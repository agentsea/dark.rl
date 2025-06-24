import asyncio
import os
import json_repair
from pydantic import BaseModel

from mcp_use import MCPClient
from dataclasses import dataclass

from dark.online_llm import OnlineLLM
from dark.sampling_params import SamplingParams

MAX_STEPS = 10  # Define the maximum number of steps

mcp_config = {
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "./bench/fs/numbers/"
      ]
    },
    "scratchpad-tool": {
      "command": "npx",
      "args": ["-y", "scratchpad-tool"]
    }
  }
}

class Action(BaseModel):
    action: str
    parameters: dict

def act_ctx(task: str, tool_descriptions: list, history: str = "") -> str:

    # The output of the action will be returned in the following format:
    # <response>I have navigated to the flights page</response>

    act_context = f"""You are operating a file system helping accomplish tasks.
Please help complete the task '{task}' with the available tools: {tool_descriptions}

For example if the task was to "Find a flight to Paris" you would output something like this depending on the state:

    <action>{{
        "action": "browser_navigate",
        "parameters": {{
            "url": "https://flights.google.com"
        }}
    }}</action>

When the task is complete, return the `end` action, but not till you are absolutely sure!

Please now review the history of actions and responses and output a new action.
    """
    return act_context

def act_ctx_reason(task: str, tool_descriptions: list, history: str = "") -> str:

    # The output of the action will be returned in the following format:
    # <response>I have navigated to the flights page</response>

    act_context = f"""You are operating a file system helping accomplish tasks.
Please help complete the task '{task}' with the available tools: {tool_descriptions}

For example if the task was to "Find a flight to Paris" you would output something like this depending on the state:

    <action>{{
        "action": "browser_navigate",
        "parameters": {{
            "url": "https://flights.google.com"
        }}
    }}</action>

When the task is complete, return the `end` action, but not till you are absolutely sure!

Please now review the history of actions and responses and output a new action.
    """
    return act_context

def verify_context(task: str, history: str, action: str, response: str):
    verify_prompt = f"""You are tasked with determining if an action taken by an agent to accomplish a task is correct.
    You will be given the task, the history of actions and responses, the action taken, and the response.
    The task is: {task}
    The history of actions and responses is: {history}
    The action was: {action}
    The response was: {response}
    Please return 'yes' if the action was correct, 'no' if it was incorrect, and 'unsure' if you are unsure.
    """
    return verify_prompt

def critique_context(task: str, history: str, action: str, response: str):
    critique_prompt = f"""You are tasked with determining if an action taken by an agent to accomplish a task is correct.
    You will be given the task, the history of actions and responses, the action taken, and the response.
    The task is: {task}
    The history of actions and responses is: {history}
    The action was: {action}
    The response was: {response}
    Please output a critique of the action and response.
    """
    return critique_prompt

tasks = [
    {"description": "Add all the numbers on line 5 in every even numbered file", "expected_output": "4"}, # 2, 1, 1
    {"description": "Add all the numbers on line 6 in every odd numbered file", "expected_output": "18"}, # 9, 4, 5
    {"description": "Add all the numbers on line 7 in every file", "expected_output": "19"}, # 7, 0, 4, 0, 5, 3
    {"description": "Multiply all the numbers on line 2 in every file", "expected_output": "39"}, # 10, 8, 6, 1, 6, 8
    {"description": "Multiply all the numbers on line 3 in every file", "expected_output": "36"}, # 11, 5, 4, 1, 6, 9
    {"description": "Multiply all the numbers on line 4 in every file then subtract line 5 in file 2", "expected_output": "15"}, # 3, 1, 8, 1, 4, 0 - 2
]

async def main():
    # Create MCP Client 
    client = MCPClient.from_dict(mcp_config)
    
    # Create the session (this is the key step that was missing!)
    session = await client.create_session("filesystem")
    
    # Get available tools from the session
    try:
        # Make sure session is connected and initialized
        if not session.connector.is_connected:
            await session.connect()
        
        # Get tools using the MCP-use API - tools are on the connector
        tools = await session.connector.list_tools()  # This returns list[Tool] directly
        available_tools = []
        tool_descriptions = []
        
        for tool in tools:
            available_tools.append(tool.name)
            
            # Build detailed tool description
            tool_desc = f"{tool.name}: {tool.description or 'No description'}"
            
            # Add parameter information from inputSchema
            if hasattr(tool, 'inputSchema') and tool.inputSchema:
                schema = tool.inputSchema
                if isinstance(schema, dict) and 'properties' in schema:
                    params = []
                    required = schema.get('required', [])
                    for param_name, param_info in schema['properties'].items():
                        param_type = param_info.get('type', 'unknown')
                        required_mark = '*' if param_name in required else ''
                        params.append(f"{required_mark}{param_name}({param_type})")
                    tool_desc += f" - Parameters: {', '.join(params)}"
            
            tool_descriptions.append(tool_desc)
        
        print(f"Available tools: {available_tools}")
        print("Tool details:")
        for desc in tool_descriptions:
            print(f"  - {desc}")
            
    except Exception as e:
        print(f"Error getting tools: {e}")
        import traceback
        traceback.print_exc()
        available_tools = ["read_file", "write_file", "list_directory"]  # fallback
        tool_descriptions = [
            "read_file: Read the contents of a file - Parameters: *path(string)",
            "write_file: Write content to a file - Parameters: *path(string), *content(string)", 
            "list_directory: List files in a directory - Parameters: *path(string)"
        ]
    
    # Load the LLM model (logging is now handled by logging.debug)
    print("Loading LLM model...")
    llm = OnlineLLM(model="Qwen/Qwen3-8B", temperature=0.2, max_tokens=3000, architecture="hf")
    print("✓ LLM model loaded successfully!")

    try:
        for task in tasks:
            print(f"\n\n====\nTask: {task['description']}\n")
            print(f"Expected output: {task['expected_output']}\n")

            history = ""  # Initialize history for each task

            for step in range(MAX_STEPS):
                input("Press Enter to continue")
                print(f"\nStep {step}")
                ctx = act_ctx(task['description'], tool_descriptions, history)
                
                # Create sampling params with shorter generation
                sampling_params = SamplingParams(
                    temperature=0.1,
                    max_tokens=500,
                    n=1,
                    presence_penalty=0.0,
                    ignore_eos=False
                )
                
                try:
                    # Generate the response
                    print(f"Context: {ctx}\n")
                    print("Generating response...")
                    act_response = await llm.generate_async(ctx, sampling_params=sampling_params)
                    print(f"Response: {act_response}")
                    
                    # Extract action from response
                    if "<action>" in act_response and "</action>" in act_response:
                        action_start = act_response.find("<action>") + 8
                        action_end = act_response.find("</action>")
                        action_str = act_response[action_start:action_end].strip()
                    else:
                        print("No valid action found in response")
                        ctx += f"<response>No valid action found in response, are you sure you output a valid tool call? Check again</response>"
                        continue
                    
                    action_dict = json_repair.loads(action_str)
                    print(f"Action: {action_dict}")
                except Exception as e:
                    print(f"Error parsing response: {e}")
                    continue

                action = Action.model_validate(action_dict)
                if action.action == "end":
                    print("Done!")
                    break

                # Take mcp action
                print(f"Taking action: {action.action} with parameters: {action.parameters}")
                try:
                    response = await session.connector.call_tool(action.action, action.parameters)
                    print(f"Tool response: {response}")
                except Exception as e:
                    print(f"Error taking action: {e}")
                    continue

                print("Action taken")
                print(f"Response: {response}")

                # Update history with the current step
                history += f"Step {step}: Action: {action_dict}, Response: {response}\n"

                verify_prompt = verify_context(task['description'], history, str(action_dict), str(response))
                outcome = await llm.generate_async(verify_prompt)
                print(f"Outcome: {outcome}")

                comment = None
                feedback = input("Press 'a' to approve, 'r' to reject, 'c' to comment, 's' to skip: ")
                if feedback == 'a':
                    print("Approved")
                    print(f"Learning: {ctx} -> \n{act_response}")
                    await llm.learn_async(ctx, act_response)

                elif feedback == 'r':
                    print("Rejected")

                elif feedback == 'c':
                    comment = input("Enter your comment: ")
                    print(f"Comment: {comment}")
                    critique_prompt = critique_context(task['description'], history, str(action_dict), str(response))

                    print(f"Learning: {critique_prompt} -> \n{comment}")
                    await llm.learn_async(critique_prompt, comment)

                elif feedback == 's':
                    print("Skipping")

                else:
                    print("Invalid feedback")
    
    finally:
        # Clean up sessions
        await client.close_all_sessions()

if __name__ == "__main__":
    asyncio.run(main())
        


