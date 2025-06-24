import asyncio
import os

from mcp_use import MCPAgent, MCPClient
from dataclasses import dataclass

from dark.online_llm import OnlineLLM

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

client = MCPClient(mcp_config)
session = client.new_session("local_fs")

class Action(BaseModel):
    name: str
    parameters: dict

def act_ctx(task: str, history: str) -> str:
    act_ctx = f"""You are operating a web browser helping accomplish tasks.
    Please help complete the task '{task}' with the tools: {session.discover_tools()}
    Please think before you act. You should output in the following format:

    <think>I need to navigate to the google flights page</think><action>
    {{
        "action": "browser_navigate",
        "parameters": {{
            "url": "https://flights.google.com"
        }}
    }}</action>


    The output of the action will be returned in the following format:
    <response>I have navigated to the flights page</response>

    If you are done, simple return the `end` action.\n\n --- BEGIN ---\n\n
    """
    return act_ctx

def verify_ctx(task: str, history: str, action: str, response: str):
    verify_ctx = f"""You are tasked with determining if an action taken by an agent to accomplish a task is correct.
    You will be given the task, the history of actions and responses, the action taken, and the response.
    The task is: {task}
    The history of actions and responses is: {history}
    The action was: {action}
    The response was: {response}
    Please return 'yes' if the action was correct, 'no' if it was incorrect, and 'unsure' if you are unsure.
    """
    return verify_ctx

def critique_ctx(task: str, history: str, action: str, response: str):
    critique_ctx = f"""You are tasked with determining if an action taken by an agent to accomplish a task is correct.
    You will be given the task, the history of actions and responses, the action taken, and the response.
    The task is: {task}
    The history of actions and responses is: {history}
    The action was: {action}
    The response was: {response}
    Please output a critique of the action and response.
    """
    return critique_ctx

tasks = [
    {"description": "Add all the numbers on line 5 in every even numbered file", "expected_output": "4"}, # 2, 1, 1
    {"description": "Add all the numbers on line 6 in every odd numbered file", "expected_output": "18"}, # 9, 4, 5
    {"description": "Add all the numbers on line 7 in every file", "expected_output": "19"}, # 7, 0, 4, 0, 5, 3
    {"description": "Multiply all the numbers on line 2 in every file", "expected_output": "39"}, # 10, 8, 6, 1, 6, 8
    {"description": "Multiply all the numbers on line 3 in every file", "expected_output": "36"}, # 11, 5, 4, 1, 6, 9
    {"description": "Multiply all the numbers on line 4 in every file then subtract line 5 in file 2", "expected_output": "15"}, # 3, 1, 8, 1, 4, 0 - 2
]

llm = OnlineLM(model="Qwen/Qwen3-8B", temperature=0.2, max_tokens=1000)

for task in tasks:
    print(f"\n\n====\nTask: {task['description']}\n")
    print(f"Expected output: {task['expected_output']}\n")

    ctx = act_ctx(task['description'])

    for step in range(10):
        print(f"\nStep {step}")

        try:
            act_response = llm.generate(ctx)
            print(f"Response: {act_response}")

            action = json_repair.loads(act_response)
            print(f"Action: {action}")
        except Exception as e:
            print(f"Error: {e}")
            continue

        action = Action.model_validate_json(content)
        if action.name == "end":
            print("Done!")
            break

        # Take mcp action
        print(f"Taking action: {action.name} with parameters: {action.parameters}")
        try:
            response = session.call_tool(action.name, action.parameters)
        except Exception as e:
            print(f"Error taking action: {e}")
            continue
        print("Action taken")
        print(f"Response: {response}")

        final_ctx = ctx + f"<response>{response}</response>"

        verify_ctx = verify_ctx(task['description'], action, task['expected_output'])
        response = llm.generate(verify_ctx)
        print(f"Outcome: {outcome}")

        comment = None
        feedback = input("Press 'a' to approve, 'r' to reject, 'c' to comment, 's' to skip")
        if feedback == 'a':
            print("Approved")
            print(f"Learning: {ctx} -> \n{response}")
            llm.learn(ctx, response)

        elif feedback == 'r':
            print("Rejected")

        elif feedback == 'c':
            comment = input("Enter your comment: ")
            print(f"Comment: {comment}")
            critique_ctx = critique_ctx(task['description'], ctx, action, response)

            print("Learning: {critique_ctx} -> \n{comment}")
            llm.learn(critique_ctx, comment)

        elif feedback == 's':
            print("Skipping")
        else:
            print("Invalid feedback")

        ctx = final_ctx
        


