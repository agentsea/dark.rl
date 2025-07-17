import asyncio
import os
import json
import json_repair
from pydantic import BaseModel

from mcp_use import MCPClient
from dataclasses import dataclass

from dark.online_llm import AsyncOnlineLLM, BatchConfig  # Updated to AsyncOnlineLLM with BatchConfig
from dark.sampling_params import SamplingParams

# Rich imports for beautiful terminal output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich.markdown import Markdown
from rich.live import Live
from rich.layout import Layout
from rich.align import Align
from rich import box

# Initialize Rich console
console = Console()

MAX_STEPS = 10  # Define the maximum number of steps
BASE_TEMPERATURE = 0.1  # Base temperature
MAX_TEMPERATURE = 1.0   # Maximum temperature
TEMPERATURE_INCREMENT = 0.2  # How much to increase temperature per repetition

# Optimized learning parameters based on our testing
OPTIMAL_LEARNING_STEPS = 10  # Optimal: 10 steps for strong memorization
OPTIMAL_LEARNING_RATE = 1e-4  # Optimal learning rate
OPTIMAL_LORA_RANK = 16  # Increased rank for better learning capacity
OPTIMAL_LORA_ALPHA = 64  # Increased alpha for stronger LoRA scaling

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

def act_ctx(task: str, tool_descriptions: list) -> str:
    # Ultra-clean prompt optimized for Qwen2.5-VL-7B  
    act_context = f"""Task: {task}

Tools: list_directory, read_file, read_multiple_files

Output ONE action only:
<action_call>{{"action": "list_directory", "parameters": {{"path": "/home/ubuntu/dark.rl/bench/fs/numbers"}}}}</action_call>

History:
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
        
        # Display tools in a beautiful table
        tools_table = Table(title="🛠️  Available MCP Tools", box=box.ROUNDED)
        tools_table.add_column("Tool Name", style="blue", no_wrap=True)
        tools_table.add_column("Description", style="white")
        tools_table.add_column("Parameters", style="green")
        
        for tool in tools:
            # Extract parameters info
            params_info = "None"
            if hasattr(tool, 'inputSchema') and tool.inputSchema:
                schema = tool.inputSchema
                if isinstance(schema, dict) and 'properties' in schema:
                    params = []
                    required = schema.get('required', [])
                    for param_name, param_info in schema['properties'].items():
                        param_type = param_info.get('type', 'unknown')
                        required_mark = '[red]*[/red]' if param_name in required else ''
                        params.append(f"{required_mark}{param_name}({param_type})")
                    params_info = ', '.join(params)
            
            tools_table.add_row(
                tool.name,
                tool.description or "No description",
                params_info
            )
        
        console.print(tools_table)
            
    except Exception as e:
        console.print(Panel(f"[red]Error getting tools: {e}[/red]", title="⚠️  Tool Loading Error", border_style="red"))
        import traceback
        console.print("[dim]Traceback:[/dim]")
        console.print(traceback.format_exc())
        
        # Fallback tools
        console.print("[blue]Using fallback tools...[/blue]")
        available_tools = ["read_file", "write_file", "list_directory"]  # fallback
        tool_descriptions = [
            "read_file: Read the contents of a file - Parameters: *path(string)",
            "write_file: Write content to a file - Parameters: *path(string), *content(string)", 
            "list_directory: List files in a directory - Parameters: *path(string)"
        ]
    
    # Load the LLM model with optimized parameters for better learning
    with console.status("[bold green]Loading LLM model with optimized learning parameters...") as status:
        # Configure batch processing for efficient learning
        batch_config = BatchConfig(
            max_batch_size=4,
            auto_batch_size=True,
            memory_threshold=0.8,
            min_batch_size=1
        )

        # model_name = "Qwen/Qwen3-8B"
        model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        
        llm = AsyncOnlineLLM(
            model=model_name, 
            temperature=0.2, 
            max_tokens=150,  # Enough for complete actions
            engine="hf",  # Use dark engine for better control and performance
            lora_rank=OPTIMAL_LORA_RANK,  # Optimized for strong learning
            lora_alpha=OPTIMAL_LORA_ALPHA,  # Optimized for strong learning
            batch_config=batch_config,
            train_every=3,  # Train after accumulating 3 examples
            use_adam8bit=True,  # Enable 8-bit Adam for memory efficiency
            thinking_mode=False,  # Explicitly disable thinking mode
        )
    
    # Display model configuration in a nice panel
    config_text = f"""
    🤖 **Model**: {model_name}
    🎛️  **Engine**: hf (HuggingFace)
    🌡️  **Temperature**: 0.2
    📝 **Max Tokens**: 150
    
    **LoRA Configuration:**
    • Rank: {OPTIMAL_LORA_RANK}
    • Alpha: {OPTIMAL_LORA_ALPHA}
    
    **Training Parameters:**
    • Steps: {OPTIMAL_LEARNING_STEPS}
    • Learning Rate: {OPTIMAL_LEARNING_RATE}
    • Train Every: 3 examples
    • Adam 8-bit: Enabled
    
    **Batch Processing:**
    • Max Batch Size: {batch_config.max_batch_size}
    • Auto Batch Size: {batch_config.auto_batch_size}
    • Memory Threshold: {batch_config.memory_threshold}
    """
    
    console.print(Panel(
        Markdown(config_text),
        title="✅ LLM Model Loaded Successfully",
        border_style="green",
        expand=False
    ))

    try:
        # Accumulate learning examples for batch training
        positive_examples = []  # For approved actions
        negative_examples = []  # For rejected/commented actions
        
        for task_idx, task in enumerate(tasks, 1):
            # Display task information in a prominent panel
            task_panel = Panel(
                f"[bold white]{task['description']}[/bold white]\n\n"
                f"[dim]Expected Output:[/dim] [blue]{task['expected_output']}[/blue]",
                title=f"📋 Task {task_idx}/{len(tasks)}",
                border_style="blue",
                expand=False
            )
            console.print(task_panel)

            history = ""  # Initialize history for each task
            previous_actions = []  # Track previous actions for repetition detection
            current_temperature = BASE_TEMPERATURE  # Current temperature for this task

            ctx = act_ctx(task['description'], tool_descriptions)
            
            for step in range(MAX_STEPS):
                # Use Rich prompt instead of input()
                Prompt.ask("\n[dim]Press Enter to continue to the next step[/dim]", default="")
                
                # Display current step information
                step_info = Panel(
                    f"[bold white]Step {step + 1}[/bold white] of [bold white]{MAX_STEPS}[/bold white]\n"
                    f"Temperature: [blue]{current_temperature}[/blue]",
                    title=f"🚀 Executing Step",
                    border_style="blue"
                )
                console.print(step_info)
                    
                # Create sampling params with dynamic temperature for more focused responses
                sampling_params = SamplingParams(
                    temperature=current_temperature,
                    max_tokens=200,  # Enough for a complete action
                    n=1,
                    presence_penalty=0.1,  # Reduce repetition but not too much
                    ignore_eos=False  # RESPECT EOS tokens - model knows when to stop!
                )
            
                try:
                    # Display context in a collapsible panel
                    console.print(Panel(
                        ctx,
                        title="📝 Context",
                        border_style="dim",
                        expand=False
                    ))
                    
                    # Stream the response with clearer formatting
                    act_response = ""
                    console.print("[bold blue]🤖 LLM Response:[/bold blue]")
                    
                    # Stream response naturally - let model stop with EOS tokens
                    chunk_count = 0
                    async for chunk in llm.stream(ctx, sampling_params=sampling_params):
                        act_response += chunk
                        chunk_count += 1
                        
                        # Print each chunk immediately for real-time feedback
                        console.print(f"[dim]Chunk {chunk_count}:[/dim] {repr(chunk)}")
                        
                        # Minimal safety limit - model should stop naturally
                        if chunk_count > 150:  
                            console.print("[yellow]Safety limit reached - model may be stuck[/yellow]")
                            break
                    
                    # Display final complete response
                    console.print(Panel(
                        act_response,
                        title=f"Complete Response ({len(act_response)} chars)",
                        border_style="blue"
                    ))
                
                    # Extract action from response
                    if "<action_call>" in act_response and "</action_call>" in act_response:
                        action_start = act_response.find("<action_call>") + 13
                        action_end = act_response.find("</action_call>")
                        action_str = act_response[action_start:action_end].strip()
                    else:
                        console.print("[red]❌ No valid action found in response[/red]")
                        console.print("[yellow]Expected format: <action_call>{\"action\": \"tool_name\", \"parameters\": {...}}</action_call>[/yellow]")
                        ctx += f"\n<response>ERROR: Use exactly this format: <action_call>{{\"action\": \"list_directory\", \"parameters\": {{\"path\": \"/home/ubuntu/dark.rl/bench/fs/numbers\"}}}}</action_call></response>\n"
                        continue
                    
                    action_dict = json_repair.loads(action_str)
                    
                    # Display parsed action in a nice format
                    action_table = Table(title="🎯 Parsed Action", box=box.SIMPLE)
                    action_table.add_column("Field", style="blue")
                    action_table.add_column("Value", style="white")
                    action_table.add_row("Action", action_dict.get('action', 'N/A'))
                    
                    params_str = json.dumps(action_dict.get('parameters', {}), indent=2)
                    action_table.add_row("Parameters", params_str)
                    console.print(action_table)
                    
                    # Check for action repetition and adjust temperature
                    action_signature = (action_dict.get('action'), str(action_dict.get('parameters', {})))
                    
                    if action_signature in previous_actions:
                        # Action is repeated, increase temperature
                        current_temperature = min(current_temperature + TEMPERATURE_INCREMENT, MAX_TEMPERATURE)
                        console.print(f"[blue]🔄 Action repetition detected! Increasing temperature to {current_temperature}[/blue]")
                    else:
                        # New action, reset temperature to base
                        if current_temperature > BASE_TEMPERATURE:
                            console.print(f"[green]✅ New action detected! Resetting temperature to {BASE_TEMPERATURE}[/green]")
                        current_temperature = BASE_TEMPERATURE
                    
                    # Add current action to history (keep last 3 actions to detect patterns)
                    previous_actions.append(action_signature)
                    if len(previous_actions) > 3:
                        previous_actions.pop(0)
                    
                except Exception as e:
                    console.print(f"[red]❌ Error parsing response: {e}[/red]")
                    continue

                ctx += f"{act_response}"

                action = Action.model_validate(action_dict)
                if action.action == "end":
                    console.print("[bold green]🎉 Task completed![/bold green]")
                    break

                # Take mcp action
                with console.status(f"[bold blue]Executing action: {action.action}[/bold blue]"):
                    try:
                        tool_response = await session.connector.call_tool(action.action, action.parameters)
                        tool_response_json = tool_response.model_dump_json()
                    except Exception as e:
                        console.print(f"[red]❌ Error taking action: {e}[/red]")
                        continue

                ctx += f"\n<response>{tool_response_json}</response>\n"

                # Display tool response in a formatted panel
                response_panel = Panel(
                    f"[dim]Type:[/dim] {type(tool_response)}\n"
                    f"[dim]Content:[/dim] {tool_response.content}\n"
                    f"[dim]Full Response:[/dim] {tool_response}",
                    title="🔧 Tool Response",
                    border_style="green"
                )
                console.print(response_panel)

                # Update history with the current step
                history += f"Step {step}: Action: {action_dict}, Response: {tool_response}\n"

                # Get verification from LLM
                verify_prompt = verify_context(task['description'], history, str(action_dict), tool_response_json)
                with console.status("[bold blue]Getting verification from LLM..."):
                    outcome = await llm.generate(verify_prompt)
                
                console.print(Panel(
                    outcome,
                    title="🔍 LLM Verification",
                    border_style="blue"
                ))

                # Get user feedback with Rich prompt
                feedback = Prompt.ask(
                    "\n[bold blue]Please provide feedback[/bold blue]",
                    choices=["a", "r", "s"],
                    default="s",
                    show_choices=True,
                    show_default=True
                )

                if feedback == 'a':
                    console.print("[green]✅ User approved the action[/green]")
                    
                    # Show learning progress
                    with console.status("[bold green]Learning from approved action..."):
                        await llm.learn(
                            [{"role": "user", "content": ctx}, {"role": "assistant", "content": act_response}],
                            adapter="task_learning",
                            steps=OPTIMAL_LEARNING_STEPS,
                            lr=OPTIMAL_LEARNING_RATE
                        )
                        positive_examples.append((ctx, act_response))
                    
                    console.print("[dim]✓ Learning completed for approved action[/dim]")

                elif feedback == 'r':
                    console.print("[red]❌ User rejected the action[/red]")
                    comment = Prompt.ask("[blue]Why did it fail?[/blue]")
                    
                    critique_prompt = critique_context(task['description'], history, str(action_dict), tool_response_json)

                    # Show learning progress for rejection
                    with console.status("[bold blue]Learning from rejected action..."):
                        await llm.learn(
                            [{"role": "user", "content": critique_prompt}, {"role": "assistant", "content": comment}],
                            adapter="critique_learning",
                            steps=OPTIMAL_LEARNING_STEPS,
                            lr=OPTIMAL_LEARNING_RATE
                        )
                        ctx += f"\n<comment>{comment}</comment>"
                        negative_examples.append((ctx, comment))
                    
                    console.print("[dim]✓ Learning completed for rejected action[/dim]")

                elif feedback == 's':
                    console.print("[dim]⏭️  User skipped the action[/dim]")

                else:
                    console.print("[red]Invalid feedback, please try again[/red]")
                
                llm.stats()
        
        # Final batch learning for improved efficiency
        console.print("\n" + "=" * 60)
        console.print(Panel(
            f"📊 **Learning Summary**\n\n"
            f"✅ Positive Examples: [green]{len(positive_examples)}[/green]\n"
            f"❌ Negative Examples: [red]{len(negative_examples)}[/red]\n"
            f"📈 Total Learning Instances: [blue]{len(positive_examples) + len(negative_examples)}[/blue]",
            title="🚀 BATCH LEARNING SUMMARY",
            border_style="blue"
        ))
        
        # Final adapter comparison
        adapters = llm.list_adapters()
        if len(adapters) > 1:
            console.print(f"\n[bold blue]⚖️  Comparing {len(adapters)} trained adapters...[/bold blue]")
            test_prompt = "Given a complex task, what should be your first step?"
            
            with console.status("[bold blue]Running adapter comparison..."):
                comparison_results = await llm.parallel_adapter_generate(
                    prompts=[test_prompt],
                    adapters=adapters
                )
            
            # Display comparison results in a table
            comparison_table = Table(title="🔍 Adapter Comparison Results", box=box.ROUNDED)
            comparison_table.add_column("Adapter", style="blue", no_wrap=True)
            comparison_table.add_column("Response", style="white")
            
            for adapter, responses in comparison_results.items():
                comparison_table.add_row(adapter, responses[0])
            
            console.print(comparison_table)

        llm.stats()
        
        # Final success message
        final_panel = Panel(
            f"🎉 **Training Session Completed Successfully!**\n\n"
            f"📈 Final Adapters: [blue]{', '.join(llm.list_adapters())}[/blue]\n"
            f"📊 Total Tasks Completed: [green]{len(tasks)}[/green]\n"
            f"🧠 Learning Examples: [blue]{len(positive_examples) + len(negative_examples)}[/blue]",
            title="✨ SUCCESS",
            border_style="green"
        )
        console.print(final_panel)
    
    finally:
        # Clean up sessions
        await client.close_all_sessions()

if __name__ == "__main__":
    asyncio.run(main())
        


