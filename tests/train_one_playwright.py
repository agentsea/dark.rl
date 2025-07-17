import asyncio
import os
import json
import json_repair
from pydantic import BaseModel
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime

from mcp_use import MCPClient

from dark.online_llm import AsyncOnlineLLM, BatchConfig
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

MAX_STEPS = 15  # Increased for web interactions
BASE_TEMPERATURE = 0.1
MAX_TEMPERATURE = 1.0
TEMPERATURE_INCREMENT = 0.2

# Optimized learning parameters
OPTIMAL_LEARNING_STEPS = 10
OPTIMAL_LEARNING_RATE = 1e-4
OPTIMAL_LORA_RANK = 16
OPTIMAL_LORA_ALPHA = 64

# MCP Configuration for Playwright
mcp_config = {
    "mcpServers": {
        "playwright": {
            "command": "npx",
            "args": [
                "@playwright/mcp@latest"
            ]
        }
    }
}

class Action(BaseModel):
    action: str
    parameters: dict



def act_ctx(task: str, tool_descriptions: list) -> str:
    """Create action context for web automation with thinking"""
    act_context = f"""Task: {task}

Tools: browser_navigate, browser_take_screenshot, browser_click, browser_type

Think first, then act. Format:
<think>Reasoning about what to do...</think>
<action_call>{{"action": "tool_name", "parameters": {{}}}}</action_call>

Next action:
<think>"""
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

# Define some sample web automation tasks
tasks = [
    {
        "description": "Navigate to Google and search for 'Playwright automation'", 
        "expected_output": "Successfully performed search and reached results page"
    },
    {
        "description": "Navigate to GitHub homepage and find the search bar", 
        "expected_output": "Located and identified the search functionality"
    },
    {
        "description": "Go to a news website and take screenshots of the main headlines", 
        "expected_output": "Captured visual content of news headlines"
    },
    {
        "description": "Navigate to a shopping site and locate product categories", 
        "expected_output": "Identified navigation and product organization"
    }
]



async def main():
    console.print(Panel(
        "[bold blue]🎭 Playwright + Qwen VL Training Session[/bold blue]\n\n"
        "This session will train the model on web automation tasks using MCP integration.",
        title="🚀 Welcome",
        border_style="blue"
    ))
    
    # Create MCP Client 
    client = MCPClient.from_dict(mcp_config)
    
    # Create the session with Playwright MCP server
    session = await client.create_session("playwright")
    
    # Get available tools from the session
    try:
        # Make sure session is connected and initialized
        if not session.connector.is_connected:
            await session.connect()
        
        # Get tools using the MCP-use API
        tools = await session.connector.list_tools()
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
        tools_table = Table(title="🛠️  Available Playwright MCP Tools", box=box.ROUNDED)
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
        available_tools = ["navigate", "click", "screenshot"]
        tool_descriptions = [
            "navigate: Navigate to a URL - Parameters: *url(string)",
            "click: Click on an element - Parameters: *selector(string)", 
            "screenshot: Take a screenshot - Parameters: description(string)"
        ]
    
    # Load the Qwen VL model
    with console.status("[bold green]Loading Qwen VL 2.5 model...") as status:
        batch_config = BatchConfig(
            max_batch_size=4,
            auto_batch_size=True,
            memory_threshold=0.8,
            min_batch_size=1
        )

        model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        
        llm = AsyncOnlineLLM(
            model=model_name, 
            temperature=0.2, 
            max_tokens=150,  # Increased for thinking + action content
            engine="hf",
            lora_rank=OPTIMAL_LORA_RANK,
            lora_alpha=OPTIMAL_LORA_ALPHA,
            batch_config=batch_config,
            train_every=3,
            use_adam8bit=True,
            thinking_mode=False,
        )
    
    # Display model configuration
    config_text = f"""
    🤖 **Model**: {model_name} (Vision-Language)
    🎛️  **Engine**: hf (HuggingFace)
    🌡️  **Temperature**: 0.2 (min 0.15 to avoid sticking)
    📝 **Max Tokens**: 150 (thinking + action content)
    
    **LoRA Configuration:**
    • Rank: {OPTIMAL_LORA_RANK}
    • Alpha: {OPTIMAL_LORA_ALPHA}
    
    **Web Automation:**
    • MCP Integration: Enabled
    • EOS Issue: FIXED! (guided format avoids completion trap)
    • Format: <think> + <action_call> (reasoning before acting)
    • Early Stopping: Stops immediately on complete </action_call>
    • Thinking Display: Shows model's reasoning process
    • Extraction: Separates thinking and action content
    """
    
    console.print(Panel(
        Markdown(config_text),
        title="✅ Qwen VL Model Loaded Successfully",
        border_style="green",
        expand=False
    ))

    try:
        positive_examples = []
        negative_examples = []
        
        for task_idx, task in enumerate(tasks, 1):
            # Display task information
            task_panel = Panel(
                f"[bold white]{task['description']}[/bold white]\n\n"
                f"[dim]Expected Outcome:[/dim] [blue]{task['expected_output']}[/blue]",
                title=f"📋 Web Automation Task {task_idx}/{len(tasks)}",
                border_style="blue",
                expand=False
            )
            console.print(task_panel)

            history = ""
            previous_actions = []
            current_temperature = BASE_TEMPERATURE
            
            ctx = act_ctx(task['description'], tool_descriptions)
            
            for step in range(MAX_STEPS):
                Prompt.ask("\n[dim]Press Enter to continue to the next step[/dim]", default="")
                
                step_info = Panel(
                    f"[bold white]Step {step + 1}[/bold white] of [bold white]{MAX_STEPS}[/bold white]\n"
                    f"Temperature: [blue]{current_temperature}[/blue]",
                    title=f"🚀 Executing Step",
                    border_style="blue"
                )
                console.print(step_info)
                
                sampling_params = SamplingParams(
                    temperature=max(current_temperature, 0.15),  # Minimum temp to avoid getting stuck
                    max_tokens=150,  # Increased for thinking + action content
                    n=1,
                    presence_penalty=0.2,  # Higher penalty to stop after action_call
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
                    
                    console.print("[bold blue]🤖 Qwen VL Response:[/bold blue]")
                    
                    # Stream response naturally - stop immediately on complete action_call
                    act_response = ""
                    chunk_count = 0
                    async for chunk in llm.stream(ctx, sampling_params=sampling_params):
                        act_response += chunk
                        chunk_count += 1
                        
                        # Print each chunk immediately for real-time feedback
                        console.print(f"[dim]Chunk {chunk_count}:[/dim] {repr(chunk)}")
                        
                        # Stop immediately when we have a complete action_call
                        if "</action_call>" in act_response:
                            console.print("[green]✅ Complete action_call detected - stopping generation[/green]")
                            break
                        
                        # Safety limit for thinking + action content
                        if chunk_count > 60:
                            console.print("[yellow]Safety limit reached - model may be stuck[/yellow]")
                            break
                    
                    # Display final complete response
                    console.print(Panel(
                        act_response,
                        title=f"Complete Response ({len(act_response)} chars)",
                        border_style="blue"
                    ))
                    
                    # Extract thinking and action from response
                    thinking_content = ""
                    action_str = ""
                    
                    if "</think>" in act_response and "<action_call>" in act_response and "</action_call>" in act_response:
                        # Extract thinking content (everything before </think>)
                        think_end = act_response.find("</think>")
                        thinking_content = act_response[:think_end].strip()
                        
                        # Extract action content (between <action_call> and </action_call>)
                        action_start = act_response.find("<action_call>") + 13
                        action_end = act_response.find("</action_call>")
                        action_str = act_response[action_start:action_end].strip()
                        
                        # Display the thinking process
                        console.print(Panel(
                            thinking_content,
                            title="🧠 Model's Thinking Process",
                            border_style="cyan"
                        ))
                        
                    elif "</action_call>" in act_response:
                        # Fallback: old format without thinking
                        action_end = act_response.find("</action_call>")
                        action_str = act_response[:action_end].strip()
                        
                    if not action_str:
                        console.print("[red]❌ No valid action found in response[/red]")
                        console.print("[yellow]Expected format: <think>reasoning</think><action_call>{\"action\": \"tool_name\", \"parameters\": {...}}</action_call>[/yellow]")
                        # Don't accumulate errors - reset context to avoid confusion
                        ctx = act_ctx(task['description'], tool_descriptions)  
                        # Don't add example here - the new format includes the prompt
                        continue
                    
                    action_dict = json_repair.loads(action_str)
                    
                    # Display parsed action
                    action_table = Table(title="🎯 Parsed Action", box=box.SIMPLE)
                    action_table.add_column("Field", style="blue")
                    action_table.add_column("Value", style="white")
                    action_table.add_row("Action", action_dict.get('action', 'N/A'))
                    
                    params_str = json.dumps(action_dict.get('parameters', {}), indent=2)
                    action_table.add_row("Parameters", params_str)
                    console.print(action_table)
                    
                    # Check for repetition and adjust temperature
                    action_signature = (action_dict.get('action'), str(action_dict.get('parameters', {})))
                    
                    if action_signature in previous_actions:
                        current_temperature = min(current_temperature + TEMPERATURE_INCREMENT, MAX_TEMPERATURE)
                        console.print(f"[blue]🔄 Repetition detected! Temperature → {current_temperature}[/blue]")
                    else:
                        if current_temperature > BASE_TEMPERATURE:
                            console.print(f"[green]✅ New action! Temperature → {BASE_TEMPERATURE}[/green]")
                        current_temperature = BASE_TEMPERATURE
                    
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
                            adapter="web_automation",
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
            test_prompt = "Given a complex web automation task, what should be your first step?"
            
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
            f"🧠 Learning Examples: [blue]{len(positive_examples) + len(negative_examples)}[/blue]\n"
            f"🎭 Web Automation: MCP Playwright Integration",
            title="✨ SUCCESS",
            border_style="green"
        )
        console.print(final_panel)
    
    finally:
        # Clean up sessions
        await client.close_all_sessions()

if __name__ == "__main__":
    asyncio.run(main()) 