#!/usr/bin/env python3
"""
One-off test for Qwen2.5-VL-7B action format training and generation
"""

import asyncio
import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich import box

from dark.online_llm import AsyncOnlineLLM, BatchConfig
from dark.sampling_params import SamplingParams

console = Console()

# Training examples in action format
TRAINING_EXAMPLES = [
    {
        "prompt": "Task: List all files in the directory /home/data\n\nTools: list_directory, read_file\n\nOutput ONE action only:",
        "response": '<action_call>{"action": "list_directory", "parameters": {"path": "/home/data"}}</action_call>'
    },
    {
        "prompt": "Task: Read the contents of config.json file\n\nTools: list_directory, read_file\n\nOutput ONE action only:",
        "response": '<action_call>{"action": "read_file", "parameters": {"path": "config.json"}}</action_call>'
    },
    {
        "prompt": "Task: Find all Python files in the current directory\n\nTools: list_directory, read_file, find_files\n\nOutput ONE action only:",
        "response": '<action_call>{"action": "find_files", "parameters": {"pattern": "*.py", "path": "."}}</action_call>'
    },
    {
        "prompt": "Task: Check if requirements.txt exists and read it\n\nTools: list_directory, read_file, file_exists\n\nOutput ONE action only:",
        "response": '<action_call>{"action": "file_exists", "parameters": {"path": "requirements.txt"}}</action_call>'
    },
    {
        "prompt": "Task: Count the number of lines in main.py\n\nTools: read_file, count_lines, list_directory\n\nOutput ONE action only:",
        "response": '<action_call>{"action": "count_lines", "parameters": {"path": "main.py"}}</action_call>'
    }
]

# Test prompts to evaluate action generation
TEST_PROMPTS = [
    "Task: Read the contents of data.json file\n\nTools: list_directory, read_file, write_file\n\nOutput ONE action only:",
    "Task: List all files in the /tmp directory\n\nTools: list_directory, read_file, delete_file\n\nOutput ONE action only:",
    "Task: Search for all .txt files in the documents folder\n\nTools: find_files, list_directory, read_file\n\nOutput ONE action only:",
    "Task: Check if backup.sql file exists\n\nTools: file_exists, read_file, list_directory\n\nOutput ONE action only:"
]

async def train_model(llm: AsyncOnlineLLM):
    """Train the model on action format examples"""
    console.print(Panel("🎯 Training Phase: Learning Action Format", border_style="blue"))
    
    # Prepare training data
    training_data = []
    for example in TRAINING_EXAMPLES:
        training_data.append([
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["response"]}
        ])
    
    # Display training examples
    training_table = Table(title="📚 Training Examples", box=box.ROUNDED)
    training_table.add_column("Example", style="blue", no_wrap=True)
    training_table.add_column("Prompt", style="white", max_width=40)
    training_table.add_column("Expected Response", style="green", max_width=40)
    
    for i, example in enumerate(TRAINING_EXAMPLES, 1):
        training_table.add_row(
            str(i),
            example["prompt"][:60] + "..." if len(example["prompt"]) > 60 else example["prompt"],
            example["response"]
        )
    
    console.print(training_table)
    
    # Train the model
    with console.status("[bold green]Training model on action format..."):
        for i, conversation in enumerate(training_data):
            await llm.learn(
                conversation,
                adapter="action_format",
                steps=5,  # Fewer steps for quick training
                lr=2e-4   # Higher learning rate for focused learning
            )
            console.print(f"[green]✓[/green] Trained on example {i+1}/{len(training_data)}")
    
    console.print("[bold green]✅ Training completed![/bold green]")

async def test_action_generation(llm: AsyncOnlineLLM):
    """Test the model's ability to generate actions"""
    console.print(Panel("🧪 Testing Phase: Action Generation", border_style="green"))
    
    # Set up sampling parameters for focused generation
    sampling_params = SamplingParams(
        temperature=0.1,  # Low temperature for consistent format
        max_tokens=100,   # Short responses for actions
        n=1,
        presence_penalty=0.0,
        ignore_eos=False
    )
    
    results = []
    
    for i, test_prompt in enumerate(TEST_PROMPTS, 1):
        console.print(f"\n[bold blue]Test {i}/{len(TEST_PROMPTS)}[/bold blue]")
        console.print(Panel(test_prompt, title="Test Prompt", border_style="blue"))
        
        # Generate response
        with console.status("[bold blue]Generating action..."):
            response = await llm.generate(test_prompt, sampling_params=sampling_params)
        
        # Display response
        console.print(Panel(response, title="Generated Response", border_style="green"))
        
        # Try to parse the action
        action_valid = False
        parsed_action = None
        
        try:
            if "<action_call>" in response and "</action_call>" in response:
                action_start = response.find("<action_call>") + 13
                action_end = response.find("</action_call>")
                action_str = response[action_start:action_end].strip()
                parsed_action = json.loads(action_str)
                action_valid = True
                console.print("[green]✅ Valid action format detected![/green]")
            else:
                console.print("[red]❌ No valid action format found[/red]")
        except Exception as e:
            console.print(f"[red]❌ Error parsing action: {e}[/red]")
        
        results.append({
            "prompt": test_prompt,
            "response": response,
            "valid": action_valid,
            "parsed_action": parsed_action
        })
        
        # User can review each result
        if action_valid:
            action_table = Table(title="Parsed Action", box=box.SIMPLE)
            action_table.add_column("Field", style="blue")
            action_table.add_column("Value", style="white")
            action_table.add_row("Action", parsed_action.get('action', 'N/A'))
            action_table.add_row("Parameters", json.dumps(parsed_action.get('parameters', {}), indent=2))
            console.print(action_table)
        
        # Ask user to continue
        if i < len(TEST_PROMPTS):
            Prompt.ask("\n[dim]Press Enter to continue to next test[/dim]", default="")
    
    return results

async def display_results(results):
    """Display final test results summary"""
    console.print(Panel("📊 Final Results Summary", border_style="blue"))
    
    valid_count = sum(1 for r in results if r["valid"])
    total_count = len(results)
    success_rate = (valid_count / total_count) * 100
    
    # Summary stats
    stats_table = Table(title="Test Statistics", box=box.ROUNDED)
    stats_table.add_column("Metric", style="blue")
    stats_table.add_column("Value", style="white")
    
    stats_table.add_row("Total Tests", str(total_count))
    stats_table.add_row("Valid Actions", str(valid_count))
    stats_table.add_row("Invalid Actions", str(total_count - valid_count))
    stats_table.add_row("Success Rate", f"{success_rate:.1f}%")
    
    console.print(stats_table)
    
    # Color-coded results
    if success_rate >= 75:
        console.print("[bold green]🎉 Excellent performance! Model learned action format well.[/bold green]")
    elif success_rate >= 50:
        console.print("[bold yellow]⚠️  Moderate performance. May need more training.[/bold yellow]")
    else:
        console.print("[bold red]❌ Poor performance. Requires additional training.[/bold red]")

async def main():
    """Main test function"""
    console.print(Panel(
        "[bold white]Qwen2.5-VL-7B Action Format Test[/bold white]\n\n"
        "This test will:\n"
        "1. Train the model on action format examples\n"
        "2. Test its ability to generate valid actions\n"
        "3. Evaluate performance",
        title="🚀 Action Format Training Test",
        border_style="blue"
    ))
    
    # Initialize model
    with console.status("[bold green]Loading Qwen2.5-VL-7B model..."):
        batch_config = BatchConfig(
            max_batch_size=2,
            auto_batch_size=True,
            memory_threshold=0.8
        )
        
        llm = AsyncOnlineLLM(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            temperature=0.1,
            max_tokens=100,
            engine="hf",
            lora_rank=16,
            lora_alpha=32,
            batch_config=batch_config,
            train_every=1,  # Train immediately
            use_adam8bit=True,
            thinking_mode=False
        )
    
    console.print("[bold green]✅ Model loaded successfully![/bold green]")
    
    try:
        # Phase 1: Training
        await train_model(llm)
        
        # Brief pause
        Prompt.ask("\n[bold blue]Training complete! Press Enter to start testing[/bold blue]", default="")
        
        # Phase 2: Testing
        results = await test_action_generation(llm)
        
        # Phase 3: Results
        await display_results(results)
        
        # Show model stats
        console.print(Panel("📈 Model Statistics", border_style="blue"))
        llm.stats()
        
        # List available adapters
        adapters = llm.list_adapters()
        if adapters:
            console.print(f"[green]✅ Trained adapters: {', '.join(adapters)}[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Error during test: {e}[/red]")
        import traceback
        console.print(traceback.format_exc())
    
    console.print(Panel(
        "[bold green]Test completed![/bold green]\n\n"
        "The model has been trained on action format examples and tested.\n"
        "Check the results above to see how well it learned the format.",
        title="✨ Done",
        border_style="green"
    ))

if __name__ == "__main__":
    asyncio.run(main()) 