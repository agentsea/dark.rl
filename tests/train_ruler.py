import asyncio
import os
import json
import json_repair
from pydantic import BaseModel
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer, is_liger_available
import re
import textwrap
import openai
import shutil
from pathlib import Path

from mcp_use import MCPClient
from dataclasses import dataclass

# Rich imports for beautiful terminal output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich import box

# Initialize Rich console
console = Console()

# --- RULER Reward Function ---

class RULEReward:
    """
    Implementation of the RULER reward function for agentic tasks.
    It uses a "judge" LLM to score full trajectories.
    """
    def __init__(self, judge_model_name, num_generations):
        print(f"Initializing RULER judge with model: {judge_model_name}...")
        self.num_generations = num_generations
        self.judge_model_name = judge_model_name

        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.openai_client = openai.OpenAI()
        self.openai_model = "gpt-4o" if self.judge_model_name == "o3" else self.judge_model_name
        if self.judge_model_name == "o3":
                print("Using 'gpt-4o' for the 'o3' judge model.")
            
        self.rubric = textwrap.dedent("""\
            A trajectory that achieves its goal should always get a significantly higher score than a trajectory that does not achieve its goal.
            A trajectory that achieves its goal more efficiently (e.g., by avoiding unproductive detours) should get a higher score than a trajectory that achieves its goal less efficiently.
            If one trajectory is only slightly better than another, the difference in scores should be small. If it is significantly better, the difference in scores should be large.
            You may give some partial credit for a trajectory that makes progress towards its goal but does not complete it.""")

    def get_judge_prompt(self, prompt, completions, tool_descriptions):
        """Constructs the prompt for the judge LLM."""
        completion_text = ""
        for i, completion in enumerate(completions):
            completion_text += f"Trajectory {i+1}:\n---\n{completion}\n---\n\n"

        return textwrap.dedent(f"""\
            You are an impartial judge. Your task is to evaluate a set of AI-generated agent trajectories based on a shared goal and a rubric.

            The agent has access to the following tools:
            {tool_descriptions}

            Shared Goal:
            {prompt}

            The agent was asked to produce a plan including thoughts and tool calls to solve the task. Review the following trajectories:

            {completion_text}

            Rubric:
            {self.rubric}

            Evaluate each trajectory on its likelihood of success and efficiency. A good trajectory will have a clear plan and use the tools correctly. A bad trajectory might hallucinate tool outputs, use tools incorrectly, or have a flawed plan.

            Provide a score between 0.0 and 1.0 for each trajectory, along with a brief explanation for your reasoning. Your output MUST be a valid JSON list of objects, where each object has a "score" and "explanation" key.
            """)

    def _parse_response(self, response, num_completions):
        try:
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
            json_str = json_match.group(1) if json_match else response
            parsed_response = json.loads(json_str)
            scores = [float(item['score']) for item in parsed_response]

            if len(scores) != num_completions:
                print(f"Warning: Judge returned {len(scores)} scores for {num_completions} completions. Padding with 0.0.")
                scores.extend([0.0] * (num_completions - len(scores)))
            
            return scores
        except (json.JSONDecodeError, ValueError, KeyError, IndexError, AttributeError) as e:
            print(f"Error parsing judge response: {e}\nResponse was:\n---\n{response}\n---")
            return [0.0] * num_completions

    def _get_rewards_from_openai(self, prompt_text, group_completions, tool_descriptions):
        judge_prompt_str = self.get_judge_prompt(prompt_text, group_completions, tool_descriptions)
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": judge_prompt_str}],
                temperature=0.0,
            )
            content = response.choices[0].message.content
            return self._parse_response(content, len(group_completions))
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return [0.0] * len(group_completions)

    def __call__(self, prompts, completions, **kwargs):
        # We need tool descriptions, which are not normally passed.
        # We'll retrieve them from a shared context or global scope as a workaround.
        tool_descriptions = kwargs.get("tool_descriptions", "No tools provided.")

        all_scores = []
        if not prompts:
            return []

        num_generations_per_prompt = self.num_generations
        
        for i, prompt_text in enumerate(prompts):
            start_idx = i * num_generations_per_prompt
            end_idx = (i + 1) * num_generations_per_prompt
            group_completions = completions[start_idx:end_idx]

            scores = self._get_rewards_from_openai(prompt_text, group_completions, tool_descriptions)
            all_scores.extend(scores)

        return all_scores

# --- MCP and Task Setup ---

mcp_config = {
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "./bench/fs/numbers/"
      ]
    }
  }
}

tasks = [
    {"description": "Add all the numbers on line 5 in every even numbered file", "expected_output": "4"},
    {"description": "Add all the numbers on line 6 in every odd numbered file", "expected_output": "18"},
    {"description": "Add all the numbers on line 7 in every file", "expected_output": "19"},
    {"description": "Multiply all the numbers on line 2 in every file", "expected_output": "39"},
    {"description": "Multiply all the numbers on line 3 in every file", "expected_output": "36"},
    {"description": "Multiply all the numbers on line 4 in every file then subtract line 5 in file 2", "expected_output": "15"},
]

async def evaluate_agent(model, tokenizer, prompts, ruler_reward_model, tool_descriptions_str):
    """Generates a plan for each task and uses RULER to score it."""
    results = []
    console.print(Panel(f"Evaluating agent on {len(prompts)} tasks...", style="yellow", title="Evaluation"))
    
    for i, prompt in enumerate(prompts):
        with console.status(f"Generating and scoring plan for task {i+1}/{len(prompts)}..."):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            # Generate a single plan, no sampling
            outputs = model.generate(
                input_ids, 
                max_new_tokens=1024, 
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
            # Decode only the generated part
            plan = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
            
            # The RULER judge needs a list of completions. We send a list with our single plan.
            scores = ruler_reward_model._get_rewards_from_openai(
                prompt, 
                [plan], 
                tool_descriptions_str
            )
            score = scores[0] if scores else 0.0
            
            results.append({"prompt": prompt, "plan": plan, "score": score})
            
    return results

def display_evaluation_results(results, title="Evaluation Results"):
    """Displays evaluation results in a Rich table."""
    table = Table(title=title, box=box.ROUNDED, show_lines=True)
    table.add_column("Task Description", style="cyan", min_width=20)
    table.add_column("Generated Plan", style="white", min_width=50)
    table.add_column("RULER Score", style="magenta", justify="right")

    for result in results:
        # Extract task description from the full prompt for cleaner display
        task_desc = re.search(r"Task:\n(.+)", result['prompt'])
        task_desc = task_desc.group(1).strip() if task_desc else "N/A"
        
        score_color = "green" if result['score'] > 0.7 else "yellow" if result['score'] > 0.4 else "red"
        table.add_row(
            task_desc,
            result['plan'],
            f"[{score_color}]{result['score']:.2f}[/{score_color}]"
        )
    console.print(table)

def display_comparison_results(before_results, after_results):
    """Displays a side-by-side comparison of before and after results."""
    table = Table(title="🔬 Before & After Training Comparison", box=box.HEAVY_HEAD, show_lines=True)
    table.add_column("Task", style="cyan", min_width=20)
    table.add_column("Before Plan", style="white", min_width=40)
    table.add_column("Before Score", style="magenta", justify="right")
    table.add_column("After Plan", style="white", min_width=40)
    table.add_column("After Score", style="magenta", justify="right")

    for before, after in zip(before_results, after_results):
        task_desc = re.search(r"Task:\n(.+)", before['prompt'])
        task_desc = task_desc.group(1).strip() if task_desc else "N/A"

        before_score_color = "green" if before['score'] > 0.7 else "yellow" if before['score'] > 0.4 else "red"
        after_score_color = "green" if after['score'] > 0.7 else "yellow" if after['score'] > 0.4 else "red"
        
        score_change = after['score'] - before['score']
        change_icon = "🔼" if score_change > 0 else "🔽" if score_change < 0 else "➖"

        table.add_row(
            task_desc,
            before['plan'],
            f"[{before_score_color}]{before['score']:.2f}[/{before_score_color}]",
            after['plan'],
            f"[{after_score_color}]{after['score']:.2f}[/{after_score_color}] {change_icon}"
        )
    console.print(table)


# --- Main Training Script ---

async def main():
    if not is_liger_available():
        print("Liger is not available. To reduce memory usage, please install TRL from source with the 'liger' extra.")

    client = MCPClient.from_dict(mcp_config)
    session = await client.create_session("filesystem")

    try:
        if not session.connector.is_connected:
            await session.connect()
        
        mcp_tools = await session.connector.list_tools()
        tool_descriptions = [f"{tool.name}: {tool.description or 'No description'}" for tool in mcp_tools]
        tool_descriptions_str = "\n".join(tool_descriptions)

        tools_table = Table(title="🛠️  Available MCP Tools", box=box.ROUNDED)
        tools_table.add_column("Tool Name", style="blue")
        tools_table.add_column("Description", style="white")
        for tool in mcp_tools:
            tools_table.add_row(tool.name, tool.description or "No description")
        console.print(tools_table)

    except Exception as e:
        console.print(Panel(f"[red]Error getting tools: {e}[/red]", title="⚠️  Tool Loading Error"))
        tool_descriptions_str = "read_file, write_file, list_directory" # Fallback
    
    # --- Create Dataset of Prompts ---
    
    def create_agent_prompt(task_description, tools_str):
        return textwrap.dedent(f"""\
            You are an expert agent tasked with solving a problem using a given set of tools.
            Your goal is to devise a step-by-step plan to accomplish the task.
            The plan should consist of a sequence of thoughts and tool calls.
            You must only use the tools provided. Do not execute the plan, just write it out.

            Available Tools:
            {tools_str}
            
            Task:
            {task_description}

            Now, provide the plan to solve the task.
            """)

    prompts = [create_agent_prompt(task["description"], tool_descriptions_str) for task in tasks]
    dataset = Dataset.from_dict({"prompt": prompts})

    # --- Configuration ---
    model_name = "Qwen/Qwen2-7B-Instruct"
    judge_model_name = "o3"
    num_generations_per_prompt = 4

    training_args = GRPOConfig(
        output_dir=f"./{model_name.split('/')[-1]}-ruler-mcp-agent",
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        logging_steps=10,
        num_train_epochs=5, # Train for more epochs on this small dataset
        num_generations=num_generations_per_prompt,
        use_liger_loss=is_liger_available(),
        max_prompt_length=1024,
        max_completion_length=1024,
        report_to="none",
        save_strategy="no",
        bf16=True,
        remove_unused_columns=False,
    )

    # --- Model & Tokenizer for Training ---
    console.print(Panel(Markdown(f"### Loading training model: {model_name}"), title_align="left", border_style="green"))
    qlora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    # --- RULER Reward Model ---
    # Pass tool descriptions via a kwarg, which TRL will pass through to the reward function
    reward_kwargs = {"tool_descriptions": tool_descriptions_str}
    
    ruler_reward_model = RULEReward(
        judge_model_name=judge_model_name,
        num_generations=num_generations_per_prompt,
    )

    # --- GRPOTrainer ---
    console.print(Panel(Markdown("### Initializing GRPOTrainer"), title_align="left", border_style="green"))
    
    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=ruler_reward_model,
        reward_kwargs=reward_kwargs,
        args=training_args,
        train_dataset=dataset,
        peft_config=qlora_config,
        model_init_kwargs={
            "quantization_config": bnb_config,
            "device_map": {"": torch.cuda.current_device()} if torch.cuda.is_available() else "cpu"
        }
    )

    # We need a tokenizer for the evaluation function.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # --- Pre-Training Evaluation ---
    console.print(Panel(Markdown("### ✍️ Pre-Training Evaluation"), title_align="left", border_style="yellow"))
    pre_train_results = await evaluate_agent(trainer.model, tokenizer, prompts, ruler_reward_model, tool_descriptions_str)
    display_evaluation_results(pre_train_results, title="📊 Before Training Results")

    # --- Training ---
    console.print(Panel(Markdown("### 🚀 Starting Training..."), title_align="left", border_style="green"))
    try:
        trainer.train()
        console.print(Panel("✅ Training Finished Successfully!", border_style="green"))

        # --- Post-Training Evaluation ---
        console.print(Panel(Markdown("### ✨ Post-Training Evaluation"), title_align="left", border_style="yellow"))
        post_train_results = await evaluate_agent(trainer.model, tokenizer, prompts, ruler_reward_model, tool_descriptions_str)
        
        # --- Comparison ---
        display_comparison_results(pre_train_results, post_train_results)

    except Exception as e:
        console.print(Panel(f"❌ Training failed: {e}", border_style="red"))
    finally:
        await client.close_all_sessions()
        console.print("MCP sessions closed.")


if __name__ == "__main__":
    asyncio.run(main()) 