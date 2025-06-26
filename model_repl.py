#!/usr/bin/env python3
"""
Model REPL - Interactive testing for Dark.RL models
Test general knowledge and capabilities across different models and architectures.
"""

import asyncio
import time
import sys
from typing import Optional
import torch
import atexit
import os
from src.dark.online_llm import OnlineLLM

# Try to import readline, fall back gracefully if not available
try:
    import readline
    READLINE_AVAILABLE = True
except ImportError:
    READLINE_AVAILABLE = False
    print("⚠️  Warning: readline not available. Arrow keys and history won't work.")


class ModelREPL:
    """Interactive REPL for testing model knowledge"""
    
    def __init__(self):
        self.llm: Optional[OnlineLLM] = None
        self.current_model = None
        self.current_engine = None
        self.conversation_history = []
        self.streaming_enabled = True
        
        # Pending settings for next model load
        self._pending_max_tokens = None
        self._pending_temperature = None
        
        # Set up readline for better input handling
        self.setup_readline()
        
        # REPL commands for auto-completion
        self.commands = [
            "/help", "/quit", "/exit", "/models", "/switch", "/load", "/arch", 
            "/info", "/compare", "/cleanup", "/vram", "/stream", "/clear", 
            "/history", "/test", "/max", "/temp"
        ]
        
        # Available models
        self.available_models = {
            "VL Models": [
                "Qwen/Qwen2.5-VL-3B-Instruct",
                "Qwen/Qwen2.5-VL-7B-Instruct", 
                "Qwen/Qwen2.5-VL-32B-Instruct",
            ],
            "Text Models": [
                "Qwen/Qwen3-0.6B",
                "Qwen/Qwen3-1.7B", 
                "Qwen/Qwen3-4B",
                "Qwen/Qwen3-8B",
                "Qwen/Qwen3-14B",
                "Qwen/Qwen3-32B",
            ],
            "MoE Models": [
                "Qwen/Qwen3-MoE-15B-A2B",
                "Qwen/Qwen3-MoE-32B-A2B",
            ]
        }
        
        # Model aliases for quick loading
        self.model_aliases = {
            # Qwen3 models
            "qwen3-0.6b": "Qwen/Qwen3-0.6B",
            "qwen3-1.7b": "Qwen/Qwen3-1.7B", 
            "qwen3-4b": "Qwen/Qwen3-4B",
            "qwen3-8b": "Qwen/Qwen3-8B",
            "qwen3-14b": "Qwen/Qwen3-14B",
            "qwen3-32b": "Qwen/Qwen3-32B",
            # Short aliases
            "0.6b": "Qwen/Qwen3-0.6B",
            "1.7b": "Qwen/Qwen3-1.7B",
            "4b": "Qwen/Qwen3-4B", 
            "8b": "Qwen/Qwen3-8B",
            "14b": "Qwen/Qwen3-14B",
            "32b": "Qwen/Qwen3-32B",
            # VL models  
            "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
            "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
            "qwen2.5-vl-32b": "Qwen/Qwen2.5-VL-32B-Instruct",
            "vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct", 
            "vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
            "vl-32b": "Qwen/Qwen2.5-VL-32B-Instruct",
            # MoE models
            "qwen3-moe-15b": "Qwen/Qwen3-MoE-15B-A2B",
            "qwen3-moe-32b": "Qwen/Qwen3-MoE-32B-A2B",
            "moe-15b": "Qwen/Qwen3-MoE-15B-A2B",
            "moe-32b": "Qwen/Qwen3-MoE-32B-A2B",
            "moe": "Qwen/Qwen3-MoE-15B-A2B",  # Default MoE
        }
    
    def setup_readline(self):
        """Set up readline for better input handling"""
        if not READLINE_AVAILABLE:
            return
            
        try:
            # Set up history file
            history_file = os.path.expanduser("~/.dark_rl_history")
            
            # Load existing history
            try:
                readline.read_history_file(history_file)
            except FileNotFoundError:
                pass  # No history file yet
            
            # Set history length
            readline.set_history_length(1000)
            
            # Save history on exit
            atexit.register(readline.write_history_file, history_file)
            
            # Set up tab completion
            readline.set_completer(self.complete_input)
            readline.parse_and_bind("tab: complete")
            
            # Enable better editing (emacs-style)
            readline.parse_and_bind("set editing-mode emacs")
            
            # Show completions without bell
            readline.parse_and_bind("set bell-style none")
            readline.parse_and_bind("set show-all-if-ambiguous on")
            
        except Exception as e:
            # Readline setup failed, but don't crash
            print(f"⚠️  Warning: Readline setup failed ({e}). Arrow keys may not work.")
    
    def complete_input(self, text, state):
        """Auto-completion function for readline"""
        options = []
        
        # Complete commands
        if text.startswith('/'):
            options = [cmd for cmd in self.commands if cmd.startswith(text)]
        
        # Complete model names for /load command
        elif readline.get_line_buffer().startswith('/load '):
            # Get what we're trying to complete
            line = readline.get_line_buffer()
            parts = line.split()
            if len(parts) >= 2:
                current_word = parts[-1] if not line.endswith(' ') else ''
                # Complete with model aliases
                options = [alias for alias in self.model_aliases.keys() 
                          if alias.startswith(current_word)]
        
        # Complete architecture names
        elif any(readline.get_line_buffer().startswith(cmd) for cmd in ['/load ', '/arch ']):
            if 'hf' not in readline.get_line_buffer() and 'dark' not in readline.get_line_buffer():
                options = ['hf', 'dark', 'auto']
        
        # Complete test topics
        elif readline.get_line_buffer().startswith('/test '):
            topics = ['science', 'history', 'literature', 'math', 'geography', 'current', 'philosophy']
            line = readline.get_line_buffer()
            parts = line.split()
            if len(parts) >= 2:
                current_word = parts[-1] if not line.endswith(' ') else ''
                options = [topic for topic in topics if topic.startswith(current_word)]
        
        try:
            return options[state]
        except IndexError:
            return None
    
    def print_banner(self):
        """Print welcome banner"""
        print("=" * 70)
        print("🧠 Dark.RL Model REPL - General Knowledge Testing")
        print("=" * 70)
        print("Test different models and architectures interactively!")
        print("Type /help for commands, /quit to exit")
        print("💡 Quick start: Use /switch to load a model, then start asking questions!")
        print("=" * 70)
    
    def print_help(self):
        """Print available commands"""
        print("\n📋 Available Commands:")
        print("  /help          - Show this help")
        print("  /quit          - Exit the REPL")
        print("  /models        - List available models")
        print("  /switch        - Interactive model selection ⭐ START HERE!")
        print("  /load <model>  - Load specific model (e.g. /load qwen3-8b)")
        print("  /arch          - Switch architecture (HF ↔ Custom) for current model")
        print("  /info          - Show current model info")
        print("  /compare       - Compare HF vs Custom for the same prompt")
        print("  /cleanup       - Free VRAM by unloading current model")
        print("  /vram          - Show current VRAM usage")
        print("  /stream        - Toggle streaming mode on/off")
        print("  /max [tokens]  - Set/view max generation tokens (e.g. /max 500)")
        print("  /temp [value]  - Set/view temperature (e.g. /temp 0.8)")
        print("  /clear         - Clear conversation history")
        print("  /history       - Show conversation history")
        print("  /test <topic>  - Quick knowledge test on a topic")
        print("\n🎯 Features:")
        print("  • Arrow keys for input editing and history navigation")
        print("  • Tab completion for commands, models, and topics")
        print("  • Persistent command history across sessions")
        print("  • Real-time streaming responses")
        print("\n💬 Chat (after loading a model):")
        print("  Just type your question and press Enter!")
        print("  Ask about science, history, current events, etc.")
        print("\n🚀 Quick Start:")
        print("  1. Type /switch for interactive selection, or /load <model> for direct")
        print("  2. Examples: /load qwen3-8b, /load qwen2.5-vl-7b")
        print("  3. Start asking questions!")
        print("  4. Use /arch to switch between HF and Custom implementations")
        print("  5. Use /compare to test both implementations with the same prompt")
        
        print("\n📝 Quick Load Examples:")
        print("  /load qwen3-8b      - Load Qwen3-8B")
        print("  /load qwen3-1.7b    - Load Qwen3-1.7B")
        print("  /load qwen2.5-vl-7b - Load Qwen2.5-VL-7B")
        print("  /load moe-15b       - Load Qwen3-MoE-15B")
        print("  /load 8b hf         - Load Qwen3-8B with HF architecture")
        print("  /load 8b dark       - Load Qwen3-8B with Custom architecture")
    
    def list_models(self):
        """List all available models"""
        print("\n🤖 Available Models:")
        for category, models in self.available_models.items():
            print(f"\n  📂 {category}:")
            for i, model in enumerate(models, 1):
                model_short = model.split('/')[-1]
                
                # Find aliases for this model
                aliases = [alias for alias, full_name in self.model_aliases.items() if full_name == model]
                alias_str = f" (aliases: {', '.join(aliases)})" if aliases else ""
                
                print(f"    {i}. {model_short}{alias_str}")
        
        print(f"\n💡 Quick Load Examples:")
        print(f"  • /load 8b          - Load Qwen3-8B (auto architecture)")
        print(f"  • /load qwen3-4b hf - Load Qwen3-4B with HF architecture")
        print(f"  • /load vl-7b       - Load Qwen2.5-VL-7B")
        print(f"  • /load moe-15b     - Load Qwen3-MoE-15B")
        print(f"  • /load 1.7b dark   - Load Qwen3-1.7B with Custom architecture")
    
    def show_info(self):
        """Show current model information"""
        if not self.llm:
            print("❌ No model loaded")
            return
        
        model_short = self.current_model.split('/')[-1]
        arch_name = "HuggingFace" if self.current_engine == "hf" else "Custom Dark.RL"
        
        print(f"\n🔍 Current Model Info:")
        print(f"  • Model: {model_short}")
        print(f"  • Engine: {self.current_engine} ({arch_name})")
        print(f"  • Temperature: {self.llm.temperature} (use /temp to change)")
        print(f"  • Max Tokens: {self.llm.max_tokens} (use /max to change)")
        
        # Engine-specific info
        if self.current_engine == "hf":
            print(f"  🚀 HF Features:")
            print(f"    - Flash Attention 2: ✅")
            print(f"    - Optimized inference: ✅")
            if "Qwen3" in self.current_model and hasattr(self.llm.hf_model, 'enable_thinking'):
                thinking_mode = getattr(self.llm.hf_model, 'enable_thinking', False)
                print(f"    - Thinking mode: {'✅' if thinking_mode else '❌'}")
        else:
            print(f"  🛠️  Custom Features:")
            print(f"    - Direct model control: ✅")
            print(f"    - Custom optimizations: ✅")
        
        # Model capabilities
        is_vl = "VL" in self.current_model
        is_moe = "MoE" in self.current_model
        print(f"  📋 Capabilities:")
        print(f"    - Text generation: ✅")
        print(f"    - Vision-language: {'✅' if is_vl else '❌'}")
        print(f"    - Mixture of Experts: {'✅' if is_moe else '❌'}")
        if is_moe:
            print(f"    - Number of experts: {getattr(self.llm.hf_model, 'num_experts', 'N/A')}")
            print(f"    - Experts per token: {getattr(self.llm.hf_model, 'num_experts_per_tok', 'N/A')}")
        print(f"    - LoRA adapters: {len(self.llm.list_adapters())} loaded")
        
        # Streaming info
        stream_status = "✅" if self.streaming_enabled else "❌"
        stream_type = ""
        if self.streaming_enabled:
            if self.current_architecture == "dark":
                stream_type = " (real token-by-token)"
            elif self.current_architecture == "hf":
                stream_type = " (simulated)"
        print(f"  🌊 Streaming: {stream_status}{stream_type}")
        
        # Architecture switching info
        if not is_vl and not is_moe:
            other_arch = "dark" if self.current_architecture == "hf" else "hf"
            print(f"  💡 Use /arch to switch to {other_arch} architecture")
        elif is_moe:
            print(f"  💡 MoE models only support dark architecture")
        print(f"  💡 Use /compare to test both architectures" if not is_vl and not is_moe else "")
        print(f"  💡 Use /stream to toggle streaming mode")
    
    def cleanup_model(self):
        """Clean up current model from VRAM"""
        if self.llm is not None:
            print("🧹 Cleaning up previous model from VRAM...")
            
            # Clear any CUDA tensors
            try:
                if hasattr(self.llm, 'hf_model') and self.llm.hf_model is not None:
                    # Move HF model to CPU and clear
                    if hasattr(self.llm.hf_model, 'cpu'):
                        self.llm.hf_model.cpu()
                    del self.llm.hf_model
                
                if hasattr(self.llm, 'llm') and self.llm.llm is not None:
                    # Clear custom model
                    del self.llm.llm
                
                # Clear the main LLM object
                del self.llm
                self.llm = None
                
            except Exception as e:
                print(f"⚠️  Cleanup warning: {e}")
            
            # Force garbage collection and clear CUDA cache
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Show memory freed
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"💾 VRAM after cleanup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    def show_vram_usage(self):
        """Show current VRAM usage"""
        if not torch.cuda.is_available():
            print("❌ CUDA not available - cannot show VRAM usage")
            return
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"\n💾 VRAM Usage:")
        print(f"  • Allocated: {allocated:.2f}GB ({allocated/total*100:.1f}%)")
        print(f"  • Reserved:  {reserved:.2f}GB ({reserved/total*100:.1f}%)")
        print(f"  • Total:     {total:.2f}GB")
        print(f"  • Free:      {total-reserved:.2f}GB")
        
        # Memory efficiency tip
        if reserved > allocated * 1.5:
            print(f"  💡 Tip: Use /cleanup to free unused reserved memory")
    
    async def manual_cleanup(self):
        """Manual VRAM cleanup command"""
        if self.llm is None:
            print("💾 No model loaded - cleaning up any remaining VRAM...")
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            self.show_vram_usage()
        else:
            print("🧹 Unloading current model and cleaning VRAM...")
            self.cleanup_model()
            self.current_model = None
            self.current_engine = None
            print("✅ Model unloaded. Use /load <model> to load a new model.")

    async def load_model(self, model_name: str, engine: str = None):
        """Load a specific model with given engine"""
        try:
            # Clean up previous model first
            if self.llm is not None:
                self.cleanup_model()
            
            print(f"🔄 Loading {model_name} with {engine or 'default'} engine...")
            start_time = time.time()
            
            # Smart defaults
            if engine is None:
                engine = "hf" if "VL" in model_name else "dark"
            
            # Use pending settings if available, otherwise defaults
            temperature = self._pending_temperature if self._pending_temperature is not None else 0.7
            max_tokens = self._pending_max_tokens if self._pending_max_tokens is not None else 150
            
            self.llm = OnlineLLM(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                engine=engine
            )
            
            load_time = time.time() - start_time
            self.current_model = model_name
            self.current_engine = engine
            
            print(f"✅ Model loaded successfully in {load_time:.2f}s")
            
            # Show VRAM usage after loading
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"💾 VRAM usage: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            
            self.show_info()
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.llm = None
    
    async def switch_model(self):
        """Interactive model switching"""
        print("\n🔄 Model Selection")
        self.list_models()
        
        # Get model choice
        print(f"\nCurrent: {self.current_model or 'None'}")
        model_input = input("\nEnter model name or number from list: ").strip()
        
        # Parse input
        selected_model = None
        
        # Check if it's a number
        try:
            choice_num = int(model_input)
            all_models = []
            for models in self.available_models.values():
                all_models.extend(models)
            if 1 <= choice_num <= len(all_models):
                selected_model = all_models[choice_num - 1]
        except ValueError:
            # Check if it's a partial name match
            for models in self.available_models.values():
                for model in models:
                    if model_input.lower() in model.lower():
                        selected_model = model
                        break
                if selected_model:
                    break
        
        if not selected_model:
            print("❌ Invalid model selection")
            return
        
        # Get architecture choice
        print(f"\n🏗️  Architecture for {selected_model.split('/')[-1]}:")
        print("  1. hf   - HuggingFace (optimized, flash attention, thinking mode for Qwen3)")
        print("  2. dark - Custom Dark.RL (custom implementation, maximum control)")
        print("  3. auto - Smart default (HF for VL models, Dark for text models)")
        
        arch_input = input("Choose architecture (1-3) [default: 3]: ").strip() or "3"
        architecture_map = {"1": "hf", "2": "dark", "3": None}
        architecture = architecture_map.get(arch_input, None)
        
        await self.load_model(selected_model, architecture)
    
    async def load_model_by_name(self, model_spec: str):
        """Load model by name/alias with optional architecture"""
        parts = model_spec.split()
        if len(parts) == 0:
            print("❌ No model specified. Use /load <model> [architecture]")
            return
        
        model_name = parts[0].lower()
        architecture = parts[1].lower() if len(parts) > 1 else None
        
        # Validate architecture if provided
        if architecture and architecture not in ["hf", "dark", "auto"]:
            print(f"❌ Invalid architecture '{architecture}'. Use 'hf', 'dark', or 'auto'")
            return
        
        # Convert "auto" to None for smart default
        if architecture == "auto":
            architecture = None
        
        # Find the model
        full_model_name = None
        
        # Check aliases first
        if model_name in self.model_aliases:
            full_model_name = self.model_aliases[model_name]
        else:
            # Try partial matching
            for alias, full_name in self.model_aliases.items():
                if model_name in alias or alias in model_name:
                    full_model_name = full_name
                    break
            
            # If still not found, try matching against full model names
            if not full_model_name:
                all_models = []
                for models in self.available_models.values():
                    all_models.extend(models)
                
                for model in all_models:
                    model_lower = model.lower()
                    if model_name in model_lower or any(part in model_lower for part in model_name.split('-')):
                        full_model_name = model
                        break
        
        if not full_model_name:
            print(f"❌ Model '{model_name}' not found")
            print("Available aliases:")
            for alias in sorted(self.model_aliases.keys()):
                print(f"  • {alias}")
            print("\nUse /models to see all available models")
            return
        
        # Load the model
        arch_info = f" with {architecture} architecture" if architecture else ""
        print(f"🚀 Loading {full_model_name.split('/')[-1]}{arch_info}...")
        await self.load_model(full_model_name, architecture)
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("🗑️  Conversation history cleared")
    
    def show_history(self):
        """Show conversation history"""
        if not self.conversation_history:
            print("📝 No conversation history")
            return
        
        print(f"\n📝 Conversation History ({len(self.conversation_history)} messages):")
        for i, (role, content, timestamp) in enumerate(self.conversation_history, 1):
            time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
            print(f"  {i}. [{time_str}] {role}: {content[:80]}{'...' if len(content) > 80 else ''}")
    
    async def switch_architecture(self):
        """Switch architecture for the current model"""
        if not self.llm or not self.current_model:
            print("❌ No model loaded. Use /switch to load a model first.")
            return
        
        # Check if model supports architecture switching
        if "VL" in self.current_model:
            print("❌ VL models only support HF architecture. Cannot switch.")
            return
        
        if "MoE" in self.current_model:
            print("❌ MoE models only support Custom Dark.RL architecture. Cannot switch.")
            return
        
        current_arch = self.current_architecture
        other_arch = "dark" if current_arch == "hf" else "hf"
        
        print(f"\n🔄 Architecture Switch for {self.current_model.split('/')[-1]}")
        print(f"Current: {current_arch} ({'HuggingFace' if current_arch == 'hf' else 'Custom Dark.RL'})")
        print(f"Switch to: {other_arch} ({'HuggingFace' if other_arch == 'hf' else 'Custom Dark.RL'})")
        
        # Architecture descriptions
        if other_arch == "hf":
            print("🚀 HuggingFace: Optimized implementation with Flash Attention 2")
            if "Qwen3" in self.current_model:
                print("   • Thinking mode support for Qwen3")
            print("   • Faster inference, better memory usage")
        else:
            print("🛠️  Custom Dark.RL: Maximum control and customization")
            print("   • Custom optimizations, direct control over model behavior")
        
        confirm = input(f"\nSwitch to {other_arch} architecture? [y/N]: ").strip().lower()
        if confirm in ['y', 'yes']:
            await self.load_model(self.current_model, other_arch)
        else:
            print("❌ Architecture switch cancelled")
    
    async def compare_architectures(self):
        """Compare HF vs Custom architectures with the same prompt"""
        if not self.llm or not self.current_model:
            print("❌ No model loaded. Use /switch to load a model first.")
            return
        
        # Check if this is a VL model (only supports HF)
        if "VL" in self.current_model:
            print("❌ VL models only support HF architecture. Cannot compare.")
            return
        
        # Check if this is a MoE model (only supports Custom)
        if "MoE" in self.current_model:
            print("❌ MoE models only support Custom Dark.RL architecture. Cannot compare.")
            return
        
        prompt = input("\n🔍 Enter prompt to compare across architectures: ").strip()
        if not prompt:
            print("❌ No prompt provided")
            return
        
        print(f"\n⚖️  Comparing architectures for: {self.current_model.split('/')[-1]}")
        print(f"Prompt: {prompt}")
        print("=" * 60)
        
        # Save current state
        original_arch = self.current_architecture
        results = {}
        
        # Test both architectures
        for arch in ["hf", "dark"]:
            try:
                print(f"\n🔄 Testing {arch.upper()} architecture...")
                
                # Load architecture if different from current
                if arch != self.current_architecture:
                    await self.load_model(self.current_model, arch)
                
                # Generate response
                start_time = time.time()
                response = await self.llm.generate_async(prompt)
                response_time = time.time() - start_time
                
                results[arch] = {
                    "response": response,
                    "time": response_time
                }
                
                # Show result
                arch_name = "HuggingFace" if arch == "hf" else "Custom Dark.RL"
                print(f"\n🤖 {arch_name} ({response_time:.2f}s):")
                print(f"{response}")
                
            except Exception as e:
                print(f"❌ Error with {arch} architecture: {e}")
                results[arch] = {"error": str(e)}
        
        # Restore original architecture
        if self.current_architecture != original_arch:
            print(f"\n🔄 Restoring {original_arch} architecture...")
            await self.load_model(self.current_model, original_arch)
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 Comparison Summary:")
        for arch, result in results.items():
            arch_name = "HuggingFace" if arch == "hf" else "Custom Dark.RL"
            if "error" in result:
                print(f"  • {arch_name}: ❌ {result['error']}")
            else:
                print(f"  • {arch_name}: ✅ {result['time']:.2f}s ({len(result['response'])} chars)")
        
        # Performance comparison
        if "hf" in results and "dark" in results:
            if "time" in results["hf"] and "time" in results["dark"]:
                hf_time = results["hf"]["time"]
                dark_time = results["dark"]["time"]
                if hf_time < dark_time:
                    speedup = dark_time / hf_time
                    print(f"🚀 HuggingFace is {speedup:.1f}x faster")
                else:
                    speedup = hf_time / dark_time
                    print(f"🛠️  Custom Dark.RL is {speedup:.1f}x faster")
    
    async def quick_test(self, topic: str):
        """Quick knowledge test on a topic"""
        if not self.llm:
            print("❌ No model loaded. Use /switch to load a model first.")
            return
        
        test_questions = {
            "science": "Explain quantum entanglement in simple terms.",
            "history": "What were the main causes of World War I?", 
            "literature": "Who wrote '1984' and what is its main theme?",
            "math": "Explain the Pythagorean theorem with an example.",
            "geography": "What are the seven continents and their characteristics?",
            "current": "What are some major technological trends in 2024?",
            "philosophy": "What is the difference between deontological and utilitarian ethics?"
        }
        
        question = test_questions.get(topic.lower()) or f"Tell me something interesting about {topic}."
        print(f"\n🧪 Quick Test - {topic.title()}:")
        print(f"Question: {question}")
        
        await self.ask_question(question, is_test=True)
    
    def toggle_streaming(self):
        """Toggle streaming mode on/off"""
        self.streaming_enabled = not self.streaming_enabled
        status = "enabled" if self.streaming_enabled else "disabled"
        print(f"🌊 Streaming mode {status}")
        
        if self.streaming_enabled:
            if self.current_architecture == "dark":
                print("  • Real token-by-token streaming for Custom architecture")
            elif self.current_architecture == "hf":
                print("  • Simulated streaming for HuggingFace architecture")
        else:
            print("  • Responses will be shown all at once")
    
    def set_max_tokens(self, tokens_str: str = None):
        """Set or view max tokens setting"""
        if not tokens_str:
            # Show current setting
            if self.llm:
                print(f"🔢 Current max tokens: {self.llm.max_tokens}")
            else:
                print("🔢 Max tokens: Not set (no model loaded)")
            print("💡 Use '/max <number>' to set max tokens (e.g. '/max 500')")
            return
        
        try:
            max_tokens = int(tokens_str)
            if max_tokens < 1:
                print("❌ Max tokens must be at least 1")
                return
            
            if max_tokens > 4096:
                print("⚠️  Warning: Very high max tokens may use significant memory")
                confirm = input(f"Set max tokens to {max_tokens}? [y/N]: ").strip().lower()
                if confirm not in ['y', 'yes']:
                    print("❌ Max tokens not changed")
                    return
            
            if self.llm:
                self.llm.max_tokens = max_tokens
                print(f"✅ Max tokens set to {max_tokens}")
            else:
                print(f"✅ Max tokens will be set to {max_tokens} when model is loaded")
                # Store for next model load
                self._pending_max_tokens = max_tokens
            
        except ValueError:
            print(f"❌ Invalid number: '{tokens_str}'. Use a positive integer.")
    
    def set_temperature(self, temp_str: str = None):
        """Set or view temperature setting"""
        if not temp_str:
            # Show current setting
            if self.llm:
                print(f"🌡️  Current temperature: {self.llm.temperature}")
            else:
                print("🌡️  Temperature: Not set (no model loaded)")
            print("💡 Use '/temp <number>' to set temperature (e.g. '/temp 0.8')")
            print("💡 Temperature ranges: 0.0 (deterministic) to 1.0+ (creative)")
            return
        
        try:
            temperature = float(temp_str)
            if temperature < 0:
                print("❌ Temperature must be non-negative")
                return
            
            if temperature > 2.0:
                print("⚠️  Warning: Very high temperature may produce incoherent text")
                confirm = input(f"Set temperature to {temperature}? [y/N]: ").strip().lower()
                if confirm not in ['y', 'yes']:
                    print("❌ Temperature not changed")
                    return
            
            if self.llm:
                self.llm.temperature = temperature
                print(f"✅ Temperature set to {temperature}")
            else:
                print(f"✅ Temperature will be set to {temperature} when model is loaded")
                # Store for next model load
                self._pending_temperature = temperature
            
        except ValueError:
            print(f"❌ Invalid number: '{temp_str}'. Use a decimal number (e.g. 0.8).")
    
    async def stream_tokens_custom(self, question: str, lora_adapter: str = None):
        """Real streaming for custom implementation"""
        if not self.llm or not self.llm.llm:
            raise Exception("Custom LLM not available for streaming")
        
        from dark.engine.sequence import Sequence
        
        # Set up sampling params
        sampling_params = self.llm.default_sampling_params
        
        # Load LoRA adapter if specified
        if lora_adapter and lora_adapter in self.llm.lora_states:
            async with self.llm.lock:
                self.llm.load_lora_state(self.llm.lora_states[lora_adapter])
        
        # Create sequence
        tokenized = self.llm.tokenizer.encode(question)
        seq = Sequence(tokenized, sampling_params)
        
        async with self.llm.lock:
            self.llm.llm.eval()
            self.llm.llm.scheduler.add(seq)
        
        response = ""
        emitted = 0
        
        # Stream generation step by step
        while not seq.is_finished:
            async with self.llm.lock:
                self.llm.llm.eval()
                with torch.cuda.stream(self.llm.infer_stream):
                    self.llm.llm.step()
            
            # Emit new tokens as they're generated
            while seq.num_completion_tokens > emitted:
                tid = seq.completion_token_ids[emitted]
                new_token = self.llm.tokenizer.decode([tid], skip_special_tokens=True)
                print(new_token, end='', flush=True)
                response += new_token
                emitted += 1
            
            await asyncio.sleep(0.01)  # Small delay for better streaming experience
        
        return response
    
    async def ask_question(self, question: str, is_test: bool = False, lora_adapter: str = None):
        """Ask a question to the current model"""
        if not self.llm:
            print("❌ No model loaded. Use /switch to load a model first.")
            return
        
        # Clean the question to avoid contamination
        question = question.strip()
        if not question:
            print("❌ Empty question")
            return
        
        try:
            start_time = time.time()
            model_name = self.current_model.split('/')[-1]
            print(f"\n🤖 {model_name}:")
            
            response = ""
            use_streaming = self.streaming_enabled
            
            if use_streaming and self.current_architecture == "dark":
                # Real streaming for custom implementation
                try:
                    response = await self.stream_tokens_custom(question, lora_adapter)
                    print()  # New line after streaming
                except Exception as stream_error:
                    print(f"\n⚠️  Custom streaming failed: {stream_error}")
                    # Fallback to standard generation
                    response = await self.llm.generate_async(question, lora_adapter=lora_adapter)
                    print(response)
            
            elif use_streaming and self.current_architecture == "hf":
                # Simulated streaming for HF implementation
                try:
                    print("💭 ", end='', flush=True)
                    response = await self.llm.generate_async(question, lora_adapter=lora_adapter)
                    
                    # Simulate streaming output
                    for i, char in enumerate(response):
                        print(char, end='', flush=True)
                        if i % 3 == 0:  # Add small delays every few characters
                            await asyncio.sleep(0.02)
                    print()  # New line after streaming
                    
                except Exception as hf_error:
                    print(f"\n⚠️  HF generation failed: {hf_error}")
                    response = "Sorry, I encountered an error generating a response."
            
            else:
                # Non-streaming response
                print("🤔 Thinking...")
                response = await self.llm.generate_async(question, lora_adapter=lora_adapter)
                print(response)
            
            response_time = time.time() - start_time
            print(f"\n⏱️  Response time: {response_time:.2f}s")
            
            # Save to history
            if not is_test:
                timestamp = time.time()
                self.conversation_history.append(("You", question, timestamp))
                self.conversation_history.append(("Assistant", response, timestamp))
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            # Don't show full traceback by default to keep it clean
            if "debug" in str(e).lower():
                import traceback
                print(f"Debug traceback: {traceback.format_exc()}")
    
    async def run(self):
        """Main REPL loop"""
        self.print_banner()
        
        print("💡 No model loaded yet. Use /switch to load a model, or /help for commands.")
        
        while True:
            try:
                # Get user input
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    command = user_input[1:].lower()
                    
                    if command == "quit" or command == "exit":
                        print("👋 Goodbye!")
                        break
                    elif command == "help":
                        self.print_help()
                    elif command == "models":
                        self.list_models()
                    elif command == "switch":
                        await self.switch_model()
                    elif command.startswith("load "):
                        model_spec = command[5:].strip()
                        await self.load_model_by_name(model_spec)
                    elif command == "load":
                        print("❌ No model specified.")
                        print("Usage: /load <model> [architecture]")
                        print("Examples:")
                        print("  • /load 8b")
                        print("  • /load qwen3-4b hf") 
                        print("  • /load vl-7b")
                        print("Use /models to see all available models and aliases")
                    elif command == "arch":
                        await self.switch_architecture()
                    elif command == "compare":
                        await self.compare_architectures()
                    elif command == "cleanup":
                        await self.manual_cleanup()
                    elif command == "vram":
                        self.show_vram_usage()
                    elif command == "stream":
                        self.toggle_streaming()
                    elif command.startswith("max"):
                        if command == "max":
                            self.set_max_tokens()
                        elif command.startswith("max "):
                            tokens = command[4:].strip()
                            self.set_max_tokens(tokens)
                    elif command.startswith("temp"):
                        if command == "temp":
                            self.set_temperature()
                        elif command.startswith("temp "):
                            temp = command[5:].strip()
                            self.set_temperature(temp)
                    elif command == "info":
                        self.show_info()
                    elif command == "clear":
                        self.clear_history()
                    elif command == "history":
                        self.show_history()
                    elif command.startswith("test "):
                        topic = command[5:].strip()
                        if topic:
                            await self.quick_test(topic)
                        else:
                            print("❌ Please specify a topic. Example: /test science")
                    else:
                        print(f"❌ Unknown command: /{command}")
                        print("Type /help for available commands")
                else:
                    # Regular question
                    await self.ask_question(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! (Ctrl+C pressed)")
                break
            except EOFError:
                print("\n\n👋 Goodbye! (EOF)")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")


def print_cli_help():
    """Print command line help"""
    print("""
🧠 Dark.RL Model REPL - General Knowledge Testing

Usage:
    python model_repl.py [options]

Options:
    -h, --help     Show this help message and exit
    
Interactive Commands (once in REPL):
    /help          Show interactive help
    /switch        Load a model
    /models        List available models  
    /quit          Exit the REPL

Examples:
    python model_repl.py            # Start interactive REPL
    uv run python model_repl.py     # Start with uv environment
    
Once in the REPL, use /switch to load a model and start chatting!
    """)

async def main():
    """Main entry point"""
    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help', 'help']:
            print_cli_help()
            return
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information")
            return
    
    repl = ModelREPL()
    await repl.run()


if __name__ == "__main__":
    # Handle Ctrl+C gracefully
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0) 