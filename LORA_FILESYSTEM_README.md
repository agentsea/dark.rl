# LoRA Filesystem Persistence

The OnlineLLM now supports automatic filesystem persistence for LoRA adapters, allowing them to be saved to disk after training and automatically loaded when requested.

## Quick Start

### 1. Set the Save Path

Set the `LORA_SAVE_PATH` environment variable to specify where LoRA adapters should be saved:

```bash
export LORA_SAVE_PATH="/path/to/your/lora/adapters"
```

Or in Python:
```python
import os
os.environ['LORA_SAVE_PATH'] = "/path/to/your/lora/adapters"
```

### 2. Use OnlineLLM as Normal

```python
from src.dark.online_llm import AsyncOnlineLLM

llm = AsyncOnlineLLM(model="Qwen/Qwen3-0.6B")

# Train an adapter - it will be automatically saved to disk
await llm.learn([
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."}
], adapter="math_expert")

# Use the adapter - loads from memory or disk automatically
response = await llm.generate("What is 5+5?", adapter="math_expert")
```

## How It Works

### Automatic Saving
- When a LoRA adapter finishes training, it's automatically saved to disk if `LORA_SAVE_PATH` is set
- Each adapter is saved as `{adapter_name}.pt` in the specified directory
- Includes metadata (model path, LoRA rank/alpha, timestamp) for compatibility checking

### Automatic Loading
- When you request an adapter that's not in memory, the system checks the filesystem
- If found on disk, the adapter is automatically loaded into memory
- If not found anywhere, a new adapter is created

### Adapter Lookup Priority
1. **Memory**: Check if adapter is already loaded in GPU/CPU memory
2. **Disk**: Check if adapter exists in `LORA_SAVE_PATH` directory
3. **New**: Create a new adapter if not found anywhere

## File Structure

```
$LORA_SAVE_PATH/
├── math_expert.pt
├── science_expert.pt
├── cooking_expert.pt
└── ...
```

Each `.pt` file contains:
- `lora_state`: The actual LoRA parameters
- `metadata`: Model compatibility information and timestamps

## API Reference

### New Methods

#### `ensure_lora_adapter(adapter: str) -> bool`
Ensures an adapter is available in memory, loading from disk if necessary.

```python
# Returns True if adapter was found (memory or disk), False if new
found = llm.ensure_lora_adapter("my_adapter")
```

#### `save_lora_to_disk(adapter: str, lora_state: Optional[Dict] = None) -> bool`
Manually save an adapter to disk.

```python
# Save current adapter state
success = llm.save_lora_to_disk("my_adapter")
```

#### `load_lora_from_disk(adapter: str) -> Optional[Dict]`
Manually load an adapter from disk.

```python
# Load adapter state from disk
state = llm.load_lora_from_disk("my_adapter")
```

#### `list_disk_adapters() -> List[str]`
List all adapters available on disk.

```python
# Get all disk adapters
disk_adapters = llm.list_disk_adapters()
```

#### `get_adapter_info(adapter: str) -> Dict`
Get detailed information about an adapter.

```python
info = llm.get_adapter_info("my_adapter")
print(f"In memory: {info['in_memory']}")
print(f"On disk: {info['on_disk']}")
print(f"Parameters: {info.get('param_count', 'N/A')}")
```

### Enhanced Methods

#### `list_adapters() -> List[str]`
Now returns all adapters (memory + disk).

#### `delete_adapter(adapter: str, delete_from_disk: bool = False) -> bool`
Can now optionally delete from disk as well.

```python
# Delete from memory only
llm.delete_adapter("my_adapter")

# Delete from both memory and disk
llm.delete_adapter("my_adapter", delete_from_disk=True)
```

#### `stats(show_detailed: bool = True)`
Enhanced to show disk information and adapter locations.

## Configuration

### Environment Variables

- `LORA_SAVE_PATH`: Directory path for saving LoRA adapters
  - If not set, adapters are only kept in memory
  - Directory is created automatically if it doesn't exist

### Metadata Validation

When loading from disk, the system validates:
- **Model compatibility**: Warns if adapter was trained on different model
- **LoRA configuration**: Warns if rank/alpha mismatch
- **File integrity**: Handles corrupted files gracefully

## Example Use Cases

### 1. Persistent Training Sessions
```python
# Session 1: Train adapters
await llm.learn(math_examples, adapter="math_expert")
await llm.learn(science_examples, adapter="science_expert")
# Adapters automatically saved to disk

# Session 2: Use pre-trained adapters (different process/restart)
llm = AsyncOnlineLLM(model="Qwen/Qwen3-0.6B")
response = await llm.generate("What is calculus?", adapter="math_expert")
# Automatically loads from disk
```

### 2. Adapter Sharing
```python
# Save adapters in shared location
os.environ['LORA_SAVE_PATH'] = "/shared/lora/adapters"

# Multiple processes can access the same adapters
llm1 = AsyncOnlineLLM(model="Qwen/Qwen3-0.6B")
llm2 = AsyncOnlineLLM(model="Qwen/Qwen3-0.6B")

# Both can use the same pre-trained adapters
response1 = await llm1.generate("Math question", adapter="shared_math")
response2 = await llm2.generate("Science question", adapter="shared_science")
```

### 3. Adapter Management
```python
# List all available adapters
all_adapters = llm.list_adapters()
for adapter in all_adapters:
    info = llm.get_adapter_info(adapter)
    print(f"{adapter}: {'Memory' if info['in_memory'] else ''} "
          f"{'Disk' if info['on_disk'] else ''}")

# Clean up old adapters
llm.delete_adapter("old_adapter", delete_from_disk=True)
```

## Best Practices

1. **Set `LORA_SAVE_PATH` early**: Before creating your OnlineLLM instance
2. **Use meaningful adapter names**: They become filenames
3. **Monitor disk space**: LoRA adapters can accumulate over time
4. **Backup important adapters**: Copy `.pt` files for safekeeping
5. **Validate compatibility**: Check warnings when loading cross-model adapters

## Troubleshooting

### Common Issues

1. **Adapters not saving**: Check that `LORA_SAVE_PATH` is set and writable
2. **Permission errors**: Ensure the save directory has write permissions
3. **Compatibility warnings**: Normal when using adapters across different model configurations
4. **Disk space**: Monitor disk usage as adapters accumulate

### Debugging

Enable debug logging to see filesystem operations:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will show:
- When adapters are saved to disk
- When adapters are loaded from disk
- File operation errors and warnings 