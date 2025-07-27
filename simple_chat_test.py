import asyncio
import torch
from src.dark.online_llm import AsyncOnlineLLM

async def main():
    """
    A simple integration test for the AsyncOnlineLLM chat and SFT learn methods.
    """
    print("--- Initializing AsyncOnlineLLM with a small model for SFT test ---")
    
    try:
        # Use a smaller model for faster testing
        model_name = "Qwen/Qwen3-8B"
        
        llm = AsyncOnlineLLM(
            model=model_name,
            engine="dark",  # Use the custom engine
        )
        print(f"✅ LLM initialized with model: {model_name}")
    except Exception as e:
        print(f"❌ Failed to initialize LLM: {e}")
        return


    prompt = "What's your favorite food?"
    messages = [{"role": "user", "content": prompt}]

    try:
        print(f"User: {prompt}")
        print("Assistant: ", end="", flush=True)
        
        initial_response = ""
        async for chunk in llm.chat_stream(messages):
            initial_response += chunk
            print(chunk, end="", flush=True)
        
        print("\n✅ Initial inference completed.")
        print(f"Initial response: '{initial_response}'")

    except Exception as e:
        print(f"\n❌ Error during initial chat_stream(): {e}")

    # --- 1. Test the learn() method with SFT ---
    print("\n--- Testing learn() method (SFT) ---")
    good_conversation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "It's a hamburger pizza sandwich of course!"}
    ]

    try:
        # With no trainer, this uses the internal SFT fine-tuning
        await llm.learn(good_conversation, adapter="food", steps=10, lr=1e-4)
        print("✅ learn() executed for 'food' adapter using SFT.")
    except Exception as e:
        print(f"❌ Error during learn(): {e}")

    # Give some time for training to complete
    await asyncio.sleep(1)

    # --- 2. Test inference with chat_stream() ---
    print("\n--- Testing chat_stream() for inference ---")
    
    try:
        print(f"User: {prompt}")
        print("Assistant: ", end="", flush=True)
        
        response = ""
        async for chunk in llm.chat_stream(messages, adapter="food"):
            response += chunk
            print(chunk, end="", flush=True)
        
        print("\n✅ Post training inference completed.")

        expected_response = "It's a hamburger pizza sandwich of course!"
        if response.strip() == expected_response:
            print("✅ Post training inference is correct.")
        else:
            print("❌ Post training inference is incorrect.")
            print(f"Expected: '{expected_response}'")
            print(f"Actual:   '{response.strip()}'")

    except Exception as e:
        print(f"\n❌ Error during chat_stream(): {e}")

    # --- 3. Clean up ---
    print("\n--- Test finished ---")
    del llm
    torch.cuda.empty_cache()


if __name__ == "__main__":
    asyncio.run(main()) 