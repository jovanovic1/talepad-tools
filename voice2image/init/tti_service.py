# tti_service.py
import asyncio
import time
from openai import AsyncOpenAI

# --- CONFIGURATION ---
# Ensure your OpenAI API key is set as an environment variable or passed securely.
# For testing, you can place it here, but **NEVER in production code**.
# Example: client = AsyncOpenAI(api_key="YOUR_OPENAI_KEY")
client = AsyncOpenAI() # Reads OPENAI_API_KEY from environment variable by default

# Choose the TTI model for this test. DALL-E 3 is recommended for quality.
TTI_MODEL = "dall-e-3"

async def generate_image(
    prompt: str,
    model: str = TTI_MODEL,
    size: str = "1024x1024",
    quality: str = "standard",
    style: str = "vivid", # or "natural"
    response_format: str = "url" # or "b64_json" if you want to embed images directly
) -> str:
    """
    Asynchronously generates an image using the specified Text-to-Image model (e.g., DALL-E 3).

    Args:
        prompt (str): The text description for the image.
        model (str): The TTI model to use (e.g., "dall-e-3").
        size (str): The resolution of the generated image (e.g., "1024x1024").
        quality (str): The quality of the image ("standard" or "hd").
        style (str): The style of the image ("vivid" or "natural").
        response_format (str): The format of the response ("url" or "b64_json").

    Returns:
        str: The URL of the generated image, or an empty string if generation fails.
    """
    start_time = time.monotonic() # Use monotonic for reliable time differences
    print(f"[TTI Service] ⏳ Starting image generation for prompt: '{prompt[:70]}...'")

    try:
        response = await client.images.generate(
            model=model,
            prompt=f"**Cartoon, Storybook Art, for a 6-12 year old, highly detailed** - {prompt}", # Emphasize style for target audience
            size=size,
            quality=quality,
            style=style,
            n=1, # Generate one image
            response_format=response_format
        )
        
        if response_format == "url":
            image_result = response.data[0].url
        else: # b64_json
            image_result = response.data[0].b64_json
            # In a real app, you might decode and save this to a file or serve it directly.
            
        end_time = time.monotonic()
        latency_ms = (end_time - start_time) * 1000
        print(f"[TTI Service] ✅ Image generated in {latency_ms:.2f}ms. Result (URL/b64_json): {image_result[:100]}...")
        return image_result

    except Exception as e:
        end_time = time.monotonic()
        latency_ms = (end_time - start_time) * 1000
        print(f"[TTI Service] ❌ Image generation failed after {latency_ms:.2f}ms: {e}")
        return ""

# --- Simple Test Function for the TTI Service ---
async def main():
    print("--- Testing TTI Service ---")
    
    test_prompts = [
        "A friendly blue monster playing hide-and-seek in a magical forest.",
        "A brave space explorer riding a unicorn on a rainbow bridge.",
        "A cute robot making a giant sandwich for its friends."
    ]

    for i, prompt in enumerate(test_prompts):
        print(f"\nTest {i+1}:")
        image_url = await generate_image(prompt)
        if image_url:
            print(f"Generated Image URL: {image_url}")
            # You might want to open this URL in a browser for visual inspection
            # import webbrowser
            # webbrowser.open(image_url)
        else:
            print("Image generation failed for this prompt.")

if __name__ == "__main__":
    # To run this, ensure you have set your OPENAI_API_KEY environment variable.
    # Example (in your terminal): export OPENAI_API_KEY="sk-..."
    asyncio.run(main())