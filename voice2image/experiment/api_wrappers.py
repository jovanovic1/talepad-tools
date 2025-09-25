# api_wrappers.py
import os
import time
import asyncio
import requests
from model_interfaces import TtiApi
from data_models import ExperimentResult
import base64

# --- OpenAI DALL-E 3 Implementation ---
class OpenAITtiApi(TtiApi):
    """Wrapper for OpenAI DALL-E 3."""
    def __init__(self, model_name="dall-e-3"):
        super().__init__(model_name=model_name, provider="openai")
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    async def generate_image(self, prompt: str) -> ExperimentResult:
        start_time = time.monotonic()
        
        # NOTE: DALL-E 3 has a built-in cost ($0.04/image for 1024x1024 standard)
        cost = 0.04
        
        try:
            response = await self.client.images.generate(
                model=self.model_name,
                prompt=f"Cartoon, Storybook style, for a 6-12 year old - {prompt}",
                size="1024x1024",
                n=1,
                response_format="url"
            )
            image_result = response.data[0].url
            
        except Exception as e:
            image_result = ""
            error_msg = str(e)
            
        latency_ms = (time.monotonic() - start_time) * 1000

        return ExperimentResult(
            provider=self.provider,
            model_name=self.model_name,
            final_prompt=prompt,
            latency_tti_ms=latency_ms,
            image_url_or_data=image_result,
            cost_per_run_usd=cost,
            error_message=error_msg if 'error_msg' in locals() else ""
        )

# --- Stability AI SD 3.5 Turbo Implementation (Based on REST Fix) ---
# --- New Function: Saves Base64 Image to Disk ---
def save_base64_image(base64_string: str, provider: str, run_id: str) -> str:
    """Decodes a Base64 string and saves it to a unique file path."""
    
    # 1. Define the save path
    SAVE_DIR = "experiment_images"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 2. Create unique filename
    file_name = f"{provider}_{run_id}_{int(time.time())}.webp"
    file_path = os.path.join(SAVE_DIR, file_name)
    
    # 3. Decode and Save
    try:
        binary_data = base64.b64decode(base64_string)
        with open(file_path, 'wb') as f:
            f.write(binary_data)
        return file_path
    except Exception as e:
        print(f"⚠️ ERROR saving image {file_name}: {e}")
        return ""


# --- Stability AI SD 3.5 Turbo Implementation ---
class StabilityAITtiApi(TtiApi):
    """Wrapper for Stability AI's SD 3.5 Turbo (REST v2beta)."""
    STABILITY_API_URL = "https://api.stability.ai/v2beta/stable-image/generate/ultra"
    
    def __init__(self, model_name="stable-diffusion-3.5-large-turbo"):
        super().__init__(model_name=model_name, provider="stability_ai")
        self.api_key = os.environ.get("STABILITY_KEY")
        self.cost = 0.04 # Conceptual cost

    async def generate_image(self, prompt: str) -> ExperimentResult:
        start_time = time.monotonic()
        image_path = ""
        base64_string = ""
        error_msg = ""
        
        # Define payload (multipart/form-data)
        data = {
            "prompt": f"**Cartoon, Vibrant Storybook Art** - {prompt}",
            "model": self.model_name,
            "output_format": "webp",
            "aspect_ratio": "1:1",
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json" # Request Base64 JSON
        }
        files = {"none": (None, '')}

        def sync_call():
            # Use data=data and files=files for multipart/form-data
            return requests.post(self.STABILITY_API_URL, headers=headers, files=files, data=data)

        try:
            response = await asyncio.to_thread(sync_call)

            if response.status_code == 200:
                response_data = response.json()
                base64_string = response_data['image']
                
                # --- 🔑 CORE CHANGE: Save the image locally ---
                # We use a placeholder run_id, which will be updated by the runner later
                image_path = await asyncio.to_thread(
                    save_base64_image, 
                    base64_string, 
                    self.provider, 
                    str(time.time_ns()) # Use nanoseconds for a unique ID placeholder
                )
                
            else:
                error_msg = f"API Error {response.status_code}: {response.text[:100]}"
                
        except Exception as e:
            error_msg = str(e)
            
        latency_ms = (time.monotonic() - start_time) * 1000

        return ExperimentResult(
            provider=self.provider,
            model_name=self.model_name,
            final_prompt=prompt,
            latency_tti_ms=latency_ms,
            # Store the local file path as the image result identifier
            image_url_or_data=image_path, 
            cost_per_run_usd=self.cost,
            error_message=error_msg
        )