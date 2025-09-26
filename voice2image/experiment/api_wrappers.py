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

    async def generate_image(self, prompt: str, test_case_id: str = "T_unknown") -> ExperimentResult:
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
            execution_type="api",
            gpu_info="N/A (Cloud API)",
            latency_tti_ms=latency_ms,
            image_url_or_data=image_result,
            cost_per_run_usd=cost,
            error_message=error_msg if 'error_msg' in locals() else ""
        )

# --- Stability AI SD 3.5 Turbo Implementation (Based on REST Fix) ---
# --- Enhanced Function: Saves Base64 Image to Run-Specific Directory ---
def save_base64_image(base64_string: str, provider: str, model_name: str, test_case_id: str, run_session_id: str) -> str:
    """
    Decodes a Base64 string and saves it to a run-specific directory with meaningful naming.

    Args:
        base64_string: The base64 encoded image data
        provider: Provider name (e.g., 'local_gpu_a100', 'openai')
        model_name: Model name (e.g., 'sdxl-turbo', 'dall-e-3')
        test_case_id: Test case identifier (e.g., 'P1', 'P2')
        run_session_id: Unique session ID for this benchmark run

    Returns:
        str: Relative path to saved image file
    """
    from datetime import datetime

    # 1. Create run-specific directory structure
    BASE_DIR = "experiment_images"
    date_str = datetime.now().strftime("%Y%m%d")
    run_dir = os.path.join(BASE_DIR, f"run_{date_str}_{run_session_id}")
    os.makedirs(run_dir, exist_ok=True)

    # 2. Create meaningful filename
    # Clean model name for filename (remove slashes, special chars)
    clean_model_name = model_name.replace("/", "_").replace("-", "_")
    timestamp = int(time.time())
    file_name = f"{provider}_{clean_model_name}_{test_case_id}_{timestamp}.webp"
    file_path = os.path.join(run_dir, file_name)

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

    async def generate_image(self, prompt: str, test_case_id: str = "T_unknown") -> ExperimentResult:
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
                image_path = await asyncio.to_thread(
                    save_base64_image,
                    base64_string,
                    self.provider,
                    self.model_name,
                    test_case_id,
                    self.run_session_id or 'session_unknown'
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
            execution_type="api",
            gpu_info="N/A (Cloud API)",
            latency_tti_ms=latency_ms,
            # Store the local file path as the image result identifier
            image_url_or_data=image_path,
            cost_per_run_usd=self.cost,
            error_message=error_msg
        )

class LocalSDXLTurbo(TtiApi):
    """Wrapper for local SDXL Turbo inference on A100."""

    def __init__(self, model_name="stabilityai/sdxl-turbo"):
        import torch
        from diffusers import AutoPipelineForText2Image

        super().__init__(model_name=model_name, provider="local_gpu_a100")
        self.cost = 0.001 # Estimate for electricity/depreciation (negligible)
        self.device = "cuda"
        
        if not torch.cuda.is_available():
            raise EnvironmentError("PyTorch CUDA not available. Cannot run local GPU test.")
        
        # 1. Load the model with optimizations (only runs once)
        print(f"Loading {self.model_name} to GPU (A100) with FP16...")
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,   # Use half-precision for speed/VRAM
            variant="fp16",
            use_safetensors=True
        )
        self.pipe.to(self.device)
        
        # 2. Apply Speed Enhancements
        try:
            # Enable xformers for memory-efficient attention (massive speedup)
            self.pipe.enable_xformers_memory_efficient_attention() 
            print("xFormers enabled.")
        except Exception:
            print("xFormers failed to load. Running without it.")

        # 3. Compile UNet (PyTorch 2.0+ optimization - HUGE speedup after first run)
        # Note: The first run will be slow due to compilation overhead.
        # self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)
        # We skip this for now to keep the first inference time predictable.
        
        print("Local model loaded successfully.")

    async def generate_image(self, prompt: str, test_case_id: str = "T_unknown") -> ExperimentResult:
        start_time = time.monotonic()
        image_result = ""
        error_msg = ""
        
        # Define the synchronous inference function
        def sync_inference(p):
            from io import BytesIO
            
            # --- Aggressive Speed Settings for Turbo ---
            image = self.pipe(
                prompt=f"Cartoon, Storybook, fun, colorful - {p}",
                num_inference_steps=2,      # Key to Turbo speed
                guidance_scale=0.0,         # Key to Turbo speed
                width=1280,
                height=640
            ).images[0]
            
            # Convert PIL image to Base64 (serving format)
            buffer = BytesIO()
            # Save as WEBP for smaller size / faster transfer if sent over a network later
            image.save(buffer, format="WEBP") 
            base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Return a Data URL for the experiment result storage
            # return f"data:image/webp;base64,{base64_data[:30]}..." 
            return base64_data  # Return just the Base64 string for local saving
            
        try:
            # Run the synchronous GPU work on a separate thread
            image_result = await asyncio.to_thread(sync_inference, prompt) 
            
            # Since the image is generated locally, we save a file in the same thread:
            # This is a placeholder, as the runner should handle saving, but we can verify it here
            print(f"[Local GPU] Generated image data URL starting with: {image_result[:30]}")

            image_path = await asyncio.to_thread(
                save_base64_image,
                image_result,
                self.provider,
                self.model_name,
                test_case_id,
                self.run_session_id or 'session_unknown'
            )

        except Exception as e:
            error_msg = str(e)
            
        latency_ms = (time.monotonic() - start_time) * 1000

        return ExperimentResult(
            provider=self.provider,
            model_name=self.model_name,
            final_prompt=prompt,
            execution_type="local",
            gpu_info="NVIDIA A100-SXM4-40GB",
            latency_tti_ms=latency_ms,
            image_url_or_data=image_result,
            cost_per_run_usd=self.cost,
            error_message=error_msg
        )


# --- SDXL Lightning Implementation ---
class LocalSDXLLightning(TtiApi):
    """Wrapper for local SDXL Lightning inference on A100."""

    def __init__(self, model_name="ByteDance/SDXL-Lightning", steps=4):
        import torch
        from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler

        super().__init__(model_name=model_name, provider="local_gpu_a100")
        self.cost = 0.001
        self.device = "cuda"
        self.steps = steps

        if not torch.cuda.is_available():
            raise EnvironmentError("PyTorch CUDA not available.")

        print(f"Loading {self.model_name} Lightning ({steps}-step) to GPU...")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config, timestep_spacing="trailing"
        )
        self.pipe.to(self.device)

        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("xFormers enabled for Lightning.")
        except Exception:
            print("xFormers failed for Lightning.")

    async def generate_image(self, prompt: str, test_case_id: str = "T_unknown") -> ExperimentResult:
        start_time = time.monotonic()
        image_result = ""
        error_msg = ""

        def sync_inference(p):
            from io import BytesIO

            image = self.pipe(
                prompt=f"Cartoon, Storybook, colorful - {p}",
                num_inference_steps=self.steps,
                guidance_scale=7.5,
                width=1024,
                height=1024
            ).images[0]

            buffer = BytesIO()
            image.save(buffer, format="WEBP")
            base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return base64_data

        try:
            image_result = await asyncio.to_thread(sync_inference, prompt)
            image_path = await asyncio.to_thread(
                save_base64_image,
                image_result,
                self.provider,
                self.model_name,
                test_case_id,
                self.run_session_id or 'session_unknown'
            )
        except Exception as e:
            error_msg = str(e)

        latency_ms = (time.monotonic() - start_time) * 1000

        return ExperimentResult(
            provider=self.provider,
            model_name=f"{self.model_name}-{self.steps}step",
            final_prompt=prompt,
            execution_type="local",
            gpu_info="NVIDIA A100-SXM4-40GB",
            latency_tti_ms=latency_ms,
            image_url_or_data=image_result,
            cost_per_run_usd=self.cost,
            error_message=error_msg
        )


# --- SDXL Base Implementation ---
class LocalSDXLBase(TtiApi):
    """Wrapper for local SDXL Base 1.0 inference on A100."""

    def __init__(self, model_name="stabilityai/stable-diffusion-xl-base-1.0", steps=25):
        import torch
        from diffusers import StableDiffusionXLPipeline

        super().__init__(model_name=model_name, provider="local_gpu_a100")
        self.cost = 0.002  # Slightly higher due to more compute
        self.device = "cuda"
        self.steps = steps

        if not torch.cuda.is_available():
            raise EnvironmentError("PyTorch CUDA not available.")

        print(f"Loading {self.model_name} Base ({steps}-step) to GPU...")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        self.pipe.to(self.device)

        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("xFormers enabled for SDXL Base.")
        except Exception:
            print("xFormers failed for SDXL Base.")

    async def generate_image(self, prompt: str, test_case_id: str = "T_unknown") -> ExperimentResult:
        start_time = time.monotonic()
        image_result = ""
        error_msg = ""

        def sync_inference(p):
            from io import BytesIO

            image = self.pipe(
                prompt=f"Cartoon, Storybook, highly detailed, colorful - {p}",
                num_inference_steps=self.steps,
                guidance_scale=7.5,
                width=1024,
                height=1024
            ).images[0]

            buffer = BytesIO()
            image.save(buffer, format="WEBP")
            base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return base64_data

        try:
            image_result = await asyncio.to_thread(sync_inference, prompt)
            image_path = await asyncio.to_thread(
                save_base64_image,
                image_result,
                self.provider,
                self.model_name,
                test_case_id,
                self.run_session_id or 'session_unknown'
            )
        except Exception as e:
            error_msg = str(e)

        latency_ms = (time.monotonic() - start_time) * 1000

        return ExperimentResult(
            provider=self.provider,
            model_name=f"{self.model_name}-{self.steps}step",
            final_prompt=prompt,
            execution_type="local",
            gpu_info="NVIDIA A100-SXM4-40GB",
            latency_tti_ms=latency_ms,
            image_url_or_data=image_result,
            cost_per_run_usd=self.cost,
            error_message=error_msg
        )


# --- LCM-SDXL Implementation ---
class LocalLCMSDXL(TtiApi):
    """Wrapper for local LCM-SDXL inference on A100."""

    def __init__(self, model_name="latent-consistency/lcm-sdxl", steps=4):
        import torch
        from diffusers import StableDiffusionXLPipeline, LCMScheduler

        super().__init__(model_name=model_name, provider="local_gpu_a100")
        self.cost = 0.001
        self.device = "cuda"
        self.steps = steps

        if not torch.cuda.is_available():
            raise EnvironmentError("PyTorch CUDA not available.")

        print(f"Loading {self.model_name} LCM ({steps}-step) to GPU...")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(self.device)

        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("xFormers enabled for LCM-SDXL.")
        except Exception:
            print("xFormers failed for LCM-SDXL.")

    async def generate_image(self, prompt: str, test_case_id: str = "T_unknown") -> ExperimentResult:
        start_time = time.monotonic()
        image_result = ""
        error_msg = ""

        def sync_inference(p):
            from io import BytesIO

            image = self.pipe(
                prompt=f"Cartoon, Storybook, vibrant colors - {p}",
                num_inference_steps=self.steps,
                guidance_scale=1.0,  # LCM works best with low guidance
                width=1024,
                height=1024
            ).images[0]

            buffer = BytesIO()
            image.save(buffer, format="WEBP")
            base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return base64_data

        try:
            image_result = await asyncio.to_thread(sync_inference, prompt)
            image_path = await asyncio.to_thread(
                save_base64_image,
                image_result,
                self.provider,
                self.model_name,
                test_case_id,
                self.run_session_id or 'session_unknown'
            )
        except Exception as e:
            error_msg = str(e)

        latency_ms = (time.monotonic() - start_time) * 1000

        return ExperimentResult(
            provider=self.provider,
            model_name=f"{self.model_name}-{self.steps}step",
            final_prompt=prompt,
            execution_type="local",
            gpu_info="NVIDIA A100-SXM4-40GB",
            latency_tti_ms=latency_ms,
            image_url_or_data=image_result,
            cost_per_run_usd=self.cost,
            error_message=error_msg
        )