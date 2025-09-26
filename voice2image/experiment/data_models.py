# data_models.py
from dataclasses import dataclass, field
from typing import List
import time
import platform
from datetime import datetime

@dataclass
class ExperimentResult:
    # Model/Test Details
    provider: str
    model_name: str
    test_case_id: str = "T1" # Unique ID for the prompt/test

    # Experiment Context
    date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    execution_type: str = "local"  # "local" or "api"
    machine_type: str = field(default_factory=lambda: f"{platform.system()}_{platform.machine()}")
    gpu_info: str = ""  # Will be populated with specific GPU details

    # Latency Metrics (Time is in milliseconds)
    latency_tti_ms: float = 0.0      # Time to Image Generation Complete (TTIS for TTI component)
    latency_total_e2e_ms: float = 0.0 # Time to Image Serve (Full End-to-End latency)

    # Quality/Output Metrics
    image_url_or_data: str = ""      # The image result
    final_prompt: str = ""           # The final prompt sent to the TTI API (after LLM refinement)
    cost_per_run_usd: float = 0.0    # Calculated cost per image/run

    # Human/LLM Evaluation Placeholder
    quality_score_mos: float = 0.0   # Mean Opinion Score (1-5) or other quality metric

    # Store errors
    error_message: str = ""

@dataclass
class ExperimentReport:
    """A container to hold all results before saving."""
    results: List[ExperimentResult] = field(default_factory=list)