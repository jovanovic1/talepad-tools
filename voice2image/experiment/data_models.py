# data_models.py
from dataclasses import dataclass, field
from typing import List
import time

@dataclass
class ExperimentResult:
    # Model/Test Details
    provider: str
    model_name: str
    test_case_id: str = "T1" # Unique ID for the prompt/test
    timestamp: float = field(default_factory=time.time)

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