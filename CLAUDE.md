# Talepad Tools - Project Context

## Overview
This project is a comprehensive benchmarking system for testing various text-to-image (TTI) generation models, comparing both speed and quality metrics. The goal is to find optimal models for fast generation vs quality generation use cases.

## Project Structure
```
talepad-tools/
├── voice2image/
│   ├── experiment/
│   │   ├── experiment_runner.py    # Main orchestrator for running benchmarks
│   │   ├── data_models.py          # Standardized data structures for results
│   │   ├── model_interfaces.py     # Abstract base class for API consistency
│   │   └── api_wrappers.py         # Concrete implementations for different providers
│   └── init/
│       ├── tti_service.py          # Standalone testing utility
│       └── samples.txt             # Test prompts
├── benchmark_results.csv           # Benchmark data (2.4MB of results)
├── readme.md                       # Basic project description
└── .gitignore                      # Git ignore rules
```

## Core Components

### Experiment Framework (`voice2image/experiment/`)
- **experiment_runner.py:11** - Main `BenchmarkRunner` class that orchestrates experiments
- **data_models.py:6** - `ExperimentResult` dataclass with standardized metrics
- **model_interfaces.py:5** - `TtiApi` abstract base class for consistent API
- **api_wrappers.py** - Concrete implementations for different TTI providers

### Model Implementations
1. **OpenAI DALL-E 3** (`api_wrappers.py:11`)
   - Cloud API, $0.04/image
   - 1024x1024 resolution
   - Cartoon/storybook style prompting

2. **Stability AI SD 3.5 Turbo** (`api_wrappers.py:75`)
   - Cloud API via REST v2beta
   - $0.04/image (conceptual)
   - WebP format output, 1:1 aspect ratio

3. **Local SDXL Turbo** (`api_wrappers.py:142`)
   - Local GPU (RTX 4070 Ti)
   - ~$0.001/image (electricity/depreciation)
   - Optimized with FP16, xFormers, 2 inference steps

### Testing Service (`voice2image/init/tti_service.py:15`)
- Standalone utility for testing OpenAI DALL-E 3
- Async image generation with timing metrics
- Test prompts for validation

## Key Metrics Tracked
- **Latency**: Time-to-image generation (TTIS) in milliseconds
- **Cost**: Per-image generation cost in USD
- **Quality**: Mean Opinion Score placeholder (0-5 scale)
- **Error Handling**: Comprehensive error logging
- **Metadata**: Provider, model name, test case ID, timestamp

## Current Test Prompts
1. "A friendly space dog wearing a helmet, flying a colorful kite over a planet made of cookies"
2. "A wizard cat casting a sparkling spell on a grumpy gnome in a neon forest, high contrast"

## Current State
- ✅ **Functional**: Local SDXL Turbo implementation running
- ✅ **Data Collection**: CSV results with extensive benchmark data
- ⚠️ **Issues**: Some "name 'image' is not defined" errors in recent runs
- 🔧 **Ready for**: Adding new model providers and expanding test scenarios

## Architecture Benefits
- **Plug-and-play**: Easy to add new TTI APIs through abstract `TtiApi` interface
- **Standardized metrics**: Consistent data format across all providers
- **Async support**: Non-blocking experiments with rate limiting
- **Local + Cloud**: Mix of local GPU and cloud API testing
- **Cost tracking**: Built-in cost analysis for different providers

## Usage
Run experiments with:
```python
# In experiment_runner.py
models_to_test = [
    # OpenAITtiApi(model_name="dall-e-3"),
    # StabilityAITtiApi(model_name="stable-diffusion-3.5-large-turbo"),
    LocalSDXLTurbo()
]
```

Results are automatically saved to `benchmark_results.csv` with comprehensive metrics for analysis.