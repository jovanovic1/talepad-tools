#!/usr/bin/env python3
# memory_efficient_benchmark.py - Run comprehensive benchmark with memory management
import asyncio
import gc
import torch
from experiment_runner import BenchmarkRunner, run_tti_benchmark
from api_wrappers import LocalSDXLTurbo, LocalSDXLBase, LocalStableDiffusion15, LocalFlux

async def run_memory_efficient_benchmark():
    """Run comprehensive benchmark loading one model at a time to avoid CUDA OOM."""
    print("🚀 Starting Memory-Efficient Comprehensive Benchmark")
    print("="*70)

    # Create single runner for all results
    runner = BenchmarkRunner()

    test_prompts = [
        "A friendly space dog wearing a helmet, flying a colorful kite over a planet made of cookies.",
        "A wizard cat casting a sparkling spell on a grumpy gnome in a neon forest, high contrast.",
    ]

    # Model configurations to test (one at a time)
    model_configs = [
        {"class": LocalSDXLTurbo, "name": "SDXL Turbo", "kwargs": {}},
        {"class": LocalStableDiffusion15, "name": "SD 1.5 (10-step)", "kwargs": {"steps": 10}},
        {"class": LocalStableDiffusion15, "name": "SD 1.5 (25-step)", "kwargs": {"steps": 25}},
        {"class": LocalSDXLBase, "name": "SDXL Base (25-step)", "kwargs": {"steps": 25}},
        {"class": LocalSDXLBase, "name": "SDXL Base (50-step)", "kwargs": {"steps": 50}},
        {"class": LocalFlux, "name": "FLUX.1 Schnell (1-step)", "kwargs": {"model_name": "black-forest-labs/FLUX.1-schnell", "steps": 1}},
    ]

    for i, config in enumerate(model_configs, 1):
        print(f"\n🔄 [{i}/{len(model_configs)}] Testing {config['name']}")
        print("-" * 50)

        try:
            # Create model instance
            model = config["class"](**config["kwargs"])

            # Run benchmark for this model
            await run_tti_benchmark(runner, model, test_prompts, iterations=2)

            # Clean up memory
            del model
            gc.collect()
            torch.cuda.empty_cache()

            print(f"✅ Completed {config['name']} - Memory cleared")

        except Exception as e:
            print(f"❌ Failed {config['name']}: {e}")
            # Still clean up on failure
            gc.collect()
            torch.cuda.empty_cache()

    # Save all results
    print(f"\n{'='*70}")
    print("💾 SAVING COMPREHENSIVE RESULTS")
    print('='*70)

    runner.save_report_to_csv()
    runner.list_benchmark_files()

    print(f"\n🎉 Comprehensive benchmark complete!")
    print(f"📊 Total models tested: {len(model_configs)}")
    print(f"📋 Results saved to: {runner.output_file}")

if __name__ == "__main__":
    asyncio.run(run_memory_efficient_benchmark())