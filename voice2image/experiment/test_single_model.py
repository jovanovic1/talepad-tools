#!/usr/bin/env python3
# test_single_model.py - Test a single SDXL model
import asyncio
from experiment_runner import BenchmarkRunner
from api_wrappers import LocalSDXLTurbo

async def test_single_model():
    print("🚀 Testing single SDXL Turbo model...")

    # Create runner
    runner = BenchmarkRunner()

    # Create single model
    model = LocalSDXLTurbo()

    # Set session ID
    model.set_run_session_id(runner.run_session_id)

    # Test single prompt, single iteration
    test_prompt = "A friendly space dog wearing a helmet"

    print(f"\n🔬 Testing prompt: '{test_prompt}'")

    try:
        result = await model.generate_image(test_prompt, "T1")
        runner.add_result(result)

        print(f"✅ Success! Latency: {result.latency_tti_ms:.2f}ms")
        print(f"📁 Image: {result.image_url_or_data[:50]}...")

        # Save results
        runner.save_report_to_csv()
        runner.list_benchmark_files()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_single_model())