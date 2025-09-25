# experiment_runner.py
import asyncio
import csv
import json
import os
from datetime import datetime
from data_models import ExperimentResult, ExperimentReport
from model_interfaces import TtiApi
from api_wrappers import OpenAITtiApi, StabilityAITtiApi # Import your wrappers

class BenchmarkRunner:
    def __init__(self, output_file="benchmark_results.csv"):
        self.output_file = output_file
        self.report = ExperimentReport()

    def add_result(self, result: ExperimentResult):
        """Adds a single experiment result to the report."""
        self.report.results.append(result)

    def save_report_to_csv(self):
        """Saves all collected results to a standardized CSV file."""
        if not self.report.results:
            print("No results to save.")
            return

        # Get all field names from the dataclass for the header row
        fieldnames = [f.name for f in ExperimentResult.__dataclass_fields__.values()]
        
        # Prepare the file
        file_exists = os.path.exists(self.output_file)
        with open(self.output_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header only if the file is new
            if not file_exists:
                writer.writeheader()

            # Write data rows
            for result in self.report.results:
                writer.writerow(result.__dict__)
        
        print(f"\n✅ Successfully saved {len(self.report.results)} results to {self.output_file}")


async def run_tti_benchmark(runner: BenchmarkRunner, api_model: TtiApi, test_prompts: list, iterations: int):
    """Runs a full benchmark cycle for one TTI API."""
    print(f"\n{'='*50}\nSTARTING BENCHMARK FOR: {api_model.provider} - {api_model.model_name}")
    total_runs = 0
    
    for prompt in test_prompts:
        print(f"\n  Testing prompt: '{prompt[:40]}...'")
        for i in range(iterations):
            total_runs += 1
            # 💡 This is the plug-and-play moment: only the API object changes.
            result = await api_model.generate_image(prompt)
            result.test_case_id = f"P{total_runs}" # Simple unique run ID
            
            runner.add_result(result)
            
            status = "✅ SUCCESS" if not result.error_message else f"❌ FAILED: {result.error_message[:40]}..."
            print(f"    Run {i+1}: TTIS={result.latency_tti_ms:.2f}ms, Status: {status}")
            
            await asyncio.sleep(0.5) # Be kind to API rate limits


if __name__ == "__main__":
    # --- 1. Define Test Environment ---
    runner = BenchmarkRunner()
    
    test_prompts = [
        "A friendly space dog wearing a helmet, flying a colorful kite over a planet made of cookies.",
        "A wizard cat casting a sparkling spell on a grumpy gnome in a neon forest, high contrast.",
    ]
    
    # --- 2. Initialize Models (The Models You Want to Test) ---
    # Remember to set your OPENAI_API_KEY and STABILITY_KEY environment variables!
    
    models_to_test = [
        OpenAITtiApi(model_name="dall-e-3"),
        StabilityAITtiApi(model_name="stable-diffusion-3.5-large-turbo"),
        # You would add Gemini, Midjourney API, etc., here.
    ]
    
    # --- 3. Run Experiments ---
    async def main_experiment():
        for model in models_to_test:
            await run_tti_benchmark(runner, model, test_prompts, iterations=3)

        # --- 4. Save Results ---
        runner.save_report_to_csv()

    asyncio.run(main_experiment())