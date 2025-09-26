# experiment_runner.py
import asyncio
import csv
import json
import os
from datetime import datetime
from data_models import ExperimentResult, ExperimentReport
from model_interfaces import TtiApi
from api_wrappers import OpenAITtiApi, StabilityAITtiApi, LocalSDXLTurbo, LocalSDXLLightning, LocalSDXLBase, LocalLCMSDXL

class BenchmarkRunner:
    def __init__(self, output_file=None):
        if output_file is None:
            # Generate unique filename with date and run number
            self.output_file = self._generate_unique_filename()
        else:
            self.output_file = output_file

        # Generate unique session ID for this benchmark run
        self.run_session_id = self._generate_session_id()

        self.report = ExperimentReport()
        print(f"📊 Benchmark results will be saved to: {self.output_file}")
        print(f"🖼️  Images will be saved to: experiment_images/run_{self._get_date_str()}_{self.run_session_id}/")

    def _get_date_str(self):
        """Get current date string."""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d")

    def _generate_session_id(self):
        """Generate a unique session ID for this benchmark run."""
        import uuid
        return str(uuid.uuid4())[:8]  # Use first 8 chars of UUID for brevity

    def _generate_unique_filename(self):
        """Generate a unique filename with date and run number in results directory."""
        from datetime import datetime
        import glob

        # Create results directory
        results_dir = "benchmark_results"
        os.makedirs(results_dir, exist_ok=True)

        # Get current date
        date_str = datetime.now().strftime("%Y%m%d")

        # Find existing files with today's date in results directory
        pattern = os.path.join(results_dir, f"benchmark_results_{date_str}_run_*.csv")
        existing_files = glob.glob(pattern)

        # Determine next run number
        if not existing_files:
            run_number = 1
        else:
            # Extract run numbers from existing files
            run_numbers = []
            for file in existing_files:
                try:
                    # Extract number from filename like "benchmark_results_20250926_run_003.csv"
                    basename = os.path.basename(file)
                    parts = basename.split('_')
                    if len(parts) >= 4 and parts[-1].endswith('.csv'):
                        run_num = int(parts[-1].replace('.csv', ''))
                        run_numbers.append(run_num)
                except (ValueError, IndexError):
                    continue

            run_number = max(run_numbers) + 1 if run_numbers else 1

        # Generate full path with zero-padded run number
        filename = f"benchmark_results_{date_str}_run_{run_number:03d}.csv"
        full_path = os.path.join(results_dir, filename)
        return full_path

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

        # Always create a new file (since filename is unique)
        with open(self.output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Always write header for new file
            writer.writeheader()

            # Write data rows
            for result in self.report.results:
                writer.writerow(result.__dict__)

        print(f"\n✅ Successfully saved {len(self.report.results)} results to {self.output_file}")
        print(f"📁 File size: {os.path.getsize(self.output_file)} bytes")

    @staticmethod
    def list_benchmark_files():
        """List all benchmark result files in the results directory."""
        import glob
        from datetime import datetime

        results_dir = "benchmark_results"

        # Check if results directory exists
        if not os.path.exists(results_dir):
            print("No benchmark results directory found.")
            return

        pattern = os.path.join(results_dir, "benchmark_results_*.csv")
        files = glob.glob(pattern)
        files.sort()

        if not files:
            print("No benchmark result files found in results directory.")
            return

        print(f"\n📊 Available Benchmark Results (in {results_dir}/):")
        print("-" * 70)
        for file in files:
            try:
                size = os.path.getsize(file)
                size_mb = size / (1024 * 1024)
                mtime = os.path.getmtime(file)
                mod_time = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                filename = os.path.basename(file)
                print(f"📄 {filename:<45} {size_mb:>6.1f}MB  {mod_time}")
            except OSError:
                filename = os.path.basename(file)
                print(f"📄 {filename:<45} [Error reading file]")
        print("-" * 70)


async def run_tti_benchmark(runner: BenchmarkRunner, api_model: TtiApi, test_prompts: list, iterations: int):
    """Runs a full benchmark cycle for one TTI API."""
    print(f"\n{'='*50}\nSTARTING BENCHMARK FOR: {api_model.provider} - {api_model.model_name}")
    total_runs = 0

    for prompt in test_prompts:
        print(f"\n  Testing prompt: '{prompt[:40]}...'")
        for i in range(iterations):
            total_runs += 1
            test_case_id = f"P{total_runs}"

            # Pass session ID to model for organized image storage
            if hasattr(api_model, 'set_run_session_id'):
                api_model.set_run_session_id(runner.run_session_id)

            result = await api_model.generate_image(prompt, test_case_id)

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
        # OpenAITtiApi(model_name="dall-e-3"),
        # StabilityAITtiApi(model_name="stable-diffusion-3.5-large-turbo"),

        # Speed-focused models (fastest to slowest)
        LocalSDXLTurbo(),                    # 2-step, fastest
        # LocalSDXLLightning(steps=4),         # 4-step Lightning (model not found)
        # LocalLCMSDXL(steps=4),               # 4-step LCM (model not found)
        # LocalSDXLLightning(steps=8),         # 8-step Lightning (model not found)

        # Quality-focused models
        LocalSDXLBase(steps=25),             # 25-step Base (balanced)
        LocalSDXLBase(steps=50),             # 50-step Base (high quality)
    ]
    
    # --- 3. Run Experiments ---
    async def main_experiment():
        for model in models_to_test:
            await run_tti_benchmark(runner, model, test_prompts, iterations=2)

        # --- 4. Save Results ---
        runner.save_report_to_csv()

        # --- 5. Show All Available Results ---
        BenchmarkRunner.list_benchmark_files()

    asyncio.run(main_experiment())