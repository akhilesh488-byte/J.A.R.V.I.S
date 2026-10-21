"""
Dry run execution on 10 sample rows with an offline stub provider.
Requires no GPU, cloud key, network connection, or Ollama server.
Usage:
  python bench/dryrun.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bench.run import run_benchmark


def main():
    dataset_path = os.path.join(ROOT, "bench", "data", "sample_dataset.jsonl")
    out_dir = os.path.join(ROOT, "results")

    print("\nStarting J.A.R.V.I.S. Dry Run on sample dataset (10 rows)...")
    csv_path = run_benchmark(
        experiment="dryrun",
        machine="A",
        provider_name="stub",
        out_dir=out_dir,
        dataset_path=dataset_path,
        repeats=1,
        warmup=0,
    )
    print(f"\nDry run complete. Generated output: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
