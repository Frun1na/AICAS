#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run benchmark.py with torch.compile disabled for comparison."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/workspace/Qwen3-VL-2B-Instruct",
        help="Path to model weights",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/workspace/data",
        help="Path to validation dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # Force the evaluation wrapper to skip the compile branch.
    if hasattr(torch, "compile"):
        torch.compile = None

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import benchmark

    benchmark.run_benchmark(
        model_class=benchmark.VLMModel,
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        output_path=args.output,
        num_samples=args.num_samples,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
