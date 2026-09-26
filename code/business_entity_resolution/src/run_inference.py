#!/usr/bin/env python3
"""
CLI Entrypoint for Business Entity Resolution.
Usage:
    python run_inference.py --train-dir dataset/train --test-dir dataset/test --output-dir output
"""

import os
import sys
import argparse
import subprocess

# Ensure src directory is in path
src_dir = os.path.abspath(os.path.dirname(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from config import Config
from pipeline import EntityResolutionPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run end-to-end Entity Resolution Pipeline (Training, Blocking, Matching, Output generation)."
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="",
        help="Path to training directory containing train_source1.tsv, train_source2.tsv, etc.",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="",
        help="Path to test directory containing test_source1.tsv, test_source2.tsv, etc.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Path to output directory for matching_results.tsv and candidate_pairs.tsv",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=15,
        help="Maximum candidates per Source 1 entity in blocking stage",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()

    if args.train_dir:
        config.train_dir = os.path.abspath(args.train_dir)
    if args.test_dir:
        config.test_dir = os.path.abspath(args.test_dir)
    if args.output_dir:
        config.output_dir = os.path.abspath(args.output_dir)
    if args.max_candidates:
        config.max_candidates_per_entity = args.max_candidates

    print("================================================================")
    print(" Amazon ML Challenge 2026: Business Entity Resolution Pipeline ")
    print("================================================================")
    print(f"Train Dir : {config.train_dir}")
    print(f"Test Dir  : {config.test_dir}")
    print(f"Output Dir: {config.output_dir}")
    print(f"Max Cands : {config.max_candidates_per_entity}")
    print("================================================================")

    pipeline = EntityResolutionPipeline(config=config)

    # 1. Train and tune threshold
    pipeline.run_training()

    # 2. Run test inference and produce TSV outputs
    matching_path, cand_path = pipeline.run_inference()

    # 3. Automatic Validation Check
    print("\n=== STEP 7: Running Submission Validator ===")
    validator_path = os.path.join(
        config.project_root, "utils", "validate_submission.py"
    )
    if os.path.exists(validator_path):
        cmd = [
            sys.executable,
            validator_path,
            "--matching",
            matching_path,
            "--candidate",
            cand_path,
            "--test-dir",
            config.test_dir,
        ]
        res = subprocess.run(cmd)
        if res.returncode == 0:
            print("\nPipeline execution and submission validation SUCCESSFUL!")
        else:
            print("\nValidator detected issues. Please check output log above.", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Warning: Validator script not found at {validator_path}")


if __name__ == "__main__":
    main()
