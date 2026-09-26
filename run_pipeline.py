#!/usr/bin/env python3
"""
Root pipeline runner for Amazon ML Challenge 2026.
Automatically detects whether the official dataset/train or dataset/sample/train is present,
runs training, validation, blocking, inference, and validates outputs.
"""

import os
import sys
import argparse

# Add src to sys.path
SRC_DIR = os.path.join(os.path.dirname(__file__), "code", "business_entity_resolution", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import Config
from pipeline import EntityResolutionPipeline


def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 Pipeline Runner")
    parser.add_argument("--train-dir", type=str, default="", help="Custom train directory")
    parser.add_argument("--test-dir", type=str, default="", help="Custom test directory")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory")
    parser.add_argument("--max-candidates", type=int, default=15, help="Candidate budget per entity")
    args = parser.parse_args()

    config = Config()

    # Determine train & test directories
    if args.train_dir:
        config.train_dir = os.path.abspath(args.train_dir)
    if args.test_dir:
        config.test_dir = os.path.abspath(args.test_dir)
    if args.output_dir:
        config.output_dir = os.path.abspath(args.output_dir)
    if args.max_candidates:
        config.max_candidates_per_entity = args.max_candidates

    print("=================================================================")
    print("      Amazon ML Challenge 2026: Business Entity Resolution       ")
    print("=================================================================")
    print(f"Train Directory : {config.train_dir}")
    print(f"Test Directory  : {config.test_dir}")
    print(f"Output Directory: {config.output_dir}")
    print(f"Candidate Cap   : {config.max_candidates_per_entity}")
    print("=================================================================")

    if not os.path.exists(config.train_dir) or not os.path.exists(config.test_dir):
        print(f"Error: Missing train or test directory:\nTrain: {config.train_dir}\nTest: {config.test_dir}")
        print("If you have downloaded the official dataset from Unstop, unzip it into 'dataset/train/' and 'dataset/test/'.")
        sys.exit(1)

    pipeline = EntityResolutionPipeline(config=config)
    pipeline.run_training()
    matching_path, cand_path = pipeline.run_inference()

    # Run validation
    import subprocess
    validator_path = os.path.join(config.project_root, "utils", "validate_submission.py")
    if os.path.exists(validator_path):
        print("\n=== STEP 7: Validating Outputs with Official Rules ===")
        res = subprocess.run([
            sys.executable,
            validator_path,
            "--matching", matching_path,
            "--candidate", cand_path,
            "--test-dir", config.test_dir,
        ])
        if res.returncode == 0:
            print("\n>>> ALL VALIDATION CHECKS PASSED SUCCESSFULLY! <<<")
            print(f"Leaderboard file ready for upload: {matching_path}")
            print("Next: Run 'python package_submission.py' to generate your final submission .zip archive.")
        else:
            print("\nValidation failed. Please review errors above.", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
