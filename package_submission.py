#!/usr/bin/env python3
"""
Submission Packaging Script for Amazon ML Challenge 2026.
Creates <team_name>_submission.zip matching the exact specification:

<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   └── ...
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""

import os
import sys
import argparse
import zipfile
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Package Amazon ML Challenge 2026 Submission")
    parser.add_argument(
        "--team-name",
        type=str,
        default="amazon_ml_team",
        help="Your team name for the submission zip file",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validate_submission.py check before packaging",
    )
    return parser.parse_args()


def validate_files(test_dir):
    matching_path = os.path.join(PROJECT_ROOT, "output", "matching_results.tsv")
    candidate_path = os.path.join(PROJECT_ROOT, "output", "candidate_pairs.tsv")
    validator_path = os.path.join(PROJECT_ROOT, "utils", "validate_submission.py")

    if not os.path.exists(matching_path):
        print(f"Error: Missing {matching_path}. Run the pipeline first!", file=sys.stderr)
        return False
    if not os.path.exists(candidate_path):
        print(f"Error: Missing {candidate_path}. Run the pipeline first!", file=sys.stderr)
        return False

    if os.path.exists(validator_path) and os.path.exists(test_dir):
        print("Running pre-packaging validation...")
        cmd = [
            sys.executable,
            validator_path,
            "--matching", matching_path,
            "--candidate", candidate_path,
            "--test-dir", test_dir,
        ]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print("Validation failed! Fix issues before packaging.", file=sys.stderr)
            return False

    return True


def create_submission_zip(team_name: str) -> str:
    zip_filename = f"{team_name}_submission.zip"
    zip_path = os.path.join(PROJECT_ROOT, zip_filename)

    required_items = [
        ("output/matching_results.tsv", "output/matching_results.tsv"),
        ("output/candidate_pairs.tsv", "output/candidate_pairs.tsv"),
        ("Documentation_template.md", "Documentation_template.md"),
        ("code/business_entity_resolution/README.md", "code/business_entity_resolution/README.md"),
        ("code/business_entity_resolution/requirements.txt", "code/business_entity_resolution/requirements.txt"),
    ]

    # Add all src files
    src_dir = os.path.join(PROJECT_ROOT, "code", "business_entity_resolution", "src")
    for root, _, files in os.walk(src_dir):
        for f in files:
            if f.endswith(".py") and not f.startswith("."):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, PROJECT_ROOT).replace("\\", "/")
                required_items.append((rel_path, rel_path))

    print(f"\nCreating submission archive: {zip_filename}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for local_rel, zip_rel in required_items:
            full_local = os.path.join(PROJECT_ROOT, local_rel)
            if not os.path.exists(full_local):
                print(f"Warning: {full_local} not found, skipping.", file=sys.stderr)
                continue
            zipf.write(full_local, arcname=zip_rel)
            print(f"  + Added: {zip_rel}")

    print(f"\nSuccessfully created {zip_path}")
    print(f"File size: {os.path.getsize(zip_path) / 1024:.2f} KB")

    print("\nZip Archive Structure:")
    with zipfile.ZipFile(zip_path, "r") as zipf:
        for info in zipf.infolist():
            print(f"  - {info.filename} ({info.file_size} bytes)")

    return zip_path


def main():
    args = parse_args()
    test_dir = os.path.join(PROJECT_ROOT, "dataset", "test")
    if not os.path.exists(test_dir):
        test_dir = os.path.join(PROJECT_ROOT, "dataset", "sample", "test")

    if not args.skip_validation:
        if not validate_files(test_dir):
            sys.exit(1)

    zip_path = create_submission_zip(args.team_name)
    print("\nReady for final submission!")


if __name__ == "__main__":
    main()
