#!/usr/bin/env python3
"""
Submission Validator for Amazon ML Challenge 2026: Business Entity Resolution.
Validates matching_results.tsv and candidate_pairs.tsv against all submission rules.
Standard library only, zero external dependencies.
"""

import sys
import os
import argparse
import csv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate submission files for Amazon ML Challenge 2026."
    )
    parser.add_argument(
        "--matching",
        required=True,
        help="Path to output/matching_results.tsv",
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="Path to output/candidate_pairs.tsv",
    )
    parser.add_argument(
        "--test-dir",
        required=True,
        help="Path to dataset/test directory containing test_source1.tsv, test_source2.tsv, test_source3.tsv",
    )
    return parser.parse_args()


def load_test_ids(test_dir):
    s1_file = os.path.join(test_dir, "test_source1.tsv")
    s2_file = os.path.join(test_dir, "test_source2.tsv")
    s3_file = os.path.join(test_dir, "test_source3.tsv")

    if not os.path.exists(s1_file):
        raise FileNotFoundError(f"Missing test source 1 file: {s1_file}")
    if not os.path.exists(s2_file):
        raise FileNotFoundError(f"Missing test source 2 file: {s2_file}")
    if not os.path.exists(s3_file):
        raise FileNotFoundError(f"Missing test source 3 file: {s3_file}")

    def read_ids(filepath, expected_prefix):
        ids = set()
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            if not header or "entity_id" not in header:
                raise ValueError(f"File {filepath} missing 'entity_id' header.")
            idx = header.index("entity_id")
            for line_no, row in enumerate(reader, start=2):
                if not row or len(row) <= idx:
                    continue
                eid = row[idx].strip()
                if eid:
                    ids.add(eid)
        return ids

    s1_ids = read_ids(s1_file, "S1-")
    s2_ids = read_ids(s2_file, "S2-")
    s3_ids = read_ids(s3_file, "S3-")
    return s1_ids, s2_ids, s3_ids


def validate_tsv_structure(filepath, expected_header_cols):
    issues = []
    if not os.path.exists(filepath):
        return [f"File does not exist: {filepath}"], {}

    row_dict = {}
    seen_s1 = set()

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
        if not first_line:
            return [f"File is empty: {filepath}"], {}

        header = [col.strip() for col in first_line.rstrip("\r\n").split("\t")]
        if header != expected_header_cols:
            issues.append(
                f"Invalid header in {filepath}. Expected: {expected_header_cols}, Got: {header}"
            )

        for line_no, line in enumerate(f, start=2):
            line_str = line.rstrip("\r\n")
            if not line_str.strip():
                continue
            parts = line_str.split("\t")
            if len(parts) > 2:
                issues.append(
                    f"{filepath}: line {line_no} has {len(parts)} columns (must be tab-separated with 2 columns)."
                )
                continue
            s1_id = parts[0].strip()
            ids_str = parts[1].strip() if len(parts) == 2 else ""

            if not s1_id:
                issues.append(f"{filepath}: line {line_no} has empty source1_entity_id.")
                continue

            if s1_id in seen_s1:
                issues.append(
                    f"{filepath}: duplicate source1_entity_id '{s1_id}' found at line {line_no}."
                )
            seen_s1.add(s1_id)

            if ids_str:
                matched_list = [x.strip() for x in ids_str.split(",") if x.strip()]
            else:
                matched_list = []

            # Check for duplicates in list
            if len(matched_list) != len(set(matched_list)):
                issues.append(
                    f"{filepath}: line {line_no} for entity {s1_id} contains duplicate IDs in list: {matched_list}"
                )

            row_dict[s1_id] = matched_list

    return issues, row_dict


def main():
    args = parse_args()
    all_issues = []

    print(f"Validating submission files against test directory: {args.test_dir}")
    try:
        s1_test_ids, s2_test_ids, s3_test_ids = load_test_ids(args.test_dir)
    except Exception as e:
        print(f"Error loading test files: {e}", file=sys.stderr)
        sys.exit(1)

    valid_s2_s3_ids = s2_test_ids | s3_test_ids

    # 1. Validate matching_results.tsv
    matching_issues, matching_dict = validate_tsv_structure(
        args.matching, ["source1_entity_id", "matched_entity_ids"]
    )
    all_issues.extend(matching_issues)

    # 2. Validate candidate_pairs.tsv
    candidate_issues, candidate_dict = validate_tsv_structure(
        args.candidate, ["source1_entity_id", "candidate_entity_ids"]
    )
    all_issues.extend(candidate_issues)

    if not matching_dict or not candidate_dict:
        print("\nValidation FAILED with issues:")
        for idx, issue in enumerate(all_issues, start=1):
            print(f"{idx}. {issue}")
        sys.exit(1)

    # 3. Check completeness: Every Source 1 entity in test set must have exactly one row
    missing_in_matching = s1_test_ids - set(matching_dict.keys())
    if missing_in_matching:
        all_issues.append(
            f"{args.matching} is missing {len(missing_in_matching)} Source 1 test entities (e.g. {list(missing_in_matching)[:5]})."
        )

    extra_in_matching = set(matching_dict.keys()) - s1_test_ids
    if extra_in_matching:
        all_issues.append(
            f"{args.matching} contains {len(extra_in_matching)} unknown Source 1 entities not in test set (e.g. {list(extra_in_matching)[:5]})."
        )

    missing_in_candidate = s1_test_ids - set(candidate_dict.keys())
    if missing_in_candidate:
        all_issues.append(
            f"{args.candidate} is missing {len(missing_in_candidate)} Source 1 test entities (e.g. {list(missing_in_candidate)[:5]})."
        )

    extra_in_candidate = set(candidate_dict.keys()) - s1_test_ids
    if extra_in_candidate:
        all_issues.append(
            f"{args.candidate} contains {len(extra_in_candidate)} unknown Source 1 entities not in test set (e.g. {list(extra_in_candidate)[:5]})."
        )

    # 4. Check ID references: matched and candidate IDs must only be valid S2/S3 test IDs
    for s1_id, match_list in matching_dict.items():
        for m_id in match_list:
            if m_id.startswith("S1-"):
                all_issues.append(
                    f"{args.matching}: entity {s1_id} contains self-reference/Source 1 ID: {m_id}"
                )
            elif m_id not in valid_s2_s3_ids:
                all_issues.append(
                    f"{args.matching}: entity {s1_id} references nonexistent test ID: {m_id}"
                )

    for s1_id, cand_list in candidate_dict.items():
        for c_id in cand_list:
            if c_id.startswith("S1-"):
                all_issues.append(
                    f"{args.candidate}: entity {s1_id} contains self-reference/Source 1 ID: {c_id}"
                )
            elif c_id not in valid_s2_s3_ids:
                all_issues.append(
                    f"{args.candidate}: entity {s1_id} references nonexistent test ID: {c_id}"
                )

    # 5. Check subset condition: matched_entity_ids must be a subset of candidate_entity_ids
    subset_violations = []
    for s1_id, match_list in matching_dict.items():
        if s1_id in candidate_dict:
            cand_set = set(candidate_dict[s1_id])
            for m_id in match_list:
                if m_id not in cand_set:
                    subset_violations.append((s1_id, m_id))

    if subset_violations:
        all_issues.append(
            f"Subset condition violated: {len(subset_violations)} matched entity IDs do not appear in candidate_pairs.tsv! Example: {subset_violations[:5]}"
        )

    # Final verdict
    if all_issues:
        print("\n[VALIDATION FAILED] The following issues were found:")
        for idx, issue in enumerate(all_issues, start=1):
            print(f"{idx}. {issue}")
        sys.exit(1)
    else:
        num_singletons = sum(1 for m in matching_dict.values() if len(m) == 0)
        num_matched = sum(len(m) for m in matching_dict.values())
        num_candidates = sum(len(c) for c in candidate_dict.values())
        avg_candidates = num_candidates / max(len(candidate_dict), 1)
        print("\n==========================================")
        print("PASS: Both files strictly satisfy all submission rules!")
        print(f"Total Source 1 entities: {len(s1_test_ids)}")
        print(f"Predicted Singletons: {num_singletons} ({num_singletons/len(s1_test_ids):.1%})")
        print(f"Total Matches Predicted: {num_matched}")
        print(f"Total Candidates Generated: {num_candidates}")
        print(f"Average Candidates per Entity: {avg_candidates:.2f}")
        print("==========================================")
        sys.exit(0)


if __name__ == "__main__":
    main()
