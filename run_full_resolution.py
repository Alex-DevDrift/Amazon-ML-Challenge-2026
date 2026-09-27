#!/usr/bin/env python3
"""
High-Performance Scalable Entity Resolution for Amazon ML Challenge 2026.
Processes all 1,732,544 Source 1 test entities and 9.9M Target records (S2 + S3).
Streams outputs directly to disk with minimal memory overhead.
Ensures 100% test entity coverage, valid IDs, strict subset condition, and format compliance.
"""

import os
import sys
import time
import re
import unicodedata
import subprocess
from rapidfuzz import fuzz

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
TEST_DIR = os.path.join(PROJECT_ROOT, "dataset", "test")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

LEGAL_SUFFIXES = {
    "corp", "corporation", "inc", "incorporated", "ltd", "limited",
    "pvt", "private", "llc", "co", "company", "sa", "sarl", "gmbh", "group", "holdings",
}
STOPWORDS = {
    "the", "and", "of", "in", "at", "for", "on", "by", "to", "with", "&",
} | LEGAL_SUFFIXES


def strip_accents(text: str) -> str:
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])


def clean_tokens(text: str):
    cleaned = strip_accents(text).lower()
    return re.findall(r"[a-z0-9]+", cleaned)


def extract_keys(name: str, addr: str, country: str):
    c_norm = country.strip().lower()
    keys = []

    name_tokens = clean_tokens(name)
    content_tokens = [t for t in name_tokens if t not in STOPWORDS and len(t) >= 2]

    # Key 1: Full content name compact
    if content_tokens:
        compact_name = "".join(content_tokens)
        keys.append((c_norm, "name_full", compact_name[:24]))

        # Key 2: First 2 content tokens
        if len(content_tokens) >= 2:
            keys.append((c_norm, "name_pair", content_tokens[0] + "_" + content_tokens[1]))
        elif len(content_tokens) == 1 and len(content_tokens[0]) >= 4:
            keys.append((c_norm, "name_single", content_tokens[0]))

    # Key 3: Address numbers + street/locality token
    nums = re.findall(r"\b\d+\b", addr)
    addr_tokens = [
        t for t in clean_tokens(addr)
        if len(t) >= 4 and t not in (
            "street", "avenue", "road", "floor", "suite", "colony",
            "lane", "nagar", "bengaluru", "mumbai", "chennai", "delhi",
            "paris", "france", "india", "state", "near", "opposite",
        )
    ]

    if nums and addr_tokens:
        keys.append((c_norm, "addr_num", nums[0], addr_tokens[0]))

    return keys


def build_target_index():
    print("\n[Phase 1/3] Building Target Inverted Index from test_source2.tsv & test_source3.tsv...")
    t0 = time.time()
    index = {}
    target_cache = {}  # Store minimal info for candidate scoring

    total_records = 0
    for src_file in ["test_source2.tsv", "test_source3.tsv"]:
        path = os.path.join(TEST_DIR, src_file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required file: {path}")

        print(f"  Indexing {src_file}...")
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            header = next(f)
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 4:
                    continue
                eid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                total_records += 1

                keys = extract_keys(name, addr, country)
                for k in keys:
                    if k not in index:
                        index[k] = []
                    if len(index[k]) < 8:  # Cap candidate pool per key to prevent generic explosion
                        index[k].append(eid)

                # Store minimal info for fast candidate scoring
                clean_n = " ".join([t for t in clean_tokens(name) if t not in STOPWORDS])
                clean_a = strip_accents(addr).lower()
                target_cache[eid] = (clean_n, clean_a)

                if total_records % 2000000 == 0:
                    print(f"    Indexed {total_records/1e6:.1f}M target records in {time.time()-t0:.1f}s...")

    print(f"Index built for {total_records} records across {len(index)} keys in {time.time()-t0:.2f}s.")
    return index, target_cache


def resolve_test_entities(index, target_cache):
    s1_path = os.path.join(TEST_DIR, "test_source1.tsv")
    matching_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    candidate_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\n[Phase 2/3] Processing 1.73M Source 1 Test Entities & Writing Outputs...")
    t0 = time.time()

    total_s1 = 0
    total_candidates = 0
    total_matches = 0
    singletons = 0

    with open(s1_path, "r", encoding="utf-8", errors="replace") as f_in, \
         open(matching_path, "w", encoding="utf-8", newline="") as f_match, \
         open(candidate_path, "w", encoding="utf-8", newline="") as f_cand:

        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        header = next(f_in)
        for line in f_in:
            parts = line.rstrip("\r\n").split("\t")
            if not parts or not parts[0]:
                continue
            s1_id = parts[0]
            name = parts[1] if len(parts) > 1 else ""
            addr = parts[2] if len(parts) > 2 else ""
            country = parts[3] if len(parts) > 3 else ""
            total_s1 += 1

            s1_clean_n = " ".join([t for t in clean_tokens(name) if t not in STOPWORDS])
            s1_clean_a = strip_accents(addr).lower()

            keys = extract_keys(name, addr, country)
            cand_pool = set()
            for k in keys:
                if k in index:
                    cand_pool.update(index[k])

            # Score candidates
            scored = []
            for cid in cand_pool:
                if cid not in target_cache:
                    continue
                t_n, t_a = target_cache[cid]

                # Fast scoring
                n_sim = fuzz.token_sort_ratio(s1_clean_n, t_n)
                a_sim = fuzz.token_set_ratio(s1_clean_a, t_a) if (s1_clean_a and t_a) else 0.0

                score = 0.70 * n_sim + 0.30 * a_sim if a_sim > 0 else n_sim
                scored.append((cid, score, n_sim, a_sim))

            # Sort descending and enforce candidate cap (max 8 candidates)
            scored.sort(key=lambda x: x[1], reverse=True)
            top_cands = [cid for cid, score, n_sim, a_sim in scored[:8]]

            # Decision threshold for matching (Macro F_0.5 rewards precision heavily)
            # Match if score >= 62.0 or (name token sort >= 80.0) or (exact address match)
            final_matches = []
            for cid, score, n_sim, a_sim in scored[:8]:
                if score >= 62.0 or n_sim >= 80.0 or a_sim >= 90.0:
                    final_matches.append(cid)

            # Guarantee subset constraint: matches must be subset of candidates
            cand_set = set(top_cands)
            final_matches = [m for m in final_matches if m in cand_set]

            # Write rows directly to disk
            f_cand.write(f"{s1_id}\t{','.join(top_cands)}\n")
            f_match.write(f"{s1_id}\t{','.join(final_matches)}\n")

            total_candidates += len(top_cands)
            total_matches += len(final_matches)
            if len(final_matches) == 0:
                singletons += 1

            if total_s1 % 200000 == 0:
                elapsed = time.time() - t0
                rate = total_s1 / elapsed
                print(f"  Processed {total_s1:,} / 1,732,544 ({total_s1/1732544*100:.1f}%) "
                      f"[{rate:.0f} entities/sec, singletons: {singletons:,}]")

    elapsed = time.time() - t0
    print("\n" + "="*50)
    print("Inference Completed Successfully!")
    print(f"Total Source 1 Entities Processed: {total_s1:,}")
    print(f"Total Predicted Matches: {total_matches:,} (Avg {total_matches/total_s1:.2f}/entity)")
    print(f"Total Candidates Generated: {total_candidates:,} (Avg {total_candidates/total_s1:.2f}/entity)")
    print(f"Predicted Singletons: {singletons:,} ({singletons/total_s1*100:.2f}%)")
    print(f"Total Time: {elapsed:.2f} seconds ({total_s1/elapsed:.0f} entities/sec)")
    print("="*50)

    return matching_path, candidate_path


def main():
    print("================================================================")
    print(" Amazon ML Challenge 2026: Official Full Dataset Resolution    ")
    print("================================================================")
    index, target_cache = build_target_index()
    matching_path, candidate_path = resolve_test_entities(index, target_cache)

    print("\n[Phase 3/3] Validating with Official validate_submission.py...")
    validator = os.path.join(PROJECT_ROOT, "utils", "validate_submission.py")
    res = subprocess.run([
        sys.executable,
        validator,
        "--matching", matching_path,
        "--candidate", candidate_path,
        "--test-dir", TEST_DIR,
    ])

    if res.returncode == 0:
        print("\n>>> OFFICIAL VALIDATION PASSED WITH EXIT CODE 0! <<<")
        print(f"File ready for upload to Unstop: {matching_path}")

        # Package submission zip
        print("\nCreating final submission package...")
        packager = os.path.join(PROJECT_ROOT, "package_submission.py")
        subprocess.run([sys.executable, packager, "--team-name", "AmazonML2026_Team"])
    else:
        print("\nValidation failed!", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
