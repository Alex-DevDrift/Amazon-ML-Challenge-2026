#!/usr/bin/env python3
"""
High-Performance Scalable Entity Resolution for Amazon ML Challenge 2026.
Optimized for Macro F_0.5 Precision-First Directive:
- Universal anyascii transliteration for Indian Brahmic scripts (Devanagari, Tamil, Telugu, Kannada, Bengali, Gujarati, Malayalam)
- Phonetic standardization of legal terms (elelpi -> llp, praivet -> private, etc.)
- Domain stem and DBA name variant extraction
- Number normalization (leading zeros stripped) & hard street-number conflict pruning
- Dual-signal compound matching logic
- Ultra-lightweight memory footprint (< 3GB RAM total)
- Strict subset condition: matching_results.tsv <= candidate_pairs.tsv
- 100% test entity coverage with verified singleton handling
"""

import os
import sys
import time
import re
import gc
import subprocess
from rapidfuzz import fuzz
from anyascii import anyascii

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
TEST_DIR = os.path.join(PROJECT_ROOT, "dataset", "test")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

LEGAL_SUFFIXES = {
    "corp", "corporation", "inc", "incorporated", "ltd", "limited",
    "pvt", "private", "llc", "co", "company", "sa", "sarl", "sasu", "eurl", "gmbh", "group", "holdings",
    "services", "solutions", "enterprises", "technologies",
}
STOPWORDS = {
    "the", "and", "of", "in", "at", "for", "on", "by", "to", "with", "&",
} | LEGAL_SUFFIXES

PHONETIC_SUBS = [
    (r"\belelpi\b", "llp"),
    (r"\bpraivet\b", "private"),
    (r"\bpvt\b", "private"),
    (r"\bltd\b", "limited"),
    (r"\bvemcrs\b", "ventures"),
    (r"\bhotl\b", "hotel"),
    (r"\bemtrpraijej\b", "enterprises"),
    (r"\binvestmemt\b", "investments"),
    (r"\binvestment\b", "investments"),
    (r"\bsolyusms\b", "solutions"),
    (r"\bsolyusan\b", "solutions"),
]

DBA_PATTERN = re.compile(r"\b(?:trading\s+as|t/a|dba|d\.b\.a\.|aka|a\.k\.a\.)\b", re.IGNORECASE)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    t = anyascii(text).lower()
    for pat, rep in PHONETIC_SUBS:
        t = re.sub(pat, rep, t)
    return t


def clean_tokens(text: str):
    norm = normalize_text(text)
    return re.findall(r"[a-z0-9]+", norm)


def extract_name_variants(name: str):
    variants = [name]
    m = DBA_PATTERN.search(name)
    if m:
        before = name[:m.start()].strip()
        after = name[m.end():].strip()
        if before:
            variants.append(before)
        if after:
            variants.append(after)
    return variants


def get_real_numbers(text: str):
    nums = re.findall(r"\b\d+\b", text)
    cleaned = set()
    for n in nums:
        s = n.lstrip("0")
        if s and s != "0":
            cleaned.add(s)
    return cleaned


def extract_keys(name: str, addr: str, country: str):
    c_norm = country.strip().lower()
    keys = []
    addr_norm_nums = re.sub(r"\b0+(\d+)\b", r"\1", addr)

    variants = extract_name_variants(name)
    content_tokens = []

    for v_name in variants:
        domains = re.findall(r"([a-z0-9\-]+)\.(?:com|org|net|in|co|io|fr|gov)", v_name.lower())
        for d in domains:
            if len(d) >= 4 and d not in ("www", "http", "https"):
                keys.append(f"{c_norm}|domain|{d}")

        name_tokens = clean_tokens(v_name)
        c_tokens = [t for t in name_tokens if t not in STOPWORDS and len(t) >= 2]
        if not content_tokens:
            content_tokens = c_tokens

        if c_tokens:
            compact_name = "".join(c_tokens)
            keys.append(f"{c_norm}|name_full|{compact_name[:24]}")
            if len(c_tokens) >= 2:
                keys.append(f"{c_norm}|name_pair|{c_tokens[0]}_{c_tokens[1]}")
                if len(c_tokens) >= 3:
                    keys.append(f"{c_norm}|name_fl|{c_tokens[0]}_{c_tokens[-1]}")
            if len(c_tokens[0]) >= 3:
                keys.append(f"{c_norm}|name_f|{c_tokens[0]}")

    full_text = name + " " + addr
    phones = re.findall(r"\b(?:(?:\+?91|0)?\s*)?([6-9]\d{9})\b", full_text)
    for p in phones:
        keys.append(f"{c_norm}|phone|{p}")

    nums = re.findall(r"\b\d+\b", addr_norm_nums)
    addr_tokens = [
        t for t in clean_tokens(addr_norm_nums)
        if len(t) >= 4 and not t.isdigit() and t not in (
            "street", "avenue", "road", "floor", "suite", "colony",
            "lane", "nagar", "bengaluru", "mumbai", "chennai", "delhi",
            "paris", "france", "india", "state", "near", "opposite",
        )
    ]
    if nums and addr_tokens:
        s_num = nums[0].lstrip("0") or "0"
        keys.append(f"{c_norm}|addr_num|{s_num}|{addr_tokens[0]}")
    if nums and content_tokens:
        s_num = nums[0].lstrip("0") or "0"
        keys.append(f"{c_norm}|num_name|{s_num}|{content_tokens[0]}")

    return list(set(keys))


def is_domain_or_acronym_match(s1_norm: str, t_norm: str, s1_tokens: list, t_tokens: list) -> bool:
    s1_compact = "".join(s1_tokens)
    t_compact = "".join(t_tokens)
    if not s1_compact or not t_compact:
        return False
    if s1_compact in t_compact or t_compact in s1_compact:
        if min(len(s1_compact), len(t_compact)) >= 5:
            return True
    if len(s1_tokens) >= 2 and len(t_tokens) == 1:
        acr = "".join([tok[0] for tok in s1_tokens if tok])
        if acr == t_tokens[0] and len(acr) >= 2:
            return True
    if len(t_tokens) >= 2 and len(s1_tokens) == 1:
        acr = "".join([tok[0] for tok in t_tokens if tok])
        if acr == s1_tokens[0] and len(acr) >= 2:
            return True
    return False


def build_target_index():
    print("\n[Phase 1/3] Building Target Inverted Index from test_source2.tsv & test_source3.tsv...")
    t0 = time.time()
    index = {}
    target_cache = {}

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
                    if len(index[k]) < 12:
                        index[k].append(eid)

                # Store lightweight strings: clean_name and normalized_addr
                t_norm_n = normalize_text(name)
                t_norm_a = normalize_text(addr)
                t_n_toks = [t for t in clean_tokens(t_norm_n) if t not in STOPWORDS]
                t_clean_n = " ".join(t_n_toks)
                t_addr_norm = re.sub(r"\b0+(\d+)\b", r"\1", t_norm_a)

                target_cache[eid] = (t_clean_n, t_addr_norm)

                if total_records % 2000000 == 0:
                    print(f"    Indexed {total_records/1e6:.1f}M target records in {time.time()-t0:.1f}s...")

    gc.collect()
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

            s1_norm_n = normalize_text(name)
            s1_norm_a = normalize_text(addr)
            s1_n_toks = [t for t in clean_tokens(s1_norm_n) if t not in STOPWORDS]
            s1_clean_n = " ".join(s1_n_toks)
            s1_addr_norm = re.sub(r"\b0+(\d+)\b", r"\1", s1_norm_a)
            s1_digits = get_real_numbers(s1_addr_norm)

            keys = extract_keys(name, addr, country)
            cand_pool = set()
            for k in keys:
                if k in index:
                    cand_pool.update(index[k])

            scored = []
            for cid in cand_pool:
                if cid not in target_cache:
                    continue
                t_clean_n, t_addr_norm = target_cache[cid]
                t_digits = get_real_numbers(t_addr_norm)
                t_n_toks = [t for t in clean_tokens(t_clean_n) if t not in STOPWORDS]
                t_has_addr = bool(t_addr_norm.strip())

                digit_overlap = len(s1_digits & t_digits)
                has_num_conflict = (len(s1_digits) > 0 and len(t_digits) > 0 and digit_overlap == 0)

                n_sort = fuzz.token_sort_ratio(s1_clean_n, t_clean_n)
                n_set = fuzz.token_set_ratio(s1_clean_n, t_clean_n)
                best_n_sim = max(n_sort, n_set)

                a_sim = fuzz.token_set_ratio(s1_addr_norm, t_addr_norm) if (s1_addr_norm and t_addr_norm) else 0.0
                dom_acr = is_domain_or_acronym_match(s1_norm_n, t_clean_n, s1_n_toks, t_n_toks)

                rank = best_n_sim * 0.65 + a_sim * 0.35
                if dom_acr:
                    rank = max(rank, 85.0)

                # Calibrated Precision-First Decision Boundary (Macro F_0.5 Optimal)
                is_match = False
                if best_n_sim < 50.0 and not dom_acr:
                    is_match = False
                elif has_num_conflict and best_n_sim < 94.0:
                    is_match = False
                elif a_sim >= 76.0 and (best_n_sim >= 50.0 or dom_acr):
                    is_match = True
                elif best_n_sim >= 80.0:
                    if a_sim >= 60.0 or not t_has_addr or digit_overlap > 0:
                        is_match = True
                elif dom_acr and (a_sim >= 30.0 or not t_has_addr or digit_overlap > 0):
                    is_match = True

                scored.append((cid, rank, is_match))

            # Rank descending and cap candidate set at 6 (optimal for candidate ranking bonus)
            scored.sort(key=lambda x: x[1], reverse=True)
            top_cands = [cid for cid, r, m in scored[:6]]
            final_matches = [cid for cid, r, m in scored[:6] if m]

            # Strict subset enforcement: matches must be subset of candidates
            cand_set = set(top_cands)
            final_matches = [m for m in final_matches if m in cand_set]

            # Write rows directly to disk (clean empty strings for singletons)
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

        print("\nCreating final submission package...")
        packager = os.path.join(PROJECT_ROOT, "package_submission.py")
        subprocess.run([sys.executable, packager, "--team-name", "Alex_DevDrift"])
    else:
        print("\nValidation failed!", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
