#!/usr/bin/env python3
"""
Official Scalable High-Performance Entity Resolution Pipeline for Amazon ML Challenge 2026.
Optimized for Leaderboard Macro F_0.5 Precision-First Maximization:
- Pre-trained 50,000-entity LightGBM pairwise reranker model
- Multi-faceted Inverted Indexing (DBA, acronyms, domain stems, street numbers, distinct street tokens, phones, PINs)
- Fast C++ SIMD string similarity features via RapidFuzz
- Shared Building & Street Number Veto Rules (eliminates commercial plaza false positives)
- Global 1-to-Many Unique Target Assignment (eliminates duplicate cross-entity false positives)
- Strict subset condition: matching_results.tsv <= candidate_pairs.tsv
- Low memory footprint (< 4GB RAM) with 100% test entity coverage
"""

import os
import sys
import time
import re
import gc
import csv
import subprocess
import numpy as np
import joblib
from rapidfuzz import fuzz
from anyascii import anyascii

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
TEST_DIR = os.path.join(PROJECT_ROOT, "dataset", "test")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MODEL_PATH = os.path.join(PROJECT_ROOT, "code", "business_entity_resolution", "models", "lgbm_matcher.joblib")

LEGAL_SUFFIXES = {
    "corp", "corporation", "inc", "incorporated", "ltd", "limited",
    "pvt", "private", "llc", "co", "company", "sa", "sarl", "sasu", "eurl",
    "gmbh", "group", "holdings", "services", "solutions", "enterprises",
    "technologies", "pllc", "llp",
}
STOPWORDS = {
    "the", "and", "of", "in", "at", "for", "on", "by", "to", "with", "&",
} | LEGAL_SUFFIXES

PHONETIC_SUBS = [
    (r"\belelpi\b", "llp"), (r"\bpraivet\b", "private"), (r"\bpvt\b", "private"),
    (r"\bltd\b", "limited"), (r"\bvemcrs\b", "ventures"), (r"\bhotl\b", "hotel"),
    (r"\bemtrpraijej\b", "enterprises"), (r"\binvestmemt\b", "investments"),
    (r"\binvestment\b", "investments"), (r"\bsolyusms\b", "solutions"),
    (r"\bsolyusan\b", "solutions"), (r"\bpayoniyr\b", "pioneer"), (r"\btek\b", "tech"),
    (r"\bdilli\b", "delhi"),
]

DBA_PATTERN = re.compile(r"\b(?:trading\s+as|t/a|t\s*/\s*a|dba|d\.b\.a\.|d\s*/\s*b\s*/\s*a|aka|a\.k\.a\.|a\s*/\s*k\s*/\s*a)\b", re.IGNORECASE)

COUNTRY_MAP = {"us": 0, "india": 1, "france": 2}


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


def extract_street_part(addr: str) -> str:
    parts = addr.split(",")
    for p in parts:
        p_clean = p.strip()
        if re.search(r"\d+", p_clean) or any(k in p_clean.lower() for k in ["rue", "ave", "st", "rd", "dr", "blvd", "ch", "imp", "way", "lane"]):
            return p_clean.lower()
    return parts[0].strip().lower() if parts else ""


def get_real_numbers(text: str):
    nums = re.findall(r"\b\d+\b", text)
    return [n.lstrip("0") for n in nums if n.lstrip("0")]


def is_domain_or_acronym_match(s1_norm: str, t_norm: str, s1_tokens: list, t_tokens: list) -> bool:
    s1_compact = "".join(s1_tokens)
    t_compact = "".join(t_tokens)
    if not s1_compact or not t_compact:
        return False
    if s1_compact in t_compact or t_compact in s1_compact:
        if min(len(s1_compact), len(t_compact)) >= 4:
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


def extract_blocking_keys(name: str, addr: str, country: str):
    c_norm = country.strip().lower()
    keys = []
    addr_norm_nums = re.sub(r"\b0+(\d+)\b", r"\1", addr)

    variants = [name]
    m = DBA_PATTERN.search(name)
    if m:
        before = name[:m.start()].strip()
        after = name[m.end():].strip()
        if before:
            variants.append(before)
        if after:
            variants.append(after)

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
            if len(c_tokens) >= 2 and len(c_tokens[1]) >= 4:
                keys.append(f"{c_norm}|name_f|{c_tokens[1]}")

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
            "ground", "shop", "unit", "building", "cross", "main",
        )
    ]
    if nums and addr_tokens:
        s_num = nums[0].lstrip("0") or "0"
        keys.append(f"{c_norm}|addr_num|{s_num}|{addr_tokens[0]}")
        if len(addr_tokens) >= 2:
            keys.append(f"{c_norm}|addr_num|{s_num}|{addr_tokens[1]}")

    if nums and content_tokens:
        s_num = nums[0].lstrip("0") or "0"
        for tok in content_tokens[:2]:
            keys.append(f"{c_norm}|num_name|{s_num}|{tok}")

    for a_tok in addr_tokens:
        if len(a_tok) >= 6:
            keys.append(f"{c_norm}|addr_dist|{a_tok}")

    return list(set(keys))


def precompute_s1(name: str, addr: str, country: str):
    norm_n = normalize_text(name)
    norm_a = normalize_text(addr)
    toks = [t for t in clean_tokens(norm_n) if t not in STOPWORDS]
    clean_n = " ".join(toks)
    tok_set = set(toks)
    street = extract_street_part(norm_a)
    nums = get_real_numbers(norm_a)
    num_set = set(nums)
    parts = norm_a.split(",")
    city = parts[-2].strip() if len(parts) >= 2 else ""
    c_str = country.strip().lower()
    c_code = COUNTRY_MAP.get(c_str, -1)

    return (clean_n, norm_n, toks, tok_set, norm_a, street, nums, num_set, city, c_code)


def precompute_target(name: str, addr: str, country: str):
    norm_n = normalize_text(name)
    norm_a = normalize_text(addr)
    toks = [t for t in clean_tokens(norm_n) if t not in STOPWORDS]
    clean_n = " ".join(toks)
    street = extract_street_part(norm_a)
    parts = norm_a.split(",")
    city = parts[-2].strip() if len(parts) >= 2 else ""
    c_str = country.strip().lower()
    c_code = COUNTRY_MAP.get(c_str, -1)

    return (clean_n, norm_n, norm_a, street, city, c_code)


def compute_pairwise_features(s1_tuple, t_tuple):
    (
        s1_clean_n, s1_norm_n, s1_toks, s1_tok_set,
        s1_norm_a, s1_street, s1_nums, s1_num_set, s1_city, s1_c_code
    ) = s1_tuple

    t_clean_n, t_norm_n, t_norm_a, t_street, t_city, t_c_code = t_tuple
    t_toks = t_clean_n.split()
    t_tok_set = set(t_toks)
    t_nums = get_real_numbers(t_norm_a)
    t_num_set = set(t_nums)

    n_ratio = fuzz.ratio(s1_clean_n, t_clean_n) / 100.0
    n_sort = fuzz.token_sort_ratio(s1_clean_n, t_clean_n) / 100.0
    n_set = fuzz.token_set_ratio(s1_clean_n, t_clean_n) / 100.0
    n_partial = fuzz.partial_ratio(s1_clean_n, t_clean_n) / 100.0

    first_tok_sim = fuzz.ratio(s1_toks[0], t_toks[0]) / 100.0 if (s1_toks and t_toks) else 0.0
    last_tok_sim = fuzz.ratio(s1_toks[-1], t_toks[-1]) / 100.0 if (s1_toks and t_toks) else 0.0

    len_s1 = len(s1_clean_n)
    len_t = len(t_clean_n)
    len_ratio = (min(len_s1, len_t) / max(len_s1, len_t)) if max(len_s1, len_t) > 0 else 1.0
    exact_name = 1.0 if (s1_clean_n and s1_clean_n == t_clean_n) else 0.0

    inter = float(len(s1_tok_set & t_tok_set))
    union = len(s1_tok_set | t_tok_set)
    jaccard_sim = (inter / union) if union > 0 else 0.0

    a_ratio = fuzz.ratio(s1_norm_a, t_norm_a) / 100.0 if (s1_norm_a and t_norm_a) else 0.0
    a_sort = fuzz.token_sort_ratio(s1_norm_a, t_norm_a) / 100.0 if (s1_norm_a and t_norm_a) else 0.0
    a_set = fuzz.token_set_ratio(s1_norm_a, t_norm_a) / 100.0 if (s1_norm_a and t_norm_a) else 0.0
    st_sort = fuzz.token_sort_ratio(s1_street, t_street) / 100.0 if (s1_street and t_street) else 0.0
    city_sim = fuzz.token_sort_ratio(s1_city, t_city) / 100.0 if (s1_city and t_city) else 0.0

    overlap = len(s1_num_set & t_num_set)
    conflict = 1.0 if (s1_nums and t_nums and overlap == 0) else 0.0

    primary_match = 0.0
    num_edit_dist = 0.0
    if s1_nums and t_nums:
        if s1_nums[0] == t_nums[0]:
            primary_match = 1.0
        elif abs(len(s1_nums[0]) - len(t_nums[0])) <= 1 and fuzz.ratio(s1_nums[0], t_nums[0]) >= 70:
            num_edit_dist = 1.0

    dom_acr = 1.0 if is_domain_or_acronym_match(s1_norm_n, t_norm_n, s1_toks, t_toks) else 0.0
    t_empty_addr = 1.0 if not t_norm_a.strip() else 0.0
    country_match = 1.0 if (s1_c_code == t_c_code and s1_c_code != -1) else 0.0

    return [
        n_ratio, n_sort, n_set, n_partial,
        first_tok_sim, last_tok_sim, len_ratio, exact_name,
        jaccard_sim, inter,
        a_ratio, a_sort, a_set, st_sort, city_sim,
        overlap, conflict, primary_match, num_edit_dist,
        dom_acr, t_empty_addr, country_match
    ]


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

                keys = extract_blocking_keys(name, addr, country)
                for k in keys:
                    if k not in index:
                        index[k] = []
                    if len(index[k]) < 25:
                        index[k].append(eid)

                target_cache[eid] = precompute_target(name, addr, country)

                if total_records % 2000000 == 0:
                    print(f"    Indexed {total_records/1e6:.1f}M records in {time.time()-t0:.1f}s...")

    gc.collect()
    print(f"Target index built: {total_records:,} records across {len(index):,} keys in {time.time()-t0:.2f}s.")
    return index, target_cache


def resolve_test_entities(index, target_cache):
    s1_path = os.path.join(TEST_DIR, "test_source1.tsv")
    matching_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    candidate_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[Phase 2/3] Loading LightGBM Pairwise Reranker Model...")
    clf = joblib.load(MODEL_PATH)
    print("LightGBM Model loaded successfully.")

    print("\nProcessing 1.73M Source 1 Test Entities with Batched LightGBM Inference...")
    t0 = time.time()

    # Data structures for Global 1-to-Many Assignment
    # best_s1_for_target: target_id -> (s1_id, prob)
    best_s1_for_target = {}

    total_s1 = 0
    total_evaluated_pairs = 0
    BATCH_SIZE = 50000

    # Batch accumulator
    batch_meta = []   # (s1_id, cid, rank_score, n_set, n_sort, conflict, dom_acr)
    batch_feats = []

    def flush_batch():
        nonlocal total_evaluated_pairs
        if not batch_feats:
            return
        X_mat = np.array(batch_feats, dtype=np.float32)
        probs = clf.predict_proba(X_mat)[:, 1]
        total_evaluated_pairs += len(probs)

        # Apply Veto Rules & Global 1-to-Many Assignment
        for meta, prob in zip(batch_meta, probs):
            s1_id, cid, rank_score, n_set, n_sort, conflict, dom_acr = meta
            # Veto Rule 1: Commercial plaza / unrelated name in same building
            if n_set < 40.0 and dom_acr == 0.0:
                prob = 0.0
            # Veto Rule 2: Distinct street number conflict
            elif conflict == 1.0 and n_sort < 0.90:
                prob = 0.0

            # Optimal calibrated threshold for Macro F_0.5
            if prob >= 0.9965:
                if cid not in best_s1_for_target or prob > best_s1_for_target[cid][1]:
                    best_s1_for_target[cid] = (s1_id, prob)

        batch_meta.clear()
        batch_feats.clear()

    temp_cands_path = os.path.join(OUTPUT_DIR, "temp_s1_candidates.tsv")
    print("Streaming candidate generation & scoring to disk...")
    with open(s1_path, "r", encoding="utf-8", errors="replace") as f_in, \
         open(temp_cands_path, "w", encoding="utf-8", newline="") as f_temp:

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

            s1_rec = precompute_s1(name, addr, country)
            keys = extract_blocking_keys(name, addr, country)
            cand_pool = set()
            for k in keys:
                if k in index:
                    cand_pool.update(index[k])

            # Pre-filter and rank candidates for candidate_pairs.tsv
            scored_for_cands = []
            for cid in cand_pool:
                if cid not in target_cache:
                    continue
                t_rec = target_cache[cid]

                # Fast preliminary check
                n_set = fuzz.token_set_ratio(s1_rec[0], t_rec[0])
                a_set = fuzz.token_set_ratio(s1_rec[4], t_rec[2]) if (s1_rec[4] and t_rec[2]) else 0.0
                dom_acr = is_domain_or_acronym_match(s1_rec[1], t_rec[1], s1_rec[2], t_rec[0].split())

                rank_score = n_set * 0.65 + a_set * 0.35
                if dom_acr:
                    rank_score = max(rank_score, 85.0)

                scored_for_cands.append((cid, rank_score))

                # Hopeless filter (rejected by Veto Rule anyway)
                if n_set < 40.0 and not dom_acr:
                    continue

                feats = compute_pairwise_features(s1_rec, t_rec)
                n_sort = feats[1]
                conflict = feats[16]

                batch_meta.append((s1_id, cid, rank_score, n_set, n_sort, conflict, 1.0 if dom_acr else 0.0))
                batch_feats.append(feats)

                if len(batch_feats) >= BATCH_SIZE:
                    flush_batch()

            scored_for_cands.sort(key=lambda x: x[1], reverse=True)
            top_cands_pool = [cid for cid, r in scored_for_cands[:10]]
            f_temp.write(f"{s1_id}\t{','.join(top_cands_pool)}\n")

            if total_s1 % 200000 == 0:
                elapsed = time.time() - t0
                print(f"  Processed {total_s1:,} / 1,732,544 ({total_s1/1732544*100:.1f}%) "
                      f"[{total_s1/elapsed:.0f} entities/sec, evaluated pairs: {total_evaluated_pairs:,}]")

    flush_batch()
    print(f"\nAll 1.73M entities processed in {time.time()-t0:.2f}s. Total pairs evaluated: {total_evaluated_pairs:,}.")

    # Free memory: index and target_cache are no longer needed
    print("\nFreeing index and target cache memory...")
    del index
    del target_cache
    del clf
    gc.collect()
    print(f"Index freed. Total unique targets assigned: {len(best_s1_for_target):,}.")

    # Materialize final matches from Global 1-to-Many Assignment
    print("\nInverting Global 1-to-Many Unique Assignments...")
    s1_matches = {}
    for cid, (s1_id, prob) in best_s1_for_target.items():
        if s1_id not in s1_matches:
            s1_matches[s1_id] = []
        s1_matches[s1_id].append((cid, prob))

    del best_s1_for_target
    gc.collect()

    total_matches = 0
    total_candidates = 0
    singletons = 0

    print("Writing final candidate_pairs.tsv and matching_results.tsv with guaranteed subset property...")
    with open(temp_cands_path, "r", encoding="utf-8") as f_temp_in, \
         open(candidate_path, "w", encoding="utf-8", newline="") as f_cand, \
         open(matching_path, "w", encoding="utf-8", newline="") as f_match:

        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        f_match.write("source1_entity_id\tmatched_entity_ids\n")

        for line in f_temp_in:
            parts = line.rstrip("\r\n").split("\t")
            s1_id = parts[0]
            pool_str = parts[1] if len(parts) > 1 else ""
            raw_pool = pool_str.split(",") if pool_str else []

            raw_matches = s1_matches.get(s1_id, [])
            raw_matches.sort(key=lambda x: x[1], reverse=True)
            true_matches = [cid for cid, p in raw_matches][:6]

            # Build final candidate list: must contain all matches + fill remaining up to 6
            cands = list(true_matches)
            for cid in raw_pool:
                if cid not in cands and len(cands) < 6:
                    cands.append(cid)

            f_cand.write(f"{s1_id}\t{','.join(cands)}\n")
            f_match.write(f"{s1_id}\t{','.join(true_matches)}\n")

            total_candidates += len(cands)
            total_matches += len(true_matches)
            if len(true_matches) == 0:
                singletons += 1

    # Cleanup temporary candidate file
    if os.path.exists(temp_cands_path):
        os.remove(temp_cands_path)

    elapsed = time.time() - t0
    print("\n" + "="*60)
    print("RESOLUTION COMPLETED SUCCESSFULLY!")
    print(f"Total S1 Entities:     {total_s1:,}")
    print(f"Total Matches:         {total_matches:,} (Avg {total_matches/total_s1:.2f}/entity)")
    print(f"Total Candidates:      {total_candidates:,} (Avg {total_candidates/total_s1:.2f}/entity)")
    print(f"Predicted Singletons:  {singletons:,} ({singletons/total_s1*100:.2f}%)")
    print(f"Total Pipeline Time:   {elapsed:.2f} seconds ({total_s1/elapsed:.0f} entities/sec)")
    print("="*60)

    return matching_path, candidate_path


def main():
    print("================================================================")
    print(" Amazon ML Challenge 2026: Official Full Resolution Pipeline    ")
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
        print(f"Submission file: {matching_path}")

        print("\nPackaging final submission ZIP...")
        packager = os.path.join(PROJECT_ROOT, "package_submission.py")
        subprocess.run([sys.executable, packager, "--team-name", "Alex_DevDrift"])
    else:
        print("\nValidation failed!", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
