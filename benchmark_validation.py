import os
import sys
import time
import re
import unicodedata
from rapidfuzz import fuzz

sys.path.append(os.path.join(os.path.dirname(__file__), "code", "business_entity_resolution", "src"))
from evaluate import evaluate_predictions, compute_entity_f_beta

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
TRAIN_DIR = os.path.join(PROJECT_ROOT, "dataset", "train")

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

    if content_tokens:
        compact_name = "".join(content_tokens)
        keys.append((c_norm, "name_full", compact_name[:24]))
        if len(content_tokens) >= 2:
            keys.append((c_norm, "name_pair", content_tokens[0] + "_" + content_tokens[1]))
        elif len(content_tokens) == 1 and len(content_tokens[0]) >= 4:
            keys.append((c_norm, "name_single", content_tokens[0]))

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

def run_benchmark(num_entities=2000):
    print(f"Loading ground truth for first {num_entities} train entities...")
    gt = {}
    with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            s1 = p[0]
            m = p[1].split(",") if len(p) > 1 and p[1] else []
            gt[s1] = m
            if len(gt) >= num_entities:
                break

    # Load S1 entities
    s1_entities = {}
    with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in gt:
                s1_entities[p[0]] = (p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else "", p[3] if len(p) > 3 else "")
            if len(s1_entities) >= num_entities:
                break

    all_gt_matches = set()
    for matches in gt.values():
        all_gt_matches.update(matches)

    print(f"Loaded {len(s1_entities)} S1 entities. Total true matches: {len(all_gt_matches)}")

    # Index targets
    print("Indexing target records (Source 2 and Source 3)...")
    t0 = time.time()
    index = {}
    target_cache = {}

    # For fast benchmarking, we can index records from S2 and S3
    for fn in ["train_source2.tsv", "train_source3.tsv"]:
        path = os.path.join(TRAIN_DIR, fn)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            next(f)
            for i, line in enumerate(f):
                p = line.rstrip("\r\n").split("\t")
                if len(p) < 4:
                    continue
                eid, name, addr, country = p[0], p[1], p[2], p[3]

                keys = extract_keys(name, addr, country)
                for k in keys:
                    if k not in index:
                        index[k] = []
                    if len(index[k]) < 8:
                        index[k].append(eid)

                clean_n = " ".join([t for t in clean_tokens(name) if t not in STOPWORDS])
                clean_a = strip_accents(addr).lower()
                target_cache[eid] = (clean_n, clean_a)

    print(f"Indexed targets in {time.time()-t0:.2f}s. Total keys: {len(index)}")

    # Evaluate old baseline logic
    preds_baseline = {}
    for s1_id, (name, addr, country) in s1_entities.items():
        s1_clean_n = " ".join([t for t in clean_tokens(name) if t not in STOPWORDS])
        s1_clean_a = strip_accents(addr).lower()
        keys = extract_keys(name, addr, country)
        cand_pool = set()
        for k in keys:
            if k in index:
                cand_pool.update(index[k])

        scored = []
        for cid in cand_pool:
            if cid not in target_cache:
                continue
            t_n, t_a = target_cache[cid]
            n_sim = fuzz.token_sort_ratio(s1_clean_n, t_n)
            a_sim = fuzz.token_set_ratio(s1_clean_a, t_a) if (s1_clean_a and t_a) else 0.0
            score = 0.70 * n_sim + 0.30 * a_sim if a_sim > 0 else n_sim
            scored.append((cid, score, n_sim, a_sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_cands = [cid for cid, score, n_sim, a_sim in scored[:8]]

        final_matches = []
        for cid, score, n_sim, a_sim in scored[:8]:
            if score >= 62.0 or n_sim >= 80.0 or a_sim >= 90.0:
                final_matches.append(cid)

        cand_set = set(top_cands)
        preds_baseline[s1_id] = [m for m in final_matches if m in cand_set]

    res = evaluate_predictions(gt, preds_baseline)
    print("\n--- BASELINE SCORE ---")
    print(f"Macro Precision: {res['macro_precision']:.4f}")
    print(f"Macro Recall:    {res['macro_recall']:.4f}")
    print(f"Macro F_0.5:     {res['macro_f_beta']:.4f}")

if __name__ == "__main__":
    run_benchmark(1000)
