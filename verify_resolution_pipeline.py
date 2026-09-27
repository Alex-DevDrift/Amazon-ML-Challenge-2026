import os
import sys
import time
import pickle
import numpy as np
import joblib
from rapidfuzz import fuzz

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, "code", "business_entity_resolution", "src"))
from evaluate import evaluate_predictions
from build_train_lgbm import (
    normalize_text, clean_tokens, extract_street_part, get_real_numbers,
    is_domain_or_acronym_match, extract_blocking_keys, STOPWORDS
)

def precompute_record(eid: str, name: str, addr: str, country: str):
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
    empty_addr = not norm_a.strip()

    return {
        "eid": eid,
        "clean_n": clean_n,
        "norm_n": norm_n,
        "toks": toks,
        "tok_set": tok_set,
        "norm_a": norm_a,
        "street": street,
        "nums": nums,
        "num_set": num_set,
        "city": city,
        "country": country.strip().lower(),
        "empty_addr": empty_addr,
    }

def compute_features(s1: dict, t: dict):
    n_ratio = fuzz.ratio(s1["clean_n"], t["clean_n"]) / 100.0
    n_sort = fuzz.token_sort_ratio(s1["clean_n"], t["clean_n"]) / 100.0
    n_set = fuzz.token_set_ratio(s1["clean_n"], t["clean_n"]) / 100.0
    n_partial = fuzz.partial_ratio(s1["clean_n"], t["clean_n"]) / 100.0

    first_tok_sim = fuzz.ratio(s1["toks"][0], t["toks"][0]) / 100.0 if (s1["toks"] and t["toks"]) else 0.0
    last_tok_sim = fuzz.ratio(s1["toks"][-1], t["toks"][-1]) / 100.0 if (s1["toks"] and t["toks"]) else 0.0

    len_s1 = len(s1["clean_n"])
    len_t = len(t["clean_n"])
    len_ratio = (min(len_s1, len_t) / max(len_s1, len_t)) if max(len_s1, len_t) > 0 else 1.0
    exact_name = 1.0 if (s1["clean_n"] and s1["clean_n"] == t["clean_n"]) else 0.0

    inter = float(len(s1["tok_set"] & t["tok_set"]))
    union = len(s1["tok_set"] | t["tok_set"])
    jaccard_sim = (inter / union) if union > 0 else 0.0

    a_ratio = fuzz.ratio(s1["norm_a"], t["norm_a"]) / 100.0 if (s1["norm_a"] and t["norm_a"]) else 0.0
    a_sort = fuzz.token_sort_ratio(s1["norm_a"], t["norm_a"]) / 100.0 if (s1["norm_a"] and t["norm_a"]) else 0.0
    a_set = fuzz.token_set_ratio(s1["norm_a"], t["norm_a"]) / 100.0 if (s1["norm_a"] and t["norm_a"]) else 0.0
    st_sort = fuzz.token_sort_ratio(s1["street"], t["street"]) / 100.0 if (s1["street"] and t["street"]) else 0.0
    city_sim = fuzz.token_sort_ratio(s1["city"], t["city"]) / 100.0 if (s1["city"] and t["city"]) else 0.0

    overlap = len(s1["num_set"] & t["num_set"])
    conflict = 1.0 if (s1["nums"] and t["nums"] and overlap == 0) else 0.0

    primary_match = 0.0
    num_edit_dist = 0.0
    if s1["nums"] and t["nums"]:
        if s1["nums"][0] == t["nums"][0]:
            primary_match = 1.0
        elif abs(len(s1["nums"][0]) - len(t["nums"][0])) <= 1 and fuzz.ratio(s1["nums"][0], t["nums"][0]) >= 70:
            num_edit_dist = 1.0

    dom_acr = 1.0 if is_domain_or_acronym_match(s1["norm_n"], t["norm_n"], s1["toks"], t["toks"]) else 0.0
    t_empty_addr = 1.0 if t["empty_addr"] else 0.0
    country_match = 1.0 if s1["country"] == t["country"] else 0.0

    return [
        n_ratio, n_sort, n_set, n_partial,
        first_tok_sim, last_tok_sim, len_ratio, exact_name,
        jaccard_sim, inter,
        a_ratio, a_sort, a_set, st_sort, city_sim,
        overlap, conflict, primary_match, num_edit_dist,
        dom_acr, t_empty_addr, country_match
    ]

def main():
    print("="*60)
    print("Verifying Optimized Resolution Pipeline on Validation Cache")
    print("="*60)

    model_path = os.path.join(PROJECT_ROOT, "code", "business_entity_resolution", "models", "lgbm_matcher.joblib")
    clf = joblib.load(model_path)

    with open("val_1000_cache.pkl", "rb") as f:
        val_data = pickle.load(f)

    val_s1_raw = val_data["s1_entities"]
    val_gt = val_data["gt"]
    val_targets_raw = val_data["target_records"]

    print(f"Pre-computing {len(val_targets_raw):,} target records...")
    t0 = time.time()
    val_targets = {}
    index = {}
    for eid, (t_n, t_a, t_c) in val_targets_raw.items():
        rec = precompute_record(eid, t_n, t_a, t_c)
        val_targets[eid] = rec
        keys = extract_blocking_keys(t_n, t_a, t_c)
        for k in keys:
            if k not in index: index[k] = []
            if len(index[k]) < 25: index[k].append(eid)

    print(f"Targets indexed and precomputed in {time.time()-t0:.2f}s.")

    print(f"Scoring 1,000 S1 entities with pre-filter + LightGBM + Veto Rules...")
    t_score = time.time()
    s1_candidates = {}
    all_scored_pairs = [] # (prob, s1_id, cid)

    for s1_id, (s1_n, s1_a, s1_c) in val_s1_raw.items():
        s1_rec = precompute_record(s1_id, s1_n, s1_a, s1_c)
        keys = extract_blocking_keys(s1_n, s1_a, s1_c)
        cands = set()
        for k in keys:
            if k in index: cands.update(index[k])

        scored = []
        for cid in cands:
            if cid not in val_targets: continue
            t_rec = val_targets[cid]

            # Fast preliminary check
            n_set = fuzz.token_set_ratio(s1_rec["clean_n"], t_rec["clean_n"])
            a_set = fuzz.token_set_ratio(s1_rec["norm_a"], t_rec["norm_a"]) if (s1_rec["norm_a"] and t_rec["norm_a"]) else 0.0
            dom_acr = is_domain_or_acronym_match(s1_rec["norm_n"], t_rec["norm_n"], s1_rec["toks"], t_rec["toks"])

            if n_set < 30.0 and a_set < 30.0 and not dom_acr:
                continue

            feats = compute_features(s1_rec, t_rec)
            prob = float(clf.predict_proba(np.array([feats], dtype=np.float32))[0, 1])

            # Veto Rules
            n_sort = feats[1]
            conflict = feats[16]
            if n_set < 40.0 and not dom_acr:
                prob = 0.0
            elif conflict == 1.0 and n_sort < 0.90:
                prob = 0.0

            scored.append((cid, prob))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_cands = [cid for cid, p in scored[:6]]
        s1_candidates[s1_id] = top_cands

        for cid, p in scored:
            if p >= 0.990:
                all_scored_pairs.append((p, s1_id, cid))

    print(f"Scoring completed in {time.time()-t_score:.2f}s.")

    # Sweep thresholds WITH Global 1-to-Many unique assignment
    print("\nEvaluating Global 1-to-Many Assignment Across Thresholds:")
    for th in [0.990, 0.993, 0.995, 0.996, 0.997, 0.998, 0.9985]:
        best_s1_for_target = {}
        for prob, s1_id, cid in all_scored_pairs:
            if prob >= th:
                if cid not in best_s1_for_target or prob > best_s1_for_target[cid][1]:
                    best_s1_for_target[cid] = (s1_id, prob)

        preds = {s1_id: [] for s1_id in val_s1_raw}
        for cid, (s1_id, prob) in best_s1_for_target.items():
            preds[s1_id].append((cid, prob))

        for s1_id in preds:
            preds[s1_id].sort(key=lambda x: x[1], reverse=True)
            # Ensure matches are subset of candidates and cap at 6
            c_set = set(s1_candidates[s1_id])
            preds[s1_id] = [cid for cid, p in preds[s1_id][:6] if cid in c_set]

        res = evaluate_predictions(val_gt, preds)
        p = res["macro_precision"]
        r = res["macro_recall"]
        f = res["macro_f_beta"]
        print(f"  Threshold {th:.4f} -> Precision: {p*100:.2f}%, Recall: {r*100:.2f}%, Macro F_0.5: {f:.4f}")

if __name__ == "__main__":
    main()
