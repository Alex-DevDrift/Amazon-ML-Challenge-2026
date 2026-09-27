import os
import sys
import time
import re
import csv
import pickle
import numpy as np
import lightgbm as lgb
from rapidfuzz import fuzz
from anyascii import anyascii

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
TRAIN_DIR = os.path.join(PROJECT_ROOT, "dataset", "train")

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

def normalize_text(text: str) -> str:
    if not text: return ""
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
    if not s1_compact or not t_compact: return False
    if s1_compact in t_compact or t_compact in s1_compact:
        if min(len(s1_compact), len(t_compact)) >= 4: return True
    if len(s1_tokens) >= 2 and len(t_tokens) == 1:
        acr = "".join([tok[0] for tok in s1_tokens if tok])
        if acr == t_tokens[0] and len(acr) >= 2: return True
    if len(t_tokens) >= 2 and len(s1_tokens) == 1:
        acr = "".join([tok[0] for tok in t_tokens if tok])
        if acr == s1_tokens[0] and len(acr) >= 2: return True
    return False

def extract_features(s1_data, t_data):
    s1_name, s1_addr, s1_country = s1_data
    t_name, t_addr, t_country = t_data

    s1_norm_n = normalize_text(s1_name)
    t_norm_n = normalize_text(t_name)
    s1_toks = [t for t in clean_tokens(s1_norm_n) if t not in STOPWORDS]
    t_toks = [t for t in clean_tokens(t_norm_n) if t not in STOPWORDS]
    s1_clean_n = " ".join(s1_toks)
    t_clean_n = " ".join(t_toks)

    s1_norm_a = normalize_text(s1_addr)
    t_norm_a = normalize_text(t_addr)
    s1_street = extract_street_part(s1_norm_a)
    t_street = extract_street_part(t_norm_a)

    s1_nums = get_real_numbers(s1_norm_a)
    t_nums = get_real_numbers(t_norm_a)
    s1_num_set = set(s1_nums)
    t_num_set = set(t_nums)

    overlap = len(s1_num_set & t_num_set)
    conflict = 1.0 if (s1_nums and t_nums and overlap == 0) else 0.0

    primary_match = 0.0
    num_edit_dist = 0.0
    if s1_nums and t_nums:
        if s1_nums[0] == t_nums[0]:
            primary_match = 1.0
        elif abs(len(s1_nums[0]) - len(t_nums[0])) <= 1 and fuzz.ratio(s1_nums[0], t_nums[0]) >= 70:
            num_edit_dist = 1.0

    n_ratio = fuzz.ratio(s1_clean_n, t_clean_n) / 100.0
    n_sort = fuzz.token_sort_ratio(s1_clean_n, t_clean_n) / 100.0
    n_set = fuzz.token_set_ratio(s1_clean_n, t_clean_n) / 100.0
    n_partial = fuzz.partial_ratio(s1_clean_n, t_clean_n) / 100.0

    first_tok_sim = fuzz.ratio(s1_toks[0], t_toks[0]) / 100.0 if (s1_toks and t_toks) else 0.0
    last_tok_sim = fuzz.ratio(s1_toks[-1], t_toks[-1]) / 100.0 if (s1_toks and t_toks) else 0.0

    len_s1 = len(s1_clean_n)
    len_t = len(t_clean_n)
    len_ratio = (min(len_s1, len_t) / max(len_s1, len_t)) if max(len_s1, len_t) > 0 else 1.0

    a_ratio = fuzz.ratio(s1_norm_a, t_norm_a) / 100.0 if (s1_norm_a and t_norm_a) else 0.0
    a_sort = fuzz.token_sort_ratio(s1_norm_a, t_norm_a) / 100.0 if (s1_norm_a and t_norm_a) else 0.0
    a_set = fuzz.token_set_ratio(s1_norm_a, t_norm_a) / 100.0 if (s1_norm_a and t_norm_a) else 0.0
    st_sort = fuzz.token_sort_ratio(s1_street, t_street) / 100.0 if (s1_street and t_street) else 0.0

    # Token Jaccard & Shared Tokens
    s1_tok_set = set(s1_toks)
    t_tok_set = set(t_toks)
    intersection = float(len(s1_tok_set & t_tok_set))
    union = len(s1_tok_set | t_tok_set)
    jaccard_sim = (intersection / union) if union > 0 else 0.0

    # City parsing
    s1_parts = s1_norm_a.split(",")
    t_parts = t_norm_a.split(",")
    s1_city = s1_parts[-2].strip() if len(s1_parts) >= 2 else ""
    t_city = t_parts[-2].strip() if len(t_parts) >= 2 else ""
    city_sim = fuzz.token_sort_ratio(s1_city, t_city) / 100.0 if (s1_city and t_city) else 0.0

    dom_acr = 1.0 if is_domain_or_acronym_match(s1_norm_n, t_norm_n, s1_toks, t_toks) else 0.0
    t_empty_addr = 1.0 if not t_norm_a.strip() else 0.0
    country_match = 1.0 if s1_country.strip().lower() == t_country.strip().lower() else 0.0
    exact_name = 1.0 if (s1_clean_n and s1_clean_n == t_clean_n) else 0.0

    return [
        n_ratio, n_sort, n_set, n_partial,
        first_tok_sim, last_tok_sim, len_ratio, exact_name,
        jaccard_sim, intersection,
        a_ratio, a_sort, a_set, st_sort, city_sim,
        overlap, conflict, primary_match, num_edit_dist,
        dom_acr, t_empty_addr, country_match
    ]

FEATURE_COLS = [
    "n_ratio", "n_sort", "n_set", "n_partial",
    "first_tok_sim", "last_tok_sim", "len_ratio", "exact_name",
    "jaccard_sim", "intersection",
    "a_ratio", "a_sort", "a_set", "st_sort", "city_sim",
    "overlap", "conflict", "primary_match", "num_edit_dist",
    "dom_acr", "t_empty_addr", "country_match"
]

def extract_blocking_keys(name: str, addr: str, country: str):
    c_norm = country.strip().lower()
    keys = []
    addr_norm_nums = re.sub(r"\b0+(\d+)\b", r"\1", addr)

    variants = [name]
    m = DBA_PATTERN.search(name)
    if m:
        before = name[:m.start()].strip()
        after = name[m.end():].strip()
        if before: variants.append(before)
        if after: variants.append(after)

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

def build_and_evaluate(num_train_s1=35000):
    print("="*60)
    print(f"Building Training Dataset from {num_train_s1} S1 Entities...")
    print("="*60)
    t0 = time.time()

    # Load validation entities to exclude from training
    with open("val_1000_cache.pkl", "rb") as f:
        val_data = pickle.load(f)
    val_s1_set = set(val_data["s1_entities"].keys())

    # Sample S1 training entities (skipping first 10,000 to ensure completely independent slice)
    train_s1 = {}
    with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), "r", encoding="utf-8", errors="replace") as f:
        r = csv.reader(f, delimiter="\t")
        header = next(r)
        for i, row in enumerate(r):
            if i < 10000: continue
            if row[0] in val_s1_set: continue
            train_s1[row[0]] = (row[1], row[2], row[3])
            if len(train_s1) >= num_train_s1:
                break

    print(f"Sampled {len(train_s1)} training S1 entities.")

    # Load ground truth for sampled S1
    gt_train = {}
    target_needed = set()
    with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), "r", encoding="utf-8", errors="replace") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            if row[0] in train_s1:
                targets = row[1].split(",") if len(row) > 1 and row[1] else []
                gt_train[row[0]] = targets
                target_needed.update(targets)

    print(f"Loaded {sum(len(v) for v in gt_train.values())} true positive matches across {len(target_needed)} targets.")

    # Scan train_source2.tsv and train_source3.tsv to load targets and build candidate index
    print("Scanning train_source2 and train_source3 to build candidate index and target cache...")
    index = {}
    target_records = {}

    # Pre-extract keys for train S1 to only index relevant target candidates
    s1_key_set = set()
    for s1_id, (s1_n, s1_a, s1_c) in train_s1.items():
        s1_key_set.update(extract_blocking_keys(s1_n, s1_a, s1_c))
    print(f"Total active blocking keys for training S1: {len(s1_key_set):,}")

    t_scan = time.time()
    indexed_extra = 0
    MAX_EXTRA_TARGETS = 400000

    for src in ["train_source2.tsv", "train_source3.tsv"]:
        path = os.path.join(TRAIN_DIR, src)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < 4: continue
                eid, name, addr, country = parts[0], parts[1], parts[2], parts[3]

                is_needed = eid in target_needed
                can_index = is_needed or (indexed_extra < MAX_EXTRA_TARGETS)

                if can_index:
                    keys = extract_blocking_keys(name, addr, country)
                    has_key_overlap = any(k in s1_key_set for k in keys)

                    if is_needed or has_key_overlap:
                        target_records[eid] = (name, addr, country)
                        if not is_needed:
                            indexed_extra += 1
                        for k in keys:
                            if k in s1_key_set:
                                if k not in index: index[k] = []
                                if len(index[k]) < 15:
                                    index[k].append(eid)

    print(f"Loaded {len(target_records):,} target records in {time.time()-t_scan:.2f}s.")

    # Extract pairwise feature dataset
    print("Extracting pairwise feature vectors for positive and hard-negative pairs...")
    X_list = []
    y_list = []

    pos_count = 0
    neg_count = 0

    for s1_id, s1_data in train_s1.items():
        true_set = set(gt_train.get(s1_id, []))
        s1_keys = extract_blocking_keys(s1_data[0], s1_data[1], s1_data[2])

        cands = set()
        for k in s1_keys:
            if k in index:
                cands.update(index[k])

        # Ensure all true positives are included in training
        for tid in true_set:
            if tid in target_records:
                cands.add(tid)

        # For negatives, sample up to 8 hard negatives per S1 entity
        neg_candidates = [cid for cid in cands if cid not in true_set and cid in target_records]
        if len(neg_candidates) > 8:
            neg_candidates = neg_candidates[:8]

        # Extract positive features
        for tid in true_set:
            if tid in target_records:
                feats = extract_features(s1_data, target_records[tid])
                X_list.append(feats)
                y_list.append(1)
                pos_count += 1

        # Extract negative features
        for cid in neg_candidates:
            feats = extract_features(s1_data, target_records[cid])
            X_list.append(feats)
            y_list.append(0)
            neg_count += 1

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"Dataset ready: {X.shape[0]:,} pairs ({pos_count:,} positives, {neg_count:,} negatives) in {time.time()-t0:.2f}s.")

    # Train LightGBM model
    print("\nTraining LightGBM Classifier...")
    t_train = time.time()
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.06,
        num_leaves=63,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1.2,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
    )
    clf.fit(X, y)
    print(f"LightGBM trained in {time.time()-t_train:.2f}s.")

    # Feature Importance
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nTop 10 Feature Importances:")
    for idx in sorted_idx[:10]:
        print(f"  {FEATURE_COLS[idx]:<18}: {importances[idx]}")

    # Save model
    models_dir = os.path.join(PROJECT_ROOT, "code", "business_entity_resolution", "models")
    os.makedirs(models_dir, exist_ok=True)
    import joblib
    model_path = os.path.join(models_dir, "lgbm_matcher.joblib")
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}.")

    # Evaluate on the independent validation set (val_1000_cache.pkl)
    print("\n" + "="*60)
    print("EVALUATING MODEL ON INDEPENDENT 1,000 VALIDATION SET")
    print("="*60)
    val_s1 = val_data["s1_entities"]
    val_gt = val_data["gt"]
    val_targets = val_data["target_records"]

    val_index = {}
    for eid, (t_n, t_a, t_c) in val_targets.items():
        keys = extract_blocking_keys(t_n, t_a, t_c)
        for k in keys:
            if k not in val_index: val_index[k] = []
            if len(val_index[k]) < 20: val_index[k].append(eid)

    # Score all candidate pairs for validation entities
    sys.path.append(os.path.join(PROJECT_ROOT, "code", "business_entity_resolution", "src"))
    from evaluate import evaluate_predictions

    val_pairs = [] # (s1_id, cid)
    val_feats = []
    for s1_id, s1_data in val_s1.items():
        keys = extract_blocking_keys(s1_data[0], s1_data[1], s1_data[2])
        cands = set()
        for k in keys:
            if k in val_index: cands.update(val_index[k])
        for cid in cands:
            if cid not in val_targets: continue
            f = extract_features(s1_data, val_targets[cid])
            val_pairs.append((s1_id, cid))
            val_feats.append(f)

    X_val = np.array(val_feats, dtype=np.float32)
    val_probs = clf.predict_proba(X_val)[:, 1]
    print(f"Predicted probabilities on {len(val_probs):,} validation pairs.")

    # Sweep thresholds WITH Global 1-to-Many unique target assignment
    best_f = 0
    best_th = 0
    best_p = 0
    best_r = 0

    print("\nThreshold Sweep with Global 1-to-Many Assignment:")
    thresholds = [0.50, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.985, 0.99, 0.995, 0.997, 0.998, 0.999]
    for th in thresholds:
        best_s1_for_target = {}
        for (s1, cid), prob in zip(val_pairs, val_probs):
            if prob >= th:
                if cid not in best_s1_for_target or prob > best_s1_for_target[cid][1]:
                    best_s1_for_target[cid] = (s1, prob)

        preds = {s1: [] for s1 in val_s1}
        for cid, (s1, prob) in best_s1_for_target.items():
            preds[s1].append((cid, prob))

        for s1 in preds:
            preds[s1].sort(key=lambda x: x[1], reverse=True)
            preds[s1] = [cid for cid, p in preds[s1][:6]]

        res = evaluate_predictions(val_gt, preds)
        p = res["macro_precision"]
        r = res["macro_recall"]
        f = res["macro_f_beta"]
        marker = " <<< NEW BEST" if f > best_f else ""
        if f > best_f:
            best_f = f
            best_th = th
            best_p = p
            best_r = r
        print(f"  Threshold {th:.5f} -> Precision: {p*100:.2f}%, Recall: {r*100:.2f}%, Macro F_0.5: {f:.4f}{marker}")

    print("\n" + "="*60)
    print(f"VALIDATION PERFORMANCE SUMMARY:")
    print(f"Optimal Threshold: p* = {best_th:.3f}")
    print(f"Macro Precision:   {best_p*100:.2f}%")
    print(f"Macro Recall:      {best_r*100:.2f}%")
    print(f"Macro F_0.5 Score: {best_f:.4f}")
    print("="*60)

if __name__ == "__main__":
    build_and_evaluate(num_train_s1=50000)
