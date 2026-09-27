import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "code", "business_entity_resolution", "src"))
import pickle, re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from anyascii import anyascii
import lightgbm as lgb
from evaluate import evaluate_predictions

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
    if not text: return ""
    t = anyascii(text).lower()
    for pat, rep in PHONETIC_SUBS:
        t = re.sub(pat, rep, t)
    return t

def clean_tokens(text: str):
    norm = normalize_text(text)
    return re.findall(r"[a-z0-9]+", norm)

def extract_street_part(addr):
    parts = addr.split(",")
    for p in parts:
        p_clean = p.strip()
        if re.search(r"\d+", p_clean) or any(k in p_clean.lower() for k in ["rue", "ave", "st", "rd", "dr", "blvd", "ch", "imp", "way", "lane"]):
            return p_clean.lower()
    return parts[0].strip().lower() if parts else ""

def get_real_numbers(text):
    return set([n.lstrip("0") for n in re.findall(r"\b[1-9]\d*\b", text) if n.lstrip("0")])

def is_domain_or_acronym_match(s1_norm: str, t_norm: str, s1_tokens: list, t_tokens: list) -> bool:
    s1_compact = "".join(s1_tokens)
    t_compact = "".join(t_tokens)
    if not s1_compact or not t_compact: return False
    if s1_compact in t_compact or t_compact in s1_compact:
        if min(len(s1_compact), len(t_compact)) >= 5: return True
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

    s1_digits = get_real_numbers(s1_norm_a)
    t_digits = get_real_numbers(t_norm_a)
    overlap = len(s1_digits & t_digits)
    conflict = 1.0 if (s1_digits and t_digits and overlap == 0) else 0.0

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

    dom_acr = 1.0 if is_domain_or_acronym_match(s1_norm_n, t_norm_n, s1_toks, t_toks) else 0.0
    t_empty_addr = 1.0 if not t_norm_a.strip() else 0.0

    return [
        n_ratio, n_sort, n_set, n_partial,
        first_tok_sim, last_tok_sim, len_ratio,
        a_ratio, a_sort, a_set, st_sort,
        overlap, conflict, dom_acr, t_empty_addr
    ]

FEATURE_COLS = [
    "n_ratio", "n_sort", "n_set", "n_partial",
    "first_tok_sim", "last_tok_sim", "len_ratio",
    "a_ratio", "a_sort", "a_set", "st_sort",
    "overlap", "conflict", "dom_acr", "t_empty_addr"
]

def main():
    print("Loading validation cache...")
    with open("val_1000_cache.pkl", "rb") as f:
        data = pickle.load(f)

    s1_entities = data["s1_entities"]
    gt = data["gt"]
    target_records = data["target_records"]

    from optimize_matcher import extract_keys

    index = {}
    for eid, (t_n, t_a, t_c) in target_records.items():
        keys = extract_keys(t_n, t_a, t_c)
        for k in keys:
            if k not in index: index[k] = []
            if len(index[k]) < 20: index[k].append(eid)

    # Build dataset of candidate pairs
    print("Building candidate pair dataset...")
    rows = []
    labels = []
    pair_meta = [] # (s1_id, cid)

    s1_ids = sorted(s1_entities.keys())
    for s1_id in s1_ids:
        s1_data = s1_entities[s1_id]
        keys = extract_keys(s1_data[0], s1_data[1], s1_data[2])
        cands = set()
        for k in keys:
            if k in index: cands.update(index[k])

        true_set = set(gt.get(s1_id, []))
        for cid in cands:
            if cid not in target_records: continue
            feats = extract_features(s1_data, target_records[cid])
            label = 1 if cid in true_set else 0
            rows.append(feats)
            labels.append(label)
            pair_meta.append((s1_id, cid))

    X = np.array(rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    print(f"Dataset built: {X.shape[0]} pairs ({y.sum()} positives, {len(y)-y.sum()} negatives).")

    # 5-Fold GroupKFold by s1_id
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    unique_s1 = np.array(s1_ids)
    oof_predictions = {}
    oof_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(unique_s1)):
        val_s1_set = set(unique_s1[val_idx])
        train_mask = np.array([m[0] not in val_s1_set for m in pair_meta])
        val_mask = ~train_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        val_meta = [pair_meta[i] for i in range(len(pair_meta)) if val_mask[i]]

        clf = lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.08,
            num_leaves=31,
            max_depth=6,
            scale_pos_weight=1.5,
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)

        val_probs = clf.predict_proba(X_val)[:, 1]
        for (s1_id, cid), prob in zip(val_meta, val_probs):
            oof_scores.append((s1_id, cid, prob))

    # Evaluate Macro F_0.5 across different probability thresholds
    print("\n--- OOF EVALUATION ACROSS PROBABILITY THRESHOLDS ---")
    best_f = 0
    best_th = 0

    # Group scores by s1
    for th in np.linspace(0.40, 0.90, 26):
        preds = {s1: [] for s1 in s1_ids}
        # Enforce unique target assignment
        best_s1_for_target = {}
        for s1, cid, score in oof_scores:
            if score >= th:
                if cid not in best_s1_for_target or score > best_s1_for_target[cid][1]:
                    best_s1_for_target[cid] = (s1, score)

        for cid, (s1, score) in best_s1_for_target.items():
            preds[s1].append(cid)

        # Cap at 6
        for s1 in preds:
            preds[s1] = preds[s1][:6]

        res = evaluate_predictions(gt, preds)
        p = res["macro_precision"]
        r = res["macro_recall"]
        f = res["macro_f_beta"]
        if f > best_f:
            best_f = f
            best_th = th
            print(f"Threshold {th:.3f} -> Precision: {p:.4f}, Recall: {r:.4f}, Macro F_0.5: {f:.4f} <<< BEST")
        elif round(th*100) % 10 == 0:
            print(f"Threshold {th:.3f} -> Precision: {p:.4f}, Recall: {r:.4f}, Macro F_0.5: {f:.4f}")

    print("\n" + "="*60)
    print(f"LIGHTGBM 5-FOLD OOF BEST MACRO F_0.5: {best_f:.4f} at threshold {best_th:.3f}")
    print("="*60)

if __name__ == "__main__":
    main()
