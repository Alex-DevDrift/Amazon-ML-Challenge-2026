import os
import sys
import pickle
import re
from rapidfuzz import fuzz

sys.path.append(os.path.join(os.path.dirname(__file__), "code", "business_entity_resolution", "src"))
from evaluate import evaluate_predictions, compute_entity_f_beta

LEGAL_SUFFIXES = {
    "corp", "corporation", "inc", "incorporated", "ltd", "limited",
    "pvt", "private", "llc", "co", "company", "sa", "sarl", "gmbh", "group", "holdings",
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
    from anyascii import anyascii
    t = anyascii(text).lower()
    for pat, rep in PHONETIC_SUBS:
        t = re.sub(pat, rep, t)
    return t

def clean_tokens(text: str):
    norm = normalize_text(text)
    return re.findall(r"[a-z0-9]+", norm)

def extract_keys(name: str, addr: str, country: str):
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
                keys.append((c_norm, "domain", d))

        name_tokens = clean_tokens(v_name)
        c_tokens = [t for t in name_tokens if t not in STOPWORDS and len(t) >= 2]
        if not content_tokens:
            content_tokens = c_tokens

        if c_tokens:
            compact_name = "".join(c_tokens)
            keys.append((c_norm, "name_full", compact_name[:24]))
            if len(c_tokens) >= 2:
                keys.append((c_norm, "name_pair", c_tokens[0] + "_" + c_tokens[1]))
                if len(c_tokens) >= 3:
                    keys.append((c_norm, "name_first_last", c_tokens[0] + "_" + c_tokens[-1]))
            if len(c_tokens[0]) >= 3:
                keys.append((c_norm, "name_first", c_tokens[0]))

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
        keys.append((c_norm, "addr_num", nums[0], addr_tokens[0]))
    if nums and len(content_tokens) > 0:
        keys.append((c_norm, "num_name", nums[0], content_tokens[0]))

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

def run_optimization(cache_file="val_1000_cache.pkl"):
    print("Loading cache...")
    with open(cache_file, "rb") as f:
        data = pickle.load(f)

    s1_entities = data["s1_entities"]
    gt = data["gt"]
    target_records = data["target_records"]

    print("Re-indexing cached targets with v2 blocking keys...")
    index = {}
    for eid, (t_n, t_a, t_c) in target_records.items():
        keys = extract_keys(t_n, t_a, t_c)
        for k in keys:
            if k not in index:
                index[k] = []
            if len(index[k]) < 20:
                index[k].append(eid)

    # Pre-extract S1 candidate pools and features
    print("Pre-scoring candidate pairs...")
    s1_candidate_data = {}
    total_gt = 0
    captured_gt = 0

    for s1_id, (s1_name, s1_addr, s1_country) in s1_entities.items():
        s1_norm_n = normalize_text(s1_name)
        s1_norm_a = normalize_text(s1_addr)
        s1_n_tokens = [t for t in clean_tokens(s1_norm_n) if t not in STOPWORDS]
        s1_clean_n = " ".join(s1_n_tokens)
        s1_addr_norm = re.sub(r"\b0+(\d+)\b", r"\1", s1_norm_a)
        s1_digits = set(re.findall(r"\b\d+\b", s1_addr_norm))

        keys = extract_keys(s1_name, s1_addr, s1_country)
        cands = set()
        for k in keys:
            if k in index:
                cands.update(index[k])

        true_m = set(gt.get(s1_id, []))
        total_gt += len(true_m)
        captured_gt += len(true_m & cands)

        cand_features = []
        for cid in cands:
            if cid not in target_records:
                continue
            t_name, t_addr, t_country = target_records[cid]
            t_norm_n = normalize_text(t_name)
            t_norm_a = normalize_text(t_addr)
            t_n_tokens = [t for t in clean_tokens(t_norm_n) if t not in STOPWORDS]
            t_clean_n = " ".join(t_n_tokens)
            t_addr_norm = re.sub(r"\b0+(\d+)\b", r"\1", t_norm_a)
            t_digits = set(re.findall(r"\b\d+\b", t_addr_norm))

            # Features
            digit_overlap = len(s1_digits & t_digits)
            has_num_conflict = (len(s1_digits) > 0 and len(t_digits) > 0 and digit_overlap == 0)

            n_sim = fuzz.token_sort_ratio(s1_clean_n, t_clean_n)
            n_set = fuzz.token_set_ratio(s1_clean_n, t_clean_n)
            best_n_sim = max(n_sim, n_set)

            a_sim = fuzz.token_set_ratio(s1_addr_norm, t_addr_norm) if (s1_addr_norm and t_addr_norm) else 0.0
            dom_acr = is_domain_or_acronym_match(s1_norm_n, t_norm_n, s1_n_tokens, t_n_tokens)

            cand_features.append({
                "cid": cid,
                "n_sim": best_n_sim,
                "a_sim": a_sim,
                "has_num_conflict": has_num_conflict,
                "digit_overlap": digit_overlap,
                "dom_acr": dom_acr,
                "t_has_addr": bool(t_norm_a.strip()),
            })

        s1_candidate_data[s1_id] = cand_features

    print(f"CANDIDATE BLOCKING RECALL: {captured_gt/total_gt*100:.2f}% ({captured_gt}/{total_gt} GT captured)")

    # Grid search over matching criteria to maximize Macro F_0.5
    print("\n--- GRID SEARCH FOR MAXIMUM MACRO F_0.5 ---")
    best_f = -1.0
    best_params = None

    for name_thresh in [80.0, 82.0, 84.0, 86.0, 88.0, 90.0]:
        for addr_thresh in [50.0, 55.0, 60.0, 65.0, 70.0]:
            for addr_high in [76.0, 80.0, 84.0]:
                for name_low in [45.0, 50.0]:
                    for cand_cap in [6, 8]:
                        preds = {}
                        cand_dict = {}
                        for s1_id, cands in s1_candidate_data.items():
                            scored = []
                            for c in cands:
                                rank = c["n_sim"] * 0.65 + c["a_sim"] * 0.35
                                if c["dom_acr"]: rank = max(rank, 85.0)

                                is_match = False
                                # Hard reject if names have near zero overlap and not domain/acronym
                                if c["n_sim"] < name_low and not c["dom_acr"]:
                                    is_match = False
                                # Hard reject if distinct street numbers conflict
                                elif c["has_num_conflict"] and c["n_sim"] < 92.0:
                                    is_match = False
                                # High address + moderate name
                                elif c["a_sim"] >= addr_high and (c["n_sim"] >= name_low or c["dom_acr"]):
                                    is_match = True
                                # High name + moderate address (or empty address / shared digit)
                                elif c["n_sim"] >= name_thresh:
                                    if c["a_sim"] >= addr_thresh or not c["t_has_addr"] or c["digit_overlap"] > 0:
                                        is_match = True
                                elif c["dom_acr"] and (c["a_sim"] >= 30.0 or not c["t_has_addr"] or c["digit_overlap"] > 0):
                                    is_match = True

                                scored.append((c["cid"], rank, is_match))

                            scored.sort(key=lambda x: x[1], reverse=True)
                            top_c = [cid for cid, r, m in scored[:cand_cap]]
                            top_m = [cid for cid, r, m in scored[:cand_cap] if m]
                            preds[s1_id] = top_m
                            cand_dict[s1_id] = top_c

                        res = evaluate_predictions(gt, preds)
                        if res["macro_f_beta"] > best_f:
                            best_f = res["macro_f_beta"]
                            best_params = (name_thresh, addr_thresh, addr_high, name_low, cand_cap, res)
                            print(f"NEW BEST F_0.5: {best_f:.4f} (P: {res['macro_precision']:.4f}, R: {res['macro_recall']:.4f}) "
                                  f"| name_th={name_thresh}, addr_th={addr_thresh}, addr_hi={addr_high}, name_lo={name_low}, cap={cand_cap}")

    print("\n" + "="*70)
    print("OPTIMAL CALIBRATED CONFIGURATION:")
    b_nt, b_at, b_ah, b_nl, b_cap, b_res = best_params
    print(f"Macro Precision: {b_res['macro_precision']:.4f}")
    print(f"Macro Recall:    {b_res['macro_recall']:.4f}")
    print(f"Macro F_0.5:     {b_res['macro_f_beta']:.4f}")
    print(f"Best parameters: name_thresh={b_nt}, addr_thresh={b_at}, addr_high={b_ah}, name_low={b_nl}, cap={b_cap}")
    print("="*70)

if __name__ == "__main__":
    run_optimization()
