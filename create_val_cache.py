import os
import sys
import time
import re
import pickle
import unicodedata
from anyascii import anyascii

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
TRAIN_DIR = os.path.join(PROJECT_ROOT, "dataset", "train")

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

def extract_keys(name: str, addr: str, country: str):
    c_norm = country.strip().lower()
    keys = []

    # Domain extraction
    domains = re.findall(r"([a-z0-9\-]+)\.(?:com|org|net|in|co|io|fr|gov)", name.lower())
    for d in domains:
        if len(d) >= 4 and d not in ("www", "http", "https"):
            keys.append((c_norm, "domain", d))

    name_tokens = clean_tokens(name)
    content_tokens = [t for t in name_tokens if t not in STOPWORDS and len(t) >= 2]

    if content_tokens:
        compact_name = "".join(content_tokens)
        keys.append((c_norm, "name_full", compact_name[:24]))
        if len(content_tokens) >= 2:
            keys.append((c_norm, "name_pair", content_tokens[0] + "_" + content_tokens[1]))
            # Also first + last token if >= 3 tokens
            if len(content_tokens) >= 3:
                keys.append((c_norm, "name_first_last", content_tokens[0] + "_" + content_tokens[-1]))
        elif len(content_tokens) == 1 and len(content_tokens[0]) >= 3:
            keys.append((c_norm, "name_single", content_tokens[0]))

    # PIN / Postal code keys
    pins = re.findall(r"\b\d{5,6}\b", addr)
    if pins and content_tokens:
        keys.append((c_norm, "pin_name", pins[0], content_tokens[0]))

    # Address numbers + street/locality token
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

def main():
    print("Loading 1,000 validation S1 entities and ground truth...")
    gt = {}
    with open(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            gt[p[0]] = p[1].split(",") if len(p) > 1 and p[1] else []
            if len(gt) >= 1000:
                break

    s1_entities = {}
    with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in gt:
                s1_entities[p[0]] = (p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else "", p[3] if len(p) > 3 else "")
            if len(s1_entities) >= 1000:
                break

    needed_targets = set()
    for m in gt.values():
        needed_targets.update(m)

    s1_all_keys = set()
    for s1_id, (n, a, c) in s1_entities.items():
        for k in extract_keys(n, a, c):
            s1_all_keys.add(k)

    print(f"Total S1 query keys: {len(s1_all_keys)}. Indexing matching targets...")
    t0 = time.time()

    candidates_by_key = {}
    target_records = {}

    for fn in ["train_source2.tsv", "train_source3.tsv"]:
        path = os.path.join(TRAIN_DIR, fn)
        print(f"Scanning {fn}...")
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            next(f)
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                if len(p) < 4:
                    continue
                eid, name, addr, country = p[0], p[1], p[2], p[3]

                # Check if this eid is in ground truth or matches any S1 key
                is_gt = eid in needed_targets
                keys = extract_keys(name, addr, country)
                matching_keys = [k for k in keys if k in s1_all_keys]

                if is_gt or matching_keys:
                    target_records[eid] = (name, addr, country)
                    for k in matching_keys:
                        if k not in candidates_by_key:
                            candidates_by_key[k] = []
                        if len(candidates_by_key[k]) < 12:
                            candidates_by_key[k].append(eid)

    print(f"Collected {len(target_records)} target records in {time.time()-t0:.2f}s.")
    val_data = {
        "s1_entities": s1_entities,
        "gt": gt,
        "candidates_by_key": candidates_by_key,
        "target_records": target_records,
    }
    with open("val_1000_cache.pkl", "wb") as f:
        pickle.dump(val_data, f)
    print("Saved val_1000_cache.pkl successfully!")

if __name__ == "__main__":
    main()
