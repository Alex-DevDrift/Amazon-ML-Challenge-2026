"""
Feature Engineering for Entity Resolution Pairwise Classification.
Extracts phonetic, character n-gram, token-level, numeric/postal overlap,
and interaction features for each candidate pair.
"""

from typing import Dict, List, Any, Set
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance

try:
    from .preprocessing import (
        extract_legal_suffix,
        extract_address_digits,
        get_character_ngrams,
    )
except ImportError:
    from preprocessing import (
        extract_legal_suffix,
        extract_address_digits,
        get_character_ngrams,
    )


def compute_jaccard(set_a: Set[Any], set_b: Set[Any]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def extract_pair_features(
    s1_row: Dict[str, Any],
    cand_row: Dict[str, Any],
) -> Dict[str, float]:
    """
    Computes fine-grained matching features for an (S1, Candidate) pair.
    """
    s1_name = s1_row.get("norm_name", "")
    cand_name = cand_row.get("norm_name", "")

    s1_addr = s1_row.get("norm_addr", "")
    cand_addr = cand_row.get("norm_addr", "")

    # 1. Name Similarities
    lev_ratio = fuzz.ratio(s1_name, cand_name) / 100.0
    token_sort = fuzz.token_sort_ratio(s1_name, cand_name) / 100.0
    token_set = fuzz.token_set_ratio(s1_name, cand_name) / 100.0
    partial = fuzz.partial_ratio(s1_name, cand_name) / 100.0

    # 3-gram character Jaccard
    s1_ngrams = get_character_ngrams(s1_name, 3)
    cand_ngrams = get_character_ngrams(cand_name, 3)
    ngram_jaccard = compute_jaccard(s1_ngrams, cand_ngrams)

    exact_match = 1.0 if s1_name == cand_name and s1_name != "" else 0.0

    # First token match
    s1_tokens = s1_name.split()
    cand_tokens = cand_name.split()
    first_token_match = (
        1.0 if s1_tokens and cand_tokens and s1_tokens[0] == cand_tokens[0] else 0.0
    )

    len_s1 = len(s1_name)
    len_cand = len(cand_name)
    len_diff = abs(len_s1 - len_cand)
    len_ratio = (min(len_s1, len_cand) / max(len_s1, len_cand)) if max(len_s1, len_cand) > 0 else 1.0

    # Legal suffix consistency
    s1_clean, s1_sfx = extract_legal_suffix(s1_row.get("business_name", ""))
    cand_clean, cand_sfx = extract_legal_suffix(cand_row.get("business_name", ""))

    suffix_match = 1.0 if (s1_sfx and cand_sfx and s1_sfx == cand_sfx) else 0.0
    suffix_conflict = 1.0 if (s1_sfx and cand_sfx and s1_sfx != cand_sfx) else 0.0

    # 2. Address Similarities
    addr_lev = fuzz.ratio(s1_addr, cand_addr) / 100.0
    addr_token_sort = fuzz.token_sort_ratio(s1_addr, cand_addr) / 100.0
    addr_token_set = fuzz.token_set_ratio(s1_addr, cand_addr) / 100.0

    # Word-level Jaccard
    s1_words = set(s1_addr.split())
    cand_words = set(cand_addr.split())
    addr_word_jaccard = compute_jaccard(s1_words, cand_words)

    # Digit / PIN / Postal Code overlap
    s1_digits = s1_row.get("addr_digits") or extract_address_digits(s1_row.get("business_address", ""))
    cand_digits = cand_row.get("addr_digits") or extract_address_digits(cand_row.get("business_address", ""))
    digit_overlap = compute_jaccard(s1_digits, cand_digits)

    # Conflict: both have numbers, but none match
    digit_conflict = (
        1.0 if s1_digits and cand_digits and not (s1_digits & cand_digits) else 0.0
    )

    addr_len_diff = abs(len(s1_addr) - len(cand_addr))
    addr_substring = (
        1.0 if (s1_addr and cand_addr and (s1_addr in cand_addr or cand_addr in s1_addr)) else 0.0
    )

    # 3. Interactions
    harmonic_name_addr = (
        2.0 * (token_sort * addr_token_set) / (token_sort + addr_token_set + 1e-6)
    )
    min_name_addr = min(token_sort, addr_token_set)
    max_name_addr = max(token_sort, addr_token_set)

    is_s2 = 1.0 if cand_row.get("entity_id", "").startswith("S2-") else 0.0

    return {
        "name_levenshtein_ratio": lev_ratio,
        "name_token_sort_ratio": token_sort,
        "name_token_set_ratio": token_set,
        "name_partial_ratio": partial,
        "name_3gram_jaccard": ngram_jaccard,
        "name_exact_match": exact_match,
        "name_first_token_match": first_token_match,
        "name_length_diff": float(len_diff),
        "name_length_ratio": len_ratio,
        "name_suffix_match": suffix_match,
        "name_suffix_conflict": suffix_conflict,
        "addr_levenshtein_ratio": addr_lev,
        "addr_token_sort_ratio": addr_token_sort,
        "addr_token_set_ratio": addr_token_set,
        "addr_word_jaccard": addr_word_jaccard,
        "addr_digit_overlap_ratio": digit_overlap,
        "addr_digit_conflict": digit_conflict,
        "addr_length_diff": float(addr_len_diff),
        "addr_substring_match": addr_substring,
        "harmonic_name_addr": harmonic_name_addr,
        "min_name_addr": min_name_addr,
        "max_name_addr": max_name_addr,
        "is_source2": is_s2,
    }


def build_feature_dataframe(
    s1_dict: Dict[str, Dict[str, Any]],
    target_dict: Dict[str, Dict[str, Any]],
    candidate_pairs: Dict[str, List[str]],
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Constructs feature matrix for all (S1, Candidate) pairs.
    """
    rows = []
    for s1_id, cands in candidate_pairs.items():
        if s1_id not in s1_dict:
            continue
        s1_row = s1_dict[s1_id]
        for c_id in cands:
            if c_id not in target_dict:
                continue
            c_row = target_dict[c_id]
            feats = extract_pair_features(s1_row, c_row)
            feats["source1_entity_id"] = s1_id
            feats["candidate_entity_id"] = c_id
            rows.append(feats)

    if not rows:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"] + feature_cols)

    df_feats = pd.DataFrame(rows)
    return df_feats
