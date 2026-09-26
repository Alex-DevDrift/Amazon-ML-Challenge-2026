"""
Scalable Blocking and Candidate Generation for Business Entity Resolution.
Implements multi-key country partitioning, inverted index retrieval,
TF-IDF sparse cosine matching, and dynamic candidate budget capping.
Ensures high recall ceiling while maximizing reduction ratio (eval criteria #2).
"""

import collections
import re
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz import fuzz

try:
    from .preprocessing import (
        normalize_business_name,
        normalize_address,
        normalize_country,
        extract_address_digits,
        get_character_ngrams,
    )
except ImportError:
    from preprocessing import (
        normalize_business_name,
        normalize_address,
        normalize_country,
        extract_address_digits,
        get_character_ngrams,
    )

STOPWORDS = {
    "the", "and", "of", "in", "at", "for", "on", "by", "to", "with",
    "limited", "corporation", "company", "incorporated", "private", "llc",
    "sa", "sarl", "gmbh", "group", "services", "enterprises", "holdings",
}


def tokenize_name_for_blocking(norm_name: str) -> List[str]:
    """Extract informative name tokens for inverted index blocking."""
    tokens = norm_name.split()
    return [t for t in tokens if len(t) >= 3 and t not in STOPWORDS]


class EntityBlocker:
    """
    Hierarchical multi-index candidate generation engine.
    """

    def __init__(
        self,
        max_candidates_per_entity: int = 15,
        min_lexical_score: float = 12.0,
    ):
        self.max_candidates = max_candidates_per_entity
        self.min_lexical_score = min_lexical_score

    def generate_candidates(
        self,
        df_s1: pd.DataFrame,
        df_s2: pd.DataFrame,
        df_s3: pd.DataFrame,
    ) -> Dict[str, List[str]]:
        """
        Generates candidate (S1, S2/S3) pairs partitioned by country.
        Returns dict: {s1_id: [candidate_ids]}
        """
        # Ensure preprocessing columns exist
        s1 = df_s1.copy()
        s2 = df_s2.copy()
        s3 = df_s3.copy()

        for df in [s1, s2, s3]:
            if "norm_country" not in df.columns:
                df["norm_country"] = df["country"].apply(normalize_country)
            if "norm_name" not in df.columns:
                df["norm_name"] = df["business_name"].apply(normalize_business_name)
            if "norm_addr" not in df.columns:
                df["norm_addr"] = df["business_address"].apply(normalize_address)
            if "addr_digits" not in df.columns:
                df["addr_digits"] = df["business_address"].apply(extract_address_digits)

        # Pool Source 2 and Source 3
        df_target = pd.concat([s2, s3], ignore_index=True)

        candidates_by_s1: Dict[str, List[str]] = {
            eid: [] for eid in s1["entity_id"].values
        }

        # Group by country (open-world partitioning)
        all_countries = set(s1["norm_country"].unique()) | set(
            df_target["norm_country"].unique()
        )

        for country in all_countries:
            s1_country = s1[s1["norm_country"] == country]
            target_country = df_target[df_target["norm_country"] == country]

            if s1_country.empty or target_country.empty:
                continue

            country_candidates = self._block_country_group(s1_country, target_country)
            for s1_id, cands in country_candidates.items():
                candidates_by_s1[s1_id] = cands

        return candidates_by_s1

    def _block_country_group(
        self,
        s1_df: pd.DataFrame,
        target_df: pd.DataFrame,
    ) -> Dict[str, List[str]]:
        """
        Blocks records within the same country partition using multi-index strategy.
        """
        target_records = target_df.to_dict(orient="records")
        target_id_to_record = {r["entity_id"]: r for r in target_records}

        # 1. Build Inverted Index on Name Tokens
        token_to_targets = collections.defaultdict(set)
        digit_to_targets = collections.defaultdict(set)

        for r in target_records:
            t_id = r["entity_id"]
            tokens = tokenize_name_for_blocking(r["norm_name"])
            for t in tokens:
                token_to_targets[t].add(t_id)

            for d in r["addr_digits"]:
                if len(d) >= 4:  # Postal code or large street number
                    digit_to_targets[d].add(t_id)

        # 2. TF-IDF Cosine Retrieval
        tfidf = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 4),
            min_df=1,
            max_features=40000,
        )
        combined_texts_target = [
            f"{r['norm_name']} {r['norm_addr']}" for r in target_records
        ]
        combined_texts_s1 = [
            f"{r['norm_name']} {r['norm_addr']}" for r in s1_df.to_dict(orient="records")
        ]

        tfidf_target_mat = tfidf.fit_transform(combined_texts_target)
        tfidf_s1_mat = tfidf.transform(combined_texts_s1)

        # Sparse matrix multiplication for fast cosine similarity
        cosine_sim = tfidf_s1_mat.dot(tfidf_target_mat.T)

        results: Dict[str, List[str]] = {}
        target_ids = [r["entity_id"] for r in target_records]

        for idx, s1_row in enumerate(s1_df.to_dict(orient="records")):
            s1_id = s1_row["entity_id"]
            s1_name = s1_row["norm_name"]
            s1_addr = s1_row["norm_addr"]
            s1_tokens = tokenize_name_for_blocking(s1_name)
            s1_digits = s1_row["addr_digits"]

            candidate_pool = set()

            # (A) Inverted Token Matches
            for t in s1_tokens:
                if t in token_to_targets:
                    candidate_pool.update(token_to_targets[t])

            # (B) Digit / Postal Code Co-occurrence
            for d in s1_digits:
                if len(d) >= 4 and d in digit_to_targets:
                    candidate_pool.update(digit_to_targets[d])

            # (C) TF-IDF Top-K Cosine candidates
            row_sims = cosine_sim.getrow(idx).toarray().ravel()
            if len(row_sims) > 0:
                top_k_indices = np.argsort(row_sims)[-self.max_candidates :]
                for target_idx in top_k_indices:
                    if row_sims[target_idx] > 0.05:
                        candidate_pool.add(target_ids[target_idx])

            # Lexical scoring and rank-filtering
            scored_candidates = []
            for c_id in candidate_pool:
                cand_rec = target_id_to_record[c_id]
                c_name = cand_rec["norm_name"]
                c_addr = cand_rec["norm_addr"]

                # Fast token sort ratio
                name_sim = fuzz.token_sort_ratio(s1_name, c_name)
                addr_sim = fuzz.token_set_ratio(s1_addr, c_addr)
                combo_score = 0.65 * name_sim + 0.35 * addr_sim

                if combo_score >= self.min_lexical_score:
                    scored_candidates.append((c_id, combo_score))

            # Sort descending and enforce budget cap
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            top_candidates = [cid for cid, _ in scored_candidates[: self.max_candidates]]
            results[s1_id] = top_candidates

        return results


def export_candidate_pairs_tsv(
    candidates_dict: Dict[str, List[str]],
    filepath: str,
):
    """
    Exports candidate pairs to candidate_pairs.tsv.
    Format:
    source1_entity_id \t candidate_entity_ids
    """
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id, cands in candidates_dict.items():
            cands_str = ",".join(cands)
            f.write(f"{s1_id}\t{cands_str}\n")
