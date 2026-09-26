"""
End-to-End Orchestrator Pipeline for Business Entity Resolution.
Handles:
- Data ingestion with explicit tab separators
- Preprocessing and open-set country normalization
- Candidate generation and candidate_pairs.tsv generation
- Pairwise feature extraction
- Model training & Macro F_0.5 threshold optimization
- Final inference and matching_results.tsv generation
- Automatic validation
"""

import os
import csv
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

try:
    from .config import Config, FEATURE_COLUMNS
    from .preprocessing import normalize_business_name, normalize_address, normalize_country, extract_address_digits
    from .blocking import EntityBlocker, export_candidate_pairs_tsv
    from .features import build_feature_dataframe
    from .model import EntityMatcherModel
    from .evaluate import evaluate_predictions, find_optimal_threshold
except ImportError:
    from config import Config, FEATURE_COLUMNS
    from preprocessing import normalize_business_name, normalize_address, normalize_country, extract_address_digits
    from blocking import EntityBlocker, export_candidate_pairs_tsv
    from features import build_feature_dataframe
    from model import EntityMatcherModel
    from evaluate import evaluate_predictions, find_optimal_threshold


def load_dataset_split(
    data_dir: str,
    prefix: str = "train",
    is_train: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[Dict[str, List[str]]]]:
    """
    Loads TSV files with explicit tab separator.
    """
    s1_path = os.path.join(data_dir, f"{prefix}_source1.tsv")
    s2_path = os.path.join(data_dir, f"{prefix}_source2.tsv")
    s3_path = os.path.join(data_dir, f"{prefix}_source3.tsv")

    print(f"Loading data from {data_dir} with prefix '{prefix}'...")
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False)
    df_s2 = pd.read_csv(s2_path, sep="\t", dtype=str, keep_default_na=False)
    df_s3 = pd.read_csv(s3_path, sep="\t", dtype=str, keep_default_na=False)

    ground_truth = None
    if is_train:
        gt_path = os.path.join(data_dir, f"{prefix}_ground_truth.tsv")
        if os.path.exists(gt_path):
            df_gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
            ground_truth = {}
            for _, row in df_gt.iterrows():
                s1_id = row["source1_entity_id"].strip()
                matches_str = row["matched_entity_ids"].strip()
                if matches_str:
                    ground_truth[s1_id] = [m.strip() for m in matches_str.split(",") if m.strip()]
                else:
                    ground_truth[s1_id] = []

    print(f"Loaded: S1={len(df_s1)}, S2={len(df_s2)}, S3={len(df_s3)}, GT={len(ground_truth) if ground_truth else 0}")
    return df_s1, df_s2, df_s3, ground_truth


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess text fields in a source dataframe."""
    df = df.copy()
    df["norm_name"] = df["business_name"].apply(normalize_business_name)
    df["norm_addr"] = df["business_address"].apply(normalize_address)
    df["norm_country"] = df["country"].apply(normalize_country)
    df["addr_digits"] = df["business_address"].apply(extract_address_digits)
    return df


def export_matching_results_tsv(
    predictions_dict: Dict[str, List[str]],
    filepath: str,
):
    """
    Exports final matching predictions to matching_results.tsv.
    Format:
    source1_entity_id \t matched_entity_ids
    """
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id, matches in predictions_dict.items():
            matches_str = ",".join(matches)
            f.write(f"{s1_id}\t{matches_str}\n")


class EntityResolutionPipeline:
    """
    Orchestrates the entire end-to-end Entity Resolution pipeline.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.blocker = EntityBlocker(
            max_candidates_per_entity=self.config.max_candidates_per_entity,
            min_lexical_score=self.config.candidate_pre_filter_threshold,
        )
        self.model = EntityMatcherModel(
            model_type=self.config.model_type,
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            scale_pos_weight=self.config.scale_pos_weight,
            random_state=self.config.random_state,
        )
        self.best_threshold = self.config.default_threshold

    def run_training(
        self,
        val_split_ratio: float = 0.2,
    ):
        """
        Trains model on train dataset and validates threshold for Macro F_0.5.
        """
        print("\n=== STEP 1: Training Data Loading & Preprocessing ===")
        df_s1, df_s2, df_s3, ground_truth = load_dataset_split(
            self.config.train_dir, prefix="train", is_train=True
        )

        df_s1 = preprocess_dataframe(df_s1)
        df_s2 = preprocess_dataframe(df_s2)
        df_s3 = preprocess_dataframe(df_s3)

        # Split train entities into Train and Validation sets
        np.random.seed(self.config.random_state)
        s1_ids = df_s1["entity_id"].values
        shuffled_ids = np.random.permutation(s1_ids)
        split_idx = int(len(shuffled_ids) * (1 - val_split_ratio))

        train_s1_ids = set(shuffled_ids[:split_idx])
        val_s1_ids = set(shuffled_ids[split_idx:])

        df_s1_train = df_s1[df_s1["entity_id"].isin(train_s1_ids)].reset_index(drop=True)
        df_s1_val = df_s1[df_s1["entity_id"].isin(val_s1_ids)].reset_index(drop=True)

        print(f"Entities: Train={len(df_s1_train)}, Val={len(df_s1_val)}")

        # Step 2: Blocking on Train & Val
        print("\n=== STEP 2: Candidate Generation (Blocking) ===")
        train_candidates = self.blocker.generate_candidates(df_s1_train, df_s2, df_s3)
        val_candidates = self.blocker.generate_candidates(df_s1_val, df_s2, df_s3)

        avg_train_cands = np.mean([len(c) for c in train_candidates.values()])
        avg_val_cands = np.mean([len(c) for c in val_candidates.values()])
        print(f"Avg candidates per entity: Train={avg_train_cands:.2f}, Val={avg_val_cands:.2f}")

        # Target lookup dictionary
        target_df = pd.concat([df_s2, df_s3], ignore_index=True)
        target_dict = {r["entity_id"]: r for r in target_df.to_dict(orient="records")}
        s1_train_dict = {r["entity_id"]: r for r in df_s1_train.to_dict(orient="records")}
        s1_val_dict = {r["entity_id"]: r for r in df_s1_val.to_dict(orient="records")}

        # Step 3: Feature Extraction
        print("\n=== STEP 3: Feature Extraction ===")
        df_feats_train = build_feature_dataframe(
            s1_train_dict, target_dict, train_candidates, FEATURE_COLUMNS
        )
        df_feats_val = build_feature_dataframe(
            s1_val_dict, target_dict, val_candidates, FEATURE_COLUMNS
        )

        # Labels
        def get_labels(df_feats, gt_dict):
            labels = []
            for _, r in df_feats.iterrows():
                s1_id = r["source1_entity_id"]
                c_id = r["candidate_entity_id"]
                is_match = 1 if (s1_id in gt_dict and c_id in gt_dict[s1_id]) else 0
                labels.append(is_match)
            return np.array(labels)

        y_train = get_labels(df_feats_train, ground_truth)
        y_val = get_labels(df_feats_val, ground_truth)

        pos_train = int(np.sum(y_train))
        pos_val = int(np.sum(y_val))
        print(f"Train pairs: {len(df_feats_train)} (Positives: {pos_train})")
        print(f"Val pairs: {len(df_feats_val)} (Positives: {pos_val})")

        # Step 4: Model Fitting
        print("\n=== STEP 4: Model Training ===")
        self.model.fit(
            df_feats_train,
            y_train,
            feature_cols=FEATURE_COLUMNS,
            eval_set=(df_feats_val, y_val) if len(df_feats_val) > 0 else None,
        )

        # Step 5: Validation & Threshold Optimization
        print("\n=== STEP 5: Threshold Optimization for Macro F_0.5 ===")
        if len(df_feats_val) > 0:
            val_probs = self.model.predict_proba(df_feats_val)
            val_pair_scores = []
            for idx, r in df_feats_val.iterrows():
                val_pair_scores.append(
                    (r["source1_entity_id"], r["candidate_entity_id"], val_probs[idx])
                )

            val_gt = {eid: ground_truth.get(eid, []) for eid in val_s1_ids}
            best_thresh, best_f05 = find_optimal_threshold(
                val_gt, val_pair_scores, val_candidates, beta=0.5
            )
            self.best_threshold = best_thresh
            print(f"Optimal Threshold: {best_thresh:.3f} | Best Validation Macro F_0.5: {best_f05:.4f}")
        else:
            self.best_threshold = self.config.default_threshold

        # Save model
        model_path = os.path.join(self.config.model_dir, "entity_matcher.pkl")
        self.model.save(model_path)

    def run_inference(self):
        """
        Runs candidate generation and matching inference on test set.
        Generates candidate_pairs.tsv and matching_results.tsv.
        """
        print("\n=== STEP 6: Test Data Inference ===")
        df_s1_test, df_s2_test, df_s3_test, _ = load_dataset_split(
            self.config.test_dir, prefix="test", is_train=False
        )

        df_s1_test = preprocess_dataframe(df_s1_test)
        df_s2_test = preprocess_dataframe(df_s2_test)
        df_s3_test = preprocess_dataframe(df_s3_test)

        # 1. Blocking / Candidate Generation
        print("Running candidate generation on test entities...")
        test_candidates = self.blocker.generate_candidates(
            df_s1_test, df_s2_test, df_s3_test
        )

        # Guarantee every test S1 entity is present
        for s1_id in df_s1_test["entity_id"].values:
            if s1_id not in test_candidates:
                test_candidates[s1_id] = []

        # Export candidate_pairs.tsv
        cand_pairs_path = os.path.join(self.config.output_dir, "candidate_pairs.tsv")
        export_candidate_pairs_tsv(test_candidates, cand_pairs_path)
        print(f"Exported candidate pairs to {cand_pairs_path}")

        # 2. Feature Extraction on Test Candidates
        target_df_test = pd.concat([df_s2_test, df_s3_test], ignore_index=True)
        target_dict_test = {r["entity_id"]: r for r in target_df_test.to_dict(orient="records")}
        s1_dict_test = {r["entity_id"]: r for r in df_s1_test.to_dict(orient="records")}

        df_feats_test = build_feature_dataframe(
            s1_dict_test, target_dict_test, test_candidates, FEATURE_COLUMNS
        )

        # 3. Model Scoring
        print(f"Scoring {len(df_feats_test)} test candidate pairs...")
        final_matches: Dict[str, List[str]] = {
            eid: [] for eid in df_s1_test["entity_id"].values
        }

        if len(df_feats_test) > 0:
            test_probs = self.model.predict_proba(df_feats_test)
            for idx, r in df_feats_test.iterrows():
                prob = test_probs[idx]
                if prob >= self.best_threshold:
                    s1_id = r["source1_entity_id"]
                    cand_id = r["candidate_entity_id"]
                    final_matches[s1_id].append(cand_id)

        # Guarantee subset condition: final matches must be a subset of candidate pairs
        for s1_id, matches in final_matches.items():
            valid_cands = set(test_candidates.get(s1_id, []))
            final_matches[s1_id] = [m for m in matches if m in valid_cands]

        # Export matching_results.tsv
        matching_path = os.path.join(self.config.output_dir, "matching_results.tsv")
        export_matching_results_tsv(final_matches, matching_path)
        print(f"Exported final matches to {matching_path}")

        return matching_path, cand_pairs_path
