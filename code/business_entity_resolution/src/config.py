"""
Configuration module for Business Entity Resolution.
Defines default paths, hyperparameter configurations, and feature lists.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Project paths
    project_root: str = field(
        default_factory=lambda: os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
    )
    dataset_dir: str = ""
    train_dir: str = ""
    test_dir: str = ""
    output_dir: str = ""
    model_dir: str = ""

    # Blocking parameters
    max_candidates_per_entity: int = 15  # Strict cap to maximize blocking reduction score
    tfidf_ngram_range: tuple = (1, 2)
    tfidf_max_features: int = 50000
    candidate_pre_filter_threshold: float = 0.12  # Min lexical score to be considered

    # Model parameters
    model_type: str = "lightgbm"  # 'lightgbm' or 'gradient_boosting'
    random_state: int = 42
    n_estimators: int = 250
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = 6
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    scale_pos_weight: float = 2.0

    # Decision threshold for macro F_0.5 (penalizes false positives 2x over false negatives)
    default_threshold: float = 0.65
    tune_threshold: bool = True

    def __post_init__(self):
        if not self.dataset_dir:
            self.dataset_dir = os.path.join(self.project_root, "dataset")
        if not self.train_dir:
            # Check if default real dataset exists, else fallback to sample
            real_train = os.path.join(self.dataset_dir, "train")
            sample_train = os.path.join(self.dataset_dir, "sample", "train")
            self.train_dir = real_train if os.path.exists(real_train) else sample_train

        if not self.test_dir:
            real_test = os.path.join(self.dataset_dir, "test")
            sample_test = os.path.join(self.dataset_dir, "sample", "test")
            self.test_dir = real_test if os.path.exists(real_test) else sample_test

        if not self.output_dir:
            self.output_dir = os.path.join(self.project_root, "output")

        if not self.model_dir:
            self.model_dir = os.path.join(
                self.project_root, "code", "business_entity_resolution", "models"
            )

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)


FEATURE_COLUMNS = [
    "name_levenshtein_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_3gram_jaccard",
    "name_exact_match",
    "name_first_token_match",
    "name_length_diff",
    "name_length_ratio",
    "name_suffix_match",
    "name_suffix_conflict",
    "addr_levenshtein_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_word_jaccard",
    "addr_digit_overlap_ratio",
    "addr_digit_conflict",
    "addr_length_diff",
    "addr_substring_match",
    "harmonic_name_addr",
    "min_name_addr",
    "max_name_addr",
    "is_source2",
]
