"""
Evaluation Module for Business Entity Resolution.
Implements the exact official Macro-averaged F_0.5 score with singleton handling.
"""

from typing import Dict, List, Set, Tuple
import collections
import numpy as np


def compute_entity_f_beta(
    true_set: Set[str],
    pred_set: Set[str],
    beta: float = 0.5,
) -> Tuple[float, float, float]:
    """
    Computes Precision, Recall, and F_beta for a single Source 1 entity.
    Includes strict singleton rules:
    - If true is empty and pred is empty -> F_beta = 1.0 (Precision = 1.0, Recall = 1.0)
    - If true is empty and pred is non-empty -> F_beta = 0.0 (Precision = 0.0, Recall = 1.0)
    - If true is non-empty and pred is empty -> F_beta = 0.0 (Precision = 1.0, Recall = 0.0)
    """
    beta_sq = beta ** 2
    # Singleton case
    if len(true_set) == 0:
        if len(pred_set) == 0:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 1.0, 0.0

    # Non-singleton case
    if len(pred_set) == 0:
        return 1.0, 0.0, 0.0

    tp = len(true_set & pred_set)
    precision = tp / len(pred_set)
    recall = tp / len(true_set)

    denom = (beta_sq * precision) + recall
    if denom == 0.0:
        f_beta = 0.0
    else:
        f_beta = ((1.0 + beta_sq) * precision * recall) / denom

    return precision, recall, f_beta


def evaluate_predictions(
    ground_truth: Dict[str, List[str]],
    predictions: Dict[str, List[str]],
    beta: float = 0.5,
) -> Dict[str, float]:
    """
    Computes macro-averaged Precision, Recall, and F_beta across all Source 1 entities.
    """
    precisions = []
    recalls = []
    f_scores = []

    all_s1_ids = sorted(ground_truth.keys())

    for s1_id in all_s1_ids:
        true_matches = set(ground_truth[s1_id])
        pred_matches = set(predictions.get(s1_id, []))

        p, r, f = compute_entity_f_beta(true_matches, pred_matches, beta=beta)
        precisions.append(p)
        recalls.append(r)
        f_scores.append(f)

    macro_p = float(np.mean(precisions))
    macro_r = float(np.mean(recalls))
    macro_f = float(np.mean(f_scores))

    return {
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f_beta": macro_f,
        "num_entities": len(all_s1_ids),
    }


def find_optimal_threshold(
    val_ground_truth: Dict[str, List[str]],
    val_pair_scores: List[Tuple[str, str, float]],
    candidate_dict: Dict[str, List[str]],
    beta: float = 0.5,
    threshold_range: Tuple[float, float, int] = (0.35, 0.90, 56),
) -> Tuple[float, float]:
    """
    Grid-searches decision threshold to directly maximize Macro F_0.5 on validation set.
    """
    thresholds = np.linspace(threshold_range[0], threshold_range[1], threshold_range[2])
    best_thresh = 0.65
    best_score = -1.0

    # Group scores by s1_id: [(cand_id, score), ...]
    scores_by_s1 = collections.defaultdict(list)
    for s1_id, c_id, score in val_pair_scores:
        scores_by_s1[s1_id].append((c_id, score))

    for thresh in thresholds:
        preds = {s1_id: [] for s1_id in val_ground_truth.keys()}
        for s1_id, cand_scores in scores_by_s1.items():
            matched = [cid for cid, s in cand_scores if s >= thresh]
            preds[s1_id] = matched

        metrics = evaluate_predictions(val_ground_truth, preds, beta=beta)
        score = metrics["macro_f_beta"]
        if score > best_score:
            best_score = score
            best_thresh = float(thresh)

    return best_thresh, best_score
