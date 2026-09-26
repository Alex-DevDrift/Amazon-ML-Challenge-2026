# Business Entity Resolution Pipeline

This repository contains the end-to-end machine learning pipeline for the **Amazon ML Challenge 2026: Business Entity Resolution Challenge**.

---

## 1. Overview & Architecture

The challenge is to resolve noisy, disparate business entity records from three sources (`Source 1`, `Source 2`, `Source 3`) to reference records in `Source 1`.

Our solution operates in three distinct, highly optimized stages:
1. **Open-World Preprocessing**:
   - Universal case folding, accent/diacritic stripping (vital for France and European entities).
   - Legal suffix detection and normalization (`corp`, `inc`, `ltd`, `pvt ltd`, `llc`, `sa`, `sarl`).
   - Address abbreviation expansion (`rd` -> `road`, `st` -> `street`, `ave` -> `avenue`, `blvd` -> `boulevard`).
   - Open-world country string normalization without restricting labels to `{US, India}`.
2. **High-Reduction Multi-Index Blocking**:
   - Country partition blocks (prevents cross-country candidate explosion).
   - Inverted token indexing on distinctive name tokens.
   - TF-IDF sub-word character n-gram cosine matching ($K \le 15$).
   - Postal / PIN code numeric co-occurrence indexing.
   - Pre-ranking & budget capping to an average of $\approx 5$ candidates per entity (maximizing the Amazon Reduction Ratio score).
3. **Pairwise Feature Engineering & Gradient Boosting Classifier**:
   - 23 lexical, syntactic, character n-gram, and numerical digit conflict features.
   - LightGBM gradient boosted decision trees with custom weighting.
   - Decision threshold $\tau$ tuned specifically for macro-averaged $F_{0.5}$ (weighting precision 2x over recall and rewarding correctly identified singletons).

---

## 2. Directory Structure

```text
code/business_entity_resolution/
├── src/
│   ├── __init__.py           # Package initializer
│   ├── config.py             # Configs, feature definitions, hyperparameters
│   ├── preprocessing.py      # Open-world normalizer & tokenizers
│   ├── blocking.py           # Multi-index blocker & candidate_pairs exporter
│   ├── features.py           # Pairwise feature extraction
│   ├── model.py              # LightGBM classifier & persistence
│   ├── evaluate.py           # Macro F_0.5 evaluator & threshold tuner
│   ├── pipeline.py           # End-to-end orchestration
│   └── run_inference.py      # Runnable CLI entrypoint
├── README.md                 # Reproduction guide
└── requirements.txt          # Pinned dependencies
```

---

## 3. Environment Setup

Install dependencies with Python 3.9+:

```bash
pip install -r requirements.txt
```

---

## 4. End-to-End Execution

### A. Run on Provided Dataset
To train the model on `dataset/train/` and run inference on `dataset/test/`:

```bash
python src/run_inference.py \
    --train-dir ../../dataset/train \
    --test-dir ../../dataset/test \
    --output-dir ../../output
```

### B. Run on Sample / Validation Data
If testing on the synthetic benchmark:

```bash
python src/run_inference.py \
    --train-dir ../../dataset/sample/train \
    --test-dir ../../dataset/sample/test \
    --output-dir ../../output
```

The script automatically executes:
1. Data loading with tab separators (`sep="\t"`)
2. Training and validation split
3. Blocking and candidate generation (`output/candidate_pairs.tsv`)
4. Pairwise feature extraction
5. LightGBM model training
6. Validation threshold optimization for Macro $F_{0.5}$
7. Test candidate generation & feature scoring
8. Output generation (`output/matching_results.tsv`)
9. Local validation via `utils/validate_submission.py` (verifying format, subset condition, and ID constraints).

---

## 5. Outputs Produced

- `output/matching_results.tsv`: Tab-separated entity match predictions for leaderboard submission.
- `output/candidate_pairs.tsv`: Tab-separated blocking candidates fed into model inference.
