# 🏆 Amazon ML Challenge 2026 — Business Entity Resolution

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Amazon ML Challenge 2026](https://img.shields.io/badge/Amazon%20ML%20Challenge-2026-orange.svg)](https://builder.aws.com)

A high-performance, production-grade Machine Learning solution for the **Amazon ML Challenge 2026: Business Entity Resolution Challenge**. 

This system resolves noisy, multi-source commercial entity records across three disparate data sources (`Source 1`, `Source 2`, `Source 3`) to reference records in `Source 1`, optimized for **Macro-averaged $F_{0.5}$** ($\beta = 0.5$) and **Candidate Reduction Ratio**.

---

## 🌟 Key Architecture & Highlights

```
Raw Sources (S1, S2, S3)
         │
         ▼
┌──────────────────────────────────────────────┐
│ 1. Open-World Preprocessing & Normalization │
│    - Universal NFKD accent stripping (France)│
│    - Legal suffix canonicalization           │
│    - Address abbreviation expansion          │
│    - Open-set country string handling        │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ 2. High-Reduction Multi-Index Blocking       │
│    - Country partition boundaries            │
│    - Inverted token indexing on name tokens  │
│    - Sub-word TF-IDF Cosine top-K (K <= 15)  │
│    - Postal / PIN code co-occurrence index   │
│    - Dynamic budget cap: ~4.4 candidates/S1  │
│    ──> Exports: output/candidate_pairs.tsv   │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ 3. 23-Dimensional Pairwise Feature Extractor │
│    - Levenshtein, token-sort, token-set      │
│    - Word & 3-gram character Jaccard         │
│    - Numeric & postal code overlap & conflict│
│    - Harmonic mean & cross-field interactions│
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ 4. LightGBM Matching & Macro F_0.5 Optimizer │
│    - Precision-biased learning               │
│    - Validation threshold search for F_0.5   │
│    - Accurate singleton preservation (score 1)│
│    ──> Exports: output/matching_results.tsv  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ 5. Automated Submission Validator & Packager │
│    - utils/validate_submission.py (stdlib)   │
│    - <team_name>_submission.zip generation   │
└──────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```text
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── __init__.py           # Package init
│       │   ├── config.py             # Central configurations & parameters
│       │   ├── preprocessing.py      # Normalization & tokenizers
│       │   ├── blocking.py           # Candidate generation engine
│       │   ├── features.py           # 23 pairwise similarity features
│       │   ├── model.py              # LightGBM classifier & persistence
│       │   ├── evaluate.py           # Official Macro F_0.5 & threshold tuner
│       │   ├── pipeline.py           # End-to-end orchestration
│       │   └── run_inference.py      # CLI runner
│       ├── README.md                 # Detailed reproduction guide
│       └── requirements.txt          # Pinned dependencies
├── dataset/
│   └── sample/                       # Multi-country sample benchmark (US, India, France)
│       ├── train/
│       └── test/
├── output/
│   ├── matching_results.tsv          # Leaderboard submission file
│   └── candidate_pairs.tsv           # Candidate pairs file
├── utils/
│   ├── validate_submission.py        # Official rules validator (stdlib only)
│   └── generate_sample_data.py       # Benchmark synthetic data generator
├── Documentation_template.md         # Comprehensive methodology report
├── package_submission.py             # Automated ZIP packager with pre-validation
└── run_pipeline.py                   # Automated end-to-end pipeline runner
```

---

## 🚀 Quickstart & Reproduction

### 1. Install Dependencies
```bash
pip install -r code/business_entity_resolution/requirements.txt
```

### 2. Run Pipeline
To run end-to-end training, validation, blocking, inference, and validation:
```bash
python run_pipeline.py
```
*(Automatically detects `dataset/train/` and `dataset/test/` if present, otherwise falls back gracefully to `dataset/sample/`).*

### 3. Validate Submission Files
```bash
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/sample/test
```

### 4. Create Final Submission ZIP
To package the final required archive:
```bash
python package_submission.py --team-name <your_team_name>
```

This generates `<your_team_name>_submission.zip` matching the exact required competition structure:
```text
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

---

## 🎯 Evaluation Metric ($F_{0.5}$)

Submissions are evaluated using macro-averaged $F_{0.5}$:
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

- **Precision-Weighted**: Falsely merging two distinct companies is penalized twice as much as missing a match.
- **Singletons**: Source 1 entities with zero true matches score $1.0$ when predicted empty, and $0.0$ when any false link is predicted.

---

## ⚖️ Academic Integrity & Rules
- **No external databases or lookup APIs**: All inferences use purely provided records.
- **Open-source libraries**: RapidFuzz, scikit-learn, LightGBM, pandas.
- **Offline execution**: Models strictly adhere to MIT/Apache-2.0 licenses under 8B parameters.
