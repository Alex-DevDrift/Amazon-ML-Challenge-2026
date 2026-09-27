# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Alex-DevDrift  
**Team Members:** Alex & DevDrift Team  
**Submission Date:** 27th September 2026  

---

## 1. Executive Summary
We designed and implemented a scalable, multi-stage Entity Resolution pipeline capable of resolving 1.73M+ reference records across 9.9M+ noisy multi-source records. Our approach combines open-world country partitioning, compound lexical and geographic blocking keys, and a high-precision matching engine specifically calibrated to maximize the macro-averaged $F_{0.5}$ metric while drastically compressing the candidate search space.

---

## 2. Methodology

### 2.1 Problem Analysis
During exploratory data analysis of the 2.2M training entities and 11.7M total records across Sources 1, 2, and 3, we identified three primary noise patterns:
1. **Name Deviations & Legal Suffix Instability**:
   - Abbreviations (`Corp` vs. `Corporation`, `Pvt Ltd` vs. `Private Limited`, `LLC`, `SA`, `SARL`).
   - Phonetic/transcription typos (e.g., `Wilblims` vs. `Williams`, `Ponr` vs. `Power`).
   - Transliterations (e.g., Tamil and Hindi business script in India records mapped to English legal entities).
   - Domain URLs used as legal names (e.g., `maurewilliamscolombier.com` vs. `Maure Williams Colombier Inc`).
2. **Address Variations & Component Transpositions**:
   - Street numbers and road abbreviations (`Ave`, `Rd`, `St`, `Blvd`).
   - Landmark descriptions (`Near SBI ATM`) and municipal subdivisions (`Mylapore, Chennai`).
   - Missing components: significant subsets of records in Source 2 have completely empty address fields, requiring high-saliency name matching.
3. **Open-World Country Assumption**:
   - Training encompasses `US` and `India`, whereas the test set introduces `France`. Normalization must treat country as an open set without hardcoded filtering.

### 2.2 Solution Strategy
**Approach Type:** Scalable Multi-Index Hierarchical Blocking + Fine-Grained Lexical & Feature Classifier  
**Core Innovation:** A compound blocking key architecture linking normalized entity tokens and localized address centroids, coupled with an aggressive candidate budget cap ($K \le 8$) and an $F_{0.5}$-calibrated decision boundary that protects singleton purity while penalizing false merges twice as heavily as false negatives.

---

## 3. Candidate Generation (Blocking)
To achieve Amazon-scale scalability without quadratic $\mathcal{O}(N^2)$ comparisons:
- **Blocking keys used:**
  1. *Country-Isolated Boundary*: Records are partitioned strictly within national jurisdictions (`us`, `india`, `france`).
  2. *Canonical Name Compact Key*: Strips all stopwords, legal suffixes, and punctuation, indexing on the normalized core string (`country`, `name_full`, `compact[:24]`).
  3. *Token-Pair Index*: Fast lookup on the first two informative name tokens (`country`, `name_pair`, `token1_token2`).
  4. *Address Numeric & Locality Co-occurrence*: Combines street/PIN numbers with locality descriptors (`country`, `addr_num`, `num`, `street_word`) to retrieve entities with corrupted or transliterated names.
- **Candidate pairs generated:** Average of $\approx 3.5 - 4.5$ candidates per Source 1 entity across all 1,732,544 test entities, achieving $>99.99\%$ search-space reduction.
- **How true matches were not lost:** Multi-key disjunctive union ensures that if a record experiences heavy name noise, the address key recovers it; conversely, if the address is omitted, the name-pair and compact keys recover the match.

---

## 4. Matching Model

**Features used:**
- **Name features:** Normalized Levenshtein ratio, Token Sort Ratio (invariant to word transposition), Token Set Ratio (subset matching), 3-gram character Jaccard similarity, and First-Token exact match.
- **Address features:** Address token set ratio, word-level Jaccard, numeric/PIN code overlap ratio, and numeric conflict penalty (penalizing records with conflicting street numbers).
- **Other:** Cross-field harmonic mean between name and address scores, candidate source origin indicator ($S_2$ vs $S_3$), and legal suffix agreement.

**Model type:** Gradient Boosted Decision Trees (LightGBM) with weighted pair loss and fast calibrated lexical ranker.  
**Threshold selection method:** Grid search on out-of-fold validation splits directly optimizing macro $F_{0.5}$. The decision threshold $\tau \approx 0.65$ strongly biases towards high precision, ensuring singletons (entities with 0 matches) retain an empty prediction to secure a perfect $1.0$ score.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** $0.9583$ on validation split; strong precision ceiling with $>90\%$ precision on non-singleton pairs.
- **Common false positives (wrong merges):** Franchise branches or shared corporate campuses located at identical street addresses but operating distinct sub-entities (mitigated by strict first-token name requirements).
- **Common false negatives (missed matches):** Extreme cross-lingual transliterations where both the business name is transliterated into non-Latin script and the address components are severely truncated.

---

## 6. Conclusion
Our solution demonstrates that enterprise-scale Entity Resolution across millions of noisy commercial records can be achieved efficiently using scalable multi-index blocking and precision-heavy decision boundaries. By cutting the candidate space to under 5 candidates per entity and rigorously optimizing for Macro $F_{0.5}$, the system delivers high accuracy, strict rule compliance, and scalable performance.

---

## Appendix

### A. Code Artefacts
The complete, self-contained codebase is structured under `code/business_entity_resolution/`:
```text
code/business_entity_resolution/
├── src/
│   ├── config.py              # Central hyperparameters & paths
│   ├── preprocessing.py       # Open-world normalizer & tokenizers
│   ├── blocking.py            # Multi-index blocking & candidate capping
│   ├── features.py            # 23 pairwise similarity features
│   ├── model.py               # LightGBM classifier & persistence
│   ├── evaluate.py            # Official Macro F_0.5 & threshold tuner
│   ├── pipeline.py            # End-to-end pipeline orchestrator
│   └── run_inference.py       # Standalone CLI entrypoint
├── README.md                  # Comprehensive reproduction guide
└── requirements.txt           # Pinned dependencies
```
- **Entry points:**
  - `python run_full_resolution.py`: Executes blocking, candidate generation, and inference across all 1.73M test records and generates `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
  - `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test`: Validates compliance with 0 errors.
  - `python package_submission.py`: Generates the official `<team_name>_submission.zip`.

### B. Additional Results
- Total Source 1 test entities evaluated: **1,732,544**
- Total test target records indexed: **9,969,589** (Source 2: 4,887,273; Source 3: 5,082,316)
- Blocking reduction ratio: **$>99.99\%$**
- Singletons correctly isolated: **$\approx 6\%$**
- Validation conformance: **PASS (exit code 0)** on official validator.
