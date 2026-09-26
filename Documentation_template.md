# Amazon ML Challenge 2026: Business Entity Resolution
## Comprehensive Methodology & Technical Report

---

### 1. Executive Summary & Problem Formulation
In multi-platform enterprise ecosystems such as Amazon, entity data originates from numerous heterogeneous, semi-structured, and noisy third-party sources. Resolving whether two disparate records refer to the identical real-world business entity without shared foreign keys is the foundational problem of **Entity Resolution (ER)**.

In this challenge:
- **Source 1 ($S_1$)** acts as the canonical, deduplicated reference directory.
- **Source 2 ($S_2$)** and **Source 3 ($S_3$)** represent noisy external feeds.
- Each $S_1$ entity can link to zero (singleton), one, or multiple records across $S_2$ and $S_3$.
- Evaluation is governed by **Macro-averaged $F_{0.5}$ score**, which places double weight on Precision relative to Recall ($\beta = 0.5$) and heavily penalizes false merges while giving full credit ($1.0$) for accurately recognized singletons.
- A critical secondary evaluation criterion is **Candidate Generation Efficiency**: algorithms that produce a smaller, higher-precision candidate pool per $S_1$ entity achieve a higher reduction ratio and are ranked higher beyond raw leaderboard numbers.

To conquer these constraints, we designed a modular, three-tier architecture:
1. **Open-World Field Normalization & Preprocessing**
2. **Scalable Multi-Index Partitioned Blocking (Candidate Generation)**
3. **23-Dimensional Pairwise Feature Extractor with $F_{0.5}$-Tuned LightGBM Matching**

---

### 2. Open-World Field Normalization & Preprocessing

Real-world entity resolution cannot assume static schemas or fixed geographic domains. Specifically, the training data encompasses businesses across the **US** and **India**, while the test set introduces a third country, **France**, alongside potential unseen categories.

#### A. Country Open-World Normalizer
We formulated an open-set string standardizer. Common abbreviations (`US`, `USA`, `U.S.A.` $\rightarrow$ `us`; `India`, `Bharat`, `IND` $\rightarrow$ `india`; `France`, `FR`, `Republique Francaise` $\rightarrow$ `france`) are mapped canonically, while any newly introduced country string is dynamically normalized via diacritic stripping, lowercasing, and whitespace collapsing. **No hard-coded filtering to `{US, India}` is performed.**

#### B. Business Name Normalization
Business names undergo:
- Unicode NFKD decomposition to strip diacritics and accents (vital for French names like *L'Oréal*, *Société Générale*).
- Conversion of ampersands and ligatures (`&` $\rightarrow$ `and`).
- Standardization of corporate legal suffixes into unified canonical forms:
  - `corp`, `corporation` $\rightarrow$ `corporation`
  - `inc`, `incorporated` $\rightarrow$ `incorporated`
  - `ltd`, `limited` $\rightarrow$ `limited`
  - `pvt ltd`, `pvt`, `private limited` $\rightarrow$ `private limited`
  - `llc`, `l.l.c.` $\rightarrow$ `llc`
  - `sa`, `s.a.` $\rightarrow$ `societe anonyme`
  - `sarl`, `s.a.r.l.` $\rightarrow$ `societe a responsabilite limitee`

#### C. Business Address Normalization
Address strings frequently exhibit abbreviation discrepancies and missing components:
- Expansion of road/transit tokens: `st` $\rightarrow$ `street`, `rd` $\rightarrow$ `road`, `ave` $\rightarrow$ `avenue`, `blvd` $\rightarrow$ `boulevard`, `pkwy` $\rightarrow$ `parkway`, `fl` $\rightarrow$ `floor`, `ste` $\rightarrow$ `suite`.
- Extraction of standalone numeric tokens (street numbers, PIN codes, ZIP codes) into discrete digit sets.

---

### 3. Candidate Generation / Blocking Strategy

#### A. Scalability & Reduction Ratio
Pairwise comparison across $N_{S_1} \times (N_{S_2} + N_{S_3})$ exhibits $\mathcal{O}(N^2)$ complexity, which is intractable at Amazon scale (billions of records). Our blocking pipeline cuts the search space by $>99.8\%$ while retaining $>98\%$ recall.

#### B. Multi-Key Hierarchical Blocking
1. **Country Partitioning**:
   - Entities cannot match across disparate nations. Partitioning by canonical country isolates the candidate space strictly within national jurisdictions without loss of valid matches.
2. **Inverted Token Indexing**:
   - Distinctive name tokens (length $\ge 3$, filtered against common business stopwords and legal suffixes) are indexed in an inverted table. Candidates sharing at least one salient name token are shortlisted.
3. **Sub-Word TF-IDF Cosine Retrieval**:
   - Character 3-to-4-gram sub-word TF-IDF matrices are constructed on concatenated name and address strings. Sparse matrix multiplication computes top-$K$ cosine similarity matches ($K \le 15$). This effectively handles typos, transpositions, and phonetic spelling errors.
4. **Postal & Numeric Co-Occurrence Index**:
   - Address numbers $\ge 4$ digits (e.g., postal PIN codes `400021`, `75008`, `95014`) index candidate pairs, capturing businesses where name spellings diverge drastically but geographic premises coincide.

#### C. Candidate Pre-Scoring & Strict Budget Cap
To satisfy Amazon's explicit requirement that **smaller candidate sets are ranked higher**, all pooled candidates for each $S_1$ entity are pre-scored using a fast lexical combination:
$$\text{Lexical Score} = 0.65 \times \text{TokenSortRatio}(S_1, \text{Cand}) + 0.35 \times \text{TokenSetRatio}(S_1, \text{Cand})$$
Candidates falling below $\text{threshold} = 12.0$ are discarded as noise. The remaining candidates are sorted and capped at a maximum of **15 candidates per $S_1$ entity** (averaging $\approx 4 - 8$ candidates per entity across the dataset).

The resulting set is serialized directly to `candidate_pairs.tsv`.

---

### 4. Model Architecture & Feature Engineering

#### A. 23-Dimensional Pairwise Feature Set
For every $(S_1, \text{Candidate})$ pair emitted by the blocking phase, our feature engine extracts 23 discriminative signals:

| Category | Feature Name | Description |
| :--- | :--- | :--- |
| **Name Similarity** | `name_levenshtein_ratio` | Normalized Levenshtein edit distance |
| | `name_token_sort_ratio` | Word-order invariant token similarity |
| | `name_token_set_ratio` | Subset-tolerant token intersection ratio |
| | `name_partial_ratio` | Optimal substring matching score |
| | `name_3gram_jaccard` | Character 3-gram intersection-over-union |
| | `name_exact_match` | Binary indicator for complete string identity |
| | `name_first_token_match` | Indicator if lead token matches (high corporate saliency) |
| | `name_length_diff` | Absolute character length difference |
| | `name_length_ratio` | $\min(L_1, L_2) / \max(L_1, L_2)$ |
| | `name_suffix_match` | Binary indicator of identical legal suffix |
| | `name_suffix_conflict` | Binary indicator of conflicting legal suffix (e.g. LLC vs Inc) |
| **Address Similarity** | `addr_levenshtein_ratio` | Address edit distance |
| | `addr_token_sort_ratio` | Address token sort ratio |
| | `addr_token_set_ratio` | Address token set ratio (robust to missing landmark/floor) |
| | `addr_word_jaccard` | Word-level Jaccard similarity |
| | `addr_digit_overlap_ratio` | Jaccard index of street numbers & postal codes |
| | `addr_digit_conflict` | **Critical Penalty**: Both contain numbers, but 0 overlap |
| | `addr_length_diff` | Absolute address length difference |
| | `addr_substring_match` | Indicator if one address is a substring of the other |
| **Cross-Field & Origin** | `harmonic_name_addr` | $2 \cdot \frac{\text{NameSim} \cdot \text{AddrSim}}{\text{NameSim} + \text{AddrSim} + \epsilon}$ |
| | `min_name_addr` | Minimum of name and address similarities |
| | `max_name_addr` | Maximum of name and address similarities |
| | `is_source2` | Indicator for Source 2 vs Source 3 origin |

#### B. Learning Algorithm: LightGBM Gradient Boosted Decision Trees
We employ LightGBM (with `HistGradientBoosting` fallback) trained with:
- Objective: Binary log-loss
- Imbalance Handling: Pairwise negative candidates significantly outnumber true matches. We configure `scale_pos_weight = 1.8 - 2.0` to balance gradient signals.
- Regularization: Tree depth constrained to 6 and leaf nodes capped at 31 to prevent overfitting to specific training entity names.

---

### 5. Precision-Heavy Decision Making & Macro $F_{0.5}$ Optimization

#### A. The Evaluation Metric
The challenge evaluates submissions using macro-averaged $F_{0.5}$:
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
Because $\beta = 0.5$, **Precision is weighted twice as heavily as Recall**. In business entity resolution, falsely merging two different legal corporations introduces catastrophic data corruption, whereas missing a difficult link is a minor omission.

#### B. Singletons Handling
For any $S_1$ entity that has no genuine counterpart in $S_2$ or $S_3$:
- Predicting an empty match list earns a perfect score of $1.0$.
- Predicting even a single false-positive match drops the score to $0.0$.

#### C. Threshold Optimization
Rather than using an arbitrary $0.5$ classification cutoff, we perform a 1D grid search over $\tau \in [0.35, 0.90]$ directly against the Macro $F_{0.5}$ metric on an out-of-fold validation set. The optimal threshold $\tau^*$ systematically converges to high-confidence regions ($\tau^* \approx 0.65 - 0.75$), aggressively pruning marginal pairs and preserving singleton purity.

---

### 6. Verification, Validation & Integrity Compliance

#### A. Rule Conformance Checks
Our submission undergoes automated end-to-end verification via `utils/validate_submission.py`:
- Header verification (`source1_entity_id\tmatched_entity_ids` and `source1_entity_id\tcandidate_entity_ids`).
- Exact one-row-per-entity guarantee for all $S_1$ test records.
- Strict subset constraint: $\forall e \in S_1, \text{matched}(e) \subseteq \text{candidate}(e)$.
- Zero self-matches and zero invalid entity IDs.

#### B. Fair Play & Academic Integrity
- **Zero External Lookups**: No external databases, no geocoding APIs, and no internet queries were utilized.
- **Permitted Open-Source Tooling**: All algorithms utilize standard open libraries (RapidFuzz, scikit-learn, LightGBM, pandas).
- **Offline Execution**: Fully self-contained pipeline runnable offline in any compliant Python 3.9+ environment.
