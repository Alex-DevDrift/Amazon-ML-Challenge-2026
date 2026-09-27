# AGENTS.md — Amazon ML Challenge 2026: Two-Agent Operating Contract

This repo uses **two independent agents** working on **separate approaches**.
They must not converge on the same model. Divergence is intentional.

## Agent Roster

| Agent | Folder | Output files | Approach (locked) |
|-------|--------|--------------|-------------------|
| **Gemini** | `code/business_entity_resolution/` + root scripts (`train_lgbm_val.py`, `optimize_matcher.py`, `run_full_resolution.py`) | `output/matching_results.tsv`, `output/candidate_pairs.tsv` | Supervised **LightGBM 5-fold CV** over pairwise similarity/conflict features (15-dim), OOF threshold sweep 0.40–0.90, unique-target assignment, cap 6. |
| **Muse** | `muse/` | `output/matching_results_muse.tsv`, `output/candidate_pairs_muse.tsv` | **Unsupervised deterministic multi-channel retrieval + composite scorer + collective graph assignment. NO LightGBM, NO sklearn classifier, NO KFold training, NO learned pairwise model.** See `muse/AGENTS.md`. |

## Global Rules (both agents)

1. **Metric is Macro F0.5 (beta=0.5).** Precision errors cost 2× recall errors.
   Singleton S1 (0 true matches) scores 1.0 iff predicted empty, else 0.0.
2. **Never break the validator.** Headers must be exactly
   `source1_entity_id\tmatched_entity_ids` and `source1_entity_id\tcandidate_entity_ids`.
   Every test S1 needs exactly one row. Only `S2-`/`S3-` IDs in lists, no intra-list dupes.
   Matches must be a subset of candidates (warns, not fails — but keep it clean).
3. **Country is open-world.** Train has US+India, test adds France. Never hard-filter
   to a closed country list; normalize (`us/india/france/...`) and always partition, never drop.
4. **No external data / APIs / pretrained embeddings >8B.** Stdlib + allowlisted OSS only
   (`rapidfuzz`, `scikit-learn` for TF-IDF/vectorizer only in Muse blocker, `pandas`, `numpy`, `anyascii`).
5. **Validate before claiming.** Every threshold / key / heuristic change must be measured
   as Macro F0.5 on a train-ground-truth split before touching test outputs.
6. **Do not overwrite the other agent's files.** Gemini never writes `*_muse.tsv`;
   Muse never writes `output/matching_results.tsv` / `output/candidate_pairs.tsv`.

## Muse Non-Compete (enforced)

Muse MUST NOT:
- train `LGBMClassifier` / `HistGradientBoostingClassifier` / any supervised pairwise classifier;
- reuse `train_lgbm_val.py` / `optimize_matcher.py` feature vectors or thresholds;
- copy Gemini blocking keys verbatim without adding at least the address-only,
  skeleton, trigram, and sibling-expansion channels described in `muse/AGENTS.md`.

Muse MUST:
- keep all code under `muse/`;
- document every experiment in `muse/EXPERIMENTS.md` with validation P/R/F0.5;
- produce `output/matching_results_muse.tsv` + `output/candidate_pairs_muse.tsv`
  via `python muse/run_muse.py` (streaming, <10 GB peak).

## Gemini Non-Compete (enforced)

Gemini MUST NOT copy Muse's composite scorer weights, veto rules, or sibling-expansion
logic back into the LightGBM pipeline without a separate validation entry.
Shared utilities allowed: `code/business_entity_resolution/src/evaluate.py` (metric only).

## Definition of Done (Muse)

1. `muse/` pipeline runs end-to-end on train-sample validation with reported Macro F0.5.
2. `output/matching_results_muse.tsv` + `output/candidate_pairs_muse.tsv` exist,
   pass `utils/validate_submission.py --test-dir dataset/test`, and differ from Gemini's files.
3. `muse/EXPERIMENTS.md` logs validation scores and the web-research basis
   (Sparkly TF-IDF blocker, sibling expansion, collective resolution, expected-F0.5 cardinality).
