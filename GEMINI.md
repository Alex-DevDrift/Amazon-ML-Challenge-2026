# Operational Rules & Directive

- **No superficial updates, boilerplate, or documentation churn**: Focus 100% on competitive ML modeling, data forensics, performance optimization, and score maximization.
- **Precision-First Mindset**: The evaluation metric is Macro $F_{0.5}$. False positives destroy the score ($F_{0.5}$ penalizes precision errors $2\times$ harder than recall errors). Over-predicting matches causes score collapses (e.g. 0.398).
- **Hard Technical Rigor**: Every change must be validated against the real validation ground truth with macro $F_{0.5}$ metrics before submitting.
- **Aggressive & Non-Conventional Exploitation**: Investigate synthetic data artifacts, leakage, exact token hashing, phone/PIN code graph components, transliteration phonetic decoders, embedding/minhash clusters, and calibrated probability cutoffs.
