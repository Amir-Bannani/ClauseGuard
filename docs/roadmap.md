# Roadmap

Planned milestones for ClauseGuard. All milestones are forward-looking; nothing below has been implemented.

---

## Milestone 1 — Dataset

**Goal:** Acquire, load, understand, and prepare the LEDGAR dataset.

- [X] Obtain the LEDGAR subset of LexGLUE
- [X] Load and inspect the data structure and label distribution
- [X] Analyze class imbalance; document the distribution
- [X] Decide whether to use all 100 LexGLUE classes, a subset, or a ClauseGuard-internal label taxonomy
- [X] Implement document-aware preprocessing and splitting
- [X] Verify no document-level leakage across train/dev/test splits
- [x] Document all preprocessing decisions

**Deliverable:** A reproducible data-preparation pipeline with verified split statistics and documented class distribution.

---

## Milestone 2 — Baseline Model

**Goal:** Train a cheap, interpretable classical baseline for clause classification.

- [X] Build TF-IDF feature pipeline
- [X] Train logistic regression classifier
- [X] Evaluate with the full metric set (macro-F1, weighted-F1, accuracy, per-class F1, confusion matrix, precision/recall)
- [X] Perform error analysis: which classes are hardest? What do misclassified provisions look like?
- [x] Record all results with no fabricated numbers
- [X] Establish the document-grouped evaluation as the standard

**Deliverable:** A trained baseline model and a complete, documented evaluation report.

---

## Milestone 3 — Transformer Model

**Goal:** Fine-tune a transformer-based classifier and compare it to the baseline.

- [ ] Select and justify the transformer candidate (DistilBERT vs. legal-domain model — chosen after experimentation)
- [ ] Implement tokenizer and fine-tuning pipeline
- [ ] Fine-tune on LEDGAR training split
- [ ] Evaluate with the same metric set as the baseline
- [ ] Compare directly against the baseline: same data, same splits, same metrics
- [ ] Error analysis: which classes improve, which still fail, and why?
- [ ] Make a defensible model-selection decision (justify the choice based on metrics + latency + size, not just F1)
- [ ] Make no premature claims about which model wins before the comparison runs

**Deliverable:** A selected trained model with a direct, documented comparison to the baseline.

---

## Milestone 4 — Model Serving

**Goal:** Turn the selected trained model into a deployable inference artifact.

- [ ] Export the trained model to ONNX
- [ ] Validate that exported model predictions match training-time predictions within an acceptable tolerance
- [ ] Optimize/quantize if justified by measured trade-offs (not blindly)
- [ ] Measure and document: inference latency, model size, memory usage
- [ ] Build a minimal ML inference service wrapping `model.onnx` (see [architecture.md](architecture.md))
- [ ] Containerize the inference service in a Docker image
- [ ] Document the model artifact lifecycle

**Deliverable:** A Docker image serving predictions from a validated `model.onnx`, with documented serving metrics.

---

## Milestone 5 — Application Layer

**Goal:** Build the surrounding application: document processing, backend, database, and frontend.

- [X] Implement PDF text extraction (PyMuPDF)
- [ ] Implement DOCX text extraction (python-docx)
- [ ] Build clause/section segmentation pipeline 
- [ ] Implement FastAPI backend: auth, document management, upload handling, orchestration, API
- [ ] Design and implement PostgreSQL schema (users, documents, clauses, analyses, model_versions)
- [ ] Record the model version used with every analysis
- [ ] Build Next.js frontend: upload, analysis dashboard, clause visualization, findings, history
- [ ] Connect all components end to end

**Deliverable:** An end-to-end working system from document upload to clause-level dashboard.

---

## Milestone 6 — Evaluation / Checklist Layer

**Goal:** Implement the rule-based clause evaluation layer.

- [ ] Define and document the initial checklist of clause checks
- [ ] For each check: specify the clause type, the feature extracted, the threshold, and the rationale
- [ ] Implement the rule-based evaluation layer against classified clauses
- [ ] Produce structured findings with clear "flagged for review" language
- [ ] Test with representative contracts
- [ ] Document the explicit boundary between what the model does (classification) and what the checklist does (evaluation)

**Deliverable:** A documented, testable, rule-based evaluation layer producing structured, explainable findings.

---

## Milestone 7 — Advanced Capabilities

**Goal:** Extend the system where concrete requirements justify it.

- [ ] Train a second ML model for concern/checklist classification (only with a defensible labeling methodology)
- [ ] Evidence / clause highlighting (possibly CUAD-informed)
- [ ] Optional: LLM-powered natural-language explanations of structured findings
- [ ] Optional: semantic search / RAG over extracted clauses ("Find every clause related to termination")
- [ ] Additional document formats and extraction pipelines if needed

**Deliverable:** Incrementally added capabilities that extend an already-functional base system.

---

## CI / CD

CI is the only part of the pipeline that runs automatically in GitHub. See
[docs/ci-cd.md](ci-cd.md) for the full design; the short version is that CI
jobs auto-activate as each component materializes, so the pipeline never
reports failures for components that do not yet exist.

Deployment concerns are documented here now so they are not forgotten, but
they will only be implemented once a hosting target exists and there is code
to deploy. CD is intentionally **not** invented ahead of time.

- [x] Initial GitHub Actions CI — structure, Python, frontend, Docker ([`.github/workflows/ci.yml`](../.github/workflows/ci.yml))
- [ ] Mark the first CI run green on GitHub (structure + any existing component jobs)
- [ ] Python package tooling (uv or pip + ruff + pytest) locked in when the first ML code lands
- [ ] Automated Docker image publishing (registry + credentials required first)
- [ ] Continuous deployment to a hosting target (see [docs/ci-cd.md](ci-cd.md) for the checklist)
- [ ] ML training workflow (`ml-training.yml`, manually triggered) — only once the training code exists

---

## Principles

- **Do the next thing.** Milestones are ordered; do not skip ahead to the exciting part.
- **Trustworthy evaluation > impressive metrics.** Document-level leakage control is non-negotiable.
- **No architecture theater.** Add infrastructure only when a concrete problem requires it.
- **Results are always measured, never claimed.** Every number in this repo is either real or explicitly `TBD`.
