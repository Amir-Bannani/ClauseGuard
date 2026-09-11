# Dataset

This document describes the dataset used for ClauseGuard's first ML task: clause classification. The primary dataset is **LEDGAR**, accessed via the **LexGLUE** benchmark.

## Terminology: LexGLUE vs. LEDGAR

These are often conflated. The relationship:

- **LexGLUE** is a *benchmark* for evaluating legal-language models in English. It packages several existing legal NLP datasets under a common evaluation protocol (Chalkidis et al., ACL 2022).
- **LEDGAR** is the *dataset/task* within LexGLUE that ClauseGuard cares about for V1: contract provision (paragraph) classification. LEDGAR is used in LexGLUE in a simplified, single-label, 100-class form.

Two other datasets are **not** required for V1:

- **ContractNLI** — excluded for V1.
- **CUAD** — useful later for span/evidence extraction, but not the primary V1 dataset.

## What LEDGAR Is

LEDGAR (Labeled EDGAR, Tuggener et al., LREC 2020) is a corpus of labeled legal provisions in contracts. The provisions were scraped from contracts in the U.S. Securities and Exchange Commission (SEC) filings, published through the EDGAR system, and made freely available.

Key facts (from the original LEDGAR paper):

- The corpus is **semi-automatically** constructed from SEC filings.
- After cleaning, the released corpus contains roughly **846k contract provisions** with a label set of about **12.6k categories**.
- The paper reports a highly skewed/biased label distribution and an average of ~124 tokens per provision (std. dev. ~104), and ~13 provisions per contract on average (std. dev. ~20).
- Each provision's labels were assigned by the **contract drafters** themselves when creating the documents, rather than by a dedicated annotation team used for these labels — labels were extracted from the source documents.

> **Important:** LEDGAR consists of labeled *contract provisions* (paragraphs), **not complete contracts**. Do not describe it as a corpus of whole contracts.

## LEDGAR in LexGLUE

For ClauseGuard V1, we use the LEDGAR task as packaged by LexGLUE. Verified facts about the LexGLUE LEDGAR setup (from the LexGLUE paper, Table 1 and dataset description):

- A subset of **80k contract provisions** with the **100 most frequent labels**.
- Formulated as **single-label, multi-class** classification: each provision maps to its single main topic/theme.
- **Chronological split** into training (60k, 2016–2017), development (10k, 2018), and test (10k, 2019).

This chronological split matters for evaluation (see Leakage below).

## What a Training Example Looks Like

LEDGAR is released as JSONL; each line is a JSON object with (in the original release) attributes:

```json
{
  "provision": "The employee agrees that for a period of five years following termination, they shall not provide services to any competing business.",
  "label": ["Non-competition"],
  "source": "relative/path/to/source/contract"
}
```

For ClauseGuard's experiments the important fields are:

- the **provision text** (the input),
- the **label** (the target class),
- the **source** (the originating contract — used for document-level grouping and leakage control).

## What Labels Represent

Each label is a **topic/theme** that the provision addresses, expressed as a legal-provision category name (e.g., "Non-competition", "Confidentiality", "Termination"). The label set is large and hierarchical in the original corpus; the LexGLUE formulation restricts it to the 100 most frequent categories as a simplification.

A likely open question for V1 (recorded here, not yet decided): whether to map LEDGAR labels to a smaller ClauseGuard-internal clause taxonomy (e.g., `non_compete`). If we do, the mapping must be explicit and documented, and evaluation must be clear about whether it uses LEDGAR labels or the mapped taxonomy.

## Dataset Limitations

- **Not complete contracts.** LEDGAR gives labeled provisions, so the dataset does not directly evaluate end-to-end contract processing (segmentation of a raw contract into provisions is a separate engineering problem handled by our document pipeline).
- **Domain skew.** Provisions come from SEC-filed contracts; the label vocabulary reflects that domain. Generalizing to contracts outside the SEC-filing sphere is an open question (the original paper explored transfer to externally annotated NDAs).
- **Noisy labels.** The corpus is semi-automatically constructed; labels were extracted from documents rather than curated by an independent annotation team for research purposes.
- **Extreme class imbalance.** Both the original corpus and the 100-class LexGLUE subset have a highly skewed label distribution. This is why **macro-F1 is the primary metric** — it does not let the frequent classes dominate the evaluation summary.

## Leakage Considerations

Multiple provisions frequently originate from the same contract. This creates a leakage risk: if provisions from one contract are split across train and test, the model can effectively "memorize" contract-level patterns and evaluation becomes artificially optimistic.

ClauseGuard treats this seriously:

1. **Document-level grouping.** The `source` field identifies the originating contract. Provisions from the same source must stay within the same split.
2. **Chronological split.** The LexGLUE LEDGAR split is chronological (train 2016–2017, dev 2018, test 2019). This reduces temporal leakage and is closer to how a deployed system behaves (trained on the past, evaluated forward).
3. **Verification.** After preprocessing, we will verify that no source contract appears in both training and validation/test sets, and we will report on this explicitly in the experiment write-ups.
4. **Skeptical reading.** If a per-provision random split ever looks better than the document-grouped split, that gap is treated as evidence of leakage risk — not as a better result.

We prioritize trustworthy evaluation over an impressive-looking metric.

## Why LEDGAR Is Right for V1

- It is a **real, documented, freely available** legal-NLP dataset with a well-defined supervised classification task.
- It matches ClauseGuard's core need: **classifying individual contract clauses**.
- Its 100-class single-label formulation via LexGLUE is tractable for both a classical baseline (TF-IDF + logistic regression) and a fine-tuned transformer, and its imbalance makes the evaluation meaningful (macro-F1 matters).
- The task is clearly separable from the checklist evaluation layer — exactly the product design ClauseGuard implements.

## Selected Statistics (Verified)

Only statistics verified against the original papers are listed here. Anything else is `TBD` after we actually load and inspect the dataset.

| Quantity | Value | Source |
| --- | --- | --- |
| LEDGAR cleaned corpus — provisions | ~846k | Tuggener et al., LREC 2020 |
| LEDGAR cleaned corpus — labels | ~12.6k categories | Tuggener et al., LREC 2020 |
| LEDGAR — avg. tokens per provision | ~124 (std. dev. ~104) | Tuggener et al., LREC 2020 |
| LEDGAR — avg. provisions per contract | ~13 (std. dev. ~20) | Tuggener et al., LREC 2020 |
| LexGLUE LEDGAR subset — provisions | 80k | Chalkidis et al., ACL 2022 |
| LexGLUE LEDGAR subset — labels | 100 most frequent | Chalkidis et al., ACL 2022 |
| LexGLUE LEDGAR subset — train / dev / test | 60k / 10k / 10k (chronological) | Chalkidis et al., ACL 2022 |

Exact split sizes, per-class distributions, and preprocessing statistics for the specific files we download are `TBD` and will be recorded after the dataset is loaded and inspected.

## References

- Don Tuggener, Pius von Däniken, Thomas Peetz, Mark Cieliebak. *LEDGAR: A Large-Scale Multi-label Corpus for Text Classification of Legal Provisions in Contracts.* LREC 2020.
- Ilias Chalkidis et al. *LexGLUE: A Benchmark Dataset for Legal Language Understanding in English.* ACL 2022.

## Next Step

Load and inspect the dataset (Milestone 1 in [roadmap.md](roadmap.md)) before writing any classifier.