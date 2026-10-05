# Concern detection and review-criteria retrieval

ClauseGuard separates extraction from policy evaluation:

```text
classified clause
→ existing category-specific extractor
→ normalized extracted facts
→ retrieve applicable configured review criteria
→ deterministic rule evaluation
→ structured findings
```

The classifier currently returns a mapping with `category`, `label_id`, `confidence`, and `probabilities`. `NuExtractExtractor.extract(text, category)` returns a JSON string matching the category schema in `legal-ai/extraction/llm_extractor.py`. The detector accepts that existing string or a parsed mapping and uses the existing duration normalizer. The classifier and extractor remain unchanged.

## Responsibilities and limits

Extraction answers **what the clause says**. Concern detection answers **whether extracted facts match a configured review criterion**. The rule evaluator makes the final decision deterministically from the extracted value and condition. Retrieval supplies relevant rule context and explanations; it does not make the concern decision. There is no LLM judge, opaque risk score, external API, or vector database in this foundation.

The initial retrieval implementation is local and replaceable through the `RuleRetriever` protocol. It filters by clause type and optional metadata, then orders applicable documents by simple token overlap. `retrieve_for_facts` builds its query only from the clause category and extracted fields; it does not send an entire contract to retrieval. This is a transparent local retrieval baseline, not an embedding-based semantic search system.

## Rules and configuration

`legal-ai/rules/engine.py` defines `ConcernRule`. A rule includes a stable ID, clause type, field, operator, expected value, optional duration unit, finding type, severity, description, source, and optional exception field. Supported operations are equality/inequality, numeric comparisons, membership/containment, boolean checks, and non-empty checks. Rules are typed Python data; arbitrary expressions are never evaluated.

The checked-in `legal-ai/rules/review_rules.json` entries are **demonstration configuration**, not legal thresholds or statements of law. Their IDs begin with `demo.` and their source is `demo_configuration`. Replace or supplement them with a team's documented review policy before relying on the results. Duration comparisons normalize to days using 30 days per month and 365 days per year; these are arithmetic comparison conventions, not legal definitions.

The same records are the small rule knowledge base. Retrieval documents include the rule ID, clause type, relevant field, condition, criterion description, rationale, severity, and source. This keeps retrieved rationale tied to the exact criterion evaluated.

To add a criterion:

1. Add a JSON record to `legal-ai/rules/review_rules.json` with a unique `rule_id`, extracted field, supported operator/value, finding metadata, rationale, and source.
2. Use a clearly identified demo source for examples. A production policy should identify its actual owner/source and should not imply legal authority without appropriate review.
3. Add a schema/evaluator test in `tests/test_concern_rules.py`, retrieval coverage in `tests/test_rule_retrieval.py`, and an end-to-end case in `tests/test_concern_detection.py` as applicable.

## Findings and exceptions

Each finding contains `rule_id`, `clause_type`, `finding_type`, `severity`, `field`, `extracted_value`, `rule_condition`, `explanation`, `evidence`, and `source`. If the extracted value appears in the original clause text, the detector returns that exact substring as evidence. Otherwise it falls back to the extractor's raw value. The explanation says the value matches a configured review criterion and labels the result a potential concern according to the configured review policy.

When a rule has an `exception_field` and that extracted field is populated, the current evaluator suppresses the finding. This is deliberately simple and conservative. Existing category schemas extract exception lists but do not classify which exception applies to which condition; production rules may need a more precise, policy-specific exception representation to avoid suppressing a criterion for an unrelated exception.

## Evaluation and tests

Run the focused tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_concern_rules.py tests/test_rule_retrieval.py tests/test_concern_detection.py tests/test_concern_evaluation.py -q
```

Run the labeled synthetic evaluation with:

```powershell
.\.venv\Scripts\python.exe legal-ai\evaluation\run_concern_demo.py
```

`legal-ai/evaluation/retrieval.py` provides `recall_at_k` over labeled retrieval cases and `binary_rule_metrics` (TP, FP, TN, FN, accuracy) over known extracted facts. The demo currently reports Recall@1 1.0 and rule counts TP=3, FP=0, TN=3, FN=0 across six supplied-fact examples. These are mechanics checks on hand-written synthetic cases only. Rule metrics assume extracted facts are correct and do not measure extraction quality. No representative contract evaluation or real policy quality result is claimed.

## Limitations

- Demo thresholds are illustrative configuration and are not authoritative legal standards.
- Rule evaluation is deterministic but only as accurate as the extracted fields and configured criteria.
- An extracted exception suppresses a matching rule whenever the configured exception field is non-empty; the extractor does not establish that it is relevant to that rule.
- Retrieval uses lexical token overlap and checked-in criteria only. No legal corpus or external source is retrieved.
- Missing or malformed fields do not trigger numeric criteria. They should be addressed through extraction-quality review rather than guessed by the rule engine.
- Findings are review prompts, not legal advice or conclusions about legality.
