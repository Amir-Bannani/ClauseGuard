"""
ClauseGuard — High-Throughput Batched & Rule-Aware Contract Audit Pipeline.
Demonstrates:
1. Full-document LegalBERT batch classification (~0.08s for 38 clauses)
2. Rule-store-aware culling (skipping categories with 0 rules e.g. exclusivity)
3. Batched NuExtract generation (batch_size=2)
4. Post-run torch.cuda.empty_cache() memory reclamation
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
from importlib import import_module

import torch
import pandas as pd

# 1. Setup paths & imports
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pdfParser.src.pipeline import ContractParserPipeline

classifier_mod = import_module("legal-ai.classification.legalbert_classifier")
extractor_mod = import_module("legal-ai.extraction.llm_extractor")
detector_mod = import_module("legal-ai.rules.detector")
engine_mod = import_module("legal-ai.rules.engine")

LegalBertClassifier = classifier_mod.LegalBertClassifier
NuExtractExtractor = extractor_mod.NuExtractExtractor
ConcernDetector = detector_mod.ConcernDetector
load_rules = engine_mod.load_rules
format_nuextract_prompt = extractor_mod.format_nuextract_prompt
EXTRACTION_SCHEMAS_WITH_DESCRIPTIONS = extractor_mod.EXTRACTION_SCHEMAS_WITH_DESCRIPTIONS

# 2. Initialize components
print("[*] Initializing ClauseGuard pipeline components...")
pdf_pipeline = ContractParserPipeline()
classifier = LegalBertClassifier(
    model_name_or_path="ml/artifacts/legalbert_classifier/checkpoint-775",
    use_fallback=False
)
extractor = NuExtractExtractor(use_fallback=False)  # set True for fast heuristic benchmark
detector = ConcernDetector()


def batch_extract_nuextract(extractor, items: List[Tuple[str, str]], batch_size: int = 2) -> List[Dict[str, Any]]:
    """Batched entity extraction using NuExtract 2.0 with HuggingFace left-padding."""
    if not items:
        return []

    if not extractor.is_loaded:
        extractor.load_model()

    if extractor.use_fallback or extractor.model is None:
        results = []
        for text, cat in items:
            raw = extractor.extract(text, cat)
            try:
                results.append(json.loads(raw) if isinstance(raw, str) else raw)
            except Exception as e:
                results.append({"error": str(e), "raw": raw})
        return results

    extracted_facts = []

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        prompts = [
            format_nuextract_prompt(
                text=text,
                schema=EXTRACTION_SCHEMAS_WITH_DESCRIPTIONS.get(cat, EXTRACTION_SCHEMAS_WITH_DESCRIPTIONS["other_general"]),
                category=cat
            )
            for text, cat in batch
        ]

        inputs = extractor.processor(text=prompts, padding=True, return_tensors="pt").to(extractor.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = extractor.model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False
            )

        decoded_strings = extractor.processor.batch_decode(outputs[:, input_len:], skip_special_tokens=True)

        for raw_str in decoded_strings:
            try:
                cleaned = raw_str.strip()
                if cleaned.startswith("```json"): cleaned = cleaned[7:]
                elif cleaned.startswith("```"): cleaned = cleaned[3:]
                if cleaned.endswith("```"): cleaned = cleaned[:-3]
                extracted_facts.append(json.loads(cleaned.strip()))
            except Exception as e:
                extracted_facts.append({"error": str(e), "raw": raw_str})

    return extracted_facts


def benchmark_contract_batched(pdf_path: str, batch_size: int = 2) -> Dict[str, Any]:
    """Executes the complete pipeline using LegalBERT batching and NuExtract batching with automatic cache cleanup."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    t_start = time.time()

    # 1. PDF Parsing
    t0 = time.time()
    parsed_doc = pdf_pipeline.parse_file(pdf_path)
    t_parse = time.time() - t0

    clauses = parsed_doc["clauses"]
    all_texts = [c["text"] for c in clauses]

    # 2. Batch LegalBERT Classification (All clauses at once)
    t1 = time.time()
    predictions = classifier.batch_predict(all_texts)
    t_classify = time.time() - t1

    # Active review rule categories from the rule store
    active_categories = {r.clause_type for r in load_rules()}

    actionable_items = []
    actionable_indices = []

    for idx, (clause, pred) in enumerate(zip(clauses, predictions)):
        cat = pred["category"]
        if cat in active_categories and cat != "other_general":
            actionable_items.append((clause["text"], cat))
            actionable_indices.append(idx)

    # 3. Batched NuExtract Generation
    t2 = time.time()
    batch_facts = batch_extract_nuextract(extractor, actionable_items, batch_size=batch_size)
    t_extract = time.time() - t2

    facts_by_index = {idx: facts for idx, facts in zip(actionable_indices, batch_facts)}

    # 4. Deterministic Rule Review
    t3 = time.time()
    records = []
    for idx, (c, pred) in enumerate(zip(clauses, predictions)):
        cat = pred["category"]
        conf = pred.get("confidence", 0.0)
        text = c["text"]

        extracted = facts_by_index.get(idx, None)
        is_skipped = (extracted is None)
        findings = []

        if not is_skipped:
            det_res = detector.detect(clause_type=cat, extraction_output=extracted, clause_text=text)
            findings = [f.to_dict() for f in det_res.findings]

        max_severity = "CLEAN"
        if findings:
            sevs = [f["severity"].upper() for f in findings]
            if "HIGH" in sevs: max_severity = "HIGH"
            elif "MEDIUM" in sevs: max_severity = "MEDIUM"
            elif "LOW" in sevs: max_severity = "LOW"
            else: max_severity = "INFO"

        records.append({
            "clause_id": c["clause_id"],
            "section_title": c.get("section_title", "General"),
            "page": c.get("page_start", 1),
            "category": cat,
            "confidence": conf,
            "skipped_extraction": is_skipped,
            "extracted": extracted,
            "findings": findings,
            "risk_level": max_severity,
            "text": text
        })
    t_rules = time.time() - t3

    # 5. Post-run VRAM cleanup
    if torch.cuda.is_available():
        alloc_before = torch.cuda.memory_allocated() / (1024 ** 3)
        res_before = torch.cuda.memory_reserved() / (1024 ** 3)
        torch.cuda.empty_cache()
        alloc_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        res_gb = torch.cuda.memory_reserved() / (1024 ** 3)
    else:
        alloc_before, res_before = 0.0, 0.0
        alloc_gb, res_gb = 0.0, 0.0

    t_total = time.time() - t_start

    return {
        "document": parsed_doc.get("document", Path(pdf_path).name),
        "total_clauses": len(clauses),
        "actionable_clauses_extracted": len(actionable_items),
        "timings": {
            "pdf_parse_sec": round(t_parse, 3),
            "classification_sec": round(t_classify, 3),
            "extraction_sec": round(t_extract, 3),
            "rules_sec": round(t_rules, 3),
            "total_sec": round(t_total, 3)
        },
        "vram_before_cleanup": {
            "allocated_before": round(alloc_before, 2),
            "reserved_before": round(res_before, 2)
        },
        "vram_after_cleanup": {
            "allocated_gb": round(alloc_gb, 2),
            "reserved_gb": round(res_gb, 2)
        },
        "records": records
    }


if __name__ == "__main__":
    pdf_file = "pdfParser/sample_work_contract.pdf"
    print(f"[*] Benchmarking Batched Pipeline on '{pdf_file}'...")
    res = benchmark_contract_batched(pdf_file, batch_size=2)
    t = res["timings"]
    vram = res["vram_after_cleanup"]
    print("=" * 70)
    print(f"Document: {res['document']} | Clauses: {res['total_clauses']}")
    print(f"Timings : Total: {t['total_sec']}s (Parse: {t['pdf_parse_sec']}s, LegalBERT Batch: {t['classification_sec']}s, NuExtract Batch: {t['extraction_sec']}s, Rules: {t['rules_sec']}s)")
    print(f"VRAM    : Allocated {vram['allocated_gb']} GB | Reserved {vram['reserved_gb']} GB (Cleaned)")
    print("=" * 70)
