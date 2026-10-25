"""
ClauseGuard — Core Contract Audit Service Layer.
Encapsulates PDF parsing, LegalBERT classification, batched NuExtract entity extraction,
deterministic rule review, and Server-Sent Events (SSE) streaming progress.
"""

from __future__ import annotations

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Generator
from importlib import import_module

import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Internal Pydantic schemas
from app.schemas import (
    AuditReportSchema,
    AuditedClauseSchema,
    AuditTimingsSchema,
    ConcernFindingSchema,
)

# Pipeline modules
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


class ContractAuditService:
    """Production service orchestrating document intelligence across the 4 pipeline stages."""

    def __init__(
        self,
        classifier_model_path: str = "ml/artifacts/legalbert_classifier/checkpoint-775",
        use_fallback: bool = False,
        device: Optional[str] = None,
    ):
        print("[*] Initializing ContractAuditService components...")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fallback = use_fallback

        # 1. Structural PDF Parser
        self.pdf_pipeline = ContractParserPipeline()

        # 2. Fine-Tuned LegalBERT Classifier
        self.classifier = LegalBertClassifier(
            model_name_or_path=classifier_model_path,
            device=self.device,
            use_fallback=use_fallback,
        )

        # 3. NuExtract 2.0 Information Extractor
        self.extractor = NuExtractExtractor(
            device=self.device,
            use_fallback=use_fallback,
        )

        # 4. Deterministic Concern Detector & Rule Store
        self.detector = ConcernDetector()

        # Cache categories that have active configured review criteria
        self.active_categories = {r.clause_type for r in load_rules()}
        print(
            f"[OK] ContractAuditService ready on {self.device}. Active rule categories: {len(self.active_categories)}"
        )

    def _batch_extract(
        self,
        items: List[Tuple[str, str]],
        batch_size: int = 2,
    ) -> List[Dict[str, Any]]:
        """Batched entity extraction using NuExtract 2.0 with HuggingFace left-padding."""
        if not items:
            return []

        if not self.extractor.is_loaded:
            self.extractor.load_model()

        # Heuristic fallback path
        if self.extractor.use_fallback or self.extractor.model is None:
            results = []
            for text, cat in items:
                raw = self.extractor.extract(text, cat)
                try:
                    results.append(json.loads(raw) if isinstance(raw, str) else raw)
                except Exception as e:
                    results.append({"error": str(e), "raw": raw})
            return results

        extracted_facts: List[Dict[str, Any]] = []

        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            prompts = [
                format_nuextract_prompt(
                    text=text,
                    schema=EXTRACTION_SCHEMAS_WITH_DESCRIPTIONS.get(
                        cat, EXTRACTION_SCHEMAS_WITH_DESCRIPTIONS["other_general"]
                    ),
                    category=cat,
                )
                for text, cat in batch
            ]

            inputs = self.extractor.processor(
                text=prompts, padding=True, return_tensors="pt"
            ).to(self.extractor.device)
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = self.extractor.model.generate(
                    **inputs,
                    max_new_tokens=160,
                    do_sample=False,
                )

            decoded_strings = self.extractor.processor.batch_decode(
                outputs[:, input_len:], skip_special_tokens=True
            )

            for raw_str in decoded_strings:
                try:
                    cleaned = raw_str.strip()
                    if cleaned.startswith("```json"):
                        cleaned = cleaned[7:]
                    elif cleaned.startswith("```"):
                        cleaned = cleaned[3:]
                    if cleaned.endswith("```"):
                        cleaned = cleaned[:-3]
                    extracted_facts.append(json.loads(cleaned.strip()))
                except Exception as e:
                    extracted_facts.append({"error": str(e), "raw": raw_str})

        return extracted_facts

    def analyze_document(
        self,
        pdf_path: str,
        batch_size: int = 2,
        filter_general: bool = True,
    ) -> AuditReportSchema:
        """Executes full contract audit synchronously and returns a validated AuditReportSchema."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Contract file not found at: {pdf_path}")

        t_start = time.time()

        # Stage 1: PDF Layout & Hierarchy Tree Parsing
        t0 = time.time()
        parsed_doc = self.pdf_pipeline.parse_file(pdf_path)
        t_parse = time.time() - t0

        clauses = parsed_doc.get("clauses", [])
        all_texts = [c["text"] for c in clauses]

        # Stage 2: Parallel LegalBERT Classification
        t1 = time.time()
        predictions = self.classifier.batch_predict(all_texts, batch_size=32)
        t_classify = time.time() - t1

        # Identify actionable covenant clauses
        actionable_items: List[Tuple[str, str]] = []
        actionable_indices: List[int] = []

        for idx, (clause, pred) in enumerate(zip(clauses, predictions)):
            cat = pred["category"]
            should_extract = (
                (cat in self.active_categories and cat != "other_general")
                if filter_general
                else True
            )
            if should_extract:
                actionable_items.append((clause["text"], cat))
                actionable_indices.append(idx)

        # Stage 3: Batched NuExtract Entity Extraction
        t2 = time.time()
        batch_facts = self._batch_extract(actionable_items, batch_size=batch_size)
        t_extract = time.time() - t2

        facts_by_index = {
            idx: facts for idx, facts in zip(actionable_indices, batch_facts)
        }

        # Stage 4: Deterministic Policy Evaluation
        t3 = time.time()
        audited_clauses: List[AuditedClauseSchema] = []
        total_findings_count = 0
        flagged_clauses_count = 0

        for idx, (c, pred) in enumerate(zip(clauses, predictions)):
            cat = pred["category"]
            conf = pred.get("confidence", 0.0)
            text = c["text"]

            extracted = facts_by_index.get(idx, None)
            is_skipped = extracted is None
            clause_findings: List[ConcernFindingSchema] = []

            if not is_skipped and extracted is not None:
                det_res = self.detector.detect(
                    clause_type=cat,
                    extraction_output=extracted,
                    clause_text=text,
                )
                for f in det_res.findings:
                    clause_findings.append(
                        ConcernFindingSchema(
                            rule_id=f.rule_id,
                            severity=f.severity.upper(),  # type: ignore
                            field=f.field,
                            extracted_value=f.extracted_value,
                            rule_condition=f.rule_condition,
                            explanation=f.explanation,
                            evidence=f.evidence,
                        )
                    )

            if clause_findings:
                flagged_clauses_count += 1
                total_findings_count += len(clause_findings)

            # Determine clause risk level
            risk_level = "CLEAN"
            if clause_findings:
                sevs = {f.severity for f in clause_findings}
                if "HIGH" in sevs:
                    risk_level = "HIGH"
                elif "MEDIUM" in sevs:
                    risk_level = "MEDIUM"
                elif "LOW" in sevs:
                    risk_level = "LOW"
                else:
                    risk_level = "INFO"

            audited_clauses.append(
                AuditedClauseSchema(
                    clause_id=c.get("clause_id", f"clause_{idx+1}"),
                    section_title=c.get("section_title", "General"),
                    page=c.get("page_start", 1),
                    category=cat,
                    confidence=round(conf, 4),
                    skipped_extraction=is_skipped,
                    text=text,
                    extracted_entities=extracted,
                    findings=clause_findings,
                    risk_level=risk_level,  # type: ignore
                )
            )

        t_rules = time.time() - t3

        # Post-run VRAM cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        t_total = time.time() - t_start

        return AuditReportSchema(
            document_name=parsed_doc.get("document", Path(pdf_path).name),
            total_clauses=len(clauses),
            actionable_clauses=len(actionable_items),
            flagged_clauses=flagged_clauses_count,
            total_findings=total_findings_count,
            timings=AuditTimingsSchema(
                pdf_parse_sec=round(t_parse, 3),
                classification_sec=round(t_classify, 3),
                extraction_sec=round(t_extract, 3),
                rules_sec=round(t_rules, 3),
                total_sec=round(t_total, 3),
            ),
            clauses=audited_clauses,
        )

    def analyze_document_stream(
        self,
        pdf_path: str,
        batch_size: int = 2,
        filter_general: bool = True,
    ) -> Generator[Dict[str, Any], None, None]:
        """Generator yielding live step-by-step progress events (SSE-ready) followed by the completed report."""
        if not os.path.exists(pdf_path):
            yield {
                "step": "error",
                "progress": 0,
                "message": f"Contract file not found: {pdf_path}",
            }
            return

        t_start = time.time()

        # Step 1: PDF Layout Parsing
        yield {
            "step": "parsing",
            "progress": 15,
            "message": "Analyzing document layout and extracting clauses...",
        }
        t0 = time.time()
        parsed_doc = self.pdf_pipeline.parse_file(pdf_path)
        t_parse = time.time() - t0

        clauses = parsed_doc.get("clauses", [])
        all_texts = [c["text"] for c in clauses]
        yield {
            "step": "parsed",
            "progress": 30,
            "message": f"Extracted {len(clauses)} clauses across {parsed_doc.get('total_blocks', 0)} visual blocks.",
        }

        # Step 2: LegalBERT Classification
        yield {
            "step": "classifying",
            "progress": 45,
            "message": f"Classifying {len(clauses)} clauses with LegalBERT...",
        }
        t1 = time.time()
        predictions = self.classifier.batch_predict(all_texts, batch_size=32)
        t_classify = time.time() - t1

        actionable_items: List[Tuple[str, str]] = []
        actionable_indices: List[int] = []

        for idx, (clause, pred) in enumerate(zip(clauses, predictions)):
            cat = pred["category"]
            should_extract = (
                (cat in self.active_categories and cat != "other_general")
                if filter_general
                else True
            )
            if should_extract:
                actionable_items.append((clause["text"], cat))
                actionable_indices.append(idx)

        yield {
            "step": "classified",
            "progress": 60,
            "message": f"Identified {len(actionable_items)} actionable covenant clauses (filtered {len(clauses) - len(actionable_items)} general clauses).",
        }

        # Step 3: NuExtract Entity Extraction
        yield {
            "step": "extracting",
            "progress": 75,
            "message": f"Extracting structured facts from {len(actionable_items)} covenants (batch size {batch_size})...",
        }
        t2 = time.time()
        batch_facts = self._batch_extract(actionable_items, batch_size=batch_size)
        t_extract = time.time() - t2

        facts_by_index = {
            idx: facts for idx, facts in zip(actionable_indices, batch_facts)
        }

        # Step 4: Deterministic Policy Evaluation
        yield {
            "step": "evaluating",
            "progress": 90,
            "message": "Evaluating review policies against extracted facts...",
        }
        t3 = time.time()
        audited_clauses: List[AuditedClauseSchema] = []
        total_findings_count = 0
        flagged_clauses_count = 0

        for idx, (c, pred) in enumerate(zip(clauses, predictions)):
            cat = pred["category"]
            conf = pred.get("confidence", 0.0)
            text = c["text"]

            extracted = facts_by_index.get(idx, None)
            is_skipped = extracted is None
            clause_findings: List[ConcernFindingSchema] = []

            if not is_skipped and extracted is not None:
                det_res = self.detector.detect(
                    clause_type=cat,
                    extraction_output=extracted,
                    clause_text=text,
                )
                for f in det_res.findings:
                    clause_findings.append(
                        ConcernFindingSchema(
                            rule_id=f.rule_id,
                            severity=f.severity.upper(),  # type: ignore
                            field=f.field,
                            extracted_value=f.extracted_value,
                            rule_condition=f.rule_condition,
                            explanation=f.explanation,
                            evidence=f.evidence,
                        )
                    )

            if clause_findings:
                flagged_clauses_count += 1
                total_findings_count += len(clause_findings)

            risk_level = "CLEAN"
            if clause_findings:
                sevs = {f.severity for f in clause_findings}
                if "HIGH" in sevs:
                    risk_level = "HIGH"
                elif "MEDIUM" in sevs:
                    risk_level = "MEDIUM"
                elif "LOW" in sevs:
                    risk_level = "LOW"
                else:
                    risk_level = "INFO"

            audited_clauses.append(
                AuditedClauseSchema(
                    clause_id=c.get("clause_id", f"clause_{idx+1}"),
                    section_title=c.get("section_title", "General"),
                    page=c.get("page_start", 1),
                    category=cat,
                    confidence=round(conf, 4),
                    skipped_extraction=is_skipped,
                    text=text,
                    extracted_entities=extracted,
                    findings=clause_findings,
                    risk_level=risk_level,  # type: ignore
                )
            )

        t_rules = time.time() - t3

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        t_total = time.time() - t_start

        report = AuditReportSchema(
            document_name=parsed_doc.get("document", Path(pdf_path).name),
            total_clauses=len(clauses),
            actionable_clauses=len(actionable_items),
            flagged_clauses=flagged_clauses_count,
            total_findings=total_findings_count,
            timings=AuditTimingsSchema(
                pdf_parse_sec=round(t_parse, 3),
                classification_sec=round(t_classify, 3),
                extraction_sec=round(t_extract, 3),
                rules_sec=round(t_rules, 3),
                total_sec=round(t_total, 3),
            ),
            clauses=audited_clauses,
        )

        # Step 5: Completed with full report
        yield {
            "step": "completed",
            "progress": 100,
            "message": f"Audit complete. Found {total_findings_count} policy concerns across {flagged_clauses_count} clause(s).",
            "report": report.model_dump(),
        }
