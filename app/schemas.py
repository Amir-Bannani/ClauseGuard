from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal


class ConcernFindingSchema(BaseModel):
    rule_id: str
    severity: Literal["HIGH", "MEDIUM", "LOW", "INFO"]
    field: str
    extracted_value: Any
    rule_condition: str
    explanation: str
    evidence: Any | None = None


class AuditTimingsSchema(BaseModel):
    pdf_parse_sec: float = Field(ge=0)
    classification_sec: float = Field(ge=0)
    extraction_sec: float = Field(ge=0)
    rules_sec: float = Field(ge=0)
    total_sec: float = Field(ge=0)


class AuditedClauseSchema(BaseModel):
    clause_id: str
    section_title: str
    page: int = Field(ge=1)
    category: str
    confidence: float = Field(ge=0, le=1)
    skipped_extraction: bool
    text: str
    extracted_entities: dict[str, Any] | None = None
    findings: list[ConcernFindingSchema] = Field(default_factory=list)
    risk_level: Literal["HIGH", "MEDIUM", "LOW", "CLEAN", "INFO"]


class AuditReportSchema(BaseModel):
    document_name: str
    total_clauses: int = Field(ge=0)
    actionable_clauses: int = Field(ge=0)
    flagged_clauses: int = Field(ge=0)
    total_findings: int = Field(ge=0)
    timings: AuditTimingsSchema
    clauses: list[AuditedClauseSchema] = Field(default_factory=list)


    