/**
 * ClauseGuard — Core TypeScript Data Contracts.
 * Strict type definitions aligning 1:1 with FastAPI Pydantic schemas (app/schemas.py)
 * and Server-Sent Events (SSE) streaming progress events.
 */

export type SeverityLevel = "HIGH" | "MEDIUM" | "LOW" | "INFO";

export type RiskLevel = "HIGH" | "MEDIUM" | "LOW" | "CLEAN" | "INFO";

export interface ConcernFinding {
  rule_id: string;
  severity: SeverityLevel;
  field: string;
  extracted_value: unknown;
  rule_condition: string;
  explanation: string;
  evidence?: unknown;
}

export interface AuditTimings {
  pdf_parse_sec: number;
  classification_sec: number;
  extraction_sec: number;
  rules_sec: number;
  total_sec: number;
}

export interface AuditedClause {
  clause_id: string;
  section_title: string;
  page: number;
  category: string;
  confidence: number;
  skipped_extraction: boolean;
  text: string;
  extracted_entities?: Record<string, unknown> | null;
  findings: ConcernFinding[];
  risk_level: RiskLevel;
}

export interface AuditReport {
  document_name: string;
  total_clauses: number;
  actionable_clauses: number;
  flagged_clauses: number;
  total_findings: number;
  timings: AuditTimings;
  clauses: AuditedClause[];
}

export type AuditStreamStep =
  | "started"
  | "parsed"
  | "classifying"
  | "classified"
  | "extracting"
  | "evaluating"
  | "completed"
  | "error";

export interface StreamProgressEvent {
  step: AuditStreamStep;
  progress: number; // 0 to 100
  message: string;
  report?: AuditReport;
}

export interface HealthResponse {
  status: string;
  cuda_available: boolean;
  device: string;
  vram_allocated_mb: number;
  vram_reserved_mb: number;
  models_loaded: boolean;
}

export interface ReviewRule {
  rule_id: string;
  clause_type: string;
  field: string;
  operator: string;
  value?: unknown;
  unit?: string;
  finding_type: string;
  severity: string;
  description: string;
  rationale: string;
  source?: string;
  exception_field?: string;
}

export interface RulesResponse {
  total_rules: number;
  rules: ReviewRule[];
}
