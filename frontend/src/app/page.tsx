"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  AuditReport,
  AuditedClause,
  AuditStreamStep,
  HealthResponse,
  RiskLevel,
  StreamProgressEvent,
} from "@/types/audit";
import { auditContractStream, checkHealth, ApiError } from "@/lib/api";
import {
  FileText,
  UploadCloud,
  CheckCircle2,
  AlertTriangle,
  AlertCircle,
  Clock,
  Shield,
  Cpu,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  FileCheck,
  Zap,
} from "lucide-react";

export default function HomePage() {
  // Backend & GPU Health
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);

  // Upload & File State
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Audit Progress State
  const [isAuditing, setIsAuditing] = useState(false);
  const [progress, setProgress] = useState<number>(0);
  const [currentStep, setCurrentStep] = useState<AuditStreamStep | null>(null);
  const [stepMessage, setStepMessage] = useState<string>("");

  // Report & Error State
  const [report, setReport] = useState<AuditReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Clause Filter & Expand State
  const [riskFilter, setRiskFilter] = useState<"ALL" | "FLAGGED" | "CLEAN">("ALL");
  const [expandedClauses, setExpandedClauses] = useState<Record<string, boolean>>({});

  // AbortController ref to allow cancellation
  const abortControllerRef = useRef<AbortController | null>(null);

  // Fetch backend health on mount
  useEffect(() => {
    let isMounted = true;
    async function loadHealth() {
      try {
        const data = await checkHealth();
        if (isMounted) {
          setHealth(data);
          setHealthError(null);
        }
      } catch (err: any) {
        if (isMounted) {
          setHealthError(
            err instanceof ApiError
              ? err.message
              : "Backend unavailable. Ensure FastAPI is running on port 8000."
          );
        }
      }
    }
    loadHealth();
    return () => {
      isMounted = false;
    };
  }, []);

  // Handle File Selection
  const handleFileDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (isAuditing) return;

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const validateAndSetFile = (selectedFile: File) => {
    if (!selectedFile.name.toLowerCase().endsWith(".pdf")) {
      setError("Please select a valid PDF contract document (.pdf)");
      setFile(null);
      return;
    }
    setError(null);
    setFile(selectedFile);
    setReport(null);
    setProgress(0);
  };

  // Start Streaming Audit
  const handleStartAudit = async () => {
    if (!file || isAuditing) return;

    setIsAuditing(true);
    setError(null);
    setReport(null);
    setProgress(5);
    setCurrentStep("started");
    setStepMessage("Initializing connection to audit pipeline...");

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const finalReport = await auditContractStream(
        file,
        {
          onProgress: (event: StreamProgressEvent) => {
            setProgress(event.progress);
            setCurrentStep(event.step);
            setStepMessage(event.message);
          },
          onComplete: (completedReport: AuditReport) => {
            setReport(completedReport);
            setIsAuditing(false);
            setProgress(100);
          },
          onError: (err: Error) => {
            setError(err.message);
            setIsAuditing(false);
          },
        },
        controller.signal
      );

      setReport(finalReport);
    } catch (err: any) {
      if (err.name !== "AbortError") {
        setError(err.message || "Audit failed during execution.");
      }
      setIsAuditing(false);
    } finally {
      abortControllerRef.current = null;
    }
  };

  // Toggle Clause Accordion
  const toggleClause = (clauseId: string) => {
    setExpandedClauses((prev) => ({
      ...prev,
      [clauseId]: !prev[clauseId],
    }));
  };

  // Filter Clauses
  const filteredClauses = report
    ? report.clauses.filter((c) => {
        if (riskFilter === "FLAGGED") return c.findings && c.findings.length > 0;
        if (riskFilter === "CLEAN") return !c.findings || c.findings.length === 0;
        return true;
      })
    : [];

  const getBadgeClass = (level: RiskLevel) => {
    switch (level) {
      case "HIGH":
        return "badge-high";
      case "MEDIUM":
        return "badge-medium";
      case "LOW":
        return "badge-low";
      case "CLEAN":
      default:
        return "badge-clean";
    }
  };

  return (
    <div className="container">
      {/* App Header */}
      <header className="app-header">
        <div className="brand-wrapper">
          <div className="brand-icon">
            <Shield size={22} />
          </div>
          <div>
            <h1 className="brand-title">ClauseGuard</h1>
            <p className="brand-subtitle">
              Automated Employment Contract Audit & Covenant Risk Analyzer
            </p>
          </div>
        </div>

        {/* System & GPU Status */}
        <div className="status-pill">
          {health?.models_loaded ? (
            <>
              <span className="status-dot online" />
              <span>
                Backend Online &bull; {health.device || "GPU"} (RTX 3050)
              </span>
            </>
          ) : (
            <>
              <span className="status-dot offline" />
              <span>{healthError ? "Backend Offline" : "Connecting..."}</span>
            </>
          )}
        </div>
      </header>

      {/* Main Content Layout */}
      <main>
        {/* Upload Card */}
        <section className="glass-card">
          <div
            className={`upload-dropzone ${isDragging ? "dragging" : ""} ${
              isAuditing ? "disabled" : ""
            }`}
            onDragOver={(e) => {
              e.preventDefault();
              if (!isAuditing) setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleFileDrop}
            onClick={() => {
              if (!isAuditing) fileInputRef.current?.click();
            }}
          >
            <input
              type="file"
              ref={fileInputRef}
              style={{ display: "none" }}
              accept=".pdf"
              onChange={handleFileInput}
              disabled={isAuditing}
            />

            <div style={{ display: "flex", justifyContent: "center", marginBottom: "16px" }}>
              <div
                style={{
                  width: "56px",
                  height: "56px",
                  borderRadius: "50%",
                  background: "rgba(99, 102, 241, 0.1)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  color: "#818cf8",
                }}
              >
                {file ? <FileCheck size={28} /> : <UploadCloud size={28} />}
              </div>
            </div>

            {file ? (
              <div>
                <h3 style={{ fontSize: "17px", fontWeight: "600", color: "#f8fafc" }}>
                  {file.name}
                </h3>
                <p style={{ fontSize: "13px", color: "#94a3b8", marginTop: "4px" }}>
                  {(file.size / (1024 * 1024)).toFixed(2)} MB &bull; Ready for legal audit
                </p>
              </div>
            ) : (
              <div>
                <h3 style={{ fontSize: "17px", fontWeight: "600", color: "#f8fafc" }}>
                  Drag and drop your contract PDF here
                </h3>
                <p style={{ fontSize: "13px", color: "#94a3b8", marginTop: "4px" }}>
                  or click to browse your local computer
                </p>
              </div>
            )}
          </div>

          {/* Action Bar */}
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginTop: "20px",
            }}
          >
            <div>
              {file && (
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={() => {
                    setFile(null);
                    setReport(null);
                    setProgress(0);
                    setError(null);
                  }}
                  disabled={isAuditing}
                >
                  <RefreshCw size={14} /> Change Contract
                </button>
              )}
            </div>

            <button
              type="button"
              className="btn-primary"
              onClick={handleStartAudit}
              disabled={!file || isAuditing}
            >
              {isAuditing ? (
                <>
                  <RefreshCw size={16} className="animate-spin" /> Auditing Contract...
                </>
              ) : (
                <>
                  <Zap size={16} /> Run AI Contract Audit
                </>
              )}
            </button>
          </div>

          {/* Error Message */}
          {error && (
            <div
              style={{
                marginTop: "16px",
                padding: "12px 16px",
                borderRadius: "8px",
                background: "rgba(239, 68, 68, 0.1)",
                border: "1px solid rgba(239, 68, 68, 0.3)",
                color: "#f87171",
                display: "flex",
                alignItems: "center",
                gap: "8px",
                fontSize: "14px",
              }}
            >
              <AlertCircle size={18} />
              <span>{error}</span>
            </div>
          )}
        </section>

        {/* Real-Time SSE Progress Card */}
        {isAuditing && (
          <section className="glass-card progress-card">
            <div className="progress-header">
              <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                <Clock size={18} color="#818cf8" />
                <span style={{ fontWeight: "600", fontSize: "15px" }}>
                  Live Pipeline Progress
                </span>
              </div>
              <span style={{ fontWeight: "700", color: "#818cf8", fontSize: "16px" }}>
                {progress}%
              </span>
            </div>

            {/* Glowing Progress Track */}
            <div className="progress-bar-track">
              <div
                className="progress-bar-fill"
                style={{ width: `${Math.max(progress, 5)}%` }}
              />
            </div>

            <p style={{ marginTop: "12px", fontSize: "13px", color: "#cbd5e1" }}>
              {stepMessage || "Processing..."}
            </p>

            {/* 4 Pipeline Milestones */}
            <div className="progress-step-list">
              <div
                className={`step-item ${
                  progress >= 30 ? "done" : progress >= 15 ? "active" : ""
                }`}
              >
                <div style={{ fontWeight: "600" }}>1. PDF Layout</div>
                <div style={{ color: "#94a3b8" }}>Clause extraction</div>
              </div>

              <div
                className={`step-item ${
                  progress >= 60 ? "done" : progress >= 45 ? "active" : ""
                }`}
              >
                <div style={{ fontWeight: "600" }}>2. LegalBERT</div>
                <div style={{ color: "#94a3b8" }}>Covenant classification</div>
              </div>

              <div
                className={`step-item ${
                  progress >= 90 ? "done" : progress >= 75 ? "active" : ""
                }`}
              >
                <div style={{ fontWeight: "600" }}>3. NuExtract 2.0</div>
                <div style={{ color: "#94a3b8" }}>Entity extraction</div>
              </div>

              <div
                className={`step-item ${
                  progress === 100 ? "done" : progress >= 90 ? "active" : ""
                }`}
              >
                <div style={{ fontWeight: "600" }}>4. Policy Check</div>
                <div style={{ color: "#94a3b8" }}>Deterministic rules</div>
              </div>
            </div>
          </section>
        )}

        {/* Audit Report View */}
        {report && (
          <section style={{ marginTop: "32px" }}>
            {/* Executive Summary Card */}
            <div className="glass-card">
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "flex-start",
                  flexWrap: "wrap",
                  gap: "16px",
                  borderBottom: "1px solid var(--border-subtle)",
                  paddingBottom: "16px",
                }}
              >
                <div>
                  <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                    <FileText size={20} color="#818cf8" />
                    <h2 style={{ fontSize: "20px", fontWeight: "700" }}>
                      {report.document_name}
                    </h2>
                  </div>
                  <p style={{ fontSize: "13px", color: "#94a3b8", marginTop: "4px" }}>
                    Full AI Compliance Audit Report &bull; Analyzed locally on RTX 3050
                  </p>
                </div>

                <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
                  <span
                    className={`badge ${
                      report.total_findings > 0 ? "badge-high" : "badge-clean"
                    }`}
                    style={{ fontSize: "13px", padding: "6px 14px" }}
                  >
                    {report.total_findings > 0
                      ? `${report.total_findings} Policy Risk(s) Found`
                      : "Compliant & Clean"}
                  </span>
                </div>
              </div>

              {/* Stats Grid */}
              <div className="stats-grid">
                <div className="stat-box">
                  <div className="stat-label">Total Clauses</div>
                  <div className="stat-value">{report.total_clauses}</div>
                </div>

                <div className="stat-box">
                  <div className="stat-label">Actionable Covenants</div>
                  <div className="stat-value" style={{ color: "#818cf8" }}>
                    {report.actionable_clauses}
                  </div>
                </div>

                <div className="stat-box">
                  <div className="stat-label">Flagged Clauses</div>
                  <div
                    className="stat-value"
                    style={{
                      color: report.flagged_clauses > 0 ? "#f87171" : "#34d399",
                    }}
                  >
                    {report.flagged_clauses}
                  </div>
                </div>

                <div className="stat-box">
                  <div className="stat-label">Total Audit Latency</div>
                  <div className="stat-value" style={{ fontSize: "22px" }}>
                    {report.timings.total_sec}s
                  </div>
                  <div style={{ fontSize: "11px", color: "#64748b", marginTop: "4px" }}>
                    NuExtract: {report.timings.extraction_sec}s
                  </div>
                </div>
              </div>

              {/* Filter Tabs */}
              <div
                style={{
                  display: "flex",
                  gap: "8px",
                  marginTop: "8px",
                  borderBottom: "1px solid var(--border-subtle)",
                  paddingBottom: "12px",
                }}
              >
                <button
                  type="button"
                  className="btn-secondary"
                  style={{
                    backgroundColor:
                      riskFilter === "ALL"
                        ? "rgba(99, 102, 241, 0.2)"
                        : "transparent",
                    borderColor:
                      riskFilter === "ALL" ? "var(--primary)" : "var(--border-subtle)",
                    color: riskFilter === "ALL" ? "#fff" : "var(--text-secondary)",
                  }}
                  onClick={() => setRiskFilter("ALL")}
                >
                  All Clauses ({report.clauses.length})
                </button>

                <button
                  type="button"
                  className="btn-secondary"
                  style={{
                    backgroundColor:
                      riskFilter === "FLAGGED"
                        ? "rgba(239, 68, 68, 0.2)"
                        : "transparent",
                    borderColor:
                      riskFilter === "FLAGGED" ? "#ef4444" : "var(--border-subtle)",
                    color: riskFilter === "FLAGGED" ? "#fff" : "var(--text-secondary)",
                  }}
                  onClick={() => setRiskFilter("FLAGGED")}
                >
                  Flagged Only ({report.flagged_clauses})
                </button>

                <button
                  type="button"
                  className="btn-secondary"
                  style={{
                    backgroundColor:
                      riskFilter === "CLEAN"
                        ? "rgba(59, 130, 246, 0.2)"
                        : "transparent",
                    borderColor:
                      riskFilter === "CLEAN" ? "#3b82f6" : "var(--border-subtle)",
                    color: riskFilter === "CLEAN" ? "#fff" : "var(--text-secondary)",
                  }}
                  onClick={() => setRiskFilter("CLEAN")}
                >
                  Clean Clauses ({report.total_clauses - report.flagged_clauses})
                </button>
              </div>

              {/* Clauses List */}
              <div style={{ marginTop: "20px" }}>
                {filteredClauses.length === 0 ? (
                  <p style={{ textAlign: "center", color: "#94a3b8", padding: "32px 0" }}>
                    No clauses matching the selected filter.
                  </p>
                ) : (
                  filteredClauses.map((clause) => {
                    const isExpanded = !!expandedClauses[clause.clause_id];
                    const hasFindings = clause.findings && clause.findings.length > 0;

                    return (
                      <div
                        key={clause.clause_id}
                        className={`clause-card ${
                          hasFindings ? "has-findings" : "clean"
                        }`}
                      >
                        <div
                          className="clause-header"
                          onClick={() => toggleClause(clause.clause_id)}
                        >
                          <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                            <span
                              style={{
                                fontSize: "12px",
                                fontWeight: "600",
                                color: "#818cf8",
                                background: "rgba(99, 102, 241, 0.15)",
                                padding: "2px 8px",
                                borderRadius: "4px",
                              }}
                            >
                              Page {clause.page}
                            </span>
                            <span style={{ fontWeight: "600", fontSize: "15px" }}>
                              {clause.section_title || clause.clause_id}
                            </span>
                            <span
                              style={{
                                fontSize: "12px",
                                color: "#94a3b8",
                                background: "#1e293b",
                                padding: "2px 8px",
                                borderRadius: "4px",
                              }}
                            >
                              {clause.category} ({(clause.confidence * 100).toFixed(0)}%)
                            </span>
                          </div>

                          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                            <span className={`badge ${getBadgeClass(clause.risk_level)}`}>
                              {clause.risk_level}
                            </span>
                            {isExpanded ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
                          </div>
                        </div>

                        {isExpanded && (
                          <div className="clause-body">
                            {/* Contract clause original text */}
                            <div className="clause-text">
                              &ldquo;{clause.text}&rdquo;
                            </div>

                            {/* Extracted Facts if available */}
                            {clause.extracted_entities && (
                              <div style={{ marginBottom: "16px" }}>
                                <div
                                  style={{
                                    fontSize: "12px",
                                    fontWeight: "600",
                                    color: "#94a3b8",
                                    textTransform: "uppercase",
                                    marginBottom: "6px",
                                  }}
                                >
                                  Extracted NuExtract Facts:
                                </div>
                                <pre
                                  style={{
                                    background: "#0a0e17",
                                    padding: "12px",
                                    borderRadius: "6px",
                                    fontSize: "12px",
                                    color: "#a5b4fc",
                                    overflowX: "auto",
                                    border: "1px solid var(--border-subtle)",
                                  }}
                                >
                                  {JSON.stringify(clause.extracted_entities, null, 2)}
                                </pre>
                              </div>
                            )}

                            {/* Detected Findings */}
                            {hasFindings ? (
                              <div>
                                <div
                                  style={{
                                    fontSize: "12px",
                                    fontWeight: "600",
                                    color: "#f87171",
                                    textTransform: "uppercase",
                                    marginBottom: "8px",
                                    display: "flex",
                                    alignItems: "center",
                                    gap: "6px",
                                  }}
                                >
                                  <AlertTriangle size={14} /> Identified Policy Risks ({clause.findings.length}):
                                </div>
                                {clause.findings.map((f, fIdx) => (
                                  <div key={fIdx} className="finding-box">
                                    <div className="finding-title">
                                      <span
                                        className={`badge ${getBadgeClass(
                                          f.severity as RiskLevel
                                        )}`}
                                      >
                                        {f.severity}
                                      </span>
                                      <span>Rule: {f.rule_id}</span>
                                    </div>
                                    <p className="finding-desc">{f.explanation}</p>
                                    <div
                                      style={{
                                        fontSize: "12px",
                                        color: "#cbd5e1",
                                        marginTop: "8px",
                                        padding: "6px 8px",
                                        background: "rgba(0,0,0,0.25)",
                                        borderRadius: "4px",
                                      }}
                                    >
                                      <strong>Condition:</strong> {f.rule_condition} &bull;{" "}
                                      <strong>Found Value:</strong>{" "}
                                      <code>{JSON.stringify(f.extracted_value)}</code>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <div
                                style={{
                                  display: "flex",
                                  alignItems: "center",
                                  gap: "8px",
                                  color: "#34d399",
                                  fontSize: "13px",
                                }}
                              >
                                <CheckCircle2 size={16} /> No policy violations detected in this clause.
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
