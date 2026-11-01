/**
 * ClauseGuard — Centralized Backend API Client.
 * Handles communication with the FastAPI backend at http://localhost:8000 (configurable via NEXT_PUBLIC_API_BASE_URL).
 * Implements native fetch() + ReadableStream SSE parsing for streaming multipart contract audits,
 * along with synchronous audit fallback, health checking, and policy rules retrieval.
 */

import {
  AuditReport,
  HealthResponse,
  RulesResponse,
  StreamProgressEvent,
} from "@/types/audit";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/+$/, "") ||
  "http://localhost:8000";

/**
 * Custom error class with HTTP status code and details.
 */
export class ApiError extends Error {
  constructor(
    message: string,
    public status?: number,
    public details?: unknown
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/**
 * Check backend and GPU health status.
 */
export async function checkHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`, {
    method: "GET",
    signal,
    headers: {
      Accept: "application/json",
    },
  });

  if (!response.ok) {
    throw new ApiError(
      `Health check failed with status ${response.status}`,
      response.status
    );
  }

  return response.json();
}

/**
 * Fetch the active deterministic review policies from the backend.
 */
export async function fetchRules(signal?: AbortSignal): Promise<RulesResponse> {
  const response = await fetch(`${API_BASE_URL}/api/rules`, {
    method: "GET",
    signal,
    headers: {
      Accept: "application/json",
    },
  });

  if (!response.ok) {
    throw new ApiError(
      `Failed to fetch rules with status ${response.status}`,
      response.status
    );
  }

  return response.json();
}

/**
 * Fallback: Synchronous contract audit returning the full report at once.
 */
export async function auditContractSync(
  file: File,
  signal?: AbortSignal
): Promise<AuditReport> {
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    throw new ApiError("Invalid file type: Only PDF documents are supported.");
  }

  const formData = new FormData();
  formData.append("file", file, file.name);

  const response = await fetch(`${API_BASE_URL}/api/audit`, {
    method: "POST",
    body: formData,
    signal,
  });

  if (!response.ok) {
    let errorDetail = `Audit failed with status ${response.status}`;
    try {
      const errJson = await response.json();
      if (errJson.detail) errorDetail = errJson.detail;
    } catch {
      // Keep default error text
    }
    throw new ApiError(errorDetail, response.status);
  }

  return response.json();
}

export interface StreamAuditCallbacks {
  onProgress?: (event: StreamProgressEvent) => void;
  onComplete?: (report: AuditReport) => void;
  onError?: (error: Error) => void;
}

/**
 * Primary: Real-time contract audit streaming progress via Server-Sent Events (SSE).
 * Uses browser-native fetch() + ReadableStream to support POST multipart uploads
 * with continuous line decoding across chunk boundaries.
 */
export async function auditContractStream(
  file: File,
  callbacks: StreamAuditCallbacks = {},
  signal?: AbortSignal
): Promise<AuditReport> {
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    const err = new ApiError("Invalid file type: Only PDF documents are supported.");
    callbacks.onError?.(err);
    throw err;
  }

  const formData = new FormData();
  formData.append("file", file, file.name);

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/api/audit/stream`, {
      method: "POST",
      body: formData,
      signal,
      headers: {
        Accept: "text/event-stream",
      },
    });
  } catch (netErr: any) {
    const errorMsg =
      netErr.name === "AbortError"
        ? "Audit request was cancelled."
        : `Network connection to backend failed (${API_BASE_URL}). Ensure FastAPI is running.`;
    const err = new ApiError(errorMsg, 0, netErr);
    callbacks.onError?.(err);
    throw err;
  }

  if (!response.ok) {
    let errorDetail = `Server returned HTTP ${response.status}`;
    try {
      const errJson = await response.json();
      if (errJson.detail) errorDetail = errJson.detail;
    } catch {
      // Not JSON
    }
    const err = new ApiError(errorDetail, response.status);
    callbacks.onError?.(err);
    throw err;
  }

  if (!response.body) {
    const err = new ApiError("ReadableStream is not supported or response body is null.");
    callbacks.onError?.(err);
    throw err;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let finalReport: AuditReport | null = null;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Split on newlines, keeping uncompleted lines in buffer
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith("data: ")) continue;

        const payloadText = trimmed.slice(6).trim();
        try {
          const event = JSON.parse(payloadText) as StreamProgressEvent;

          // Dispatch progress callback
          callbacks.onProgress?.(event);

          if (event.step === "completed" && event.report) {
            finalReport = event.report;
            callbacks.onComplete?.(event.report);
          } else if (event.step === "error") {
            const err = new ApiError(event.message || "Audit failed during pipeline execution.");
            callbacks.onError?.(err);
            throw err;
          }
        } catch (jsonErr: any) {
          // If this was an ApiError thrown above, re-throw it
          if (jsonErr instanceof ApiError) throw jsonErr;
          console.warn("[ClauseGuard API] Failed to parse SSE event chunk:", payloadText);
        }
      }
    }
  } finally {
    reader.releaseLock();
  }

  if (!finalReport) {
    const err = new ApiError("Stream closed before receiving the completed audit report.");
    callbacks.onError?.(err);
    throw err;
  }

  return finalReport;
}
