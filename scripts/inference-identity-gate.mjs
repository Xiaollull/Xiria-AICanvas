export const INFERENCE_IDENTITY_GATE_ERROR = "INFERENCE_BACKEND_IDENTITY_UNAVAILABLE";

export function inferenceIdentityGate(health, protocol, workspaceId) {
  if (!health || health.status !== "ready") return { allowed: false, reason: "offline_or_not_ready" };
  if (health.protocol !== protocol) return { allowed: false, reason: "protocol_mismatch" };
  if (health.workspace_id !== workspaceId) return { allowed: false, reason: "workspace_mismatch" };
  return { allowed: true, reason: "matched" };
}

export function guardedInferenceResponse(pathname, health, protocol, workspaceId) {
  if (!pathname.startsWith("/api/inference/") || pathname === "/api/inference/health") return null;
  const gate = inferenceIdentityGate(health, protocol, workspaceId);
  return gate.allowed ? null : {
    statusCode: 503,
    body: { error: INFERENCE_IDENTITY_GATE_ERROR, reason: gate.reason },
  };
}
