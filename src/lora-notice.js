export const LORA_NOTICE_KINDS = ["error", "warning", "success", "info"];

/**
 * Typed manager feedback. A plain string is information: severity comes from the operation that
 * produced the message, never from guessing at its wording.
 */
export function loraNotice(value, kind = "info") {
  if (!value) return null;
  if (typeof value === "string") return { kind: LORA_NOTICE_KINDS.includes(kind) ? kind : "info", title: "", detail: value, diagnostics: [] };
  const detail = typeof value.detail === "string" ? value.detail : "";
  if (!detail && !value.title) return null;
  return { kind: LORA_NOTICE_KINDS.includes(value.kind) ? value.kind : "info", title: value.title || "", detail,
    diagnostics: Array.isArray(value.diagnostics) ? value.diagnostics.filter((item) => item && typeof item.message === "string") : [] };
}
