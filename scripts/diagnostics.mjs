import { randomUUID } from "node:crypto";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";

export function getLogsDirectory(projectRoot) {
  return path.join(projectRoot, "logs");
}

function safeLogKind(kind) {
  const normalized = String(kind || "diagnostic").toLowerCase().replace(/[^a-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "");
  return normalized || "diagnostic";
}

function timestampForFilename(date) {
  return date.toISOString().replace(/[-:]/g, "").replace("T", "-").replace(/\.\d{3}Z$/, "");
}

function formatDetails(details) {
  try {
    return JSON.stringify(details ?? {}, null, 2);
  } catch {
    return String(details);
  }
}

export function writeDiagnosticLog(projectRoot, { kind, message, details = {} }) {
  try {
    const createdAt = new Date();
    const directory = getLogsDirectory(projectRoot);
    mkdirSync(directory, { recursive: true });
    const filename = `${timestampForFilename(createdAt)}-${safeLogKind(kind)}-${randomUUID().slice(0, 8)}.log`;
    const content = [
      "XiriaCanvas AI diagnostic log",
      `Time: ${createdAt.toISOString()}`,
      `Type: ${safeLogKind(kind)}`,
      `Message: ${String(message || "No message")}`,
      "",
      "Details:",
      formatDetails(details),
      "",
    ].join("\n");
    const logPath = path.join(directory, filename);
    writeFileSync(logPath, content, "utf8");
    return { directory, filename, path: logPath };
  } catch {
    // Diagnostics must never obscure the original setup or inference failure.
    return null;
  }
}
