import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { getLogsDirectory, writeDiagnosticLog } from "./diagnostics.mjs";

test("diagnostic logs are written under the project logs directory", () => {
  const projectRoot = mkdtempSync(path.join(os.tmpdir(), "xirai-diagnostics-"));
  try {
    const result = writeDiagnosticLog(projectRoot, {
      kind: "setup failure",
      message: "CUDA verification failed",
      details: { variant: "cu126", code: 1 },
    });
    assert.equal(result.directory, getLogsDirectory(projectRoot));
    assert.match(result.filename, /^\d{8}-\d{6}-setup-failure-[a-f0-9]{8}\.log$/);
    const content = readFileSync(result.path, "utf8");
    assert.match(content, /Type: setup-failure/);
    assert.match(content, /CUDA verification failed/);
    assert.match(content, /"variant": "cu126"/);
  } finally {
    rmSync(projectRoot, { recursive: true, force: true });
  }
});
