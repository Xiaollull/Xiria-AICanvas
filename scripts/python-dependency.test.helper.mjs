import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";

// Probe dependency presence only, not backend imports: a backend regression or
// an installed-but-broken package must fail the test, never turn into a skip.
export function pythonDependencyOptions(python, cwd, modules) {
  const probe = spawnSync(python, ["-B", "-c",
    "import importlib.util,json,sys; print(json.dumps([m for m in sys.argv[1:] if importlib.util.find_spec(m) is None]))", ...modules],
  { cwd, encoding: "utf8", windowsHide: true, timeout: 15000 });
  assert.equal(probe.status, 0, probe.error?.message || probe.stderr || "Python dependency probe failed");
  const missing = JSON.parse(probe.stdout);
  return { skip: missing.length ? `Python dependencies unavailable: ${missing.join(", ")}` : false };
}
