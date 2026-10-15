import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { previousUpdaterAccepts } from "./update-compatibility.mjs";

const root = path.resolve(import.meta.dirname, "..");
const read = (relative) => readFileSync(path.join(root, relative), "utf8");
// The validator every release up to 1.3.0 shipped: in-process, protocol hard-coded.
const inProcessValidator = (protocol) => `const INFERENCE_PROTOCOL = ${protocol};\nexport async function validateUpdatedProject() {}\n`;
const backend = (protocol) => `import os\n\nINFERENCE_PROTOCOL = ${protocol}\n`;

test("an in-process validator accepts only its own protocol", () => {
  assert.deepEqual(previousUpdaterAccepts({ previousValidatorSource: inProcessValidator(39), backendSource: backend(39) }), { ok: true, delegated: false });
  // 1.0.9 -> 1.1.0 on 2026-09-18: the installed 35 validator met a 36 backend and rolled back.
  const refused = previousUpdaterAccepts({ previousValidatorSource: inProcessValidator(35), backendSource: backend(36) });
  assert.equal(refused.ok, false);
  assert.match(refused.reason, /asserts inference protocol 35.*declares 36/s);
});

test("a delegating validator accepts any protocol, because the new tree validates itself", () => {
  assert.deepEqual(previousUpdaterAccepts({ previousValidatorSource: read("scripts/update-validation.mjs"), backendSource: backend(40) }), { ok: true, delegated: true });
});

test("unreadable inputs fail closed", () => {
  assert.equal(previousUpdaterAccepts({ previousValidatorSource: "", backendSource: backend(39) }).ok, false);
  assert.equal(previousUpdaterAccepts({ previousValidatorSource: inProcessValidator(39), backendSource: "no protocol here" }).ok, false);
  assert.equal(previousUpdaterAccepts({}).ok, false);
});

test("the release rehearsal runs the compatibility check on the in-place path", () => {
  const workflow = read(".github/workflows/release.yml");
  assert.match(workflow, /import \{ previousUpdaterAccepts \} from "\.\/scripts\/update-compatibility\.mjs"/);
  assert.match(workflow, /`v\$\{process\.env\.PREVIOUS_VERSION\}:scripts\/update-validation\.mjs`/);
  assert.match(workflow, /if \(!compatibility\.ok\) throw new Error\(compatibility\.reason\)/);
});
