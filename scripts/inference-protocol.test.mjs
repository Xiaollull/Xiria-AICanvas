import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";

const root = path.resolve(import.meta.dirname, "..");

test("frontend supervisor, startup, update validation, and Python backend use protocol 34", () => {
  const sources = [
    ["vite.config.js", /const inferenceProtocol = (\d+);/],
    ["scripts/start.mjs", /const inferenceProtocol = (\d+);/],
    ["scripts/update-validation.mjs", /const INFERENCE_PROTOCOL = (\d+);/],
    ["backend/inference_server.py", /INFERENCE_PROTOCOL = (\d+)/],
  ];
  const protocols = sources.map(([relative, pattern]) => {
    const match = readFileSync(path.join(root, relative), "utf8").match(pattern);
    assert.ok(match, `${relative} must declare the inference protocol`);
    return Number(match[1]);
  });
  assert.deepEqual(protocols, [34, 34, 34, 34]);
});
