// QPlan2/P2: one fixture, two implementations. The JS rules (src/lora-mix-rules.js) and the authoritative Python
// contract (backend/lora_mix_search.py) must agree on parsing, sequences, displacement and exact counts.
// `node scripts/lora-mix-parity.mjs --write` refreshes backend/tests/fixtures/lora_mix_parity.json.
import { writeFile } from "node:fs/promises";
import path from "node:path";
import { analyzeMembers, countSpace, formatUnits, parseUnits } from "../src/lora-mix-rules.js";
import { createRng } from "../src/lora-mix-sampler.js";

export const PARITY_FILE = path.resolve(import.meta.dirname, "../backend/tests/fixtures/lora_mix_parity.json");
const PARSE_CASES = ["0.6", "-0.2", "5", "-5", ".5", "-0", "0.0001", "-0.0001", "0.1234", "", "-", "0.", "1e2", "0.12345", "5.0001", "-5.0001", "6", "abc", "--1", "1,5", " 0.3 ", "00.30", "007", "5.0", "4.9999"];

function vectors(free, limit, cap) {
  const out = [], current = new Array(free.length);
  const visit = (position, used) => {
    if (out.length >= cap) return;
    if (position === free.length) { out.push(current.slice()); return; }
    const { sequence } = free[position];
    for (let i = 0; i < sequence.size; i += 1) {
      const value = sequence.at(i), distance = Math.abs(value - sequence.base);
      if (limit !== null && used + distance > limit) continue;
      current[position] = value; visit(position + 1, used + distance);
    }
  };
  visit(0, 0);
  return out;
}

export function buildParityFixture() {
  const rng = createRng("20261009");
  const rules = [];
  for (let trial = 0; trial < 40; trial += 1) {
    const members = Array.from({ length: 1 + rng.int(4) }, () => {
      if (rng.float() < 0.2) return { rule: { locked: true, base: formatUnits(-5000 + rng.int(100) * 100) } };
      const min = -5000 + rng.int(60) * 100, span = rng.int(9) * 100, step = (1 + rng.int(5)) * 100, base = Math.min(min + span, min + rng.int(span / 100 + 1) * 100 + (rng.float() < 0.3 ? 50 : 0));
      return { rule: { locked: false, base: formatUnits(base), min: formatUnits(min), max: formatUnits(min + span), step: formatUnits(step), preference: ["balanced", "min", "max"][rng.int(3)] } };
    });
    const analysis = analyzeMembers(members);
    const limits = analysis.valid ? [0, 100, 250, 600, analysis.dmax, analysis.dmax + 5000].filter((value, i, all) => all.indexOf(value) === i) : [];
    rules.push({
      members, valid: analysis.valid, dmax: analysis.dmax, space: analysis.space === null ? null : analysis.space.toString(),
      sequences: analysis.free.map((row) => Array.from({ length: row.sequence.size }, (_, i) => row.sequence.at(i))),
      counts: Object.fromEntries(limits.map((limit) => { const count = countSpace(analysis.free, Math.min(limit, analysis.dmax)); return [String(limit), count.kind === "exact" ? count.count.toString() : null]; })),
      first_vectors: analysis.valid ? vectors(analysis.free, analysis.dmax ? Math.floor(analysis.dmax / 2) : null, 25) : [],
      half_limit: analysis.valid && analysis.dmax ? Math.floor(analysis.dmax / 2) : null,
    });
  }
  // Rule sets with invalid fields keep their field-keyed errors identical in both languages.
  const invalid = [
    { locked: false, base: "0.6", min: "0.8", max: "1.0", step: "0.1", preference: "max" }, { locked: false, base: "0.6", min: "0.2", max: "1.0", step: "0", preference: "max" },
    { locked: false, base: "0.6", min: "1.0", max: "0.2", step: "0.1", preference: "max" }, { locked: false, base: "0.6", min: "0.2", max: "1.0", step: "0.00001", preference: "max" },
    { locked: false, base: "x", min: "0.2", max: "1.0", step: "0.1", preference: "max" }, { locked: false, base: "0.6", min: "0.2", max: "1.0", step: "0.1", preference: "weird" },
    { locked: true, base: "0.5", min: "junk", max: "", step: "-3", preference: "nope" }, { locked: true, base: "0.123456" },
  ].map((rule) => { const row = analyzeMembers([{ rule }]).rows[0]; return { rule, error_fields: Object.keys(row.errors).sort(), locked: row.locked }; });
  return { schema_version: 1, parse: PARSE_CASES.map((text) => { const result = parseUnits(text); return { text, ok: result.ok, units: result.ok ? result.units : null, reason: result.ok ? null : result.reason }; }), rules, invalid };
}

if (process.argv.includes("--write")) {
  await writeFile(PARITY_FILE, `${JSON.stringify(buildParityFixture(), null, 1)}\n`, "utf8");
  console.log(`wrote ${PARITY_FILE}`);
}
