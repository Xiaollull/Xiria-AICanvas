import test from "node:test";
import assert from "node:assert/strict";
import { createLoraMixDraft, validLoraMix } from "../src/lora-mix-draft.js";
import { adoptAsBase, applyAdoptAsBase, followUp, groupFromCandidate, isFavorite, recommendedColumns, toggleFavorite } from "../src/lora-mix-followup.js";
import { mixIssues } from "../src/lora-mix-model.js";

// QPlan2/P2 Phase D: explicit, previewable edits that start from a finished candidate. Synthetic references only.
const uuid = (n) => `00000000-0000-4000-8000-${String(n).padStart(12, "0")}`;
const hash = (n) => n.toString(16).padStart(2, "0").repeat(32);
const lora = (n, extra = {}) => ({ value: `style/l${n}.safetensors`, name: `L${n}`, category: "style", weight: 0.6, precision: 4, enabled: true, identitySchema: 1, instanceId: uuid(n), contentHash: hash(n), contentRevision: 2, ...extra });
const baseline = (loras = []) => ({ parameters: { engine: "Anima", diffusion_model: "d", text_encoder: "t", vae: "v", prompt: "raw", negative_prompt: "", steps: 20, cfg: 4, width: 512, height: 512, seed: "7", sampler: "euler", scheduler: "normal", guidance: "none", pag: { scale: 0.3, applied_layers: "mid" }, postprocess_chain: [] },
  loras, group_snapshots: [], prompt_without_groups: "raw", group_prompt: "", seed_mode: "fixed", library_revision: 0 });
const draft = (globals = []) => {
  const value = createLoraMixDraft(baseline(globals));
  value.mix.members = [{ ref: lora(1), rule: { locked: false, base: "0.6", min: "0.2", max: "1.0", step: "0.2", preference: "balanced" } }, { ref: lora(2), rule: { locked: false, base: "-0.2", min: "-0.8", max: "0.6", step: "0.2", preference: "max" } }, { ref: lora(3), rule: { locked: true, base: "0.5" } }];
  value.mix.max_displacement = "0.6";
  value.mix.candidates = [{ id: "C01", source: "baseline", raw: { [uuid(1)]: "0.6", [uuid(2)]: "-0.2" } }];
  return value;
};
const weights = { [uuid(1)]: 8000, [uuid(2)]: -4000, [uuid(3)]: 5000 };

test("recommended presentation columns: 4x4, 8x4, 8x8, never an axis", () => {
  assert.deepEqual([1, 2, 4, 9, 16, 32, 64, 128, 100, 256].map(recommendedColumns), [1, 2, 4, 3, 4, 8, 8, 12, 10, 12]);
  assert.equal(recommendedColumns(0), 1); assert.equal(recommendedColumns(Number.NaN), 1);
});

test("adopt as baseline: only unlocked members move, only when the current range still holds the value, nothing is auto-widened", () => {
  const source = draft(), before = structuredClone(source);
  const plan = adoptAsBase(source, weights);
  assert.equal(plan.ok, true); assert.deepEqual(plan.changes.map((item) => [item.name, item.from, item.to]), [["L1", "0.6", "0.8"], ["L2", "-0.2", "-0.4"]]);
  assert.deepEqual(source, before, "planning never mutates the draft");
  const applied = applyAdoptAsBase(source, weights);
  assert.deepEqual(applied.draft.mix.members.map((member) => member.rule.base), ["0.8", "-0.4", "0.5"], "the locked member keeps its base");
  assert.deepEqual(applied.draft.mix.members[0].rule.min, "0.2"); assert.deepEqual(applied.draft.mix.candidates, [], "the preview is cleared: it was generated around the old base");
  assert.deepEqual(applied.draft.mix.max_displacement, "0.6");
  assert.equal(mixIssues(applied.draft).analysis.valid, true);
  const outside = adoptAsBase(source, { ...weights, [uuid(1)]: 20000 });
  assert.equal(outside.ok, false); assert.match(outside.issues[0], /当前范围无法容纳 2/);
  assert.equal(applyAdoptAsBase(source, { ...weights, [uuid(1)]: 20000 }).draft, source, "a refused adoption returns the draft unchanged");
  assert.equal(adoptAsBase(source, { [uuid(1)]: 8000 }).ok, false, "a candidate that does not mention a member cannot be adopted");
});

test("narrower and finer follow-ups are exact integer arithmetic around the chosen vector, clipped to the old range", () => {
  const source = draft();
  const narrower = followUp(source, weights, { narrow: true, finer: false });
  assert.equal(narrower.ok, true);
  // L1: span 0.8 -> half-width 0.2 around 0.8, clipped to [0.2, 1.0] -> [0.6, 1.0]. L2: span 1.4 -> 0.35 around -0.4 -> [-0.75, -0.05].
  assert.deepEqual(narrower.changes.map((item) => item.after), [{ base: "0.8", min: "0.6", max: "1", step: "0.2" }, { base: "-0.4", min: "-0.75", max: "-0.05", step: "0.2" }]);
  assert.deepEqual(narrower.draft.mix.members[2].rule, { locked: true, base: "0.5" });
  assert.deepEqual(narrower.draft.mix.candidates, []);
  assert.equal(mixIssues(narrower.draft).analysis.valid, true);
  const finer = followUp(source, weights, { narrow: false, finer: true });
  assert.deepEqual(finer.changes.map((item) => [item.after.min, item.after.max, item.after.step]), [["0.2", "1.0", "0.1"], ["-0.8", "0.6", "0.1"]], "finer halves the step and keeps the range text exactly as typed");
  const both = followUp(source, weights, { narrow: true, finer: true });
  assert.deepEqual(both.changes[0].after, { base: "0.8", min: "0.6", max: "1", step: "0.1" });
  // The bound follows the new reachable maximum only when the old one would be refused, and the preview says so.
  const tight = draft();
  tight.mix.max_displacement = "1.2";
  const shrunk = followUp(tight, weights, { narrow: true });
  assert.equal(shrunk.displacement.before, "1.2"); assert.notEqual(shrunk.displacement.after, "1.2");
  assert.equal(mixIssues(shrunk.draft).issues.some((issue) => issue.scope === "displacement"), false);
  assert.equal(followUp(source, { ...weights, [uuid(2)]: 90000 }, {}).ok, false);
  const extreme = draft(); extreme.mix.members[0].rule.step = "0.0001"; extreme.mix.members[0].rule.min = "0.5999"; extreme.mix.members[0].rule.max = "0.6001";
  assert.equal(followUp(extreme, { ...weights, [uuid(1)]: 6000 }, { narrow: true, finer: true }).changes[0].after.step, "0.0001", "the step never reaches zero");
});

test("saving a group is explicit: tests only, or the complete mix with globals; fresh, bounded, never overlapping", () => {
  const withGlobals = draft([lora(9, { weight: "0.5" }), lora(8, { weight: "0.25", enabled: false })]);
  const candidate = { id: "C03", weights };
  const tests = groupFromCandidate(withGlobals, candidate, { includeGlobals: false });
  assert.equal(tests.ok, true); assert.equal(tests.scope, "tests"); assert.deepEqual(tests.members.map((member) => [member.name, member.weight, member.precision]), [["L1", 0.8, 4], ["L2", -0.4, 4], ["L3", 0.5, 4]]);
  assert.match(tests.name, /仅测试成员/); assert.equal(tests.presetPrompt, "");
  const complete = groupFromCandidate(withGlobals, candidate, { includeGlobals: true });
  assert.equal(complete.scope, "complete"); assert.deepEqual(complete.members.map((member) => member.name), ["L9", "L8", "L1", "L2", "L3"]);
  assert.equal(complete.members[1].enabled, false, "a disabled global stays disabled and auditable in the saved mix");
  assert.ok(complete.members.every((member) => member.groupId === undefined && member.instanceId));
  assert.equal(withGlobals.baseline.loras[0].weight, "0.5", "the draft is untouched");
  const crowded = draft(Array.from({ length: 14 }, (_, i) => lora(100 + i)));
  assert.equal(groupFromCandidate(crowded, candidate, { includeGlobals: true }).ok, false);
  assert.equal(groupFromCandidate(crowded, candidate, { includeGlobals: false }).ok, true, "a tests-only group never inherits the globals' size");
  const overlapping = draft([lora(1, { weight: "0.5" })]);
  assert.equal(groupFromCandidate(overlapping, candidate, { includeGlobals: true }).ok, false);
});

test("favorites are keyed by the stable candidate key and live in the draft's search block", () => {
  let value = draft();
  assert.equal(isFavorite(value, "k1"), false);
  value = toggleFavorite(value, "k1"); value = toggleFavorite(value, "k2");
  assert.deepEqual(value.mix.favorites, ["k1", "k2"]); assert.equal(isFavorite(value, "k2"), true);
  assert.deepEqual(toggleFavorite(value, "k1").mix.favorites, ["k2"]);
  assert.equal(validLoraMix(value.mix), true);
  assert.equal(validLoraMix({ ...value.mix, favorites: [3] }), false);
  assert.equal(validLoraMix((({ favorites, ...rest }) => rest)(value.mix)), true, "a draft from before favorites existed is still valid");
});

import { interpretImageMetadata as parseImageMetadata } from "../src/image-metadata.js";
test("the image reader recognises a search-round cell and a candidate-grid PNG, and ignores unknown versions", () => {
  const cell = { engine: "Anima", prompt: "p", seed: 1, xyz_plot: { experiment_id: "a".repeat(32), cell_id: "2:0:0:0" },
    lora_mix_search: { schema_version: 1, session_id: "b".repeat(32), round: 2, candidate_id: "C03", displacement: "0.4", context_hash: "c".repeat(64), global_loras: [{}, {}], test_loras: [{ name: "L1", weight: "0.8", locked: false }, { name: "L3", weight: "0.5", locked: true }] } };
  const info = parseImageMetadata({ parameters: JSON.stringify(cell) });
  assert.deepEqual(info.xyzCell, { experimentId: "a".repeat(32), cellId: "2:0:0:0" });
  assert.equal(info.mixCell.candidateId, "C03"); assert.equal(info.mixCell.round, 2); assert.equal(info.mixCell.globalCount, 2);
  assert.deepEqual(info.mixCell.testLoras, [{ name: "L1", weight: "0.8", locked: false }, { name: "L3", weight: "0.5", locked: true }]);
  const future = parseImageMetadata({ parameters: JSON.stringify({ ...cell, lora_mix_search: { ...cell.lora_mix_search, schema_version: 2 } }) });
  assert.equal(future.mixCell, null, "an unknown search record is never interpreted");
  assert.equal(parseImageMetadata({ parameters: JSON.stringify({ engine: "Anima", prompt: "p", seed: 1 }) }).mixCell, null);
  const grid = parseImageMetadata({ xyz_plot: JSON.stringify({ schema_version: 1, kind: "xyz_plot_grid", experiment_id: "a".repeat(32), recipe: { mode: "original", canvas: { width: 10, height: 10 } }, cells: [],
    search: { session_id: "b".repeat(32), round: 2, context_hash: "c".repeat(64), columns: 4, rows: 4, candidates_count: 16 } }) });
  assert.deepEqual(grid.xyzGrid.search, { sessionId: "b".repeat(32), round: 2, columns: 4, rows: 4, candidates: 16 });
  const plainGrid = parseImageMetadata({ xyz_plot: JSON.stringify({ schema_version: 1, kind: "xyz_plot_grid", experiment_id: "a".repeat(32), recipe: {}, cells: [] }) });
  assert.equal(plainGrid.xyzGrid.search, null);
});
