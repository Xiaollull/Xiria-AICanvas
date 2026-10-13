import test from "node:test";
import assert from "node:assert/strict";
import { createLoraMixDraft, defaultMix, emptyLoraMixDrafts, inspectLoraMixDrafts, loraMixRecipe, validLoraMix, withLoraMixDraft } from "../src/lora-mix-draft.js";
import { applySelection, budgetOf, candidateRaw, candidateWeights, memberAnalysis, mixIssues, mixRulesKey, newCandidate, parseCandidate, roleConflicts } from "../src/lora-mix-model.js";
import { proposeCandidates } from "../src/lora-mix-sampler.js";
import { parseUnits } from "../src/lora-mix-rules.js";

// QPlan2/P2 Phases A/B: the independent search draft and its derived verdicts. Synthetic references only.
const uuid = (n) => `00000000-0000-4000-8000-${String(n).padStart(12, "0")}`;
const hash = (n) => n.toString(16).padStart(2, "0").repeat(32);
const lora = (n, extra = {}) => ({ value: `style/l${n}.safetensors`, name: `L${n}`, category: "style", weight: 0.6, precision: 4, enabled: true, identitySchema: 1, instanceId: uuid(n), contentHash: hash(n), contentRevision: 2, ...extra });
const baseline = (engine = "Anima", loras = []) => ({ parameters: { engine, diffusion_model: "d.safetensors", text_encoder: "t.safetensors", vae: "v.safetensors", prompt: "raw", negative_prompt: "", steps: 20, cfg: 4,
  width: 512, height: 512, seed: "7", sampler: "euler", scheduler: "normal", guidance: "none", pag: { scale: 0.3, applied_layers: "mid" }, postprocess_chain: [] },
  loras, group_snapshots: [], prompt_without_groups: "raw", group_prompt: "", seed_mode: "fixed", library_revision: 0 });
const rule = (base, min, max, step = "0.1", preference = "balanced") => ({ locked: false, base, min, max, step, preference });
function draftWith(members, extra = {}, loras = []) {
  const draft = createLoraMixDraft(baseline("Anima", loras));
  draft.mix.members = members.map(([ref, memberRule]) => ({ ref, rule: memberRule }));
  Object.assign(draft.mix, extra);
  return draft;
}
const A = lora(11), B = lora(12);
const members = () => [[A, rule("0.6", "0.2", "1.0", "0.1", "max")], [B, rule("-0.2", "-0.8", "0.6", "0.1", "min")], [lora(13), { locked: true, base: "0.5" }]];

test("A1 the search draft is independent, XYZ-shaped and survives a container round trip", () => {
  const source = baseline(), before = structuredClone(source), draft = createLoraMixDraft(source);
  assert.deepEqual(source, before, "capturing a baseline never edits its source");
  assert.equal(draft.kind, "lora_mix_draft"); assert.equal(draft.engine, "Anima"); assert.deepEqual(draft.axes, { x: null, y: null, z: null });
  assert.equal(validLoraMix(draft.mix), true);
  assert.equal(draft.mix.budget.preset, "32"); assert.equal(draft.mix.include_baseline, true); assert.equal(draft.mix.sampling_seed, "1");
  assert.notEqual(draft.mix.sampling_seed, draft.baseline.parameters.seed, "image seed and sampling seed are separate");
  draft.baseline.parameters.prompt = "search only";
  assert.equal(source.parameters.prompt, "raw");
  const container = withLoraMixDraft(emptyLoraMixDrafts(), "Anima", draft);
  const inspected = inspectLoraMixDrafts(structuredClone(container), "lora_mix_search");
  assert.equal(inspected.fatal, false); assert.deepEqual(inspected.container, container);
  assert.equal(inspectLoraMixDrafts(undefined, "basic").fatal, false);
  assert.throws(() => withLoraMixDraft(emptyLoraMixDrafts(), "SD", draft), /mismatch/);
  assert.throws(() => withLoraMixDraft(emptyLoraMixDrafts(), "Anima", { ...draft, kind: "xyz_draft" }), /mismatch/);
});

test("A2 unknown versions and malformed drafts keep their raw data and stop autosave instead of being repaired", () => {
  const good = withLoraMixDraft(emptyLoraMixDrafts(), "Anima", createLoraMixDraft(baseline()));
  const future = structuredClone(good); future.by_engine.Anima.mix.version = 2;
  const noMix = structuredClone(good); delete noMix.by_engine.Anima.mix;
  const badCandidate = structuredClone(good); badCandidate.by_engine.Anima.mix.candidates = [{ id: "C01", source: "weird", raw: {} }];
  const badBaseline = structuredClone(good); badBaseline.by_engine.Anima.baseline.parameters.engine = "SD";
  const wrongKind = structuredClone(good); wrongKind.by_engine.Anima.kind = "xyz_draft";
  const futureContainer = { ...structuredClone(good), schema_version: 2 };
  for (const raw of [future, noMix, badCandidate, badBaseline, wrongKind, futureContainer, [], "x"]) {
    const result = inspectLoraMixDrafts(raw, "lora_mix_search");
    assert.equal(result.fatal, true); assert.deepEqual(result.raw, raw); assert.deepEqual(result.container, emptyLoraMixDrafts()); assert.match(result.error, /不会从基础生图自动重建/);
  }
  // An XYZ container is not a search container.
  assert.equal(inspectLoraMixDrafts({ schema_version: 1, by_engine: { Anima: { kind: "xyz_draft" } } }, "lora_mix_search").fatal, true);
});

test("A3 the wire recipe is the XYZ projection plus the search block, without UI-only sources", () => {
  const draft = draftWith(members());
  draft.baseline.seed_source = { schema_version: 1, mode: "increment" }; draft.repeat_count = 3;
  const recipe = loraMixRecipe(draft);
  assert.equal(recipe.baseline.seed_source, undefined); assert.deepEqual(recipe.sample_seeds, ["7", "8", "9"]);
  assert.equal(recipe.kind, "xyz_draft"); assert.deepEqual(recipe.mix, draft.mix);
  assert.equal(draft.kind, "lora_mix_draft", "the stored draft keeps its own kind");
});

test("B1 X1 selection keeps existing rules, orders by pick, drops unpicked, never duplicates", () => {
  const draft = draftWith(members()), mix = draft.mix;
  const next = applySelection(mix, [{ ...B, weight: 0.9 }, lora(14, { weight: 0.75 }), { ...B }]);
  assert.deepEqual(next.members.map((member) => member.ref.instanceId), [uuid(12), uuid(14)]);
  assert.deepEqual(next.members[0].rule, mix.members[1].rule, "a kept member keeps its X2 rule, not the picked weight");
  assert.equal(next.members[1].rule.base, "0.75"); assert.equal(next.members[1].rule.locked, false);
  assert.deepEqual(Object.keys(next.members[1].ref).sort(), ["category", "contentHash", "contentRevision", "identitySchema", "instanceId", "name", "value"]);
  assert.equal(applySelection(mix, []).members.length, 0);
});

test("B2 budget: presets, custom, seeds share the budget, nothing is truncated or silently adjusted", () => {
  const draft = draftWith(members());
  assert.deepEqual(budgetOf(draft), { images: 32, seeds: 1, slots: 32, leftover: 0, errors: [] });
  draft.repeat_count = 3;
  assert.deepEqual(budgetOf(draft), { images: 32, seeds: 3, slots: 10, leftover: 2, errors: [] });
  draft.mix.budget = { preset: "custom", custom: "40" }; draft.repeat_count = 8;
  assert.equal(budgetOf(draft).slots, 5);
  for (const custom of ["", "0", "-4", "3.5", "abc", "12345678"]) { draft.mix.budget = { preset: "custom", custom }; assert.ok(budgetOf(draft).errors.length, custom); }
  draft.mix.budget = { preset: "custom", custom: "5" }; draft.repeat_count = 8;
  assert.match(budgetOf(draft).errors[0], /少于样本数/);
  draft.mix.budget = { preset: "custom", custom: "2000" }; draft.repeat_count = 1;
  assert.match(budgetOf(draft).errors[0], /超出现有执行上限/);
  draft.mix.budget = { preset: "custom", custom: "1030" }; draft.repeat_count = 5;
  assert.match(budgetOf(draft).errors[0], /超出现有执行上限/, "total cells beyond 1024 is refused, never reduced");
  draft.mix.budget = { preset: "128", custom: "" }; draft.repeat_count = 33;
  assert.match(budgetOf(draft).errors[0], /1–32/);
});

test("B3 role conflicts: the same instance/path always, equal content only when both sides could be fused", () => {
  const draft = draftWith(members()), analysis = memberAnalysis(draft);
  assert.deepEqual(roleConflicts([], draft.mix.members, analysis.rows), []);
  assert.deepEqual(roleConflicts([{ ...A, weight: 0 }], draft.mix.members, analysis.rows).map((item) => item.kind), ["instance"], "a zero-weight global still cannot share an instance");
  assert.deepEqual(roleConflicts([lora(99, { value: A.value })], draft.mix.members, analysis.rows).map((item) => item.kind), ["path"]);
  assert.deepEqual(roleConflicts([lora(99, { contentHash: A.contentHash })], draft.mix.members, analysis.rows).map((item) => [item.kind, item.index]), [["content", 0]]);
  assert.deepEqual(roleConflicts([lora(99, { contentHash: A.contentHash, enabled: false })], draft.mix.members, analysis.rows), [], "a disabled global is never fused");
  assert.deepEqual(roleConflicts([lora(99, { contentHash: A.contentHash, weight: 0 })], draft.mix.members, analysis.rows), []);
  // A locked zero-weight test member can never be fused, so an equal-content global is no conflict.
  const dormant = draftWith([[lora(21), { locked: true, base: "0" }]]);
  assert.deepEqual(roleConflicts([lora(99, { contentHash: hash(21) })], dormant.mix.members, memberAnalysis(dormant).rows), []);
  const issues = mixIssues(draftWith(members(), {}, [lora(99, { contentHash: A.contentHash })])).issues;
  assert.ok(issues.some((issue) => issue.scope === "role" && issue.kind === "content" && /不能重叠/.test(issue.message)));
});

test("B4 issues: missing members, rule errors by field, displacement bound, concurrency and stale identity are explicit", () => {
  assert.deepEqual(mixIssues(draftWith([])).issues.map((issue) => issue.scope), ["members"]);
  const bad = draftWith([[A, rule("0.6", "0.8", "1.0")], [{ ...B, contentHash: "zz" }, rule("0.6", "0.2", "1", "0")]]);
  const issues = mixIssues(bad).issues;
  assert.ok(issues.some((issue) => issue.scope === "rule" && issue.member === 0 && issue.field === "base"));
  assert.ok(issues.some((issue) => issue.scope === "rule" && issue.member === 1 && issue.field === "step"));
  assert.ok(issues.some((issue) => issue.scope === "members" && issue.member === 1 && /强实例/.test(issue.message)));
  const good = draftWith(members());
  assert.equal(mixIssues(good).issues.some((issue) => issue.scope === "rule" || issue.scope === "role"), false);
  good.mix.max_displacement = "1.3";
  assert.ok(mixIssues(good).issues.some((issue) => issue.scope === "displacement" && /不会自动截断/.test(issue.message)));
  good.mix.max_displacement = "0.6"; good.requested_concurrency = 11;
  assert.ok(mixIssues(good).issues.some((issue) => issue.scope === "budget" && /请求并发/.test(issue.message)));
  good.requested_concurrency = 1;
  assert.deepEqual(mixIssues(good).issues.map((issue) => issue.scope), ["candidates"], "only the missing preview remains");
  assert.equal(mixIssues(good).valid, false);
  assert.equal(mixIssues(good).space.count, 77n);
});

test("B5 candidates keep raw input, flag off-sequence/duplicate/over-bound rows, and never clamp", () => {
  const draft = draftWith(members()), analysis = memberAnalysis(draft);
  const row = (id, a, b, source = "manual") => ({ id, source, raw: { [uuid(11)]: a, [uuid(12)]: b } });
  draft.mix.candidates = [row("C01", "0.6", "-0.2", "baseline"), row("C02", "0.65", "-0.2"), row("C03", "0.6", "-0.2"), row("C04", "1.0", "0.6"), row("C05", "0.", "-0.2"), row("C06", "0.8", "-0.4"), { id: "C07", source: "manual", raw: { [uuid(11)]: "0.7" } }];
  const report = mixIssues(draft);
  const problems = Object.fromEntries(report.candidates.map((entry) => [entry.candidate.id, entry.problems.map((problem) => problem.message)]));
  assert.deepEqual(problems.C01, []);
  assert.match(problems.C02.join(), /候选序列/);
  assert.match(problems.C03.join(), /与 C01 完全重复/);
  assert.match(problems.C04.join(), /总位移 1.2 超过上限 0.6/);
  assert.match(problems.C05.join(), /需要普通十进制数/);
  assert.deepEqual(problems.C06, [], "0.2 + 0.2 = 0.4 <= 0.6");
  assert.match(problems.C07.join(), /缺少权重/);
  assert.equal(draft.mix.candidates[1].raw[uuid(11)], "0.65", "raw input is retained verbatim");
  assert.equal(report.candidates[0].displacement, 0); assert.equal(report.candidates[3].displacement, 12000);
  assert.deepEqual(candidateWeights(draft, report.candidates[5].parsed, analysis), { [uuid(11)]: 8000, [uuid(12)]: -4000, [uuid(13)]: 5000 }, "the locked member rides at its base");
  assert.equal(report.valid, false);
});

test("B6 the combined active-LoRA limit counts enabled non-zero globals plus non-zero test members", () => {
  const globals = Array.from({ length: 15 }, (_, i) => lora(100 + i));
  const draft = draftWith(members().slice(0, 2), {}, globals), analysis = memberAnalysis(draft);
  draft.mix.candidates = [{ id: "C01", source: "manual", raw: { [uuid(11)]: "0.6", [uuid(12)]: "-0.2" } }, { id: "C02", source: "manual", raw: { [uuid(11)]: "0.6", [uuid(12)]: "0" } }];
  draft.mix.max_displacement = "1.2";
  const report = mixIssues(draft);
  assert.ok(report.candidates[0].problems.some((problem) => /超过 16 上限/.test(problem.message)));
  assert.deepEqual(report.candidates[1].problems, [], "a zero-weight member is auditable but not active");
  assert.equal(parseCandidate(draft.mix.candidates[1], analysis).units.join(), "6000,0");
  const disabled = draftWith(members().slice(0, 2), {}, [...globals.slice(0, 14), lora(200, { enabled: false })]);
  disabled.mix.candidates = [draft.mix.candidates[0]];
  assert.deepEqual(mixIssues(disabled).candidates[0].problems, []);
});

test("B7 a proposal turns into editable candidates and a valid, rules-keyed preview", () => {
  const draft = draftWith(members()), analysis = memberAnalysis(draft), limit = parseUnits(draft.mix.max_displacement).units;
  const proposal = proposeCandidates({ free: analysis.free, limit, count: 9, seed: draft.mix.sampling_seed, cursor: draft.mix.cursor });
  let mix = { ...draft.mix };
  for (const entry of proposal.vectors) { mix = { ...mix, candidates: [...mix.candidates, newCandidate(mix, analysis, entry, entry.source)], next_id: mix.next_id + 1 }; }
  draft.mix = mix;
  assert.deepEqual(draft.mix.candidates.map((candidate) => candidate.id), ["C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09"]);
  assert.equal(draft.mix.candidates[0].source, "baseline");
  assert.deepEqual(draft.mix.candidates[0].raw, candidateRaw(analysis, [6000, -2000]));
  const report = mixIssues(draft);
  assert.deepEqual(report.issues, []); assert.equal(report.valid, true); assert.equal(report.expectedOutput, 9);
  const key = mixRulesKey(draft);
  draft.mix.members[0].rule.step = "0.05";
  assert.notEqual(mixRulesKey(draft), key);
  draft.mix.candidates = [];
  assert.equal(validLoraMix(defaultMix()), true);
});

import { addManualCandidate, clearCandidates, drawSamplingSeed, editCandidateValue, generateCandidates, removeCandidate, togglePin } from "../src/lora-mix-generate.js";

test("B8 generation fills an editable preview, keeps pinned rows, and a repeat is identical", () => {
  const draft = draftWith(members().slice(0, 2));
  draft.mix.budget = { preset: "16", custom: "" };
  const first = generateCandidates(draft);
  assert.equal(first.ok, true); assert.equal(first.draft.mix.candidates.length, 16); assert.equal(first.draft.mix.cursor, 1); assert.notEqual(first.draft.mix.proposed_key, "");
  assert.equal(first.draft.mix.candidates[0].source, "baseline");
  assert.deepEqual(generateCandidates(draft), first, "same draft => same preview (no hidden randomness)");
  assert.deepEqual(draft.mix.candidates, [], "generation returns a new draft; the input is untouched");
  assert.equal(mixIssues(first.draft).valid, true);
  // Pin two rows, redraw: they survive with their ids; the rest changes with the advanced cursor.
  let pinned = togglePin(togglePin(first.draft, "C04"), "C07");
  const again = generateCandidates(pinned);
  const ids = again.draft.mix.candidates.map((candidate) => candidate.id);
  assert.ok(ids.includes("C04") && ids.includes("C07"));
  assert.equal(again.draft.mix.candidates.length, 16);
  assert.equal(again.draft.mix.candidates.find((candidate) => candidate.id === "C04").pinned, true);
  assert.deepEqual(again.draft.mix.candidates.find((candidate) => candidate.id === "C04").raw, first.draft.mix.candidates.find((candidate) => candidate.id === "C04").raw);
  assert.ok(new Set(again.draft.mix.candidates.map((candidate) => candidate.id)).size === 16);
  assert.ok(Number(ids.at(-1).slice(1)) > 16, "new rows continue the id sequence instead of reusing ids");
  assert.equal(again.draft.mix.candidates[0].id, "C01", "an unedited baseline row keeps its id across a redraw");
  assert.equal(again.draft.mix.candidates[0].source, "baseline");
  // An edited baseline row is no longer the baseline vector: it is replaced, and a fresh baseline leads.
  const edited = editCandidateValue(first.draft, "C01", uuid(11), "0.8");
  const redrawn = generateCandidates(edited);
  assert.equal(redrawn.draft.mix.candidates[0].source, "baseline");
  assert.notEqual(redrawn.draft.mix.candidates[0].id, "C01");
  assert.equal(redrawn.draft.mix.candidates.filter((candidate) => candidate.source === "baseline").length, 1);
  // Excluding what is tested/reserved yields a disjoint continuation.
  const tested = new Set(mixIssues(first.draft).candidates.map((entry) => entry.parsed.key));
  const next = generateCandidates({ ...first.draft, mix: { ...first.draft.mix, candidates: [], budget: { preset: "16", custom: "" } } }, { exclude: tested });
  assert.equal(next.ok, true); assert.ok(next.draft.mix.candidates.length > 0);
  assert.ok(mixIssues(next.draft).candidates.every((entry) => !tested.has(entry.parsed.key)));
});

test("B9 generation refuses while rules/budget/role are invalid, and reports exhaustion honestly", () => {
  const bad = draftWith([[A, rule("0.6", "0.8", "1.0")]]);
  const refused = generateCandidates(bad);
  assert.equal(refused.ok, false); assert.ok(refused.issues.every((issue) => ["members", "rule", "role", "displacement", "budget"].includes(issue.scope)));
  const tiny = draftWith([[A, rule("0", "-0.1", "0.1")], [B, rule("0", "-0.1", "0.1")]]);
  tiny.mix.max_displacement = "0.2";
  const result = generateCandidates(tiny);
  assert.equal(result.ok, true); assert.equal(result.proposal.exhausted, true); assert.match(result.note, /已穷尽.*不会用重复项补足/);
  assert.equal(result.draft.mix.candidates.length, 9, "9 legal vectors, never padded to 32");
  const locked = draftWith([[A, { locked: true, base: "0.4" }], [B, { locked: true, base: "-0.2" }]]);
  const one = generateCandidates(locked);
  assert.equal(one.ok, true); assert.equal(one.allLocked, true); assert.deepEqual(one.draft.mix.candidates, []);
  assert.equal(mixIssues(one.draft).valid, true, "an all-locked rule set is one profile and needs no preview rows");
  assert.equal(mixIssues(one.draft).analysis.dmax, 0);
});

test("B10 preview edits keep raw text, recalculate verdicts and never clamp", () => {
  let draft = generateCandidates(draftWith(members().slice(0, 2))).draft;
  assert.equal(mixIssues(draft).valid, true);
  draft = editCandidateValue(draft, "C02", uuid(11), "0.6");
  assert.equal(draft.mix.candidates[1].source, "manual", "an edited sample becomes a manual row");
  assert.equal(draft.mix.candidates[1].raw[uuid(11)], "0.6");
  draft = editCandidateValue(draft, "C02", uuid(11), "0.65");
  assert.equal(draft.mix.candidates[1].raw[uuid(11)], "0.65");
  assert.ok(mixIssues(draft).issues.some((issue) => issue.candidate === "C02" && /候选序列/.test(issue.message)));
  draft = editCandidateValue(draft, "C01", uuid(11), "0.7");
  assert.equal(draft.mix.candidates[0].source, "baseline", "the baseline marker is a role, not a claim about the values");
  const added = addManualCandidate(draft);
  assert.equal(added.ok, true); assert.equal(added.draft.mix.candidates.at(-1).source, "manual");
  assert.deepEqual(mixIssues(added.draft).candidates.at(-1).problems, [], "the baseline row was edited, so a fresh baseline-valued row is not a duplicate");
  const twice = addManualCandidate(added.draft);
  assert.ok(mixIssues(twice.draft).issues.some((issue) => issue.candidate === twice.draft.mix.candidates.at(-1).id && /完全重复/.test(issue.message)));
  assert.equal(removeCandidate(added.draft, "C02").mix.candidates.some((candidate) => candidate.id === "C02"), false);
  assert.deepEqual(clearCandidates(added.draft).mix.candidates, []);
  assert.equal(addManualCandidate(draftWith([])).ok, false);
});

test("B11 a sampling seed is drawn only on request, as a uint64 decimal", () => {
  assert.equal(drawSamplingSeed((bytes) => { bytes[0] = 0xffffffff; bytes[1] = 0xffffffff; return bytes; }), "18446744073709551615");
  assert.equal(drawSamplingSeed((bytes) => { bytes[0] = 0; bytes[1] = 5; return bytes; }), "5");
  assert.ok(/^\d{1,20}$/.test(drawSamplingSeed()));
});
