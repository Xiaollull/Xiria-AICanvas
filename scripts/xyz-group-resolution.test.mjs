import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createXYZDraft, inspectXYZDrafts } from "../src/xyz-draft.js";
import { captureXYZPromptSource } from "../src/xyz-prompt-source.js";
import { copyXYZManagerGroup, updateXYZManagerGroups, updateXYZManagerMounts, mountXYZCatalogItem } from "../src/xyz-manager-context.js";
import { analyzeXYZGroupCopy, contentConflicts, explainResolution, groupResolutionReference, projectResolvedGroup, projectionSignature, resolveGroupMembers } from "../src/xyz-group-resolution.js";
import { loraNotice } from "../src/lora-notice.js";

// QPlan2/P2.1 Phase A fixtures. Synthetic names only; nothing here touches a real library or group.
const uuid = (n) => `00000000-0000-4000-8000-${String(n).padStart(12, "0")}`;
const hash = (n) => n.toString(16).padStart(2, "0").repeat(32);
const legacy = (i, extra = {}) => ({ value: `shared:5738b73b9323/style/nk-${i}.safetensors`, name: `NK ${i}`, category: "style", weight: 0.5 + i / 100, precision: i % 2 ? 4 : 2, enabled: true, ...extra });
const sharedGroup = { id: "g-nkstyle", name: "NKStyleV2", enabled: false, collecting: false, presetPrompt: "nk style words",
  members: Array.from({ length: 8 }, (_, i) => legacy(i + 1, i === 3 ? { enabled: false } : i === 5 ? { weight: 0 } : i === 6 ? { weight: -1.2345 } : {})) };
const verdictFor = (member, n, extra = {}) => ({ ...member, status: "resolved", value: member.value, instanceId: uuid(n), contentHash: hash(n), contentRevision: 2,
  kind: "current", originalValue: member.value, sourceId: uuid(n), auditKind: "recorded-path", candidates: [], promptRewriteAllowed: false, ...extra });
const entriesFor = (group, offset = 10, edit = () => ({})) => group.members.map((member, index) => ({ index, verdict: verdictFor(member, offset + index, edit(member, index)) }));
const STRICT = new Set(["value", "weight", "precision", "enabled", "identitySchema", "instanceId", "contentHash", "contentRevision", "originalReference", "name", "category", "groupId", "missing"]);

const strong = (n, extra = {}) => ({ value: `local/${n}.safetensors`, name: `Local ${n}`, category: "style", weight: 0.7, precision: 4, enabled: true,
  identitySchema: 1, instanceId: uuid(n), contentHash: hash(n), contentRevision: 3, ...extra });
const baseline = (loras = []) => ({ parameters: { engine: "Anima", diffusion_model: "d.safetensors", text_encoder: "t.safetensors", vae: "v.safetensors", prompt: "raw", negative_prompt: "", steps: 20, cfg: 4,
  width: 512, height: 512, seed: "1", sampler: "euler", scheduler: "normal", guidance: "none", pag: { scale: 0.3, applied_layers: "mid" }, postprocess_chain: [] },
  loras, group_snapshots: [], prompt_without_groups: "raw", group_prompt: "", seed_mode: "fixed", library_revision: 0 });
const fresh = (loras = []) => createXYZDraft(captureXYZPromptSource(baseline(loras), { positive: "raw", negative: "", records: [], active: { positive: [], negative: [] } }));

test("A1 binding-only references keep identity fields and omit the service's disabled filter", () => {
  const reference = groupResolutionReference({ ...legacy(1, { enabled: false }), ...strong(1), weight: 9 });
  assert.equal(Object.hasOwn(reference, "enabled"), false);
  assert.deepEqual(Object.keys(reference).sort(), ["contentHash", "contentRevision", "identitySchema", "instanceId", "value"]);
  assert.deepEqual(groupResolutionReference(legacy(2)), { value: legacy(2).value });
});

test("A2 legacy shared eight-member group projects without dropping members, changing weights or leaking verdict fields", () => {
  const original = structuredClone(sharedGroup);
  const projection = projectResolvedGroup(sharedGroup, { entries: entriesFor(sharedGroup) });
  assert.equal(projection.ok, true);
  assert.deepEqual(sharedGroup, original, "source library group is never mutated");
  assert.equal(projection.group.members.length, 8);
  projection.group.members.forEach((member, index) => {
    const source = sharedGroup.members[index];
    for (const field of ["value", "name", "category", "weight", "precision", "enabled"]) assert.equal(member[field], source[field], `${field} #${index}`);
    assert.equal(member.identitySchema, 1);
    assert.equal(member.instanceId, uuid(10 + index));
    assert.equal(member.contentHash, hash(10 + index));
    assert.equal(member.contentRevision, 2);
    assert.ok([...Object.keys(member)].every((key) => STRICT.has(key)), `strict LoraReference keys only: ${Object.keys(member)}`);
  });
  assert.equal(projection.group.members[3].enabled, false);
  assert.equal(projection.group.members[5].weight, 0);
  assert.equal(projection.group.presetPrompt, "nk style words");
  assert.ok(projection.bindings.every((item) => item.kind === "bound" && item.hadHash === false));
  const draft = copyXYZManagerGroup(fresh(), projection.group);
  assert.equal(draft.baseline.loras.length, 8);
  assert.deepEqual(draft.baseline.loras.map((item) => item.weight), sharedGroup.members.map((item) => item.weight));
  assert.equal(inspectXYZDrafts({ schema_version: 1, by_engine: { Anima: draft } }).fatal, false);
  assert.match(draft.baseline.parameters.prompt, /nk style words/);
});

test("A3 an untouched-legacy fallback verdict can never authorise a snapshot", () => {
  const entries = entriesFor(sharedGroup);
  entries[2] = { index: 2, verdict: { ...sharedGroup.members[2], status: "unresolved", promptRewriteAllowed: false } };
  const projection = projectResolvedGroup(sharedGroup, { entries });
  assert.equal(projection.ok, false);
  assert.equal(projection.group, null);
  assert.deepEqual(projection.issues.map((item) => [item.index, item.code]), [[2, "unresolved"]]);
  assert.match(projection.issues[0].message, /没有登记/);
  assert.throws(() => copyXYZManagerGroup(fresh(), sharedGroup), /未解析身份/);
});

test("A4 service refusals become per-member guidance and every offender is reported", async () => {
  const calls = [];
  const resolve = async (engine, references) => {
    calls.push({ engine, references });
    const index = calls.length - 1;
    if (index === 1) throw Object.assign(new Error("LoRA reference is ambiguous-path; choose an explicit target"), { status: 422, resolution: { status: "ambiguous-path" } });
    if (index === 4) throw Object.assign(new Error("LoRA instance or bytes changed; explicit repair required"), { status: 422 });
    return { revision: 7, references: [verdictFor(references[0], 20 + index)], loras: [] };
  };
  const result = await resolveGroupMembers("Anima", sharedGroup, { resolve });
  assert.equal(calls.length, 8);
  assert.ok(calls.every((call) => call.engine === "Anima" && call.references.length === 1 && !Object.hasOwn(call.references[0], "enabled")));
  assert.equal(result.stale, false);
  const projection = projectResolvedGroup(sharedGroup, result);
  assert.equal(projection.ok, false);
  assert.deepEqual(projection.issues.map((item) => item.index), [1, 4]);
  assert.match(projection.issues[0].message, /路径被复用|多个候选/);
  assert.match(projection.issues[1].message, /explicit repair/);
  assert.equal(explainResolution({ message: "boom" }), "无法连接资产库：boom");
});

test("A5 a publication revision change during resolution is stale, and abort stops without partial results", async () => {
  let revision = 3;
  const resolve = async (_engine, references) => ({ revision: revision++, references: [verdictFor(references[0], 30)], loras: [] });
  const result = await resolveGroupMembers("Anima", sharedGroup, { resolve });
  assert.equal(result.stale, true);
  assert.equal(projectResolvedGroup(sharedGroup, result).ok, false);
  const controller = new AbortController();
  await assert.rejects(resolveGroupMembers("Anima", sharedGroup, { signal: controller.signal, resolve: async (_e, references) => { controller.abort(); return { revision: 1, references: [verdictFor(references[0], 31)] }; } }), { name: "AbortError" });
});

test("A6 relocation records the original, never rebinding another instance or changed content", () => {
  const strongGroup = { ...sharedGroup, members: [strong(1), strong(2)] };
  const moved = projectResolvedGroup(strongGroup, { entries: [{ index: 0, verdict: verdictFor(strongGroup.members[0], 1, { value: "local/moved/1.safetensors", instanceId: uuid(1), contentHash: hash(1), contentRevision: 3 }) },
    { index: 1, verdict: verdictFor(strongGroup.members[1], 2, { contentRevision: 4, instanceId: uuid(2), contentHash: hash(2) }) }] });
  assert.equal(moved.ok, true);
  assert.deepEqual(moved.bindings.map((item) => item.kind), ["relocate", "revision"]);
  assert.equal(moved.group.members[0].value, "local/moved/1.safetensors");
  assert.deepEqual(moved.group.members[0].originalReference, { value: "local/1.safetensors", instanceId: uuid(1), contentHash: hash(1) });
  assert.equal(moved.group.members[1].contentRevision, 4);
  const other = projectResolvedGroup(strongGroup, { entries: [{ index: 0, verdict: verdictFor(strongGroup.members[0], 9) }, { index: 1, verdict: verdictFor(strongGroup.members[1], 2, { instanceId: uuid(2), contentHash: hash(8) }) }] });
  assert.deepEqual(other.issues.map((item) => [item.index, item.code]), [[0, "identity-changed"], [1, "content-changed"]]);
  const bad = projectResolvedGroup(sharedGroup, { entries: entriesFor(sharedGroup, 10, (_m, index) => index === 0 ? { contentHash: "xyz" } : {}) });
  assert.deepEqual(bad.issues.map((item) => [item.index, item.code]), [[0, "invalid-evidence"]]);
  assert.equal(projectionSignature(moved), projectionSignature(structuredClone(moved)));
});

test("B1 equal-content copies between NKStyleV2-like and NekoX-like groups require an explicit choice", () => {
  // Three of the shared members share bytes with already mounted local copies of another group.
  const nk = projectResolvedGroup(sharedGroup, { entries: entriesFor(sharedGroup, 10) }).group;
  const neko = { id: "g-neko", name: "NekoX V2", enabled: true, collecting: false, presetPrompt: "", members: [strong(41, { contentHash: nk.members[0].contentHash }), strong(42, { contentHash: nk.members[1].contentHash }), strong(43, { contentHash: nk.members[2].contentHash })] };
  let draft = fresh();
  draft.baseline.group_snapshots.push(neko);
  draft.baseline.loras = neko.members.map((item) => ({ ...item, groupId: neko.id }));
  const analysis = analyzeXYZGroupCopy(draft, nk);
  assert.deepEqual(analysis.actions, { "copy-only": true, enable: false, replace: false });
  assert.equal(analysis.conflicts.length, 0, "group-owned mounts are never replaceable");
  assert.deepEqual(analysis.enableBlockers.map((item) => item.code), ["content-owned", "content-owned", "content-owned"]);
  assert.match(analysis.enableBlockers[0].message, /NekoX V2/);
  assert.throws(() => copyXYZManagerGroup(draft, nk), /内容相同/);
  assert.throws(() => copyXYZManagerGroup(draft, nk, { mode: "replace" }), /内容相同/);
  const copied = copyXYZManagerGroup(draft, nk, { mode: "copy-only" });
  assert.equal(copied.baseline.loras.length, 3, "retaining the other group mounts nothing");
  assert.equal(copied.baseline.group_snapshots.length, 2);
  const snapshot = copied.baseline.group_snapshots.find((item) => item.id === nk.id);
  assert.equal(snapshot.enabled, false);
  assert.deepEqual(snapshot.members, nk.members);
  assert.equal(copied.baseline.parameters.prompt, draft.baseline.parameters.prompt, "a disabled copy adds no group text");
  // Explicit user action elsewhere removes the conflict; then the snapshot enables without repeated fusion.
  draft = updateXYZManagerGroups(copied, (groups) => groups.map((item) => item.id === neko.id ? { ...item, enabled: false } : item));
  assert.equal(draft.baseline.loras.length, 0);
  draft = updateXYZManagerGroups(draft, (groups) => groups.map((item) => item.id === nk.id ? { ...item, enabled: true } : item));
  assert.equal(draft.baseline.loras.length, 8);
  assert.throws(() => updateXYZManagerGroups(draft, (groups) => groups.map((item) => item.id === neko.id ? { ...item, enabled: true } : item)), /内容完全相同/);
});

test("B2 standalone collisions are replaced only by an explicit action that discloses and never sums weights", () => {
  const group = { id: "g-one", name: "One", enabled: false, collecting: false, presetPrompt: "", members: [strong(1, { weight: 0.3 }), strong(2, { weight: 0.4 })] };
  const sameInstance = strong(1, { weight: 0.9, precision: 2 }), equalCopy = strong(5, { contentHash: group.members[1].contentHash, weight: 0.8 }), keep = strong(7);
  const draft = fresh([sameInstance, equalCopy, keep]);
  const before = structuredClone(draft);
  const analysis = analyzeXYZGroupCopy(draft, group);
  assert.deepEqual(analysis.conflicts.map((item) => [item.kind, item.memberIndex, item.existing.instanceId]), [["same-instance", 0, uuid(1)], ["equal-content", 1, uuid(5)]]);
  assert.deepEqual(analysis.actions, { "copy-only": true, enable: false, replace: true });
  assert.deepEqual(analysis.total, { current: 3, afterEnable: 5, afterReplace: 3 });
  assert.throws(() => copyXYZManagerGroup(draft, group), /替换并启用/);
  assert.deepEqual(draft, before, "a refused copy is atomic");
  const next = copyXYZManagerGroup(draft, group, { mode: "replace" });
  assert.deepEqual(next.baseline.loras.map((item) => [item.instanceId, item.weight, item.groupId]), [[uuid(7), 0.7, undefined], [uuid(1), 0.3, "g-one"], [uuid(2), 0.4, "g-one"]]);
  assert.equal(next.baseline.group_snapshots[0].enabled, true);
  const only = copyXYZManagerGroup(draft, group, { mode: "copy-only" });
  assert.deepEqual(only.baseline.loras, before.baseline.loras);
  assert.equal(only.baseline.group_snapshots[0].enabled, false);
});

test("B3 caps, dormant copies, duplicate ids and in-group repeats are explicit", () => {
  const group = { id: "g-cap", name: "Cap", enabled: false, collecting: false, presetPrompt: "", members: [strong(1), strong(2)] };
  const crowded = fresh(Array.from({ length: 15 }, (_, i) => strong(30 + i)));
  const capped = analyzeXYZGroupCopy(crowded, group);
  assert.deepEqual(capped.enableBlockers.map((item) => item.code), ["limit"]);
  assert.equal(capped.actions["copy-only"], true);
  // Disabled or zero-weight equal-content copies are never fused, so they are not conflicts.
  const dormant = analyzeXYZGroupCopy(fresh([strong(9, { contentHash: hash(1), enabled: false }), strong(8, { contentHash: hash(2), weight: 0 })]), group);
  assert.equal(dormant.conflicts.length, 0);
  assert.equal(dormant.actions.enable, true);
  const twice = { ...group, members: [strong(1), strong(2, { contentHash: hash(1) })] };
  assert.deepEqual(analyzeXYZGroupCopy(fresh(), twice).enableBlockers.map((item) => item.code), ["content-in-group"]);
  const existing = copyXYZManagerGroup(fresh(), group);
  assert.deepEqual(analyzeXYZGroupCopy(existing, group).blockers.map((item) => item.code), ["snapshot-exists"]);
  assert.throws(() => copyXYZManagerGroup(existing, group), /已有独立快照/);
  assert.throws(() => copyXYZManagerGroup(fresh(), { ...group, members: [strong(1, { weight: 9 })] }), /−5 到 5/);
  assert.throws(() => copyXYZManagerGroup(fresh(), group, { mode: "bogus" }), /未知/);
});

test("B4 plain edits cannot create repeated fusion but an already-duplicated draft stays editable", () => {
  const a = strong(1), b = strong(2, { contentHash: a.contentHash });
  const draft = fresh([a, { ...b, weight: 0 }]);
  assert.equal(contentConflicts(draft.baseline.loras).length, 0);
  assert.throws(() => updateXYZManagerMounts(draft, (mounts) => mounts.map((item, i) => i === 1 ? { ...item, weight: 0.5 } : item)), /内容完全相同/);
  assert.throws(() => updateXYZManagerMounts(draft, (mounts) => mounts.map((item, i) => i === 1 ? { ...item, weight: 0.5 } : item)), (error) => error.message.includes("Local 1") && error.message.includes("Local 2"));
  assert.throws(() => mountXYZCatalogItem(fresh([a]), b), /内容完全相同/);
  const old = fresh([a, b]);
  const edited = updateXYZManagerMounts(old, (mounts) => mounts.map((item, i) => i === 0 ? { ...item, weight: 0.2 } : item));
  assert.equal(edited.baseline.loras[0].weight, 0.2, "pre-existing duplication is reported by the planner, not by every unrelated edit");
});

test("C1 notices take severity from the operation, never from wording", () => {
  assert.deepEqual(loraNotice("已删除组合", "success"), { kind: "success", title: "", detail: "已删除组合", diagnostics: [] });
  assert.equal(loraNotice("失败 error 超过上限").kind, "info");
  assert.equal(loraNotice("x", "bogus").kind, "info");
  assert.equal(loraNotice(""), null);
  assert.equal(loraNotice({ kind: "error", title: "T", detail: "D", diagnostics: [{ name: "n", message: "m" }, null, { message: 3 }] }).diagnostics.length, 1);
  assert.equal(loraNotice({ kind: "nope", detail: "D" }).kind, "info");
  assert.equal(loraNotice({}), null);
});

test("C2 the manager renders typed notices in their own slot, not as a grid-stealing alert", async () => {
  const modal = await readFile(new URL("../src/LoraManagerModal.jsx", import.meta.url), "utf8");
  assert.doesNotMatch(modal, /xyz-sidebar-error/);
  assert.match(modal, /<LoraManagerNotice notice=\{c\.notice\} onDismiss=\{c\.onDismissNotice\} \/>/);
  const notice = await readFile(new URL("../src/LoraManagerNotice.jsx", import.meta.url), "utf8");
  assert.match(notice, /role=\{value\.kind === "error" \? "alert" : "status"\}/);
  const css = await readFile(new URL("../src/styles.css", import.meta.url), "utf8");
  assert.match(css, /\.lora-browser \{[^}]*display: flex; flex-direction: column;/);
  assert.match(css, /\.lora-browser-content \{[^}]*grid-template-rows: auto minmax\(0, 1fr\);/);
  assert.match(css, /\.lora-manager-notice\.error \{[^}]*var\(--state-error-ink\)/);
});
