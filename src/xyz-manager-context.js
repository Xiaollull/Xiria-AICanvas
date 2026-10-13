import { mountedEntryForGroups } from "./lora-groups.js";
import { editXYZMounts, xyzStrongLora } from "./xyz-sidebar-context.js";
import { recomposeXYZGroups } from "./xyz-prompt-source.js";
import { analyzeXYZGroupCopy, contentConflictMessage, introducedContentConflict } from "./xyz-group-resolution.js";

/** All group/mount transactions use one draft owner. No library or Basic writer exists here. */
export function updateXYZManagerMounts(draft, updater, { syncGroups = true } = {}) {
  const next = structuredClone(draft), before = next.baseline.loras;
  const mounts = typeof updater === "function" ? updater(structuredClone(before)) : updater;
  if (!Array.isArray(mounts) || mounts.length > 16) throw new Error("最多挂载 16 个 LoRA；原始基线未改动。");
  const identities = new Set();
  for (const item of mounts) {
    const previous = before.find((old) => (item.instanceId ? old.instanceId === item.instanceId : old.value === item.value)) || before.find((old) => old.value === item.value);
    if (!previous && !xyzStrongLora(item)) throw new Error("候选缺少强实例 / 内容修订。");
    const key = item.instanceId || item.value;
    if (identities.has(key)) throw new Error("同一实例不能重复挂载。"); identities.add(key);
    if (!Number.isFinite(Number(item.weight)) || item.weight === "" || Math.abs(Number(item.weight)) > 5) throw new Error("权重需为 −5 到 5。");
    if (previous && ["identitySchema", "instanceId", "contentHash", "contentRevision", "value"].some((field) => previous[field] !== item[field])) throw new Error("不能修改已保存挂载的身份。");
  }
  const repeated = introducedContentConflict(before, mounts);
  if (repeated) throw new Error(contentConflictMessage(repeated, next.baseline.group_snapshots));
  next.baseline.loras = structuredClone(mounts);
  if (syncGroups) next.baseline.group_snapshots = next.baseline.group_snapshots.map((group) => group.enabled
    ? { ...group, members: mounts.filter((item) => item.groupId === group.id).map(({ groupId, ...item }) => item) } : group);
  return recomposeXYZGroups(next);
}
export function updateXYZManagerGroups(draft, updater) {
  const next = structuredClone(draft), before = next.baseline.group_snapshots, mountsBefore = structuredClone(next.baseline.loras);
  const groups = typeof updater === "function" ? updater(structuredClone(before)) : updater;
  if (!Array.isArray(groups) || new Set(groups.map((group) => group.id)).size !== groups.length) throw new Error("组合快照 ID 不能重复。");
  const added = [];
  for (const group of groups) {
    const old = before.find((item) => item.id === group.id);
    if (group.enabled && !old?.enabled) {
      if (group.members.some((item) => !xyzStrongLora(item))) throw new Error("组合含未解析身份；不能按同名文件启用。");
      const members = group.members.map((item) => ({ ...item, groupId: group.id }));
      if (members.some((item) => item.weight === "" || !Number.isFinite(Number(item.weight)) || Math.abs(Number(item.weight)) > 5)) throw new Error("组合权重需为 −5 到 5；不会静默截断。");
      if (new Set(members.map((item) => item.instanceId)).size !== members.length || new Set(members.map((item) => item.value)).size !== members.length) throw new Error("组合包含重复挂载实例或路径。");
      if (members.some((item) => next.baseline.loras.some((mounted) => mounted.instanceId === item.instanceId && (mounted.contentHash !== item.contentHash || mounted.contentRevision !== item.contentRevision || mounted.value !== item.value)))) throw new Error("同一实例的保存内容修订不同；请明确卸载后再选择，不会自动替换快照。");
      if (members.some((item) => next.baseline.loras.some((mounted) => mounted.value === item.value && mounted.instanceId !== item.instanceId))) throw new Error("同路径的保存实例与组合实例不同；不会按路径重绑。");
      const kept = next.baseline.loras.filter((item) => item.groupId !== group.id && !members.some((member) => member.instanceId === item.instanceId));
      if (kept.length + members.length > 16) throw new Error("启用组合会超过 16 个挂载上限。");
      next.baseline.loras = [...kept, ...members]; added.push(group.id);
    } else if (!group.enabled && old?.enabled) next.baseline.loras = next.baseline.loras.filter((item) => item.groupId !== group.id);
  }
  const removed = new Set(before.filter((group) => !groups.some((item) => item.id === group.id)).map((group) => group.id));
  next.baseline.loras = next.baseline.loras.filter((item) => !removed.has(item.groupId));
  const repeated = introducedContentConflict(mountsBefore, next.baseline.loras);
  if (repeated) throw new Error(contentConflictMessage(repeated, groups));
  next.baseline.group_snapshots = structuredClone(groups);
  return recomposeXYZGroups(next, added);
}
export function mountXYZCatalogItem(draft, item) {
  const index = draft.baseline.loras.findIndex((entry) => entry.instanceId ? entry.instanceId === item.instanceId : entry.value === item.value);
  const next = editXYZMounts(draft, index >= 0 ? { kind: "remove", index } : { kind: "add", item });
  if (index < 0) next.baseline.loras[next.baseline.loras.length - 1] = mountedEntryForGroups(next.baseline.loras.at(-1), next.baseline.group_snapshots);
  const result = updateXYZManagerMounts(next, (mounts) => mounts);
  // The add happened before the shared validator ran, so compare with the draft the user started from.
  const repeated = introducedContentConflict(draft.baseline.loras, result.baseline.loras);
  if (repeated) throw new Error(contentConflictMessage(repeated, result.baseline.group_snapshots));
  return result;
}
/**
 * Single-choice picker commit: the axis target is exactly one LoRA. Clicking the current selection keeps
 * it; clicking a different item replaces it, so the temporary set never holds two members at once.
 */
export function replaceXYZCatalogSelection(draft, item) {
  const chosen = (entry) => entry.instanceId ? entry.instanceId === item.instanceId : entry.value === item.value;
  if (draft.baseline.loras.length === 1 && chosen(draft.baseline.loras[0])) return draft;
  const cleared = draft.baseline.loras.length ? updateXYZManagerMounts(draft, () => []) : draft;
  return mountXYZCatalogItem(cleared, item);
}
/**
 * Informational cross-axis marks: the `lora_weight` targets the OTHER two axes already hold, keyed by
 * instance id and by path so a picker can badge the card without ever touching another axis's state.
 */
export function xyzAxisLoraMarks(draft, axisKey) {
  const marks = {};
  for (const key of ["x", "y", "z"]) {
    if (key === axisKey) continue;
    const axis = draft.axes?.[key];
    if (!axis || axis.field !== "lora_weight") continue;
    const reference = axis.editor_source?.target_reference;
    for (const identity of [reference?.instanceId, axis.target, reference?.value]) {
      if (!identity) continue;
      const label = key.toUpperCase();
      if (marks[identity]?.includes(label)) continue;
      marks[identity] = [...(marks[identity] || []), label];
    }
  }
  return marks;
}
/**
 * Copy a library group as a private snapshot. The group must already carry strong identities
 * (the async resolver projects them first). Modes are explicit user choices: "enable" refuses any
 * collision, "replace" removes only the standalone mounts the analysis lists, "copy-only" stores a
 * disabled snapshot and mounts nothing. Group-owned mounts are never modified here.
 */
export function copyXYZManagerGroup(draft, group, { mode = "enable" } = {}) {
  if (!["enable", "replace", "copy-only"].includes(mode)) throw new Error("未知的复制方式。");
  if (group.members.length > 16 || group.members.some((item) => !xyzStrongLora(item))) throw new Error("组合有未解析身份或超过 16 项；原始基线未改动。");
  const analysis = analyzeXYZGroupCopy(draft, group);
  if (analysis.blockers.length) throw new Error(analysis.blockers[0].message);
  const next = structuredClone(draft);
  next.baseline.group_snapshots.push({ ...structuredClone(group), enabled: false, collecting: false });
  if (mode === "copy-only") return updateXYZManagerGroups(next, (groups) => groups);
  if (analysis.enableBlockers.length) throw new Error(analysis.enableBlockers[0].message);
  if (mode === "enable" && analysis.conflicts.length) {
    const first = analysis.conflicts[0];
    throw new Error(`已挂载「${first.existing.name || first.existing.value}」与组合成员冲突；请明确选择「替换并启用」或「仅复制」。`);
  }
  if (mode === "replace") {
    if (!analysis.conflicts.length) throw new Error("没有可替换的独立挂载。");
    const removed = new Set(analysis.conflicts.map((item) => item.existing.instanceId));
    next.baseline.loras = next.baseline.loras.filter((item) => !removed.has(item.instanceId));
  }
  return updateXYZManagerGroups(next, (groups) => groups.map((item) => item.id === group.id ? { ...item, enabled: true } : item));
}
