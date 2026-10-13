import { CONTENT_HASH, INSTANCE_ID } from "./lora-identity.js";
import { xyzStrongLora } from "./xyz-sidebar-context.js";

// Pure pieces of the shared/local library-group -> private XYZ snapshot transaction.
// Nothing here fetches, stores or normalizes a saved draft. The resolver verdict is only
// evidence: a member becomes a strong reference after its fields are validated, never by
// guessing a same-named or equal-hash copy.

export const XYZ_MOUNT_LIMIT = 16;
const IDENTITY_KEYS = ["identitySchema", "instanceId", "contentHash", "contentRevision", "originalReference"];
const label = (item) => item?.name || item?.value || "未命名 LoRA";

/** Same activity rule as backend/xyz_planner.checked_loras: disabled or zero weight is never fused. */
export const effectiveLora = (item) => item?.enabled !== false && Number(item?.weight) !== 0;

/** Equal-byte copies (different instances) that would both be fused. */
export function contentConflicts(list) {
  const active = (list || []).filter((item) => effectiveLora(item) && item.contentHash);
  const pairs = [];
  for (let i = 0; i < active.length; i += 1) for (let j = i + 1; j < active.length; j += 1) {
    const a = active[i], b = active[j];
    if (a.contentHash === b.contentHash && a.instanceId !== b.instanceId) pairs.push({ a, b, key: [a.instanceId, b.instanceId].sort().join("|") });
  }
  return pairs;
}

/** An edit may not CREATE repeated fusion; an old draft that already has it stays editable. */
export function introducedContentConflict(before, after) {
  const known = new Set(contentConflicts(before).map((pair) => pair.key));
  return contentConflicts(after).find((pair) => !known.has(pair.key)) || null;
}
/** Equal names are common between a shared and a local copy, so say which copy and who owns it. */
const rootOf = (item) => String(item.value || "").startsWith("shared:") ? "共享库" : "本地";
const originOf = (item, snapshots) => `${rootOf(item)} · ${item.groupId ? `组合「${snapshots?.find((group) => group.id === item.groupId)?.name || item.groupId}」` : "独立挂载"}`;
export const contentConflictMessage = (pair, snapshots) => `「${label(pair.a)}」（${originOf(pair.a, snapshots)}）与「${label(pair.b)}」（${originOf(pair.b, snapshots)}）是内容完全相同的不同副本，不能重复融合；请只保留其中一个。`;

/** Binding-only: no `enabled` key, because the service skips enabled:false and a disabled member would be lost. */
export function groupResolutionReference(member) {
  return { value: member.value, ...Object.fromEntries(IDENTITY_KEYS.filter((key) => member[key] !== undefined).map((key) => [key, structuredClone(member[key])])) };
}

const STATUS_TEXT = {
  unresolved: "这个文件没有登记到 LoRA 身份库。请先在「LoRA 修复」中登记或重新选择；不会自动扫描或登记。",
  "ambiguous-path": "同一路径存在多个候选实例（路径被复用）。请在「LoRA 修复」中明确选择，不会按名称猜测。",
  missing: "登记的文件已缺失。请在「LoRA 修复」中重新选择目标。",
  "content-changed": "文件内容与记录的哈希不同。需要显式修复后才能使用。",
  "unsupported-identity": "身份记录的格式无法识别，原始数据保持不变。",
  "history-cycle": "替换历史形成环；请在「LoRA 修复」中检查。",
};
export function explainResolution(error) {
  if (!error) return "资产库没有返回这个成员的解析结果。";
  const known = STATUS_TEXT[error.resolution?.status];
  if (known) return known;
  if (error.status === undefined) return `无法连接资产库：${error.message || "请求失败"}`;
  return error.message || "资产库拒绝了这个引用。";
}

/** Resolve members one at a time so every offender is reported, then compare publication revisions. */
export async function resolveGroupMembers(engine, group, { resolve, signal } = {}) {
  const entries = [];
  let revision, stale = false;
  for (let index = 0; index < group.members.length; index += 1) {
    if (signal?.aborted) throw Object.assign(new Error("已取消"), { name: "AbortError" });
    try {
      const payload = await resolve(engine, [groupResolutionReference(group.members[index])], { signal });
      if (revision === undefined) revision = payload?.revision;
      else if (payload?.revision !== revision) stale = true;
      entries.push({ index, verdict: payload?.references?.length === 1 ? payload.references[0] : null });
    } catch (error) {
      if (signal?.aborted || error?.name === "AbortError") throw error;
      entries.push({ index, error: { message: error?.message || "", status: error?.status, resolution: error?.resolution } });
    }
  }
  return { revision, stale, entries };
}

/**
 * Whitelist the verdict into strict LoraReference fields. Weight, precision, enablement, name,
 * category and order always come from the source member; only the binding evidence is adopted.
 */
export function projectResolvedGroup(group, { entries = [], stale = false } = {}) {
  const members = [], bindings = [], issues = [];
  if (stale) issues.push({ index: -1, code: "stale", message: "资产库在核验期间发生了变化；请重新核验。" });
  group.members.forEach((member, index) => {
    const entry = entries[index], verdict = entry?.verdict;
    const issue = (code, message) => issues.push({ index, name: label(member), value: member.value, code, message });
    if (!entry || entry.error || !verdict) return issue("service", explainResolution(entry?.error));
    if (verdict.status !== "resolved") return issue(verdict.status || "unresolved", STATUS_TEXT[verdict.status] || STATUS_TEXT.unresolved);
    const evidence = typeof verdict.value === "string" && verdict.value && verdict.value.length <= 500 && INSTANCE_ID.test(verdict.instanceId || "")
      && CONTENT_HASH.test(verdict.contentHash || "") && Number.isSafeInteger(verdict.contentRevision) && verdict.contentRevision > 0;
    if (!evidence) return issue("invalid-evidence", "资产库返回的身份证据不完整，不能作为强引用。");
    if (member.instanceId && member.instanceId !== verdict.instanceId) return issue("identity-changed", "解析结果的实例与组合记录不同；不会自动改绑。");
    if (member.contentHash && member.contentHash !== verdict.contentHash) return issue("content-changed", STATUS_TEXT["content-changed"]);
    const projected = { ...structuredClone(member), value: verdict.value, identitySchema: 1, instanceId: verdict.instanceId, contentHash: verdict.contentHash, contentRevision: verdict.contentRevision };
    if (verdict.value !== member.value && !projected.originalReference) {
      projected.originalReference = { value: member.value, ...(member.instanceId ? { instanceId: member.instanceId } : {}), ...(member.contentHash ? { contentHash: member.contentHash } : {}) };
    }
    if (!xyzStrongLora(projected)) return issue("invalid-evidence", "解析后的成员仍不是有效的强引用。");
    members.push(projected);
    bindings.push({
      index, name: label(member), from: member.value, value: verdict.value, instanceId: verdict.instanceId, contentRevision: verdict.contentRevision,
      weight: member.weight, precision: member.precision, enabled: member.enabled !== false, hadHash: Boolean(member.contentHash),
      kind: !member.instanceId ? "bound" : verdict.value !== member.value ? "relocate" : member.contentRevision !== verdict.contentRevision ? "revision" : "current",
    });
  });
  const ok = issues.length === 0;
  return { ok, issues, bindings, group: ok ? { ...structuredClone(group), members, enabled: false, collecting: false } : null };
}

export const projectionSignature = (projection) => JSON.stringify([projection.ok, projection.issues.map((item) => [item.index, item.code]),
  projection.bindings.map((item) => [item.index, item.value, item.instanceId, item.contentRevision, item.kind])]);

/**
 * What copying a projected group into this draft would do. `blockers` forbid even a disabled
 * copy; `enableBlockers` forbid mounting it. `conflicts` are standalone mounts that the group can
 * replace only by an explicit user action; group-owned mounts are never touched automatically.
 */
export function analyzeXYZGroupCopy(draft, group) {
  const mounts = draft.baseline.loras, snapshots = draft.baseline.group_snapshots, members = group.members;
  const blockers = [], enableBlockers = [], conflicts = [];
  const block = (code, message) => blockers.push({ code, message });
  const noEnable = (code, message, extra = {}) => enableBlockers.push({ code, message, ...extra });
  if (snapshots.some((item) => item.id === group.id)) block("snapshot-exists", "该组合已有独立快照；请直接编辑快照，避免覆盖本地修改。");
  if (members.length > XYZ_MOUNT_LIMIT) block("member-limit", `组合含 ${members.length} 项，超过 ${XYZ_MOUNT_LIMIT} 项上限。`);
  if (members.some((item) => !xyzStrongLora(item))) block("identity", "组合含未解析身份；不能按同名文件复制。");
  if (new Set(members.map((item) => item.instanceId)).size !== members.length || new Set(members.map((item) => item.value)).size !== members.length) block("duplicate-member", "组合包含重复的实例或路径。");
  for (const item of members) if (item.weight === "" || !Number.isFinite(Number(item.weight)) || Math.abs(Number(item.weight)) > 5) block("weight", `「${label(item)}」的权重需为 −5 到 5；不会静默截断。`);
  const inner = contentConflicts(members)[0];
  if (inner) noEnable("content-in-group", contentConflictMessage(inner, [group]));
  members.forEach((member, memberIndex) => {
    for (const existing of mounts) {
      const owner = existing.groupId ? snapshots.find((item) => item.id === existing.groupId) : null;
      const ownerName = owner?.name || existing.groupId;
      if (existing.instanceId === member.instanceId) {
        if (["value", "contentHash", "contentRevision"].some((field) => existing[field] !== member[field])) noEnable("revision", `「${label(member)}」已按另一内容修订挂载；请明确卸载后再选择，不会自动替换。`);
        else if (existing.groupId) noEnable("owned", `「${label(member)}」已属于组合「${ownerName}」；请先在「组合」中自行停用它。`, { owner: ownerName });
        else conflicts.push({ kind: "same-instance", memberIndex, existing });
      } else if (existing.value === member.value) noEnable("path", `「${label(member)}」的路径已被另一个实例占用；不会按路径重绑。`);
      else if (existing.contentHash && existing.contentHash === member.contentHash && effectiveLora(existing) && effectiveLora(member)) {
        if (existing.groupId) noEnable("content-owned", `待复制的「${label(member)}」（${rootOf(member)}）与组合「${ownerName}」中的「${label(existing)}」（${rootOf(existing)}）内容相同（不同副本），不能重复融合；请先在「组合」中自行停用它。`, { owner: ownerName });
        else conflicts.push({ kind: "equal-content", memberIndex, existing });
      }
    }
  });
  const removed = new Set(conflicts.map((item) => item.existing.instanceId));
  const afterEnable = mounts.length + members.length, afterReplace = mounts.length - removed.size + members.length;
  if (!conflicts.length && afterEnable > XYZ_MOUNT_LIMIT) noEnable("limit", `启用后将有 ${afterEnable} 个挂载，超过 ${XYZ_MOUNT_LIMIT} 个上限。`);
  if (conflicts.length && afterReplace > XYZ_MOUNT_LIMIT) noEnable("limit", `替换后将有 ${afterReplace} 个挂载，超过 ${XYZ_MOUNT_LIMIT} 个上限。`);
  const free = !blockers.length;
  return {
    blockers, enableBlockers, conflicts, total: { current: mounts.length, afterEnable, afterReplace },
    promptText: Boolean(group.presetPrompt?.trim()),
    actions: { "copy-only": free, enable: free && !conflicts.length && !enableBlockers.length, replace: free && conflicts.length > 0 && !enableBlockers.length },
  };
}

export const analysisSignature = (analysis) => JSON.stringify([analysis.blockers.map((item) => item.code), analysis.enableBlockers.map((item) => item.code),
  analysis.conflicts.map((item) => [item.kind, item.memberIndex, item.existing.instanceId, item.existing.weight]), analysis.actions]);
