// XYZ and LoRA mix accept only LoRAs with a registered identity (instance + content revision), so
// a library that was never registered — and every path-only mount saved before it was — left both
// modules unusable: catalog cards locked without a word, saved drafts refused by the planner with
// "LoRA identity/content/revision is missing". This module closes both gaps without guessing:
// registration is the same explicit full-root scan + baseline commit the repair panel performs, and
// a path-only reference is upgraded only when the library proves that exact path and its bytes.
import { CONTENT_HASH, INSTANCE_ID } from "./lora-identity.js";
import { announceLoraPublication, loraFetch, resolveLoraReferences } from "./lora-library-client.js";
import { xyzSubmissionId } from "./xyz-client.js";

const RESOLVE_BATCH = 16;
const SCAN_BYTE_BUDGET = 32 * 1024 ** 3;
const SCAN_ROUNDS = 8;

const refsOf = (draft) => [
  ...(draft?.baseline?.loras || []),
  ...(draft?.baseline?.group_snapshots || []).flatMap((group) => group?.members || []),
  ...(draft?.mix?.members || []).map((member) => member?.ref),
].filter((item) => item && typeof item.value === "string" && item.value);

/** Saved references that carry no identity at all, once per path. */
export function pathOnlyFamilyLoras(draft) {
  const seen = new Set();
  return refsOf(draft).filter((item) => !item.instanceId && !item.contentHash && !seen.has(item.value) && seen.add(item.value))
    .map((item) => ({ value: item.value }));
}

/** Identity fields for a resolver answer, only when it names this very path with proven bytes. */
export function provenIdentity(resolved, value) {
  if (resolved?.status !== "resolved" || resolved.kind !== "current" || resolved.value !== value
    || !INSTANCE_ID.test(resolved.instanceId || "") || !CONTENT_HASH.test(resolved.contentHash || "")
    || !Number.isSafeInteger(resolved.contentRevision) || resolved.contentRevision < 1) return null;
  return { identitySchema: 1, instanceId: resolved.instanceId, contentHash: resolved.contentHash, contentRevision: resolved.contentRevision };
}

/** Add identities to the path-only references they belong to. Anything else is left untouched. */
export function withFamilyLoraIdentities(draft, identities) {
  if (!identities?.size || !pathOnlyFamilyLoras(draft).some((item) => identities.has(item.value))) return draft;
  const upgrade = (item) => item && !item.instanceId && !item.contentHash && identities.has(item.value) ? { ...item, ...identities.get(item.value) } : item;
  const next = structuredClone(draft);
  if (Array.isArray(next.baseline.loras)) next.baseline.loras = next.baseline.loras.map(upgrade);
  if (Array.isArray(next.baseline.group_snapshots)) {
    next.baseline.group_snapshots = next.baseline.group_snapshots.map((group) => Array.isArray(group?.members) ? { ...group, members: group.members.map(upgrade) } : group);
  }
  if (Array.isArray(next.mix?.members)) next.mix.members = next.mix.members.map((member) => member?.ref ? { ...member, ref: upgrade(member.ref) } : member);
  return next;
}

/** Ask the library for each path. One unresolvable path must not cost the others their answer. */
export async function resolveFamilyLoraIdentities(engine, references, { resolve = resolveLoraReferences } = {}) {
  const identities = new Map();
  const record = (reference, resolved) => {
    const identity = provenIdentity(resolved, reference.value);
    if (identity) identities.set(reference.value, identity);
  };
  for (let index = 0; index < references.length; index += RESOLVE_BATCH) {
    const batch = references.slice(index, index + RESOLVE_BATCH).map((item) => ({ value: item.value, weight: "1", enabled: true }));
    try {
      const answer = await resolve(engine, batch);
      batch.forEach((reference, position) => record(reference, answer?.references?.[position]));
    } catch {
      for (const reference of batch) {
        try { record(reference, (await resolve(engine, [reference]))?.references?.[0]); } catch { /* ambiguous, missing or changed: needs the repair panel */ }
      }
    }
  }
  return identities;
}

/**
 * Register every LoRA of one engine: a full scan of that engine's roots, then the baseline commit.
 * Hashing is resumable in the service, so a scan that only ran out of byte budget is repeated.
 */
export async function registerEngineLoraIdentities(engine, { request = loraFetch, newId = xyzSubmissionId, onProgress } = {}) {
  const call = async (action, body) => {
    const response = await request(`/api/lora-relink/${action}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    let payload = {};
    try { payload = await response.json(); } catch { payload = {}; }
    if (!response.ok) throw new Error(payload.error || `LoRA 身份登记失败（${response.status}）`);
    return payload;
  };
  let scan;
  for (let round = 1; round <= SCAN_ROUNDS; round += 1) {
    onProgress?.(round === 1 ? "正在读取文件内容（只读，不改变任何挂载）…" : `继续读取剩余文件（第 ${round} 轮）…`);
    scan = await call("scan", { engines: [engine], maxBytes: SCAN_BYTE_BUDGET });
    const budgetOnly = !scan.complete && scan.issues?.length && scan.issues.every((issue) => issue.reason === "observation-413");
    if (!budgetOnly) break;
  }
  if (!scan.complete || !scan.domain?.fullRoots) {
    const first = scan.issues?.[0];
    throw new Error(`扫描未能完整读取 ${engine} 的 LoRA 目录${first ? `（${first.relative || first.rootId}：${first.reason}）` : ""}；可在“修复 LoRA”面板查看完整说明`);
  }
  onProgress?.("正在登记身份…");
  const receipt = await call("commit", { scanId: scan.scanId, selections: [], registerBaseline: true, operationId: newId() });
  announceLoraPublication(receipt.revision);
  return { files: scan.observations?.length || 0, revision: receipt.revision };
}
