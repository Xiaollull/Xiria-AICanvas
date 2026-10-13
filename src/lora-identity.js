// Management identity is additive. It never changes the protocol-39 path/weight wire.
export const LORA_IDENTITY_SCHEMA = 1;
export const INSTANCE_ID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
export const CONTENT_HASH = /^[0-9a-f]{64}$/;
const fields = ["identitySchema", "instanceId", "contentHash", "contentRevision", "originalReference"];

export function inspectLoraIdentity(item) {
  if (!item || !fields.some((key) => Object.hasOwn(item, key))) return { fields: {}, fatal: false };
  const validRef = (ref) => ref && typeof ref === "object" && !Array.isArray(ref)
    && typeof ref.value === "string" && ref.value.length > 0 && ref.value.length <= 500 && !ref.value.includes("\0")
    && (!Object.hasOwn(ref, "instanceId") || INSTANCE_ID.test(ref.instanceId))
    && (!Object.hasOwn(ref, "contentHash") || CONTENT_HASH.test(ref.contentHash));
  if ((item.identitySchema !== undefined && item.identitySchema !== LORA_IDENTITY_SCHEMA)
    || (item.instanceId !== undefined && !INSTANCE_ID.test(item.instanceId))
    || (item.contentHash !== undefined && !CONTENT_HASH.test(item.contentHash))
    || (item.contentRevision !== undefined && (!Number.isSafeInteger(item.contentRevision) || item.contentRevision < 1))
    || (item.originalReference !== undefined && !validRef(item.originalReference))) {
    return { fields: {}, fatal: true };
  }
  return { fields: Object.fromEntries(fields.filter((key) => Object.hasOwn(item, key)).map((key) => [key,
    key === "originalReference" ? { ...item[key] } : item[key]])), fatal: false };
}

/** Identity wins over path. This resolver never guesses a copy by name/hash. */
export function resolveLibraryReference(registry, engine, reference, { history = false, caseInsensitive = false } = {}) {
  if (registry?.schemaVersion !== 1 || inspectLoraIdentity(reference).fatal) return { status: "unsupported-identity" };
  const instances = Object.values(registry.instances || {}).filter((item) => item.engine === engine);
  const samePath = (value) => caseInsensitive ? value.toLowerCase() === reference.value.toLowerCase() : value === reference.value;
  const pathMatches = instances.filter((item) => samePath(item.value) || item.history?.some(samePath));
  const candidates = pathMatches.map(({ instanceId, engine, rootId, value, contentHash, status }) => ({ instanceId, engine, rootId, value, contentHash, status }));
  // Full bytes disambiguate an already recorded path provenance, never a name
  // or a hash-only copy search. Equal-byte occupants still require selection.
  const provenMatches = reference.contentHash ? pathMatches.filter((item) => item.contentHash === reference.contentHash) : pathMatches;
  let instance = reference.instanceId ? instances.find((item) => item.instanceId === reference.instanceId)
    : provenMatches.length === 1 ? provenMatches[0] : null;
  if (!reference.instanceId && provenMatches.length > 1) return { status: "ambiguous-path", auditKind: "path-reuse", candidates };
  if (!instance) return { status: "unresolved" };
  const source = instance;
  let expectedHash = reference.contentHash || instance.contentHash;
  let kind = source.value !== reference.value ? "relocate" : "current";
  if (history) {
    const visited = new Set();
    for (;;) {
      const key = `${instance.instanceId}:${expectedHash}`;
      if (visited.has(key)) return { status: "history-cycle" };
      visited.add(key);
      const rule = (registry.substitutions || []).find((item) => item.engine === engine && item.sourceId === instance.instanceId && (!item.sourceHash || item.sourceHash === expectedHash));
      if (!rule) break;
      instance = instances.find((item) => item.instanceId === rule.targetId);
      if (!instance) return { status: "unresolved" };
      expectedHash = rule.targetHash; kind = "replace";
    }
  }
  if (instance.status !== "present") return { status: "missing", instanceId: instance.instanceId };
  if (expectedHash && expectedHash !== instance.contentHash) return { status: "content-changed", instanceId: instance.instanceId };
  return { status: "resolved", value: instance.value, instanceId: instance.instanceId,
    contentHash: instance.contentHash, contentRevision: instance.contentRevision, kind, originalValue: reference.value,
    sourceId: source.instanceId, auditKind: reference.instanceId ? "instance" : reference.contentHash ? "full-content-path" : "recorded-path", candidates };
}
