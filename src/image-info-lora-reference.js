// A reader result carries BOTH recorded provenance and current execution identity.
// Re-resolving before Apply must not feed the target ID with the old recorded
// path: that would turn A→C replacement into a fictitious C relocation and lose
// the source's confirmed tag permission. No verdict/permission is trusted here;
// the server resolves this recorded reference again against current authority.
export function recordedImageLoraReference(item) {
  const value = item.path || item.value || item.name;
  const source = item.originalReference?.value === value ? item.originalReference : item;
  const hash = source.contentHash || source.hash;
  return { value, ...(source.instanceId ? { identitySchema: 1, instanceId: source.instanceId } : {}),
    ...(typeof hash === "string" && /^[0-9a-f]{64}$/.test(hash) ? { contentHash: hash } : {}), weight: item.weight };
}

export function resolvedImageLoraIdentity(item, resolved) {
  const { weight: _weight, identitySchema: _schema, ...originalReference } = recordedImageLoraReference(item);
  return { ...item, identitySchema: 1, instanceId: resolved.instanceId, contentHash: resolved.contentHash,
    contentRevision: resolved.contentRevision, originalReference };
}
