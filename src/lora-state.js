export const LORA_SYNC_CHANNEL = "xirai-lora-workspace-v1";

export function normalizeMountedLoras(value) {
  return inspectMountedLoras(value).items;
}

// This is deliberately the only mounted-entry normalization boundary. Model
// scopes call it as well, so an invalid path can never be copied from one
// model identity to another by a persistence migration or a sync message.
export function inspectMountedLoras(value) {
  if (!Array.isArray(value)) return { items: [], rejected: value === undefined ? 0 : 1 };
  const seen = new Set();
  let rejected = 0;
  const items = [];
  for (const item of value) {
    const path = item?.value;
    if (typeof path !== "string" || !path.trim() || path.length > 500 || path.includes("\0") || seen.has(path)) {
      rejected += 1;
      continue;
    }
    if (items.length >= 16) {
      rejected += 1;
      continue;
    }
    seen.add(path);
    const weight = Number(item.weight);
    // A mounted entry may name the group it came from. The mounted list stays
    // the single source of truth for generation; the tag only says which group
    // definition an edit here has to be written back into, and an unrecognised
    // one degrades to a standalone mount rather than to a dangling reference.
    const groupId = typeof item.groupId === "string" && /^[0-9a-z-]{1,64}$/.test(item.groupId) ? item.groupId : "";
    items.push({
      value: path,
      name: typeof item.name === "string" && item.name.length <= 500 ? item.name : path,
      category: typeof item.category === "string" && item.category.length <= 300 ? item.category : "other",
      weight: Number.isFinite(weight) ? Math.max(-5, Math.min(5, weight)) : 1,
      precision: [1, 2, 4].includes(item.precision) ? item.precision : 1,
      enabled: item.enabled !== false,
      ...(groupId ? { groupId } : {}),
    });
  }
  return { items, rejected };
}

export function sameMountedLoras(first, second) {
  return JSON.stringify(normalizeMountedLoras(first)) === JSON.stringify(normalizeMountedLoras(second));
}
