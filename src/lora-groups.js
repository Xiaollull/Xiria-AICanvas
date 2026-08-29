import { normalizeMountedLoras } from "./lora-state.js";
import { READY_LORA_ENGINES, engineScopeKey } from "./lora-model-scope.js";

// User-defined LoRA combinations.
//
// The mounted list stays the single source of truth for generation: nothing here
// is read when a job is submitted except the preset prompt. A group is a saved
// *definition* — members with their weights, plus a preset positive prompt —
// and enabling one mounts those members with a `groupId` tag. That keeps drag
// ordering, the 16-entry cap, cross-window sync, directory pruning and the
// generation payload exactly as they were, and means a corrupt group file can
// never take the mounted library down with it.
//
// Groups are scoped per engine like the mounted map. A group of SD LoRAs cannot
// be enabled under Anima, because the paths address different directories.

export const LORA_GROUPS_SCHEMA_VERSION = 1;
export const MAXIMUM_GROUPS = 32;
export const MAXIMUM_GROUP_NAME = 60;
export const MAXIMUM_PRESET_PROMPT = 2000;
export const MAXIMUM_GROUP_MEMBERS = 16;

const GROUP_ID_PATTERN = /^[0-9a-z-]{1,64}$/;

function isPlainObject(value) { return Boolean(value) && typeof value === "object" && !Array.isArray(value); }

export function emptyLoraGroupMap() {
  return { schemaVersion: LORA_GROUPS_SCHEMA_VERSION, byEngine: Object.fromEntries(READY_LORA_ENGINES.map((engine) => [engine, []])) };
}

export function newLoraGroupId() {
  return `g-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

/** Collapses the whitespace a textarea invites, so a preset never posts a wall of blank lines. */
function cleanPresetPrompt(value) {
  if (typeof value !== "string") return "";
  return value.replace(/\r\n?/g, "\n").replace(/\n{3,}/g, "\n\n").slice(0, MAXIMUM_PRESET_PROMPT).trim();
}

function cleanGroupName(value, fallback) {
  const name = typeof value === "string" ? value.replace(/\s+/g, " ").trim().slice(0, MAXIMUM_GROUP_NAME) : "";
  return name || fallback;
}

/**
 * Members reuse the mounted-entry normalizer, so a path that could not be
 * mounted can never be stored in a group either. The `groupId` tag itself is
 * stripped: membership is expressed by which group holds the entry.
 */
function normalizeMembers(value) {
  return normalizeMountedLoras(value).slice(0, MAXIMUM_GROUP_MEMBERS).map(({ groupId: _groupId, ...member }) => member);
}

export function normalizeLoraGroup(value, index = 0) {
  if (!isPlainObject(value)) return null;
  const id = typeof value.id === "string" && GROUP_ID_PATTERN.test(value.id) ? value.id : "";
  if (!id) return null;
  return {
    id,
    name: cleanGroupName(value.name, `LoRA 组 ${index + 1}`),
    enabled: value.enabled === true,
    // While a group is collecting, LoRAs mounted from the library join it rather
    // than landing standalone. That is what makes "create an empty group and try
    // things out in it" a real flow instead of a rename-and-drag chore.
    collecting: value.collecting === true,
    presetPrompt: cleanPresetPrompt(value.presetPrompt),
    members: normalizeMembers(value.members),
  };
}

export function normalizeLoraGroupList(value) {
  if (!Array.isArray(value)) return { groups: [], rejected: value === undefined ? 0 : 1 };
  const seen = new Set();
  const groups = [];
  let rejected = 0;
  for (const entry of value) {
    const group = normalizeLoraGroup(entry, groups.length);
    if (!group || seen.has(group.id) || groups.length >= MAXIMUM_GROUPS) {
      rejected += 1;
      continue;
    }
    seen.add(group.id);
    groups.push(group);
  }
  let collector = "";
  for (const group of groups) {
    if (!group.collecting) continue;
    // Only one target can win, and a disabled group has no mounted block for the
    // new entry to appear in, so it cannot be one.
    if (collector || !group.enabled) group.collecting = false;
    else collector = group.id;
  }
  return { groups, rejected };
}

/**
 * Unlike the mounted map, a damaged group file is *not* fatal. Groups are a
 * convenience layered over a mounted library that works without them, so the
 * safe response is to drop the unreadable groups and keep generating.
 */
export function normalizeLoraGroupMap(value, { fieldMissing = false } = {}) {
  if (fieldMissing || value === undefined || value === null) return { container: emptyLoraGroupMap(), warning: "", rejected: 0 };
  if (!isPlainObject(value) || !Number.isInteger(value.schemaVersion) || value.schemaVersion > LORA_GROUPS_SCHEMA_VERSION || !isPlainObject(value.byEngine)) {
    return { container: emptyLoraGroupMap(), warning: "LoRA 组合格式无法识别，已重置为空；挂载库未受影响。", rejected: 0 };
  }
  const container = emptyLoraGroupMap();
  let rejected = 0;
  for (const engine of READY_LORA_ENGINES) {
    const normalized = normalizeLoraGroupList(value.byEngine[engine]);
    rejected += normalized.rejected;
    container.byEngine[engine] = normalized.groups;
  }
  return { container, warning: rejected ? `已隔离 ${rejected} 个无效的 LoRA 组合。` : "", rejected };
}

export function loraGroupsForScope(container, engine) {
  const scope = engineScopeKey(engine);
  return scope ? normalizeLoraGroupList(container?.byEngine?.[scope]).groups : [];
}

export function withLoraGroupsForScope(container, engine, groups) {
  const scope = engineScopeKey(engine);
  const next = emptyLoraGroupMap();
  for (const target of READY_LORA_ENGINES) next.byEngine[target] = loraGroupsForScope(container, target);
  if (scope) next.byEngine[scope] = normalizeLoraGroupList(groups).groups;
  return next;
}

export function updateLoraGroupsForScope(container, engine, updater) {
  const scope = engineScopeKey(engine);
  const previous = loraGroupsForScope(container, scope);
  if (!scope || typeof updater !== "function") return { container: withLoraGroupsForScope(container, scope, previous), groups: previous, changed: false };
  const groups = normalizeLoraGroupList(updater(previous)).groups;
  return { container: withLoraGroupsForScope(container, scope, groups), groups, changed: !sameLoraGroups(previous, groups) };
}

export function sameLoraGroups(first, second) {
  return JSON.stringify(normalizeLoraGroupList(first).groups) === JSON.stringify(normalizeLoraGroupList(second).groups);
}

export function sameLoraGroupMap(first, second) {
  return READY_LORA_ENGINES.every((engine) => sameLoraGroups(first?.byEngine?.[engine], second?.byEngine?.[engine]));
}

export function createLoraGroup({ id, name, members = [], presetPrompt = "", enabled = false, collecting = false } = {}, index = 0) {
  return normalizeLoraGroup({
    id: typeof id === "string" && GROUP_ID_PATTERN.test(id) ? id : newLoraGroupId(),
    name,
    enabled,
    collecting,
    presetPrompt,
    members,
  }, index);
}

export function enabledLoraGroups(groups) {
  return normalizeLoraGroupList(groups).groups.filter((group) => group.enabled);
}

/**
 * Preset prompts of the enabled groups, in group order, ahead of what the user
 * typed. The prompt box on the generate page is never rewritten — this runs on
 * the request body only, so what is submitted stays visible in the saved
 * settings while the box the user owns keeps saying what they wrote.
 */
export function composeGroupPrompt(groups, basePrompt = "") {
  const parts = [
    ...enabledLoraGroups(groups).map((group) => group.presetPrompt),
    typeof basePrompt === "string" ? basePrompt : "",
  ];
  return parts
    // A preset that already ends in a comma must not produce ", ," once joined.
    .map((part) => part.trim().replace(/^[,\s]+|[,\s]+$/g, ""))
    .filter(Boolean)
    .join(", ");
}

/** The group a mounted entry belongs to, or "" when it is a standalone mount. */
export function groupIdOfMountedLora(item, groups) {
  const tag = typeof item?.groupId === "string" ? item.groupId : "";
  if (!tag) return "";
  return normalizeLoraGroupList(groups).groups.some((group) => group.id === tag) ? tag : "";
}

/**
 * Splits the mounted list into per-group blocks plus the standalone remainder,
 * preserving mounted order inside each block. Only groups that actually have a
 * mounted member get a block, so a disabled group does not render an empty one.
 */
export function partitionMountedByGroup(loras, groups) {
  const known = normalizeLoraGroupList(groups).groups;
  const mounted = normalizeMountedLoras(loras);
  const blocks = [];
  const byId = new Map();
  for (const group of known) {
    const block = { group, items: [] };
    byId.set(group.id, block);
  }
  const standalone = [];
  for (const item of mounted) {
    const block = byId.get(groupIdOfMountedLora(item, known));
    if (block) block.items.push(item);
    else standalone.push(item);
  }
  for (const group of known) {
    const block = byId.get(group.id);
    if (block.items.length) blocks.push(block);
  }
  return { blocks, standalone };
}

/** Members of a group as mountable entries, tagged so edits find their way home. */
export function groupMembersAsMounted(group) {
  const normalized = normalizeLoraGroup(group);
  if (!normalized) return [];
  return normalized.members.map((member) => ({ ...member, groupId: normalized.id }));
}

/**
 * Enabling a group appends its members; disabling removes exactly the entries
 * carrying its tag. A member already mounted standalone is adopted rather than
 * duplicated, because the mounted list is keyed by path and a second copy would
 * silently be dropped by normalization.
 */
export function applyGroupEnabled(loras, group, enabled) {
  const normalized = normalizeLoraGroup(group);
  const mounted = normalizeMountedLoras(loras);
  if (!normalized) return { loras: mounted, changed: false, overflow: false };
  if (!enabled) {
    const next = mounted.filter((item) => item.groupId !== normalized.id);
    return { loras: next, changed: next.length !== mounted.length, overflow: false };
  }
  const members = groupMembersAsMounted(normalized);
  const memberPaths = new Set(members.map((member) => member.value));
  const kept = mounted.filter((item) => !memberPaths.has(item.value) && item.groupId !== normalized.id);
  const next = [...kept, ...members];
  if (next.length > MAXIMUM_GROUP_MEMBERS) return { loras: mounted, changed: false, overflow: true };
  return { loras: next, changed: JSON.stringify(next) !== JSON.stringify(mounted), overflow: false };
}

/**
 * Writes a mounted-area edit back into the group that owns it. Weight, precision
 * and enabled follow the mounted entry; a member no longer mounted under the tag
 * is dropped from the group, which is what unmounting a grouped LoRA means.
 * Standalone mounts are invisible here by construction — they carry no tag.
 */
export function syncMountedIntoGroups(groups, loras) {
  const known = normalizeLoraGroupList(groups).groups;
  const mounted = normalizeMountedLoras(loras);
  const byGroup = new Map(known.map((group) => [group.id, []]));
  for (const item of mounted) {
    const bucket = byGroup.get(groupIdOfMountedLora(item, known));
    if (bucket) bucket.push(item);
  }
  return known.map((group) => {
    // A disabled group is not represented in the mounted list at all, so an
    // empty bucket there means "not mounted", never "the user emptied it".
    if (!group.enabled) return group;
    return { ...group, members: normalizeMembers(byGroup.get(group.id) || []) };
  });
}

/**
 * An enabled group is *represented* in the mounted list: its members are mounted
 * carrying its tag, which is the invariant `applyGroupEnabled` maintains and
 * `syncMountedIntoGroups` relies on. Replacing the mounted list wholesale —
 * applying a gallery card, or a picture's parameters — destroys that
 * representation, and a group left switched on afterwards is two bugs at once:
 * it keeps prefixing its preset prompt onto every request while none of its
 * LoRAs are loaded, and the next mounted edit syncs it back to zero members and
 * empties the user's own definition.
 *
 * So a group that is no longer fully mounted under its tag is switched off. Its
 * member list is kept, because switching it back on is how the user restores it.
 */
export function disableUnmountedGroups(groups, loras) {
  const known = normalizeLoraGroupList(groups).groups;
  const mounted = normalizeMountedLoras(loras);
  const tagged = new Set(mounted.map((item) => `${item.groupId || ""} ${item.value}`));
  let changed = false;
  const next = known.map((group) => {
    if (!group.enabled) return group;
    const represented = group.members.length > 0
      && group.members.every((member) => tagged.has(`${group.id} ${member.value}`));
    if (represented) return group;
    changed = true;
    return { ...group, enabled: false };
  });
  return { groups: next, changed };
}

/** Groups left holding no members after an edit; the caller decides whether to prune. */
export function emptyLoraGroupIds(groups) {
  return normalizeLoraGroupList(groups).groups.filter((group) => !group.members.length).map((group) => group.id);
}

/** The group new mounts should join, or "" when mounts stay standalone. */
export function collectingGroupId(groups) {
  return normalizeLoraGroupList(groups).groups.find((group) => group.collecting)?.id || "";
}

/**
 * Points new mounts at one group, or at none when `id` is empty. Enabling is
 * implied: a collector has to be mounted for its block to exist.
 */
export function setCollectingGroup(groups, id) {
  return normalizeLoraGroupList(groups).groups.map((group) => group.id === id
    ? { ...group, collecting: true, enabled: true }
    : { ...group, collecting: false });
}

/**
 * Tags a freshly mounted entry with the collecting group, if there is one. Both
 * mount surfaces call this so a LoRA picked from the library lands in the same
 * place regardless of which one the user is looking at.
 */
export function mountedEntryForGroups(entry, groups) {
  const target = collectingGroupId(groups);
  return target ? { ...entry, groupId: target } : entry;
}
