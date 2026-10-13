// Presentation helpers for the shared LoRA repair panel (P10.1). They label,
// filter and group what the server returned; ranking itself stays server-side.
// Labels are never request values: a selection always carries the observed
// `rootId + value` pair.
import { parseSharedRef } from "./shared-model-refs.js";

export const PICKER_ALTERNATIVES = 5;
export const PICKER_SEARCH_LIMIT = 60;

/** rootId -> { label, shared, detail }. Duplicate shared labels get a stable
 * short id suffix; the full id stays available as `detail`. */
export function repairRootLabels(roots = []) {
  const labels = new Map(), shared = new Map();
  for (const root of roots) {
    if (labels.has(root.rootId)) continue;
    if (root.shared) {
      const label = root.label || "共享目录";
      shared.set(label, [...(shared.get(label) || []), root.rootId]);
      labels.set(root.rootId, { label, shared: true, detail: root.rootId });
    } else labels.set(root.rootId, { label: "本地", shared: false, detail: root.rootId });
  }
  for (const ids of shared.values()) if (ids.length > 1) for (const id of ids) labels.set(id, { ...labels.get(id), label: `${labels.get(id).label} · ${id.slice(0, 4)}` });
  return labels;
}

export function repairRelativeValue(value) { return parseSharedRef(value)?.relativePath || value; }

/** Basename first, folder second. Only `.safetensors` is hidden, so two files
 * with one stem and different extensions stay distinguishable. */
export function describeRepairFile(item, labels = new Map()) {
  const relative = item.relative || repairRelativeValue(item.value);
  const name = relative.split("/").at(-1);
  const folder = relative.includes("/") ? `${relative.slice(0, relative.lastIndexOf("/"))}/` : "";
  const shared = Boolean(parseSharedRef(item.value)) || item.rootId?.startsWith?.("local-") === false;
  const root = labels.get(item.rootId) || { label: shared ? "共享目录" : "本地", shared, detail: item.rootId };
  return { name, displayName: name.replace(/\.safetensors$/i, ""), folder, relative, rootLabel: root.label, rootDetail: root.detail, shared: root.shared, key: repairKey(item) };
}
export const repairKey = (item) => `${item.rootId}\0${item.value}`;
export const sameRepairTarget = (a, b) => Boolean(a && b && a.rootId === b.rootId && a.value === b.value);

/** Relevant-first sections for one row. `selected` stays visible even when it
 * is neither suggested nor in the current search. */
export function repairPickerSections(row, observations, { query = "" } = {}) {
  const pool = observations.filter((item) => item.engine === row.engine);
  const find = (target) => target && pool.find((item) => sameRepairTarget(item, target));
  const ranked = new Map((row.candidates || []).map((candidate) => [repairKey(candidate), candidate]));
  const recommended = find(row.recommendation);
  const alternatives = (row.candidates || []).filter((candidate) => !sameRepairTarget(candidate, row.recommendation))
    .map(find).filter(Boolean).slice(0, PICKER_ALTERNATIVES);
  const text = query.trim().toLowerCase();
  let results = [], total = 0;
  if (text) {
    const hits = pool.filter((item) => item.value.toLowerCase().includes(text) || repairRelativeValue(item.value).toLowerCase().includes(text));
    total = hits.length;
    const starts = (item) => repairRelativeValue(item.value).split("/").at(-1).toLowerCase().startsWith(text) ? 0 : 1;
    results = hits.sort((a, b) => starts(a) - starts(b) || Number(b.rootId.startsWith("local-")) - Number(a.rootId.startsWith("local-"))
      || repairRelativeValue(a.value).localeCompare(repairRelativeValue(b.value))).slice(0, PICKER_SEARCH_LIMIT);
  }
  const selected = find(row.target);
  const shown = [recommended, ...alternatives, ...results].filter(Boolean);
  return { recommended: recommended || null, alternatives, results, total, ranked,
    selected: selected && !shown.some((item) => sameRepairTarget(item, selected)) ? selected : null };
}

/** Browse-all tree: every usable local / enabled shared file of the engine,
 * local roots first, then folders, then names. */
export function repairBrowseTree(observations, engine, labels = new Map()) {
  const roots = new Map();
  for (const item of observations) {
    if (item.engine !== engine) continue;
    const file = describeRepairFile(item, labels);
    const root = roots.get(item.rootId) || { rootId: item.rootId, label: file.rootLabel, shared: file.shared, folders: new Map() };
    root.folders.set(file.folder, [...(root.folders.get(file.folder) || []), item]);
    roots.set(item.rootId, root);
  }
  return [...roots.values()].sort((a, b) => Number(a.shared) - Number(b.shared) || a.label.localeCompare(b.label)).map((root) => ({
    ...root, folders: [...root.folders.entries()].sort(([a], [b]) => a.localeCompare(b))
      .map(([folder, items]) => ({ folder, items: items.sort((a, b) => repairRelativeValue(a.value).localeCompare(repairRelativeValue(b.value))) })),
  }));
}

/** Folders that "Match by directory" can target, local first. */
export function repairFolderChoices(observations, engines, labels = new Map()) {
  const seen = new Map();
  for (const item of observations) {
    if (!engines.includes(item.engine)) continue;
    const file = describeRepairFile(item, labels);
    const key = `${item.rootId}\0${file.folder}`;
    const entry = seen.get(key) || { rootId: item.rootId, folder: file.folder.replace(/\/$/, ""), rootLabel: file.rootLabel, shared: file.shared, files: 0 };
    entry.files += 1; seen.set(key, entry);
  }
  return [...seen.values()].sort((a, b) => Number(a.shared) - Number(b.shared) || a.rootLabel.localeCompare(b.rootLabel) || a.folder.localeCompare(b.folder));
}
