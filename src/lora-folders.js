// Folders inside a LoRA category, as something to walk into rather than a flat wall of files.
//
// `/api/loras` answers with each category's files as one flat list, and that list is what mounting,
// scanning and cross-window sync all read — so it stays exactly as it is. A file's `name` inside a
// local category is its path below that category's folder (`Style_九V6.0/matcha.safetensors`), and
// this module projects those paths into a tree for navigation only: at each level the folders come
// first, sorted by name, then the files that sit directly in that folder. Nothing here changes a
// file's `value`, the relative ID a job mounts.
//
// Shared roots are not projected: they keep their own `shared:<id>/…` view and identity.

const collator = typeof Intl !== "undefined" ? new Intl.Collator("zh-CN", { numeric: true, sensitivity: "base" }) : null;
const compareNames = (first, second) => (collator ? collator.compare(first, second) : first.localeCompare(second));

function segmentsOf(value) {
  return String(value || "").split("\\").join("/").split("/").filter(Boolean);
}

/** A folder path with every segment checked; anything with an empty, `.` or `..` part is the root. */
export function normalizeLoraFolderPath(value) {
  const segments = segmentsOf(value);
  return segments.some((segment) => segment === "." || segment === "..") ? "" : segments.join("/");
}

function matchesSearch(model, query) {
  if (!query) return true;
  return String(model.name || "").toLowerCase().includes(query);
}

/**
 * One level of a category, ready to render.
 *
 * `folders` are the immediate subfolders of `folderPath` (with how many files sit anywhere below
 * them); `files` are the models directly inside it. With a search, a folder is listed only when
 * something below it matches, carrying that count, and files deeper down stay inside their folders
 * instead of surfacing at this level — a match keeps its location.
 */
export function loraFolderView(category, folderPath = "", search = "") {
  const models = Array.isArray(category?.models) ? category.models : [];
  const query = String(search || "").trim().toLowerCase();
  const wanted = segmentsOf(normalizeLoraFolderPath(folderPath));
  // Fall back to the deepest ancestor that still exists: a rescan may have removed the folder.
  const hasPrefix = (segments) => models.some((model) => {
    const parts = segmentsOf(model.name);
    return parts.length > segments.length && segments.every((segment, index) => parts[index] === segment);
  });
  let depth = wanted.length;
  while (depth > 0 && !hasPrefix(wanted.slice(0, depth))) depth -= 1;
  const current = wanted.slice(0, depth);
  const folders = new Map();
  const files = [];
  for (const model of models) {
    const parts = segmentsOf(model.name);
    if (parts.length <= current.length || !current.every((segment, index) => parts[index] === segment)) continue;
    const rest = parts.slice(current.length);
    if (rest.length === 1) {
      if (matchesSearch(model, query)) files.push(model);
      continue;
    }
    const name = rest[0];
    const entry = folders.get(name) || { name, path: [...current, name].join("/"), count: 0, matches: 0, bytes: 0 };
    entry.count += 1;
    entry.bytes += Number(model.size) || 0;
    if (matchesSearch(model, query)) entry.matches += 1;
    folders.set(name, entry);
  }
  const folderList = [...folders.values()]
    .filter((folder) => !query || folder.matches > 0)
    .sort((first, second) => compareNames(first.name, second.name));
  files.sort((first, second) => compareNames(segmentsOf(first.name).at(-1) || "", segmentsOf(second.name).at(-1) || ""));
  return {
    path: current.join("/"),
    requested: wanted.join("/"),
    missing: current.length !== wanted.length,
    crumbs: current.map((name, index) => ({ name, path: current.slice(0, index + 1).join("/") })),
    parent: current.length ? current.slice(0, -1).join("/") : null,
    folders: folderList,
    files,
    total: models.length,
  };
}

/** The file name a card shows once its folder is the one being browsed. */
export function loraFileLabel(model) {
  return segmentsOf(model?.name).at(-1) || String(model?.name || "");
}

/**
 * Where a relative folder inside an engine's LoRA root lives in the category view: which category
 * tab, and which folder below it. Mirrors `groupLoraModels`: the first segment is a category folder
 * (a default one or a folder category of its own); the root holds only files.
 */
export function loraCategoryLocation(relativeFolder, defaultCategories = []) {
  const segments = segmentsOf(normalizeLoraFolderPath(relativeFolder));
  if (!segments.length) return { categoryId: "root", folder: "" };
  const known = defaultCategories.find((category) => category.directory === segments[0]);
  return { categoryId: known?.id || `folder:${encodeURIComponent(segments[0])}`, folder: segments.slice(1).join("/") };
}
