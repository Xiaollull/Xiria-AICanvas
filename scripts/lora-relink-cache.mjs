import { readFile, stat, mkdir, copyFile, rename } from "node:fs/promises";
import { randomUUID } from "node:crypto";
import path from "node:path";
import { durableJson } from "./lora-coordinator.mjs";
import { CONTENT_HASH } from "../src/lora-identity.js";
import { parseSharedRef } from "../src/shared-model-refs.js";

/** Derived cache only. This never touches weights, sidecars, user cards/assets,
 * PNGs or workflows. Source authority is verified bytes recorded before the
 * move; target bytes are freshly verified by the committing coordinator.
 */
export async function reuseVerifiedLoraCaches(results, roots, getCachePath, review) {
  const warnings = [];
  for (const result of results) {
    if (!CONTENT_HASH.test(result.sourceHash || "") || result.sourceHash !== result.targetHash) continue;
    const sourceRoot = roots.find((root) => root.engine === result.engine && root.id === result.sourceRootId);
    const targetRoot = roots.find((root) => root.engine === result.engine && root.id === result.targetRootId);
    if (!sourceRoot || !targetRoot) continue;
    const fromRelative = parseSharedRef(result.sourceValue)?.relativePath || result.sourceValue;
    const toRelative = parseSharedRef(result.targetValue)?.relativePath || result.targetValue;
    const fromFile = path.join(sourceRoot.directory, fromRelative), toFile = path.join(targetRoot.directory, toRelative);
    const from = getCachePath(fromFile), to = getCachePath(toFile);
    if (from === to) continue;
    try {
      const source = JSON.parse(await readFile(path.join(from, "metadata.json"), "utf8"));
      // JSON's claim alone is not authority: it must equal BOTH actual verified
      // hashes. Unknown legacy source hashes deliberately cannot reuse it.
      if (source.hash !== result.sourceHash) continue;
      let existing = null;
      try { existing = JSON.parse(await readFile(path.join(to, "metadata.json"), "utf8")); } catch {}
      if (existing?.hash === result.targetHash) continue;
      let local = toRelative;
      for (const prefix of targetRoot.scopeDirectories || []) if (prefix && local.startsWith(`${prefix}/`)) { local = local.slice(prefix.length + 1); break; }
      const category = ["style", "character", "concept", "other"].includes(local.split("/")[0]) ? local.split("/")[0] : "other";
      await mkdir(to, { recursive: true });
      const names = new Set(["preview.webp", "preview.png", "preview.jpg", "preview.jpeg", ...(source.previewFile && path.basename(source.previewFile) === source.previewFile ? [source.previewFile] : [])]);
      let previewFile = null;
      for (const name of names) {
        if (!/\.(?:webp|png|jpe?g)$/i.test(name)) continue;
        const info = await stat(path.join(from, name)).catch(() => null);
        if (!info?.isFile() || info.size > 20 * 1024 * 1024) continue;
        const temporary = path.join(to, `${randomUUID()}.tmp`);
        await copyFile(path.join(from, name), temporary); await rename(temporary, path.join(to, name));
        previewFile ||= name;
      }
      const derived = review ? review(source, source.localMetadata, category) : {};
      await durableJson(path.join(to, "metadata.json"), { ...source, ...derived, ...(previewFile ? { previewFile } : {}),
        relocationAudit: { instanceId: result.targetId, sourceReference: result.sourceValue, contentHash: result.targetHash, method: "verified-full-bytes" } });
    } catch (error) {
      if (error.code !== "ENOENT") warnings.push({ sourceId: result.sourceId, reason: "derived-cache-unavailable" });
    }
  }
  return warnings;
}
