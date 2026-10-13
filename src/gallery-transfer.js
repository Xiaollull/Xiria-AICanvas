// Gallery import/export helpers shared by the dialog and its tests.
export const UPLOAD_ROUTE = "/api/inference/gallery/transfer/import-upload";

/** A pasted path without the quotes Windows "Copy as path" adds. */
export function cleanZipPath(value) {
  const text = String(value || "").trim();
  return text.length >= 2 && text[0] === text.at(-1) && (text[0] === "\"" || text[0] === "'") ? text.slice(1, -1).trim() : text;
}

/** Why a typed path cannot be imported, or "" when it may be sent. The server checks it again. */
export function zipPathIssue(value) {
  const path = cleanZipPath(value);
  if (!path) return "";
  if (!/\.zip$/i.test(path)) return "路径必须指向 .zip 文件本身（以 .zip 结尾）";
  if (!/^(?:[A-Za-z]:[\\/]|\\\\|\/)/.test(path)) return "请填写完整的绝对路径，例如 D:\\Exports\\gallery.zip";
  return "";
}

export function isZipFile(file) {
  return Boolean(file) && /\.zip$/i.test(file.name || "");
}

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "";
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

export async function uploadPackage(file, folder) {
  const query = new URLSearchParams({ name: file.name });
  if (folder) query.set("folder", folder);
  const response = await fetch(`${UPLOAD_ROUTE}?${query}`, { method: "POST", cache: "no-store", headers: { "Content-Type": "application/zip" }, body: file });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(payload.detail) ? payload.detail.map((item) => item.msg || String(item)).join("；") : payload.detail;
    throw new Error(detail || `导入失败（HTTP ${response.status}）`);
  }
  return payload;
}
