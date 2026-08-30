import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const projectRoot = path.dirname(path.dirname(fileURLToPath(import.meta.url)));

/** The one place the running build's version is read from.
 *
 * `package.json` is a managed file, so an applied update replaces it along with the program and
 * the number moves with the code it describes. The About panel, the update check and the release
 * comparison all read this rather than carrying a copy: a hardcoded version in the UI would keep
 * claiming the old release after a successful update, and the check would offer it again forever.
 */
export function readAppVersion(root = projectRoot) {
  try {
    const parsed = JSON.parse(readFileSync(path.join(root, "package.json"), "utf8"));
    return typeof parsed.version === "string" && parsed.version ? parsed.version : "0.0.0";
  } catch {
    return "0.0.0";
  }
}

export const appVersion = readAppVersion();
