import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { libraryError } from "./lora-library.mjs";

/** Private owner IPC, no inference startup and no arbitrary SQL in Node. */
export function galleryOwnerParticipant(stateDirectory, python) {
  const cwd = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const call = (action, operationId, patches, token, extra = {}) => new Promise((resolve, reject) => {
    const child = spawn(python, ["-B", "-m", "backend.lora_relink_owner"], { cwd, windowsHide: true, stdio: ["pipe", "pipe", "pipe"] });
    const stdout = [], stderr = [];
    let bytes = 0;
    const timeout = setTimeout(() => child.kill(), 30000);
    child.stdout.on("data", (chunk) => { bytes += chunk.length; if (bytes > 1024 * 1024) child.kill(); else stdout.push(chunk); });
    child.stderr.on("data", (chunk) => { if (stderr.reduce((size, value) => size + value.length, 0) < 65536) stderr.push(chunk); });
    child.on("error", (error) => { clearTimeout(timeout); reject(error); });
    child.on("close", (code) => {
      clearTimeout(timeout);
      if (code !== 0) {
        const error = libraryError(`Gallery owner refused repair (${code}); inspect the retained operation journal`, 503);
        error.ownerDiagnostic = Buffer.concat(stderr).toString("utf8"); // private, not HTTP error text
        reject(error);
        return;
      }
      try { resolve(JSON.parse(Buffer.concat(stdout).toString("utf8"))); } catch (error) { reject(error); }
    });
    child.stdin.on("error", reject);
    child.stdin.end(JSON.stringify({ stateDirectory, token, action, operationId, patches, ...extra }));
  });
  return { list: (token) => call("list", null, [], token),
    xyzReferences: (draft, token) => call("xyz-references", null, [], token, { draft }),
    identity: (file) => call("identity", null, [], null, { file }),
    check: (patches, token) => call("check", null, patches, token),
    prepare: (id, patches, token) => call("prepare", id, patches, token),
    publish: (id, token) => call("publish", id, [], token) };
}
