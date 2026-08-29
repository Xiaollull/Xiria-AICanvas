import { spawn } from "node:child_process";
import { rm } from "node:fs/promises";
import { assertSafePluginFolder } from "./plugin-registry.mjs";

// Folder actions for the settings plugin page: reveal in the OS file manager, and remove.
//
// Both act on a *folder*, never on plugin code. Revealing opens the platform file browser on a
// directory; it does not execute, read, or serve anything inside the plugin. Removing deletes the
// folder the user asked to delete and nothing else.
//
// Neither action ever receives a path from the client. The caller passes a plugin id, and the path
// is derived from the fixed plugin root and re-validated through `assertSafePluginFolder`, which
// applies exactly the same link/junction/reparse rule discovery uses. `plugin-registry.mjs` stays
// read-only and spawn-free; everything that acts on the filesystem lives here.

/**
 * The platform command that opens a file manager on a directory. Pure, so the mapping is testable
 * without spawning anything. The path is always passed as a separate argument — never interpolated
 * into a shell string, and no shell is used.
 */
export function revealFolderCommand(platform, absolutePath) {
  if (platform === "win32") return { command: "explorer.exe", args: [absolutePath] };
  if (platform === "darwin") return { command: "open", args: [absolutePath] };
  return { command: "xdg-open", args: [absolutePath] };
}

export async function revealPluginFolder({
  projectRoot,
  id,
  platform = process.platform,
  spawnProcess = spawn,
} = {}) {
  const folderPath = await assertSafePluginFolder({ projectRoot, id });
  const { command, args } = revealFolderCommand(platform, folderPath);
  // Detached and unreferenced so a file manager the user leaves open never holds the dev server
  // alive. Windows Explorer reports a non-zero exit code even on success, so the exit status is
  // deliberately not treated as a result.
  const child = spawnProcess(command, args, { detached: true, stdio: "ignore", windowsHide: true });
  child.on?.("error", () => {});
  child.unref?.();
  return { revealed: true };
}

export async function removePluginFolder({ projectRoot, id } = {}) {
  const folderPath = await assertSafePluginFolder({ projectRoot, id });
  // `rm` unlinks symbolic links rather than following them, so a link planted inside a plugin
  // folder can never redirect this deletion outside the folder.
  await rm(folderPath, { recursive: true, force: false });
  return { removed: true };
}
