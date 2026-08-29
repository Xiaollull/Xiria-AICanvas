import { spawn } from "node:child_process";
import path from "node:path";

export function createUpdateRestartHandoff({
  supervised,
  projectRoot,
  environment = process.env,
  nodePath = process.execPath,
  spawnProcess = spawn,
  exitProcess = (code) => process.exit(code),
}) {
  let handedOff = false;
  return () => {
    if (handedOff) return false;
    handedOff = true;
    if (!supervised) {
      const launcher = spawnProcess(nodePath, [path.join(projectRoot, "scripts", "start.mjs"), "--no-open"], {
        cwd: projectRoot,
        env: environment,
        detached: true,
        stdio: "ignore",
        windowsHide: true,
      });
      launcher.unref();
    }
    exitProcess(supervised ? 77 : 0);
    return true;
  };
}
