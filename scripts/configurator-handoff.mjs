import { spawn } from "node:child_process";
import path from "node:path";

export function createConfiguratorHandoff({
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
      const configurator = spawnProcess(nodePath, [path.join(projectRoot, "scripts", "setup-gui.mjs"), "--no-open", "--return-to-app"], {
        cwd: projectRoot,
        env: environment,
        detached: true,
        stdio: "ignore",
        windowsHide: true,
      });
      configurator.unref();
    }
    exitProcess(supervised ? 75 : 0);
    return true;
  };
}
