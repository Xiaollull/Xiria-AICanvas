import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { findPython, isolatedPythonEnv, loadLocalEnv, verifyProjectVenv } from "./python.mjs";

const projectRoot = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
Object.assign(process.env, loadLocalEnv(projectRoot));
const verifiedVenv = verifyProjectVenv(projectRoot);
let python;
if (verifiedVenv.ok) {
  python = verifiedVenv.python;
} else {
  python = findPython(projectRoot);
  if (!python) {
    console.error(`${verifiedVenv.error}. Run \`npm run setup\` to create an isolated project environment.`);
    process.exit(1);
  }
  console.warn(`${verifiedVenv.error}; the project may not be fully isolated from other Python installations.`);
}
const environment = isolatedPythonEnv({ ...process.env, ...loadLocalEnv(projectRoot) });
// The progress bar is drawn with block characters, and Windows consoles accept them only as
// UTF-8; left to the system code page they arrive as replacement characters here too.
environment.PYTHONIOENCODING ||= "utf-8";
environment.INFERENCE_WORKSPACE_ID ||= createHash("sha256")
  .update(process.platform === "win32" ? path.normalize(projectRoot).toLowerCase() : path.normalize(projectRoot))
  .digest("hex");

const child = spawn(python.command, [...python.args, path.join(projectRoot, "backend", "inference_server.py")], {
  cwd: projectRoot,
  env: environment,
  stdio: "inherit",
  windowsHide: true,
});

child.on("error", (error) => {
  console.error(`Failed to start inference service: ${error.message}`);
  process.exitCode = 1;
});
child.on("exit", (code, signal) => process.exit(signal ? 1 : (code ?? 0)));

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.once(signal, () => {
    child.kill(signal);
    const timer = setTimeout(() => child.kill("SIGKILL"), 30000);
    timer.unref();
  });
}
