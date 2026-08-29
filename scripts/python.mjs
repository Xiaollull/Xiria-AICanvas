import { spawnSync } from "node:child_process";
import { existsSync, lstatSync, readFileSync, realpathSync } from "node:fs";
import path from "node:path";

export function loadLocalEnv(projectRoot) {
  const envPath = path.join(projectRoot, ".env");
  if (!existsSync(envPath)) return {};
  const values = {};
  for (const line of readFileSync(envPath, "utf8").split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const separator = trimmed.indexOf("=");
    if (separator < 1) continue;
    const key = trimmed.slice(0, separator).trim();
    let value = trimmed.slice(separator + 1).trim();
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    if (!(key in process.env)) values[key] = value;
  }
  return values;
}

function works(candidate, minimumMinor = 10, maximumMinor = 13) {
  const probe = spawnSync(candidate.command, [...candidate.args, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"], {
    encoding: "utf8",
    windowsHide: true,
    timeout: 10000,
  });
  if (probe.status !== 0) return false;
  const match = probe.stdout.trim().match(/^(\d+)\.(\d+)$/);
  return Boolean(match) && Number(match[1]) === 3 && Number(match[2]) >= minimumMinor && Number(match[2]) <= maximumMinor;
}

export function getVenvPythonPath(projectRoot) {
  return process.platform === "win32"
    ? path.join(projectRoot, ".venv", "Scripts", "python.exe")
    : path.join(projectRoot, ".venv", "bin", "python");
}

export function findPython(projectRoot, { includeVenv = true, includeOverride = true } = {}) {
  const candidates = [];
  if (includeVenv) {
    candidates.push({
      command: getVenvPythonPath(projectRoot),
      args: [],
      source: ".venv",
    });
  }
  if (includeOverride && process.env.XIRAI_PYTHON) {
    candidates.push({ command: process.env.XIRAI_PYTHON, args: [], source: "XIRAI_PYTHON" });
  }
  if (process.platform === "win32") {
    candidates.push(
      { command: "py", args: ["-3.13"], source: "Python launcher" },
      { command: "py", args: ["-3.12"], source: "Python launcher" },
      { command: "py", args: ["-3.11"], source: "Python launcher" },
      { command: "py", args: ["-3.10"], source: "Python launcher" },
      { command: "python", args: [], source: "PATH" },
      { command: "python3", args: [], source: "PATH" },
    );
  } else {
    candidates.push(
      { command: "python3.13", args: [], source: "PATH" },
      { command: "python3.12", args: [], source: "PATH" },
      { command: "python3.11", args: [], source: "PATH" },
      { command: "python3.10", args: [], source: "PATH" },
      { command: "python3", args: [], source: "PATH" },
      { command: "python", args: [], source: "PATH" },
    );
  }
  return candidates.find((candidate) => works(candidate)) || null;
}

export function verifyProjectVenv(projectRoot) {
  const venvRoot = path.join(projectRoot, ".venv");
  const pythonPath = getVenvPythonPath(projectRoot);
  if (!existsSync(pythonPath)) return { ok: false, error: "Project .venv does not exist" };
  try {
    if (lstatSync(venvRoot).isSymbolicLink()) return { ok: false, error: "Project .venv must not be a symlink or junction" };
    const venvConfig = readFileSync(path.join(venvRoot, "pyvenv.cfg"), "utf8");
    if (/include-system-site-packages\s*=\s*true/i.test(venvConfig)) {
      return { ok: false, error: "Project .venv must not inherit system site-packages" };
    }
  } catch {
    return { ok: false, error: "Project .venv is incomplete or invalid" };
  }

  const script = "import json,sys; print(json.dumps({'prefix':sys.prefix,'base_prefix':sys.base_prefix,'executable':sys.executable}))";
  const result = spawnSync(pythonPath, ["-I", "-c", script], { encoding: "utf8", windowsHide: true, timeout: 10000 });
  if (result.status !== 0) return { ok: false, error: "Project .venv Python is not executable" };
  try {
    const details = JSON.parse(result.stdout.trim());
    const expectedPrefix = realpathSync(venvRoot);
    const actualPrefix = realpathSync(details.prefix);
    const normalize = (value) => process.platform === "win32" ? value.toLowerCase() : value;
    if (normalize(expectedPrefix) !== normalize(actualPrefix) || normalize(details.prefix) === normalize(details.base_prefix)) {
      return { ok: false, error: "Project .venv is not an isolated virtual environment" };
    }
    return { ok: true, python: { command: pythonPath, args: [], source: ".venv" }, details };
  } catch {
    return { ok: false, error: "Could not verify project .venv" };
  }
}

export function isolatedPythonEnv(environment = process.env) {
  const clean = { ...environment, PYTHONDONTWRITEBYTECODE: "1", PYTHONNOUSERSITE: "1" };
  for (const name of ["PYTHONHOME", "PYTHONPATH", "PIP_CONFIG_FILE", "PIP_TARGET", "PIP_PREFIX", "PIP_USER", "VIRTUAL_ENV", "CONDA_PREFIX", "__PYVENV_LAUNCHER__"]) {
    delete clean[name];
  }
  return clean;
}

export function runPython(python, args, options = {}) {
  return spawnSync(python.command, [...python.args, ...args], {
    cwd: options.cwd,
    env: options.env || process.env,
    stdio: options.stdio || "inherit",
    encoding: options.encoding,
    timeout: options.timeout,
    windowsHide: true,
  });
}
