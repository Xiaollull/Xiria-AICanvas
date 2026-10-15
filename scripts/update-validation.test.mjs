import assert from "node:assert/strict";
import { access, lstat, mkdir, mkdtemp, readdir, readFile, rm, stat, symlink, unlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { writeSetupMarker } from "./setup-state.mjs";
import {
  UPDATE_VALIDATION_CLI,
  UPDATE_VALIDATION_CONTRACT,
  UPDATE_VALIDATION_EVENT_PREFIX,
  UpdateValidationError,
  ensureUpdatedProjectReady,
  mirrorEnvironmentTree,
  repairArguments,
  repairUpdatedEnvironment,
  runValidationCommand,
  validateProjectTree,
  validateUpdatedProject,
} from "./update-validation.mjs";

const scriptsDirectory = import.meta.dirname;

/** A tree that passes every check the validator can run without Python: manifests, marker, Vite entry. */
async function validatableProject(prefix, selection = {}) {
  const projectRoot = await mkdtemp(path.join(os.tmpdir(), prefix));
  await Promise.all([
    mkdir(path.join(projectRoot, "backend"), { recursive: true }),
    mkdir(path.join(projectRoot, "models"), { recursive: true }),
    mkdir(path.join(projectRoot, "node_modules", "vite", "bin"), { recursive: true }),
  ]);
  await Promise.all([
    writeFile(path.join(projectRoot, "backend", "server.py"), "value = 1\n"),
    writeFile(path.join(projectRoot, "node_modules", "vite", "bin", "vite.js"), ""),
    writeFile(path.join(projectRoot, "models", "model-paths.json"), JSON.stringify({ checkpoints: {}, loras: {}, upscalers: "models/upscalers", configs: "models/configs" })),
    writeFile(path.join(projectRoot, "models", "recommended-models.json"), JSON.stringify({ schema: 1, artifacts: [], staticFamilies: [], civitaiFamilies: [] })),
    writeFile(path.join(projectRoot, "models", "yolo-models.json"), JSON.stringify({ schema: 1, models: [] })),
    writeFile(path.join(projectRoot, "models", "background-removal-models.json"), JSON.stringify({ schema: 1, models: [] })),
  ]);
  writeSetupMarker(projectRoot, { complete: true, product: "XiriaCanvas AI", selection }, {});
  return projectRoot;
}

/** Which validator step a fake command invocation is, by what it runs. */
function validationStep(commandArguments) {
  const text = commandArguments.join(" ");
  if (text.includes("pip --isolated check")) return "pip-check";
  if (text.includes("pipeline_configs.py")) return "runtime-configs";
  if (text.includes("import rtx_vsr")) return "rtx-vsr";
  if (text.includes("import inference_server")) return "backend-load";
  if (text.includes("import accelerate")) return "backend-imports";
  if (text.includes("import torch")) return "torch-cuda";
  if (text.includes("ast.parse")) return "python-sources";
  if (text.includes("vite.js build")) return "frontend-build";
  return "other";
}

test("environment validation failure repairs once and validates again", async () => {
  const events = [];
  let validations = 0;
  let repairs = 0;
  const result = await ensureUpdatedProjectReady({
    validate: async () => {
      validations += 1;
      if (validations === 1) throw new UpdateValidationError("environment", "missing dependency");
    },
    repair: async () => { repairs += 1; },
    report: (event) => events.push(event),
  });
  assert.deepEqual(result, { verified: true, repaired: true });
  assert.equal(validations, 2);
  assert.equal(repairs, 1);
  assert.ok(events.some((event) => event.phase === "repair"));
});

test("program validation failure never invokes environment repair", async () => {
  let repairs = 0;
  await assert.rejects(ensureUpdatedProjectReady({
    validate: async () => { throw new UpdateValidationError("program", "invalid source"); },
    repair: async () => { repairs += 1; },
  }), /invalid source/);
  assert.equal(repairs, 0);
});

test("forced dependency repair runs before validation and remains a restart gate", async () => {
  const order = [];
  const result = await ensureUpdatedProjectReady({
    forceRepair: true,
    repair: async () => { order.push("repair"); },
    validate: async () => { order.push("validate"); },
  });
  assert.deepEqual(order, ["repair", "validate"]);
  assert.deepEqual(result, { verified: true, repaired: true });
});

test("failed validation after a transactional repair restores the old environment", async () => {
  const calls = [];
  await assert.rejects(ensureUpdatedProjectReady({
    forceRepair: true,
    repair: async () => ({
      commit: async () => calls.push("commit"),
      rollback: async () => calls.push("rollback"),
      backupRoot: "backup",
    }),
    validate: async () => { throw new UpdateValidationError("program", "new source failed"); },
  }), /new source failed/);
  assert.deepEqual(calls, ["rollback"]);
});

test("updated project validation classifies invalid venv as environment failure", async () => {
  await assert.rejects(validateUpdatedProject({
    projectRoot: process.cwd(),
    verifyVenv: () => ({ ok: false, error: "broken venv" }),
  }), (error) => error.kind === "environment" && /broken venv/.test(error.message));
});

test("updated project validation classifies malformed model manifest as program failure", async () => {
  const projectRoot = await mkdtemp(path.join(os.tmpdir(), "xirai-update-validation-"));
  try {
    await Promise.all([
      mkdir(path.join(projectRoot, "backend"), { recursive: true }),
      mkdir(path.join(projectRoot, "models"), { recursive: true }),
    ]);
    await writeFile(path.join(projectRoot, "backend", "server.py"), "value = 1\n");
    await Promise.all([
      writeFile(path.join(projectRoot, "models", "model-paths.json"), JSON.stringify({ checkpoints: {}, loras: {}, upscalers: "models/upscalers", configs: "models/configs" })),
      writeFile(path.join(projectRoot, "models", "recommended-models.json"), JSON.stringify({ schema: 1, artifacts: [], staticFamilies: [], civitaiFamilies: [] })),
      writeFile(path.join(projectRoot, "models", "yolo-models.json"), "invalid"),
      writeFile(path.join(projectRoot, "models", "background-removal-models.json"), JSON.stringify({ schema: 1, models: [] })),
    ]);
    await assert.rejects(validateUpdatedProject({
      projectRoot,
      verifyVenv: () => ({ ok: true, python: { command: "python", args: [] } }),
      runCommand: async () => ({}),
    }), (error) => error.kind === "program" && /yolo-models/.test(error.message));
  } finally {
    await rm(projectRoot, { recursive: true, force: true });
  }
});

test("updated project validation requires a valid setup marker", async () => {
  const projectRoot = await mkdtemp(path.join(os.tmpdir(), "xirai-update-marker-required-"));
  try {
    await Promise.all([
      mkdir(path.join(projectRoot, "backend"), { recursive: true }),
      mkdir(path.join(projectRoot, "models"), { recursive: true }),
    ]);
    await writeFile(path.join(projectRoot, "backend", "server.py"), "value = 1\n");
    await Promise.all([
      writeFile(path.join(projectRoot, "models", "model-paths.json"), JSON.stringify({ checkpoints: {}, loras: {}, upscalers: "models/upscalers", configs: "models/configs" })),
      writeFile(path.join(projectRoot, "models", "recommended-models.json"), JSON.stringify({ schema: 1, artifacts: [], staticFamilies: [], civitaiFamilies: [] })),
      writeFile(path.join(projectRoot, "models", "yolo-models.json"), JSON.stringify({ schema: 1, models: [] })),
      writeFile(path.join(projectRoot, "models", "background-removal-models.json"), JSON.stringify({ schema: 1, models: [] })),
    ]);
    await assert.rejects(validateUpdatedProject({
      projectRoot,
      verifyVenv: () => ({ ok: true, python: { command: "python", args: [] } }),
      runCommand: async () => ({}),
    }), (error) => error.kind === "environment" && /配置完成标记/.test(error.message));
  } finally {
    await rm(projectRoot, { recursive: true, force: true });
  }
});

test("updated project validation only requires runtime configs installed checkpoints need", async () => {
  const projectRoot = await mkdtemp(path.join(os.tmpdir(), "xirai-update-runtime-config-"));
  const invocations = [];
  try {
    await Promise.all([
      mkdir(path.join(projectRoot, "backend"), { recursive: true }),
      mkdir(path.join(projectRoot, "models"), { recursive: true }),
    ]);
    await writeFile(path.join(projectRoot, "backend", "server.py"), "value = 1\n");
    await Promise.all([
      writeFile(path.join(projectRoot, "models", "model-paths.json"), JSON.stringify({ checkpoints: {}, loras: {}, upscalers: "models/upscalers", configs: "models/configs" })),
      writeFile(path.join(projectRoot, "models", "recommended-models.json"), JSON.stringify({ schema: 1, artifacts: [], staticFamilies: [], civitaiFamilies: [] })),
      writeFile(path.join(projectRoot, "models", "yolo-models.json"), JSON.stringify({ schema: 1, models: [] })),
      writeFile(path.join(projectRoot, "models", "background-removal-models.json"), JSON.stringify({ schema: 1, models: [] })),
    ]);
    await assert.rejects(validateUpdatedProject({
      projectRoot,
      verifyVenv: () => ({ ok: true, python: { command: "python", args: [] } }),
      runCommand: async (_command, commandArguments) => { invocations.push(commandArguments); },
    }), (error) => error.kind === "environment");
  } finally {
    await rm(projectRoot, { recursive: true, force: true });
  }
  const configCheck = invocations.find((commandArguments) => commandArguments.some((value) => value.endsWith("pipeline_configs.py")));
  assert.ok(configCheck, "runtime config validation must run");
  assert.deepEqual(configCheck.slice(1), ["--check", "--required", "--installed", projectRoot]);
});

test("failed backup notification leaves the original Python environment intact", async () => {
  const projectRoot = await mkdtemp(path.join(os.tmpdir(), "xirai-update-backup-notification-"));
  try {
    await mkdir(path.join(projectRoot, ".venv"), { recursive: true });
    await writeFile(path.join(projectRoot, ".venv", "original.txt"), "old environment");
    await assert.rejects(repairUpdatedEnvironment({
      projectRoot,
      onBackup: async () => { throw new Error("journal write failed"); },
      runCommand: async () => assert.fail("repair must not start after backup journal failure"),
    }), /环境自动修复失败/);
    assert.equal(await readFile(path.join(projectRoot, ".venv", "original.txt"), "utf8"), "old environment");
    await assert.rejects(access(path.join(projectRoot, ".venv", ".xirai-offline-update-owner.json")));
  } finally {
    await rm(projectRoot, { recursive: true, force: true });
  }
});

test("update repair arguments preserve the explicit RTX VSR marker choice", () => {
  const enabled = { selection: { variant: "cu130", torch: "2.13.0+cu130", xformers: true, rtxVsr: true } };
  assert.deepEqual(repairArguments(enabled), [
    "--events", "--update-repair", "--torch=cu130", "--torch-version=2.13.0+cu130", "--with-rtx-vsr",
  ]);
  assert.deepEqual(repairArguments(enabled, true), [
    "--events", "--update-repair", "--torch=auto", "--refresh-selection", "--without-xformers", "--with-rtx-vsr",
  ]);
  assert.equal(repairArguments({ selection: {} }).at(-1), "--without-rtx-vsr");
});

test("machine-state checks become warnings instead of rolling an update back", async () => {
  const projectRoot = await validatableProject("xirai-update-warnings-", { variant: "cu130", torch: "2.13.0+cu130", rtxVsr: true });
  try {
    const failing = new Set(["pip-check", "runtime-configs", "torch-cuda", "rtx-vsr"]);
    const steps = [];
    const result = await validateUpdatedProject({
      projectRoot,
      environment: {},
      verifyVenv: () => ({ ok: true, python: { command: "python", args: [] } }),
      probeBackend: async () => { steps.push("backend-probe"); },
      runCommand: async (_command, commandArguments) => {
        const step = validationStep(commandArguments);
        steps.push(step);
        if (failing.has(step)) throw new Error(`${step} failed: Missing local runtime configs: sdxl`);
        return { stdout: "", stderr: "" };
      },
    });
    assert.equal(result.verified, true);
    assert.deepEqual(result.warnings.map((warning) => warning.check), ["python-dependency-consistency", "runtime-configs", "torch-cuda", "rtx-vsr"]);
    assert.match(result.warnings[1].message, /Missing local runtime configs: sdxl/);
    // Every program check still ran after the advisory failures.
    for (const step of ["python-sources", "backend-imports", "backend-load", "backend-probe", "frontend-build"]) assert.ok(steps.includes(step), step);
  } finally {
    await rm(projectRoot, { recursive: true, force: true });
  }
});

test("program checks still block: backend load, backend start and frontend build", async () => {
  for (const failure of ["backend-load", "backend-probe", "frontend-build"]) {
    const projectRoot = await validatableProject("xirai-update-program-block-");
    try {
      await assert.rejects(validateUpdatedProject({
        projectRoot,
        environment: {},
        verifyVenv: () => ({ ok: true, python: { command: "python", args: [] } }),
        probeBackend: async () => { if (failure === "backend-probe") throw new Error("health never became ready"); },
        runCommand: async (_command, commandArguments) => {
          if (validationStep(commandArguments) === failure) throw new Error(`${failure} failed`);
          return { stdout: "", stderr: "" };
        },
      }), (error) => error instanceof UpdateValidationError && error.kind === "program", failure);
    } finally {
      await rm(projectRoot, { recursive: true, force: true });
    }
  }
});

test("a backend import failure still blocks as an environment failure, so repair can run", async () => {
  const projectRoot = await validatableProject("xirai-update-import-block-");
  try {
    await assert.rejects(validateUpdatedProject({
      projectRoot,
      environment: {},
      verifyVenv: () => ({ ok: true, python: { command: "python", args: [] } }),
      probeBackend: async () => {},
      runCommand: async (_command, commandArguments) => {
        if (validationStep(commandArguments) === "backend-imports") throw new Error("No module named 'diffusers'");
        return { stdout: "", stderr: "" };
      },
    }), (error) => error.kind === "environment" && /diffusers/.test(error.message));
  } finally {
    await rm(projectRoot, { recursive: true, force: true });
  }
});

test("warnings reach the caller through the repair orchestration", async () => {
  const warnings = [{ check: "runtime-configs", message: "missing sd" }];
  const result = await ensureUpdatedProjectReady({ validate: async () => ({ verified: true, warnings }), repair: async () => assert.fail("no repair") });
  assert.deepEqual(result, { verified: true, repaired: false, warnings });
});

async function treeWithValidator(prefix, body) {
  const projectRoot = await mkdtemp(path.join(os.tmpdir(), prefix));
  const entry = path.join(projectRoot, ...UPDATE_VALIDATION_CLI.split("/"));
  await mkdir(path.dirname(entry), { recursive: true });
  await writeFile(entry, body);
  return projectRoot;
}

const emitLine = (event) => `process.stdout.write(${JSON.stringify(`${UPDATE_VALIDATION_EVENT_PREFIX}${JSON.stringify({ contract: UPDATE_VALIDATION_CONTRACT, ...event })}\n`)});\n`;

test("an installed tree is validated by the validator that tree ships, not by this process", async () => {
  const projectRoot = await treeWithValidator("xirai-update-delegate-", [
    "if (process.env.XIRAI_REMOTE_UPDATE_TOKEN) process.exit(9);\n",
    emitLine({ type: "progress", progress: 40, message: "new tree checking" }),
    "process.stdout.write('ordinary output is ignored\\n');\n",
    emitLine({ type: "result", verified: true, warnings: [{ check: "runtime-configs", message: "missing sdxl" }] }),
  ].join(""));
  const events = [];
  try {
    const result = await validateProjectTree({
      projectRoot,
      environment: { ...process.env, XIRAI_REMOTE_UPDATE_TOKEN: "a".repeat(64) },
      validateInProcess: async () => assert.fail("the tree ships its own validator"),
      report: (event) => events.push(event),
    });
    assert.deepEqual(result, { verified: true, warnings: [{ check: "runtime-configs", message: "missing sdxl" }] });
    assert.deepEqual(events, [{ phase: "environment", progress: 40, message: "new tree checking" }]);
  } finally {
    await rm(projectRoot, { recursive: true, force: true });
  }
});

test("the shipped validator's failure kind survives the process boundary", async () => {
  const projectRoot = await treeWithValidator("xirai-update-delegate-fail-", [
    emitLine({ type: "error", kind: "environment", message: "No module named 'diffusers'" }),
    "process.exitCode = 1;\n",
  ].join(""));
  const silent = await treeWithValidator("xirai-update-delegate-silent-", "process.stdout.write('done\\n');\n");
  try {
    await assert.rejects(validateProjectTree({ projectRoot }), (error) => error instanceof UpdateValidationError
      && error.kind === "environment" && error.message === "No module named 'diffusers'");
    // Exit 0 without a result is not a pass.
    await assert.rejects(validateProjectTree({ projectRoot: silent }), (error) => error.kind === "program" && /未返回有效结果/.test(error.message));
  } finally {
    await rm(projectRoot, { recursive: true, force: true });
    await rm(silent, { recursive: true, force: true });
  }
});

test("a tree without the entry point falls back to in-process validation", async () => {
  const projectRoot = await mkdtemp(path.join(os.tmpdir(), "xirai-update-legacy-tree-"));
  try {
    const result = await validateProjectTree({ projectRoot, validateInProcess: async (options) => ({ verified: true, root: options.projectRoot }) });
    assert.deepEqual(result, { verified: true, root: projectRoot });
  } finally {
    await rm(projectRoot, { recursive: true, force: true });
  }
});

test("the shipped entry point speaks contract 1", async () => {
  // A tree with no .venv: the real validator stops at its first check and reports it as an event.
  const projectRoot = await mkdtemp(path.join(os.tmpdir(), "xirai-update-cli-contract-"));
  let stdout = "";
  try {
    await assert.rejects(runValidationCommand(process.execPath, [path.join(scriptsDirectory, "update-validate-cli.mjs"), projectRoot], {
      cwd: projectRoot,
      timeout: 60000,
      onStdout: (text) => { stdout += text; },
    }), /错误码 1/);
  } finally {
    await rm(projectRoot, { recursive: true, force: true });
  }
  const events = stdout.split(/\r?\n/).filter((line) => line.startsWith(UPDATE_VALIDATION_EVENT_PREFIX))
    .map((line) => JSON.parse(line.slice(UPDATE_VALIDATION_EVENT_PREFIX.length)));
  assert.ok(events.length >= 2);
  assert.ok(events.every((event) => event.contract === 1));
  assert.equal(events.at(-1).type, "error");
  assert.equal(events.at(-1).kind, "environment");
});

async function sampleEnvironment(prefix) {
  const workspace = await mkdtemp(path.join(os.tmpdir(), prefix));
  const source = path.join(workspace, "source");
  await mkdir(path.join(source, "Lib", "site-packages", "pkg"), { recursive: true });
  await mkdir(path.join(source, "empty"), { recursive: true });
  await writeFile(path.join(source, "pyvenv.cfg"), "home = /usr/bin\n");
  await writeFile(path.join(source, "Lib", "site-packages", "pkg", "__init__.py"), "VERSION = 1\n");
  if (process.platform !== "win32") await symlink("Lib", path.join(source, "lib64"));
  return { workspace, source };
}

test("the repair snapshot is hard-linked, keeps the tree shape, and survives a pip-style rewrite", async () => {
  const { workspace, source } = await sampleEnvironment("xirai-venv-mirror-");
  const destination = path.join(workspace, "destination");
  try {
    const counts = await mirrorEnvironmentTree(source, destination);
    assert.deepEqual(counts, { linked: 2, copied: 0 });
    const file = path.join("Lib", "site-packages", "pkg", "__init__.py");
    const [original, mirrored] = await Promise.all([stat(path.join(source, file)), stat(path.join(destination, file))]);
    assert.equal(mirrored.ino, original.ino, "a link, not a copy");
    assert.deepEqual(await readdir(path.join(destination, "empty")), []);
    if (process.platform !== "win32") assert.ok((await lstat(path.join(destination, "lib64"))).isSymbolicLink());
    // pip and uv replace a file by unlinking it first; the snapshot keeps the old bytes.
    await unlink(path.join(destination, file));
    await writeFile(path.join(destination, file), "VERSION = 2\n");
    assert.equal(await readFile(path.join(source, file), "utf8"), "VERSION = 1\n");
  } finally {
    await rm(workspace, { recursive: true, force: true });
  }
});

test("a volume that cannot hard-link falls back to copying", async () => {
  const { workspace, source } = await sampleEnvironment("xirai-venv-mirror-copy-");
  const destination = path.join(workspace, "destination");
  let attempts = 0;
  try {
    const counts = await mirrorEnvironmentTree(source, destination, {
      linkFile: async () => { attempts += 1; throw Object.assign(new Error("cross-device link"), { code: "EXDEV" }); },
    });
    assert.deepEqual(counts, { linked: 0, copied: 2 });
    assert.equal(attempts, 1, "linking stops being attempted after the volume refuses once");
    assert.equal(await readFile(path.join(destination, "pyvenv.cfg"), "utf8"), "home = /usr/bin\n");
  } finally {
    await rm(workspace, { recursive: true, force: true });
  }
});

test("environment repair snapshots by hard link and restores the original on rollback", async () => {
  const projectRoot = await mkdtemp(path.join(os.tmpdir(), "xirai-update-repair-link-"));
  const venvFile = path.join(projectRoot, ".venv", "Lib", "site.py");
  try {
    await mkdir(path.dirname(venvFile), { recursive: true });
    await writeFile(venvFile, "original\n");
    const originalInode = (await stat(venvFile)).ino;
    const transaction = await repairUpdatedEnvironment({
      projectRoot,
      environment: {},
      runCommand: async () => {
        // The repair rewrites a package the way pip does.
        await unlink(venvFile);
        await writeFile(venvFile, "repaired\n");
        return { stdout: "", stderr: "" };
      },
    });
    assert.ok(transaction.backupRoot);
    const backupFile = path.join(transaction.backupRoot, "Lib", "site.py");
    assert.equal((await stat(backupFile)).ino, originalInode, "the backup is the original environment itself");
    assert.equal(await readFile(backupFile, "utf8"), "original\n");
    await transaction.rollback();
    assert.equal(await readFile(venvFile, "utf8"), "original\n");
    await assert.rejects(access(transaction.backupRoot));
    await assert.rejects(access(path.join(projectRoot, ".venv", ".xirai-offline-update-owner.json")));
  } finally {
    await rm(projectRoot, { recursive: true, force: true });
  }
});

test("a committed environment repair leaves no backup behind", async () => {
  const projectRoot = await mkdtemp(path.join(os.tmpdir(), "xirai-update-repair-commit-"));
  try {
    await mkdir(path.join(projectRoot, ".venv"), { recursive: true });
    await writeFile(path.join(projectRoot, ".venv", "pyvenv.cfg"), "home = /usr/bin\n");
    const transaction = await repairUpdatedEnvironment({ projectRoot, environment: {}, runCommand: async () => ({ stdout: "", stderr: "" }) });
    await transaction.commit();
    await assert.rejects(access(transaction.backupRoot));
    assert.equal(await readFile(path.join(projectRoot, ".venv", "pyvenv.cfg"), "utf8"), "home = /usr/bin\n");
    assert.deepEqual((await readdir(projectRoot)).filter((name) => name.startsWith(".venv")), [".venv"]);
  } finally {
    await rm(projectRoot, { recursive: true, force: true });
  }
});
