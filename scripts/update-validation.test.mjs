import assert from "node:assert/strict";
import { access, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import {
  UpdateValidationError,
  ensureUpdatedProjectReady,
  repairArguments,
  repairUpdatedEnvironment,
  validateUpdatedProject,
} from "./update-validation.mjs";

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
