import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdir, mkdtemp, rename, link, symlink, writeFile } from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import { randomUUID } from "node:crypto";
import { scanLoraInventory, registerInitialInventory, registerObservedInventory, emptyLoraRegistry, matchLoraInventory, observeLoraFile } from "./lora-library.mjs";
import { resolveLibraryReference } from "../src/lora-identity.js";

test("P10 real Linux native filesystem: inode/rename/hardlink/case/shared/path boundaries", { skip: process.platform !== "linux" }, async () => {
  const base = path.join(os.tmpdir(), "opencode"); await mkdir(base, { recursive: true });
  const directory = await mkdtemp(path.join(base, "p10-linux-"));
  const root = { engine: "Anima", id: "linux", directory };
  await mkdir(path.join(directory, "style"));
  await writeFile(path.join(directory, "style/A.safetensors"), "bytes");
  await writeFile(path.join(directory, "style/a.safetensors"), "other");
  const first = await scanLoraInventory([root]); assert.equal(first.complete, true);
  assert.equal(first.observations.length, 2);
  const registry = registerInitialInventory(emptyLoraRegistry(), first);
  const A = Object.values(registry.instances).find((item) => item.value === "style/A.safetensors");
  assert.ok(A.fsIdentity); assert.equal(A.fsIdentity.kind, "posix");
  await mkdir(path.join(directory, "concept"));
  await rename(path.join(directory, A.value), path.join(directory, "concept/A.safetensors"));
  const renamed = await scanLoraInventory([root]);
  assert.equal(matchLoraInventory(registry, renamed).find((item) => item.sourceId === A.instanceId).confidence, "filesystem-and-content");
  await link(path.join(directory, "concept/A.safetensors"), path.join(directory, "style/link.safetensors"));
  const linked = await scanLoraInventory([root]);
  assert.notEqual(matchLoraInventory(registry, linked).find((item) => item.sourceId === A.instanceId).confidence, "filesystem-and-content");
  // A confirmed move plus a different current occupant must retain BOTH IDs.
  const moved = structuredClone(registry);
  moved.instances[A.instanceId] = { ...A, ...linked.observations.find(item => item.value === "concept/A.safetensors"), history: [A.value] };
  await writeFile(path.join(directory, A.value), "new occupant");
  const reused = await scanLoraInventory([root]);
  for (const item of reused.observations) item.instanceId = Object.values(moved.instances).find(record => record.value === item.value)?.instanceId || randomUUID();
  const baseline = registerObservedInventory(moved, reused);
  assert.equal(resolveLibraryReference(baseline, "Anima", { value: A.value }).status, "ambiguous-path");
  assert.equal(resolveLibraryReference(baseline, "Anima", A).value, "concept/A.safetensors");
  const occupant = Object.values(baseline.instances).find(item => item.value === A.value);
  assert.notEqual(occupant.instanceId, A.instanceId);
  assert.equal(resolveLibraryReference(baseline, "Anima", occupant).value, A.value);
  assert.equal(Object.values(baseline.instances).filter(item => item.value === "style/link.safetensors").length, 1);
  assert.deepEqual(Object.keys(registerObservedInventory(baseline, reused).instances), Object.keys(baseline.instances));
  await symlink(path.join(directory, "concept/A.safetensors"), path.join(directory, "style/symlink.safetensors"));
  await assert.rejects(observeLoraFile(root, "style/symlink.safetensors"), /links/);
  assert.equal((await scanLoraInventory([root])).complete, false);
  await assert.rejects(observeLoraFile(root, "../outside.safetensors"), /Invalid/);
  const shared = await scanLoraInventory([{ ...root, id: "123456abcdef", shared: true }]);
  assert.ok(shared.observations.every((item) => item.value.startsWith("shared:123456abcdef/")));
  const receipt = { platform: process.platform, filesystemRoot: directory, first, renamed, linked, reused, baseline, shared };
  if (process.env.P10_EVIDENCE_DIR) {
    await mkdir(process.env.P10_EVIDENCE_DIR, { recursive: true });
    await writeFile(path.join(process.env.P10_EVIDENCE_DIR, `linux-native-${Date.now()}.json`), JSON.stringify(receipt, null, 2));
  }
});
