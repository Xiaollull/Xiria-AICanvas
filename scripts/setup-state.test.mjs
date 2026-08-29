import assert from "node:assert/strict";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { getSetupMarkerPath, getSetupResumePath, readSetupMarker, readSetupResume, writeSetupMarker, writeSetupResume } from "./setup-state.mjs";

test("setup resume state is written atomically and read back", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "xirai-resume-"));
  const environment = { XIRAI_CACHE_DIR: "state-cache" };
  const resume = { schema: 1, product: "XiriaCanvas AI", completed: { torch: { status: "complete" } } };
  try {
    writeSetupResume(directory, resume, environment);
    assert.deepEqual(readSetupResume(directory, environment), resume);
    writeSetupResume(directory, { ...resume, completed: { ...resume.completed, backend: { status: "complete" } } }, environment);
    assert.equal(readSetupResume(directory, environment).completed.backend.status, "complete");
    assert.equal(getSetupResumePath(directory, environment), path.join(directory, "state-cache", "setup-resume.json"));
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("setup marker is written atomically for update repairs", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "xirai-marker-"));
  const environment = { XIRAI_CACHE_DIR: "state-cache" };
  const marker = { complete: true, product: "XiriaCanvas AI", selection: { torch: "test" } };
  try {
    writeSetupMarker(directory, marker, environment);
    assert.deepEqual(readSetupMarker(directory, environment), marker);
    assert.equal(getSetupMarkerPath(directory, environment), path.join(directory, "state-cache", "setup-complete.json"));
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("invalid setup resume state is ignored", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "xirai-resume-invalid-"));
  const environment = { XIRAI_CACHE_DIR: ".cache" };
  try {
    await mkdir(path.dirname(getSetupResumePath(directory, environment)), { recursive: true });
    await writeFile(getSetupResumePath(directory, environment), "{}", "utf8");
    assert.equal(readSetupResume(directory, environment), null);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("setup marker from another product is ignored", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "xirai-marker-invalid-"));
  const environment = { XIRAI_CACHE_DIR: ".cache" };
  try {
    await mkdir(path.dirname(getSetupMarkerPath(directory, environment)), { recursive: true });
    await writeFile(getSetupMarkerPath(directory, environment), JSON.stringify({ complete: true, product: "Other" }), "utf8");
    assert.equal(readSetupMarker(directory, environment), null);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
