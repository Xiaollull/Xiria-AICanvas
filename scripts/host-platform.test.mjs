import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { HOST_PATH_EXAMPLES, HOST_PLATFORM, hostIsWindows, hostPathExample } from "../src/host-platform.js";

const readSource = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

const viteConfig = await readSource("vite.config.js");
const reader = await readSource("src/ImageInfoReader.jsx");

test("path hints follow the host, not the browser", () => {
  // The WebUI binds 0.0.0.0 and advertises LAN URLs, so the machine rendering
  // the input is routinely not the machine holding the folders. Offering a
  // Linux user `F:\...` names a path their filesystem cannot represent.
  assert.equal(hostPathExample("modelsRoot", "win32"), "F:\\ComfyUI\\models");
  assert.equal(hostPathExample("modelsRoot", "linux"), "/home/you/ComfyUI/models");
  assert.equal(hostPathExample("loraDirectory", "win32"), "D:\\Stable-diffusion-webui\\models\\Lora");
  assert.equal(hostPathExample("loraDirectory", "linux"), "/home/you/stable-diffusion-webui/models/Lora");
  assert.equal(hostPathExample("imageDirectory", "win32"), "D:\\outputs\\2026-08");
  assert.equal(hostPathExample("imageDirectory", "linux"), "/home/you/outputs/2026-08");

  assert.equal(hostIsWindows("win32"), true);
  for (const platform of ["linux", "darwin", ""]) assert.equal(hostIsWindows(platform), false);

  // Every example must actually be absolute in the flavour it claims.
  for (const [kind, example] of Object.entries(HOST_PATH_EXAMPLES)) {
    assert.match(example.win32, /^[A-Za-z]:\\/, kind);
    assert.match(example.posix, /^\//, kind);
    assert.doesNotMatch(example.posix, /\\/, kind);
  }
  assert.equal(hostPathExample("not-a-kind", "linux"), "");
});

test("the host platform is substituted by Vite and degrades safely without it", () => {
  // Imported straight into Node here, so the constant was never substituted;
  // reading it must not throw, and the POSIX example is the safe default.
  assert.equal(HOST_PLATFORM, "");
  assert.equal(hostPathExample("modelsRoot"), "/home/you/ComfyUI/models");
  assert.match(viteConfig, /__XIRAI_HOST_PLATFORM__: JSON\.stringify\(process\.platform\)/);
  assert.match(reader, /hostPathExample\("imageDirectory"\)/);
  assert.doesNotMatch(reader, /[A-Z]:\\\\outputs/);
});
