import assert from "node:assert/strict";
import test from "node:test";
import { resolveRtxVsrChoice, rtxVsrEligibility } from "./rtx-vsr.mjs";

const windowsBaseline = {
  platform: "win32",
  architecture: "x64",
  pythonVersion: "3.12",
  driverVersion: "570.65",
  cudaAvailable: true,
  torchCudaVersion: "13.0",
  computeCapability: "7.5",
};

test("RTX VSR CLI choices default off and reject contradictory flags", () => {
  assert.equal(resolveRtxVsrChoice([]), null);
  assert.equal(resolveRtxVsrChoice(["--with-rtx-vsr"]), true);
  assert.equal(resolveRtxVsrChoice(["--without-rtx-vsr"]), false);
  assert.throws(() => resolveRtxVsrChoice(["--with-rtx-vsr", "--without-rtx-vsr"]), /不能同时使用/);
});

test("RTX VSR eligibility uses verified CUDA Torch metadata", () => {
  assert.equal(rtxVsrEligibility(windowsBaseline).supported, true);
  assert.equal(rtxVsrEligibility({ ...windowsBaseline, cudaAvailable: false }).supported, false);
  assert.equal(rtxVsrEligibility({ ...windowsBaseline, torchCudaVersion: null }).supported, false);
  assert.equal(rtxVsrEligibility({ ...windowsBaseline, computeCapability: "7.4" }).supported, false);
  assert.equal(rtxVsrEligibility({ ...windowsBaseline, isWsl: true }).supported, false);
  assert.equal(rtxVsrEligibility({ ...windowsBaseline, architecture: "arm64" }).supported, false);
  assert.equal(rtxVsrEligibility({ ...windowsBaseline, platform: "darwin" }).supported, false);
  assert.equal(rtxVsrEligibility({ ...windowsBaseline, pythonVersion: "3.9" }).supported, false);
});

test("RTX VSR driver floors enforce current Windows, Linux, and known TCC boundaries", () => {
  const legacyWindows = rtxVsrEligibility({ ...windowsBaseline, driverVersion: "570.64" });
  assert.equal(legacyWindows.supported, true);
  assert.match(legacyWindows.warning, /真实探针/);
  const legacyTcc = rtxVsrEligibility({ ...windowsBaseline, driverVersion: "594.99", driverModel: "TCC" });
  assert.equal(legacyTcc.supported, true);
  assert.match(legacyTcc.warning, /真实探针/);
  assert.equal(rtxVsrEligibility({ ...windowsBaseline, driverVersion: "595.0", driverModel: "TCC" }).supported, true);
  assert.equal(rtxVsrEligibility({ ...windowsBaseline, driverVersion: "594.99" }).supported, true);

  const baseline = {
    ...windowsBaseline,
    platform: "linux",
    architecture: "x64",
    driverVersion: "570.190",
  };
  assert.equal(rtxVsrEligibility(baseline).supported, true);
  assert.match(rtxVsrEligibility({ ...baseline, driverVersion: "570.189" }).warning, /真实探针/);
  assert.equal(rtxVsrEligibility({ ...baseline, driverVersion: "580.82" }).supported, true);
  assert.match(rtxVsrEligibility({ ...baseline, driverVersion: "580.81" }).warning, /真实探针/);
  assert.equal(rtxVsrEligibility({ ...baseline, driverVersion: "590.44" }).supported, true);
  assert.match(rtxVsrEligibility({ ...baseline, driverVersion: "590.43" }).warning, /真实探针/);
  assert.equal(rtxVsrEligibility({ ...baseline, driverVersion: "591.0" }).supported, true);
});
