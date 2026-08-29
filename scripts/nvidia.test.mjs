import assert from "node:assert/strict";
import test from "node:test";
import { detectNvidiaSmi, mergeNvidiaDetection, parseCudaDriverApiVersion, parseCudaVersion, parseNvidiaGpuQuery, parseNvidiaQuery } from "./nvidia.mjs";

test("classic nvidia-smi CUDA capability is parsed", () => {
  assert.deepEqual(parseCudaVersion("Driver Version: 560.94 CUDA Version: 12.6"), { major: 12, minor: 6, text: "12.6" });
});

test("new CUDA UMD version output is parsed", () => {
  assert.deepEqual(parseCudaVersion("KMD Version: 610.74 CUDA UMD Version: 13.3"), { major: 13, minor: 3, text: "13.3" });
});

test("CUDA driver API integer version is parsed", () => {
  assert.deepEqual(parseCudaDriverApiVersion(13030), { major: 13, minor: 3, text: "13.3" });
});

test("GPU query keeps model and driver separate", () => {
  assert.deepEqual(parseNvidiaQuery("NVIDIA GeForce RTX 5090, 610.74\n"), {
    name: "NVIDIA GeForce RTX 5090",
    driverVersion: "610.74",
  });
});

test("driver API capability augments nvidia-smi identity", () => {
  assert.deepEqual(mergeNvidiaDetection(
    { available: true, name: "NVIDIA GeForce RTX 5090", driverVersion: "610.74", cuda: null, source: "nvidia-smi" },
    { available: true, name: "NVIDIA GeForce RTX 5090", cuda: { major: 13, minor: 3, text: "13.3" }, source: "driver-api" },
  ), {
    available: true,
    name: "NVIDIA GeForce RTX 5090",
    driverVersion: "610.74",
    cuda: { major: 13, minor: 3, text: "13.3" },
    computeCapability: null,
    source: "driver-api",
  });
});

test("the GPU architecture is read alongside the driver's CUDA ceiling", () => {
  // They answer different questions: the driver says which CUDA runtime it can host, the card says
  // which kernels a wheel must contain. A Tesla V100 under a current driver reports CUDA 13 and is
  // sm_70, an architecture CUDA 13 no longer builds for.
  assert.deepEqual(parseNvidiaGpuQuery("Tesla V100S-PCIE-32GB, 580.173.02, 7.0\n"), {
    name: "Tesla V100S-PCIE-32GB",
    driverVersion: "580.173.02",
    computeCapability: "7.0",
  });
  // A driver too old for `compute_cap` leaves the field absent rather than wrong.
  assert.deepEqual(parseNvidiaGpuQuery("NVIDIA GeForce RTX 5090, 610.74\n"), {
    name: "NVIDIA GeForce RTX 5090",
    driverVersion: "610.74",
    computeCapability: null,
  });

  const responses = [
    { status: 0, stdout: "Tesla V100S-PCIE-32GB, 580.173.02, 7.0\n", stderr: "" },
    { status: 0, stdout: "NVIDIA-SMI 580.173.02 Driver Version: 580.173.02 CUDA Version: 13.0\nTesla V100S-PCIE-32GB", stderr: "" },
  ];
  const detected = detectNvidiaSmi("nvidia-smi", () => responses.shift());
  assert.equal(detected.computeCapability, "7.0");
  assert.deepEqual(detected.cuda, { major: 13, minor: 0, text: "13.0" });
});

test("a driver too old for the compute_cap query still identifies the GPU", () => {
  const responses = [
    { status: 2, stdout: "", stderr: "Field \"compute_cap\" is not a valid field to query." },
    { status: 0, stdout: "NVIDIA GeForce GTX 1080, 470.256.02\n", stderr: "" },
    { status: 0, stdout: "NVIDIA-SMI 470.256.02 Driver Version: 470.256.02 CUDA Version: 11.4\nNVIDIA GeForce GTX 1080", stderr: "" },
  ];
  const detected = detectNvidiaSmi("nvidia-smi", () => responses.shift());
  assert.equal(detected.available, true);
  assert.equal(detected.name, "NVIDIA GeForce GTX 1080");
  assert.equal(detected.computeCapability, null);
});

test("new nvidia-smi UMD output identifies a GPU instead of CPU fallback", () => {
  const responses = [
    { status: 0, stdout: "NVIDIA GeForce RTX 5090, 610.74\n", stderr: "" },
    { status: 0, stdout: "NVIDIA-SMI 610.74 KMD Version: 610.74 CUDA UMD Version: 13.3\nNVIDIA GeForce RTX 5090", stderr: "" },
  ];
  const detected = detectNvidiaSmi("nvidia-smi", () => responses.shift());
  assert.equal(detected.available, true);
  assert.equal(detected.name, "NVIDIA GeForce RTX 5090");
  assert.deepEqual(detected.cuda, { major: 13, minor: 3, text: "13.3" });
});
