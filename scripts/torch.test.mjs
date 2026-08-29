import assert from "node:assert/strict";
import test from "node:test";
import { torchInternals } from "./torch.mjs";

test("stable versions sort newest first", () => {
  const versions = ["2.9.1+cu128", "2.10.0+cu128", "2.9.0+cu128"];
  versions.sort(torchInternals.compareVersions);
  assert.deepEqual(versions, ["2.10.0+cu128", "2.9.1+cu128", "2.9.0+cu128"]);
});

test("CUDA variants compare by numeric capability", () => {
  assert.equal(torchInternals.cudaCapability("cu126"), 126);
  assert.equal(torchInternals.cudaCapability("cu130"), 130);
  assert.equal(torchInternals.cudaCapability("cpu"), 0);
});

test("CUDA wheel labels parse without assuming a one-digit minor forever", () => {
  assert.deepEqual(torchInternals.cudaVariantVersion("cu92"), { major: 9, minor: 2, text: "9.2" });
  assert.deepEqual(torchInternals.cudaVariantVersion("cu130"), { major: 13, minor: 0, text: "13.0" });
  assert.deepEqual(torchInternals.cudaVariantVersion("cu1310"), { major: 13, minor: 10, text: "13.10" });
  assert.equal(torchInternals.cudaVariantIsCompatible("cu1310", { major: 13, minor: 3 }), false);
});

test("pre-release versions are not accepted as stable candidates", () => {
  assert.deepEqual(torchInternals.versionParts("2.10.0"), [2, 10, 0, ""]);
  assert.equal(torchInternals.versionParts("2.10.0rc1"), null);
});

test("automatic variants use the newest wheel supported by the driver", () => {
  const variants = ["cpu", "cu126", "cu128", "cu130", "cu131"];
  assert.deepEqual(torchInternals.compatibleVariants(variants, {
    available: true,
    cuda: { major: 13, minor: 3, text: "13.3" },
  }), ["cu131", "cu130", "cu128", "cu126"]);
});

test("a CUDA wheel with no kernels for the installed GPU is not offered", () => {
  // A Tesla V100 is sm_70 and CUDA 13 dropped everything below sm_75. Its driver still reports a
  // CUDA 13 ceiling, so the driver check alone picks cu130 — which installs, reports
  // `torch.cuda.is_available() == True`, and then fails the first real operation with
  // "no kernel image is available for execution on the device".
  const volta = { available: true, cuda: { major: 13, minor: 0, text: "13.0" }, computeCapability: "7.0" };
  assert.deepEqual(torchInternals.compatibleVariants(["cu126", "cu128", "cu130"], volta), ["cu128", "cu126"]);

  // An Ampere card under the same driver keeps every wheel.
  const ampere = { ...volta, computeCapability: "8.6" };
  assert.deepEqual(torchInternals.compatibleVariants(["cu126", "cu128", "cu130"], ampere), ["cu130", "cu128", "cu126"]);

  // Turing is exactly the CUDA 13 floor.
  assert.equal(torchInternals.cudaVariantSupportsDevice("cu130", "7.5"), true);
  assert.equal(torchInternals.cudaVariantSupportsDevice("cu130", "7.0"), false);
  // Kepler is below the CUDA 12 floor.
  assert.equal(torchInternals.cudaVariantSupportsDevice("cu126", "3.7"), false);
  assert.equal(torchInternals.cudaVariantSupportsDevice("cu126", "5.0"), true);
});

test("an unreadable compute capability does not veto a wheel", () => {
  // The real CUDA operation setup runs after installing is the authority. Refusing every wheel
  // because one query failed would strand a working machine.
  assert.equal(torchInternals.cudaVariantSupportsDevice("cu130", null), true);
  assert.equal(torchInternals.cudaVariantSupportsDevice("cu130", "unknown"), true);
  assert.deepEqual(torchInternals.compatibleVariants(["cu130"], {
    available: true,
    cuda: { major: 13, minor: 0, text: "13.0" },
    computeCapability: null,
  }), ["cu130"]);
  assert.deepEqual(torchInternals.parseComputeCapability("8.6"), { major: 8, minor: 6, text: "8.6" });
  assert.deepEqual(torchInternals.minimumComputeCapability("cu130"), { major: 7, minor: 5 });
  assert.equal(torchInternals.minimumComputeCapability("cpu"), null);
});

test("an NVIDIA GPU never silently falls back to CPU", () => {
  assert.deepEqual(torchInternals.compatibleVariants(["cpu", "cu126"], {
    available: true,
    cuda: null,
  }), []);
});

test("CPU wheels remain available only without an NVIDIA GPU", () => {
  assert.deepEqual(torchInternals.compatibleVariants(["cu130", "cpu"], {
    available: false,
    cuda: null,
  }), ["cpu"]);
});

test("newest stable PyTorch wins before CUDA runtime freshness", () => {
  const plans = [
    { version: "2.12.0+cu130", variant: "cu130" },
    { version: "2.13.0+cu128", variant: "cu128" },
    { version: "2.13.0+cu130", variant: "cu130" },
  ];
  plans.sort(torchInternals.comparePlans);
  assert.deepEqual(plans.map((plan) => `${plan.version}/${plan.variant}`), [
    "2.13.0+cu130/cu130",
    "2.13.0+cu128/cu128",
    "2.12.0+cu130/cu130",
  ]);
});

test("duplicate mirror plans collapse before choosing one combination", () => {
  const plans = torchInternals.uniquePlans([
    { version: "2.13.0+cu130", variant: "cu130", indexRoot: "official" },
    { version: "2.13.0+cu130", variant: "cu130", indexRoot: "mirror" },
    { version: "2.13.0+cu128", variant: "cu128", indexRoot: "official" },
    { version: "2.12.0+cu130", variant: "cu130", indexRoot: "official" },
  ]);
  assert.deepEqual(plans.map((plan) => `${plan.version}/${plan.variant}`), [
    "2.13.0+cu130/cu130",
    "2.13.0+cu128/cu128",
    "2.12.0+cu130/cu130",
  ]);
});

test("stable plans rank newest PyTorch releases first", () => {
  const plans = torchInternals.uniquePlans([
    { version: "2.13.0+cu130", variant: "cu130" },
    { version: "2.13.0+cu128", variant: "cu128" },
    { version: "2.12.1+cu130", variant: "cu130" },
    { version: "2.12.1+cu128", variant: "cu128" },
  ]);
  assert.deepEqual(plans.map((plan) => `${plan.version}/${plan.variant}`), [
    "2.13.0+cu130/cu130",
    "2.13.0+cu128/cu128",
    "2.12.1+cu130/cu130",
    "2.12.1+cu128/cu128",
  ]);
});

test("a custom index may point directly at one CUDA variant", () => {
  assert.equal(torchInternals.directRootVariant("https://example.test/pytorch/cu130"), "cu130");
  assert.equal(torchInternals.directRootVariant("https://example.test/pytorch/"), null);
});
