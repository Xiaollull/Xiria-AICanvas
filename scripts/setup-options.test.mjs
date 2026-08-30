import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  buildSetupArguments,
  buildSetupCatalog,
  parseCudaVariants,
  parseTorchWheelVersions,
  recommendRepairPlan,
  validateSetupConfiguration,
} from "./setup-options.mjs";
import {
  installOptionalAccelerator,
  removablePackages,
  resolveAcceleratorChoice,
  sageAttentionSmokeSource,
  SAGE_ATTENTION_REQUIREMENT,
  tritonPackageName,
  tritonSmokeSource,
} from "./anima-accelerators.mjs";

test("manual setup catalog keeps only modern CUDA wheel indexes", () => {
  const html = '<a href="cu117/">cu117</a><a href="cu118/">cu118</a><a href="cu126/">cu126</a><a href="cu132/">cu132</a><a href="rocm7.1/">rocm</a>';
  assert.deepEqual(parseCudaVariants(html), ["cu132", "cu126", "cu118"]);
});

test("wheel parser filters Python, platform, architecture, and pre-releases", () => {
  const html = [
    '<a href="torch-2.13.0%2Bcu126-cp312-cp312-win_amd64.whl">torch-2.13.0+cu126-cp312-cp312-win_amd64.whl</a>',
    '<a href="torch-2.12.1%2Bcu126-cp312-cp312-win_amd64.whl">torch-2.12.1+cu126-cp312-cp312-win_amd64.whl</a>',
    '<a href="torch-2.13.0.dev1%2Bcu126-cp312-cp312-win_amd64.whl">nightly</a>',
    '<a href="torch-2.13.0%2Bcu126-cp311-cp311-win_amd64.whl">other Python</a>',
    '<a href="torch-2.13.0%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl">Linux</a>',
  ].join("");
  assert.deepEqual(parseTorchWheelVersions(html, {
    variant: "cu126", pythonTag: "cp312", platform: "win32", architecture: "x64",
  }), ["2.13.0+cu126", "2.12.1+cu126"]);
});

test("catalog marks one recommended combination and discourages alternatives", () => {
  const catalog = buildSetupCatalog({
    hardware: { available: true, cuda: { major: 13, minor: 3, text: "13.3" } },
    pythonTag: "cp312",
    variantVersions: {
      cu132: ["2.13.0+cu132", "2.12.1+cu132"],
      cu130: ["2.13.0+cu130"],
      cu134: ["2.13.0+cu134"],
    },
  });
  assert.deepEqual(catalog.recommended, { version: "2.13.0+cu132", variant: "cu132", compatible: true });
  assert.equal(catalog.cudaOptions.find((item) => item.variant === "cu134").compatible, false);
  assert.equal(catalog.cudaOptions.find((item) => item.variant === "cu132").versions[1].discouraged, true);
});

test("repair recommendation moves to a lower compatible CUDA runtime after the preferred plan fails", () => {
  const catalog = buildSetupCatalog({
    hardware: { available: true, cuda: { major: 13, minor: 3, text: "13.3" } },
    pythonTag: "cp312",
    variantVersions: { cu132: ["2.13.0+cu132"], cu130: ["2.13.0+cu130"], cu128: ["2.12.1+cu128"] },
  });
  assert.deepEqual(recommendRepairPlan(catalog, { torch: "2.13.0+cu132", variant: "cu132" }), {
    version: "2.13.0+cu130", variant: "cu130", compatible: true,
  });
  assert.deepEqual(recommendRepairPlan(catalog, { torch: "2.12.1+cu128", variant: "cu128" }), catalog.recommended);
  assert.deepEqual(recommendRepairPlan(catalog, {
    torch: "2.13.0+cu130", variant: "cu130", repairAttempted: true,
  }), {
    version: "2.12.1+cu128", variant: "cu128", compatible: true,
  });
  assert.equal(recommendRepairPlan(catalog, {
    torch: "2.12.1+cu128", variant: "cu128", repairAttempted: true,
  }), null);
});

test("manual setup rejects driver-incompatible and missing wheel combinations", () => {
  const catalog = buildSetupCatalog({
    hardware: { available: true, cuda: { major: 12, minor: 6, text: "12.6" } },
    pythonTag: "cp312",
    variantVersions: { cu126: ["2.13.0+cu126"], cu128: ["2.13.0+cu128"] },
  });
  assert.equal(validateSetupConfiguration({ mode: "manual", cudaVariant: "cu128", torchVersion: "2.13.0+cu128" }, catalog).ok, false);
  assert.equal(validateSetupConfiguration({ mode: "manual", cudaVariant: "cu126", torchVersion: "2.12.0+cu126" }, catalog).ok, false);
  assert.equal(validateSetupConfiguration({ mode: "manual", cudaVariant: "cu126", torchVersion: "2.13.0+cu126", autoRepair: true }, catalog).ok, true);
});

test("automatic setup requires a recommended NVIDIA CUDA combination", () => {
  const withoutGpu = buildSetupCatalog({
    hardware: { available: false, cuda: null },
    pythonTag: "cp312",
    variantVersions: { cu126: ["2.13.0+cu126"] },
  });
  const withoutCompatibleWheel = buildSetupCatalog({
    hardware: { available: true, cuda: { major: 12, minor: 4, text: "12.4" } },
    pythonTag: "cp312",
    variantVersions: { cu126: ["2.13.0+cu126"] },
  });
  assert.equal(validateSetupConfiguration({ mode: "auto" }, withoutGpu).ok, false);
  assert.equal(validateSetupConfiguration({ mode: "auto" }, withoutCompatibleWheel).ok, false);
});

test("GUI setup arguments always forward one explicit choice for every optional component", () => {
  assert.deepEqual(buildSetupArguments({ mode: "auto", installXformers: true, installRtxVsr: false }), [
    "--torch=auto", "--without-rtx-vsr", "--without-triton", "--without-sageattention",
  ]);
  assert.deepEqual(buildSetupArguments({
    mode: "manual",
    cudaVariant: "cu130",
    torchVersion: "2.13.0+cu130",
    installXformers: false,
    installRtxVsr: true,
    installTriton: true,
    installSageAttention: true,
  }), [
    "--torch=cu130", "--torch-version=2.13.0+cu130", "--refresh-selection", "--without-xformers",
    "--with-rtx-vsr", "--with-triton", "--with-sageattention",
  ]);
});

test("choosing SageAttention chooses Triton, because Sage 1.x has no kernels of its own", () => {
  const catalog = buildSetupCatalog({
    hardware: { available: true, cuda: { major: 13, minor: 0, text: "13.0" } },
    pythonTag: "cp312",
    variantVersions: { cu130: ["2.13.0+cu130"] },
  });
  const implied = validateSetupConfiguration({ mode: "auto", installSageAttention: true }, catalog);
  assert.equal(implied.ok, true);
  assert.equal(implied.configuration.installTriton, true);
  assert.equal(implied.configuration.installSageAttention, true);
  assert.deepEqual(buildSetupArguments(implied.configuration).slice(-2), ["--with-triton", "--with-sageattention"]);

  // Triton alone is a valid choice; it unlocks compilation without changing any kernel.
  const tritonOnly = validateSetupConfiguration({ mode: "auto", installTriton: true }, catalog);
  assert.equal(tritonOnly.configuration.installTriton, true);
  assert.equal(tritonOnly.configuration.installSageAttention, false);

  // Both default off, like every other optional component.
  const bare = validateSetupConfiguration({ mode: "auto" }, catalog);
  assert.equal(bare.configuration.installTriton, false);
  assert.equal(bare.configuration.installSageAttention, false);
});

test("accelerator choices are a tri-state so a marker can carry the previous answer", () => {
  assert.equal(resolveAcceleratorChoice([], "triton"), null);
  assert.equal(resolveAcceleratorChoice(["--with-triton"], "triton"), true);
  assert.equal(resolveAcceleratorChoice(["--without-triton"], "triton"), false);
  assert.equal(resolveAcceleratorChoice(["--with-sageattention"], "triton"), null);
  assert.equal(resolveAcceleratorChoice(["--with-sageattention"], "sageattention"), true);
  assert.throws(() => resolveAcceleratorChoice(["--with-triton", "--without-triton"], "triton"), /不能同时使用/);
});

test("the Triton check verifies the Inductor import path, not merely that triton imports", () => {
  const source = tritonSmokeSource("2.13.0+cu130");
  assert.match(source, /import torch,triton/);
  assert.match(source, /torch\.__version__=="2\.13\.0\+cu130"/);
  // PyTorch reads its own Inductor templates with the locale encoding, so on a
  // non-UTF-8 machine the import dies before compilation is ever attempted. The
  // check must exercise that path with the same repair the backend applies.
  assert.match(source, /encoding='utf-8'/);
  assert.match(source, /import torch\._inductor\.compile_fx/);
  assert.match(source, /has_triton_package\(\)/);
  // `load_flex_template` calls the reader by keyword. A positional-only stand-in
  // raises TypeError mid-import, and setup would then uninstall a working Triton.
  assert.match(source, /lambda name,template_dir:/);
  assert.doesNotMatch(source, /lambda name,d:/);
});

test("the SageAttention check makes a real CUDA call in the layout the runtime uses", () => {
  const source = sageAttentionSmokeSource();
  assert.match(source, /from sageattention import sageattn/);
  assert.match(source, /torch\.cuda\.is_available\(\)/);
  // Head dim 128, BF16, HND — exactly what AnimaCosmosAttnProcessor passes.
  assert.match(source, /\(1,4,256,128\)/);
  assert.match(source, /dtype=torch\.bfloat16/);
  assert.match(source, /tensor_layout='HND'/);
  assert.match(source, /torch\.isfinite\(o\)\.all\(\)/);
});

test("Triton comes from the distribution that actually ships a wheel for the platform", () => {
  assert.equal(tritonPackageName("win32"), "triton-windows");
  assert.equal(tritonPackageName("linux"), "triton");
  // PyPI publishes SageAttention only up to the pure-Triton 1.x line; 2.x needs
  // an MSVC/CUDA source build and is deliberately not attempted.
  assert.equal(SAGE_ATTENTION_REQUIREMENT, "sageattention<2");
});

test("setup configuration defaults RTX VSR off and preserves an explicit opt-in", () => {
  const catalog = buildSetupCatalog({
    hardware: { available: true, cuda: { major: 13, minor: 0, text: "13.0" } },
    pythonTag: "cp312",
    variantVersions: { cu130: ["2.13.0+cu130"] },
  });
  assert.equal(validateSetupConfiguration({ mode: "auto" }, catalog).configuration.installRtxVsr, false);
  assert.equal(validateSetupConfiguration({ mode: "auto", installRtxVsr: true }, catalog).configuration.installRtxVsr, true);
});

test("an unreachable Hugging Face defers runtime configs instead of failing setup", async () => {
  const setup = await readFile(new URL("./setup.mjs", import.meta.url), "utf8");
  assert.doesNotMatch(setup, /fail\("模型运行配置下载失败/);
  assert.match(setup, /pipelineConfigsDeferred = result\.status !== 0/);
  assert.match(setup, /warn\(checkPipelineConfigs\(\["--required"\]\)/);
  assert.match(setup, /checkpointTask\("backend", backendFingerprint, pipelineConfigsDeferred \? "partial" : "complete"\)/);
  assert.match(setup, /taskCheckpoint\("backend", backendFingerprint, \["complete", "partial"\]\)/);
});

test("mirror routes reach Hugging Face through its mirror before the official host", async () => {
  const setup = await readFile(new URL("./setup.mjs", import.meta.url), "utf8");
  assert.match(setup, /official: \{[^}]*hf: "https:\/\/huggingface\.co"/);
  assert.match(setup, /aliyun: \{[^}]*hf: "https:\/\/hf-mirror\.com"/);
  assert.match(setup, /tsinghua: \{[^}]*hf: "https:\/\/hf-mirror\.com"/);
  assert.match(setup, /HF_ENDPOINT: process\.env\.HF_ENDPOINT \|\| source\.hf/);
});

test("xformers setup validates extension loading with a real CUDA attention operation", async () => {
  const setup = await readFile(new URL("./setup.mjs", import.meta.url), "utf8");
  assert.doesNotMatch(setup, /import torch,xformers,xformers\._C/);
  assert.match(setup, /xformers:cuda-smoke-v2/);
  assert.match(setup, /_cpp_library_load_exception is None/);
  assert.match(setup, /memory_efficient_attention\(query,query,query\)/);
  assert.match(setup, /actualXformers = pythonCheck\(venvPython, xformersCudaSmoke\(torchPlan\.version\)/);
});

test("the configurator offers Triton and SageAttention, defaulted off, with Sage implying Triton", async () => {
  const ui = await readFile(new URL("./setup-ui.html", import.meta.url), "utf8");
  // Neither carries `checked`: both are opt-in, like RTX VSR and unlike xformers.
  assert.match(ui, /id="tritonToggle"(?![^>]*checked)/);
  assert.match(ui, /id="sageAttentionToggle"(?![^>]*checked)/);
  assert.match(ui, /installTriton: elements\.tritonToggle\.checked \|\| elements\.sageAttentionToggle\.checked/);
  assert.match(ui, /installSageAttention: elements\.sageAttentionToggle\.checked/);
  // Ticking Sage shows Triton being selected; clearing Triton clears Sage.
  assert.match(ui, /sageAttentionToggle\.checked\) elements\.tritonToggle\.checked = true/);
  assert.match(ui, /!elements\.tritonToggle\.checked\) elements\.sageAttentionToggle\.checked = false/);
  // A re-run restores the previous answer from the setup marker.
  assert.match(ui, /elements\.tritonToggle\.checked = state\.configuration\.installTriton === true/);
  assert.match(ui, /elements\.sageAttentionToggle\.checked = state\.configuration\.installSageAttention === true/);
});

function acceleratorHarness({ checkpoints = {}, verifyResults = [], installStatus = 0 } = {}) {
  const calls = { uninstalled: [], installed: [], warnings: [], skipped: [], completed: [], checkpointed: [], started: [] };
  const verify = () => (verifyResults.length ? verifyResults.shift() : true);
  return {
    calls,
    hooks: {
      startTask: (id, detail) => calls.started.push([id, detail]),
      completeTask: (id) => calls.completed.push(id),
      skipTask: (id, detail) => calls.skipped.push([id, detail]),
      emit: (type, payload) => { if (type === "warning") calls.warnings.push(payload.message); },
      taskCheckpoint: (id, fingerprint, states) => Boolean(checkpoints[`${id}:${(states || ["complete"]).join(",")}`]),
      checkpointTask: (id, fingerprint, state) => calls.checkpointed.push([id, state || "complete"]),
      verify,
      uninstall: (names) => calls.uninstalled.push(names),
      install: async (args) => { calls.installed.push(args); return { status: installStatus }; },
    },
  };
}

const acceleratorRequest = {
  id: "triton", packageArgs: ["triton-windows"], smokeSource: "import triton",
  fingerprint: "fp", uninstallNames: ["triton-windows", "triton"],
  offLabel: "off", startDetail: "start",
  incompatibleMessage: "incompatible", unavailableMessage: "unavailable",
};

test("an unselected accelerator is actively removed, not merely skipped", async () => {
  const { calls, hooks } = acceleratorHarness();
  assert.equal(await installOptionalAccelerator({ ...acceleratorRequest, ...hooks, requested: false }), false);
  // Clearing the checkbox has to take the package away, or a stale install keeps
  // satisfying the capability probe and the Performance choice stays enabled.
  assert.deepEqual(calls.uninstalled, [["triton-windows", "triton"]]);
  assert.deepEqual(calls.skipped, [["triton", "off"]]);
  assert.deepEqual(calls.installed, []);
  assert.deepEqual(calls.started, []);
});

test("a verified accelerator completes, and a resumed one is not downloaded again", async () => {
  const fresh = acceleratorHarness();
  assert.equal(await installOptionalAccelerator({ ...acceleratorRequest, ...fresh.hooks, requested: true }), true);
  assert.deepEqual(fresh.calls.installed, [["triton-windows"]]);
  assert.deepEqual(fresh.calls.completed, ["triton"]);
  assert.deepEqual(fresh.calls.checkpointed, [["triton", "complete"]]);
  assert.deepEqual(fresh.calls.uninstalled, []);

  const resumed = acceleratorHarness({ checkpoints: { "triton:complete": true } });
  assert.equal(await installOptionalAccelerator({ ...acceleratorRequest, ...resumed.hooks, requested: true }), true);
  assert.deepEqual(resumed.calls.installed, []);
  assert.match(resumed.calls.skipped[0][1], /断点恢复/);

  // A resumed checkpoint whose package no longer verifies must reinstall rather
  // than trust the marker.
  const stale = acceleratorHarness({ checkpoints: { "triton:complete": true }, verifyResults: [false, true] });
  assert.equal(await installOptionalAccelerator({ ...acceleratorRequest, ...stale.hooks, requested: true }), true);
  assert.deepEqual(stale.calls.installed, [["triton-windows"]]);
});

test("an accelerator that installs but fails verification is uninstalled again", async () => {
  const { calls, hooks } = acceleratorHarness({ verifyResults: [false] });
  assert.equal(await installOptionalAccelerator({ ...acceleratorRequest, ...hooks, requested: true }), false);
  assert.deepEqual(calls.installed, [["triton-windows"]]);
  assert.deepEqual(calls.uninstalled, [["triton-windows", "triton"]]);
  assert.deepEqual(calls.warnings, ["incompatible"]);
  assert.deepEqual(calls.checkpointed, [["triton", "skipped"]]);
  assert.deepEqual(calls.completed, []);
});

test("an accelerator with no installable wheel warns and is checkpointed as skipped", async () => {
  const { calls, hooks } = acceleratorHarness({ installStatus: 1 });
  assert.equal(await installOptionalAccelerator({ ...acceleratorRequest, ...hooks, requested: true }), false);
  assert.deepEqual(calls.warnings, ["unavailable"]);
  assert.deepEqual(calls.checkpointed, [["triton", "skipped"]]);
  assert.deepEqual(calls.completed, []);

  // A previously skipped fingerprint is not retried within the same environment.
  const remembered = acceleratorHarness({ checkpoints: { "triton:skipped": true } });
  assert.equal(await installOptionalAccelerator({ ...acceleratorRequest, ...remembered.hooks, requested: true }), false);
  assert.deepEqual(remembered.calls.installed, []);
});

test("a package the runtime itself requires is never removed for being unselected", async () => {
  // On Linux the PyTorch wheel requires Triton outright. Removing it because the optional
  // accelerator was not ticked leaves `pip check` reporting "torch requires triton, which is not
  // installed", and setup fails its own final consistency check over a package it called optional.
  const { calls, hooks } = acceleratorHarness();
  assert.equal(await installOptionalAccelerator({
    ...acceleratorRequest,
    ...hooks,
    requested: false,
    protectedNames: ["filelock", "triton", "typing-extensions"],
    retainedLabel: "kept",
  }), false);
  assert.deepEqual(calls.uninstalled, [["triton-windows"]]);
  assert.deepEqual(calls.skipped, [["triton", "kept"]]);

  // Windows torch declares no Triton requirement, so nothing is protected and both names go.
  const windows = acceleratorHarness();
  assert.equal(await installOptionalAccelerator({
    ...acceleratorRequest,
    ...windows.hooks,
    requested: false,
    protectedNames: ["filelock", "typing-extensions"],
    retainedLabel: "kept",
  }), false);
  assert.deepEqual(windows.calls.uninstalled, [["triton-windows", "triton"]]);
  assert.deepEqual(windows.calls.skipped, [["triton", "off"]]);

  // A protected package that fails its smoke test stays installed too — it is still what torch
  // depends on; it simply does not unlock the accelerator.
  const failing = acceleratorHarness({ verifyResults: [false] });
  assert.equal(await installOptionalAccelerator({
    ...acceleratorRequest,
    ...failing.hooks,
    requested: true,
    protectedNames: ["triton", "triton-windows"],
  }), false);
  assert.deepEqual(failing.calls.uninstalled, []);
  assert.deepEqual(failing.calls.checkpointed, [["triton", "skipped"]]);

  // Distribution names normalise: `Triton_Windows` and `triton-windows` are one package.
  assert.deepEqual(removablePackages(["triton-windows", "triton"], ["Triton_Windows"]), ["triton"]);
  assert.deepEqual(removablePackages(["sageattention"], []), ["sageattention"]);
});

test("an explicitly configured torch index outranks the one a plan was resolved against", async () => {
  const setup = await readFile(new URL("./setup.mjs", import.meta.url), "utf8");
  // The plan is chosen by version, so the newest wheel can be found on the default index even when
  // XIRAI_TORCH_INDEX names a fast mirror — and then gigabytes come down the slow route the user
  // explicitly configured away from.
  assert.match(setup, /\.\.\.\(process\.env\.XIRAI_TORCH_INDEX \? forVariant\(process\.env\.XIRAI_TORCH_INDEX\) : \[\]\),\s*\n\s*plan\.indexUrl,/);
  // torchvision has to come from the same place, and before the backend dependencies can resolve
  // one of their own.
  assert.match(setup, /import \{[^}]*\bcheckForTorchvision\b[^}]*\} from "\.\/torch\.mjs"/);
  assert.match(setup, /const torchvisionCheck = checkForTorchvision\(torchPlan\)/);
  assert.match(setup, /from torchvision\.ops import nms/);
  assert.match(setup, /torchvision==\$\{installedTorchvision\.stdout\.trim\(\)\}/);
  // What that check actually accepts and rejects is proved by running it, in torch.test.mjs. It
  // lives beside the CUDA variant parsing it depends on for that reason: twice now a matching pair
  // has been reported as "no matching torchvision", and neither spelling of the bug was visible to
  // a test that only read this file.
});

test("the environment check proves the symbols Krea 2 loads through, not just the packages", async () => {
  const setup = await readFile(new URL("./setup.mjs", import.meta.url), "utf8");
  // `requirements.txt` floors transformers at 4.51 and diffusers at 0.38, but Krea 2 mounts a
  // Qwen3-VL encoder and a Wan autoencoder that arrived later than the transformers floor. Both
  // verification passes imported `Qwen3Config` and `Qwen3Model` and neither of these, so a
  // resolver that landed an older transformers would report a fully configured environment and
  // then fail at load time — with Krea 2 shipping as a ready engine.
  for (const symbol of ["AutoencoderKLWan", "Qwen3VLTextModel", "convert_wan_vae_to_diffusers"]) {
    const occurrences = setup.split(symbol).length - 1;
    assert.ok(occurrences >= 1, `the configurator never verifies ${symbol}, which Krea 2 imports`);
  }
  // Both passes, not only the first: the second is what the "verify" task reports green on.
  assert.equal(setup.split("AutoencoderKLWan").length - 1, 2);
  assert.equal(setup.split("Qwen3VLTextModel").length - 1, 2);
});

test("setup keeps Triton when PyTorch requires it rather than failing pip check", async () => {
  const setup = await readFile(new URL("./setup.mjs", import.meta.url), "utf8");
  // The markers decide it — the same wheel line requires Triton on Linux and not on Windows — so
  // they are evaluated by the interpreter that will run the environment.
  assert.match(setup, /function torchRequiredPackages\(\)/);
  assert.match(setup, /requirement\.marker is None or requirement\.marker\.evaluate\(\)/);
  assert.match(setup, /protectedNames: torchRequirements/);
});

test("setup wires the accelerators without a path that can fail the run", async () => {
  const setup = await readFile(new URL("./setup.mjs", import.meta.url), "utf8");
  assert.match(setup, /installOptionalAccelerator\(\{/);
  // The only `fail(` reachable from these options is the mutually-exclusive-flags guard.
  assert.match(setup, /if \(acceleratorChoiceError\) fail\(acceleratorChoiceError\)/);
  assert.doesNotMatch(setup, /fail\("[^"]*Triton/);
  assert.doesNotMatch(setup, /fail\("[^"]*SageAttention/);
  assert.match(setup, /uninstall: \(names\) =>/);
  assert.match(setup, /pip", "--isolated", "uninstall", "-y", \.\.\.names/);
  // A real setup run died here: `uvInstallFastest` probes through
  // `choosePackageSource`, which calls `fail()` when no route answers — so an
  // optional accelerator could end the whole run — and it probes by the first
  // argument, making `sageattention<2` a URL that cannot exist.
  assert.doesNotMatch(setup, /install: \(args\) => uvInstallFastest/);
  assert.match(setup, /const routeNames = \[\.\.\.new Set\(\[\.\.\.uniqueSources\("pip"\), "official"\]\)\]/);
  // Sage is pure Triton, so it never runs against a Triton that failed validation.
  assert.match(setup, /requested: requestedSageAttention && actualTriton/);
  // The marker records what was asked for and what actually survived validation.
  assert.match(setup, /tritonAvailable: actualTriton/);
  assert.match(setup, /sageAttentionAvailable: actualSageAttention/);
  // Distinct fingerprints, or one accelerator's checkpoint would answer for both.
  assert.match(setup, /triton:\$\{tritonPackage\}:inductor-smoke-v1/);
  assert.match(setup, /sageattention:cuda-smoke-v1/);
});
