// Optional Anima acceleration packages. Neither is required to generate: each
// unlocks a Performance choice, and the backend reports `triton_not_installed`
// or `sageattention_not_installed` rather than failing when one is absent.

export function resolveAcceleratorChoice(args = [], name) {
  const keys = new Set(args.map((argument) => String(argument).replace(/^--/, "").split("=", 1)[0]));
  const enabled = keys.has(`with-${name}`);
  const disabled = keys.has(`without-${name}`);
  if (enabled && disabled) throw new Error(`--with-${name} 与 --without-${name} 不能同时使用`);
  return enabled ? true : disabled ? false : null;
}

// Windows has no official Triton wheel; `triton-windows` is the distribution that
// carries one. Elsewhere the package is simply `triton`.
export function tritonPackageName(platform = process.platform) {
  return platform === "win32" ? "triton-windows" : "triton";
}

// PyPI publishes SageAttention only up to the pure-Triton 1.x line. 2.x needs an
// MSVC/CUDA source build, so it is deliberately not attempted here.
export const SAGE_ATTENTION_REQUIREMENT = "sageattention<2";

/** Import check for Triton plus the Inductor path `torch.compile` actually takes.
 *
 * The Inductor import is the part that matters: PyTorch reads its own bundled
 * Jinja templates with `open()` and no encoding, so on a non-UTF-8 locale the
 * import raises UnicodeDecodeError before any compilation starts. The backend
 * repairs that reader in `anima_pipeline.repair_inductor_template_encoding`; this
 * check applies the same repair so setup verifies what the runtime will do rather
 * than a weaker claim.
 */
export function tritonSmokeSource(expectedTorchVersion) {
  return [
    "import torch,triton",
    `assert torch.__version__==${JSON.stringify(expectedTorchVersion)}`,
    "from torch._inductor import utils as _u",
    // The parameter must stay named `template_dir`: `load_flex_template` calls this
    // by keyword, and a positional-only stand-in raises TypeError mid-import — which
    // would make setup uninstall a perfectly good Triton.
    "_u.load_template=lambda name,template_dir:open(__import__('pathlib').Path(template_dir)/(name+'.py.jinja'),encoding='utf-8').read()",
    "import torch._inductor.compile_fx",
    "from torch.utils._triton import has_triton_package",
    "assert has_triton_package()",
    "from torch._inductor.utils import triton_version_uses_attrs_dict as _t",
    "_t()",
  ].join(";");
}

// Normalised distribution name, so `triton_windows`, `Triton-Windows` and `triton-windows` are one
// package rather than three.
function packageKey(name) {
  return String(name).trim().toLowerCase().replace(/[-_.]+/g, "-");
}

/** The packages this accelerator may remove, minus the ones something else depends on.
 *
 * On Linux `torch` declares `triton` as a hard requirement, so removing Triton because the user did
 * not tick the optional accelerator leaves the environment inconsistent — `pip check` reports
 * "torch requires triton, which is not installed" and setup fails at its final consistency check,
 * for a package the user was told was optional. An optional feature switch must not be able to
 * break the runtime it merely accelerates.
 */
export function removablePackages(uninstallNames = [], protectedNames = []) {
  const keep = new Set(protectedNames.map(packageKey));
  return uninstallNames.filter((name) => !keep.has(packageKey(name)));
}

/** Install one optional accelerator, or make sure it is absent.
 *
 * Every exit is a skip or a completion — never a setup failure — and every path
 * that leaves the package unusable also removes it, so a half-installed wheel can
 * never satisfy a later import. Dependencies are injected so each branch is
 * testable without a Python environment or a network.
 *
 * @returns {Promise<boolean>} whether the package is installed *and* verified.
 */
export async function installOptionalAccelerator({
  id,
  requested,
  packageArgs,
  smokeSource,
  fingerprint,
  uninstallNames,
  protectedNames = [],
  offLabel,
  retainedLabel,
  startDetail,
  incompatibleMessage,
  unavailableMessage,
  startTask,
  completeTask,
  skipTask,
  emit,
  taskCheckpoint,
  checkpointTask,
  verify,
  uninstall,
  install,
}) {
  const removable = removablePackages(uninstallNames, protectedNames);
  const retained = removable.length < uninstallNames.length;
  if (!requested) {
    if (removable.length) uninstall(removable);
    skipTask(id, retained ? retainedLabel || offLabel : offLabel);
    return false;
  }
  startTask(id, startDetail);
  if (taskCheckpoint(id, fingerprint) && verify(smokeSource)) {
    emit("route", { id, label: "本地环境", connections: 0, mode: "断点恢复 · 无需下载" });
    skipTask(id, "断点恢复 · 已安装并通过验证");
    return true;
  }
  if (taskCheckpoint(id, fingerprint, ["skipped"])) {
    skipTask(id, "断点恢复 · 当前环境不兼容");
    return false;
  }
  const result = await install(packageArgs);
  if (result?.status !== 0) {
    emit("warning", { message: unavailableMessage });
    checkpointTask(id, fingerprint, "skipped");
    skipTask(id, "当前平台无兼容 wheel");
    return false;
  }
  if (!verify(smokeSource)) {
    if (removable.length) uninstall(removable);
    emit("warning", { message: incompatibleMessage });
    checkpointTask(id, fingerprint, "skipped");
    skipTask(id, removable.length ? "验证未通过，已安全移除" : "验证未通过，因运行时依赖保留但不启用");
    return false;
  }
  checkpointTask(id, fingerprint);
  completeTask(id);
  return true;
}

/** Real CUDA call in the layout and dtype the Anima processor uses. */
export function sageAttentionSmokeSource() {
  return [
    "import torch",
    "from sageattention import sageattn",
    "assert torch.cuda.is_available()",
    "q=torch.randn((1,4,256,128),device='cuda',dtype=torch.bfloat16)",
    "o=sageattn(q,q,q,tensor_layout='HND',is_causal=False)",
    "torch.cuda.synchronize()",
    "assert o.shape==q.shape and torch.isfinite(o).all().item()",
  ].join(";");
}
