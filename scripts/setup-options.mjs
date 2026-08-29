import { torchInternals } from "./torch.mjs";

function decodeWheelName(value) {
  const decodedEntities = String(value || "")
    .replace(/&amp;/g, "&")
    .replace(/&#x2B;/gi, "+")
    .replace(/&#43;/g, "+");
  try {
    return decodeURIComponent(decodedEntities);
  } catch {
    return decodedEntities;
  }
}

function platformWheelMatches(filename, platform, architecture) {
  if (platform === "win32") return filename.endsWith(architecture === "arm64" ? "-win_arm64.whl" : "-win_amd64.whl");
  if (platform === "linux") return filename.endsWith(architecture === "arm64" ? "_aarch64.whl" : "_x86_64.whl");
  return false;
}

export function parseCudaVariants(html) {
  const variants = [...String(html || "").matchAll(/(?:href=["'][^"']*\/)?(cu\d+)\/?["'<\s]/gi)]
    .map((match) => match[1].toLowerCase())
    .filter((variant) => {
      const version = torchInternals.cudaVariantVersion(variant);
      return version && (version.major > 11 || version.major === 11 && version.minor >= 8);
    });
  return [...new Set(variants)].sort((first, second) => torchInternals.cudaCapability(second) - torchInternals.cudaCapability(first));
}

export function parseTorchWheelVersions(html, { variant, pythonTag, platform, architecture }) {
  const versions = [];
  const wheelNames = String(html || "").match(/torch-[^\s"'<>]+\.whl/gi) || [];
  for (const encodedName of wheelNames) {
    const filename = decodeWheelName(encodedName);
    if (!filename.includes(`-${pythonTag}-${pythonTag}-`) || !platformWheelMatches(filename, platform, architecture)) continue;
    const version = filename.match(/^torch-(\d+\.\d+\.\d+\+(?:cu\d+|cpu))-/i)?.[1];
    if (!version || !version.toLowerCase().endsWith(`+${variant.toLowerCase()}`)) continue;
    versions.push(version);
  }
  return [...new Set(versions)].sort(torchInternals.compareVersions);
}

export function buildSetupCatalog({ hardware, pythonTag, variantVersions }) {
  const plans = [];
  for (const [variant, versions] of Object.entries(variantVersions)) {
    const compatible = Boolean(hardware?.available && torchInternals.cudaVariantIsCompatible(variant, hardware.cuda));
    for (const version of versions) plans.push({ version, variant, compatible });
  }
  plans.sort(torchInternals.comparePlans);
  const compatiblePlans = plans.filter((plan) => plan.compatible);
  const recommended = compatiblePlans[0] || null;
  const cudaOptions = Object.entries(variantVersions)
    .sort(([first], [second]) => torchInternals.cudaCapability(second) - torchInternals.cudaCapability(first))
    .map(([variant, versions]) => {
      const cudaRuntime = torchInternals.cudaVariantVersion(variant)?.text || variant;
      const compatible = Boolean(hardware?.available && torchInternals.cudaVariantIsCompatible(variant, hardware.cuda));
      return {
        variant,
        cudaRuntime,
        compatible,
        recommended: recommended?.variant === variant,
        reason: compatible ? null : `超过当前驱动 CUDA ${hardware?.cuda?.text || "未知"} 上限`,
        versions: versions.map((version) => {
          const isRecommended = version === recommended?.version && variant === recommended?.variant;
          let reason = null;
          if (!compatible) reason = `内置 CUDA ${cudaRuntime} 与当前驱动不兼容`;
          else if (!isRecommended && version !== recommended?.version) reason = "不是当前最新稳定 PyTorch";
          else if (!isRecommended) reason = "可用，但不是当前首选 CUDA runtime";
          return { version, recommended: isRecommended, discouraged: !isRecommended, reason };
        }),
      };
    });
  return { hardware, pythonTag, plans, cudaOptions, recommended };
}

export function recommendRepairPlan(catalog, failedSelection) {
  const compatible = (catalog?.plans || []).filter((plan) => plan.compatible);
  if (!compatible.length) return null;
  const failedVariant = failedSelection?.variant;
  const failedVersion = failedSelection?.torch;
  const repairAttempted = failedSelection?.repairAttempted === true;
  const failedCapability = torchInternals.cudaCapability(failedVariant || "");
  const recommended = catalog.recommended || compatible[0];
  if (!failedVariant || !failedVersion || (!repairAttempted && (failedVariant !== recommended.variant || failedVersion !== recommended.version))) return recommended;
  return compatible.find((plan) => torchInternals.cudaCapability(plan.variant) < failedCapability) || null;
}

export function buildSetupArguments(configuration) {
  const args = [];
  if (configuration.mode === "auto") {
    args.push("--torch=auto");
  } else {
    args.push(`--torch=${configuration.cudaVariant}`);
    args.push(`--torch-version=${configuration.torchVersion}`);
    args.push("--refresh-selection");
  }
  if (!configuration.installXformers) args.push("--without-xformers");
  args.push(configuration.installRtxVsr === true ? "--with-rtx-vsr" : "--without-rtx-vsr");
  args.push(configuration.installTriton === true ? "--with-triton" : "--without-triton");
  args.push(configuration.installSageAttention === true ? "--with-sageattention" : "--without-sageattention");
  return args;
}

export function validateSetupConfiguration(configuration, catalog) {
  const mode = configuration?.mode === "manual" ? "manual" : configuration?.mode === "auto" ? "auto" : null;
  if (!mode) return { ok: false, error: "请选择自动配置或手动配置" };
  // SageAttention 1.x is a pure-Triton package with no kernels of its own, so
  // choosing it without Triton would install something that cannot run. Selecting
  // Sage therefore selects Triton rather than being rejected for it.
  const installSageAttention = configuration.installSageAttention === true;
  const normalized = {
    mode,
    installXformers: configuration.installXformers !== false,
    installRtxVsr: configuration.installRtxVsr === true,
    installTriton: configuration.installTriton === true || installSageAttention,
    installSageAttention,
    autoRepair: configuration.autoRepair === true,
    repairAttempted: configuration.repairAttempted === true,
    repair: configuration.repair === true,
  };
  if (mode === "auto") {
    if (!catalog?.hardware?.available) return { ok: false, error: "自动配置需要可用的 NVIDIA GPU 和驱动" };
    if (!catalog.recommended) return { ok: false, error: "没有找到与当前 Python、平台和 NVIDIA 驱动兼容的稳定 PyTorch CUDA wheel" };
    return { ok: true, configuration: normalized };
  }
  const option = catalog?.cudaOptions?.find((item) => item.variant === configuration.cudaVariant);
  if (!option) return { ok: false, error: "所选 CUDA runtime 不在当前 wheel 目录中" };
  if (!option.compatible) return { ok: false, error: option.reason || "所选 CUDA runtime 与当前驱动不兼容" };
  if (!option.versions.some((item) => item.version === configuration.torchVersion)) {
    return { ok: false, error: "所选 PyTorch 版本没有适用于当前 Python 和平台的 wheel" };
  }
  return {
    ok: true,
    configuration: {
      ...normalized,
      cudaVariant: option.variant,
      torchVersion: configuration.torchVersion,
    },
  };
}
