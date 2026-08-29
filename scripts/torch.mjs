import { runPython } from "./python.mjs";

function versionParts(value) {
  const match = value.match(/^(\d+)\.(\d+)\.(\d+)(?:\+([a-z0-9.]+))?$/i);
  return match ? [Number(match[1]), Number(match[2]), Number(match[3]), match[4] || ""] : null;
}

function compareVersions(first, second) {
  const a = versionParts(first);
  const b = versionParts(second);
  if (!a || !b) return 0;
  for (let index = 0; index < 3; index += 1) {
    if (a[index] !== b[index]) return b[index] - a[index];
  }
  return 0;
}

function cudaCapability(variant) {
  const digits = variant.match(/^cu(\d+)$/)?.[1];
  return digits ? Number(digits) : 0;
}

function cudaVariantVersion(variant) {
  const digits = variant.match(/^cu(\d+)$/)?.[1];
  if (!digits || digits.length < 2) return null;
  const majorLength = digits.length >= 3 ? 2 : 1;
  return {
    major: Number(digits.slice(0, majorLength)),
    minor: Number(digits.slice(majorLength)),
    text: `${Number(digits.slice(0, majorLength))}.${Number(digits.slice(majorLength))}`,
  };
}

function cudaVariantIsCompatible(variant, cuda) {
  const version = cudaVariantVersion(variant);
  if (!version || !cuda) return false;
  return version.major < cuda.major || version.major === cuda.major && version.minor <= cuda.minor;
}

// The driver's CUDA ceiling is only half the question. A wheel ships kernels for a fixed list of
// GPU architectures, and CUDA 13 dropped everything below Turing: a Tesla V100 (sm_70) under a
// CUDA 13 driver reports `torch.cuda.is_available() == True` on a cu130 build and then fails the
// first real operation with "no kernel image is available for execution on the device". So a
// variant is only usable if it was built for the card that is actually installed.
const CUDA_VARIANT_MINIMUM_COMPUTE = [
  { major: 13, minimum: { major: 7, minor: 5 } },
  { major: 12, minimum: { major: 5, minor: 0 } },
];

function parseComputeCapability(value) {
  if (!value) return null;
  if (typeof value === "object") {
    const major = Number(value.major);
    const minor = Number(value.minor);
    return Number.isInteger(major) && Number.isInteger(minor) ? { major, minor, text: `${major}.${minor}` } : null;
  }
  const match = String(value).trim().match(/^(\d+)\.(\d+)$/);
  return match ? { major: Number(match[1]), minor: Number(match[2]), text: `${match[1]}.${match[2]}` } : null;
}

function minimumComputeCapability(variant) {
  const version = cudaVariantVersion(variant);
  if (!version) return null;
  return CUDA_VARIANT_MINIMUM_COMPUTE.find((entry) => entry.major === version.major)?.minimum || null;
}

/** Whether a CUDA wheel carries kernels for this GPU. Unknown capability is not a veto: the wheel
 *  is still verified by a real CUDA operation before setup accepts it. */
function cudaVariantSupportsDevice(variant, computeCapability) {
  const minimum = minimumComputeCapability(variant);
  const capability = parseComputeCapability(computeCapability);
  if (!minimum || !capability) return true;
  return capability.major > minimum.major || capability.major === minimum.major && capability.minor >= minimum.minor;
}

function compatibleVariants(variants, hardware) {
  return [...new Set(variants)]
    .filter((variant) => {
      if (variant === "cpu") return !hardware?.available;
      return Boolean(hardware?.available
        && cudaVariantIsCompatible(variant, hardware.cuda)
        && cudaVariantSupportsDevice(variant, hardware.computeCapability));
    })
    .sort((first, second) => cudaCapability(second) - cudaCapability(first));
}

function comparePlans(first, second) {
  return compareVersions(first.version, second.version) || cudaCapability(second.variant) - cudaCapability(first.variant);
}

function directRootVariant(indexRoot) {
  try {
    return new URL(indexRoot).pathname.match(/\/(cu\d+|cpu)\/?$/i)?.[1]?.toLowerCase() || null;
  } catch {
    return String(indexRoot).match(/(?:^|\/)(cu\d+|cpu)\/?$/i)?.[1]?.toLowerCase() || null;
  }
}

function uniquePlans(plans) {
  const seen = new Set();
  return [...plans].sort(comparePlans).filter((plan) => {
    const key = `${plan.version}|${plan.variant}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function fetchIndex(python, url, environment) {
  const script = "import sys,urllib.request; sys.stdout.buffer.write(urllib.request.urlopen(sys.argv[1],timeout=12).read())";
  const result = runPython(python, ["-I", "-c", script, url], {
    env: environment,
    stdio: "pipe",
    encoding: "utf8",
    timeout: 15000,
  });
  return result.status === 0 ? result.stdout : "";
}

export function discoverTorchVariants(python, indexRoot, hardware, environment) {
  const directVariant = directRootVariant(indexRoot);
  if (directVariant) return compatibleVariants([directVariant], hardware);
  const html = fetchIndex(python, `${indexRoot.replace(/\/$/, "")}/`, environment);
  const variants = [...html.matchAll(/href=["'][^"']*\/?(cu\d+|cpu)\/?["']/gi)].map((match) => match[1].toLowerCase());
  return compatibleVariants(variants, hardware);
}

export function availableTorchVersions(python, indexUrl, environment) {
  const result = runPython(python, ["-m", "pip", "--isolated", "index", "versions", "torch", "--index-url", indexUrl], {
    env: environment,
    stdio: "pipe",
    encoding: "utf8",
    timeout: 30000,
  });
  if (result.status !== 0) return [];
  const output = `${result.stdout}\n${result.stderr || ""}`;
  const line = output.match(/Available versions:\s*([^\r\n]+)/i)?.[1] || "";
  return line.split(",").map((version) => version.trim()).filter((version) => versionParts(version)).sort(compareVersions);
}

export function resolveStableTorchPlans(python, indexRoots, _stableIndexUrls, hardware, environment) {
  const plans = [];
  for (const indexRoot of indexRoots) {
    const variants = discoverTorchVariants(python, indexRoot, hardware, environment);
    for (const variant of variants) {
      const directVariant = directRootVariant(indexRoot);
      const indexUrl = directVariant ? indexRoot.replace(/\/$/, "") : `${indexRoot.replace(/\/$/, "")}/${variant}`;
      const versions = availableTorchVersions(python, indexUrl, environment).sort(compareVersions);
      for (const version of versions) plans.push({ version, variant, indexUrl, indexRoot });
    }
  }
  return uniquePlans(plans);
}

export function resolveLatestStableTorch(python, indexRoots, stableIndexUrls, hardware, environment) {
  return resolveStableTorchPlans(python, indexRoots, stableIndexUrls, hardware, environment)[0] || null;
}

export const torchInternals = {
  versionParts,
  compareVersions,
  cudaCapability,
  cudaVariantVersion,
  cudaVariantIsCompatible,
  cudaVariantSupportsDevice,
  minimumComputeCapability,
  parseComputeCapability,
  compatibleVariants,
  comparePlans,
  directRootVariant,
  uniquePlans,
};
