function numericVersion(value) {
  const parts = String(value || "").match(/^\s*(\d+)\.(\d+)/);
  return parts ? [Number(parts[1]), Number(parts[2])] : null;
}

function versionAtLeast(value, minimum) {
  const parsed = numericVersion(value);
  return Boolean(parsed && (parsed[0] > minimum[0] || parsed[0] === minimum[0] && parsed[1] >= minimum[1]));
}

export function rtxVsrEligibility({
  platform = process.platform,
  architecture = process.arch,
  pythonVersion,
  driverVersion,
  driverModel,
  cudaAvailable = false,
  torchCudaVersion,
  computeCapability,
  isWsl = false,
} = {}) {
  if (!["win32", "linux"].includes(platform) || architecture !== "x64") return { supported: false, reason: "仅支持 x64 Windows / Linux" };
  if (isWsl) return { supported: false, reason: "WSL 原生运行库目前不稳定" };
  if (!versionAtLeast(pythonVersion, [3, 10])) return { supported: false, reason: "需要 Python 3.10 或更高版本" };
  if (cudaAvailable !== true || !numericVersion(torchCudaVersion)) return { supported: false, reason: "需要已实际验证的 CUDA PyTorch" };
  const capability = numericVersion(computeCapability);
  if (!capability || capability[0] < 7 || capability[0] === 7 && capability[1] < 5) return { supported: false, reason: "GPU 不具备受支持的 RTX Tensor Core" };
  let warning = null;
  if (platform === "win32") {
    if (!versionAtLeast(driverVersion, [570, 65])) warning = "NVIDIA 文档要求 Windows 驱动 570.65+；将以隔离进程真实探针作为最终判据";
    if (String(driverModel || "").trim().toUpperCase() === "TCC" && !versionAtLeast(driverVersion, [595, 0])) {
      warning = "NVIDIA 文档要求 Windows TCC 驱动 595+；将以隔离进程真实探针作为最终判据";
    }
  }
  if (platform === "linux") {
    const driver = numericVersion(driverVersion);
    const supportedBranch = driver && (driver[0] === 570 && driver[1] >= 190
      || driver[0] === 580 && driver[1] >= 82
      || driver[0] === 590 && driver[1] >= 44
      || driver[0] > 590);
    if (!supportedBranch) warning = "NVIDIA 文档要求 Linux 驱动 570.190+、580.82+ 或 590.44+；将以隔离进程真实探针作为最终判据";
  }
  return { supported: true, reason: null, warning };
}

export function resolveRtxVsrChoice(args = []) {
  const keys = new Set(args.map((argument) => String(argument).replace(/^--/, "").split("=", 1)[0]));
  const withRtxVsr = keys.has("with-rtx-vsr");
  const withoutRtxVsr = keys.has("without-rtx-vsr");
  if (withRtxVsr && withoutRtxVsr) throw new Error("--with-rtx-vsr 与 --without-rtx-vsr 不能同时使用");
  return withRtxVsr ? true : withoutRtxVsr ? false : null;
}

export const RTX_VSR_VERSION = "0.1.0.1";
export const RTX_VSR_INDEX = "https://pypi.nvidia.com/";
export const RTX_VSR_PRODUCT_SOURCE = "https://github.com/Comfy-Org/Nvidia_RTX_Nodes_ComfyUI";
