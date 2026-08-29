const uvVersion = "0.11.29";

const targets = {
  "aarch64-pc-windows-msvc": {
    assetName: "uv-aarch64-pc-windows-msvc.zip",
    sha256: "55b597ae81bc29531a7c352a1431a8a73cc2755d7a5b9ec454580cbe02e5154f",
    executable: "uv.exe",
    binaries: ["uv.exe", "uvx.exe", "uvw.exe"],
  },
  "x86_64-pc-windows-msvc": {
    assetName: "uv-x86_64-pc-windows-msvc.zip",
    sha256: "a047d55651bc3e0ca24595b25ec4cfcb10f9dca9fb56514e661269b37d4fae68",
    executable: "uv.exe",
    binaries: ["uv.exe", "uvx.exe", "uvw.exe"],
  },
  "aarch64-unknown-linux-gnu": {
    assetName: "uv-aarch64-unknown-linux-gnu.tar.gz",
    sha256: "94500fb064ae3c971a873cba64d94694c50677e0a4dbf78735c80509e7429919",
    executable: "uv",
    binaries: ["uv", "uvx"],
  },
  "aarch64-unknown-linux-musl": {
    assetName: "uv-aarch64-unknown-linux-musl.tar.gz",
    sha256: "593d79a797ece3f1dfaaf3e0a973263422a135d9262c7dbc6cd75d9c11acc0b4",
    executable: "uv",
    binaries: ["uv", "uvx"],
  },
  "x86_64-unknown-linux-gnu": {
    assetName: "uv-x86_64-unknown-linux-gnu.tar.gz",
    sha256: "04f8b82f5d47f0512dcd32c67a4a6f16a0ea27c81537c338fd0ad6b23cebe829",
    executable: "uv",
    binaries: ["uv", "uvx"],
  },
  "x86_64-unknown-linux-musl": {
    assetName: "uv-x86_64-unknown-linux-musl.tar.gz",
    sha256: "46711858adb2a3acaa9cee00f9060688ad1fd5706aecc005b96a6a7f285a00b7",
    executable: "uv",
    binaries: ["uv", "uvx"],
  },
};

function versionAtLeast(actual, minimum) {
  const actualParts = String(actual || "").split(".").map(Number);
  const minimumParts = String(minimum).split(".").map(Number);
  if (actualParts.some((part) => !Number.isInteger(part))) return false;
  for (let index = 0; index < Math.max(actualParts.length, minimumParts.length); index += 1) {
    const difference = (actualParts[index] || 0) - (minimumParts[index] || 0);
    if (difference) return difference > 0;
  }
  return true;
}

export function resolveUvTarget({ platform = process.platform, architecture = process.arch, glibcVersion = null } = {}) {
  const cpu = architecture === "arm64" ? "aarch64" : architecture === "x64" ? "x86_64" : null;
  if (!cpu) throw new Error(`Unsupported uv bootstrap CPU architecture: ${architecture}`);
  let target;
  if (platform === "win32") {
    target = `${cpu}-pc-windows-msvc`;
  } else if (platform === "linux") {
    const minimumGlibc = cpu === "aarch64" ? "2.28" : "2.17";
    const libc = glibcVersion && versionAtLeast(glibcVersion, minimumGlibc) ? "gnu" : "musl";
    target = `${cpu}-unknown-linux-${libc}`;
  } else {
    throw new Error(`Unsupported uv bootstrap platform: ${platform}`);
  }
  return { target, version: uvVersion, ...targets[target] };
}

function splitBases(value) {
  return String(value || "").trim().split(/\s+/).filter(Boolean).map((item) => item.replace(/\/$/, ""));
}

export function uvDownloadBases(environment = process.env) {
  const direct = environment.UV_DOWNLOAD_URL || environment.INSTALLER_DOWNLOAD_URL;
  if (direct) return splitBases(direct);
  const githubBase = environment.UV_INSTALLER_GHE_BASE_URL || environment.UV_INSTALLER_GITHUB_BASE_URL;
  if (githubBase) {
    return splitBases(githubBase).map((base) => `${base}/astral-sh/uv/releases/download/${uvVersion}`);
  }
  return [
    `https://releases.astral.sh/github/uv/releases/download/${uvVersion}`,
    `https://github.com/astral-sh/uv/releases/download/${uvVersion}`,
  ];
}

export function uvDownloadRoutes(target, environment = process.env) {
  const configured = environment.UV_DOWNLOAD_URL
    || environment.INSTALLER_DOWNLOAD_URL
    || environment.UV_INSTALLER_GHE_BASE_URL
    || environment.UV_INSTALLER_GITHUB_BASE_URL;
  if (configured) {
    return uvDownloadBases(environment).map((base, index) => ({
      id: `uv-custom-${index + 1}`,
      label: index === 0 ? "uv 自定义线路" : `uv 自定义备用 ${index + 1}`,
      url: `${base}/${target.assetName}`,
    }));
  }

  const githubUrl = `https://github.com/astral-sh/uv/releases/download/${uvVersion}/${target.assetName}`;
  return [
    { id: "uv-ghfast", label: "GitHub 加速 · ghfast", url: `https://ghfast.top/${githubUrl}` },
    { id: "uv-ghproxy-net", label: "GitHub 加速 · ghproxy.net", url: `https://ghproxy.net/${githubUrl}` },
    { id: "uv-astral", label: "uv 官方 · Astral", url: `https://releases.astral.sh/github/uv/releases/download/${uvVersion}/${target.assetName}` },
    { id: "uv-github", label: "uv 官方 · GitHub", url: githubUrl },
  ];
}

export function uvVersionMatches(output, version = uvVersion) {
  return String(output || "").trim().split(/\s+/)[1] === version;
}

export const uvBootstrapInternals = { targets, uvVersion, versionAtLeast };
