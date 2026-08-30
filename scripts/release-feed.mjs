/** Pure validation and routing for the online release feed. */

const DEFAULT_REPOSITORY = "Xiaollull/Xiria-AICanvas";
const DEFAULT_GITHUB_API_BASE = "https://api.github.com";
const CHECKSUM_SUFFIX = ".sha256";
const SHA256_PATTERN = /^[a-f0-9]{64}$/i;
const STABLE_VERSION_PATTERN = /^(?:v)?(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/;
const REPOSITORY_OWNER_PATTERN = /^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$/;
const REPOSITORY_NAME_PATTERN = /^[A-Za-z0-9](?:[A-Za-z0-9_.-]{0,98}[A-Za-z0-9_.-])?$/;

function configurationError(message) {
  return Object.assign(new Error(message), { code: "UPDATE_CONFIGURATION_INVALID", statusCode: 500 });
}

function releaseError(message) {
  return Object.assign(new Error(message), { code: "RELEASE_INVALID" });
}

/** Only stable MAJOR.MINOR.PATCH releases are actionable. A lower-case `v` is accepted for tags. */
export function parseVersion(value) {
  if (typeof value !== "string") return null;
  const match = value.match(STABLE_VERSION_PATTERN);
  if (!match) return null;
  const text = `${match[1]}.${match[2]}.${match[3]}`;
  return {
    // Keep canonical decimal strings: unlike Number this loses no precision, and unlike converting
    // an attacker-controlled million-digit component to BigInt it has predictable linear cost.
    major: match[1],
    minor: match[2],
    patch: match[3],
    prerelease: null,
    text,
  };
}

/** Negative when `first` is older. Invalid or unstable versions never compare as newer. */
export function compareVersions(first, second) {
  const a = parseVersion(first);
  const b = parseVersion(second);
  if (!a || !b) return 0;
  for (const key of ["major", "minor", "patch"]) {
    if (a[key].length !== b[key].length) return a[key].length < b[key].length ? -1 : 1;
    if (a[key] !== b[key]) return a[key] < b[key] ? -1 : 1;
  }
  return 0;
}

export function updateAvailable(latest, current) {
  if (!parseVersion(latest) || !parseVersion(current)) return false;
  return compareVersions(latest, current) > 0;
}

export function releaseRepository(environment = process.env) {
  const raw = environment.XIRAI_UPDATE_REPO;
  if (raw == null || raw === "") return DEFAULT_REPOSITORY;
  if (typeof raw !== "string" || raw !== raw.trim()) {
    throw configurationError("XIRAI_UPDATE_REPO 格式无效，应为 owner/repository");
  }
  const parts = raw.split("/");
  if (parts.length !== 2 || !REPOSITORY_OWNER_PATTERN.test(parts[0])
    || !REPOSITORY_NAME_PATTERN.test(parts[1]) || [".", ".."].includes(parts[1])) {
    throw configurationError("XIRAI_UPDATE_REPO 格式无效，应为 owner/repository");
  }
  return raw;
}

/**
 * GitHub Enterprise is opt-in through an HTTPS API base (for example
 * `https://github.example.com/api/v3`). This explicit base is also the host allow-list entry used
 * for PAT-bearing requests. An arbitrary XIRAI_UPDATE_FEED never becomes trusted merely by looking
 * like a GitHub URL.
 */
export function releaseApiBaseUrl(environment = process.env) {
  const raw = environment.XIRAI_UPDATE_GITHUB_API_BASE;
  if (raw == null || raw === "") return DEFAULT_GITHUB_API_BASE;
  if (typeof raw !== "string" || raw !== raw.trim()) {
    throw configurationError("XIRAI_UPDATE_GITHUB_API_BASE 必须是有效的 HTTPS 地址");
  }
  let parsed;
  try {
    parsed = new URL(raw);
  } catch {
    throw configurationError("XIRAI_UPDATE_GITHUB_API_BASE 必须是有效的 HTTPS 地址");
  }
  if (parsed.protocol !== "https:" || parsed.username || parsed.password || parsed.search || parsed.hash) {
    throw configurationError("XIRAI_UPDATE_GITHUB_API_BASE 必须是有效的 HTTPS 地址");
  }
  const pathname = parsed.pathname.replace(/\/+$/, "");
  return `${parsed.origin}${pathname}`;
}

/** The release endpoint. A custom feed is data-only and is never implicitly PAT-trusted. */
export function releaseFeedUrl(environment = process.env) {
  // Validate a nonempty repository setting even when a custom feed is selected. Silently ignoring
  // an invalid security-relevant setting makes a later switch away from the custom feed target the
  // default repository, which is precisely the fallback this validator exists to prevent.
  const repository = releaseRepository(environment);
  const configured = environment.XIRAI_UPDATE_FEED;
  if (configured != null && configured !== "") {
    if (typeof configured !== "string" || configured !== configured.trim()) {
      throw configurationError("XIRAI_UPDATE_FEED 必须是有效的 HTTPS 地址");
    }
    let parsed;
    try {
      parsed = new URL(configured);
    } catch {
      throw configurationError("XIRAI_UPDATE_FEED 必须是有效的 HTTPS 地址");
    }
    if (parsed.protocol !== "https:" || parsed.username || parsed.password) {
      throw configurationError("XIRAI_UPDATE_FEED 必须是有效的 HTTPS 地址且不能包含凭据");
    }
    return parsed.href;
  }
  return `${releaseApiBaseUrl(environment)}/repos/${repository}/releases/latest`;
}

export function releaseRepositoryUrl(environment = process.env) {
  return `${releaseApiBaseUrl(environment)}/repos/${releaseRepository(environment)}`;
}

export function hasCustomReleaseFeed(environment = process.env) {
  return environment.XIRAI_UPDATE_FEED != null && environment.XIRAI_UPDATE_FEED !== "";
}

/** Token-bearing requests may go only to official GitHub or the explicitly configured GHE host. */
export function trustedGithubUrl(value, environment = process.env) {
  let parsed;
  try {
    parsed = new URL(value);
  } catch {
    return false;
  }
  if (parsed.protocol !== "https:" || parsed.username || parsed.password) return false;
  const origins = new Set(["https://api.github.com", "https://github.com"]);
  try {
    origins.add(new URL(releaseApiBaseUrl(environment)).origin.toLowerCase());
  } catch {
    return false;
  }
  return origins.has(parsed.origin.toLowerCase());
}

export function updateAuthorizationHeaders(value, {
  environment = process.env,
  allowToken = true,
} = {}) {
  const token = typeof environment.XIRAI_UPDATE_TOKEN === "string"
    ? environment.XIRAI_UPDATE_TOKEN.trim()
    : "";
  return token && allowToken && trustedGithubUrl(value, environment)
    ? { Authorization: `Bearer ${token}` }
    : {};
}

export function missingReleaseMeaning({ customFeed = false, repositoryReachable = false } = {}) {
  if (customFeed) return "source-missing";
  return repositoryReachable ? "no-release" : "source-missing";
}

function absoluteHttpsUrl(value, label) {
  let parsed;
  try {
    parsed = new URL(value);
  } catch {
    throw releaseError(`${label}不是有效的 HTTPS 地址`);
  }
  if (parsed.protocol !== "https:" || parsed.username || parsed.password) {
    throw releaseError(`${label}不是有效的 HTTPS 地址`);
  }
  return parsed.href;
}

function assetChecksum(asset) {
  if (asset?.digest == null || asset.digest === "") return null;
  const digest = String(asset.digest);
  const match = digest.match(/^sha256:([a-f0-9]{64})$/i);
  if (!match) throw releaseError("发布资产的 SHA-256 摘要格式无效");
  return match[1].toLowerCase();
}

function exactUploadedAsset(assets, name, label, { required = true } = {}) {
  const named = assets.filter((asset) => asset && typeof asset === "object" && asset.name === name);
  if (!named.length) {
    if (!required) return null;
    throw releaseError(`发布信息中缺少精确命名的${label} ${name}`);
  }
  if (named.length !== 1) throw releaseError(`发布信息中存在多个同名${label} ${name}`);
  if (named[0].state !== "uploaded") throw releaseError(`${label} ${name} 尚未完成上传`);
  return named[0];
}

function assetUrls(asset, label) {
  const browserUrl = absoluteHttpsUrl(asset.browser_download_url, `${label}下载地址`);
  const apiUrl = asset.url == null || asset.url === "" ? null : absoluteHttpsUrl(asset.url, `${label} API 地址`);
  return { browserUrl, apiUrl };
}

/** Select exactly XirAI-${version}.7z and its exact sidecar; there is no archive fallback. */
export function selectReleaseAsset(assets = [], version = "") {
  const parsedVersion = parseVersion(version);
  if (!parsedVersion) throw releaseError("发布版本必须是稳定的 MAJOR.MINOR.PATCH");
  if (!Array.isArray(assets)) throw releaseError("发布资产列表格式无效");
  const name = `XirAI-${parsedVersion.text}.7z`;
  const archive = exactUploadedAsset(assets, name, "更新包");
  const sidecarName = `${name}${CHECKSUM_SUFFIX}`;
  const sidecar = exactUploadedAsset(assets, sidecarName, "校验和文件");
  const size = archive.size;
  if (!Number.isSafeInteger(size) || size <= 0) throw releaseError("发布的更新包大小无效");
  const sidecarSize = sidecar.size;
  if (!Number.isSafeInteger(sidecarSize) || sidecarSize <= 0 || sidecarSize > 16 * 1024) {
    throw releaseError("发布的校验和文件大小无效或超过 16 KiB");
  }
  const archiveUrls = assetUrls(archive, "更新包");
  const sidecarUrls = sidecar ? assetUrls(sidecar, "校验和文件") : null;
  return {
    name,
    url: archiveUrls.browserUrl,
    apiUrl: archiveUrls.apiUrl,
    bytes: size,
    advertisedSha256: assetChecksum(archive),
    checksumUrl: sidecarUrls?.browserUrl || null,
    checksumApiUrl: sidecarUrls?.apiUrl || null,
  };
}

/** Reduce a stable, published release to the fields the update flow is allowed to use. */
export function parseRelease(payload) {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw releaseError("更新服务器返回的发布信息格式无效");
  }
  if (payload.draft) throw releaseError("发布信息仍是草稿，不能用于在线更新");
  if (payload.prerelease) throw releaseError("在线更新仅接受稳定版本，不能使用预发布版本");
  const version = parseVersion(payload.tag_name);
  if (!version) throw releaseError("发布标签必须是稳定的 MAJOR.MINOR.PATCH");
  const asset = selectReleaseAsset(payload.assets, version.text);
  return {
    version: version.text,
    prerelease: false,
    publishedAt: typeof payload.published_at === "string" ? payload.published_at : null,
    notes: typeof payload.body === "string" ? payload.body.trim() : "",
    asset,
  };
}

function authenticatedAssetRoute(asset, kind, { environment, allowToken }) {
  const browserUrl = kind === "checksum" ? asset.checksumUrl : asset.url;
  const apiUrl = kind === "checksum" ? asset.checksumApiUrl : asset.apiUrl;
  const token = typeof environment.XIRAI_UPDATE_TOKEN === "string" && environment.XIRAI_UPDATE_TOKEN.trim();
  const url = token && allowToken && apiUrl && trustedGithubUrl(apiUrl, environment) ? apiUrl : browserUrl;
  if (!url) return null;
  return {
    url,
    headers: {
      ...(apiUrl && url === apiUrl ? { Accept: "application/octet-stream" } : {}),
      ...updateAuthorizationHeaders(url, { environment, allowToken }),
    },
  };
}

function hasAuthorizationHeader(headers) {
  return Object.keys(headers || {}).some((name) => name.toLowerCase() === "authorization");
}

function publicGithubBrowserAssetUrl(value) {
  try {
    const parsed = new URL(value);
    return parsed.origin.toLowerCase() === "https://github.com"
      && !parsed.username && !parsed.password && !parsed.search && !parsed.hash
      && /^\/[^/]+\/[^/]+\/releases\/download\//.test(parsed.pathname);
  } catch {
    return false;
  }
}

/** Build the proxy convention `<mirror-base>/<complete-browser-url>` without URL interpolation.
 *
 * The mirror is an explicitly trusted transport, not an authenticated GitHub endpoint. Its base
 * therefore has to be credential-free HTTPS, and the source URL must not contain query data that
 * could be a signed/private download credential. Route headers are deliberately not accepted here.
 */
function explicitMirrorUrl(configured, sourceUrl) {
  if (configured == null || configured === "") return null;
  if (typeof configured !== "string" || configured !== configured.trim()) {
    throw configurationError("XIRAI_UPDATE_MIRROR 必须是有效的 HTTPS 地址");
  }
  let mirror;
  let source;
  try {
    mirror = new URL(configured);
    source = new URL(sourceUrl);
  } catch {
    throw configurationError("XIRAI_UPDATE_MIRROR 必须是有效的 HTTPS 地址");
  }
  if (mirror.protocol !== "https:" || mirror.username || mirror.password || mirror.search || mirror.hash) {
    throw configurationError("XIRAI_UPDATE_MIRROR 必须是无凭据、无查询参数的 HTTPS 地址");
  }
  // Queries on asset URLs may be short-lived credentials. An explicit mirror may learn the public
  // browser path, but it must never receive those credentials (or a fragment that obscures them).
  if (source.protocol !== "https:" || source.username || source.password || source.search || source.hash) return null;
  mirror.pathname = `${mirror.pathname.replace(/\/+$/, "")}/${source.href}`;
  return mirror.href;
}

export function checksumRoutes(release, {
  environment = process.env,
  allowToken = true,
} = {}) {
  if (!release?.asset?.checksumUrl) return [];
  const request = authenticatedAssetRoute(release.asset, "checksum", { environment, allowToken });
  return request ? [{ id: "release-checksum", label: "校验和 · GitHub", ...request }] : [];
}

/** Parse exactly one sha256sum line whose complete filename is the selected archive name. */
export function parseChecksumFile(text, assetName) {
  if (typeof text !== "string" || typeof assetName !== "string" || !assetName) return null;
  const match = text.match(/^([a-f0-9]{64}) ([ *])([^\r\n]+)(?:\r?\n)?$/i);
  if (!match || !SHA256_PATTERN.test(match[1]) || match[3] !== assetName) return null;
  return match[1].toLowerCase();
}

/** Only checksum-pinned archive routes are installable. */
export function releaseDownloadRoutes(release, {
  checksum,
  environment = process.env,
  allowToken = true,
  privateContext = false,
} = {}) {
  if (!release?.asset?.url || !SHA256_PATTERN.test(String(checksum || ""))) return [];
  const request = authenticatedAssetRoute(release.asset, "archive", { environment, allowToken });
  if (!request) return [];
  const official = { id: "release-github", label: "官方 · GitHub", ...request };
  const officialUsesAssetApi = Boolean(release.asset.apiUrl && request.url === release.asset.apiUrl);
  const privateOfficial = privateContext || officialUsesAssetApi || hasAuthorizationHeader(request.headers)
    // The built-ins are GitHub browser-URL proxies. Unknown hosts, API paths, and signed URLs are
    // not established-public assets and are kept on the official route unless a mirror was chosen.
    || !publicGithubBrowserAssetUrl(release.asset.url);
  const mirrorUrl = explicitMirrorUrl(environment.XIRAI_UPDATE_MIRROR, release.asset.url);
  if (mirrorUrl) {
    return [
      // Never copy `official.headers`: a user-selected mirror receives no GitHub Authorization.
      { id: "release-mirror", label: "自定义加速线路", url: mirrorUrl },
      official,
    ];
  }
  // A configured mirror whose source URL contained possible credentials was intentionally skipped.
  if (environment.XIRAI_UPDATE_MIRROR != null && environment.XIRAI_UPDATE_MIRROR !== "") return [official];
  // Built-in proxies are public services. Even benchmarking them discloses the complete release
  // browser path, so an API/PAT/private route must never put them in the candidate set.
  if (privateOfficial) return [official];
  return [
    { id: "release-ghfast", label: "GitHub 加速 · ghfast", url: `https://ghfast.top/${release.asset.url}` },
    { id: "release-ghproxy-net", label: "GitHub 加速 · ghproxy.net", url: `https://ghproxy.net/${release.asset.url}` },
    official,
  ];
}

export const releaseFeedInternals = {
  CHECKSUM_SUFFIX,
  DEFAULT_GITHUB_API_BASE,
  DEFAULT_REPOSITORY,
  SHA256_PATTERN,
  STABLE_VERSION_PATTERN,
};
