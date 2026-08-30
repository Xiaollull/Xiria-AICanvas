/** Online update: what the release feed says, and how to fetch what it names.
 *
 * Everything here is pure. The network calls live in the update endpoints so that version
 * comparison, asset selection and route construction — the parts that decide whether a machine
 * replaces its own program files — can be tested without one.
 */

const DEFAULT_REPOSITORY = "Xiaollull/Xiria-AICanvas";
const RELEASE_ARCHIVE_PATTERN = /\.(?:zip|7z|tar|tar\.gz|tgz|tar\.xz|txz)$/i;
const CHECKSUM_SUFFIX = ".sha256";
const SHA256_PATTERN = /\b[a-f0-9]{64}\b/i;

/** `1.2.3`, `v1.2.3`, `1.2.3-beta.1`; anything else is not a version we will act on. */
export function parseVersion(value) {
  const match = String(value ?? "").trim().replace(/^v/i, "")
    .match(/^(\d+)\.(\d+)\.(\d+)(?:-([0-9a-z.-]+))?$/i);
  if (!match) return null;
  return {
    major: Number(match[1]),
    minor: Number(match[2]),
    patch: Number(match[3]),
    // A release carrying a pre-release tag sorts below the same numbers without one, per semver.
    prerelease: match[4] ? match[4].split(".") : null,
    text: match[0],
  };
}

function comparePrerelease(first, second) {
  if (!first && !second) return 0;
  if (!first) return 1;
  if (!second) return -1;
  for (let index = 0; index < Math.max(first.length, second.length); index += 1) {
    const left = first[index];
    const right = second[index];
    if (left === undefined) return -1;
    if (right === undefined) return 1;
    const leftNumeric = /^\d+$/.test(left);
    const rightNumeric = /^\d+$/.test(right);
    if (leftNumeric && rightNumeric) {
      if (Number(left) !== Number(right)) return Number(left) - Number(right);
      continue;
    }
    if (leftNumeric !== rightNumeric) return leftNumeric ? -1 : 1;
    if (left !== right) return left < right ? -1 : 1;
  }
  return 0;
}

/** Negative when `first` is older. Unparsable versions never compare as newer. */
export function compareVersions(first, second) {
  const a = parseVersion(first);
  const b = parseVersion(second);
  if (!a || !b) return 0;
  if (a.major !== b.major) return a.major - b.major;
  if (a.minor !== b.minor) return a.minor - b.minor;
  if (a.patch !== b.patch) return a.patch - b.patch;
  return comparePrerelease(a.prerelease, b.prerelease);
}

/** An update is offered only for a strictly newer, parsable version.
 *
 * A build running a version the feed does not know about — a local build ahead of the channel, or
 * a tag that is not a version at all — is left alone rather than "updated" backwards.
 */
export function updateAvailable(latest, current) {
  if (!parseVersion(latest) || !parseVersion(current)) return false;
  return compareVersions(latest, current) > 0;
}

export function releaseRepository(environment = process.env) {
  const configured = String(environment.XIRAI_UPDATE_REPO || "").trim();
  return /^[\w.-]+\/[\w.-]+$/.test(configured) ? configured : DEFAULT_REPOSITORY;
}

/** The release the update check reads. A full URL override is honoured verbatim. */
export function releaseFeedUrl(environment = process.env) {
  const configured = String(environment.XIRAI_UPDATE_FEED || "").trim();
  if (configured) return configured;
  return `https://api.github.com/repos/${releaseRepository(environment)}/releases/latest`;
}

function assetChecksum(asset) {
  // GitHub reports an asset digest as `sha256:<hex>`; it arrives over TLS from the API host, so it
  // is trustworthy in a way a mirror's copy of the bytes is not.
  const digest = String(asset?.digest || "");
  const match = digest.toLowerCase().startsWith("sha256:") && digest.slice(7).match(SHA256_PATTERN);
  return match ? match[0].toLowerCase() : null;
}

/** The archive a release offers, plus wherever its checksum can be had from. */
export function selectReleaseAsset(assets = []) {
  const archives = assets.filter((asset) => RELEASE_ARCHIVE_PATTERN.test(String(asset?.name || "")));
  if (!archives.length) return null;
  // Smallest first: a release may carry both a full package and a program-only archive, and the
  // updater replaces program files only.
  const chosen = archives.slice().sort((first, second) => (first.size || 0) - (second.size || 0))[0];
  const name = String(chosen.name);
  const sidecar = assets.find((asset) => String(asset?.name || "") === `${name}${CHECKSUM_SUFFIX}`);
  return {
    name,
    url: String(chosen.browser_download_url || ""),
    bytes: Number(chosen.size) || 0,
    sha256: assetChecksum(chosen),
    checksumUrl: sidecar ? String(sidecar.browser_download_url || "") : null,
  };
}

/** One release, reduced to what the update flow needs, or null when it offers nothing usable. */
export function parseRelease(payload) {
  if (!payload || typeof payload !== "object") return null;
  if (payload.draft) return null;
  const version = parseVersion(payload.tag_name || payload.name);
  if (!version) return null;
  const asset = selectReleaseAsset(Array.isArray(payload.assets) ? payload.assets : []);
  if (!asset || !asset.url) return null;
  return {
    version: version.text,
    prerelease: Boolean(payload.prerelease) || Boolean(version.prerelease),
    publishedAt: typeof payload.published_at === "string" ? payload.published_at : null,
    notes: typeof payload.body === "string" ? payload.body.trim() : "",
    asset,
  };
}

/** The checksum has to come from a route the mirrors cannot rewrite. */
export function checksumRoutes(release) {
  if (!release?.asset?.checksumUrl) return [];
  return [{ id: "release-checksum", label: "校验和 · GitHub", url: release.asset.checksumUrl }];
}

export function parseChecksumFile(text, assetName) {
  for (const line of String(text ?? "").split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const hash = trimmed.match(SHA256_PATTERN)?.[0];
    if (!hash) continue;
    // `SHA256SUMS` style lists name every file; a single-hash file names none.
    const named = trimmed.includes(assetName);
    if (named || !/\s/.test(trimmed)) return hash.toLowerCase();
  }
  return null;
}

/** Where the archive may be fetched from.
 *
 * The accelerated hosts are untrusted byte transports, exactly as they are for the uv bootstrap:
 * they are offered only when the archive's SHA-256 is known from the API or a checksum asset, so a
 * rewritten payload cannot be installed. Without a checksum the download is pinned to GitHub
 * itself, because replacing a user's program files with unverified bytes from a third-party proxy
 * is not a trade worth making for speed.
 */
export function releaseDownloadRoutes(release, { checksum, environment = process.env } = {}) {
  const url = release?.asset?.url;
  if (!url) return [];
  const official = { id: "release-github", label: "官方 · GitHub", url };
  const configuredBase = String(environment.XIRAI_UPDATE_MIRROR || "").trim().replace(/\/+$/, "");
  if (!checksum) return [official];
  const accelerated = configuredBase
    ? [{ id: "release-mirror", label: "自定义加速线路", url: `${configuredBase}/${url}` }]
    : [
      { id: "release-ghfast", label: "GitHub 加速 · ghfast", url: `https://ghfast.top/${url}` },
      { id: "release-ghproxy-net", label: "GitHub 加速 · ghproxy.net", url: `https://ghproxy.net/${url}` },
    ];
  return [...accelerated, official];
}

export const releaseFeedInternals = { DEFAULT_REPOSITORY, RELEASE_ARCHIVE_PATTERN, comparePrerelease };
