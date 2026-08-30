import assert from "node:assert/strict";
import test from "node:test";
import {
  checksumRoutes,
  compareVersions,
  hasCustomReleaseFeed,
  missingReleaseMeaning,
  parseChecksumFile,
  parseRelease,
  parseVersion,
  releaseApiBaseUrl,
  releaseDownloadRoutes,
  releaseFeedUrl,
  releaseRepository,
  releaseRepositoryUrl,
  selectReleaseAsset,
  trustedGithubUrl,
  updateAuthorizationHeaders,
  updateAvailable,
} from "./release-feed.mjs";

const sha = "a".repeat(64);

function uploadedAsset(name, {
  size = 1024,
  digest = null,
  state = "uploaded",
  browserHost = "github.com",
  apiHost = "api.github.com",
} = {}) {
  return {
    name,
    state,
    size,
    digest,
    browser_download_url: `https://${browserHost}/owner/repo/releases/download/v1.2.3/${name}`,
    url: `https://${apiHost}/repos/owner/repo/releases/assets/${encodeURIComponent(name)}`,
  };
}

function release(overrides = {}) {
  const version = overrides.version || "1.2.3";
  const archiveName = `XirAI-${version}.7z`;
  return {
    tag_name: `v${version}`,
    draft: false,
    prerelease: false,
    published_at: "2026-09-01T10:00:00Z",
    body: "  修复若干问题  ",
    assets: [
      uploadedAsset(archiveName, { size: 734003200, digest: `sha256:${sha}` }),
      uploadedAsset(`${archiveName}.sha256`, { size: 89 }),
    ],
    ...overrides,
  };
}

test("stable versions are exact MAJOR.MINOR.PATCH values with precision-safe numeric components", () => {
  assert.equal(parseVersion("1.2.3").text, "1.2.3");
  assert.equal(parseVersion("v1.2.3").text, "1.2.3");
  const huge = "900719925474099312345678901234567890.2.3";
  assert.equal(parseVersion(huge).major, huge.split(".")[0]);

  for (const rejected of [
    "", null, undefined, "V1.2.3", " 1.2.3", "1.2.3 ", "01.2.3", "1.02.3", "1.2.03",
    "1.2", "1.2.3.4", "1.2.3-", "1.2.3-beta", "1.2.3+build", "latest",
  ]) {
    assert.equal(parseVersion(rejected), null, String(rejected));
  }
});

test("version ordering does not lose precision and unstable values never become newer", () => {
  assert.ok(compareVersions("1.0.1", "1.0.0") > 0);
  assert.ok(compareVersions("1.1.0", "1.0.999999999999999999999999999999") > 0);
  assert.ok(compareVersions("900719925474099312345678901234567891.0.0", "900719925474099312345678901234567890.999.999") > 0);
  assert.equal(compareVersions("v1.2.3", "1.2.3"), 0);
  assert.equal(compareVersions("1.2.3-beta.1", "1.2.3"), 0);
  assert.equal(updateAvailable("1.2.4", "1.2.3"), true);
  assert.equal(updateAvailable("1.2.3", "1.2.3"), false);
  assert.equal(updateAvailable("1.2.3", "1.2.4"), false);
  assert.equal(updateAvailable("1.2.4-beta", "1.2.3"), false);
  assert.equal(updateAvailable("1.2.4", "dev"), false);
});

test("repository configuration defaults only when unset and rejects every invalid nonempty value", () => {
  assert.equal(releaseRepository({}), "Xiaollull/Xiria-AICanvas");
  assert.equal(releaseRepository({ XIRAI_UPDATE_REPO: "" }), "Xiaollull/Xiria-AICanvas");
  assert.equal(releaseRepository({ XIRAI_UPDATE_REPO: "someone/fork.name" }), "someone/fork.name");
  for (const invalid of [" ", "not a repo", "owner", "/repo", "owner/", "owner/repo/extra", "-owner/repo", "owner/.repo", "owner/repo "]) {
    assert.throws(() => releaseRepository({ XIRAI_UPDATE_REPO: invalid }), /XIRAI_UPDATE_REPO 格式无效/, invalid);
  }
  assert.equal(releaseFeedUrl({}), "https://api.github.com/repos/Xiaollull/Xiria-AICanvas/releases/latest");
  assert.equal(releaseRepositoryUrl({ XIRAI_UPDATE_REPO: "o/r" }), "https://api.github.com/repos/o/r");
});

test("custom feeds and GitHub Enterprise API bases must be credential-free HTTPS URLs", () => {
  assert.equal(releaseFeedUrl({ XIRAI_UPDATE_FEED: "https://updates.example/latest.json" }), "https://updates.example/latest.json");
  assert.equal(hasCustomReleaseFeed({ XIRAI_UPDATE_FEED: "https://updates.example/latest.json" }), true);
  assert.equal(hasCustomReleaseFeed({}), false);
  for (const invalid of ["http://updates.example/latest", "https://u:p@updates.example/latest", "ftp://updates.example/latest", " https://updates.example/latest"] ) {
    assert.throws(() => releaseFeedUrl({ XIRAI_UPDATE_FEED: invalid }), /HTTPS/, invalid);
  }
  const ghe = { XIRAI_UPDATE_GITHUB_API_BASE: "https://git.example/api/v3/", XIRAI_UPDATE_REPO: "o/r" };
  assert.equal(releaseApiBaseUrl(ghe), "https://git.example/api/v3");
  assert.equal(releaseFeedUrl(ghe), "https://git.example/api/v3/repos/o/r/releases/latest");
  for (const invalid of ["http://git.example/api/v3", "https://u:p@git.example/api/v3", "https://git.example/api/v3?q=1"]) {
    assert.throws(() => releaseApiBaseUrl({ XIRAI_UPDATE_GITHUB_API_BASE: invalid }), /HTTPS/);
  }
});

test("only the exact uploaded archive and exact uploaded sidecar are selected", () => {
  const payload = release({
    assets: [
      uploadedAsset("source.tar.gz", { size: 1 }),
      uploadedAsset("XirAI-full-1.2.3.7z", { size: 2 }),
      uploadedAsset("evil-XirAI-1.2.3.7z", { size: 3 }),
      uploadedAsset("XirAI-1.2.3.7z", { size: 700_000_000, digest: `sha256:${"b".repeat(64)}` }),
      uploadedAsset("XirAI-1.2.3.7z.sha256", { size: 89 }),
    ],
  });
  const asset = selectReleaseAsset(payload.assets, "1.2.3");
  assert.equal(asset.name, "XirAI-1.2.3.7z");
  assert.equal(asset.bytes, 700_000_000);
  assert.equal(asset.advertisedSha256, "b".repeat(64));
  assert.ok(asset.checksumUrl.endsWith("XirAI-1.2.3.7z.sha256"));
});

test("asset selection rejects fallback, ambiguity, non-uploaded state, and unsafe metadata", () => {
  const exactArchive = uploadedAsset("XirAI-1.2.3.7z");
  const exactSidecar = uploadedAsset("XirAI-1.2.3.7z.sha256", { size: 89 });
  const select = (assets) => selectReleaseAsset(assets, "1.2.3");

  assert.throws(() => select([uploadedAsset("XirAI-full-1.2.3.7z"), exactSidecar]), /缺少精确命名的更新包/);
  assert.throws(() => select([uploadedAsset("XirAI-1.2.3.7z.exe"), exactSidecar]), /缺少精确命名的更新包/);
  assert.throws(() => select([exactArchive]), /缺少精确命名的校验和文件/);
  assert.throws(() => select([exactArchive, exactArchive, exactSidecar]), /多个同名更新包/);
  assert.throws(() => select([exactArchive, exactSidecar, exactSidecar]), /多个同名校验和文件/);
  assert.throws(() => select([{ ...exactArchive, state: "starter" }, exactSidecar]), /尚未完成上传/);
  assert.throws(() => select([exactArchive, { ...exactSidecar, state: "new" }]), /尚未完成上传/);
  assert.throws(() => select([{ ...exactArchive, browser_download_url: "http://github.com/file" }, exactSidecar]), /HTTPS/);
  assert.throws(() => select([{ ...exactArchive, browser_download_url: "https://u:p@github.com/file" }, exactSidecar]), /HTTPS/);
  assert.throws(() => select([{ ...exactArchive, size: 0 }, exactSidecar]), /大小无效/);
  assert.throws(() => select([{ ...exactArchive, size: "1024" }, exactSidecar]), /大小无效/);
  assert.throws(() => select([exactArchive, { ...exactSidecar, size: "89" }]), /大小无效/);
  assert.throws(() => select([exactArchive, { ...exactSidecar, size: 16 * 1024 + 1 }]), /16 KiB/);
  assert.throws(() => select([{ ...exactArchive, digest: `sha256:${sha}tail` }, exactSidecar]), /摘要格式无效/);
});

test("release parsing exposes invalid releases instead of treating them as latest", () => {
  const parsed = parseRelease(release());
  assert.equal(parsed.version, "1.2.3");
  assert.equal(parsed.prerelease, false);
  assert.equal(parsed.notes, "修复若干问题");
  assert.equal(parsed.asset.advertisedSha256, sha);

  assert.throws(() => parseRelease(null), /格式无效/);
  assert.throws(() => parseRelease(release({ draft: true })), /草稿/);
  assert.throws(() => parseRelease(release({ prerelease: true })), /预发布/);
  assert.throws(() => parseRelease(release({ tag_name: "v1.2.3-beta.1" })), /稳定/);
  assert.throws(() => parseRelease(release({ tag_name: "v01.2.3" })), /稳定/);
  assert.throws(() => parseRelease(release({ tag_name: "nightly" })), /稳定/);
  assert.throws(() => parseRelease(release({ assets: [uploadedAsset("../XirAI-1.2.3.7z"), uploadedAsset("../XirAI-1.2.3.7z.sha256", { size: 89 })] })), /缺少精确命名/);
});

test("sha256sum sidecars require one exact complete filename", () => {
  assert.equal(parseChecksumFile(`${sha}  XirAI-1.2.3.7z\n`, "XirAI-1.2.3.7z"), sha);
  assert.equal(parseChecksumFile(`${sha} *XirAI-1.2.3.7z\r\n`, "XirAI-1.2.3.7z"), sha);
  for (const malicious of [
    `${sha}\n`,
    `${sha} XirAI-1.2.3.7z\n`,
    `${sha}  prefix-XirAI-1.2.3.7z\n`,
    `${sha}  ./XirAI-1.2.3.7z\n`,
    `${sha}  XirAI-1.2.3.7z.exe\n`,
    `${sha}  XirAI-1.2.3.7z\n${"b".repeat(64)}  other\n`,
    `${sha}  XirAI-1.2.3.7z\n\n`,
    `xx${sha}  XirAI-1.2.3.7z\n`,
    `${sha.toUpperCase()}  XirAI-1.2.3.7z extra\n`,
  ]) {
    assert.equal(parseChecksumFile(malicious, "XirAI-1.2.3.7z"), null, malicious);
  }
});

test("public download routes fail closed without a resolved SHA-256 and retain built-in acceleration", () => {
  const parsed = parseRelease(release());
  assert.deepEqual(releaseDownloadRoutes(parsed, { checksum: null, environment: {} }), []);
  assert.deepEqual(releaseDownloadRoutes(parsed, { checksum: "short", environment: {} }), []);
  const routes = releaseDownloadRoutes(parsed, { checksum: sha, environment: {} });
  assert.deepEqual(routes.map((route) => route.id), ["release-ghfast", "release-ghproxy-net", "release-github"]);
  assert.ok(routes.slice(0, 2).every((route) => !new Headers(route.headers).has("authorization")));
  const custom = releaseDownloadRoutes(parsed, {
    checksum: sha,
    environment: { XIRAI_UPDATE_MIRROR: "https://mirror.example/gh/" },
  });
  assert.deepEqual(custom.map((route) => route.id), ["release-mirror", "release-github"]);
  assert.equal(custom[0].url, `https://mirror.example/gh/${parsed.asset.url}`);
  assert.equal(new Headers(custom[0].headers).has("authorization"), false);
});

test("PAT headers are limited to HTTPS GitHub or the explicit GHE host", () => {
  const environment = { XIRAI_UPDATE_TOKEN: " secret ", XIRAI_UPDATE_GITHUB_API_BASE: "https://git.example/api/v3" };
  assert.deepEqual(updateAuthorizationHeaders("https://api.github.com/repos/o/r", { environment }), { Authorization: "Bearer secret" });
  assert.deepEqual(updateAuthorizationHeaders("https://github.com/o/r/file", { environment }), { Authorization: "Bearer secret" });
  assert.deepEqual(updateAuthorizationHeaders("https://git.example/api/v3/repos/o/r", { environment }), { Authorization: "Bearer secret" });
  assert.deepEqual(updateAuthorizationHeaders("http://api.github.com/repos/o/r", { environment }), {});
  assert.deepEqual(updateAuthorizationHeaders("https://evil.example/file", { environment }), {});
  assert.deepEqual(updateAuthorizationHeaders("https://api.github.com.evil.example/file", { environment }), {});
  assert.deepEqual(updateAuthorizationHeaders("https://api.github.com:444/file", { environment }), {});
  assert.deepEqual(updateAuthorizationHeaders("https://u:p@api.github.com/file", { environment }), {});
  assert.deepEqual(updateAuthorizationHeaders("https://api.github.com/file", { environment, allowToken: false }), {});
  assert.equal(trustedGithubUrl("https://git.example/file", environment), true);
  assert.equal(trustedGithubUrl("http://git.example/file", environment), false);
});

test("private GitHub assets use only the authenticated official API route by default", () => {
  const parsed = parseRelease(release());
  const environment = { XIRAI_UPDATE_TOKEN: "secret" };
  const [checksum] = checksumRoutes(parsed, { environment });
  assert.equal(checksum.url, parsed.asset.checksumApiUrl);
  assert.equal(checksum.headers.Authorization, "Bearer secret");
  assert.equal(checksum.headers.Accept, "application/octet-stream");
  const routes = releaseDownloadRoutes(parsed, { checksum: sha, environment });
  assert.deepEqual(routes.map((route) => route.id), ["release-github"]);
  const [official] = routes;
  assert.equal(official.url, parsed.asset.apiUrl);
  assert.equal(official.headers.Authorization, "Bearer secret");
  assert.ok(routes.every((route) => !/ghfast\.top|ghproxy\.net/.test(route.url)));

  const customChecksum = checksumRoutes(parsed, { environment, allowToken: false })[0];
  assert.equal(customChecksum.url, parsed.asset.checksumUrl);
  assert.equal(customChecksum.headers.Authorization, undefined);
  const publicRoutes = releaseDownloadRoutes(parsed, { checksum: sha, environment, allowToken: false });
  assert.deepEqual(publicRoutes.map((route) => route.id), ["release-ghfast", "release-ghproxy-net", "release-github"]);
  const customOfficial = publicRoutes.at(-1);
  assert.equal(customOfficial.url, parsed.asset.url);
  assert.equal(customOfficial.headers.Authorization, undefined);
});

test("an explicit mirror is isolated from private official credentials", () => {
  const parsed = parseRelease(release());
  const token = "private-token-that-must-not-leak";
  const routes = releaseDownloadRoutes(parsed, {
    checksum: sha,
    environment: {
      XIRAI_UPDATE_TOKEN: token,
      XIRAI_UPDATE_MIRROR: "https://mirror.example/gh/",
    },
  });
  assert.deepEqual(routes.map((route) => route.id), ["release-mirror", "release-github"]);
  assert.equal(routes[0].url, `https://mirror.example/gh/${parsed.asset.url}`);
  assert.equal(routes[0].url.includes(token), false);
  assert.equal(new Headers(routes[0].headers).has("authorization"), false);
  assert.equal(routes[1].url, parsed.asset.apiUrl);
  assert.equal(routes[1].headers.Authorization, `Bearer ${token}`);
});

test("explicit private context suppresses public proxies without relying on token presence", () => {
  const parsed = parseRelease(release());
  const routes = releaseDownloadRoutes(parsed, {
    checksum: sha,
    environment: {},
    privateContext: true,
  });
  assert.deepEqual(routes.map((route) => route.id), ["release-github"]);
  assert.equal(routes[0].url, parsed.asset.url);
  assert.equal(routes[0].headers.Authorization, undefined);
});

test("an API-shaped official URL is conservative even without a token or explicit private flag", () => {
  const parsed = parseRelease(release());
  const apiOnly = {
    ...parsed,
    asset: {
      ...parsed.asset,
      url: parsed.asset.apiUrl,
      apiUrl: null,
    },
  };
  const routes = releaseDownloadRoutes(apiOnly, { checksum: sha, environment: {} });
  assert.deepEqual(routes.map((route) => route.id), ["release-github"]);
  assert.equal(routes[0].url, parsed.asset.apiUrl);
});

test("mirror configuration is credential-free HTTPS and signed source URLs are never disclosed", () => {
  const parsed = parseRelease(release());
  for (const mirror of [
    "http://mirror.example/gh",
    "https://user:password@mirror.example/gh",
    "https://mirror.example/gh?token=secret",
    "https://mirror.example/gh#fragment",
    " https://mirror.example/gh",
    "not a URL",
  ]) {
    assert.throws(
      () => releaseDownloadRoutes(parsed, { checksum: sha, environment: { XIRAI_UPDATE_MIRROR: mirror } }),
      /XIRAI_UPDATE_MIRROR/,
      mirror,
    );
  }

  const signed = {
    ...parsed,
    asset: { ...parsed.asset, url: `${parsed.asset.url}?token=release-secret` },
  };
  const routes = releaseDownloadRoutes(signed, {
    checksum: sha,
    environment: { XIRAI_UPDATE_MIRROR: "https://mirror.example/gh" },
  });
  assert.deepEqual(routes.map((route) => route.id), ["release-github"]);
  assert.ok(routes.every((route) => !route.url.startsWith("https://mirror.example/")));
  assert.deepEqual(
    releaseDownloadRoutes(signed, { checksum: sha, environment: {} }).map((route) => route.id),
    ["release-github"],
  );
});

test("404 meaning remains explicit for custom feeds and reachable repositories", () => {
  assert.equal(missingReleaseMeaning({ customFeed: false, repositoryReachable: true }), "no-release");
  assert.equal(missingReleaseMeaning({ customFeed: false, repositoryReachable: false }), "source-missing");
  assert.equal(missingReleaseMeaning({ customFeed: true, repositoryReachable: true }), "source-missing");
  assert.equal(missingReleaseMeaning(), "source-missing");
});
