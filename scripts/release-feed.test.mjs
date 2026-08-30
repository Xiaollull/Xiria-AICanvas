import assert from "node:assert/strict";
import test from "node:test";
import {
  checksumRoutes,
  compareVersions,
  parseChecksumFile,
  parseRelease,
  parseVersion,
  releaseDownloadRoutes,
  releaseFeedUrl,
  releaseRepository,
  selectReleaseAsset,
  updateAvailable,
} from "./release-feed.mjs";

function release(overrides = {}) {
  return {
    tag_name: "v1.1.0",
    published_at: "2026-09-01T10:00:00Z",
    body: "  修复若干问题  ",
    assets: [
      { name: "XirAI-1.1.0.7z", browser_download_url: "https://github.com/o/r/releases/download/v1.1.0/XirAI-1.1.0.7z", size: 734003200, digest: "sha256:" + "a".repeat(64) },
    ],
    ...overrides,
  };
}

test("versions parse with and without a leading v, and reject anything else", () => {
  assert.equal(parseVersion("1.0.0").text, "1.0.0");
  assert.equal(parseVersion("v1.2.3").text, "1.2.3");
  assert.deepEqual(parseVersion("1.2.3-beta.1").prerelease, ["beta", "1"]);
  for (const rejected of ["", null, undefined, "latest", "1.0", "1.0.0.0", "nightly-2026-09-01"]) {
    assert.equal(parseVersion(rejected), null, String(rejected));
  }
});

test("version ordering follows semver, including pre-release ranking", () => {
  assert.ok(compareVersions("1.0.1", "1.0.0") > 0);
  assert.ok(compareVersions("1.1.0", "1.0.9") > 0);
  assert.ok(compareVersions("2.0.0", "1.99.99") > 0);
  assert.equal(compareVersions("1.0.0", "v1.0.0"), 0);
  // A pre-release sorts below the release it leads to, and numerically among its own kind.
  assert.ok(compareVersions("1.0.0-beta.1", "1.0.0") < 0);
  assert.ok(compareVersions("1.0.0-beta.2", "1.0.0-beta.10") < 0);
  assert.ok(compareVersions("1.0.0-alpha", "1.0.0-beta") < 0);
});

test("an update is offered only for a strictly newer, parsable version", () => {
  assert.equal(updateAvailable("1.0.1", "1.0.0"), true);
  assert.equal(updateAvailable("1.0.0", "1.0.0"), false);
  // A local build ahead of the channel is left alone rather than rolled backwards.
  assert.equal(updateAvailable("1.0.0", "1.1.0"), false);
  // Neither side may be a tag that is not a version.
  assert.equal(updateAvailable("nightly", "1.0.0"), false);
  assert.equal(updateAvailable("1.0.1", "dev"), false);
});

test("the release feed defaults to the project repository and honours an override", () => {
  assert.equal(releaseRepository({}), "Xiaollull/Xiria-AICanvas");
  assert.equal(releaseRepository({ XIRAI_UPDATE_REPO: "someone/fork" }), "someone/fork");
  // A malformed override falls back rather than building a nonsense URL.
  assert.equal(releaseRepository({ XIRAI_UPDATE_REPO: "not a repo" }), "Xiaollull/Xiria-AICanvas");
  assert.equal(releaseFeedUrl({}), "https://api.github.com/repos/Xiaollull/Xiria-AICanvas/releases/latest");
  assert.equal(releaseFeedUrl({ XIRAI_UPDATE_FEED: "https://mirror.example/latest.json" }), "https://mirror.example/latest.json");
});

test("the smallest archive asset wins and its digest is read from the API", () => {
  const asset = selectReleaseAsset([
    { name: "notes.txt", browser_download_url: "https://x/notes.txt", size: 10 },
    { name: "XirAI-full-1.1.0.7z", browser_download_url: "https://x/full.7z", size: 9_000_000_000 },
    { name: "XirAI-1.1.0.7z", browser_download_url: "https://x/program.7z", size: 700_000_000, digest: `sha256:${"b".repeat(64)}` },
  ]);
  assert.equal(asset.name, "XirAI-1.1.0.7z");
  assert.equal(asset.bytes, 700_000_000);
  assert.equal(asset.sha256, "b".repeat(64));
  assert.equal(selectReleaseAsset([{ name: "readme.md", browser_download_url: "https://x/r.md" }]), null);
});

test("a checksum sidecar asset is found beside the archive it covers", () => {
  const asset = selectReleaseAsset([
    { name: "XirAI-1.1.0.7z", browser_download_url: "https://x/a.7z", size: 5 },
    { name: "XirAI-1.1.0.7z.sha256", browser_download_url: "https://x/a.7z.sha256", size: 65 },
  ]);
  assert.equal(asset.checksumUrl, "https://x/a.7z.sha256");
  assert.deepEqual(checksumRoutes({ asset }), [
    { id: "release-checksum", label: "校验和 · GitHub", url: "https://x/a.7z.sha256" },
  ]);
  assert.deepEqual(checksumRoutes({ asset: { ...asset, checksumUrl: null } }), []);
});

test("a checksum file is read in both bare-hash and SHA256SUMS form", () => {
  const hash = "c".repeat(64);
  assert.equal(parseChecksumFile(`${hash}\n`, "XirAI-1.1.0.7z"), hash);
  assert.equal(parseChecksumFile(`${hash}  XirAI-1.1.0.7z\n`, "XirAI-1.1.0.7z"), hash);
  assert.equal(parseChecksumFile([
    `${"d".repeat(64)}  OtherPackage.7z`,
    `${hash}  XirAI-1.1.0.7z`,
  ].join("\n"), "XirAI-1.1.0.7z"), hash);
  assert.equal(parseChecksumFile("no hash here", "XirAI-1.1.0.7z"), null);
  assert.equal(parseChecksumFile("", "XirAI-1.1.0.7z"), null);
});

test("a release is reduced to the fields the update flow acts on", () => {
  const parsed = parseRelease(release());
  assert.equal(parsed.version, "1.1.0");
  assert.equal(parsed.prerelease, false);
  assert.equal(parsed.publishedAt, "2026-09-01T10:00:00Z");
  assert.equal(parsed.notes, "修复若干问题");
  assert.equal(parsed.asset.name, "XirAI-1.1.0.7z");
  assert.equal(parsed.asset.sha256, "a".repeat(64));
});

test("a release with nothing installable in it is not an update", () => {
  assert.equal(parseRelease(null), null);
  assert.equal(parseRelease({}), null);
  // A draft is not published, whatever it carries.
  assert.equal(parseRelease(release({ draft: true })), null);
  // A tag that is not a version cannot be compared against the running build.
  assert.equal(parseRelease(release({ tag_name: "nightly" })), null);
  // A release with notes but no archive offers nothing to install.
  assert.equal(parseRelease(release({ assets: [{ name: "notes.txt", browser_download_url: "https://x/n.txt" }] })), null);
  // An asset entry without a download URL is equally useless.
  assert.equal(parseRelease(release({ assets: [{ name: "XirAI.7z", browser_download_url: "" }] })), null);
});

test("a pre-release tag is reported as such so it can be kept off a stable channel", () => {
  assert.equal(parseRelease(release({ tag_name: "v1.2.0-beta.1" })).prerelease, true);
  assert.equal(parseRelease(release({ prerelease: true })).prerelease, true);
});

test("accelerated routes are offered only when the archive's checksum is known", () => {
  const parsed = parseRelease(release());
  const verified = releaseDownloadRoutes(parsed, { checksum: parsed.asset.sha256, environment: {} });
  assert.deepEqual(verified.map((route) => route.id), ["release-ghfast", "release-ghproxy-net", "release-github"]);
  assert.ok(verified[0].url.endsWith(parsed.asset.url));

  // Without a checksum a mirror could serve anything at all, and the bytes become program files.
  const unverified = releaseDownloadRoutes(parsed, { checksum: null, environment: {} });
  assert.deepEqual(unverified.map((route) => route.id), ["release-github"]);
  assert.equal(unverified[0].url, parsed.asset.url);
});

test("a configured mirror replaces the built-in accelerated hosts", () => {
  const parsed = parseRelease(release());
  const routes = releaseDownloadRoutes(parsed, {
    checksum: parsed.asset.sha256,
    environment: { XIRAI_UPDATE_MIRROR: "https://mirror.example/gh/" },
  });
  assert.deepEqual(routes.map((route) => route.id), ["release-mirror", "release-github"]);
  assert.equal(routes[0].url, `https://mirror.example/gh/${parsed.asset.url}`);
  assert.deepEqual(releaseDownloadRoutes(null, { checksum: "x" }), []);
});
