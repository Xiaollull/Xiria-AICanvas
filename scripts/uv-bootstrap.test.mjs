import assert from "node:assert/strict";
import test from "node:test";
import { resolveUvTarget, uvBootstrapInternals, uvDownloadBases, uvDownloadRoutes, uvVersionMatches } from "./uv-bootstrap.mjs";

test("uv bootstrap selects official Windows archives", () => {
  const x64 = resolveUvTarget({ platform: "win32", architecture: "x64" });
  assert.equal(x64.assetName, "uv-x86_64-pc-windows-msvc.zip");
  assert.deepEqual(x64.binaries, ["uv.exe", "uvx.exe", "uvw.exe"]);
  assert.equal(resolveUvTarget({ platform: "win32", architecture: "arm64" }).assetName, "uv-aarch64-pc-windows-msvc.zip");
});

test("uv bootstrap respects glibc minimums and musl fallback", () => {
  assert.equal(resolveUvTarget({ platform: "linux", architecture: "x64", glibcVersion: "2.17" }).target, "x86_64-unknown-linux-gnu");
  assert.equal(resolveUvTarget({ platform: "linux", architecture: "x64", glibcVersion: "2.16" }).target, "x86_64-unknown-linux-musl");
  assert.equal(resolveUvTarget({ platform: "linux", architecture: "arm64", glibcVersion: "2.27" }).target, "aarch64-unknown-linux-musl");
  assert.equal(resolveUvTarget({ platform: "linux", architecture: "arm64", glibcVersion: "2.28" }).target, "aarch64-unknown-linux-gnu");
  assert.deepEqual(resolveUvTarget({ platform: "linux", architecture: "x64", glibcVersion: "2.17" }).binaries, ["uv", "uvx"]);
});

test("uv download bases follow official installer precedence", () => {
  assert.deepEqual(uvDownloadBases({ UV_DOWNLOAD_URL: "https://one.test/base https://two.test/base/", INSTALLER_DOWNLOAD_URL: "https://ignored.test" }), [
    "https://one.test/base",
    "https://two.test/base",
  ]);
  assert.deepEqual(uvDownloadBases({ INSTALLER_DOWNLOAD_URL: "https://generic.test" }), ["https://generic.test"]);
  assert.deepEqual(uvDownloadBases({ UV_INSTALLER_GITHUB_BASE_URL: "https://github-proxy.test" }), [
    `https://github-proxy.test/astral-sh/uv/releases/download/${uvBootstrapInternals.uvVersion}`,
  ]);
});

test("uv routes append the selected archive and versions must match exactly", () => {
  const target = resolveUvTarget({ platform: "win32", architecture: "x64" });
  const routes = uvDownloadRoutes(target, { UV_DOWNLOAD_URL: "https://mirror.test/releases" });
  assert.equal(routes[0].url, `https://mirror.test/releases/${target.assetName}`);
  assert.equal(routes[0].label, "uv 自定义线路");
  assert.equal(uvVersionMatches(`uv ${target.version} (fixture)`), true);
  assert.equal(uvVersionMatches("uv 0.1.0"), false);
});

test("uv default routes include verified accelerators and official fallbacks", () => {
  const target = resolveUvTarget({ platform: "win32", architecture: "x64" });
  const routes = uvDownloadRoutes(target, {});
  assert.deepEqual(routes.map((route) => route.id), ["uv-ghfast", "uv-ghproxy-net", "uv-astral", "uv-github"]);
  assert.ok(routes.every((route) => route.url.endsWith(target.assetName)));
  assert.ok(routes.some((route) => route.url.startsWith("https://releases.astral.sh/")));
  assert.ok(routes.some((route) => route.url.startsWith("https://github.com/")));
});

test("uv bootstrap rejects unsupported application platforms", () => {
  assert.throws(() => resolveUvTarget({ platform: "darwin", architecture: "x64" }), /Unsupported/);
  assert.throws(() => resolveUvTarget({ platform: "linux", architecture: "ia32" }), /Unsupported/);
});
