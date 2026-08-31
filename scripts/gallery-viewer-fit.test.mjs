import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { access, mkdtemp, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import test from "node:test";
import { promisify } from "node:util";
import { fileURLToPath, pathToFileURL } from "node:url";

const execFileAsync = promisify(execFile);
const projectRoot = resolve(fileURLToPath(new URL("..", import.meta.url)));
const chromeCandidates = [
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
];

async function chromePath() {
  for (const candidate of chromeCandidates) {
    try {
      await access(candidate);
      return candidate;
    } catch {}
  }
  return null;
}

function svg(width, height) {
  return `data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='${width}' height='${height}' viewBox='0 0 ${width} ${height}'%3E%3Crect width='100%25' height='100%25' fill='%238a73ff'/%3E%3C/svg%3E`;
}

function fixture(cssHref) {
  return `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="stylesheet" href="${cssHref}"><style>:root{--lime:#8a73ff;--accent-rgb:138 115 255}.gallery-focus-shell.fixture-shell{position:static;width:100%;height:auto;display:block;margin:0;overflow:visible;padding:0;border:0;background:transparent;box-shadow:none}.fixtures{display:grid;grid-template-columns:1fr 1fr;gap:20px;padding:20px}.fixture-stage{width:calc(50vw - 30px);height:700px}.fixture-stage .gallery-viewer-image-viewport{width:100%;height:100%;box-sizing:border-box}</style></head><body><div class="gallery-focus gallery-focus-shell fixture-shell"><div class="fixtures"><div class="fixture-stage"><div id="wide" class="gallery-viewer-image-viewport"><span class="gallery-focus-frame ready"><img class="gallery-focus-full" src="${svg(1200, 200)}"></span></div></div><div class="fixture-stage"><div id="portrait" class="gallery-viewer-image-viewport"><span class="gallery-focus-frame ready"><img class="gallery-focus-full" src="${svg(200, 1200)}"></span></div></div></div></div><script>const box=node=>{const r=node.getBoundingClientRect();return[r.left,r.top,r.right,r.bottom,r.width,r.height]};const measure=id=>{const stage=document.getElementById(id);const frame=stage.querySelector('.gallery-focus-frame');const image=stage.querySelector('img');const stageStyle=getComputedStyle(stage);const frameStyle=getComputedStyle(frame);const imageStyle=getComputedStyle(image);return{stage:box(stage),frame:box(frame),image:box(image),stageStyle:[stageStyle.backgroundColor,stageStyle.backgroundImage],frameStyle:[frameStyle.width,frameStyle.height,frameStyle.maxWidth,frameStyle.maxHeight],imageStyle:[imageStyle.width,imageStyle.height,imageStyle.maxWidth,imageStyle.maxHeight,imageStyle.objectFit]}};document.documentElement.dataset.geometry=JSON.stringify({wide:measure('wide'),portrait:measure('portrait')});</script></body></html>`;
}

function assertContained(value, label) {
  const { stage, image } = value;
  const geometry = JSON.stringify(value);
  assert.ok(image[0] >= stage[0] - 0.5, `${label} escapes the left edge; ${geometry}`);
  assert.ok(image[1] >= stage[1] - 0.5, `${label} escapes the top edge; ${geometry}`);
  assert.ok(image[2] <= stage[2] + 0.5, `${label} escapes the right edge; ${geometry}`);
  assert.ok(image[3] <= stage[3] + 0.5, `${label} escapes the bottom edge; ${geometry}`);
  assert.ok(image[4] < stage[4] && image[5] < stage[5], `${label} should be scaled down with visible margin; ${geometry}`);
  assert.deepEqual(value.stageStyle, ["rgba(0, 0, 0, 0)", "none"], `${label} stage must be fully transparent; ${geometry}`);
}

test("local Chromium fully contains extreme panoramic and portrait originals", async (context) => {
  const chrome = await chromePath();
  if (!chrome) {
    context.skip("Chromium is unavailable");
    return;
  }
  const folder = await mkdtemp(join(tmpdir(), "xirai-gallery-fit-"));
  try {
    const htmlPath = join(folder, "fixture.html");
    const screenshot = join(folder, "gallery-fit.png");
    await writeFile(htmlPath, fixture(pathToFileURL(join(projectRoot, "src", "styles.css")).href), "utf8");
    const { stdout } = await execFileAsync(chrome, [
      "--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=1800,820",
      `--screenshot=${screenshot}`, "--dump-dom", pathToFileURL(htmlPath).href,
    ], { windowsHide: true, timeout: 30000, maxBuffer: 2 * 1024 * 1024 });
    const match = stdout.match(/data-geometry="([^"]+)"/);
    assert.ok(match, "gallery fit geometry was not emitted");
    const geometry = JSON.parse(match[1].replaceAll("&quot;", '"'));
    assertContained(geometry.wide, "panoramic image");
    assertContained(geometry.portrait, "portrait image");
    assert.ok(Math.abs(geometry.wide.image[4] / geometry.wide.image[5] - 6) < 0.02, "panoramic ratio changed");
    assert.ok(Math.abs(geometry.portrait.image[4] / geometry.portrait.image[5] - 1 / 6) < 0.02, "portrait ratio changed");
    assert.ok((await stat(screenshot)).size > 1024, "gallery fit screenshot was not produced");
  } finally {
    await rm(folder, { recursive: true, force: true });
  }
});
