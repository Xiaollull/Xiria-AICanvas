import test from "node:test";
import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { access, mkdtemp, readFile, rm } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";

const execFileAsync = promisify(execFile);
const root = resolve(fileURLToPath(new URL("..", import.meta.url)));
const skipBrowserFixture = process.env.XIRAI_SKIP_BROWSER_FIXTURES === "1";
const chromeCandidates = [
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
];
// viewer-geometry.js imports viewerEditorLayerBounds, so the fixture has to serve that module too
// or the import chain 404s and the page never runs.
const modules = ["viewer-geometry.js", "viewer-editor.js"];

async function chromePath() {
  for (const candidate of chromeCandidates) {
    try { await access(candidate); return candidate; } catch {}
  }
  return null;
}

// The page is assembled from the shipped stylesheet and the shipped modules, so what it measures is
// what the viewer itself would draw: the ring a real browser paints at a real zoom, and the cursor
// the stylesheet really resolves to.
const fixture = `<!doctype html><html><head><meta charset="utf-8"><link rel="stylesheet" href="/src/styles.css"><style>
:root{--n-01:#090a0c;--n-02:#111218;--n-03:#181923;--n-05:#252636;--n-06:#303146;--n-07:#3b3d55;--n-12:#75788f;--n-13:#7d8097;--n-21:#d2d5e4;--n-27:#ececf6;--lime:#c8acfb;--accent-rgb:200 172 251;box-sizing:border-box}*{box-sizing:border-box}html,body{margin:0;width:100%;height:100%;overflow:hidden}
#canvas{position:absolute;left:0;top:0;width:640px;height:420px}
</style></head><body>
<div class="image-viewer-canvas tool-brush" id="canvas">
  <div class="viewer-scene" id="scene"></div>
  <i class="viewer-brush-cursor brush" id="ring"></i>
</div>
<pre id="result">pending</pre>
<script>window.fixtureErrors=[];addEventListener("error",event=>fixtureErrors.push(event.message));addEventListener("unhandledrejection",event=>fixtureErrors.push(String(event.reason)));setTimeout(()=>{const node=document.getElementById("result");if(node.textContent==="pending")node.textContent=JSON.stringify({fatal:"fixture timeout",errors:fixtureErrors})},2500)</script>
<script type="module">
import { viewerBrushCursorSize } from "/src/viewer-geometry.js";
try {
  const canvas = document.getElementById("canvas");
  const scene = document.getElementById("scene");
  const ring = document.getElementById("ring");
  const zoom = 2;
  scene.style.transform = "translate(0px, 0px) scale(" + zoom + ")";

  // 1. Every tool's cursor, as the stylesheet actually resolves it on the canvas.
  const cursors = {};
  for (const tool of ["move", "brush", "eraser"]) {
    canvas.className = "image-viewer-canvas tool-" + tool;
    cursors[tool] = getComputedStyle(canvas).cursor;
  }

  // 2. The ring, sized and placed exactly the way the app places it, measured by the browser.
  canvas.className = "image-viewer-canvas tool-brush";
  const canvasRect = canvas.getBoundingClientRect();
  const at = { x: 220, y: 160 };
  const layerScale = 0.5;
  const diameter = viewerBrushCursorSize(40, zoom, layerScale);
  ring.style.width = diameter + "px";
  ring.style.height = diameter + "px";
  ring.style.left = at.x + "px";
  ring.style.top = at.y + "px";
  const ringRect = ring.getBoundingClientRect();
  const ringHiddenStyle = (() => { ring.classList.add("away"); const display = getComputedStyle(ring).display; ring.classList.remove("away"); return display; })();
  const ringGeometry = {
    diameter,
    width: ringRect.width,
    height: ringRect.height,
    centerX: ringRect.left + ringRect.width / 2 - canvasRect.left,
    centerY: ringRect.top + ringRect.height / 2 - canvasRect.top,
    pointerEvents: getComputedStyle(ring).pointerEvents,
    hidden: ringHiddenStyle,
  };

  // 3. A picture's grips, as the real cascade resolves them.
  const picture = document.createElement("div");
  picture.className = "viewer-image-layer active";
  picture.style.width = "120px";
  picture.style.height = "80px";
  picture.style.transform = "translate(0px, 0px) rotate(0deg) scale(1)";
  scene.appendChild(picture);
  const grip = (host) => {
    const anchor = document.createElement("i");
    anchor.className = "layer-corner-anchor br";
    const corner = document.createElement("i");
    corner.className = "layer-corner br";
    anchor.appendChild(corner);
    host.appendChild(anchor);
    const style = getComputedStyle(corner);
    return { width: style.width, height: style.height, radius: style.borderRadius };
  };
  const stacking = { activePicture: getComputedStyle(picture).zIndex, pictureGrip: grip(picture) };

  document.getElementById("result").textContent = JSON.stringify({ cursors, ringGeometry, stacking, errors: window.fixtureErrors });
} catch (error) {
  document.getElementById("result").textContent = JSON.stringify({ fatal: String(error), errors: window.fixtureErrors });
}
</script></body></html>`;

async function withServer(run) {
  const sources = new Map(await Promise.all(modules.map(async (name) => [name, await readFile(join(root, "src", name), "utf8")])));
  const server = createServer(async (request, response) => {
    try {
      const name = request.url.startsWith("/src/") ? request.url.slice(5) : "";
      if (sources.has(name)) {
        response.setHeader("content-type", "text/javascript; charset=utf-8");
        response.end(sources.get(name));
      } else if (request.url === "/src/styles.css") {
        response.setHeader("content-type", "text/css; charset=utf-8");
        response.end(await readFile(join(root, "src", "styles.css")));
      } else {
        response.setHeader("content-type", "text/html; charset=utf-8");
        response.end(fixture);
      }
    } catch (error) {
      response.statusCode = 500;
      response.end(String(error));
    }
  });
  await new Promise((resolvePromise) => server.listen(0, "127.0.0.1", resolvePromise));
  try { return await run(server.address().port); } finally { await new Promise((resolvePromise) => server.close(resolvePromise)); }
}

test("real Chromium paints the brush ring and the tool cursors the way the model describes them", { skip: skipBrowserFixture && "browser fixtures disabled" }, async (context) => {
  const chrome = await chromePath();
  if (!chrome) { context.skip("Chromium is unavailable"); return; }
  const profile = await mkdtemp(join(tmpdir(), "xirai-paint-tools-"));
  try {
    await withServer(async (port) => {
      const { stdout } = await execFileAsync(chrome, ["--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1", `--user-data-dir=${profile}`, "--window-size=1280,800", "--virtual-time-budget=3000", "--dump-dom", `http://127.0.0.1:${port}/`], { windowsHide: true, timeout: 30000, maxBuffer: 4 * 1024 * 1024 });
      const match = stdout.match(/<pre id="result">([^<]+)<\/pre>/);
      assert.ok(match && match[1] !== "pending", `browser fixture did not finish:\n${stdout.slice(-3000)}`);
      const result = JSON.parse(match[1].replaceAll("&quot;", '"').replaceAll("&amp;", "&"));
      assert.equal(result.fatal, undefined, `fixture failed: ${JSON.stringify(result)}`);
      assert.deepEqual(result.errors, [], "browser console errors");

      // The brush and the eraser hand the pointer over to the ring; moving still grabs.
      assert.deepEqual(result.cursors, { move: "grab", brush: "none", eraser: "none" });

      // A 40px brush on a half-scale layer at 2x zoom covers 40 screen pixels, and the ring is
      // exactly that wide, centred on the pointer rather than hanging off one corner of it.
      assert.equal(result.ringGeometry.diameter, 40);
      assert.equal(result.ringGeometry.width, 40);
      assert.equal(result.ringGeometry.height, 40);
      assert.ok(Math.abs(result.ringGeometry.centerX - 220) < 0.5, `ring centre x ${result.ringGeometry.centerX}`);
      assert.ok(Math.abs(result.ringGeometry.centerY - 160) < 0.5, `ring centre y ${result.ringGeometry.centerY}`);
      assert.equal(result.ringGeometry.pointerEvents, "none", "the ring must never swallow a stroke");
      assert.equal(result.ringGeometry.hidden, "none", "the ring must leave with the pointer");

      // A selected picture keeps its round 20px grips, unchanged by the text tool's removal.
      assert.deepEqual(result.stacking.pictureGrip, { width: "20px", height: "20px", radius: "50%" });
      assert.equal(result.stacking.activePicture, "8");
    });
  } finally {
    await rm(profile, { recursive: true, force: true });
  }
});
