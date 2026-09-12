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
const modules = ["viewer-editor.js", "viewer-geometry.js"];

async function chromePath() {
  for (const candidate of chromeCandidates) {
    try { await access(candidate); return candidate; } catch {}
  }
  return null;
}

// The page is assembled from the shipped stylesheet and the shipped modules, so what it measures is
// what the viewer itself would draw: the ring a real browser paints at a real zoom, the cursor the
// stylesheet really resolves to, and a text box laid out by the real line-breaking rules.
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
import { measureTextLayer, normalizeTextLayer, textCanvasFont, VIEWER_TEXT_PADDING } from "/src/viewer-editor.js";
import { viewerBrushCursorSize, viewerMarqueeRect } from "/src/viewer-geometry.js";
try {
  const canvas = document.getElementById("canvas");
  const scene = document.getElementById("scene");
  const ring = document.getElementById("ring");
  const zoom = 2;
  scene.style.transform = "translate(0px, 0px) scale(" + zoom + ")";

  // 1. Every tool's cursor, as the stylesheet actually resolves it on the canvas.
  const cursors = {};
  for (const tool of ["move", "brush", "eraser", "text"]) {
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

  // 3. A text box laid out by the browser from the lines the model wrapped for it.
  const measuringContext = document.createElement("canvas").getContext("2d");
  const base = normalizeTextLayer({ text: "The quick brown fox jumps over the lazy dog", fontSize: 24, boxWidth: 200, boxHeight: 120, x: 0, y: 0 });
  measuringContext.font = textCanvasFont(base);
  const measured = measureTextLayer(base.text, base, (value) => measuringContext.measureText(value).width);
  const layer = document.createElement("div");
  layer.className = "viewer-image-layer viewer-text-layer active";
  layer.style.width = measured.naturalWidth + "px";
  layer.style.height = measured.naturalHeight + "px";
  layer.style.transform = "translate(0px, 0px) rotate(0deg) scale(1)";
  const content = document.createElement("div");
  content.className = "viewer-text-content";
  content.style.font = textCanvasFont(base);
  content.style.lineHeight = base.lineHeight;
  content.textContent = measured.lines.join("\\n");
  layer.appendChild(content);
  scene.appendChild(layer);

  const layerRect = layer.getBoundingClientRect();
  // Each rendered line is measured on its own, so a line wider than the box it was wrapped for
  // would show up here. The space a line ends on is dropped first: it hangs past a wrap on screen
  // and is measured away by the wrapper, so counting it would compare two different things.
  const lineWidths = measured.lines.map((line) => {
    const probe = document.createElement("span");
    probe.style.font = content.style.font;
    probe.style.whiteSpace = "pre";
    probe.style.position = "absolute";
    probe.textContent = line.replace(/\\s+$/, "");
    document.body.appendChild(probe);
    const width = probe.getBoundingClientRect().width;
    probe.remove();
    return width;
  });
  const textBox = {
    lines: measured.lines,
    naturalWidth: measured.naturalWidth,
    naturalHeight: measured.naturalHeight,
    renderedWidth: layerRect.width,
    renderedHeight: layerRect.height,
    contentLimit: measured.naturalWidth - VIEWER_TEXT_PADDING * 2,
    widest: Math.max(...lineWidths),
    overflow: getComputedStyle(content).overflow,
    whiteSpace: getComputedStyle(content).whiteSpace,
    clipped: content.scrollHeight > content.clientHeight,
  };

  // 4. The swept rectangle, placed in the scene the way a layer is placed.
  const rect = viewerMarqueeRect({ x: -60, y: -20 }, { x: 40, y: 30 });
  const marquee = document.createElement("i");
  marquee.className = "viewer-text-marquee";
  marquee.style.width = rect.width + "px";
  marquee.style.height = rect.height + "px";
  marquee.style.transform = "translate(" + rect.centerX + "px, " + rect.centerY + "px)";
  scene.appendChild(marquee);
  const marqueeRect = marquee.getBoundingClientRect();
  const marqueeGeometry = {
    width: marqueeRect.width,
    height: marqueeRect.height,
    // Scene origin is the middle of the canvas, so a scene point lands at centre + point x zoom.
    left: marqueeRect.left - canvasRect.left - canvasRect.width / 2,
    top: marqueeRect.top - canvasRect.top - canvasRect.height / 2,
    pointerEvents: getComputedStyle(marquee).pointerEvents,
  };

  document.getElementById("result").textContent = JSON.stringify({ cursors, ringGeometry, textBox, marqueeGeometry, errors: window.fixtureErrors });
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

test("real Chromium paints the brush ring, the tool cursors and a text box the way the model describes them", { skip: skipBrowserFixture && "browser fixtures disabled" }, async (context) => {
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

      // The brush and the eraser hand the pointer over to the ring; the text tool positions from a
      // crosshair; moving still grabs.
      assert.deepEqual(result.cursors, { move: "grab", brush: "none", eraser: "none", text: "crosshair" });

      // A 40px brush on a half-scale layer at 2x zoom covers 40 screen pixels, and the ring is
      // exactly that wide, centred on the pointer rather than hanging off one corner of it.
      assert.equal(result.ringGeometry.diameter, 40);
      assert.equal(result.ringGeometry.width, 40);
      assert.equal(result.ringGeometry.height, 40);
      assert.ok(Math.abs(result.ringGeometry.centerX - 220) < 0.5, `ring centre x ${result.ringGeometry.centerX}`);
      assert.ok(Math.abs(result.ringGeometry.centerY - 160) < 0.5, `ring centre y ${result.ringGeometry.centerY}`);
      assert.equal(result.ringGeometry.pointerEvents, "none", "the ring must never swallow a stroke");
      assert.equal(result.ringGeometry.hidden, "none", "the ring must leave with the pointer");

      // The box keeps the size it was given, and the browser lays out the lines the model wrapped
      // without breaking any of them again: every line fits inside the padded width.
      assert.deepEqual([result.textBox.naturalWidth, result.textBox.naturalHeight], [200, 120]);
      // Zoom scales the box on screen and nothing else: 200 x 120 layer pixels at 2x.
      assert.deepEqual([result.textBox.renderedWidth, result.textBox.renderedHeight], [400, 240]);
      assert.ok(result.textBox.lines.length > 1, "a 200px box must wrap this sentence");
      assert.ok(result.textBox.widest <= result.textBox.contentLimit + 0.5, `line of ${result.textBox.widest}px escaped a ${result.textBox.contentLimit}px box`);
      assert.equal(result.textBox.whiteSpace, "pre", "the committed box must not re-wrap what was wrapped for it");
      assert.equal(result.textBox.overflow, "hidden", "a box shows only what fits in it");

      // The swept rectangle lands where the drag put it: scene coordinates, scaled by the zoom.
      assert.deepEqual([result.marqueeGeometry.width, result.marqueeGeometry.height], [200, 100]);
      assert.ok(Math.abs(result.marqueeGeometry.left - -120) < 0.5, `marquee left ${result.marqueeGeometry.left}`);
      assert.ok(Math.abs(result.marqueeGeometry.top - -40) < 0.5, `marquee top ${result.marqueeGeometry.top}`);
      assert.equal(result.marqueeGeometry.pointerEvents, "none");
    });
  } finally {
    await rm(profile, { recursive: true, force: true });
  }
});
