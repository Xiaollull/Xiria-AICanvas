import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { access, mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import test from "node:test";
import { promisify } from "node:util";
import { fileURLToPath, pathToFileURL } from "node:url";

import { HOVER_PREVIEW_MARGIN, HOVER_PREVIEW_WIDTH, hoverPreviewPlacement } from "../src/lora-hover-preview.js";

// The mounted row grew a column and the hover panel is drawn into <body> at a
// computed position. Both are the kind of thing a source assertion cannot check:
// what matters is that the row still lines up and that the panel lands on screen
// and does not sit on top of the row it describes.

const execFileAsync = promisify(execFile);
const projectRoot = resolve(fileURLToPath(new URL("..", import.meta.url)));
const chromeCandidates = [
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
];
const skipBrowserFixture = process.env.XIRAI_SKIP_BROWSER_FIXTURES === "1";

async function chromePath() {
  for (const candidate of chromeCandidates) {
    try {
      await access(candidate);
      return candidate;
    } catch {}
  }
  return null;
}

const PIXEL = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='3' height='4'%3E%3Crect width='3' height='4' fill='%23888'/%3E%3C/svg%3E";

function row(index) {
  return `<div class="lora-mounted-item with-preview">
    <div class="mounted-item-head">
      <button class="mounted-drag-handle" type="button">::</button>
      <button class="mounted-toggle active" type="button"><i></i></button>
      <div class="mounted-preview"><img alt="" src="${PIXEL}"></div>
      <div class="mounted-item-info"><strong>和风光影 v${index} 一个相当长的自定义名称</strong><small>Style · 0.1</small></div>
      <div class="mounted-weight-field"><input class="mounted-weight-input" value="0.75"><span class="mounted-weight-steppers"><button type="button">+</button><button type="button">-</button></span></div>
      <button class="mounted-card-edit customized" type="button" data-edit="${index}">P</button>
      <button class="remove" type="button">x</button>
    </div>
    <div class="lora-weight-control"><span>-5</span><input class="lora-weight" type="range" min="-5" max="5" step="0.1" value="0.75"><span>5</span></div>
  </div>`;
}

function fixture(cssHref) {
  return `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="stylesheet" href="${cssHref}"><style>:root{--lime:#d6ff3f;--accent-rgb:214 255 63;--accent-contrast:#10120b;--line:#30312e;--muted:#7d7f78;--orange:#ff7548}body{margin:0;background:#080908}.rail{width:380px;padding:12px}</style></head><body>
<div class="rail"><div class="lora-mounted-list">${[1, 2, 3].map(row).join("")}</div></div>
<div class="lora-hover-preview" id="hover"><div class="lora-hover-cover"><img alt="" src="${PIXEL}"></div><div class="lora-hover-copy"><strong>和风光影 v2</strong><code>kazutake-hazano_v2_epoch28.safetensors</code><div class="lora-hover-tags"><span>风格</span><span>和风</span></div><p class="lora-hover-prompt"><svg width="12" height="12"></svg><span>kazutake, soft rim light, delicate shading</span></p><p class="lora-hover-note"><svg width="12" height="12"></svg><span>配 0.75 权重，和写实底模不搭</span></p></div></div>
<script>
const heads=[...document.querySelectorAll('.mounted-item-head')].map(head=>{
  const cells=[...head.children].map(node=>{const r=node.getBoundingClientRect();return [Math.round(r.left),Math.round(r.top),Math.round(r.right),Math.round(r.bottom)]});
  const r=head.getBoundingClientRect();
  return {cells,box:[Math.round(r.left),Math.round(r.top),Math.round(r.right),Math.round(r.bottom)]};
});
let clicked=0;document.addEventListener('click',e=>{if(e.target.closest('.mounted-card-edit'))clicked+=1});
for(const button of document.querySelectorAll('.mounted-card-edit'))button.click();
const panel=document.getElementById('hover').getBoundingClientRect();
const list=document.querySelector('.lora-mounted-list');
document.body.dataset.geometry=JSON.stringify({width:innerWidth,heads,clicked,panel:[Math.round(panel.width),Math.round(panel.height)],list:[list.clientWidth,list.scrollWidth]});
</script></body></html>`;
}

test("the mounted row keeps its columns aligned once the card button is added", {
  skip: skipBrowserFixture && "XIRAI_SKIP_BROWSER_FIXTURES keeps CI independent of runner browsers",
}, async (context) => {
  const styles = await readFile(join(projectRoot, "src", "styles.css"), "utf8");
  // Seven cells with a preview, six without: handle, switch, [preview], name,
  // weight, card, remove.
  assert.match(styles, /\.lora-mounted-item\.with-preview \.mounted-item-head \{ grid-template-columns: 32px 34px 44px minmax\(0, 1fr\) 74px 24px 22px; \}/);
  assert.match(styles, /\.mounted-card-edit \{/);
  // Seven fixed cells and their gaps is most of a phone-width row, so the narrow
  // breakpoint reclaims enough for the name to stay a name. The stylesheet holds
  // more than one 600px block, so this asks which one the rule actually sits in.
  const tightened = styles.indexOf(".lora-mounted-item.with-preview .mounted-item-head { grid-template-columns: 28px 34px 40px minmax(0, 1fr) 62px 24px 22px; }");
  assert.ok(tightened > 0, "the narrow row layout is missing");
  const before = styles.slice(0, tightened);
  assert.ok(before.lastIndexOf("@media (max-width: 600px)") > before.lastIndexOf("\n}"), "the narrow row layout escaped its media query");

  const chrome = await chromePath();
  if (!chrome) {
    context.skip("Chromium is unavailable; source CSS contract was checked");
    return;
  }
  const folder = await mkdtemp(join(tmpdir(), "xirai-lora-card-"));
  try {
    const htmlPath = join(folder, "fixture.html");
    await writeFile(htmlPath, fixture(pathToFileURL(join(projectRoot, "src", "styles.css")).href), "utf8");
    for (const width of [1400, 900, 620]) {
      const screenshot = join(folder, `card-${width}.png`);
      const { stdout } = await execFileAsync(chrome, [
        "--headless=new", "--disable-gpu", "--hide-scrollbars", `--window-size=${width},900`, `--screenshot=${screenshot}`, "--dump-dom", pathToFileURL(htmlPath).href,
      ], { windowsHide: true, timeout: 30000, maxBuffer: 2 * 1024 * 1024 });
      const match = stdout.match(/data-geometry="([^"]+)"/);
      assert.ok(match, `geometry result missing at ${width}`);
      const geometry = JSON.parse(match[1].replace(/&quot;/g, '"'));

      assert.equal(geometry.clicked, 3, `the card button stays clickable at ${width}`);
      assert.ok(geometry.list[1] <= geometry.list[0], `the mounted list overflows horizontally at ${width}`);
      const [first, ...rest] = geometry.heads;
      assert.equal(first.cells.length, 7, `the row draws seven cells at ${width}`);
      for (const [index, head] of rest.entries()) {
        // Every row is the same shape, so a name long enough to wrap or a wider
        // weight box must not shift the two buttons on the right out of column.
        for (const cell of [4, 5, 6]) {
          assert.equal(head.cells[cell][0] - head.box[0], first.cells[cell][0] - first.box[0], `row ${index + 2} column ${cell} drifted at ${width}`);
          assert.equal(head.cells[cell][2] - head.box[0], first.cells[cell][2] - first.box[0], `row ${index + 2} column ${cell} width drifted at ${width}`);
        }
      }
      for (const head of geometry.heads) {
        for (let index = 1; index < head.cells.length; index += 1) {
          assert.ok(head.cells[index][0] >= head.cells[index - 1][2], "row cells overlap");
        }
        assert.ok(head.cells.at(-1)[2] <= head.box[2] + 1, "the remove button is pushed outside the row");
      }

      // The panel is measured before it is placed, so the placement it is given
      // has to keep the real rendered box on screen.
      assert.equal(geometry.panel[0], HOVER_PREVIEW_WIDTH, `the hover panel is ${geometry.panel[0]}px wide at ${width}`);
      const anchor = { left: first.box[0], top: first.box[1], right: first.box[2], bottom: first.box[3] };
      const placement = hoverPreviewPlacement({
        anchor,
        viewport: { width: geometry.width, height: 900 },
        size: { width: geometry.panel[0], height: geometry.panel[1] },
      });
      assert.ok(placement.left >= HOVER_PREVIEW_MARGIN, `panel off the left edge at ${width}`);
      assert.ok(placement.left + geometry.panel[0] <= geometry.width - HOVER_PREVIEW_MARGIN, `panel off the right edge at ${width}`);
      assert.ok(placement.top >= HOVER_PREVIEW_MARGIN, `panel off the top edge at ${width}`);
      assert.ok(placement.top + geometry.panel[1] <= 900 - HOVER_PREVIEW_MARGIN, `panel off the bottom edge at ${width}`);
      // Whichever way it went, it must not sit on the row the pointer is resting
      // on: clear to one side, or clear above or below it.
      const clearHorizontally = placement.left >= anchor.right || placement.left + geometry.panel[0] <= anchor.left;
      const clearVertically = placement.top >= anchor.bottom || placement.top + geometry.panel[1] <= anchor.top;
      assert.ok(clearHorizontally || clearVertically, `the panel covers the row it describes at ${width} (${placement.side})`);
      assert.ok((await stat(screenshot)).size > 1024, `screenshot missing at ${width}`);
    }
  } finally {
    await rm(folder, { recursive: true, force: true });
  }
});
