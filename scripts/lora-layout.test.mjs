import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { access, mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
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

function fixture(cssHref) {
  return `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="stylesheet" href="${cssHref}"><style>:root{--lime:#d6ff3f;--accent-rgb:214 255 63;--line:#30312e}body{margin:0;background:#080908}.fixture-card img{background:#1d1e1c}</style></head><body><div class="lora-modal-backdrop"><section class="lora-modal"><header class="lora-modal-head"><div><span>MODEL ASSET LIBRARY</span><h2>LoRA 管理</h2></div><div class="lora-modal-actions"><button class="modal-close" type="button">X</button></div></header><div class="lora-modal-body"><nav class="lora-categories"><button class="lora-mounted-nav">已挂载</button><button>角色</button><button>风格</button><button>概念</button><button>其他</button></nav><main class="lora-browser"><header class="lora-browser-head"><div><h3>LoRA 分类</h3></div><label class="lora-search"><input value="" placeholder="搜索"></label></header><div class="lora-library-grid">${Array.from({ length: 12 }, (_, index) => `<article class="lora-library-card fixture-card"><div class="lora-card-preview"><img alt="" src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='16'/%3E"><button class="lora-card-query" type="button" data-query="${index}">查询</button><span class="lora-card-state">挂载</span></div><div class="lora-card-copy"><strong>测试 LoRA ${index}</strong><small>128 MB</small><button class="lora-card-detail" type="button" data-detail="${index}">查看详细信息</button></div></article>`).join("")}</div></main></div><footer class="lora-modal-foot"><button type="button">完成</button></footer></section></div><script>let clicked=0;document.addEventListener('click',event=>{if(event.target.matches('.lora-card-query,.lora-card-detail'))clicked+=1});for(const button of document.querySelectorAll('.lora-card-query,.lora-card-detail'))button.click();const cards=[...document.querySelectorAll('.lora-library-card')].map(node=>{const r=node.getBoundingClientRect();return [r.left,r.top,r.right,r.bottom]});const modal=document.querySelector('.lora-modal'),grid=document.querySelector('.lora-library-grid');document.body.dataset.geometry=JSON.stringify({width:innerWidth,cards,modal:[modal.clientWidth,modal.scrollWidth],grid:[grid.clientWidth,grid.scrollWidth],clicked});</script></body></html>`;
}

function hasIntersection(first, second) {
  return first[0] < second[2] && first[2] > second[0] && first[1] < second[3] && first[3] > second[1];
}

test("local Chromium fixture keeps LoRA modal cards in flow across 1100/800/600 breakpoints", {
  skip: skipBrowserFixture && "XIRAI_SKIP_BROWSER_FIXTURES keeps CI independent of runner browsers",
}, async (context) => {
  const styles = await readFile(join(projectRoot, "src", "styles.css"), "utf8");
  assert.match(styles, /\.lora-library-grid \{[^}]*grid-template-columns: repeat\(auto-fill, minmax\(180px, 1fr\)\)/);
  assert.match(styles, /\.lora-card-preview \{[^}]*aspect-ratio: 3 \/ 4/);
  assert.match(styles, /\.lora-card-preview img \{[^}]*object-fit: cover/);
  assert.match(styles, /@media \(max-width: 1100px\)/);
  assert.match(styles, /@media \(max-width: 800px\)/);
  assert.match(styles, /@media \(max-width: 600px\)/);
  assert.match(styles, /\.lora-categories \{[^}]*overflow-x: auto/);

  const chrome = await chromePath();
  if (!chrome) {
    context.skip("Chromium is unavailable; source CSS contract was checked");
    return;
  }
  const folder = await mkdtemp(join(tmpdir(), "xirai-lora-layout-"));
  try {
    const htmlPath = join(folder, "fixture.html");
    const cssHref = pathToFileURL(join(projectRoot, "src", "styles.css")).href;
    await writeFile(htmlPath, fixture(cssHref), "utf8");
    for (const width of [2000, 1100, 800, 600]) {
      const screenshot = join(folder, `lora-${width}.png`);
      const { stdout } = await execFileAsync(chrome, [
        "--headless=new", "--disable-gpu", "--hide-scrollbars", `--window-size=${width},1200`, `--screenshot=${screenshot}`, "--dump-dom", pathToFileURL(htmlPath).href,
      ], { windowsHide: true, timeout: 30000, maxBuffer: 2 * 1024 * 1024 });
      const documentMatch = stdout.match(/data-geometry="([^"]+)"/);
      assert.ok(documentMatch, `geometry result missing at ${width}`);
      const geometry = JSON.parse(documentMatch[1].replace(/&quot;/g, '"'));
      assert.equal(geometry.clicked, 24, `overlay buttons remain clickable at ${width}`);
      assert.ok(geometry.modal[1] <= geometry.modal[0], `modal horizontal overflow at ${width}`);
      assert.ok(geometry.grid[1] <= geometry.grid[0], `grid horizontal overflow at ${width}`);
      for (let index = 0; index < geometry.cards.length; index += 1) {
        for (let other = index + 1; other < geometry.cards.length; other += 1) {
          assert.equal(hasIntersection(geometry.cards[index], geometry.cards[other]), false, `cards ${index}/${other} overlap at ${width}`);
        }
      }
      assert.ok((await stat(screenshot)).size > 1024, `screenshot missing at ${width}`);
    }
  } finally {
    await rm(folder, { recursive: true, force: true });
  }
});
