import test from "node:test";
import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { access, mkdtemp, rm } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";
import { readFile } from "node:fs/promises";

const execFileAsync = promisify(execFile);
const root = resolve(fileURLToPath(new URL("..", import.meta.url)));
const skipBrowserFixture = process.env.XIRAI_SKIP_BROWSER_FIXTURES === "1";
const chromeCandidates = [
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
];

async function chromePath() {
  for (const candidate of chromeCandidates) {
    try { await access(candidate); return candidate; } catch {}
  }
  return null;
}

function fixture({ sidebar, expanded }, editorSource) {
  return `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="stylesheet" href="/src/styles.css"><style>
  :root{--n-01:#090a0c;--n-02:#111218;--n-03:#181923;--n-05:#252636;--n-06:#303146;--n-07:#3b3d55;--n-08:#4b4d68;--n-13:#7d8097;--n-15:#9295ad;--n-17:#a9acc3;--n-18:#b2b5ca;--n-20:#c1c4d8;--n-27:#ececf6;--n-28:#f3f1fa;--n-31:#faf9fd;--n-32:#fff;--lime:#c8acfb;--accent-rgb:200 172 251;--line:#303146;--z-immersive:100;box-sizing:border-box}*{box-sizing:border-box}html,body{margin:0;width:100%;height:100%;overflow:hidden}button,input,select{font:12px sans-serif}.image-viewer-backdrop{position:fixed;inset:0}.image-viewer.fixture{max-width:none;max-height:none;height:${expanded ? "100vh" : "calc(100vh - 36px)"}}.viewer-edge-panel .tall{height:900px}.viewer-history{min-height:0}
  </style></head><body><div class="image-viewer-backdrop ${expanded ? "expanded" : ""}"><section class="image-viewer fixture ${expanded ? "expanded" : ""} ${sidebar ? "sidebar-open" : ""}"><header class="image-viewer-head"><strong>fixture</strong></header><div class="image-viewer-body"><aside class="viewer-history"></aside><section class="viewer-workspace" id="workspace"><div class="viewer-toolbar" id="toolbar">
    <div class="viewer-toolbar-group viewer-tool-group"><button>-</button><output>100%</output><button>+</button></div>
    <div class="viewer-toolbar-group viewer-editor-tools"><button aria-pressed="true"><span>移动</span></button><button><span>画笔</span></button><button><span>橡皮</span></button><button><span>文字</span></button></div>
    <div class="viewer-toolbar-group viewer-editor-properties viewer-text-properties"><label class="viewer-property-field viewer-text-content-field"><span>内容</span><input value="Browser fixture text"></label><label class="viewer-property-field"><span>字号</span><input class="bounded-number" value="48"><em>px</em></label><label class="viewer-property-color"><span>颜色</span><input type="color" value="#c8acfb"></label><label class="viewer-property-select"><span>字重</span><select><option>400</option></select></label><div class="viewer-property-align"><button>L</button><button>C</button><button>R</button></div><label class="viewer-property-field"><span>角度</span><input class="bounded-number" value="45"><em>°</em></label></div>
    <div class="viewer-toolbar-group"><button><span>对齐与线条</span></button><button><span>拼图模板</span></button><button><span>应用编辑</span></button></div>
    <div class="viewer-toolbar-group viewer-toolbar-results"><button><span>保存拼图</span></button><button><span>重新拼图</span></button></div>
  </div><div id="popover" class="viewer-edge-panel"><div class="tall"></div></div><div class="image-viewer-canvas tool-brush" id="canvas"></div></section></div></section></div><pre id="result">pending</pre><script>window.fixtureErrors=[];addEventListener("error",event=>fixtureErrors.push(event.message));addEventListener("unhandledrejection",event=>fixtureErrors.push(String(event.reason)));setTimeout(()=>{const node=document.getElementById("result");if(node.textContent==="pending")node.textContent=JSON.stringify({fatal:"fixture timeout",errors:fixtureErrors})},1500)</script><script>${editorSource.replace(/^export /gm, "")}</script>
  <script>
    (async()=>{ try {
    const source=document.createElement("canvas"); source.width=32; source.height=32; const sourceContext=source.getContext("2d"); sourceContext.fillStyle="#ff0000"; sourceContext.fillRect(0,0,32,32);
    const brush=normalizePaintStroke({tool:"brush",color:"#c8acfb",size:6,opacity:1,points:[{x:8,y:16},{x:24,y:16}]},{width:32,height:32});
    const eraser=normalizePaintStroke({tool:"eraser",size:8,opacity:1,points:[{x:16,y:16}]},{width:32,height:32});
    const painted=document.createElement("canvas"); renderRasterLayer(painted,source,{naturalWidth:32,naturalHeight:32,paintStrokes:[brush,eraser]});
    const undone=document.createElement("canvas"); renderRasterLayer(undone,source,{naturalWidth:32,naturalHeight:32,paintStrokes:[brush]});
    const pixel=(canvas,x,y)=>[...canvas.getContext("2d").getImageData(x,y,1,1).data];
    const createdText=normalizeTextLayer({text:"NEW",fontSize:24,fontWeight:700,color:"#c8acfb",naturalWidth:72,naturalHeight:54,x:0,y:0,scale:1,rotation:0});
    const text=normalizeTextLayer({...createdText,text:"ROTATE",naturalWidth:118,rotation:45});
    const bounds=viewerEditorLayerBounds(text); const textCanvas=document.createElement("canvas"); textCanvas.width=Math.ceil(bounds.width); textCanvas.height=Math.ceil(bounds.height); drawViewerLayer(textCanvas.getContext("2d"),text,null,bounds);
    const textPixels=textCanvas.getContext("2d").getImageData(0,0,textCanvas.width,textCanvas.height).data; let textAlpha=0; for(let index=3;index<textPixels.length;index+=4) if(textPixels[index]) textAlpha++;
    await new Promise(resolve=>setTimeout(resolve,20));
    const workspace=document.getElementById("workspace"),toolbar=document.getElementById("toolbar"),stage=document.getElementById("canvas"),popover=document.getElementById("popover");
    const toolbarHeight=Math.ceil(toolbar.getBoundingClientRect().height); workspace.style.setProperty("--viewer-toolbar-height",toolbarHeight+"px"); popover.style.top=(toolbarHeight+1)+"px";
    await new Promise(resolve=>setTimeout(resolve,20));
    const box=node=>{const r=node.getBoundingClientRect();return{left:r.left,top:r.top,right:r.right,bottom:r.bottom,width:r.width,height:r.height}};
    const wr=box(workspace),tr=box(toolbar),sr=box(stage),pr=box(popover),history=box(document.querySelector(".viewer-history")); const groups=[...toolbar.children].map(box);
    const intersects=(a,b)=>Math.min(a.right,b.right)-Math.max(a.left,b.left)>.5&&Math.min(a.bottom,b.bottom)-Math.max(a.top,b.top)>.5;
    const layout={workspace:wr,toolbar:tr,stage:sr,popover:pr,history,groupsInside:groups.every(r=>r.left>=tr.left-.5&&r.right<=tr.right+.5&&r.top>=tr.top-.5&&r.bottom<=tr.bottom+.5),groupsOverlap:groups.some((r,index)=>groups.slice(index+1).some(other=>intersects(r,other))),canvasBelow:sr.top>=tr.bottom-.5,popoverInside:pr.left>=wr.left-.5&&pr.right<=wr.right+.5&&pr.top>=tr.bottom-.5&&pr.bottom<=wr.bottom+.5,sidebarClear:${sidebar ? "tr.left>=history.right-.5" : "true"}};
    let budgetError="";try{viewerCanvasDimensions(8193,8193,"浏览器测试画布")}catch(error){budgetError=error.message}
    const result={layout,paint:{brush:pixel(undone,8,16),erased:pixel(painted,16,16),undo:pixel(undone,16,16)},text:{created:createdText.text,changed:text.text,rotation:text.rotation,bounds,textAlpha},budgetError,errors:fixtureErrors}; document.getElementById("result").textContent=JSON.stringify(result);
    } catch(error) { document.getElementById("result").textContent=JSON.stringify({fatal:String(error),errors:fixtureErrors}); } })();
  </script></body></html>`;
}

async function withServer(run) {
  const editorSource = await readFile(join(root, "src", "viewer-editor.js"), "utf8");
  const server = createServer(async (request, response) => {
    try {
      if (request.url === "/src/styles.css") {
        response.setHeader("content-type", "text/css; charset=utf-8");
        response.end(await readFile(join(root, "src", "styles.css")));
      } else if (request.url === "/src/viewer-editor.js") {
        response.setHeader("content-type", "text/javascript; charset=utf-8");
        response.end(await readFile(join(root, "src", "viewer-editor.js")));
      } else {
        const url = new URL(request.url, "http://fixture");
        response.setHeader("content-type", "text/html; charset=utf-8");
        response.end(fixture({ sidebar: url.searchParams.get("sidebar") === "1", expanded: url.searchParams.get("expanded") === "1" }, editorSource));
      }
    } catch (error) {
      response.statusCode = 500;
      response.end(String(error));
    }
  });
  await new Promise((resolvePromise) => server.listen(0, "127.0.0.1", resolvePromise));
  try { return await run(server.address().port); } finally { await new Promise((resolvePromise) => server.close(resolvePromise)); }
}

function resultFromDom(dom) {
  const match = dom.match(/<pre id="result">([^<]+)<\/pre>/);
  assert.ok(match && match[1] !== "pending", `browser fixture did not finish:\n${dom.slice(-3000)}`);
  return JSON.parse(match[1].replaceAll("&quot;", '"').replaceAll("&amp;", "&"));
}

test("real Chromium validates edited pixels and toolbar geometry at required widths", { skip: skipBrowserFixture && "browser fixtures disabled" }, async (context) => {
  const chrome = await chromePath();
  if (!chrome) { context.skip("Chromium is unavailable"); return; }
  const profile = await mkdtemp(join(tmpdir(), "xirai-viewer-browser-"));
  try {
    await withServer(async (port) => {
      for (const scenario of [
        { width: 1580, sidebar: false, expanded: true },
        { width: 1366, sidebar: true, expanded: false },
        { width: 1280, sidebar: true, expanded: false },
        { width: 1024, sidebar: true, expanded: false },
      ]) {
        const query = `sidebar=${scenario.sidebar ? 1 : 0}&expanded=${scenario.expanded ? 1 : 0}`;
        const { stdout } = await execFileAsync(chrome, ["--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1", `--user-data-dir=${profile}`, `--window-size=${scenario.width},820`, "--virtual-time-budget=2500", "--dump-dom", `http://127.0.0.1:${port}/?${query}`], { windowsHide: true, timeout: 30000, maxBuffer: 4 * 1024 * 1024 });
        const result = resultFromDom(stdout);
        assert.equal(result.fatal, undefined, `${scenario.width}px fixture failed: ${JSON.stringify(result)}`);
        assert.deepEqual(result.errors, [], `${scenario.width}px browser console errors`);
        assert.equal(result.layout.groupsInside, true, `${scenario.width}px toolbar group escaped`);
        assert.equal(result.layout.groupsOverlap, false, `${scenario.width}px toolbar groups overlap`);
        assert.equal(result.layout.canvasBelow, true, `${scenario.width}px canvas is not below measured toolbar`);
        assert.equal(result.layout.popoverInside, true, `${scenario.width}px popover escaped workspace`);
        assert.equal(result.layout.sidebarClear, true, `${scenario.width}px toolbar overlaps sidebar`);
        assert.deepEqual(result.paint.brush, [200, 172, 251, 255]);
        assert.equal(result.paint.erased[3], 0, "eraser must remove both original and brush alpha");
        assert.deepEqual(result.paint.undo, [200, 172, 251, 255], "undo replay must remove exactly the eraser stroke");
        assert.deepEqual([result.text.created, result.text.changed, result.text.rotation], ["NEW", "ROTATE", 45], "text must be created, changed, then rotated");
        assert.ok(result.text.bounds.width > 120 && result.text.bounds.height > 120, "rotated text AABB must expand");
        assert.ok(result.text.textAlpha > 100, "rotated semantic text must produce visible pixels");
        assert.match(result.budgetError, /64 MP/, "production canvas budget must reject unsafe browser allocations");
      }
    });
  } finally {
    await rm(profile, { recursive: true, force: true });
  }
});
