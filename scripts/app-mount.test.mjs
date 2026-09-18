import assert from "node:assert/strict";
import { execFile, spawn } from "node:child_process";
import { access } from "node:fs/promises";
import { createServer } from "node:http";
import { resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

// A page that throws while rendering leaves a blank window and nothing else: the build succeeds,
// because bundling never asks whether an identifier exists, and every source assertion in this
// suite still passes, because the text it looks for is right there. The only thing that catches it
// is running the app. So this loads the real dev server in a headless browser, visits each page and
// works the post-processing chain, and fails on anything the page logged as an error.
//
// It is the same skippable browser fixture `lora-card-layout.test.mjs` uses, for the same reason:
// CI must not depend on a runner's browsers.

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

/** Collects what the page throws or logs as an error, then drives it and reports through the title. */
const probe = [
  "<script>",
  "window.__errors=[];",
  "const realError=console.error.bind(console);",
  "console.error=(...a)=>{window.__errors.push(a.map((i)=>(i&&i.stack)||String(i)).join(' '));realError(...a);};",
  "addEventListener('error',(e)=>{window.__errors.push((e.error&&e.error.stack)||e.message);});",
  "addEventListener('unhandledrejection',(e)=>{window.__errors.push(String((e.reason&&e.reason.stack)||e.reason));});",
  "const wait=(ms)=>new Promise((done)=>setTimeout(done,ms));",
  "const nav=(label)=>[...document.querySelectorAll('nav.main-nav button')].find((item)=>item.textContent.includes(label));",
  "addEventListener('load',async()=>{",
  "  await wait(2500);",
  "  const seen=[];",
  "  for(const label of ['\\u56fe\\u751f\\u56fe','\\u753b\\u5eca','\\u5de5\\u5177\\u7bb1','\\u6587\\u751f\\u56fe']){",
  "    const button=nav(label);",
  "    if(!button){window.__errors.push('missing nav: '+label);continue;}",
  "    button.click();",
  "    await wait(2200);",
  "    const root=document.getElementById('root');",
  "    seen.push(label+'='+(root?root.innerHTML.length:-1));",
  "  }",
  "  const clickAll=async(selector)=>{const items=[...document.querySelectorAll(selector)];for(const item of items){item.click();await wait(600);}return items.length;};",
  "  seen.push('expanded='+await clickAll('.postprocess-stage-summary'));",
  "  await wait(1000);",
  "  const add=document.querySelector('.resample-chain-add button');",
  "  if(!add){window.__errors.push('no add-unit button');}else{add.click();await wait(1200);seen.push('added');}",
  "  seen.push('cards='+document.querySelectorAll('.postprocess-stage-card').length);",
  "  const move=[...document.querySelectorAll('.resample-stage-actions button')].filter((item)=>!item.disabled);",
  "  if(move.length){move[move.length-1].click();await wait(800);seen.push('moved');}",
  "  const remove=document.querySelector('.resample-stage-remove');",
  "  if(remove){remove.click();await wait(800);seen.push('removed');}",
  "  const root=document.getElementById('root');",
  "  document.title='LEN:'+(root?root.innerHTML.length:-1)+' SEEN:'+seen.join(',')+' ERRORS::'+window.__errors.join(' ||| ');",
  "});",
  "</script>",
].join("\n");

test("every page mounts and the resample chain can be worked without throwing", {
  skip: skipBrowserFixture && "XIRAI_SKIP_BROWSER_FIXTURES keeps CI independent of runner browsers",
}, async (context) => {
  const chrome = await chromePath();
  if (!chrome) {
    context.skip("Chromium is unavailable");
    return;
  }

  // Ask the OS for a free port and hand that to Vite, rather than parsing one back out of its
  // ANSI-decorated banner.
  const scout = createServer();
  await new Promise((done) => scout.listen(0, "127.0.0.1", done));
  const port = scout.address().port;
  await new Promise((done) => scout.close(done));

  const vite = spawn(process.execPath, ["node_modules/vite/bin/vite.js", "--port", String(port), "--strictPort"], {
    cwd: projectRoot, stdio: ["ignore", "pipe", "pipe"], windowsHide: true,
  });
  let output = "";
  vite.stdout.on("data", (chunk) => { output += chunk; });
  vite.stderr.on("data", (chunk) => { output += chunk; });

  const started = await new Promise((done) => {
    const timer = setInterval(() => {
      if (output.includes("ready in") && output.includes(String(port))) {
        clearInterval(timer);
        done(true);
      }
    }, 250);
    setTimeout(() => { clearInterval(timer); done(false); }, 60000);
  });

  // The probe has to reach the page, and the page is served by Vite rather than read from disk, so
  // a proxy in front of it carries the script instead of the repository's own index.html.
  const proxy = createServer(async (request, response) => {
    try {
      const upstream = await fetch(`http://127.0.0.1:${port}${request.url}`, {
        headers: { ...request.headers, host: `127.0.0.1:${port}` },
      });
      const type = upstream.headers.get("content-type") || "application/octet-stream";
      if (type.includes("text/html")) {
        response.writeHead(upstream.status, { "content-type": type });
        response.end((await upstream.text()).replace("<head>", `<head>${probe}`));
        return;
      }
      response.writeHead(upstream.status, { "content-type": type });
      response.end(Buffer.from(await upstream.arrayBuffer()));
    } catch (error) {
      response.writeHead(502);
      response.end(String(error));
    }
  });

  try {
    assert.ok(started, `the dev server did not start:\n${output.slice(0, 1500)}`);
    await new Promise((done) => proxy.listen(0, "127.0.0.1", done));
    const { stdout } = await execFileAsync(chrome, [
      "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
      "--virtual-time-budget=60000", "--dump-dom", `http://127.0.0.1:${proxy.address().port}/`,
    ], { windowsHide: true, timeout: 180000, maxBuffer: 32 * 1024 * 1024 });

    const title = stdout.match(/<title>([\s\S]*?)<\/title>/);
    assert.ok(title, "the page never reported back");
    const [state, errors] = title[1].split("ERRORS::");
    assert.equal(errors.trim(), "", `the page logged an error:\n${errors.trim().slice(0, 2000)}`);

    const rendered = Number(state.match(/LEN:(-?\d+)/)?.[1] ?? -1);
    assert.ok(rendered > 5000, `the root rendered ${rendered} characters, so the app did not mount`);
    // Each page is its own lazy chunk; a blank one is a chunk that threw on its first render.
    for (const page of ["图生图", "画廊", "工具箱", "文生图"]) {
      const length = Number(state.match(new RegExp(`${page}=(-?\\d+)`))?.[1] ?? -1);
      assert.ok(length > 5000, `${page} rendered ${length} characters`);
    }
    // The chain's own controls: expanding a card, adding a unit, reordering and deleting.
    for (const step of ["added", "moved", "removed"]) {
      assert.match(state, new RegExp(step), `the chain editor never got as far as "${step}": ${state}`);
    }
    const cards = Number(state.match(/cards=(\d+)/)?.[1] ?? 0);
    assert.ok(cards >= 4, `adding a unit left ${cards} stage cards`);
  } finally {
    proxy.close();
    vite.kill();
  }
});
