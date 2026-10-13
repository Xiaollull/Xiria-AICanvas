import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { access, mkdtemp, readFile, rm } from "node:fs/promises";
import { createServer } from "node:http";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";
import { build as buildFixture } from "vite";
import react from "@vitejs/plugin-react";
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

  // Real App/lazy chunks and interactions, with the existing CPU-only API fixture.
  // Never load the production Vite config: its runtime/download recovery belongs
  // to the application, not a render smoke test. No unexpected API can escape.
  const profile = await mkdtemp(join(tmpdir(), "xirai-app-mount-"));
  const output = join(profile, "build");
  await buildFixture({ configFile: false, root: projectRoot, cacheDir: join(profile, "vite-cache"),
    plugins: [react()], logLevel: "error", css: { postcss: { plugins: [] } },
    build: { outDir: output, emptyOutDir: true,
      rolldownOptions: { input: join(projectRoot, "scripts", "app-mount.test.fixture.html") } } });
  const html = await readFile(join(output, "scripts", "app-mount.test.fixture.html"), "utf8");

  // Serve the real production App and its lazy chunks over owned HTTP. There is
  // no dev-client/HMR socket, default-root recovery, browser profile or live API.
  const proxy = createServer(async (request, response) => {
    try {
      if (request.url === "/") {
        response.writeHead(200, { "content-type": "text/html; charset=utf-8" });
        response.end(html.replace("<head>", `<head>${probe}`));
        return;
      }
      if (request.url.startsWith("/api/")) { response.writeHead(409); response.end("Unexpected API forbidden in render fixture"); return; }
      const file = resolve(output, `.${decodeURIComponent(new URL(request.url, "http://fixture").pathname)}`);
      if (!file.startsWith(output + "/") && !file.startsWith(output + "\\")) { response.writeHead(403); response.end(); return; }
      const type = file.endsWith(".js") ? "text/javascript" : file.endsWith(".css") ? "text/css" : file.endsWith(".svg") ? "image/svg+xml" : "application/octet-stream";
      const content = await readFile(file);
      response.writeHead(200, { "content-type": type });
      response.end(content);
    } catch (error) {
      response.writeHead(502);
      response.end(String(error));
    }
  });

  try {
    await new Promise((done) => proxy.listen(0, "127.0.0.1", done));
    const { stdout } = await execFileAsync(chrome, [
      "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
      "--no-first-run", "--no-default-browser-check", `--user-data-dir=${profile}`,
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
    await rm(profile, { recursive: true, force: true });
  }
});
