import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { browserUrlCommand, openBrowser } from "./browser-url.mjs";
import { isApplicationReady } from "./service-readiness.js";
import { assertStartupPortsFree, startupDisposition } from "./startup-admission.mjs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const source = await readFile(new URL("./start.mjs", import.meta.url), "utf8");
const setup = await readFile(new URL("./setup-gui.mjs", import.meta.url), "utf8");
const html = await readFile(new URL("./setup-ui.html", import.meta.url), "utf8");
test("shared default URL descriptor has one URL, no profile/window/LoRA intent on three platforms", () => {
  for (const platform of ["win32", "darwin", "linux"]) for (const suffix of ["/", "/config"]) {
    const url = `http://localhost:18765${suffix}`, calls = [], child = new EventEmitter(); let unref = false;
    child.unref = () => { unref = true; };
    openBrowser(url, { platform, spawnProcess: (...args) => { calls.push(args); return child; } });
    assert.equal(calls.length, 1); assert.equal(unref, true);
    const expected = browserUrlCommand(url, platform);
    assert.deepEqual(calls[0].slice(0, 2), [expected.command, expected.args]);
    assert.equal(expected.args.filter((arg) => arg === url).length, 1);
    assert.doesNotMatch(JSON.stringify(expected), /--app|--new-window|profile|\/lora/);
  }
  for (const text of [source, setup]) {
    assert.match(text, /import \{ openBrowser \} from "\.\/browser-url\.mjs"/);
    assert.doesNotMatch(text, /function openBrowser|XIRAI_BROWSER_PROFILE|--user-data-dir/);
  }
});
test("opener reports spawn failure and refuses shell/protocol injection without spawning", () => {
  const child = new EventEmitter(), messages = []; child.unref = () => {};
  openBrowser("http://localhost:7709/", { spawnProcess: () => child, report: (text) => messages.push(text) });
  child.emit("error", Error("fixture")); assert.equal(messages.length, 1);
  for (const url of ["file:///x", "http://localhost/&bad", 'http://localhost/"bad']) assert.throws(() => browserUrlCommand(url));
});
test("ready owner predicate denies unready, foreign workspace/protocol and malformed health", () => {
  const good = { status: "ready", protocol: 39, workspace_id: "fixture" };
  assert.equal(isApplicationReady(good, "fixture"), true);
  assert.equal(startupDisposition(good, "fixture"), "reuse");
  for (const health of [{ ...good, status: "loading" }, { ...good, protocol: 38 }, { ...good, workspace_id: "foreign" }, {}]) {
    assert.equal(isApplicationReady(health, "fixture"), false); assert.equal(startupDisposition(health, "fixture"), "blocked");
  }
  assert.equal(startupDisposition(null, "fixture"), "probe-ports");
  assert.equal(isApplicationReady(good, undefined), false);
});
test("busy port admission never kills or reuses an unknown owner", async () => {
  const calls = [];
  const create = () => { const s = new EventEmitter(); s.listen = (target, ready) => { calls.push(target); s.emit("error", Object.assign(Error("busy"), { code: "EADDRINUSE" })); }; return s; };
  await assert.rejects(assertStartupPortsFree([{ port: 7709, host: "0.0.0.0" }], create), /未停止任何服务/);
  assert.equal(calls.length, 1);
});
test("Windows/Linux wrappers no longer pre-kill; no-open/reuse and supervisor no-reopen remain wired", async () => {
  for (const name of ["../Start-XirAI.bat", "../Start-XirAI.sh"]) {
    const wrapper = await readFile(new URL(name, import.meta.url), "utf8");
    assert.doesNotMatch(wrapper, /cleanup-processes|taskkill|pkill/); assert.match(wrapper, /start\.mjs/);
  }
  assert.match(source, /if \(!noOpen\) openBrowser\(mainUrl\);/);
  assert.match(source, /if \(openOnReady && !noOpen\) openBrowser\(mainUrl\);/);
  assert.match(source, /if \(decision\.restart\) \{\s*openOnReady = false;/);
  assert.ok(source.indexOf('=== "reuse"') < source.indexOf("assertStartupPortsFree(["));
  assert.match(html, /import \{ isApplicationReady \} from "\/setup-readiness\.js"/);
  assert.equal((html.match(/await waitForApplication\(\)/g) || []).length, 2);
  assert.equal((html.match(/window\.location\.replace\("\/"\)/g) || []).length, 1);
  assert.doesNotMatch(html, /window\.open/);
  assert.match(setup, /端口.*已被占用/);
});
test("actual launcher control flow reuses healthy service once and refuses foreign/unknown occupied ports without spawning", async () => {
  const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
  const script = source.replace(/^import .*;\r?\n/gm, '').replaceAll('import.meta.url', JSON.stringify(new URL('./start.mjs', import.meta.url).href));
  for (const [health, flags, busy, expectedCode, expectedOpen] of [
    [{status:'ready',protocol:39,workspace_id:'owned'},[],false,0,1],
    [{status:'ready',protocol:39,workspace_id:'owned'},['--no-open','--exit-after-ready'],false,0,0],
    [{status:'loading',protocol:39,workspace_id:'owned'},[],false,1,0],
    [{status:'ready',protocol:38,workspace_id:'owned'},[],false,1,0],
    [{status:'ready',protocol:39,workspace_id:'foreign'},[],false,1,0],
    [null,[],true,1,0],
  ]) {
    const events = [], exit = Object.assign(Error('controlled exit'), { controlled: true });
    const mocks = {
      spawn: () => { throw Error('Never spawn/kill any real service'); }, spawnSync: () => { throw Error('Never check/install real environment'); },
      createHash: () => ({ update: () => ({ digest: () => 'owned' }) }), path, fileURLToPath,
      process: { argv: flags, env: { WEB_PORT: '18999', INFERENCE_PORT: '18998', XIRAI_STATE_DIR: 'scratch' }, platform: 'win32', versions: { node: '24.18.0' }, exit: code => { events.push({exit:code}); throw exit; } },
      loadLocalEnv: () => ({}), createHttpFetch: () => async () => ({ ok: !!health, json: async () => health }),
      startupDisposition, APPLICATION_PROTOCOL: 39, openBrowser: url => events.push({url}),
      assertStartupPortsFree: async ports => { events.push({ports}); if(busy) throw Error('owned fixture port busy; no kill'); },
      writeDiagnosticLog: (_root, entry) => events.push({diagnostic:entry.message}),
      console: { log() {}, error() {} }, AbortController, setTimeout, clearTimeout,
    };
    await assert.rejects(new AsyncFunction(...Object.keys(mocks), script)(...Object.values(mocks)), e => e === exit);
    assert.equal(events.find(x=>Object.hasOwn(x,'exit')).exit, expectedCode);
    assert.equal(events.filter(x=>x.url).length, expectedOpen);
    if(expectedOpen) assert.equal(events.find(x=>x.url).url, 'http://localhost:18999/');
    assert.equal(events.some(x=>x.ports), busy);
  }
});
test("actual Setup return wait requires ready/protocol/workspace and redirects this tab only", async () => {
  const part = html.slice(html.indexOf('async function waitForApplication()'), html.indexOf('async function closeConfigurator()'));
  const create = new Function('state','fetch','isApplicationReady','window','setTimeout','AbortSignal', part+';return waitForApplication;');
  const good={status:'ready',protocol:39,workspace_id:'owned'};
  for(const health of [{...good,status:'loading'},{...good,protocol:38},{...good,workspace_id:'foreign'},null,good]) {
    const redirects=[], reads=[];
    const wait=create({workspaceId:'owned'},async url=>{reads.push(url);return {ok:health!==null,json:async()=>health};},isApplicationReady,{location:{replace:url=>redirects.push(url)}},resolve=>resolve(),{timeout:()=>undefined});
    if(health===good) { await wait();assert.deepEqual(redirects,['/']);assert.equal(reads.length,1); }
    else { await assert.rejects(wait(),/未跳转/);assert.deepEqual(redirects,[]);assert.equal(reads.length,128); }
  }
});
