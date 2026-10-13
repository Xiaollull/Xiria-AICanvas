// Historical round-1 (append queue) harness. The current replacement/cache contract
// is validated by scripts/wd14-desk-browser-smoke.mjs; do not use this as current acceptance.
// Opt-in live WD14 UI acceptance. Existing services are never restarted.
// /models and /tag are REAL; folder/download/recovery requests below are MOCKED.
// All browser profiles, synthetic fixtures, TXT exports and evidence stay in system temp.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdir, writeFile, readFile, readdir } from 'node:fs/promises';
import { createServer } from 'node:net';
import path from 'node:path';

const origin = process.env.WD14_UI_ORIGIN || 'http://127.0.0.1:7709';
const evidence = process.env.WD14_UI_EVIDENCE;
if (!evidence || !process.env.LOCALAPPDATA || !path.resolve(evidence).toLowerCase().startsWith((path.join(process.env.LOCALAPPDATA, 'Temp', 'opencode') + path.sep).toLowerCase())) throw new Error('WD14_UI_EVIDENCE must be a child of approved opencode system temp');
assert.ok(['127.0.0.1', 'localhost'].includes(new URL(origin).hostname), 'live origin must be loopback');
await mkdir(evidence, { recursive: true });
await mkdir(path.join(evidence, 'exports'), { recursive: true });
const server = createServer();
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const debugPort = server.address().port;
await new Promise(resolve => server.close(resolve));
const browser = spawn('C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe', ['--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check', '--disable-extensions', `--remote-debugging-port=${debugPort}`, `--user-data-dir=${path.join(evidence, 'profile')}`, '--window-size=1440,900', 'about:blank'], { stdio: 'ignore', windowsHide: true });
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
async function wait(check, timeout = 45000) {
  const until = Date.now() + timeout;
  while (Date.now() < until) { const value = await check(); if (value) return value; await sleep(150); }
  throw new Error('Browser condition timed out: ' + check.toString());
}
let ws;
const report = { origin, live: ['models', 'tag (first two images)'], mocked: ['active recovery', 'batch', 'download', 'mock job polling/cancellation', 'slow tag for stop-only scenario'], checks: [], warnings: [], screenshots: [], runtimeErrors: [], realTagResponses: [] };
let mockInstall = null, mockBatch = null, batchPayload = null, recovery = false, pollPublishing = false, mockTag = false, mockTagCalls = 0;
try {
  const tab = await wait(async () => {
    try { return (await (await fetch(`http://127.0.0.1:${debugPort}/json/list`)).json()).find(x => x.type === 'page'); } catch { return null; }
  });
  ws = new WebSocket(tab.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { ws.addEventListener('open', resolve, { once: true }); ws.addEventListener('error', reject, { once: true }); });
  let serial = 0;
  const pending = new Map();
  function send(method, params = {}) {
    const id = ++serial;
    return new Promise((resolve, reject) => { pending.set(id, { resolve, reject }); ws.send(JSON.stringify({ id, method, params })); });
  }
  ws.addEventListener('message', event => {
    const message = JSON.parse(event.data);
    if (message.id) {
      const p = pending.get(message.id); pending.delete(message.id);
      if (message.error) p?.reject(new Error(message.error.message)); else p?.resolve(message.result);
    }
    if (message.method === 'Runtime.exceptionThrown') report.runtimeErrors.push(message.params.exceptionDetails.exception?.description || message.params.exceptionDetails.text);
    if (message.method === 'Network.responseReceived' && message.params.response.url.includes('/wd14/tag?') && !mockTag) report.realTagResponses.push({ status: message.params.response.status, url: message.params.response.url });
    if (message.method === 'Fetch.requestPaused') {
      const { request, requestId } = message.params;
      const url = new URL(request.url);
      let payload = {}, status = 200;
      if (url.pathname.endsWith('/tag') && mockTag) {
        mockTagCalls++;
        void sleep(1500).then(() => send('Fetch.fulfillRequest', { requestId, responseCode: 200, responseHeaders: [{ name: 'Content-Type', value: 'application/json' }], body: Buffer.from(JSON.stringify({ rating: { general: .8, sensitive: .1, questionable: .07, explicit: .03 }, tags: [] })).toString('base64') })).catch(error => report.runtimeErrors.push(error.message));
        return;
      }
      if (url.pathname.endsWith('/active')) payload = { jobs: recovery ? [{ id: 'mock-recovery', kind: 'download', state: 'error', total: 2, done: 0, skipped: 0, failed: 0, warning: '发现历史安装 marker；请人工检查恢复，保留现场', error: '安装中断，需要人工检查' }] : [] };
      else if (url.pathname.includes('/download/')) { mockInstall = { id: 'mock-install', kind: 'download', state: 'running', total: 2, done: 1, skipped: 0, failed: 0 }; payload = mockInstall; }
      else if (url.pathname.endsWith('/mock-install/cancel')) { status = 409; mockInstall.state = 'publishing'; payload = { detail: '正在发布' }; }
      else if (url.pathname.endsWith('/mock-install')) { payload = { ...mockInstall }; if (pollPublishing) { mockInstall.state = 'done'; mockInstall.warning = '历史备份需人工检查；不会自动删除'; } pollPublishing = true; }
      else if (url.pathname.endsWith('/batch')) { batchPayload = JSON.parse(request.postData); mockBatch = { id: 'mock-batch', kind: 'batch', state: 'running', total: 3, done: 1, skipped: 1, failed: 0, current: 'synthetic.png', logs: ['模拟 UI 批量进度；没有写入服务端目录'] }; payload = mockBatch; status = 202; }
      else if (url.pathname.endsWith('/mock-batch/cancel')) { mockBatch.state = 'cancelled'; payload = { ok: true, state: 'cancelled' }; }
      else if (url.pathname.endsWith('/mock-batch')) payload = mockBatch;
      void send('Fetch.fulfillRequest', { requestId, responseCode: status, responseHeaders: [{ name: 'Content-Type', value: 'application/json' }], body: Buffer.from(JSON.stringify(payload)).toString('base64') }).catch(error => report.runtimeErrors.push(error.message));
    }
  });
  const evaluate = async expression => {
    const r = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
    return r.result.value;
  };
  const check = (name, detail) => { report.checks.push({ name, detail }); console.log('PASS', name, JSON.stringify(detail ?? true)); };
  const viewport = async (width, height = 900) => { await send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 1, mobile: false }); await sleep(250); };
  const screenshot = async name => { const shot = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: false }); const file = path.join(evidence, `${name}.png`); await writeFile(file, Buffer.from(shot.data, 'base64')); report.screenshots.push(file); };
  const clickText = async (selector, text) => evaluate(`[...document.querySelectorAll(${JSON.stringify(selector)})].find(x=>x.textContent.includes(${JSON.stringify(text)})).click()`);
  const fill = async (selector, value) => evaluate(`(() => {const e=document.querySelector(${JSON.stringify(selector)});const proto=e.tagName==='TEXTAREA'?HTMLTextAreaElement.prototype:HTMLInputElement.prototype;Object.getOwnPropertyDescriptor(proto,'value').set.call(e,${JSON.stringify(value)});e.dispatchEvent(new Event('input',{bubbles:true}));})()`);
  const select = async (selector, value) => evaluate(`(() => {const e=document.querySelector(${JSON.stringify(selector)});e.value=${JSON.stringify(value)};e.dispatchEvent(new Event('change',{bubbles:true}));})()`);
  const nav = async text => { await clickText('.main-nav button', text); await sleep(400); };
  await send('Page.enable'); await send('Runtime.enable'); await send('DOM.enable'); await send('Network.enable');
  await send('Fetch.enable', { patterns: [{ urlPattern: '*/api/inference/wd14/active*' }, { urlPattern: '*/api/inference/wd14/batch*' }, { urlPattern: '*/api/inference/wd14/download/*' }, { urlPattern: '*/api/inference/wd14/jobs/mock-*' }] });
  await send('Browser.grantPermissions', { origin, permissions: ['clipboardReadWrite', 'clipboardSanitizedWrite'] });
  await send('Browser.setDownloadBehavior', { behavior: 'allow', downloadPath: path.join(evidence, 'exports') });
  await viewport(1440);
  await send('Page.navigate', { url: origin });
  await wait(() => evaluate("Boolean(document.querySelector('.main-nav')) && !document.querySelector('.app-loader')"));
  // Track object URLs without changing their browser implementation.
  await evaluate("window.__wd14URLs={created:[],revoked:[]};const c=URL.createObjectURL.bind(URL),r=URL.revokeObjectURL.bind(URL);URL.createObjectURL=(b)=>{const u=c(b);__wd14URLs.created.push(u);return u};URL.revokeObjectURL=(u)=>{__wd14URLs.revoked.push(u);r(u)}");
  await nav('图片反推');
  await wait(() => evaluate("document.querySelector('.wd14-status .ready') !== null"));
  const layout = await evaluate("(() => {const rect=s=>{const r=document.querySelector(s).getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height}};return {work:rect('.wd14-work'),side:rect('.wd14-sidebar'),h1:document.querySelectorAll('.wd14-page h1').length,modelCount:document.querySelector('.wd14-controls select').options.length,range:getComputedStyle(document.querySelector('.wd14-slider input')).height,select:getComputedStyle(document.querySelector('.wd14-controls select')).height,scrollWidth:document.documentElement.scrollWidth}})()");
  assert.equal(layout.h1, 0); assert.equal(layout.modelCount, 16); assert.ok(layout.work.x < layout.side.x); assert.ok(layout.work.width > layout.side.width * 1.8); assert.equal(layout.select, '44px');
  assert.equal(await evaluate("document.querySelector('.wd14-grid').firstElementChild.matches('.wd14-work')"), true);
  check('1440 primary left, wide, no hero, real 16-model catalog and full controls', layout);
  await screenshot('desktop-1440-empty');
  await viewport(1920, 1080); await screenshot('desktop-1920-empty');
  await viewport(1440);
  // Real keyboard input on a real range, then reset to default for live inference.
  await evaluate("document.querySelector('.wd14-slider input').focus()");
  await send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'ArrowRight', code: 'ArrowRight', windowsVirtualKeyCode: 39 });
  await send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'ArrowRight', code: 'ArrowRight', windowsVirtualKeyCode: 39 });
  assert.equal(await evaluate("document.querySelector('.wd14-slider output').textContent"), '0.36');
  check('keyboard range and focus', await evaluate("({value:document.querySelector('.wd14-slider input').value,focus:getComputedStyle(document.activeElement).outlineStyle})"));
  await fill('.wd14-slider input', '0.35');
  const fixtures = [];
  for (const [index, dimensions] of [[0, [64, 96]], [1, [96, 64]]]) {
    const data = await evaluate(`(() => {const c=document.createElement('canvas');c.width=${dimensions[0]};c.height=${dimensions[1]};const x=c.getContext('2d');x.fillStyle='${index ? '#326d70' : '#994840'}';x.fillRect(0,0,c.width,c.height);x.fillStyle='#eeeecc';x.fillRect(15,15,24,30);return c.toDataURL('image/png').split(',')[1]})()`);
    const filename = path.join(evidence, `synthetic-${index + 1}.png`); await writeFile(filename, Buffer.from(data, 'base64')); fixtures.push(filename);
  }
  const document = await send('DOM.getDocument');
  const input = await send('DOM.querySelector', { nodeId: document.root.nodeId, selector: '.wd14-page input[type=file]' });
  await send('DOM.setFileInputFiles', { nodeId: input.nodeId, files: fixtures });
  await wait(() => evaluate("document.querySelector('.wd14-queue')?.textContent.includes('2 张') && document.querySelector('.wd14-drop img')?.naturalWidth > 0"));
  check('selected image contains original proportions', await evaluate("({fit:getComputedStyle(document.querySelector('.wd14-drop img')).objectFit,natural:[document.querySelector('.wd14-drop img').naturalWidth,document.querySelector('.wd14-drop img').naturalHeight]})"));
  await clickText('.wd14-work button', '开始反推');
  await wait(() => evaluate("document.querySelector('.wd14-work-foot')?.textContent.includes('完成 2 张')"), 180000);
  assert.equal(report.realTagResponses.length, 2); assert.ok(report.realTagResponses.every(x => x.status === 200));
  assert.equal(await evaluate("document.querySelectorAll('.wd14-ratings span').length"), 4);
  const result = await evaluate("({ratings:[...document.querySelectorAll('.wd14-ratings span')].map(x=>x.textContent),text:document.querySelector('.wd14-editor textarea').value,tabs:[...document.querySelectorAll('.wd14-tabs button')].map(x=>x.textContent)})");
  check('REAL browser-to-installed CPU ONNX, two images in order and four ratings', result);
  const edited = result.text + ', sol_ui_edit';
  await fill('.wd14-editor textarea', edited);
  await evaluate("document.querySelectorAll('.wd14-tabs button')[1].click()");
  await wait(() => evaluate("document.querySelector('.wd14-drop img').naturalWidth === 96"));
  await evaluate("document.querySelectorAll('.wd14-tabs button')[0].click()");
  assert.equal(await evaluate("document.querySelector('.wd14-editor textarea').value"), edited);
  await clickText('.wd14-results button', '复制');
  await wait(() => evaluate("document.querySelector('.wd14-work-foot').textContent.includes('已复制标签')"));
  assert.equal(await evaluate('navigator.clipboard.readText()'), edited);
  await clickText('.wd14-results button', '导出 TXT');
  await wait(async () => (await readdir(path.join(evidence, 'exports'))).includes('synthetic-1.txt'));
  assert.equal(await readFile(path.join(evidence, 'exports', 'synthetic-1.txt'), 'utf8'), edited);
  await fill('.wd14-search input', 'sol_no_match_');
  assert.equal(await evaluate("document.querySelectorAll('.wd14-chips > span').length"), 0);
  await fill('.wd14-search input', '');
  check('edit survives image switching; actual clipboard + TXT export; tag search');
  await screenshot('desktop-1440-real-result');
  await viewport(1920, 1080); await screenshot('desktop-1920-real-result');
  for (const width of [320, 375, 390, 414, 768]) {
    await viewport(width, 844);
    await evaluate("document.querySelector('.wd14-page').scrollTop=0");
    const geometry = await evaluate("({width:innerWidth,root:document.documentElement.scrollWidth,page:document.querySelector('.wd14-page').clientWidth,scroll:document.querySelector('.wd14-page').scrollWidth,workY:document.querySelector('.wd14-work').getBoundingClientRect().y,sideY:document.querySelector('.wd14-sidebar').getBoundingClientRect().y})");
    if (geometry.root > width && width === 768) {
      const overflow = await evaluate("[...document.querySelectorAll('.topbar,.main-nav,.top-actions')].map(e=>({selector:e.className,right:e.getBoundingClientRect().right,width:e.getBoundingClientRect().width}))");
      report.warnings.push({ name: 'Inherited shared-navigation overflow at 768; outside WD14 permission scope', geometry, overflow });
      await screenshot('tablet-768-shared-nav-overflow');
    } else assert.equal(geometry.root, width);
    assert.ok(geometry.scroll <= geometry.page); assert.ok(geometry.workY < geometry.sideY);
    check(`narrow ${width} WD14 content no overflow, analysis first${geometry.root > width ? ' (shared-nav warning)' : ''}`, geometry);
    if (width === 390) {
      await screenshot('mobile-390-source');
      await evaluate("document.querySelector('.wd14-editor').scrollIntoView({block:'start'})"); await screenshot('mobile-390-result');
      await evaluate("document.querySelector('.wd14-safety summary').focus()");
      assert.equal(await evaluate("document.activeElement.matches('.wd14-safety summary')"), true);
      await send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Enter', code: 'Enter', text: '\r', unmodifiedText: '\r', windowsVirtualKeyCode: 13 });
      await send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'Enter', code: 'Enter', windowsVirtualKeyCode: 13 });
      await wait(() => evaluate("document.querySelector('.wd14-safety').open"));
      await evaluate("document.querySelector('.wd14-batch button[type=submit]').focus()");
      const reach = await evaluate("(() => {const r=document.activeElement.getBoundingClientRect();return {top:r.top,bottom:r.bottom,height:innerHeight,focus:getComputedStyle(document.activeElement).outlineStyle,copy:document.querySelector('.wd14-safety').textContent}})()");
      assert.ok(reach.top >= 0 && reach.bottom <= reach.height); assert.match(reach.copy, /无法识别映射网络盘/); check('390 keyboard disclosure and final batch action reachable', reach);
      await screenshot('mobile-390-batch');
    }
  }
  await viewport(1440);
  await evaluate("document.querySelector('.wd14-page').scrollTop=0");
  const missing = await evaluate("[...document.querySelector('.wd14-controls select').options].find(x=>x.textContent.includes('未安装')).value");
  await select('.wd14-controls select', missing);
  await clickText('.wd14-controls button', '安装所选模型');
  await wait(() => evaluate("document.querySelector('.wd14-controls .wd14-progress')?.textContent.includes('50%')"));
  await clickText('.wd14-controls button', '取消下载');
  await wait(() => evaluate("document.querySelector('.wd14-work-foot').textContent.includes('模型正在发布，不能取消')"));
  assert.equal(await evaluate("[...document.querySelectorAll('.wd14-controls button')].some(x=>x.textContent.includes('取消下载'))"), false);
  await screenshot('mock-publishing-409');
  await wait(() => evaluate("document.querySelector('.wd14-controls .wd14-progress').textContent.includes('历史备份需人工检查')"));
  check('MOCK install byte progress, racing publishing 409, noncancel state, warning visible');
  await select('.wd14-controls select', 'wd-swinv2-v3');
  const scratchPath = evidence.replaceAll('/', '\\');
  await fill('.wd14-batch form > label:first-child input', scratchPath);
  await fill('.wd14-batch form > label:nth-child(2) input', scratchPath);
  assert.equal(await evaluate("document.querySelector('.wd14-batch select').value"), 'ignore');
  await select('.wd14-batch select', 'append');
  await evaluate("document.querySelector('.wd14-batch input[type=checkbox]').click()");
  await clickText('.wd14-batch button', '开始批量');
  await wait(() => evaluate("document.querySelector('.wd14-batch .wd14-progress') !== null"));
  assert.equal(batchPayload.conflict, 'append'); assert.equal(batchPayload.recursive, true); assert.equal(batchPayload.output_folder, scratchPath); assert.equal(batchPayload.general, .35); assert.equal(batchPayload.character, .85); assert.equal(batchPayload.limit, 50);
  await clickText('.wd14-batch button', '取消任务');
  await wait(() => evaluate("document.querySelector('.wd14-batch .wd14-progress').textContent.includes('cancelled')"));
  check('MOCK batch request wiring (default ignore, explicit append, recursive, output) + progress/cancel', batchPayload);
  await screenshot('mock-batch-cancelled');
  await clickText('.wd14-queue button', '清空');
  await wait(() => evaluate("!document.querySelector('.wd14-drop img')"));
  const urlLifecycle = await evaluate('({created:__wd14URLs.created.length,revoked:__wd14URLs.revoked.length,unrevoked:__wd14URLs.created.filter(u=>!__wd14URLs.revoked.includes(u))})');
  assert.equal(urlLifecycle.unrevoked.length, 0); check('object URLs revoked on switching/clear/export', urlLifecycle);
  // Drop + clipboard admission uses browser File/DataTransfer objects; tag stays real above.
  await evaluate("(() => {const d=new DataTransfer();d.items.add(new File(['x'],'drop.png',{type:'image/png'}));document.querySelector('.wd14-drop').dispatchEvent(new DragEvent('drop',{bubbles:true,cancelable:true,dataTransfer:d}));const p=new DataTransfer();p.items.add(new File(['x'],'paste.png',{type:'image/png'}));document.querySelector('.wd14-drop').dispatchEvent(new ClipboardEvent('paste',{bubbles:true,cancelable:true,clipboardData:p}));})()");
  await wait(() => evaluate("document.querySelector('.wd14-queue')?.textContent.includes('2 张')"));
  check('drop and paste admission, browser File/DataTransfer events');
  await clickText('.wd14-queue button', '清空');
  mockTag = true;
  await send('Fetch.enable', { patterns: [{ urlPattern: '*/api/inference/wd14/active*' }, { urlPattern: '*/api/inference/wd14/batch*' }, { urlPattern: '*/api/inference/wd14/download/*' }, { urlPattern: '*/api/inference/wd14/jobs/mock-*' }, { urlPattern: '*/api/inference/wd14/tag*' }] });
  await evaluate("(() => {const d=new DataTransfer();d.items.add(new File([new Uint8Array(20*1024*1024+1)],'too-large.png',{type:'image/png'}));for(let i=1;i<31;i++)d.items.add(new File(['x'],`stop-${i}.png`,{type:'image/png'}));document.querySelector('.wd14-drop').dispatchEvent(new DragEvent('drop',{bubbles:true,cancelable:true,dataTransfer:d}));})()");
  assert.equal(await evaluate("document.querySelectorAll('.wd14-file-strip button').length"), 30);
  await clickText('.wd14-work button', '开始反推');
  await wait(() => mockTagCalls === 1);
  await clickText('.wd14-work button', '停止剩余图片');
  await wait(() => evaluate("document.querySelector('.wd14-work-foot').textContent.includes('已停止剩余图片')"));
  assert.equal(mockTagCalls, 1);
  assert.equal(await evaluate("document.querySelectorAll('.wd14-tabs button').length"), 2);
  assert.match(await evaluate("document.querySelector('.wd14-results').textContent"), /图片超过 20 MiB/);
  check('30-image cap, 20MiB failure visible, MOCK slow tag finishes current then stops remaining', { mockTagCalls });
  await screenshot('mock-stop-and-20MiB-error');
  await clickText('.wd14-queue button', '清空');
  // Unchanged sibling pages are actually opened at desktop and narrow widths.
  for (const width of [1440, 390]) {
    await viewport(width, width === 390 ? 844 : 900);
    for (const [text, selector] of [['文生图', '.workspace'], ['图生图', '.image-workspace'], ['工具箱', '.toolbox-page']]) {
      await nav(text); await wait(() => evaluate(`Boolean(document.querySelector('${selector}'))`));
      const geometry = await evaluate(`(() => {const r=document.querySelector('${selector}').getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height,root:document.documentElement.scrollWidth,viewport:innerWidth}})()`);
      assert.ok(geometry.width > 0 && geometry.height > 0); assert.ok(geometry.root <= width); check(`${text} ${width} navigation/layout non-regression`, geometry);
      await screenshot(`sibling-${text}-${width}`);
    }
  }
  recovery = true;
  await nav('图片反推');
  await wait(() => evaluate("document.querySelector('.wd14-controls .wd14-progress')?.textContent.includes('人工检查恢复')"));
  check('MOCK interrupted marker recovery warning visible after remount');
  await screenshot('mobile-390-marker-warning');
  assert.equal(report.runtimeErrors.length, 0, report.runtimeErrors.join('\n'));
  check('no uncaught browser exceptions');
} catch (error) {
  report.failure = error.stack; console.error(error); process.exitCode = 1;
} finally {
  await writeFile(path.join(evidence, 'probe.json'), JSON.stringify(report, null, 2));
  ws?.close(); browser.kill();
  await Promise.race([new Promise(resolve => browser.exitCode !== null ? resolve() : browser.once('exit', resolve)), sleep(5000)]);
  console.log('Evidence', path.join(evidence, 'probe.json'));
}
