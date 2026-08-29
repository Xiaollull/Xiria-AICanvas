import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { access, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import test from "node:test";
import { promisify } from "node:util";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  acceptsLoraDragOver,
  appLoraDragLocked,
  establishLoraDragSession,
  isLoraDragHandleSource,
  isValidLoraDragSession,
  isValidLoraDropTarget,
  LORA_SORT_TRANSFER_TYPE,
  pageLoraDragLocked,
  reorderLoraItems,
  shouldCommitLoraDrop,
  suppressLoraDragHandleKeyboard,
} from "../src/lora-drag-handle.js";

const execFileAsync = promisify(execFile);
const projectRoot = resolve(fileURLToPath(new URL("..", import.meta.url)));
const chromeCandidates = [
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
];

function dragEvent({ currentTarget, target = currentTarget, dataTransfer = null } = {}) {
  let prevented = false;
  return {
    currentTarget,
    target,
    dataTransfer,
    preventDefault: () => { prevented = true; },
    get defaultPrevented() { return prevented; },
  };
}

function transfer({ hideReads = false, throwOnRead = false } = {}) {
  const data = new Map();
  return {
    effectAllowed: "none",
    data,
    get types() { return [...data.keys()]; },
    setData(type, value) { data.set(type, String(value)); },
    getData(type) {
      if (throwOnRead) throw new Error("drag data is private until drop");
      return hideReads ? "" : data.get(type) || "";
    },
  };
}

function handleNode() {
  const handle = { dataset: { loraDragHandle: "true" } };
  const svg = { closest: (selector) => selector === "[data-lora-drag-handle]" ? handle : null };
  const path = { closest: (selector) => selector === "[data-lora-drag-handle]" ? handle : null };
  const outside = { closest: () => null };
  handle.contains = (node) => node === svg || node === path;
  return { handle, svg, path, outside };
}

function transferWithPrivateValue(value) {
  const next = transfer();
  next.setData(LORA_SORT_TRANSFER_TYPE, value);
  return next;
}

test("LoRA drag permission is fail-closed for every App and dedicated-page lock", () => {
  for (const status of ["idle", "running"]) {
    for (const modelSwitching of [false, true]) {
      for (const loraWorkspaceLocked of [false, true]) {
        for (const shouldPersistMountedLoras of [false, true]) {
          const state = { status, modelSwitching, loraWorkspaceLocked, shouldPersistMountedLoras };
          const expected = status === "running" || modelSwitching || loraWorkspaceLocked || !shouldPersistMountedLoras;
          assert.equal(appLoraDragLocked(state), expected, `App lock permutation ${JSON.stringify(state)}`);
        }
      }
    }
  }

  for (const workspaceLocked of [false, true]) {
    for (const syncReady of [false, true]) {
      for (const canPersist of [false, true]) {
        for (const scopeKey of [null, "SD"]) {
          const state = { workspaceLocked, syncReady, canPersist, scopeKey };
          const expected = workspaceLocked || !syncReady || !canPersist || !scopeKey;
          assert.equal(pageLoraDragLocked(state), expected, `page lock permutation ${JSON.stringify(state)}`);
        }
      }
    }
  }
});

test("the marked six-dot handle and its SVG/path descendants alone establish a native transfer session", () => {
  const items = [{ value: "first" }, { value: "second" }];
  const { handle, svg, path, outside } = handleNode();
  for (const target of [handle, svg, path]) {
    assert.equal(isLoraDragHandleSource(dragEvent({ currentTarget: handle, target })), true, "button, SVG and path are all within the marked handle");
  }
  const validTransfer = transfer();
  const valid = dragEvent({ currentTarget: handle, target: path, dataTransfer: validTransfer });
  const session = establishLoraDragSession({ event: valid, index: 0, sourceValue: "first", items, locked: false });
  assert.ok(session);
  assert.deepEqual(Object.keys(session).sort(), ["index", "nonce", "sourceValue", "transferValue"]);
  assert.equal(session.index, 0);
  assert.equal(session.sourceValue, "first");
  assert.match(session.nonce, /^[0-9a-f]{36}$/);
  assert.equal(validTransfer.effectAllowed, "move");
  assert.equal(validTransfer.data.get(LORA_SORT_TRANSFER_TYPE), session.transferValue);
  assert.deepEqual(JSON.parse(session.transferValue), { index: 0, sourceValue: "first", nonce: session.nonce });
  assert.equal(validTransfer.data.get("text/plain"), "first");

  // These cover row/card surfaces and every real control shape. They cannot
  // substitute another marked ancestor, and the exact visible SVG/path stays legal.
  for (const target of [outside, {}, { dataset: {} }]) {
    const event = dragEvent({ currentTarget: handle, target, dataTransfer: transfer() });
    assert.equal(isLoraDragHandleSource(event), false);
    assert.equal(establishLoraDragSession({ event, index: 0, sourceValue: "first", items, locked: false }), null);
  }
  for (const currentTarget of [{}, { dataset: {} }]) {
    assert.equal(establishLoraDragSession({ event: dragEvent({ currentTarget, dataTransfer: transfer() }), index: 0, sourceValue: "first", items, locked: false }), null);
  }
  assert.equal(establishLoraDragSession({ event: valid, index: 0, sourceValue: "first", items, locked: true }), null);
});

test("native LoRA drop admission requires a live session and exact private transfer identity", () => {
  const items = [{ value: "first" }, { value: "second" }, { value: "third" }];
  const { handle, path } = handleNode();
  const nativeTransfer = transfer();
  const session = establishLoraDragSession({ event: dragEvent({ currentTarget: handle, target: path, dataTransfer: nativeTransfer }), index: 0, sourceValue: "first", items, locked: false });
  const position = { session, items, targetIndex: 1, locked: false };
  assert.equal(isValidLoraDragSession(position), true);
  assert.equal(acceptsLoraDragOver({ ...position, dataTransfer: nativeTransfer }), true);
  assert.equal(isValidLoraDropTarget({ ...position, dataTransfer: nativeTransfer }), true);
  assert.equal(shouldCommitLoraDrop({ ...position, dataTransfer: nativeTransfer }), true);
  assert.deepEqual(reorderLoraItems(items, session.index, 1).map((item) => item.value), ["second", "first", "third"]);

  const noTransfer = transfer();
  const wrongMime = transfer();
  wrongMime.setData("application/x-external-drag", session.transferValue);
  const parsed = JSON.parse(session.transferValue);
  const wrongToken = transferWithPrivateValue(JSON.stringify({ ...parsed, nonce: "0".repeat(36) }));
  const wrongIndex = transferWithPrivateValue(JSON.stringify({ ...parsed, index: 1 }));
  const wrongSource = transferWithPrivateValue(JSON.stringify({ ...parsed, sourceValue: "second" }));
  for (const [label, dataTransfer] of [["missing private transfer", noTransfer], ["wrong MIME", wrongMime], ["wrong token", wrongToken], ["wrong source index", wrongIndex], ["wrong source value", wrongSource]]) {
    assert.equal(acceptsLoraDragOver({ ...position, dataTransfer }), false, label);
    assert.equal(isValidLoraDropTarget({ ...position, dataTransfer }), false, label);
    assert.equal(shouldCommitLoraDrop({ ...position, dataTransfer }), false, label);
  }
  assert.equal(isValidLoraDropTarget({ ...position, session: null, dataTransfer: nativeTransfer }), false, "external transfer cannot borrow a session");
  assert.equal(isValidLoraDropTarget({ ...position, locked: true, dataTransfer: nativeTransfer }), false, "lock invalidates a previously valid native transfer");

  const replacementTransfer = transfer();
  const replacement = establishLoraDragSession({ event: dragEvent({ currentTarget: handle, target: path, dataTransfer: replacementTransfer }), index: 0, sourceValue: "first", items, locked: false });
  assert.notEqual(replacement.nonce, session.nonce);
  assert.equal(isValidLoraDropTarget({ ...position, session: replacement, dataTransfer: nativeTransfer }), false, "old transfer is stale after a new session");

  // Chromium and other browsers can hide getData during dragover. Visible MIME
  // plus the current session may paint feedback, but a hidden value never passes drop.
  const privateRead = transfer({ hideReads: true });
  privateRead.setData(LORA_SORT_TRANSFER_TYPE, replacement.transferValue);
  assert.equal(acceptsLoraDragOver({ ...position, session: replacement, dataTransfer: privateRead }), true);
  assert.equal(isValidLoraDropTarget({ ...position, session: replacement, dataTransfer: privateRead }), false);
  const throwingRead = transfer({ throwOnRead: true });
  throwingRead.setData(LORA_SORT_TRANSFER_TYPE, replacement.transferValue);
  assert.equal(acceptsLoraDragOver({ ...position, session: replacement, dataTransfer: throwingRead }), true);
  assert.equal(isValidLoraDropTarget({ ...position, session: replacement, dataTransfer: throwingRead }), false);
});

test("handle activation keys never establish or commit a reorder", () => {
  for (const key of ["Enter", " "]) {
    let prevented = false;
    assert.equal(suppressLoraDragHandleKeyboard({ key, preventDefault: () => { prevented = true; } }), true);
    assert.equal(prevented, true);
  }
  assert.equal(suppressLoraDragHandleKeyboard({ key: "ArrowDown", preventDefault: () => assert.fail("unexpected prevent") }), false);
});

test("one shared panel wires native source, exact-transfer drop and a11y through drag handles", async () => {
  const [app, page, panel, styles, helper] = await Promise.all([
    readFile(join(projectRoot, "src", "App.jsx"), "utf8"),
    readFile(join(projectRoot, "src", "LoraManagerPage.jsx"), "utf8"),
    readFile(join(projectRoot, "src", "LoraMountPanel.jsx"), "utf8"),
    readFile(join(projectRoot, "src", "styles.css"), "utf8"),
    readFile(join(projectRoot, "src", "lora-drag-handle.js"), "utf8"),
  ]);
  assert.match(helper, /application\/x-xiria-lora-sort/);
  assert.match(helper, /target\.closest\?\.\("\[data-lora-drag-handle\]"\)/);
  assert.match(helper, /GripVertical SVG or its inner path/);
  assert.match(helper, /Drop is the commit boundary/);
  assert.match(helper, /type is visible but its value is unreadable/);

  // The gesture lives in exactly one place now. Both mounted queues used to
  // carry their own copy, which is how a fix to one silently left the other
  // behind; a second implementation reappearing must fail this test.
  for (const [name, source] of [["App.jsx", app], ["LoraManagerPage.jsx", page]]) {
    assert.match(source, /<LoraMountPanel\b/, `${name} must render the shared panel`);
    assert.ok(!source.includes("data-lora-drag-handle"), `${name} must not re-implement the drag handle`);
    assert.ok(!source.includes("establishLoraDragSession"), `${name} must not own a drag session`);
    assert.ok(!source.includes('type="range"') || name !== "LoraManagerPage.jsx", `${name} must not re-implement the weight slider`);
  }

  assert.match(panel, /data-lora-drag-handle="true"/);
  assert.match(panel, /draggable=\{!locked\}/);
  for (const symbol of ["establishLoraDragSession", "acceptsLoraDragOver", "isValidLoraDropTarget", "shouldCommitLoraDrop", "suppressLoraDragHandleKeyboard"]) {
    assert.ok(panel.includes(symbol), `the panel must use ${symbol}`);
  }
  assert.match(panel, /aria-label=\{`拖动排序：\$\{/);
  assert.match(panel, /aria-describedby="lora-mount-pointer-only"/);
  assert.match(panel, /触控设备不支持原生排序/);

  // Only the handle is draggable; the row it sits in must never be.
  const rowOpening = panel.slice(panel.lastIndexOf("<", panel.indexOf("className={`lora-mounted-item")), panel.indexOf("onDragOver"));
  assert.ok(rowOpening, "the mounted-row parent should be identifiable");
  assert.doesNotMatch(rowOpening, /\bdraggable\s*=/);
  assert.doesNotMatch(rowOpening, /\bonDragStart\s*=/);
  assert.match(panel, /isValidLoraDropTarget\(\{[\s\S]*dataTransfer: event\.dataTransfer/);
  assert.match(panel, /clearSession\(\);\s*return;/);

  // A lock arriving mid-gesture still ends the drag, now inside the panel
  // rather than in each host's sync handler.
  assert.match(panel, /if \(locked\) clearSession\(\);/);
  assert.match(app, /loraDragGateRef\.current = \{[\s\S]*loraWorkspaceLocked: locked/);
  assert.match(page, /const currentLoraDragLocked = \(\) => pageLoraDragLocked/);
  assert.match(styles, /\.mounted-drag-handle \{[^}]*width: 32px;[^}]*height: 32px;[^}]*cursor: grab/);
  assert.match(styles, /\.mounted-drag-handle:disabled \{[^}]*cursor: not-allowed/);
  assert.match(styles, /\.mounted-drag-handle svg \{ pointer-events: none;/);
  assert.match(styles, /@media \(pointer: coarse\) \{ \.mounted-drag-handle \{ cursor: default;/);
});

async function chromePath() {
  for (const candidate of chromeCandidates) {
    try {
      await access(candidate);
      return candidate;
    } catch {}
  }
  return null;
}

function browserFixture(helperHref) {
  return `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>
  body{margin:0;font:14px sans-serif}.fixture{max-width:760px;margin:12px auto}.row{display:grid;grid-template-columns:32px 36px minmax(80px,1fr) minmax(120px,1.5fr) 72px 30px;gap:8px;align-items:center;min-height:64px;margin:7px;padding:10px;border:1px solid #555}.row.drop-target{border-color:#0b0}.row.dragging{opacity:.4}.handle{width:32px;height:32px;cursor:grab}.blank{height:26px}.details{grid-column:1/-1}.row input[type=range]{width:100%}@media(max-width:760px){.row{grid-template-columns:32px 34px minmax(80px,1fr) 60px 28px}.row input[type=range]{grid-column:1/-1}} </style></head><body><main class="fixture"><div id="rows"></div></main><script type="module">
  import { acceptsLoraDragOver, establishLoraDragSession, isValidLoraDragSession, isValidLoraDropTarget, LORA_SORT_TRANSFER_TYPE, reorderLoraItems, shouldCommitLoraDrop } from "${helperHref}";
  let items=[{value:"first"},{value:"second"},{value:"third"}],session=null,locked=false,results={nonHandle:[],invalid:{},valid:false,lockedDrop:false,overflow:false,overlap:false};
  const rows=document.querySelector('#rows'),order=()=>items.map(item=>item.value).join(','),clear=()=>{session=null;for(const row of rows.children)row.classList.remove('dragging','drop-target');};
  const live=(targetIndex)=>isValidLoraDragSession({session,items,targetIndex,locked});
  const render=()=>{rows.replaceChildren(...items.map((item,index)=>{const row=document.createElement('article');row.className='row';row.draggable=false;row.dataset.index=index;const handle=document.createElement('button');handle.type='button';handle.className='handle';handle.dataset.loraDragHandle='true';handle.draggable=!locked;handle.setAttribute('aria-label','拖动排序：'+item.value);const svg=document.createElementNS('http://www.w3.org/2000/svg','svg'),path=document.createElementNS('http://www.w3.org/2000/svg','path');path.setAttribute('d','M2 2h2v2H2zM8 2h2v2H8zM2 8h2v2H2zM8 8h2v2H8zM2 14h2v2H2zM8 14h2v2H8z');svg.append(path);handle.append(svg);handle.addEventListener('dragstart',event=>{const next=establishLoraDragSession({event,index,sourceValue:item.value,items,locked});if(!next){event.preventDefault();clear();return;}session=next;row.classList.add('dragging');});handle.addEventListener('dragend',clear);const toggle=document.createElement('button');toggle.className='toggle';toggle.textContent='switch';const name=document.createElement('strong');name.className='name';name.textContent=item.value;const range=document.createElement('input');range.className='range';range.type='range';const number=document.createElement('input');number.className='number';number.type='number';const remove=document.createElement('button');remove.className='remove';remove.textContent='remove';const blank=document.createElement('div');blank.className='blank';const details=document.createElement('button');details.className='details';details.textContent='details';row.append(handle,toggle,name,range,number,remove,blank,details);row.addEventListener('dragover',event=>{if(!acceptsLoraDragOver({session,items,targetIndex:index,locked,dataTransfer:event.dataTransfer})){clear();return;}event.preventDefault();row.classList.add('drop-target');});row.addEventListener('drop',event=>{const current=session;if(!isValidLoraDropTarget({session:current,items,targetIndex:index,locked,dataTransfer:event.dataTransfer})){clear();return;}event.preventDefault();if(shouldCommitLoraDrop({session:current,items,targetIndex:index,locked,dataTransfer:event.dataTransfer}))items=reorderLoraItems(items,current.index,index);render();clear();});return row;}));};
  const fire=(node,type,dataTransfer)=>{const event=new DragEvent(type,{bubbles:true,cancelable:true,dataTransfer});node.dispatchEvent(event);return event};
  const start=()=>{const dataTransfer=new DataTransfer();fire(rows.children[0].querySelector('path'),'dragstart',dataTransfer);return dataTransfer};
  const target=()=>rows.children[1];
  const invalid=(label,create)=>{render();locked=false;const nativeTransfer=start(),candidate=create(nativeTransfer);const before=order(),over=fire(target(),'dragover',candidate),feedback=target().classList.contains('drop-target'),drop=fire(target(),'drop',candidate);results.invalid[label]={overPrevented:over.defaultPrevented,dropPrevented:drop.defaultPrevented,feedback,before,after:order()};};
  render();const first=rows.children[0],nonHandleTarget=target();for(const node of [first,first.querySelector('.blank'),first.querySelector('.name'),first.querySelector('.toggle'),first.querySelector('.range'),first.querySelector('.number'),first.querySelector('.remove'),first.querySelector('.details')]){node.dispatchEvent(new PointerEvent('pointerdown',{bubbles:true}));const event=fire(node,'dragstart',new DataTransfer()),over=fire(nonHandleTarget,'dragover',new DataTransfer());results.nonHandle.push({order:order(),prevented:event.defaultPrevented,overPrevented:over.defaultPrevented,target:nonHandleTarget.classList.contains('drop-target')});}
  invalid('none',()=>new DataTransfer());
  invalid('wrongMime',nativeTransfer=>{const next=new DataTransfer();next.setData('application/x-external-drag',nativeTransfer.getData(LORA_SORT_TRANSFER_TYPE));return next;});
  invalid('wrongToken',nativeTransfer=>{const value=JSON.parse(nativeTransfer.getData(LORA_SORT_TRANSFER_TYPE));const next=new DataTransfer();next.setData(LORA_SORT_TRANSFER_TYPE,JSON.stringify({...value,nonce:'0'.repeat(36)}));return next;});
  invalid('wrongIndex',nativeTransfer=>{const value=JSON.parse(nativeTransfer.getData(LORA_SORT_TRANSFER_TYPE));const next=new DataTransfer();next.setData(LORA_SORT_TRANSFER_TYPE,JSON.stringify({...value,index:1}));return next;});
  invalid('wrongSource',nativeTransfer=>{const value=JSON.parse(nativeTransfer.getData(LORA_SORT_TRANSFER_TYPE));const next=new DataTransfer();next.setData(LORA_SORT_TRANSFER_TYPE,JSON.stringify({...value,sourceValue:'second'}));return next;});
  render();clear();const external=new DataTransfer();external.setData(LORA_SORT_TRANSFER_TYPE,JSON.stringify({index:0,sourceValue:'first',nonce:'a'.repeat(36)}));const externalBefore=order(),externalOver=fire(target(),'dragover',external),externalFeedback=target().classList.contains('drop-target'),externalDrop=fire(target(),'drop',external);results.invalid.external={overPrevented:externalOver.defaultPrevented,dropPrevented:externalDrop.defaultPrevented,feedback:externalFeedback,before:externalBefore,after:order()};
  render();locked=false;const oldTransfer=start(),newTransfer=start(),staleBefore=order(),staleOver=fire(target(),'dragover',oldTransfer),staleFeedback=target().classList.contains('drop-target'),staleDrop=fire(target(),'drop',oldTransfer);results.invalid.stale={overPrevented:staleOver.defaultPrevented,dropPrevented:staleDrop.defaultPrevented,feedback:staleFeedback,before:staleBefore,after:order(),different:oldTransfer.getData(LORA_SORT_TRANSFER_TYPE)!==newTransfer.getData(LORA_SORT_TRANSFER_TYPE)};
  render();locked=false;const validTransfer=start(),validOver=fire(target(),'dragover',validTransfer);results.valid=validOver.defaultPrevented&&target().classList.contains('drop-target');fire(target(),'drop',validTransfer);results.reordered=order();
  render();locked=false;const lockedTransfer=start();locked=true;const lockedBefore=order(),lockedOver=fire(target(),'dragover',lockedTransfer),lockedFeedback=target().classList.contains('drop-target'),lockedDrop=fire(target(),'drop',lockedTransfer);results.lockedDrop={overPrevented:lockedOver.defaultPrevented,dropPrevented:lockedDrop.defaultPrevented,feedback:lockedFeedback,before:lockedBefore,after:order()};const rects=[...rows.children].map(row=>row.getBoundingClientRect());results.overlap=rects.some((first,index)=>rects.slice(index+1).some(second=>first.left<second.right&&first.right>second.left&&first.top<second.bottom&&first.bottom>second.top));results.overflow=document.documentElement.scrollWidth>window.innerWidth;document.documentElement.dataset.result=JSON.stringify(results);
  </script></body></html>`;
}

test("temporary Chromium fixture admits only nested six-dot path transfers and rejects every mismatched drop at 1100/800/600", async (context) => {
  const chrome = await chromePath();
  if (!chrome) {
    context.skip("Chromium is unavailable; pure drag/session and source contracts were checked");
    return;
  }
  const folder = await mkdtemp(join(tmpdir(), "xirai-lora-drag-"));
  try {
    const fixturePath = join(folder, "fixture.html");
    await writeFile(fixturePath, browserFixture(pathToFileURL(join(projectRoot, "src", "lora-drag-handle.js")).href), "utf8");
    for (const width of [1100, 800, 600]) {
      const { stdout } = await execFileAsync(chrome, ["--headless=new", "--disable-gpu", "--allow-file-access-from-files", "--dump-dom", `--window-size=${width},820`, pathToFileURL(fixturePath).href], { windowsHide: true, timeout: 30000, maxBuffer: 2 * 1024 * 1024 });
      const match = stdout.match(/data-result="([^"]+)"/);
      assert.ok(match, `fixture result missing at ${width}`);
      const result = JSON.parse(match[1].replace(/&quot;/g, '"'));
      assert.equal(result.nonHandle.length, 8);
      for (const attempt of result.nonHandle) assert.deepEqual(attempt, { order: "first,second,third", prevented: false, overPrevented: false, target: false }, `non-handle drag was admitted at ${width}`);
      for (const [label, attempt] of Object.entries(result.invalid)) {
        assert.deepEqual({ overPrevented: attempt.overPrevented, dropPrevented: attempt.dropPrevented, feedback: attempt.feedback, before: attempt.before, after: attempt.after }, { overPrevented: false, dropPrevented: false, feedback: false, before: "first,second,third", after: "first,second,third" }, `${label} transfer was admitted at ${width}`);
      }
      assert.equal(result.invalid.stale.different, true, `fixture did not create a stale transfer at ${width}`);
      assert.equal(result.valid, true, `nested SVG/path handle drag target was not admitted at ${width}`);
      assert.equal(result.reordered, "second,first,third", `valid handle transfer did not reorder at ${width}`);
      assert.deepEqual(result.lockedDrop, { overPrevented: false, dropPrevented: false, feedback: false, before: "second,first,third", after: "second,first,third" }, `lock transition permitted a stale drop at ${width}`);
      assert.equal(result.overlap, false, `mounted rows overlapped at ${width}`);
      assert.equal(result.overflow, false, `mounted-row fixture overflowed at ${width}`);
    }
  } finally {
    await rm(folder, { recursive: true, force: true });
  }
});
