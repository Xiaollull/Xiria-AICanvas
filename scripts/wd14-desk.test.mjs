import test from 'node:test';
import assert from 'node:assert/strict';
import { Wd14Desk, WD14_IMAGE_BYTES } from '../src/wd14-desk.js';
const file = name => ({ name, type: 'image/png', size: 100 });
const params = { model: 'wd-swinv2-v3', general: .35, character: .85, limit: 50, alpha: false };
const result = text => ({ rating: { general: .8, sensitive: .1, questionable: .07, explicit: .03 }, tags: [{ name: text, confidence: .9 }] });
const deferred = () => { let resolve, reject; const promise = new Promise((a,b) => { resolve=a; reject=b; }); return { promise, resolve, reject }; };

test('new images replace selection, successful results survive, edits restore, rerun updates one stable id', async () => {
  const desk = new Wd14Desk();
  const a=desk.add([file('a.png')]); await desk.run(a,params,async()=>result('a'));
  desk.edit(a,'edited a');
  const b=desk.add([file('b.png')]); assert.equal(desk.state.selectedId,b);
  assert.equal(desk.state.entries.find(x=>x.id===b).text,'');
  await desk.run(b,params,async()=>result('b'));
  desk.select(a); assert.equal(desk.state.entries.find(x=>x.id===a).text,'edited a');
  let observed; await desk.run(a,{...params,general:.7},async(f,p)=>{observed=[f.name,p.general];return result('new a');});
  assert.deepEqual(observed,['a.png',.7]); assert.equal(desk.state.entries.length,2);
  assert.equal(desk.state.entries[0].id,a); assert.equal(desk.state.entries[0].text,'new a');
  assert.equal(desk.state.entries[1].text,'b'); assert.equal(desk.state.entries[0].params.general,.7);
});
test('A late after B (transport ignores abort) cannot mix preview/result or leave orphan pending state', async () => {
  const desk=new Wd14Desk(), slow=deferred(); let signal;
  const a=desk.add([file('a.png')]); const first=desk.run(a,params,(f,p,s)=>{signal=s;return slow.promise;});
  const b=desk.add([file('b.png')]); assert.ok(signal.aborted);
  await desk.run(b,params,async()=>result('B'));
  slow.resolve(result('late A')); await first;
  assert.equal(desk.state.selectedId,b); assert.equal(desk.state.entries.length,1);
  assert.equal(desk.state.entries[0].text,'B'); assert.equal(desk.state.entries[0].status,'done');
});
test('selecting history during analysis never gets stolen by completion or failure', async () => {
  const desk=new Wd14Desk(); const a=desk.add([file('a.png')]); await desk.run(a,params,async()=>result('A'));desk.edit(a,'edit');
  const b=desk.add([file('b.png')]), slow=deferred(); const pending=desk.run(b,params,()=>slow.promise);
  desk.select(a); slow.resolve(result('B')); await pending;
  assert.equal(desk.state.selectedId,a); assert.equal(desk.state.entries[0].text,'edit');assert.equal(desk.state.entries[1].text,'B');
});
test('rerun failure retains latest edits/result/last-success params; attempt params are immutable snapshots', async () => {
  const desk=new Wd14Desk(), id=desk.add([file('a.png')]);await desk.run(id,params,async()=>result('A'));
  const slow=deferred(), changed={...params,general:.6}; const pending=desk.run(id,changed,()=>slow.promise);changed.general=.9;desk.edit(id,'while pending');slow.reject(new Error('failure'));await pending;
  const entry=desk.state.entries[0];assert.equal(entry.text,'while pending');assert.equal(entry.params.general,.35);assert.equal(entry.attemptParams.general,.6);assert.ok(Object.isFrozen(entry.attemptParams));assert.equal(entry.status,'error');assert.match(entry.error,/failure/);
});
test('replace while rerunning keeps the old result but stops its pending state', async () => {
  const desk=new Wd14Desk(),a=desk.add([file('a.png')]);await desk.run(a,params,async()=>result('A'));const slow=deferred();const pending=desk.run(a,params,()=>slow.promise);
  desk.add([file('b.png')]);assert.equal(desk.state.entries[0].status,'stopped');slow.resolve(result('late'));await pending;assert.equal(desk.state.entries[0].text,'A');
});
test('clear/remove/stop/dispose invalidate stale success and rejection even after reactivation', async () => {
  for(const action of ['clear','removeSelected','stop','dispose']) {
    const desk=new Wd14Desk(),id=desk.add([file('a.png')]),slow=deferred();let signal;
    const pending=desk.run(id,params,(f,p,s)=>{signal=s;return slow.promise;});desk[action]();assert.ok(signal.aborted);desk.activate();
    const next=desk.add([file('b.png')]);await desk.run(next,params,async()=>result('B'));
    slow.reject(new Error('stale'));await pending;assert.equal(desk.state.selectedId,next);assert.equal(desk.state.entries.at(-1).text,'B');assert.equal(desk.state.entries.at(-1).error,'');
  }
});
test('explicit one-file admission, 20MiB bound, cache full rejection is lossless, removal releases capacity', async () => {
  const desk=new Wd14Desk();assert.equal(desk.add([file('a.png'),file('b.png')]),null);assert.match(desk.state.message,/每次仅添加一张/);
  assert.equal(desk.add([{...file('big.png'),size:WD14_IMAGE_BYTES+1}]),null);
  for(let i=0;i<30;i++){const id=desk.add([file(`${i}.png`)]);await desk.run(id,params,async()=>result(String(i)));}
  const selected=desk.state.selectedId;assert.equal(desk.add([file('31.png')]),null);assert.equal(desk.state.selectedId,selected);assert.equal(desk.state.entries.length,30);assert.match(desk.state.message,/30 项/);
  desk.removeSelected();assert.equal(desk.state.entries.length,29);assert.ok(desk.add([file('replacement.png')]));assert.equal(desk.state.entries.length,30);
});
