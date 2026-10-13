import test from 'node:test';
import assert from 'node:assert/strict';
import { createCatalogRequest, catalogReadIntent } from '../src/tagger-catalog-request.js';
const deferred=()=>{let resolve,reject;const promise=new Promise((r,j)=>{resolve=r;reject=j;});return {resolve,reject,promise};};
test('Strict mount and concurrent poll share one actual request; settled reads do not persist status',async()=>{
  const first=deferred();let calls=0;const get=createCatalogRequest(()=>{calls++;return calls===1?first.promise:Promise.resolve({current:calls});});
  const a=get(),b=get();assert.equal(a,b);await Promise.resolve();assert.equal(calls,1);first.resolve({models:19});await Promise.all([a,b]);
  await get();assert.equal(calls,2);
});
test('explicit refresh behind ordinary flight queues only one force read, errors allow retry',async()=>{
  const first=deferred(),second=deferred();const calls=[];const get=createCatalogRequest(force=>{calls.push(force);return calls.length===1?first.promise:second.promise;});
  const plain=get(),a=get(true),b=get(true);await Promise.resolve();assert.deepEqual(calls,[false]);first.resolve({old:1});await plain;
  await new Promise(r=>setTimeout(r,0));assert.deepEqual(calls,[false,true]);second.resolve({new:1});await Promise.all([a,b]);
  let attempt=0;const retry=createCatalogRequest(()=>++attempt===1?Promise.reject(Error('once')):Promise.resolve('ok'));
  await assert.rejects(retry(),/once/);assert.equal(await retry(),'ok');
});
test('LAN manual and completion refresh remain read-only; local explicit refresh uses Origin-bearing POST',()=>{
  assert.deepEqual(catalogReadIntent(true,true),{path:'/models?refresh=true',options:{method:'POST'}});
  for(const force of [true,false])assert.deepEqual(catalogReadIntent(force,false),{path:'/models',options:undefined});
  assert.deepEqual(catalogReadIntent(false,true),{path:'/models',options:undefined});
});
