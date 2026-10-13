import { useRef, useState } from 'react';
import XYZLoraManager from './XYZLoraManager.jsx';
import { xyzAxisLoraMarks } from './xyz-manager-context.js';

/** Per-control local transaction using the exact mother manager, never baseline/group writers. */
export default function XYZAxisLoraPicker({ draft, manager, members = [], onSelect, label, axisKey = "", multiple = false, text }) {
 const [local,setLocal] = useState(null), [error,setError] = useState('');
 const current=useRef({draft,onSelect});current.current={draft,onSelect};
 const token=useRef('');
 const open=()=>{
  token.current=JSON.stringify([draft.engine,draft.axes]);
  setError('');setLocal({...structuredClone(draft),axes:{x:null,y:null,z:null},baseline:{...structuredClone(draft.baseline),loras:structuredClone(members),group_snapshots:[],prompt_without_groups:'',group_prompt:'',prompt_source:undefined,parameters:{...structuredClone(draft.baseline.parameters),prompt:'',negative_prompt:''}}});
 };
 const marks = xyzAxisLoraMarks(draft, axisKey);
 return <><button type="button" className="xyz-lora-picker" aria-label={label} onClick={open}>{text ?? (members.length ? `${members.map(m=>m.name||m.value).join(' · ')} · 管理选择` : '打开 LoRA 管理器 · 待选择')}</button>
 {error && <small role="alert">{error}</small>}
 {local && manager && <XYZLoraManager draft={local} onChange={setLocal} {...manager} transaction single={!multiple} axisMarks={marks} title={label} onClose={()=>setLocal(null)} onConfirm={next=>{
  if(token.current!==JSON.stringify([current.current.draft.engine,current.current.draft.axes])){setError('轴或引擎已变更；临时选择未写入，请重新打开。');setLocal(null);return;}
  if(!multiple && next.baseline.loras.length!==1){setError('此控件须确认恰好一个 LoRA；请调整临时集合。');return;}
  current.current.onSelect(structuredClone(next.baseline.loras),next.baseline.group_prompt);setLocal(null);
 }} />}
 </>;
}
