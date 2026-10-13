import SizeGrid from "./SizeGrid.jsx";
import { effectiveCanvasMaxEdge } from "./canvas-limit.js";
import { patchXYZParameters } from "./xyz-sidebar-context.js";
import { XYZNumberInput } from "./SamplingControls.jsx";

const presets = [{ label: "512", w: 512, h: 512 }, { label: "1:1", w: 1024, h: 1024 }, { label: "4:3", w: 1152, h: 896 }, { label: "3:4", w: 896, h: 1152 }, { label: "16:9", w: 1344, h: 768 }, { label: "9:16", w: 768, h: 1344 }];
export default function XYZCanvasPanel({ draft, onChange, canvasMaxEdge }) {
  const p = draft.baseline.parameters, max = effectiveCanvasMaxEdge(canvasMaxEdge, draft.engine);
  const update = (patch) => onChange(patchXYZParameters(draft, patch));
  return <section className="xyz-canvas-panel xyz-mother-controls" aria-label="XYZ 原始生成画布">
    <div className="section-heading"><span>画布</span><h2>原始单图尺寸</h2></div>
    <div className="dimension"><XYZNumberInput ariaLabel="XYZ 生成宽度" value={p.width} min={64} max={max} integer onCommit={(width) => update({ width })} /><i>×</i><XYZNumberInput ariaLabel="XYZ 生成高度" value={p.height} min={64} max={max} integer onCommit={(height) => update({ height })} /><b>PX</b></div>
    <SizeGrid width={p.width} height={p.height} max={max} onChange={(width, height) => update({ width, height })} />
    <div className="size-presets">{presets.map((item) => <button type="button" key={item.label} disabled={item.w > max || item.h > max} className={p.width === item.w && p.height === item.h ? "active" : ""} onClick={() => update({ width: item.w, height: item.h })}>{item.label}</button>)}</div>
    <small className="xyz-canvas-note">全部格子共享此宽高；{draft.inherit_postprocess ? "后处理可改变最终尺寸。" : "后处理关闭。"}</small>
    {(p.width > max || p.height > max) && <p className="xyz-warning" role="alert">保存画布超过当前 {max}px 上限；保留原值，请明确修改。</p>}
  </section>;
}
