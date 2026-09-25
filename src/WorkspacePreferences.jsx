import { useEffect, useRef, useState } from "react";
import { AlertTriangle, FolderOpen, RefreshCw, RotateCcw, Scan } from "lucide-react";
import {
  CANVAS_BOUND_PRESETS,
  CANVAS_BOUND_STEP,
  DEFAULT_CANVAS_MAX_EDGE,
  HARD_CANVAS_MAX_EDGE,
  MIN_CANVAS_BOUND,
  canvasBoundConflicts,
  snapCanvasMaxEdge,
} from "./canvas-limit.js";
import { DEFAULT_OUTPUT_TEMPLATE, OUTPUT_TEMPLATE_MAX_LENGTH, OUTPUT_TEMPLATE_PLACEHOLDERS } from "./output-template.js";

// Two workspace preferences that shape every run without being a parameter of any one run: how
// large a canvas the pickers offer, and where finished pictures are written.

/**
 * The canvas bound. Raising it only widens what the pickers offer. Lowering it below a canvas that
 * is already chosen is refused with the reason, rather than cutting that canvas down unseen: the
 * user either picks a smaller canvas first or keeps the bound.
 */
export function CanvasBoundSetting({ value, canvases, onChange }) {
  const [draft, setDraft] = useState(String(value));
  const [conflict, setConflict] = useState(null);
  useEffect(() => setDraft(String(value)), [value]);
  const request = (next) => {
    const bound = snapCanvasMaxEdge(next);
    setDraft(String(bound));
    const excluded = canvasBoundConflicts(bound, canvases);
    if (excluded.length) {
      setConflict({ bound, excluded });
      return;
    }
    setConflict(null);
    if (bound !== value) onChange(bound);
  };
  return <div className="settings-option canvas-bound-setting">
    <div>
      <strong>画布可选上限</strong>
      <p>文生图与图生图画布的方形上限（{MIN_CANVAS_BOUND}～{HARD_CANVAS_MAX_EDGE}，{CANVAS_BOUND_STEP} 像素步长）；实际画布仍可自由选择宽高。超过 2048 的画布会在提交时按本机显存与内存预先检查，大尺寸只是可选刻度，不代表本机一定能生成。图生图“后处理”模式的原图 8192 边 / 32MP 限制不受此设置影响。</p>
      <div className="canvas-bound-presets" role="group" aria-label="画布上限速选">
        {CANVAS_BOUND_PRESETS.map((preset) => <button type="button" key={preset} className={value === preset ? "active" : ""} aria-pressed={value === preset} onClick={() => request(preset)}>{preset}{preset === DEFAULT_CANVAS_MAX_EDGE ? " · 默认" : ""}</button>)}
      </div>
      {conflict && <div className="canvas-bound-conflict" role="alert">
        <AlertTriangle size={13} />
        <span>当前已选画布 {conflict.excluded.map((item) => `${item.label} ${item.width} × ${item.height}`).join("、")} 超出 {conflict.bound} × {conflict.bound}。请先重新选择画布再降低上限；上限保持 {value} × {value}。</span>
        <button type="button" onClick={() => { setConflict(null); setDraft(String(value)); }}>取消修改</button>
      </div>}
    </div>
    <label className="canvas-bound-input">
      <input
        type="number"
        inputMode="numeric"
        aria-label="画布上限边长"
        min={MIN_CANVAS_BOUND}
        max={HARD_CANVAS_MAX_EDGE}
        step={CANVAS_BOUND_STEP}
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        onBlur={() => request(draft)}
        onKeyDown={(event) => { if (event.key === "Enter") { event.preventDefault(); request(draft); } }}
      />
      <span>× {draft && Number.isFinite(Number(draft)) ? snapCanvasMaxEdge(draft) : value}{conflict ? "（未应用）" : ""}</span>
    </label>
  </div>;
}

/**
 * The output path template, previewed by the server that will resolve it.
 *
 * The preview endpoint runs the same parser and directory walk a save does, without creating
 * anything, so a template this field accepts is one a job accepts. An invalid draft is never saved:
 * the last valid template keeps naming pictures until the draft is fixed or reset.
 */
export function OutputTemplateSetting({ value, engine, onChange }) {
  const [draft, setDraft] = useState(value);
  const [status, setStatus] = useState({ loading: true, root: "", valid: true, error: "", preview: "", legacy: true });
  const token = useRef(0);
  const inputRef = useRef(null);
  useEffect(() => setDraft(value), [value]);
  useEffect(() => {
    const current = ++token.current;
    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      setStatus((previous) => ({ ...previous, loading: true }));
      try {
        const query = new URLSearchParams({ engine: engine || "iL" });
        if (draft.trim() && draft.trim() !== DEFAULT_OUTPUT_TEMPLATE) query.set("template", draft);
        const response = await fetch(`/api/inference/output-template?${query}`, { cache: "no-store", signal: controller.signal });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.detail || payload.error || "推理服务暂时无法预览输出路径");
        if (current !== token.current) return;
        setStatus({ loading: false, root: payload.root || "", valid: payload.valid === true, error: payload.error || "", preview: payload.preview || "", legacy: payload.legacy === true });
        if (payload.valid === true && draft !== value) onChange(draft.trim() ? draft : DEFAULT_OUTPUT_TEMPLATE);
      } catch (error) {
        if (error.name === "AbortError" || current !== token.current) return;
        setStatus((previous) => ({ ...previous, loading: false, valid: false, error: error.message || "推理服务暂时无法预览输出路径" }));
      }
    }, 300);
    return () => { window.clearTimeout(timer); controller.abort(); };
    // `onChange` and `value` are read at settle time; the preview follows the draft and the engine.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draft, engine]);
  const insert = (name) => {
    const input = inputRef.current;
    const token = `{${name}}`;
    if (!input) { setDraft((current) => `${current}${token}`); return; }
    const start = input.selectionStart ?? draft.length;
    const end = input.selectionEnd ?? draft.length;
    const next = `${draft.slice(0, start)}${token}${draft.slice(end)}`.slice(0, OUTPUT_TEMPLATE_MAX_LENGTH);
    setDraft(next);
    window.requestAnimationFrame(() => { input.focus(); input.setSelectionRange(start + token.length, start + token.length); });
  };
  const saved = draft === value;
  return <div className="settings-option output-template-setting">
    <div>
      <strong>输出路径</strong>
      <p>生成的图片写入输出根目录下的相对路径。默认模板等同于原有的“日期文件夹 / XirAI-日期-时-分”命名，同名时追加 -02、-03；自定义模板的最后一段是文件名，例如 <code>Test/image</code> 依次写入 <code>Test/image_0001.png</code>、<code>image_0002.png</code>。文件名由服务端生成且只写 PNG，模板在提交任务时固定，生成途中修改不会影响已排队的任务。</p>
      <div className="output-template-root"><FolderOpen size={13} /><span>输出根目录</span><code title={status.root}>{status.root || (status.loading ? "正在读取…" : "未知")}</code></div>
      <label className="output-template-field">
        <input
          ref={inputRef}
          value={draft}
          maxLength={OUTPUT_TEMPLATE_MAX_LENGTH}
          spellCheck="false"
          aria-label="输出路径模板"
          aria-invalid={!status.valid && !status.loading}
          onChange={(event) => setDraft(event.target.value)}
        />
        <button type="button" title="恢复默认命名" aria-label="恢复默认输出命名" disabled={draft === DEFAULT_OUTPUT_TEMPLATE} onClick={() => setDraft(DEFAULT_OUTPUT_TEMPLATE)}><RotateCcw size={13} /></button>
      </label>
      <div className="output-template-placeholders" aria-label="可用占位符">
        {OUTPUT_TEMPLATE_PLACEHOLDERS.map((item) => <button type="button" key={item.name} title={item.label} onClick={() => insert(item.name)}><code>{`{${item.name}}`}</code><small>{item.label}</small></button>)}
      </div>
      <p className={`output-template-preview ${status.valid ? "" : "invalid"}`} role="status" aria-live="polite">
        {status.loading ? <><RefreshCw className="spin" size={12} />正在预览…</>
          : status.valid ? <><Scan size={12} />下一张图片：<code>{status.preview}</code>{status.legacy ? "（默认命名）" : ""}{saved ? "" : "（未保存）"}</>
            : <><AlertTriangle size={12} />{status.error}{saved ? "" : " · 模板未保存，仍使用上一个有效模板"}</>}
      </p>
    </div>
  </div>;
}
