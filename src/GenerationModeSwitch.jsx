import { useRef } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import WorkspaceSelect from "./WorkspaceSelect.jsx";
import { GENERATION_MODES, stepGenerationMode } from "./generation-modes.js";

/** One framed control: borderless step arrows on both sides of a symmetric dropdown (a chevron each side), whose menu
    spans the whole frame, plus a dot per mode. */
export default function GenerationModeSwitch({ value, onChange, disabled }) {
  const index = GENERATION_MODES.findIndex((mode) => mode.id === value), frame = useRef(null);
  return <div className="generation-mode-field">
    <span>生成模式</span>
    <div ref={frame} className={`generation-mode-switch${disabled ? " disabled" : ""}`} role="group" aria-label="生成模式">
      <button type="button" className="generation-mode-step" aria-label="上一个生成模式" title="上一个生成模式" disabled={disabled} onClick={() => onChange(stepGenerationMode(value, -1))}><ChevronLeft size={16} /></button>
      <WorkspaceSelect ariaLabel="生成模式" value={value} disabled={disabled} onChange={onChange} anchorRef={frame} mirrorChevron options={GENERATION_MODES.map((mode) => ({ value: mode.id, label: mode.label }))} />
      <button type="button" className="generation-mode-step" aria-label="下一个生成模式" title="下一个生成模式" disabled={disabled} onClick={() => onChange(stepGenerationMode(value, 1))}><ChevronRight size={16} /></button>
      <span className="generation-mode-dots" aria-hidden="true">{GENERATION_MODES.map((mode, i) => <i key={mode.id} className={i === index ? "active" : ""} />)}</span>
    </div>
  </div>;
}
