import { useRef } from "react";

export default function SizeGrid({ width, height, onChange, min = 0, max = 2048, step = 64, disabled = false }) {
  const gridRef = useRef(null);
  const clamp = (value) => Math.max(min, Math.min(max, value));
  const snap = (value) => clamp(min + Math.round((value - min) / step) * step);
  const safeWidth = snap(Number(width) || min);
  const safeHeight = snap(Number(height) || min);
  const widthPercent = ((safeWidth - min) / (max - min)) * 100;
  const heightPercent = ((safeHeight - min) / (max - min)) * 100;
  const updateFromPointer = (event) => {
    const rect = gridRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
    const y = Math.max(0, Math.min(1, (rect.bottom - event.clientY) / rect.height));
    onChange(snap(min + x * (max - min)), snap(min + y * (max - min)));
  };
  const updateAxis = (axis, value) => onChange(axis === "width" ? value : safeWidth, axis === "height" ? value : safeHeight);
  const handleKeyDown = (event) => {
    if (disabled) return;
    const delta = event.shiftKey ? step * 4 : step;
    const changes = { ArrowRight: ["width", safeWidth + delta], ArrowLeft: ["width", safeWidth - delta], ArrowUp: ["height", safeHeight + delta], ArrowDown: ["height", safeHeight - delta], Home: [event.shiftKey ? "height" : "width", min], End: [event.shiftKey ? "height" : "width", max] };
    const change = changes[event.key];
    if (!change) return;
    event.preventDefault();
    updateAxis(change[0], snap(change[1]));
  };
  const handlePointerDown = (event) => {
    if (disabled) return;
    event.currentTarget.setPointerCapture?.(event.pointerId);
    updateFromPointer(event);
  };
  return <div ref={gridRef} className={`size-grid${disabled ? " disabled" : ""}`} onPointerDown={handlePointerDown} onPointerMove={(event) => event.currentTarget.hasPointerCapture?.(event.pointerId) && updateFromPointer(event)} aria-disabled={disabled}>
    <div className="grid-selection" style={{ width: `${widthPercent}%`, height: `${heightPercent}%` }} />
    <div className="grid-origin"><span>x1, y1</span></div>
    <div className="grid-handle" role="slider" tabIndex={disabled ? -1 : 0} aria-label="画布尺寸" aria-valuemin={min} aria-valuemax={max} aria-valuenow={safeWidth} aria-valuetext={`${safeWidth} × ${safeHeight} 像素`} onKeyDown={handleKeyDown} onPointerDown={(event) => { event.stopPropagation(); handlePointerDown(event); }} style={{ left: `${widthPercent}%`, bottom: `${heightPercent}%` }} />
    <span className={`grid-size-label ${safeHeight >= 1792 ? "near-top" : ""} ${safeWidth <= 768 ? "near-left" : ""}`} style={safeHeight >= 1792 ? { left: safeWidth <= 768 ? "8px" : `${widthPercent}%`, top: `calc(${100 - heightPercent}% + 12px)` } : { left: safeWidth <= 768 ? "8px" : `${widthPercent}%`, bottom: `calc(${heightPercent}% + 12px)` }}>{safeWidth} × {safeHeight}</span>
    <div className="grid-label label-tl">高</div><div className="grid-label label-br">宽</div>
  </div>;
}
