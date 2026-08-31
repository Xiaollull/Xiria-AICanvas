import { useRef, useState } from "react";

export default function SizeGrid({ width, height, onChange, min = 0, max = 2048, step = 64, disabled = false }) {
  const gridRef = useRef(null);
  const dragRef = useRef(null);
  const emittedRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const clamp = (value) => Math.max(min, Math.min(max, value));
  const snap = (value) => clamp(min + Math.round((value - min) / step) * step);
  const safeWidth = snap(Number(width) || min);
  const safeHeight = snap(Number(height) || min);
  const widthPercent = ((safeWidth - min) / (max - min)) * 100;
  const heightPercent = ((safeHeight - min) / (max - min)) * 100;
  emittedRef.current = { width: safeWidth, height: safeHeight };
  const emitChange = (nextWidth, nextHeight) => {
    const next = { width: snap(nextWidth), height: snap(nextHeight) };
    if (emittedRef.current?.width === next.width && emittedRef.current?.height === next.height) return;
    emittedRef.current = next;
    onChange(next.width, next.height);
  };
  const updateFromPointer = (event, rect = gridRef.current?.getBoundingClientRect()) => {
    if (!rect?.width || !rect.height) return;
    const x = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
    const y = Math.max(0, Math.min(1, (rect.bottom - event.clientY) / rect.height));
    emitChange(min + x * (max - min), min + y * (max - min));
  };
  const updateAxis = (axis, value) => emitChange(axis === "width" ? value : safeWidth, axis === "height" ? value : safeHeight);
  const handleKeyDown = (event) => {
    if (disabled) return;
    const delta = event.shiftKey ? step * 4 : step;
    const changes = { ArrowRight: ["width", safeWidth + delta], ArrowLeft: ["width", safeWidth - delta], ArrowUp: ["height", safeHeight + delta], ArrowDown: ["height", safeHeight - delta], Home: [event.shiftKey ? "height" : "width", min], End: [event.shiftKey ? "height" : "width", max] };
    const change = changes[event.key];
    if (!change) return;
    event.preventDefault();
    updateAxis(change[0], snap(change[1]));
  };
  const beginPointerDrag = (event, kind) => {
    if (disabled || (event.pointerType === "mouse" && event.button !== 0)) return;
    const grid = gridRef.current;
    const rect = grid?.getBoundingClientRect();
    if (!grid || !rect?.width || !rect.height) return;
    event.preventDefault();
    // Capture on the grid, not the nested handle, so its bubbling move events stay reliable.
    grid.setPointerCapture?.(event.pointerId);
    dragRef.current = { pointerId: event.pointerId, kind, rect, startX: event.clientX, startY: event.clientY, startWidth: safeWidth, startHeight: safeHeight };
    setDragging(true);
    if (kind === "surface") updateFromPointer(event, rect);
  };
  const movePointerDrag = (event) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    event.preventDefault();
    if (drag.kind === "surface") {
      updateFromPointer(event, drag.rect);
      return;
    }
    const widthDelta = ((event.clientX - drag.startX) / drag.rect.width) * (max - min);
    const heightDelta = ((drag.startY - event.clientY) / drag.rect.height) * (max - min);
    emitChange(drag.startWidth + widthDelta, drag.startHeight + heightDelta);
  };
  const endPointerDrag = (event) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    dragRef.current = null;
    setDragging(false);
    if (gridRef.current?.hasPointerCapture?.(event.pointerId)) gridRef.current.releasePointerCapture(event.pointerId);
  };
  const handlePointerDown = (event) => {
    event.stopPropagation();
    event.currentTarget.focus?.({ preventScroll: true });
    beginPointerDrag(event, "handle");
  };
  return <div ref={gridRef} className={`size-grid${disabled ? " disabled" : ""}${dragging ? " dragging" : ""}`} onPointerDown={(event) => beginPointerDrag(event, "surface")} onPointerMove={movePointerDrag} onPointerUp={endPointerDrag} onPointerCancel={endPointerDrag} onLostPointerCapture={endPointerDrag} aria-disabled={disabled}>
    <div className="grid-selection" style={{ width: `${widthPercent}%`, height: `${heightPercent}%` }} />
    <div className="grid-origin"><span>x1, y1</span></div>
    <button type="button" className="grid-handle-hitbox" role="slider" tabIndex={disabled ? -1 : 0} aria-label="画布尺寸" aria-valuemin={min} aria-valuemax={max} aria-valuenow={safeWidth} aria-valuetext={`${safeWidth} × ${safeHeight} 像素`} title="拖动调整画布尺寸；方向键微调，Shift + 方向键大步调整" disabled={disabled} onKeyDown={handleKeyDown} onPointerDown={handlePointerDown} style={{ left: `${widthPercent}%`, bottom: `${heightPercent}%` }}><span className="grid-handle" aria-hidden="true" /></button>
    <span className={`grid-size-label ${safeHeight >= 1792 ? "near-top" : ""} ${safeWidth <= 768 ? "near-left" : ""}`} style={safeHeight >= 1792 ? { left: safeWidth <= 768 ? "8px" : `${widthPercent}%`, top: `calc(${100 - heightPercent}% + 12px)` } : { left: safeWidth <= 768 ? "8px" : `${widthPercent}%`, bottom: `calc(${heightPercent}% + 12px)` }}>{safeWidth} × {safeHeight}</span>
    <div className="grid-label label-tl">高</div><div className="grid-label label-br">宽</div>
  </div>;
}
