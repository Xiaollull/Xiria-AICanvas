import { useCallback, useEffect, useRef, useState } from "react";
import { Bot, GripVertical, Maximize2, Minimize2, SquareArrowOutUpRight, X } from "lucide-react";

import AiAssistant from "./AiAssistant.jsx";
import {
  RESIZE_HANDLES,
  clampWindowRect,
  defaultWindowRect,
  isMaximizedRect,
  maximizedWindowRect,
  moveWindow,
  readStoredWindowRect,
  resizeWindow,
  writeStoredWindowRect,
} from "./assistant-window.js";

// Floating, non-modal assistant window.
//
// Deliberately has no backdrop element: the point of this mode is that the workspace behind it stays
// visible and clickable while the user edits a prompt. Nothing here may capture pointer events
// outside the window's own bounds.

const viewportSize = () => ({ width: window.innerWidth, height: window.innerHeight });

export default function AiAssistantOverlay({ open, onClose, onApplyPrompt, promptSnapshot }) {
  const [rect, setRect] = useState(() => readStoredWindowRect(viewportSize(), window.localStorage));
  // The size to come back to when the window is un-maximised. Absent on a fresh load, in which case
  // the opening rect is a better answer than a full-screen panel the user cannot shrink in one click.
  const [restoreRect, setRestoreRect] = useState(null);
  // Which service the chat is actually talking to. Owned by AiAssistant, shown here, so the title
  // bar answers "which model is this" without a second settings fetch.
  const [status, setStatus] = useState({ ready: false, model: "", configured: false });
  const gesture = useRef(null);
  const windowRef = useRef(null);
  const maximized = isMaximizedRect(rect, viewportSize());

  // The rect at gesture start is the reference for every move, so a clamped axis never accumulates
  // an offset between the pointer and the grabbed edge.
  const beginGesture = useCallback((event, handle) => {
    if (event.button !== 0) return;
    if (handle === "move" && event.target.closest("button")) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture?.(event.pointerId);
    gesture.current = { handle, pointerId: event.pointerId, startX: event.clientX, startY: event.clientY, startRect: rect };
  }, [rect]);

  const continueGesture = useCallback((event) => {
    const active = gesture.current;
    if (!active || active.pointerId !== event.pointerId) return;
    const deltaX = event.clientX - active.startX;
    const deltaY = event.clientY - active.startY;
    setRect(active.handle === "move"
      ? moveWindow(active.startRect, deltaX, deltaY, viewportSize())
      : resizeWindow(active.startRect, active.handle, deltaX, deltaY, viewportSize()));
  }, []);

  const endGesture = useCallback((event) => {
    const active = gesture.current;
    if (!active || active.pointerId !== event.pointerId) return;
    gesture.current = null;
    event.currentTarget.releasePointerCapture?.(event.pointerId);
    setRect((current) => {
      writeStoredWindowRect(current, window.localStorage);
      return current;
    });
  }, []);

  // A window parked against the right edge would sit off-screen after the browser is narrowed.
  useEffect(() => {
    if (!open) return undefined;
    const reflow = () => setRect((current) => moveWindow(current, 0, 0, viewportSize()));
    window.addEventListener("resize", reflow);
    return () => window.removeEventListener("resize", reflow);
  }, [open]);

  useEffect(() => {
    if (!open) return undefined;
    const dismiss = (event) => {
      if (event.key === "Escape" && !event.defaultPrevented) onClose?.();
    };
    window.addEventListener("keydown", dismiss);
    return () => window.removeEventListener("keydown", dismiss);
  }, [onClose, open]);

  const commitRect = useCallback((next) => {
    setRect(next);
    writeStoredWindowRect(next, window.localStorage);
  }, []);

  const toggleMaximize = useCallback(() => {
    const viewport = viewportSize();
    if (isMaximizedRect(rect, viewport)) {
      commitRect(restoreRect ? clampWindowRect(restoreRect, viewport) : defaultWindowRect(viewport));
      return;
    }
    setRestoreRect(rect);
    commitRect(maximizedWindowRect(viewport));
  }, [commitRect, rect, restoreRect]);

  // Hands the session to a full browser tab. The workspace tab stays exactly as it was; the
  // transcript travels through localStorage, so the conversation continues rather than restarting.
  const popOut = useCallback(() => {
    window.open("/assistant", "_blank", "noopener,noreferrer");
    onClose?.();
  }, [onClose]);

  if (!open) return null;

  return (
    <section
      className="assistant-window"
      ref={windowRef}
      role="dialog"
      aria-label="AI 助手"
      aria-modal="false"
      style={{ left: `${rect.x}px`, top: `${rect.y}px`, width: `${rect.width}px`, height: `${rect.height}px` }}
    >
      <header
        className="assistant-window-bar"
        onPointerDown={(event) => beginGesture(event, "move")}
        onPointerMove={continueGesture}
        onPointerUp={endGesture}
        onPointerCancel={endGesture}
        onDoubleClick={(event) => { if (!event.target.closest("button")) toggleMaximize(); }}
      >
        <GripVertical size={14} className="assistant-grip" />
        <Bot size={15} />
        <span className="assistant-window-title">
          <strong>AI 助手</strong>
          {/* The saved configuration in use, not just the model: with several profiles stored, the
              model id alone does not say which of them the next message will go to. */}
          {status.ready && <em className={status.configured ? "" : "off"} title={[status.profile, status.model].filter(Boolean).join(" · ")}>{status.profile || status.model || "未配置"}</em>}
        </span>
        <span className="assistant-spacer" />
        <button type="button" onClick={toggleMaximize} title={maximized ? "还原窗口大小" : "最大化窗口"} aria-label={maximized ? "还原窗口大小" : "最大化窗口"}>
          {maximized ? <Minimize2 size={13} /> : <Maximize2 size={13} />}
        </button>
        <button type="button" onClick={popOut} title="在新标签页中打开" aria-label="在新标签页中打开"><SquareArrowOutUpRight size={14} /></button>
        <button type="button" className="assistant-close" onClick={() => onClose?.()} title="关闭" aria-label="关闭"><X size={15} /></button>
      </header>

      <AiAssistant variant="overlay" onApplyPrompt={onApplyPrompt} onStatusChange={setStatus} promptSnapshot={promptSnapshot} />

      {RESIZE_HANDLES.map((handle) => (
        <div
          key={handle}
          className={`assistant-resize assistant-resize-${handle}`}
          onPointerDown={(event) => beginGesture(event, handle)}
          onPointerMove={continueGesture}
          onPointerUp={endGesture}
          onPointerCancel={endGesture}
        />
      ))}
    </section>
  );
}
