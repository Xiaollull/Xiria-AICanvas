import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { ImageOff, MessageSquareText, Palette, StickyNote } from "lucide-react";

import { HOVER_PREVIEW_DELAY, hasHoverPreview, hoverPreviewPlacement } from "./lora-hover-preview.js";

// The large picture that appears beside a hovered LoRA.
//
// A mounted row is one line of text; the card behind it may hold a cover, the
// prompt the user keeps for it, tags and a note. Rather than crowd the row, all
// of that is shown at a size worth looking at only while the pointer rests on
// it. It is drawn into `document.body` because the rows it anchors to live
// inside scrolling panels and a modal, and a panel clipped by its own container
// would be worse than none.

/**
 * Opens the preview after a short rest, so running the pointer down a list of
 * sixteen mounted LoRAs does not flash sixteen pictures on the way past.
 */
export function useLoraHoverPreview({ enabled = true, delay = HOVER_PREVIEW_DELAY } = {}) {
  const [preview, setPreview] = useState(null);
  const timerRef = useRef(null);

  const hide = useCallback(() => {
    window.clearTimeout(timerRef.current);
    setPreview((current) => current === null ? current : null);
  }, []);

  const show = useCallback((element, presentation) => {
    window.clearTimeout(timerRef.current);
    if (!enabled || !element || !hasHoverPreview(presentation)) return;
    const rect = element.getBoundingClientRect();
    const anchor = { left: rect.left, top: rect.top, right: rect.right, bottom: rect.bottom };
    timerRef.current = window.setTimeout(() => setPreview({ anchor, presentation }), delay);
  }, [delay, enabled]);

  useEffect(() => () => window.clearTimeout(timerRef.current), []);
  useEffect(() => { if (!enabled) hide(); }, [enabled, hide]);

  /** Spread onto any row that should carry a preview. */
  const bind = useCallback((presentation) => ({
    onPointerEnter: (event) => show(event.currentTarget, presentation),
    onPointerLeave: hide,
    // Keyboard users reach these rows by tabbing to the controls inside them,
    // so focus opens the same card the pointer would.
    onFocus: (event) => show(event.currentTarget, presentation),
    onBlur: hide,
  }), [hide, show]);

  return { preview, show, hide, bind };
}

export default function LoraHoverPreview({ preview }) {
  const panelRef = useRef(null);
  const [placement, setPlacement] = useState(null);

  // Measured synchronously after the panel is in the DOM but before it is
  // painted. It used to wait for `requestAnimationFrame`, which is not delivered
  // to a throttled or occluded page — and until it arrived the panel stayed
  // hidden, so hovering looked like it did nothing at all.
  useLayoutEffect(() => {
    if (!preview) {
      setPlacement(null);
      return undefined;
    }
    const place = () => {
      const panel = panelRef.current;
      if (!panel) return;
      const rect = panel.getBoundingClientRect();
      setPlacement(hoverPreviewPlacement({
        anchor: preview.anchor,
        viewport: { width: window.innerWidth, height: window.innerHeight },
        size: { width: rect.width, height: rect.height },
      }));
    };
    place();
    window.addEventListener("resize", place);
    return () => window.removeEventListener("resize", place);
  }, [preview]);

  // The cover arrives after the first measurement, and an image with no
  // intrinsic height until then would leave the panel placed for a shorter box.
  const remeasure = useCallback(() => {
    const panel = panelRef.current;
    if (!panel || !preview) return;
    const rect = panel.getBoundingClientRect();
    setPlacement(hoverPreviewPlacement({
      anchor: preview.anchor,
      viewport: { width: window.innerWidth, height: window.innerHeight },
      size: { width: rect.width, height: rect.height },
    }));
  }, [preview]);

  if (!preview) return null;
  const { presentation } = preview;
  // The panel's height depends on how much the card holds, so it has to exist
  // before it can be placed. The layout effect above measures and positions it
  // in the same commit, so this hidden state is not painted.
  const style = placement
    ? { left: `${placement.left}px`, top: `${placement.top}px` }
    : { left: "0px", top: "0px", visibility: "hidden" };

  return createPortal(
    // Hidden from assistive technology on purpose: it repeats what the row's own
    // title and text already say, and a panel that follows the pointer is not
    // something a screen reader should be dragged through.
    <div ref={panelRef} className={`lora-hover-preview ${placement ? `side-${placement.side}` : ""}`} style={style} aria-hidden="true">
      <div className="lora-hover-cover">
        {presentation.coverUrl
          ? <img src={presentation.coverUrl} alt="" onLoad={remeasure} />
          : <span className="lora-hover-empty"><ImageOff size={20} />尚未设置封面</span>}
      </div>
      <div className="lora-hover-copy">
        <strong>{presentation.title}</strong>
        {presentation.renamed && <code title={presentation.fileName}>{presentation.fileName}</code>}
        {presentation.tags?.length > 0 && <div className="lora-hover-tags">{presentation.tags.map((tag) => <span key={tag}>{tag}</span>)}</div>}
        {presentation.prompt && <p className="lora-hover-prompt"><MessageSquareText size={12} /><span>{presentation.prompt}</span></p>}
        {presentation.note && <p className="lora-hover-note"><StickyNote size={12} /><span>{presentation.note}</span></p>}
        {/* A LoRA nobody has customised used to open nothing at all, which reads
            as a broken hover rather than as an empty card. */}
        {!presentation.customized && (
          <p className="lora-hover-hint"><Palette size={12} /><span>在「管理 → 已挂载」点这一行的调色板按钮，可设置封面、提示词和名称</span></p>
        )}
      </div>
    </div>,
    document.body,
  );
}
