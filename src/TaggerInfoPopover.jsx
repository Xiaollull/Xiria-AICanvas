import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { CircleHelp, ExternalLink, X } from "lucide-react";
import { popoverPosition } from "./tagger-settings.js";

export default function TaggerInfoPopover({ engine }) {
  const [open, setOpen] = useState(false);
  const [position, setPosition] = useState({ left: 12, top: 12 });
  const anchor = useRef(null), panel = useRef(null), closeButton = useRef(null);
  const id = useId();
  function close(restore = true) { setOpen(false); if (restore) anchor.current?.focus(); }
  useLayoutEffect(() => {
    if (!open) return undefined;
    const place = () => {
      if (!anchor.current || !panel.current) return;
      const bounds = panel.current.getBoundingClientRect();
      setPosition(popoverPosition(anchor.current.getBoundingClientRect(), bounds.width, bounds.height, window.innerWidth, window.innerHeight));
    };
    place();
    const resize = new ResizeObserver(place);
    resize.observe(panel.current);
    window.addEventListener("resize", place);
    window.addEventListener("scroll", place, true);
    return () => { resize.disconnect(); window.removeEventListener("resize", place); window.removeEventListener("scroll", place, true); };
  }, [open]);
  useEffect(() => {
    if (!open) return undefined;
    closeButton.current?.focus();
    const outside = (event) => { if (!panel.current?.contains(event.target) && !anchor.current?.contains(event.target)) close(false); };
    const keyboard = (event) => { if (event.key === "Escape") { event.preventDefault(); close(); } };
    document.addEventListener("pointerdown", outside);
    document.addEventListener("focusin", outside);
    document.addEventListener("keydown", keyboard);
    return () => { document.removeEventListener("pointerdown", outside); document.removeEventListener("focusin", outside); document.removeEventListener("keydown", keyboard); };
  }, [open]);
  return <><button ref={anchor} className="wd14-info-button" type="button" aria-label={`${engine.name} 简介`} aria-haspopup="dialog" aria-expanded={open} aria-controls={open ? id : undefined} onClick={() => open ? close() : setOpen(true)}><CircleHelp size={18} aria-hidden="true" /></button>
    {open && createPortal(<section ref={panel} id={id} className="wd14-info-popover" role="dialog" aria-modal="false" aria-label={`${engine.name} 简介`} style={position}>
      <header><h3>{engine.name}</h3><button ref={closeButton} type="button" aria-label="关闭模型简介" onClick={() => close()}><X size={16} aria-hidden="true" /></button></header>
      <p className="wd14-info-author">作者 · {engine.author}</p><p>{engine.description}</p><p><strong>运行方式</strong> · {engine.runtime}，可手动选择 CPU / GPU；GPU 需要可用依赖和足够显存。</p>
      <p className="wd14-info-license"><strong>许可与注意事项</strong><br />{engine.license}</p>
      {/^https:\/\/huggingface\.co\//.test(engine.repo_url || "") && <a href={engine.repo_url} target="_blank" rel="noopener noreferrer">官方仓库 <ExternalLink size={14} aria-hidden="true" /></a>}
    </section>, document.body)}</>;
}
