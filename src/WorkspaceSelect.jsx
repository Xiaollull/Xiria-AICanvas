import { useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { ChevronDown } from "lucide-react";

export default function WorkspaceSelect({
  value,
  options,
  onChange,
  disabled = false,
  placeholder = "请选择",
  ariaLabel,
  ariaInvalid,
  ariaDescribedBy,
  className = "",
}) {
  const id = useId();
  const rootRef = useRef(null);
  const menuRef = useRef(null);
  const triggerRef = useRef(null);
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const [menuStyle, setMenuStyle] = useState(null);
  const selectedIndex = options.findIndex((option) => option.value === value);
  const selected = options[selectedIndex];
  const enabledIndexes = options.reduce((indexes, option, index) => option.disabled ? indexes : [...indexes, index], []);

  const openMenu = (preferLast = false) => {
    if (disabled || !enabledIndexes.length) return;
    const fallback = preferLast ? enabledIndexes.at(-1) : enabledIndexes[0];
    setActiveIndex(selectedIndex >= 0 && !options[selectedIndex]?.disabled ? selectedIndex : fallback);
    setMenuStyle(null);
    setOpen(true);
  };
  const closeMenu = (restoreFocus = false) => {
    setOpen(false);
    if (restoreFocus) requestAnimationFrame(() => triggerRef.current?.focus());
  };
  const selectAt = (index) => {
    const option = options[index];
    if (disabled || !option || option.disabled) return;
    onChange(option.value);
    closeMenu(true);
  };
  const moveActive = (direction) => {
    if (!enabledIndexes.length) return;
    const current = enabledIndexes.indexOf(activeIndex);
    const next = current < 0
      ? (direction > 0 ? 0 : enabledIndexes.length - 1)
      : (current + direction + enabledIndexes.length) % enabledIndexes.length;
    setActiveIndex(enabledIndexes[next]);
  };
  const onKeyDown = (event) => {
    if (disabled) return;
    if (!open) {
      if (["ArrowDown", "ArrowUp", "Enter", " "].includes(event.key)) {
        event.preventDefault();
        openMenu(event.key === "ArrowUp");
      }
      return;
    }
    if (event.key === "Escape") { event.preventDefault(); closeMenu(true); return; }
    if (event.key === "Tab") { closeMenu(); return; }
    if (event.key === "ArrowDown") { event.preventDefault(); moveActive(1); return; }
    if (event.key === "ArrowUp") { event.preventDefault(); moveActive(-1); return; }
    if (event.key === "Home") { event.preventDefault(); setActiveIndex(enabledIndexes[0]); return; }
    if (event.key === "End") { event.preventDefault(); setActiveIndex(enabledIndexes.at(-1)); return; }
    if (["Enter", " "].includes(event.key)) { event.preventDefault(); selectAt(activeIndex); }
  };

  useEffect(() => {
    if (!open) return undefined;
    const positionMenu = () => {
      const rect = rootRef.current?.getBoundingClientRect();
      if (!rect) return;
      const width = Math.min(Math.max(rect.width, 180), window.innerWidth - 16);
      const estimatedHeight = Math.min(options.length * 35 + 10, 248);
      const spaceBelow = window.innerHeight - rect.bottom - 8;
      const spaceAbove = rect.top - 8;
      const placeAbove = spaceBelow < Math.min(estimatedHeight, 150) && spaceAbove > spaceBelow;
      const maxHeight = Math.max(96, Math.min(248, placeAbove ? spaceAbove : spaceBelow));
      const left = Math.min(Math.max(8, rect.left), window.innerWidth - width - 8);
      const top = placeAbove ? Math.max(8, rect.top - Math.min(estimatedHeight, maxHeight) - 6) : rect.bottom + 6;
      setMenuStyle({ left, top, width, maxHeight });
    };
    const onPointerDown = (event) => {
      if (!rootRef.current?.contains(event.target) && !menuRef.current?.contains(event.target)) closeMenu();
    };
    const onScroll = (event) => {
      if (!menuRef.current?.contains(event.target)) positionMenu();
    };
    const frame = requestAnimationFrame(positionMenu);
    document.addEventListener("pointerdown", onPointerDown, true);
    document.addEventListener("scroll", onScroll, true);
    window.addEventListener("resize", positionMenu);
    return () => {
      cancelAnimationFrame(frame);
      document.removeEventListener("pointerdown", onPointerDown, true);
      document.removeEventListener("scroll", onScroll, true);
      window.removeEventListener("resize", positionMenu);
    };
  }, [open, options.length]);

  useEffect(() => {
    if (!open || activeIndex < 0) return;
    menuRef.current?.querySelector(`[data-option-index="${activeIndex}"]`)?.scrollIntoView({ block: "nearest" });
  }, [activeIndex, open]);

  useEffect(() => {
    if (disabled && open) setOpen(false);
  }, [disabled, open]);

  return <div ref={rootRef} className={`workspace-select ${open ? "open" : ""}${className ? ` ${className}` : ""}`}>
    <button ref={triggerRef} type="button" className="workspace-select-trigger" aria-label={ariaLabel} aria-haspopup="listbox" aria-expanded={open} aria-controls={`${id}-menu`} aria-activedescendant={open && activeIndex >= 0 ? `${id}-option-${activeIndex}` : undefined} aria-invalid={ariaInvalid} aria-describedby={ariaDescribedBy} disabled={disabled || !enabledIndexes.length} onClick={() => open ? closeMenu() : openMenu()} onKeyDown={onKeyDown}>
      <span title={selected?.label || placeholder}>{selected?.label || placeholder}</span><ChevronDown size={14} />
    </button>
    {open && createPortal(<div ref={menuRef} id={`${id}-menu`} className="workspace-select-menu" role="listbox" aria-label={ariaLabel} style={menuStyle || { visibility: "hidden" }}>
      {options.map((option, index) => <button key={`${option.value}-${index}`} id={`${id}-option-${index}`} type="button" role="option" aria-selected={option.value === value} disabled={option.disabled} data-option-index={index} className={index === activeIndex ? "active" : ""} onPointerMove={() => !option.disabled && setActiveIndex(index)} onMouseDown={(event) => event.preventDefault()} onClick={() => selectAt(index)}><span>{option.label}</span>{option.value === value && <i />}</button>)}
    </div>, document.body)}
  </div>;
}
