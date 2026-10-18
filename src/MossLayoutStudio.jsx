// The MOSS Layout Studio (QPlan2/MOSS/P2 §6.2): a full-screen editor for the coordinate layout.
//
// Every edit goes straight to the workspace through `onChange`; undo/redo is this session's own
// history. Coordinates are top-left pixels on the generation canvas, exactly the official readable
// draft; the centred `yxyx` form the model reads is only ever produced by the compiler.
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  AlignCenterHorizontal, AlignCenterVertical, AlignEndHorizontal, AlignEndVertical, AlignHorizontalDistributeCenter,
  AlignStartHorizontal, AlignStartVertical, AlignVerticalDistributeCenter, ArrowDown, ArrowUp, Check, ClipboardCopy,
  CopyPlus, Download, Eye, EyeOff, FileJson, Hand, ImagePlus, LayoutTemplate, Lock, LockOpen, Maximize, MousePointer2,
  Plus, Redo2, SquarePlus, Trash2, TriangleAlert, Undo2, Upload, X, ZoomIn, ZoomOut,
} from "lucide-react";
import {
  MOSS_LAYOUT_CANVAS_KINDS,
  MOSS_LAYOUT_CANVAS_NAMES,
  MOSS_LAYOUT_GRID,
  MOSS_LAYOUT_HANDLES,
  MOSS_LAYOUT_HANDLE_AXES,
  MOSS_LAYOUT_LABELS,
  MOSS_LAYOUT_LABEL_NAMES,
  MOSS_LAYOUT_MAX_ITEMS,
  MOSS_LAYOUT_MIN_BOX,
  MOSS_LAYOUT_TEMPLATES,
  alignMossLayoutItems,
  boxFromDrag,
  centredBox,
  clampBox,
  duplicateMossLayoutItems,
  mossLayoutDraft,
  mossLayoutFromTemplate,
  mossLayoutItemsAt,
  moveBox,
  moveMossLayoutItem,
  newMossLayoutItem,
  parseMossLayoutImport,
  patchMossLayoutItem,
  resizeBox,
  snapDelta,
  snapTargets,
} from "./moss-layout.js";
import { copyText } from "./clipboard.js";
import { MossLayoutMiniature, MossTokenMeter } from "./MossLayoutCard.jsx";

const ALIGN_MODES = [
  { id: "left", label: "左对齐", Icon: AlignStartVertical },
  { id: "hcenter", label: "水平居中", Icon: AlignCenterVertical },
  { id: "right", label: "右对齐", Icon: AlignEndVertical },
  { id: "top", label: "顶对齐", Icon: AlignStartHorizontal },
  { id: "vcenter", label: "垂直居中", Icon: AlignCenterHorizontal },
  { id: "bottom", label: "底对齐", Icon: AlignEndHorizontal },
  { id: "hspace", label: "水平等距（≥3 个）", Icon: AlignHorizontalDistributeCenter },
  { id: "vspace", label: "垂直等距（≥3 个）", Icon: AlignVerticalDistributeCenter },
];
const META_SUGGESTIONS = ["poster", "photograph", "illustration", "comic", "infographic"];
const RULER_STEPS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
const VIEW_PAD = 40;
const MIN_ZOOM = 0.1, MAX_ZOOM = 12;
const SNAP_PX = 7;
const CAPTION_LINE = 15.4;

const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const excerpt = (text, length) => {
  const flat = String(text ?? "").replace(/\s+/g, " ").trim();
  return flat.length > length ? `${flat.slice(0, length - 1)}…` : flat;
};
const isEditable = (target) => Boolean(target?.closest?.("input, textarea, select, [contenteditable='true']"));
const isHexColour = (value) => /^#(?:[0-9a-f]{3}|[0-9a-f]{6})$/i.test(String(value).trim());
const unionBox = (boxes) => [Math.min(...boxes.map((b) => b[0])), Math.min(...boxes.map((b) => b[1])), Math.max(...boxes.map((b) => b[2])), Math.max(...boxes.map((b) => b[3]))];

function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function useElementSize(ref) {
  const [size, setSize] = useState({ width: 0, height: 0 });
  useLayoutEffect(() => {
    const element = ref.current;
    if (!element) return undefined;
    const measure = () => setSize({ width: element.clientWidth, height: element.clientHeight });
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, [ref]);
  return size;
}

/** Free-text chips (meta, tags, palette). Enter or a comma adds; colour mode also offers a picker. */
function ChipField({ label, hint, values, onChange, placeholder, suggestions = [], colour = false, problem = "" }) {
  const [text, setText] = useState("");
  const [picked, setPicked] = useState("#1e3a5f");
  const add = (value) => {
    const entries = String(value).split(/[,，\n]/).map((entry) => entry.trim()).filter(Boolean).filter((entry) => !values.includes(entry));
    if (entries.length) onChange([...values, ...entries].slice(0, 64));
    setText("");
  };
  const unused = suggestions.filter((entry) => !values.includes(entry));
  return <div className={`moss-field ${problem ? "invalid" : ""}`}>
    <span className="moss-field-label">{label}{hint && <small>{hint}</small>}</span>
    <div className="moss-chips">
      {values.map((value) => <span key={value} className="moss-chip">
        {colour && isHexColour(value) && <i style={{ background: value }} />}
        {value}
        <button type="button" onClick={() => onChange(values.filter((entry) => entry !== value))} aria-label={`移除 ${value}`}><X size={10} /></button>
      </span>)}
      <input value={text} placeholder={placeholder} onChange={(event) => setText(event.target.value)}
        onKeyDown={(event) => {
          if ((event.key === "Enter" || event.key === ",") && text.trim()) { event.preventDefault(); add(text); }
          if (event.key === "Backspace" && !text && values.length) onChange(values.slice(0, -1));
        }}
        onBlur={() => text.trim() && add(text)} />
    </div>
    {colour && <div className="moss-colour-add">
      <input type="color" value={picked} onChange={(event) => setPicked(event.target.value)} aria-label="选择颜色" />
      <button type="button" onClick={() => add(picked.toUpperCase())}><Plus size={11} />添加 {picked.toUpperCase()}</button>
    </div>}
    {unused.length > 0 && <div className="moss-suggestions">{unused.map((entry) => <button key={entry} type="button" onClick={() => add(entry)}>+ {entry}</button>)}</div>}
    {problem && <em>{problem}</em>}
  </div>;
}

/** One coordinate, typed freely and committed on Enter or blur. */
function CoordField({ label, value, onCommit }) {
  const [text, setText] = useState(String(value));
  useEffect(() => setText(String(value)), [value]);
  const commit = () => {
    const number = Number(text);
    if (text.trim() !== "" && Number.isFinite(number)) onCommit(Math.round(number));
    else setText(String(value));
  };
  return <label className="moss-coord">
    <span>{label}</span>
    <input inputMode="numeric" value={text} onChange={(event) => setText(event.target.value)} onBlur={commit}
      onKeyDown={(event) => { if (event.key === "Enter") commit(); }} />
  </label>;
}

export default function MossLayoutStudio({ layout: workspaceLayout, onChange, prefix = "", compiled, tokens, stageSizes = [], adetailerActive = false, onClose }) {
  // A drag is previewed here and reaches the workspace once, when it ends: the Basic page is one large
  // component, and re-rendering it on every pointer move made dragging stutter.
  const [previewLayout, setPreviewLayout] = useState(null);
  const layout = previewLayout ?? workspaceLayout;
  const W = layout.width, H = layout.height;
  const layoutRef = useRef(layout);
  layoutRef.current = layout;
  const [tool, setTool] = useState("select");
  const [drawLabel, setDrawLabel] = useState("object");
  const [selected, setSelected] = useState(() => (layout.items[0] ? [layout.items[0].id] : []));
  const [tab, setTab] = useState(layout.summary.caption.trim() ? "item" : "scene");
  const [pane, setPane] = useState("canvas");
  const [view, setView] = useState({ zoom: 1, x: 0, y: 0 });
  const [overlays, setOverlays] = useState({ snap: true, grid: true, thirds: false });
  const [menu, setMenu] = useState("");
  const [reference, setReference] = useState(null);
  const [guides, setGuides] = useState([]);
  const [drawing, setDrawing] = useState(null);
  const [cursor, setCursor] = useState(null);
  const [spaceHeld, setSpaceHeld] = useState(false);
  const [notice, setNotice] = useState("");
  const [focusCaption, setFocusCaption] = useState(0);
  const [jsonEdit, setJsonEdit] = useState(null);
  const [jsonError, setJsonError] = useState("");
  const [importText, setImportText] = useState("");
  const [importError, setImportError] = useState("");
  const [dragRow, setDragRow] = useState(null);
  const [, setHistoryVersion] = useState(0);
  const history = useRef({ undo: [], redo: [], group: "", at: 0 });
  const gesture = useRef(null);
  const studioRef = useRef(null);
  const viewportRef = useRef(null);
  const paperRef = useRef(null);
  const captionRef = useRef(null);
  const keyHandler = useRef(null);
  const wheelHandler = useRef(null);
  const viewport = useElementSize(viewportRef);

  const fit = viewport.width > 0 && viewport.height > 0
    ? Math.max(0.01, Math.min((viewport.width - VIEW_PAD * 2) / W, (viewport.height - VIEW_PAD * 2) / H))
    : 0.5;
  const scale = fit * view.zoom;
  const paper = {
    width: W * scale, height: H * scale,
    left: (viewport.width - W * scale) / 2 + view.x, top: (viewport.height - H * scale) / 2 + view.y,
  };
  const items = layout.items;
  const selectedIds = selected.filter((id) => items.some((item) => item.id === id));
  const primary = items.find((item) => item.id === selectedIds.at(-1)) || null;
  const issues = compiled.issues;
  const issuesFor = (id) => issues.filter((issue) => issue.itemId === id);
  const numberOf = (id) => items.findIndex((item) => item.id === id) + 1;
  const full = items.length >= MOSS_LAYOUT_MAX_ITEMS;

  useEffect(() => {
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    studioRef.current?.focus();
    return () => { document.body.style.overflow = previous; };
  }, []);
  useEffect(() => () => { if (reference?.url) URL.revokeObjectURL(reference.url); }, [reference?.url]);
  useEffect(() => {
    if (!notice) return undefined;
    const timer = setTimeout(() => setNotice(""), 3800);
    return () => clearTimeout(timer);
  }, [notice]);
  useEffect(() => {
    if (!focusCaption || window.innerWidth < 900) return;
    captionRef.current?.focus();
  }, [focusCaption]);

  // --- history -------------------------------------------------------------------------------------
  const record = (group = "") => {
    const entry = history.current;
    const now = Date.now();
    if (group && entry.group === group && now - entry.at < 1500) { entry.at = now; return; }
    entry.undo.push(layoutRef.current);
    if (entry.undo.length > 200) entry.undo.shift();
    entry.redo = [];
    entry.group = group;
    entry.at = now;
    setHistoryVersion((value) => value + 1);
  };
  const apply = (next) => { layoutRef.current = next; onChange(next); };
  const preview = (next) => { layoutRef.current = next; setPreviewLayout(next); };
  const commit = (next, group = "") => {
    if (next === layoutRef.current) return;
    record(group);
    apply(next);
  };
  const step = (from, to) => {
    const entry = history.current;
    if (!entry[from].length) return;
    entry[to].push(layoutRef.current);
    entry.group = "";
    apply(entry[from].pop());
    setHistoryVersion((value) => value + 1);
  };
  const undo = () => step("undo", "redo");
  const redo = () => step("redo", "undo");

  // --- edits -------------------------------------------------------------------------------------
  const patchSummary = (patch, group = "") => commit({ ...layoutRef.current, summary: { ...layoutRef.current.summary, ...patch } }, group);
  const patchItem = (id, patch, group = "") => commit(patchMossLayoutItem(layoutRef.current, id, patch), group);
  const addItem = (label, box = null) => {
    if (layoutRef.current.items.length >= MOSS_LAYOUT_MAX_ITEMS) return;
    const item = newMossLayoutItem(layoutRef.current, label, box);
    commit({ ...layoutRef.current, items: [...layoutRef.current.items, item] });
    setSelected([item.id]);
    setTab("item");
    setFocusCaption((value) => value + 1);
  };
  const removeItems = (ids) => {
    const removable = new Set(ids.filter((id) => !layoutRef.current.items.find((item) => item.id === id)?.locked));
    if (!removable.size) return;
    commit({ ...layoutRef.current, items: layoutRef.current.items.filter((item) => !removable.has(item.id)) });
    setSelected((current) => current.filter((id) => !removable.has(id)));
  };
  const duplicateItems = (ids) => {
    const result = duplicateMossLayoutItems(layoutRef.current, ids);
    if (!result.ids.length) return;
    commit(result.layout);
    setSelected(result.ids);
  };
  const nudge = (dx, dy) => {
    const current = layoutRef.current;
    const ids = new Set(selectedIds);
    const moved = { ...current, items: current.items.map((item) => ids.has(item.id) && !item.locked ? { ...item, box: moveBox(item.box, dx, dy, W, H) } : item) };
    commit(moved, "nudge");
  };
  const align = (mode) => commit(alignMossLayoutItems(layoutRef.current, selectedIds, mode));
  const applyTemplate = (template) => {
    const next = mossLayoutFromTemplate(template.id, W, H);
    if (!next) return;
    commit({ ...next, enabled: true });
    setSelected([]);
    setMenu("");
    setTab("scene");
    setNotice(`已应用模板「${template.name}」；Ctrl+Z 可撤销`);
  };
  const applyImport = (text) => {
    const result = parseMossLayoutImport(text, W, H);
    if (result.error) { setImportError(result.error); return false; }
    commit({ ...result.layout, enabled: true });
    setSelected(result.layout.items[0] ? [result.layout.items[0].id] : []);
    setImportError("");
    setImportText("");
    setMenu("");
    setNotice(["已导入布局", result.note].filter(Boolean).join("；"));
    return true;
  };
  const copy = async (text, label) => setNotice(await copyText(text) ? `已复制${label}` : `无法复制${label}：浏览器拒绝了剪贴板访问`);

  // --- canvas geometry ----------------------------------------------------------------------------
  const toCanvas = (event) => {
    const rect = paperRef.current.getBoundingClientRect();
    return { x: (event.clientX - rect.left) / scale, y: (event.clientY - rect.top) / scale };
  };
  // Strong targets (edges, centre, thirds, other boxes) win over the 16 px grid, and only they draw a guide.
  const snapAxis = (edges, axis, exclude) => {
    const threshold = SNAP_PX / scale;
    const current = layoutRef.current;
    const strongTargets = snapTargets(current, axis, exclude, { useGrid: false });
    if (overlays.thirds) {
      const limit = axis === "x" ? current.width : current.height;
      strongTargets.push(Math.round(limit / 3), Math.round(limit * 2 / 3));
    }
    const strong = snapDelta(edges, strongTargets, threshold);
    if (strong) return { ...strong, guide: true };
    const grid = snapDelta(edges, edges.map((edge) => Math.round(edge / MOSS_LAYOUT_GRID) * MOSS_LAYOUT_GRID), threshold);
    return grid ? { ...grid, guide: false } : null;
  };

  const zoomAt = (zoom, screenX, screenY) => {
    const next = clamp(zoom, MIN_ZOOM, MAX_ZOOM);
    setView((current) => {
      const before = fit * current.zoom, after = fit * next;
      const left = (viewport.width - W * before) / 2 + current.x, top = (viewport.height - H * before) / 2 + current.y;
      const px = (screenX - left) / before, py = (screenY - top) / before;
      return { zoom: next, x: screenX - px * after - (viewport.width - W * after) / 2, y: screenY - py * after - (viewport.height - H * after) / 2 };
    });
  };
  const zoomBy = (factor) => zoomAt(view.zoom * factor, viewport.width / 2, viewport.height / 2);

  wheelHandler.current = (event) => {
    const rect = viewportRef.current.getBoundingClientRect();
    const factor = Math.exp(-event.deltaY * (event.deltaMode === 1 ? 0.05 : 0.0015));
    zoomAt(view.zoom * factor, event.clientX - rect.left, event.clientY - rect.top);
  };
  useEffect(() => {
    const element = viewportRef.current;
    if (!element) return undefined;
    const wheel = (event) => { event.preventDefault(); wheelHandler.current(event); };
    element.addEventListener("wheel", wheel, { passive: false });
    return () => element.removeEventListener("wheel", wheel);
  }, []);

  const cancelGesture = () => {
    const current = gesture.current;
    if (!current) return;
    gesture.current = null;
    if (current.recorded) {
      history.current.undo.pop();
      setHistoryVersion((value) => value + 1);
    }
    layoutRef.current = current.before;
    setPreviewLayout(null);
    setDrawing(null);
    setGuides([]);
  };

  const onPointerDown = (event) => {
    if (event.button === 2 || !paperRef.current) return;
    setMenu("");
    // The canvas takes the keyboard: Delete and the arrows act on boxes, not on the caption just typed.
    if (isEditable(document.activeElement)) document.activeElement.blur();
    studioRef.current?.focus({ preventScroll: true });
    const point = toCanvas(event);
    const base = { pointerId: event.pointerId, client: { x: event.clientX, y: event.clientY }, start: point, moved: false, before: layoutRef.current, recorded: false };
    const current = layoutRef.current;
    if (event.button === 1 || tool === "pan" || spaceHeld) {
      gesture.current = { ...base, kind: "pan", origin: view };
    } else {
      const handle = event.target.closest("[data-handle]");
      const boxElement = tool === "select" ? event.target.closest("[data-item]") : null;
      if (handle) {
        const item = current.items.find((entry) => entry.id === handle.dataset.owner);
        if (!item || item.locked) return;
        gesture.current = { ...base, kind: "resize", id: item.id, handle: handle.dataset.handle, origin: item.box };
      } else if (boxElement) {
        const id = boxElement.dataset.item;
        let nextSelection = selectedIds;
        if (event.shiftKey) nextSelection = selectedIds.includes(id) ? selectedIds.filter((entry) => entry !== id) : [...selectedIds, id];
        else if (!selectedIds.includes(id)) nextSelection = [id];
        setSelected(nextSelection);
        if (!event.shiftKey) setTab("item");
        const movable = current.items.filter((item) => nextSelection.includes(item.id) && !item.locked);
        gesture.current = {
          ...base, kind: "move", id, alt: event.altKey, shift: event.shiftKey, wasSelected: selectedIds.includes(id),
          ids: movable.map((item) => item.id), origin: new Map(movable.map((item) => [item.id, item.box])),
        };
      } else {
        if (full) { setNotice(`最多 ${MOSS_LAYOUT_MAX_ITEMS} 个框`); return; }
        let start = { x: clamp(point.x, 0, W), y: clamp(point.y, 0, H) };
        if (overlays.snap) {
          const sx = snapAxis([start.x], "x", []), sy = snapAxis([start.y], "y", []);
          start = { x: start.x + (sx?.delta || 0), y: start.y + (sy?.delta || 0) };
        }
        gesture.current = { ...base, kind: "draw", start };
      }
    }
    viewportRef.current.setPointerCapture(event.pointerId);
    event.preventDefault();
  };

  const onPointerMove = (event) => {
    if (!paperRef.current) return;
    const point = toCanvas(event);
    setCursor(point.x >= 0 && point.y >= 0 && point.x <= W && point.y <= H ? { x: Math.round(point.x), y: Math.round(point.y) } : null);
    const active = gesture.current;
    if (!active || active.pointerId !== event.pointerId) return;
    if (!active.moved && Math.hypot(event.clientX - active.client.x, event.clientY - active.client.y) < 3) return;
    const first = !active.moved;
    active.moved = true;
    if (active.kind === "pan") {
      setView({ ...active.origin, x: active.origin.x + event.clientX - active.client.x, y: active.origin.y + event.clientY - active.client.y });
      return;
    }
    const snapping = overlays.snap && !(event.ctrlKey || event.metaKey);
    const nextGuides = [];
    if (active.kind === "move") {
      if (!active.ids.length) return;
      if (first) {
        record();
        active.recorded = true;
        if (active.alt) {
          const copies = duplicateMossLayoutItems(layoutRef.current, active.ids, 0);
          preview(copies.layout);
          active.ids = copies.ids;
          active.origin = new Map(copies.ids.map((id) => [id, copies.layout.items.find((item) => item.id === id).box]));
          setSelected(copies.ids);
        }
      }
      const [ux1, uy1, ux2, uy2] = unionBox([...active.origin.values()]);
      let dx = clamp(point.x - active.start.x, -ux1, W - ux2), dy = clamp(point.y - active.start.y, -uy1, H - uy2);
      if (snapping) {
        const sx = snapAxis([ux1 + dx, ux2 + dx, (ux1 + ux2) / 2 + dx], "x", active.ids);
        const sy = snapAxis([uy1 + dy, uy2 + dy, (uy1 + uy2) / 2 + dy], "y", active.ids);
        if (sx) { dx = clamp(dx + sx.delta, -ux1, W - ux2); if (sx.guide) nextGuides.push({ axis: "x", value: sx.target }); }
        if (sy) { dy = clamp(dy + sy.delta, -uy1, H - uy2); if (sy.guide) nextGuides.push({ axis: "y", value: sy.target }); }
      }
      dx = Math.round(dx); dy = Math.round(dy);
      const current = layoutRef.current;
      preview({ ...current, items: current.items.map((item) => active.origin.has(item.id) ? { ...item, box: moveBox(active.origin.get(item.id), dx, dy, W, H) } : item) });
    } else if (active.kind === "resize") {
      if (first) { record(); active.recorded = true; }
      const keepRatio = event.shiftKey;
      let box = resizeBox(active.origin, active.handle, point.x - active.start.x, point.y - active.start.y, W, H, { keepRatio });
      if (snapping && !keepRatio) {
        const axes = MOSS_LAYOUT_HANDLE_AXES[active.handle];
        let [x1, y1, x2, y2] = box;
        if (axes.x) {
          const edge = axes.x < 0 ? x1 : x2;
          const hit = snapAxis([edge], "x", [active.id]);
          if (hit) {
            if (axes.x < 0) x1 = clamp(edge + hit.delta, 0, x2 - MOSS_LAYOUT_MIN_BOX); else x2 = clamp(edge + hit.delta, x1 + MOSS_LAYOUT_MIN_BOX, W);
            if (hit.guide) nextGuides.push({ axis: "x", value: hit.target });
          }
        }
        if (axes.y) {
          const edge = axes.y < 0 ? y1 : y2;
          const hit = snapAxis([edge], "y", [active.id]);
          if (hit) {
            if (axes.y < 0) y1 = clamp(edge + hit.delta, 0, y2 - MOSS_LAYOUT_MIN_BOX); else y2 = clamp(edge + hit.delta, y1 + MOSS_LAYOUT_MIN_BOX, H);
            if (hit.guide) nextGuides.push({ axis: "y", value: hit.target });
          }
        }
        box = clampBox([x1, y1, x2, y2], W, H);
      }
      preview(patchMossLayoutItem(layoutRef.current, active.id, { box }));
    } else if (active.kind === "draw") {
      let end = { x: clamp(point.x, 0, W), y: clamp(point.y, 0, H) };
      if (snapping) {
        const sx = snapAxis([end.x], "x", []), sy = snapAxis([end.y], "y", []);
        if (sx) { end = { ...end, x: end.x + sx.delta }; if (sx.guide) nextGuides.push({ axis: "x", value: sx.target }); }
        if (sy) { end = { ...end, y: end.y + sy.delta }; if (sy.guide) nextGuides.push({ axis: "y", value: sy.target }); }
      }
      active.end = end;
      setDrawing(boxFromDrag(active.start, end, W, H));
    }
    setGuides(nextGuides);
  };

  const onPointerUp = (event) => {
    const active = gesture.current;
    if (!active || active.pointerId !== event.pointerId) return;
    gesture.current = null;
    setGuides([]);
    if (active.recorded) {
      apply(layoutRef.current);
      setPreviewLayout(null);
    }
    if (active.kind === "draw") {
      setDrawing(null);
      const end = active.end;
      if (active.moved && end && Math.abs(end.x - active.start.x) >= MOSS_LAYOUT_MIN_BOX / 2 && Math.abs(end.y - active.start.y) >= MOSS_LAYOUT_MIN_BOX / 2) {
        addItem(drawLabel, boxFromDrag(active.start, end, W, H));
      } else if (!active.moved) {
        setSelected([]);
      }
    } else if (active.kind === "move" && !active.moved) {
      if (active.alt && !active.shift) {
        // Alt-click walks down through the boxes stacked under the pointer.
        const stack = mossLayoutItemsAt(layoutRef.current, active.start.x, active.start.y);
        if (stack.length > 1) {
          const at = stack.findIndex((item) => item.id === (primary?.id ?? active.id));
          setSelected([stack[(at + 1) % stack.length].id]);
        }
      } else if (!active.shift && active.wasSelected && selectedIds.length > 1) {
        setSelected([active.id]);
      }
    }
  };

  // --- keyboard ------------------------------------------------------------------------------------
  keyHandler.current = (event) => {
    const editing = isEditable(event.target);
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      if (gesture.current) cancelGesture();
      else if (menu) setMenu("");
      else if (editing) event.target.blur();
      else if (selectedIds.length) setSelected([]);
      else onClose();
      return;
    }
    if (editing) return;
    const mod = event.ctrlKey || event.metaKey;
    const key = event.key.toLowerCase();
    let handled = true;
    if (mod && key === "z") (event.shiftKey ? redo : undo)();
    else if (mod && key === "y") redo();
    else if (mod && key === "d") duplicateItems(selectedIds);
    else if (mod && key === "a") setSelected(items.filter((item) => !item.hidden).map((item) => item.id));
    else if (mod) handled = false;
    else if (key === "v") setTool("select");
    else if (key === "b") setTool("draw");
    else if (key === "h") setTool("pan");
    else if (key === "delete" || key === "backspace") removeItems(selectedIds);
    else if (event.key.startsWith("Arrow") && selectedIds.length) {
      const amount = event.shiftKey ? MOSS_LAYOUT_GRID : 1;
      nudge(event.key === "ArrowLeft" ? -amount : event.key === "ArrowRight" ? amount : 0, event.key === "ArrowUp" ? -amount : event.key === "ArrowDown" ? amount : 0);
    } else if (event.key === " ") setSpaceHeld(true);
    else handled = false;
    if (handled) { event.preventDefault(); event.stopPropagation(); }
  };
  useEffect(() => {
    const down = (event) => keyHandler.current(event);
    const up = (event) => { if (event.key === " ") setSpaceHeld(false); };
    window.addEventListener("keydown", down, true);
    window.addEventListener("keyup", up, true);
    return () => { window.removeEventListener("keydown", down, true); window.removeEventListener("keyup", up, true); };
  }, []);

  // --- rendering helpers -------------------------------------------------------------------------
  const rulerStep = RULER_STEPS.find((value) => value * scale >= 56) || 8192;
  const ticks = (limit) => Array.from({ length: Math.floor(limit / rulerStep) + 1 }, (_, index) => index * rulerStep);
  const gridSize = [16, 32, 64, 128, 256].find((value) => value * scale >= 14) || 512;
  const draftText = JSON.stringify(mossLayoutDraft(layout), null, 2);
  const hiddenCount = items.filter((item) => item.hidden).length;
  const counts = Object.fromEntries(MOSS_LAYOUT_LABELS.map((label) => [label, items.filter((item) => item.label === label && !item.hidden).length]));
  const canvasCursor = gesture.current?.kind === "pan" || spaceHeld || tool === "pan" ? "pan" : tool === "draw" ? "draw" : "select";
  const firstIssue = issues[0];
  const goToIssue = (issue) => {
    if (!issue) return;
    if (issue.itemId) { setSelected([issue.itemId]); setTab("item"); setPane("inspector"); }
    else { setTab(issue.path.startsWith("summary") ? "scene" : "json"); setPane("inspector"); }
  };
  const summaryProblem = (field) => issues.find((issue) => issue.path === `summary.${field}` || issue.path.startsWith(`summary.${field}[`))?.message || "";
  const itemProblem = (item, field) => issuesFor(item.id).find((issue) => new RegExp(`^items\\[\\d+\\]\\.${field}(\\[|$)`).test(issue.path))?.message || "";

  const itemInspector = () => {
    if (selectedIds.length > 1) {
      const chosen = items.filter((item) => selectedIds.includes(item.id));
      return <div className="moss-inspector-body">
        <p className="moss-inspector-note">已选中 {chosen.length} 个元素。拖动画布上的任意一个可整体移动；Shift+点击 增减选择。</p>
        <div className="moss-field">
          <span className="moss-field-label">对齐与分布<small>{chosen.length === 1 ? "相对画布" : "相对选区"}</small></span>
          <div className="moss-align-grid">{ALIGN_MODES.map(({ id, label, Icon }) => <button key={id} type="button" title={label} disabled={(id === "hspace" || id === "vspace") && chosen.length < 3} onClick={() => align(id)}><Icon size={15} /><span>{label}</span></button>)}</div>
        </div>
        <div className="moss-field">
          <span className="moss-field-label">统一类型</span>
          <div className="moss-segmented four">{MOSS_LAYOUT_LABELS.map((label) => <button key={label} type="button" className={`label-${label} ${chosen.every((item) => item.label === label) ? "active" : ""}`}
            onClick={() => commit({ ...layout, items: items.map((item) => selectedIds.includes(item.id) ? { ...item, label } : item) })}><i />{MOSS_LAYOUT_LABEL_NAMES[label]}</button>)}</div>
        </div>
        <div className="moss-inspector-actions">
          <button type="button" onClick={() => duplicateItems(selectedIds)}><CopyPlus size={13} />复制</button>
          <button type="button" className="danger" onClick={() => removeItems(selectedIds)}><Trash2 size={13} />删除</button>
        </div>
      </div>;
    }
    if (!primary) {
      return <div className="moss-inspector-empty">
        <SquarePlus size={22} />
        <strong>未选中元素</strong>
        <p>在画布空白处拖拽即可画框；已有框上画框请先按 B 切换到画框工具。点选一个框后在这里填写描述。</p>
        <div className="moss-add-row">{MOSS_LAYOUT_LABELS.map((label) => <button key={label} type="button" className={`label-${label}`} disabled={full} onClick={() => addItem(label)}><i />＋{MOSS_LAYOUT_LABEL_NAMES[label]}</button>)}</div>
      </div>;
    }
    const item = primary;
    const [x1, y1, x2, y2] = item.box;
    const setBox = (box) => patchItem(item.id, { box: clampBox(box, W, H) }, `box:${item.id}`);
    const centred = centredBox(item.box, W, H);
    return <div className="moss-inspector-body">
      <div className="moss-item-title">
        <b>#{numberOf(item.id)}</b>
        <strong>{MOSS_LAYOUT_LABEL_NAMES[item.label]}</strong>
        {item.hidden && <span className="moss-flag">已隐藏 · 不参与编译</span>}
        {item.locked && <span className="moss-flag">已锁定</span>}
      </div>
      <div className="moss-field">
        <span className="moss-field-label">类型</span>
        <div className="moss-segmented four">{MOSS_LAYOUT_LABELS.map((label) => <button key={label} type="button" className={`label-${label} ${item.label === label ? "active" : ""}`} onClick={() => { patchItem(item.id, { label }); setDrawLabel(label); }}><i />{MOSS_LAYOUT_LABEL_NAMES[label]}</button>)}</div>
      </div>
      <label className={`moss-field ${itemProblem(item, "caption") ? "invalid" : ""}`}>
        <span className="moss-field-label">描述 *<small>{item.caption.length} 字符</small></span>
        <textarea ref={captionRef} rows={4} value={item.caption} placeholder={item.label === "text" ? "文字的样式与位置，例如：深蓝色粗体大写标题，居中" : "这个区域里有什么、什么样子"} onChange={(event) => patchItem(item.id, { caption: event.target.value }, `caption:${item.id}`)} />
        {itemProblem(item, "caption") && <em>{itemProblem(item, "caption")}</em>}
      </label>
      {item.label === "text" && <label className={`moss-field ${itemProblem(item, "text") ? "invalid" : ""}`}>
        <span className="moss-field-label">渲染文字 *<small>按原样渲染，换行即换行</small></span>
        <textarea rows={3} value={item.text} placeholder="要出现在画面里的文字" onChange={(event) => patchItem(item.id, { text: event.target.value }, `text:${item.id}`)} />
        {itemProblem(item, "text") && <em>{itemProblem(item, "text")}</em>}
      </label>}
      <div className={`moss-field ${itemProblem(item, "bbox") ? "invalid" : ""}`}>
        <span className="moss-field-label">位置与尺寸<small>左上角像素 · {x2 - x1} × {y2 - y1}</small></span>
        <div className="moss-coords">
          <CoordField label="x1" value={x1} onCommit={(value) => setBox([value, y1, x2, y2])} />
          <CoordField label="y1" value={y1} onCommit={(value) => setBox([x1, value, x2, y2])} />
          <CoordField label="x2" value={x2} onCommit={(value) => setBox([x1, y1, value, y2])} />
          <CoordField label="y2" value={y2} onCommit={(value) => setBox([x1, y1, x2, value])} />
          <CoordField label="宽" value={x2 - x1} onCommit={(value) => setBox([x1, y1, x1 + value, y2])} />
          <CoordField label="高" value={y2 - y1} onCommit={(value) => setBox([x1, y1, x2, y1 + value])} />
        </div>
        <div className="moss-coord-actions">
          <button type="button" onClick={() => setBox([0, 0, W, H])}>铺满画布</button>
          <button type="button" onClick={() => setBox(moveBox(item.box, Math.round((W - (x2 - x1)) / 2) - x1, 0, W, H))}>水平居中</button>
          <button type="button" onClick={() => setBox(moveBox(item.box, 0, Math.round((H - (y2 - y1)) / 2) - y1, W, H))}>垂直居中</button>
        </div>
        <code className="moss-centred" title="MOSS 读取的格式：以画布中心为原点，[y1, x1, y2, x2]">官方坐标 yxyx · [{centred.join(", ")}]</code>
        {itemProblem(item, "bbox") && <em>{itemProblem(item, "bbox")}</em>}
      </div>
      <ChipField label="标签" hint="可选，补充关键词" values={item.tags} placeholder="输入后回车" onChange={(tags) => patchItem(item.id, { tags })} problem={itemProblem(item, "tags")} />
      <ChipField label="配色" hint="可选" values={item.palette} colour placeholder="#1E3A5F 或颜色名" onChange={(palette) => patchItem(item.id, { palette })} problem={itemProblem(item, "palette")} />
      <div className="moss-inspector-actions">
        <button type="button" onClick={() => duplicateItems([item.id])}><CopyPlus size={13} />复制</button>
        <button type="button" onClick={() => patchItem(item.id, { locked: !item.locked })}>{item.locked ? <LockOpen size={13} /> : <Lock size={13} />}{item.locked ? "解锁" : "锁定"}</button>
        <button type="button" onClick={() => patchItem(item.id, { hidden: !item.hidden })}>{item.hidden ? <Eye size={13} /> : <EyeOff size={13} />}{item.hidden ? "显示" : "隐藏"}</button>
        <button type="button" className="danger" disabled={item.locked} onClick={() => removeItems([item.id])}><Trash2 size={13} />删除</button>
      </div>
    </div>;
  };

  const sceneInspector = () => <div className="moss-inspector-body">
    <label className={`moss-field ${summaryProblem("caption") ? "invalid" : ""}`}>
      <span className="moss-field-label">全局画面描述 *<small>{layout.summary.caption.length} 字符</small></span>
      <textarea rows={6} value={layout.summary.caption} placeholder="整张图是什么：主题、风格、背景与氛围。例如：A minimalist travel poster with a cream background…" onChange={(event) => patchSummary({ caption: event.target.value }, "summary.caption")} />
      {summaryProblem("caption") && <em>{summaryProblem("caption")}</em>}
    </label>
    <ChipField label="画面类型 meta" hint="可选" values={layout.summary.meta} placeholder="输入后回车" suggestions={META_SUGGESTIONS} onChange={(meta) => patchSummary({ meta })} problem={summaryProblem("meta")} />
    <div className="moss-field">
      <span className="moss-field-label">画布类型 canvas<small>可选</small></span>
      <div className="moss-segmented four">
        {[...MOSS_LAYOUT_CANVAS_KINDS, ""].map((kind) => <button key={kind || "none"} type="button" className={layout.summary.canvas === kind ? "active" : ""} onClick={() => patchSummary({ canvas: kind })}>{kind ? MOSS_LAYOUT_CANVAS_NAMES[kind] : "不指定"}</button>)}
      </div>
    </div>
    <ChipField label="全局配色 palette" hint="可选" values={layout.summary.palette} colour placeholder="#F5EBDD 或颜色名" onChange={(palette) => patchSummary({ palette })} problem={summaryProblem("palette")} />
    <div className="moss-field">
      <span className="moss-field-label">画布</span>
      <p className="moss-inspector-note">{W} × {H} px，与基本页的画布尺寸一致；在基本页修改尺寸时，所有框按比例缩放。</p>
    </div>
  </div>;

  const jsonInspector = () => <div className="moss-inspector-body">
    <label className={`moss-field ${jsonError ? "invalid" : ""}`}>
      <span className="moss-field-label">可读草稿<small>左上角 xyxy 像素，可直接编辑</small></span>
      <textarea className="moss-json" rows={14} spellCheck={false} value={jsonEdit ?? draftText} onChange={(event) => { setJsonEdit(event.target.value); setJsonError(""); }} />
      {jsonError && <em>{jsonError}</em>}
      {hiddenCount > 0 && <small className="moss-inspector-note">{hiddenCount} 个隐藏元素不在草稿中；应用修改会移除它们。</small>}
    </label>
    <div className="moss-inspector-actions">
      <button type="button" className="primary" disabled={jsonEdit === null} onClick={() => {
        const result = parseMossLayoutImport(jsonEdit, W, H);
        if (result.error) { setJsonError(result.error); return; }
        const visible = items.filter((item) => !item.hidden);
        commit({ ...result.layout, enabled: true, items: result.layout.items.map((item, index) => ({ ...item, id: visible[index]?.id ?? item.id })) });
        setJsonEdit(null);
        setNotice(["已应用 JSON", result.note].filter(Boolean).join("；"));
      }}><Check size={13} />应用修改</button>
      <button type="button" disabled={jsonEdit === null} onClick={() => { setJsonEdit(null); setJsonError(""); }}>还原</button>
      <button type="button" onClick={() => copy(jsonEdit ?? draftText, "草稿 JSON")}><ClipboardCopy size={13} />复制</button>
    </div>
    <div className="moss-field">
      <span className="moss-field-label">校验<small>{issues.length ? `${issues.length} 处` : "通过"}</small></span>
      {issues.length ? <ul className="moss-issues">{issues.map((issue, index) => <li key={`${issue.path}-${index}`}>
        <button type="button" onClick={() => goToIssue(issue)}><code>{issue.itemId ? `#${numberOf(issue.itemId)} ${issue.path.replace(/^items\[\d+\]\.?/, "") || "元素"}` : issue.path || "草稿"}</code>{issue.message}</button>
      </li>)}</ul> : <p className="moss-inspector-note good"><Check size={12} />草稿符合 MOSS 官方格式</p>}
    </div>
    <div className="moss-field">
      <span className="moss-field-label">提交的提示词<small>前缀 + 官方 abs_centralized / yxyx JSON</small></span>
      <pre className="moss-compiled">{compiled.prompt ?? "（布局无效，修正后显示）"}</pre>
      <div className="moss-inspector-actions"><button type="button" disabled={!compiled.prompt} onClick={() => copy(compiled.prompt, "提示词")}><ClipboardCopy size={13} />复制提示词</button></div>
    </div>
  </div>;

  const toggleOverlay = (key) => setOverlays((current) => ({ ...current, [key]: !current[key] }));
  const menuButton = (id, Icon, label, disabled = false) => <button type="button" className={menu === id ? "active" : ""} aria-expanded={menu === id} disabled={disabled} onClick={() => setMenu((current) => current === id ? "" : id)}><Icon size={14} /><span>{label}</span></button>;

  return createPortal(<div className="moss-studio-backdrop" role="presentation">
    <div ref={studioRef} className="moss-studio" role="dialog" aria-modal="true" aria-label="MOSS 布局编辑器" tabIndex={-1}
      onPointerDownCapture={(event) => { if (menu && !event.target.closest(".moss-popover, [aria-expanded]")) setMenu(""); }}>
      <header className="moss-studio-head">
        <div className="moss-studio-title">
          <strong>MOSS 布局 · {W} × {H}</strong>
          <span>左上角像素坐标编辑 · 提交时编译为官方 abs_centralized / yxyx 提示词</span>
        </div>
        <div className="moss-studio-head-actions">
          {menuButton("templates", LayoutTemplate, "模板")}
          {menuButton("import", Upload, "导入")}
          {menuButton("export", Download, "导出")}
          <button type="button" className="moss-head-copy" disabled={!compiled.prompt} onClick={() => copy(compiled.prompt, "提示词")}><ClipboardCopy size={14} /><span>复制提示词</span></button>
          <button type="button" className="primary" onClick={onClose}><Check size={14} /><span>完成</span></button>
          <button type="button" className="moss-studio-close" onClick={onClose} aria-label="关闭布局编辑器"><X size={16} /></button>
        </div>
        {menu === "templates" && <div className="moss-popover moss-templates" role="menu">
          <header><strong>模板</strong><span>应用后替换当前布局，可撤销</span></header>
          {MOSS_LAYOUT_TEMPLATES.map((template) => {
            const preview = mossLayoutFromTemplate(template.id, W, H);
            return <button key={template.id} type="button" role="menuitem" onClick={() => applyTemplate(template)}>
              <MossLayoutMiniature layout={preview} className="template" />
              <span><strong>{template.name}</strong><small>{template.description}</small></span>
            </button>;
          })}
        </div>}
        {menu === "import" && <div className="moss-popover moss-import">
          <header><strong>导入布局</strong><span>可读草稿（abs / xyxy），或官方编译后的布局（abs_centralized / yxyx）</span></header>
          <textarea rows={8} spellCheck={false} value={importText} placeholder={'{"width":1024,"height":1024,"bbox_coord_space":"abs","bbox_format":"xyxy","summary":{…},"items":[…]}'} onChange={(event) => { setImportText(event.target.value); setImportError(""); }} />
          {importError && <em>{importError}</em>}
          <div className="moss-inspector-actions">
            <label className="moss-file-button"><FileJson size={13} />从文件读取<input type="file" accept=".json,.txt,application/json,text/plain" onChange={async (event) => {
              const file = event.target.files?.[0];
              event.target.value = "";
              if (file) setImportText(await file.text());
            }} /></label>
            <button type="button" className="primary" disabled={!importText.trim()} onClick={() => applyImport(importText)}><Check size={13} />导入并替换</button>
          </div>
        </div>}
        {menu === "export" && <div className="moss-popover moss-export" role="menu">
          <button type="button" role="menuitem" onClick={() => { copy(draftText, "草稿 JSON"); setMenu(""); }}><ClipboardCopy size={13} /><span><strong>复制可读草稿</strong><small>官方 compile_prompt.py 的输入格式</small></span></button>
          <button type="button" role="menuitem" onClick={() => { download(`moss-layout-${W}x${H}.json`, `${draftText}\n`, "application/json"); setMenu(""); }}><Download size={13} /><span><strong>下载草稿 .json</strong><small>可再次导入</small></span></button>
          <button type="button" role="menuitem" disabled={!compiled.prompt} onClick={() => { copy(compiled.prompt, "提示词"); setMenu(""); }}><ClipboardCopy size={13} /><span><strong>复制编译后的提示词</strong><small>含前缀，即提交给 MOSS 的文本</small></span></button>
          <button type="button" role="menuitem" disabled={!compiled.prompt} onClick={() => { download(`moss-layout-prompt-${W}x${H}.txt`, compiled.prompt, "text/plain"); setMenu(""); }}><Download size={13} /><span><strong>下载提示词 .txt</strong><small>UTF-8 纯文本</small></span></button>
        </div>}
      </header>

      <div className="moss-studio-toolbar" role="toolbar" aria-label="布局工具">
        <div className="moss-tool-group">
          <button type="button" aria-pressed={tool === "select"} title="选择与移动 (V)" onClick={() => setTool("select")}><MousePointer2 size={14} /><span>选择</span></button>
          <button type="button" aria-pressed={tool === "draw"} title="画框 (B)：可在已有框上开始" onClick={() => setTool("draw")}><SquarePlus size={14} /><span>画框</span></button>
          <button type="button" aria-pressed={tool === "pan"} title="平移 (H，或按住空格)" onClick={() => setTool("pan")}><Hand size={14} /><span>平移</span></button>
        </div>
        <div className="moss-label-picker" role="radiogroup" aria-label="新框类型">
          {MOSS_LAYOUT_LABELS.map((label) => <button key={label} type="button" role="radio" aria-checked={drawLabel === label} className={`label-${label}`} title={`新画的框为「${MOSS_LAYOUT_LABEL_NAMES[label]}」`} onClick={() => setDrawLabel(label)}><i />{MOSS_LAYOUT_LABEL_NAMES[label]}</button>)}
        </div>
        <div className="moss-tool-group">
          <button type="button" title="撤销 (Ctrl+Z)" disabled={!history.current.undo.length} onClick={undo}><Undo2 size={14} /></button>
          <button type="button" title="重做 (Ctrl+Shift+Z)" disabled={!history.current.redo.length} onClick={redo}><Redo2 size={14} /></button>
        </div>
        <div className="moss-tool-group moss-align-tools">
          {ALIGN_MODES.map(({ id, label, Icon }) => <button key={id} type="button" title={`${label}${selectedIds.length === 1 ? "（相对画布）" : ""}`} disabled={!selectedIds.length || ((id === "hspace" || id === "vspace") && selectedIds.length < 3)} onClick={() => align(id)}><Icon size={14} /></button>)}
        </div>
        <div className="moss-toggle-group">
          <label><input type="checkbox" checked={overlays.snap} onChange={() => toggleOverlay("snap")} />吸附</label>
          <label><input type="checkbox" checked={overlays.grid} onChange={() => toggleOverlay("grid")} />网格</label>
          <label><input type="checkbox" checked={overlays.thirds} onChange={() => toggleOverlay("thirds")} />三分线</label>
        </div>
        <div className="moss-tool-reference">
          {menuButton("reference", ImagePlus, reference ? "参考图 ✓" : "参考图")}
          {menu === "reference" && <div className="moss-popover moss-reference-panel">
            <header><strong>参考图</strong><span>只在编辑器里显示，不会上传或发送给模型</span></header>
            <label className="moss-file-button"><ImagePlus size={13} />{reference ? "更换图片" : "选择本地图片"}<input type="file" accept="image/*" onChange={(event) => {
              const file = event.target.files?.[0];
              event.target.value = "";
              if (file) setReference((current) => ({ url: URL.createObjectURL(file), name: file.name, opacity: current?.opacity ?? 0.45, visible: true }));
            }} /></label>
            {reference && <>
              <p className="moss-inspector-note">{reference.name}</p>
              <label className="moss-range">不透明度<input type="range" min={0.1} max={1} step={0.05} value={reference.opacity} onChange={(event) => setReference({ ...reference, opacity: Number(event.target.value) })} /><output>{Math.round(reference.opacity * 100)}%</output></label>
              <div className="moss-inspector-actions">
                <button type="button" onClick={() => setReference({ ...reference, visible: !reference.visible })}>{reference.visible ? <EyeOff size={13} /> : <Eye size={13} />}{reference.visible ? "隐藏" : "显示"}</button>
                <button type="button" className="danger" onClick={() => setReference(null)}><Trash2 size={13} />移除</button>
              </div>
            </>}
          </div>}
        </div>
        <div className="moss-tool-group moss-zoom">
          <button type="button" title="适应窗口" onClick={() => setView({ zoom: 1, x: 0, y: 0 })}><Maximize size={14} /></button>
          <button type="button" title="缩小" onClick={() => zoomBy(1 / 1.25)}><ZoomOut size={14} /></button>
          <button type="button" className="moss-zoom-value" title="切换到 100%（1 画布像素 = 1 屏幕像素）" onClick={() => zoomAt(1 / fit, viewport.width / 2, viewport.height / 2)}>{Math.round(scale * 100)}%</button>
          <button type="button" title="放大" onClick={() => zoomBy(1.25)}><ZoomIn size={14} /></button>
        </div>
      </div>

      <nav className="moss-studio-panes" aria-label="编辑器面板">
        {[["canvas", "画布"], ["list", `元素 ${items.length}`], ["inspector", "属性"]].map(([id, label]) => <button key={id} type="button" aria-pressed={pane === id} onClick={() => setPane(id)}>{label}</button>)}
      </nav>

      <div className="moss-studio-body" data-pane={pane}>
        <aside className="moss-studio-list" aria-label="元素列表">
          <header>
            <strong>元素 <b>{items.length}</b></strong>
            <span>列表顺序 = JSON 顺序 = 阅读顺序</span>
          </header>
          <div className="moss-label-counts">{MOSS_LAYOUT_LABELS.map((label) => <span key={label} className={`label-${label} ${counts[label] ? "" : "zero"}`}><i />{MOSS_LAYOUT_LABEL_NAMES[label]} {counts[label]}</span>)}</div>
          {items.length ? <ol>
            {items.map((item, index) => {
              const bad = issuesFor(item.id).length > 0;
              return <li key={item.id} draggable
                className={`${selectedIds.includes(item.id) ? "selected" : ""} ${item.hidden ? "hidden" : ""} ${dragRow?.over === index ? `drop-${dragRow.place}` : ""}`}
                onDragStart={(event) => { event.dataTransfer.effectAllowed = "move"; setDragRow({ id: item.id, over: null, place: "" }); }}
                onDragOver={(event) => {
                  if (!dragRow) return;
                  event.preventDefault();
                  const rect = event.currentTarget.getBoundingClientRect();
                  const place = event.clientY < rect.top + rect.height / 2 ? "before" : "after";
                  if (dragRow.over !== index || dragRow.place !== place) setDragRow({ ...dragRow, over: index, place });
                }}
                onDrop={(event) => {
                  event.preventDefault();
                  if (!dragRow) return;
                  const from = items.findIndex((entry) => entry.id === dragRow.id);
                  let to = dragRow.place === "after" ? index + 1 : index;
                  if (from < to) to -= 1;
                  commit(moveMossLayoutItem(layout, dragRow.id, to));
                  setDragRow(null);
                }}
                onDragEnd={() => setDragRow(null)}>
                <span className="moss-grip" aria-hidden="true">⋮⋮</span>
                <button type="button" className="moss-list-main" onClick={(event) => {
                  setSelected(event.shiftKey ? (selectedIds.includes(item.id) ? selectedIds.filter((id) => id !== item.id) : [...selectedIds, item.id]) : [item.id]);
                  setTab("item");
                }}>
                  <b>{index + 1}</b>
                  <i className={`moss-swatch label-${item.label}`} />
                  <span>
                    <strong>{MOSS_LAYOUT_LABEL_NAMES[item.label]}{bad && <TriangleAlert size={11} />}</strong>
                    <small className={item.caption.trim() ? "" : "missing"}>{item.label === "text" && item.text.trim() ? `“${excerpt(item.text, 24)}” · ` : ""}{excerpt(item.caption, 40) || "未填写描述"}</small>
                  </span>
                </button>
                <div className="moss-row-actions">
                  <button type="button" title={item.hidden ? "显示（参与编译）" : "隐藏（不参与编译）"} onClick={() => patchItem(item.id, { hidden: !item.hidden })}>{item.hidden ? <EyeOff size={13} /> : <Eye size={13} />}</button>
                  <button type="button" title={item.locked ? "解锁" : "锁定：画布上不可选中，可在其上直接画框"} onClick={() => patchItem(item.id, { locked: !item.locked })}>{item.locked ? <Lock size={13} /> : <LockOpen size={13} />}</button>
                  <span className="moss-row-more">
                    <button type="button" title="上移" disabled={index === 0} onClick={() => commit(moveMossLayoutItem(layout, item.id, index - 1))}><ArrowUp size={13} /></button>
                    <button type="button" title="下移" disabled={index === items.length - 1} onClick={() => commit(moveMossLayoutItem(layout, item.id, index + 1))}><ArrowDown size={13} /></button>
                    <button type="button" title="复制 (Ctrl+D)" disabled={full} onClick={() => duplicateItems([item.id])}><CopyPlus size={13} /></button>
                    <button type="button" title="删除 (Delete)" className="danger" disabled={item.locked} onClick={() => removeItems([item.id])}><Trash2 size={13} /></button>
                  </span>
                </div>
              </li>;
            })}
          </ol> : <p className="moss-list-empty">还没有框。在画布上拖拽画出第一个框，或用下方按钮添加。</p>}
          <footer className="moss-add-row">{MOSS_LAYOUT_LABELS.map((label) => <button key={label} type="button" className={`label-${label}`} disabled={full} onClick={() => addItem(label)}><i />＋{MOSS_LAYOUT_LABEL_NAMES[label]}</button>)}</footer>
        </aside>

        <section className={`moss-studio-canvas cursor-${canvasCursor}`} ref={viewportRef}
          onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp} onPointerCancel={cancelGesture}
          onPointerLeave={() => setCursor(null)} onContextMenu={(event) => event.preventDefault()}>
          <div ref={paperRef} className="moss-paper" style={{ width: paper.width, height: paper.height, left: paper.left, top: paper.top }}>
            <div className="moss-ruler top" aria-hidden="true">{ticks(W).map((value) => <span key={value} style={{ left: value * scale }}>{value}</span>)}</div>
            <div className="moss-ruler left" aria-hidden="true">{ticks(H).map((value) => <span key={value} style={{ top: value * scale }}>{value}</span>)}</div>
            {reference?.visible && <img className="moss-reference" src={reference.url} alt="" style={{ opacity: reference.opacity }} draggable={false} />}
            {overlays.grid && <div className="moss-grid" style={{ backgroundSize: `${gridSize * scale}px ${gridSize * scale}px, ${gridSize * scale}px ${gridSize * scale}px` }} />}
            {overlays.thirds && <div className="moss-thirds"><i /><i /><i /><i /></div>}
            <i className="moss-centre-mark" />
            {items.map((item, index) => {
              if (item.hidden) return null;
              const isSelected = selectedIds.includes(item.id);
              const isPrimary = primary?.id === item.id && selectedIds.length === 1;
              const bad = issuesFor(item.id).length > 0;
              const [x1, y1, x2, y2] = item.box;
              // Whole caption lines only (11px × 1.4 below the 21px tag strip): a box too short for one shows none.
              const lines = Math.min(4, Math.floor(((y2 - y1) * scale - 27) / CAPTION_LINE));
              return <div key={item.id} data-item={item.id}
                className={`moss-box label-${item.label} ${isSelected ? "selected" : ""} ${item.locked ? "locked" : ""} ${bad ? "invalid" : ""}`}
                style={{ left: x1 * scale, top: y1 * scale, width: (x2 - x1) * scale, height: (y2 - y1) * scale, zIndex: index + 1 }}>
                <span className="moss-box-tag"><b>{index + 1}</b><span>{MOSS_LAYOUT_LABEL_NAMES[item.label]}</span>{item.locked && <Lock size={9} />}{bad && <TriangleAlert size={9} />}</span>
                {(x2 - x1) * scale > 70 && lines >= 1 && <span className="moss-box-caption" style={{ WebkitLineClamp: lines, maxHeight: lines * CAPTION_LINE }}>
                  {item.label === "text" && item.text.trim() && <b>“{excerpt(item.text, 40)}”</b>}
                  {excerpt(item.caption, 90) || <em>未填写描述</em>}
                </span>}
                {isPrimary && !item.locked && MOSS_LAYOUT_HANDLES.map((handle) => <i key={handle} className={`moss-handle h-${handle}`} data-handle={handle} data-owner={item.id} />)}
                {isPrimary && <span className="moss-box-size">{x2 - x1} × {y2 - y1}</span>}
              </div>;
            })}
            {drawing && <div className={`moss-box drafting label-${drawLabel}`} style={{ left: drawing[0] * scale, top: drawing[1] * scale, width: (drawing[2] - drawing[0]) * scale, height: (drawing[3] - drawing[1]) * scale, zIndex: items.length + 2 }}>
              <span className="moss-box-size">{drawing[2] - drawing[0]} × {drawing[3] - drawing[1]}</span>
            </div>}
            {guides.map((guide, index) => <i key={`${guide.axis}-${guide.value}-${index}`} className={`moss-guide ${guide.axis}`} style={guide.axis === "x" ? { left: guide.value * scale } : { top: guide.value * scale }} />)}
          </div>
          {!items.length && <div className="moss-canvas-hint"><SquarePlus size={18} /><span>拖拽画出第一个框</span></div>}
          {notice && <button type="button" className="moss-studio-notice" onClick={() => setNotice("")}>{notice}</button>}
        </section>

        <aside className="moss-studio-inspector" aria-label="属性">
          <div className="moss-tabs" role="tablist">
            {[["scene", "画面"], ["item", selectedIds.length > 1 ? `元素 ×${selectedIds.length}` : "元素"], ["json", "JSON"]].map(([id, label]) => <button key={id} type="button" role="tab" aria-selected={tab === id} className={id === "scene" && summaryProblem("caption") ? "attention" : ""} onClick={() => setTab(id)}>{label}</button>)}
          </div>
          {tab === "scene" ? sceneInspector() : tab === "json" ? jsonInspector() : itemInspector()}
        </aside>
      </div>

      <footer className="moss-studio-status">
        <button type="button" className={`moss-status-validity ${issues.length ? "bad" : "good"}`} onClick={() => goToIssue(firstIssue)} disabled={!issues.length}>
          {issues.length ? <TriangleAlert size={13} /> : <Check size={13} />}{issues.length ? `${issues.length} 处待修正` : "布局有效"}
        </button>
        <MossTokenMeter tokens={tokens} compact />
        <span className="moss-status-prefix" title={prefix || "LoRA 组与已启用的正向预设会作为前缀行放在布局 JSON 之前"}>{prefix ? `前缀「${excerpt(prefix, 48)}」` : "无前缀"}</span>
        {stageSizes.length > 0 && <span title="Hires 重绘时按重绘画布重新编译布局，坐标按比例换算">Hires → {stageSizes.map((stage) => `${stage.width}×${stage.height}`).join("、")} 自动换算坐标</span>}
        {adetailerActive && <span title="ADetailer 局部重绘时只保留与检测区域相关的框，并换算到局部画布">ADetailer 按检测区域裁剪布局</span>}
        <span className="moss-status-cursor">{cursor ? `x ${cursor.x}  y ${cursor.y}` : ""}{primary && selectedIds.length === 1 ? `${cursor ? " · " : ""}#${numberOf(primary.id)} ${primary.box[2] - primary.box[0]}×${primary.box[3] - primary.box[1]}` : ""}</span>
      </footer>
    </div>
  </div>, document.body);
}
