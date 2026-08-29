import { useEffect, useRef, useState } from "react";
import {
  ArrowDown,
  ArrowLeft,
  ArrowUp,
  BookOpen,
  Check,
  ChevronRight,
  Copy,
  FolderPlus,
  ImageIcon,
  ImagePlus,
  Images,
  Layers3,
  Pencil,
  Plus,
  RefreshCw,
  Save,
  SlidersHorizontal,
  Sparkles,
  Trash2,
  Upload,
  WandSparkles,
  X,
  Zap,
} from "lucide-react";
import { SAMPLER_NAMES as SAMPLERS, SCHEDULER_NAMES as SCHEDULERS } from "./sampling-options";
import { normalizeHiresSeed, normalizeUint64Seed, secureRandomUint64Seed } from "./hires-settings";
import { DEFAULT_SETTINGS, GUIDANCE, clone, displayTitle, galleryRequest, normalizedSettings, useDialogLifecycle } from "./gallery-core";
import { composeGroupPrompt } from "./lora-groups";
import { formatWeight } from "./lora-weight";

const MAX_MANUAL_IMAGE_BYTES = 20 * 1024 * 1024;
const MAX_MANUAL_TOTAL_BYTES = 100 * 1024 * 1024;
// Prompt first: it is what a user almost always wants from a curated card, and
// it is the only group that cannot invalidate the current workspace by pulling
// in a model or LoRA set that is not installed.
const APPLY_DEFAULT_GROUPS = ["prompts"];
const APPLY_GROUPS = [
  ["prompts", "Prompt", "正向、负向与 ADetailer Prompt"],
  ["model", "模型组件", "模型引擎与完整模型组件"],
  ["loras", "LoRA 组合", "挂载顺序、权重与启用状态"],
  ["sampling", "采样参数", "引导增强、步数、CFG、降噪、Seed、采样器与批次"],
  ["canvas", "画布大小", "基础宽度与高度"],
  ["hires", "Hires.fix", "开关、模型、倍率与精修参数"],
  ["adetailer", "ADetailer", "开关、检测模型、蒙版与重绘参数"],
  ["rtx", "RTX VSR", "开关、倍率、质量与后处理顺序"],
  ["auxiliary", "其他选项", "过程预览与透明背景模型"],
];

function selectedImageHiresSeed(settings, imageIndex) {
  const mode = settings.imageHiresSeedModes?.[imageIndex];
  const seed = settings.imageHiresSeeds?.[imageIndex];
  return normalizeHiresSeed(mode ?? settings.hires.seedMode, seed ?? settings.hires.seed);
}

function applyCollectionCardOrder(cards, collectionId, cardIds) {
  const byId = new Map(cards.filter((card) => card.collection_id === collectionId).map((card) => [card.id, card]));
  const ordered = cardIds.map((cardId) => byId.get(cardId)).filter(Boolean);
  let index = 0;
  return cards.map((card) => card.collection_id === collectionId ? ordered[index++] || card : card);
}

function compactName(value) {
  return String(value || "未设置").replaceAll("\\", "/").split("/").pop();
}

function formatDate(value) {
  if (!value) return "--";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "--" : date.toLocaleString("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}

async function copyText(value) {
  if (!value) return false;
  try {
    await navigator.clipboard.writeText(value);
    return true;
  } catch {
    const textarea = document.createElement("textarea");
    textarea.value = value;
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();
    const copied = document.execCommand("copy");
    textarea.remove();
    return copied;
  }
}

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    if (!file.type.match(/^image\/(?:png|jpeg|webp|gif)$/)) {
      reject(new Error(`${file.name} 不是受支持的 PNG、JPEG、WebP 或 GIF`));
      return;
    }
    if (file.size > MAX_MANUAL_IMAGE_BYTES) {
      reject(new Error(`${file.name} 超过 20 MiB`));
      return;
    }
    const reader = new FileReader();
    reader.addEventListener("load", () => resolve({ dataUrl: reader.result, name: file.name }));
    reader.addEventListener("error", () => reject(new Error(`无法读取 ${file.name}`)));
    reader.readAsDataURL(file);
  });
}

async function filesToNewImages(fileList) {
  const files = [...fileList];
  if (!files.length) throw new Error("没有检测到可添加的图片文件");
  if (files.reduce((total, file) => total + file.size, 0) > MAX_MANUAL_TOTAL_BYTES) {
    throw new Error("单次添加图片总大小不能超过 100 MiB");
  }
  const imported = [];
  for (const file of files) imported.push(await fileToDataUrl(file));
  return imported.map((item) => ({ kind: "new", dataUrl: item.dataUrl, name: item.name, url: item.dataUrl }));
}

function NumericField({ label, value, onChange, min, max, step = 1, disabled = false }) {
  return <label className="gallery-field"><span>{label}</span><input type="number" value={value} min={min} max={max} step={step} disabled={disabled} onChange={(event) => {
    const next = Number(event.target.value);
    if (Number.isFinite(next)) onChange(Math.max(min ?? next, Math.min(max ?? next, next)));
  }} /></label>;
}

function ToggleField({ label, detail, checked, onChange, disabled = false }) {
  const locked = disabled && !checked;
  return <label className={`gallery-toggle-field ${checked ? "active" : ""} ${disabled ? "disabled" : ""}`}><span><strong>{label}</strong><small>{detail}</small></span><input type="checkbox" checked={checked} disabled={locked} onChange={(event) => { if (!disabled || !event.target.checked) onChange(event.target.checked); }} /><i /></label>;
}

/**
 * Stored gallery assets are full-resolution originals — a 2048x2944 PNG is six
 * megapixels. Anything rendered small asks for the cached derivative instead;
 * `thumb_url` is absent on records written before thumbnails existed, and on
 * files the user has only just picked, so both degrade to the original.
 */
function thumbUrl(image) {
  return image?.thumb_url || image?.url || "";
}

/**
 * The opened picture, painted progressively. The thumbnail is already in cache
 * from the grid tile, so it appears on the first frame and the original swaps in
 * behind it once it has decoded off the main thread. Without this the stage was
 * empty for as long as a five-megabyte PNG took to arrive and decode, which is
 * what "it freezes for a few seconds" actually was.
 */
function FocusImage({ image, alt }) {
  const [loaded, setLoaded] = useState(false);
  const preview = thumbUrl(image);
  const hasPreview = preview && preview !== image.url;
  useEffect(() => setLoaded(false), [image.url]);
  return (
    <div className={`gallery-focus-frame ${loaded ? "ready" : ""}`}>
      {hasPreview && !loaded && <img className="gallery-focus-preview" src={preview} alt="" aria-hidden="true" decoding="async" />}
      <img
        className="gallery-focus-full"
        src={image.url}
        alt={alt}
        decoding="async"
        fetchPriority="high"
        onLoad={() => setLoaded(true)}
        onError={() => setLoaded(true)}
      />
    </div>
  );
}

/** The prompt as submitted: the enabled groups' prefix ahead of the user's text. */
function effectivePrompt(settings) {
  return settings.loraGroupPrompt
    ? composeGroupPrompt([{ id: "recorded", name: "", enabled: true, presetPrompt: settings.loraGroupPrompt, members: [] }], settings.positive)
    : settings.positive;
}

function PromptBlock({ tone, label, value, onNotice }) {
  return <section className={`gallery-prompt ${tone}`}>
    <header><span>{label}</span><button type="button" disabled={!value} onClick={async () => onNotice(await copyText(value) ? `${label}已复制` : "复制失败")}><Copy size={13} />复制</button></header>
    <p>{value || "未填写"}</p>
  </section>;
}

function PromptLibraryDialog({ entry, onClose, onSaved }) {
  const dialogRef = useDialogLifecycle(true, onClose);
  const [title, setTitle] = useState(entry?.title || "");
  const [positivePrompt, setPositivePrompt] = useState(entry?.positive_prompt || "");
  const [negativePrompt, setNegativePrompt] = useState(entry?.negative_prompt || "");
  const [notes, setNotes] = useState(entry?.notes || "");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const save = async (event) => {
    event.preventDefault();
    if (!title.trim() || (!positivePrompt.trim() && !negativePrompt.trim()) || busy) return;
    setBusy(true);
    setError("");
    try {
      const payload = await galleryRequest(entry ? `/prompts/${entry.id}` : "/prompts", {
        method: entry ? "PATCH" : "POST",
        body: JSON.stringify({
          title: title.trim(),
          positive_prompt: positivePrompt,
          negative_prompt: negativePrompt,
          notes: notes.trim() || null,
        }),
      });
      onSaved(payload);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  };
  return <div className="gallery-dialog-backdrop" onMouseDown={(event) => event.target === event.currentTarget && !busy && onClose()}>
    <form ref={dialogRef} className="gallery-prompt-dialog" role="dialog" aria-modal="true" aria-labelledby="gallery-prompt-dialog-title" tabIndex="-1" onSubmit={save}>
      <header><div><span>PROMPT LIBRARY</span><h2 id="gallery-prompt-dialog-title">{entry ? "编辑词条" : "添加词条"}</h2><p>保存一组可复用的正向与负向 Prompt。</p></div><button type="button" aria-label="关闭词条编辑器" disabled={busy} onClick={onClose}><X size={18} /></button></header>
      <fieldset disabled={busy}>
        <label className="gallery-field"><span>标题</span><input autoFocus value={title} maxLength={160} onChange={(event) => setTitle(event.target.value)} placeholder="例如：雨夜霓虹街景" /></label>
        <label className="gallery-field prompt-positive"><span>正向 Prompt</span><textarea value={positivePrompt} maxLength={8000} onChange={(event) => setPositivePrompt(event.target.value)} placeholder="主体、风格、构图、光线等" /></label>
        <label className="gallery-field prompt-negative"><span>负向 Prompt</span><textarea value={negativePrompt} maxLength={8000} onChange={(event) => setNegativePrompt(event.target.value)} placeholder="不希望出现的内容" /></label>
        <label className="gallery-field"><span>备注（可选）</span><textarea className="gallery-prompt-notes" value={notes} maxLength={2000} onChange={(event) => setNotes(event.target.value)} placeholder="用途、模型偏好或使用说明" /></label>
        {error && <p className="gallery-form-error">{error}</p>}
      </fieldset>
      <footer><span>{positivePrompt.length + negativePrompt.length} 个 Prompt 字符</span><div><button type="button" disabled={busy} onClick={onClose}>取消</button><button type="submit" className="primary" disabled={busy || !title.trim() || (!positivePrompt.trim() && !negativePrompt.trim())}>{busy ? <RefreshCw className="spin" size={14} /> : <Save size={14} />}保存词条</button></div></footer>
    </form>
  </div>;
}

function PromptLibrary({ entries, loading, error, onRefresh, onAdd, onEdit, onDelete, onApply, onNotice }) {
  const [search, setSearch] = useState("");
  const query = search.trim().toLocaleLowerCase();
  const visible = entries.filter((entry) => !query || [entry.title, entry.positive_prompt, entry.negative_prompt, entry.notes].some((value) => String(value || "").toLocaleLowerCase().includes(query)));
  const copyPrompt = async (label, value) => onNotice(await copyText(value) ? `${label}已复制` : "复制失败");
  return <section className="gallery-prompt-library">
    <header className="gallery-hero"><div><span>PROMPT LIBRARY / LOCAL</span><h2>词库</h2><p>保存常用提示词组合；每个词条包含标题、正向 Prompt、负向 Prompt 与备注。</p></div><div><button type="button" onClick={onAdd}><Plus size={15} />添加词条</button><button type="button" onClick={onRefresh} disabled={loading}><RefreshCw className={loading ? "spin" : ""} size={15} />刷新</button></div></header>
    <div className="gallery-prompt-scroll">
      <label className="gallery-prompt-search"><BookOpen size={16} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="搜索标题、Prompt 或备注" /><b>{visible.length} / {entries.length}</b></label>
      {error && <div className="gallery-page-error"><X size={22} /><strong>无法读取词库</strong><p>{error}</p></div>}
      {!error && loading && <div className="gallery-loading"><RefreshCw className="spin" size={25} /><span>正在读取词库</span></div>}
      {!error && !loading && entries.length === 0 && <div className="gallery-first-empty"><span><BookOpen size={33} /></span><strong>添加第一个 Prompt 词条</strong><p>为常用提示词设置标题，保存正向、负向 Prompt 与使用备注。</p><button type="button" onClick={onAdd}><Plus size={15} />添加词条</button></div>}
      {!error && !loading && entries.length > 0 && visible.length === 0 && <div className="gallery-prompt-empty">没有匹配“{search.trim()}”的词条</div>}
      {!error && !loading && visible.length > 0 && <div className="gallery-prompt-grid">{visible.map((entry) => <article className="gallery-prompt-card" key={entry.id}>
        <header><div><span>PROMPT ENTRY</span><h3>{entry.title}</h3><small>更新于 {formatDate(entry.updated_at)}</small></div><div><button type="button" title="编辑词条" aria-label={`编辑词条 ${entry.title}`} onClick={() => onEdit(entry)}><Pencil size={14} /></button><button type="button" className="danger" title="删除词条" aria-label={`删除词条 ${entry.title}`} onClick={() => onDelete(entry)}><Trash2 size={14} /></button></div></header>
        <div className="gallery-prompt-card-body">
          <section className="positive"><header><span>正向 Prompt</span><button type="button" disabled={!entry.positive_prompt} onClick={() => void copyPrompt("正向 Prompt", entry.positive_prompt)}><Copy size={12} />复制</button></header><p>{entry.positive_prompt || "未填写"}</p></section>
          <section className="negative"><header><span>负向 Prompt</span><button type="button" disabled={!entry.negative_prompt} onClick={() => void copyPrompt("负向 Prompt", entry.negative_prompt)}><Copy size={12} />复制</button></header><p>{entry.negative_prompt || "未填写"}</p></section>
          {entry.notes && <p className="gallery-prompt-card-notes">{entry.notes}</p>}
        </div>
        <footer><button type="button" onClick={() => onApply(entry, "generate")}><WandSparkles size={13} />应用到文生图</button><button type="button" onClick={() => onApply(entry, "image")}><ImagePlus size={13} />应用到图生图</button></footer>
      </article>)}</div>}
    </div>
  </section>;
}

function GalleryCardTile({ card, tileIndex = 0, onOpen, onMenu, dragState, reorderBusy, onReorderStart, onReorderOver, onReorderEnd }) {
  const settings = normalizedSettings(card.settings);
  const first = card.images?.[0];
  const gesture = useRef(null);
  const suppressClick = useRef(false);
  const [holding, setHolding] = useState(false);
  const isDragging = dragState?.cardId === card.id;
  const isDropTarget = dragState?.targetId === card.id && !isDragging;
  useEffect(() => () => {
    if (gesture.current?.timer) window.clearTimeout(gesture.current.timer);
  }, []);
  const clearGesture = (event, finishDrag = false) => {
    const current = gesture.current;
    if (!current || event.pointerId !== current.pointerId) return;
    if (current.timer) window.clearTimeout(current.timer);
    setHolding(false);
    if (current.node.hasPointerCapture(current.pointerId)) current.node.releasePointerCapture(current.pointerId);
    gesture.current = null;
    if (current.active && finishDrag) {
      event.preventDefault();
      suppressClick.current = true;
      onReorderEnd(card);
    }
    window.setTimeout(() => { suppressClick.current = false; }, 120);
  };
  const pointerDown = (event) => {
    if (event.button !== 0 || !event.isPrimary || reorderBusy) return;
    const node = event.currentTarget;
    const pointerId = event.pointerId;
    node.setPointerCapture(pointerId);
    setHolding(true);
    const next = { node, pointerId, startX: event.clientX, startY: event.clientY, x: event.clientX, y: event.clientY, active: false, timer: 0 };
    next.timer = window.setTimeout(() => {
      if (gesture.current !== next) return;
      suppressClick.current = true;
      next.active = onReorderStart(card, { x: next.x, y: next.y }) !== false;
      setHolding(false);
    }, 360);
    gesture.current = next;
  };
  const pointerMove = (event) => {
    const current = gesture.current;
    if (!current || event.pointerId !== current.pointerId) return;
    current.x = event.clientX;
    current.y = event.clientY;
    if (!current.active) {
      if (Math.hypot(event.clientX - current.startX, event.clientY - current.startY) > 8) {
        window.clearTimeout(current.timer);
        current.timer = 0;
        suppressClick.current = true;
        setHolding(false);
      }
      return;
    }
    event.preventDefault();
    const target = document.elementFromPoint(event.clientX, event.clientY)?.closest("[data-gallery-card-id]");
    if (target?.dataset.galleryCollectionId === card.collection_id) {
      onReorderOver(card, target.dataset.galleryCardId, { x: event.clientX, y: event.clientY });
    } else {
      onReorderOver(card, null, { x: event.clientX, y: event.clientY });
    }
    const scroll = current.node.closest(".gallery-scroll");
    if (scroll) {
      const bounds = scroll.getBoundingClientRect();
      if (event.clientY < bounds.top + 56) scroll.scrollBy({ top: -14 });
      else if (event.clientY > bounds.bottom - 56) scroll.scrollBy({ top: 14 });
    }
  };
  const contextMenu = (event) => {
    if (event.button !== 2) return;
    event.preventDefault();
    if (gesture.current?.timer) window.clearTimeout(gesture.current.timer);
    gesture.current = null;
    setHolding(false);
    if (!reorderBusy && !dragState) onMenu(card, event);
  };
  return <button
    type="button"
    className={`gallery-card ${holding ? "holding" : ""} ${isDragging ? "dragging" : ""} ${isDropTarget ? "drop-target" : ""} ${reorderBusy ? "reorder-busy" : ""}`}
    data-gallery-card-id={card.id}
    data-gallery-collection-id={card.collection_id}
    style={{ "--gallery-card-index": Math.min(tileIndex, 12) }}
    aria-grabbed={isDragging}
    title="单击查看；按住左键拖拽排序；右键删除"
    onPointerDown={pointerDown}
    onPointerMove={pointerMove}
    onPointerUp={(event) => clearGesture(event, true)}
    onPointerCancel={(event) => clearGesture(event, true)}
    onContextMenu={contextMenu}
    onClick={(event) => { if (suppressClick.current || isDragging) { event.preventDefault(); return; } onOpen(card); }}
  >
    <span className="gallery-card-visual">
      {first ? <img src={thumbUrl(first)} alt="" loading="lazy" decoding="async" /> : <span className="gallery-card-empty"><ImageIcon size={25} /><b>{displayTitle(card)}</b></span>}
      {card.image_count > 1 && <b className="gallery-image-count"><Images size={12} />{card.image_count}</b>}
      <i>{settings.model || "--"}</i>
    </span>
    <span className="gallery-card-copy">
      <strong>{displayTitle(card)}</strong>
      <small>{compactName(settings.model === "Anima" ? settings.diffusionModel : settings.checkpoint)} · {settings.size?.width || "--"} × {settings.size?.height || "--"}</small>
      <span><b>{settings.steps || "--"} STEP</b><b>CFG {settings.cfg ?? "--"}</b><b>{settings.loras?.filter((item) => item.enabled !== false).length || 0} LoRA</b></span>
    </span>
  </button>;
}

function GalleryCardMenu({ menu, onClose, onEdit, onDelete }) {
  const menuRef = useRef(null);
  useEffect(() => {
    const closeOutside = (event) => {
      if (!menuRef.current?.contains(event.target)) onClose();
    };
    const keyDown = (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
      }
    };
    const closeForViewportChange = () => onClose();
    window.addEventListener("pointerdown", closeOutside);
    window.addEventListener("keydown", keyDown);
    window.addEventListener("scroll", closeForViewportChange, true);
    window.addEventListener("resize", closeForViewportChange);
    window.requestAnimationFrame(() => menuRef.current?.querySelector("button")?.focus());
    return () => {
      window.removeEventListener("pointerdown", closeOutside);
      window.removeEventListener("keydown", keyDown);
      window.removeEventListener("scroll", closeForViewportChange, true);
      window.removeEventListener("resize", closeForViewportChange);
    };
  }, [onClose]);
  return <div ref={menuRef} className="gallery-card-menu" role="menu" aria-label={`卡片操作：${displayTitle(menu.card, 42)}`} style={{ left: menu.x, top: menu.y }}>
    <header><span>CARD ACTIONS</span><small>{displayTitle(menu.card, 42)}</small></header>
    <button type="button" role="menuitem" onClick={() => onEdit(menu.card)}><Pencil size={15} /><span><strong>编辑卡片</strong><small>修改图片与生成参数</small></span></button>
    <button type="button" role="menuitem" className="danger" onClick={() => onDelete(menu.card)}><Trash2 size={15} /><span><strong>删除卡片</strong><small>删除独立精选副本</small></span></button>
  </div>;
}

function AddCardTile({ collectionId, tileIndex = 0, onAdd, onDropError }) {
  const dragDepth = useRef(0);
  const [draggingFiles, setDraggingFiles] = useState(false);
  const [readingFiles, setReadingFiles] = useState(false);
  const acceptsFiles = (event) => [...(event.dataTransfer?.types || [])].includes("Files");
  const dragEnter = (event) => {
    if (!acceptsFiles(event) || readingFiles) return;
    event.preventDefault();
    dragDepth.current += 1;
    setDraggingFiles(true);
  };
  const dragOver = (event) => {
    if (!acceptsFiles(event) || readingFiles) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
  };
  const dragLeave = (event) => {
    if (!acceptsFiles(event)) return;
    event.preventDefault();
    dragDepth.current = Math.max(0, dragDepth.current - 1);
    if (!dragDepth.current) setDraggingFiles(false);
  };
  const drop = async (event) => {
    if (!acceptsFiles(event) || readingFiles) return;
    event.preventDefault();
    event.stopPropagation();
    dragDepth.current = 0;
    setDraggingFiles(false);
    setReadingFiles(true);
    try {
      const images = await filesToNewImages(event.dataTransfer.files);
      onAdd(collectionId, images);
    } catch (error) {
      onDropError(error.message);
    } finally {
      setReadingFiles(false);
    }
  };
  return <button
    type="button"
    className={`gallery-add-card ${draggingFiles ? "drop-active" : ""} ${readingFiles ? "drop-reading" : ""}`}
    style={{ "--gallery-card-index": Math.min(tileIndex, 12) }}
    aria-label="添加卡片；也可以拖入本地图片"
    aria-busy={readingFiles}
    onClick={() => !readingFiles && onAdd(collectionId, [])}
    onDragEnter={dragEnter}
    onDragOver={dragOver}
    onDragLeave={dragLeave}
    onDrop={(event) => void drop(event)}
  ><span>{readingFiles ? <RefreshCw className="spin" size={24} /> : draggingFiles ? <Upload size={24} /> : <Plus size={24} />}</span><strong>{readingFiles ? "正在读取图片" : draggingFiles ? "释放以创建卡片" : "添加卡片"}</strong><small>{readingFiles ? "校验格式与文件大小" : draggingFiles ? "PNG · JPEG · WebP · GIF" : "点击添加 · 或拖入本地图片"}</small></button>;
}

function CollectionDialog({ collection, onClose, onSaved, onDeleted }) {
  const dialogRef = useDialogLifecycle(true, onClose);
  const [id, setId] = useState(collection?.id || "");
  const [description, setDescription] = useState(collection?.description || "");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const save = async (event) => {
    event.preventDefault();
    if (!id.trim() || busy) return;
    setBusy(true);
    setError("");
    try {
      const result = await galleryRequest(collection ? `/collections/${encodeURIComponent(collection.id)}` : "/collections", {
        method: collection ? "PATCH" : "POST",
        body: JSON.stringify({ id: id.trim(), description: description.trim() || null }),
      });
      onSaved(result);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  };
  const remove = async () => {
    if (!collection || busy || !window.confirm(`删除收藏夹“${collection.id}”及其中全部精选卡片？原始生成文件不会被删除。`)) return;
    setBusy(true);
    setError("");
    try {
      await galleryRequest(`/collections/${encodeURIComponent(collection.id)}`, { method: "DELETE" });
      onDeleted(collection.id);
    } catch (requestError) {
      setError(requestError.message);
      setBusy(false);
    }
  };
  return <div className="gallery-dialog-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
    <form ref={dialogRef} className="gallery-small-dialog" role="dialog" aria-modal="true" aria-labelledby="gallery-collection-dialog-title" tabIndex="-1" onSubmit={save}>
      <header><div><span>COLLECTION</span><h2 id="gallery-collection-dialog-title">{collection ? "编辑收藏夹" : "创建收藏夹"}</h2></div><button type="button" aria-label="关闭收藏夹编辑" onClick={onClose}><X size={18} /></button></header>
      <div className="gallery-dialog-body">
        <label className="gallery-field"><span>收藏夹 ID</span><input value={id} maxLength={64} autoFocus onChange={(event) => setId(event.target.value)} placeholder="例如 portraits-2026" /></label>
        <label className="gallery-field"><span>简介（可选）</span><textarea value={description} maxLength={1000} onChange={(event) => setDescription(event.target.value)} placeholder="记录主题、用途或筛选说明" /></label>
        {error && <p className="gallery-form-error">{error}</p>}
      </div>
      <footer>{collection ? <button type="button" className="danger" disabled={busy} onClick={remove}><Trash2 size={14} />删除</button> : <span />}<button type="submit" className="primary" disabled={busy || !id.trim()}>{busy ? <RefreshCw className="spin" size={14} /> : <Save size={14} />}{collection ? "保存修改" : "创建收藏夹"}</button></footer>
    </form>
  </div>;
}

function GalleryLoraManager({ engine, value, onClose, onApply }) {
  const dialogRef = useDialogLifecycle(true, onClose);
  const [mounted, setMounted] = useState(() => clone(value || []).slice(0, 16));
  const [categories, setCategories] = useState([]);
  const [category, setCategory] = useState("mounted");
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    fetch(`/api/loras?engine=${encodeURIComponent(engine)}`, { signal: controller.signal, cache: "no-store" })
      .then(async (response) => {
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "无法读取 LoRA 目录");
        return payload;
      })
      .then((payload) => { setCategories(payload.categories || []); setError(""); })
      .catch((fetchError) => fetchError.name !== "AbortError" && setError(fetchError.message))
      .finally(() => !controller.signal.aborted && setLoading(false));
    return () => controller.abort();
  }, [engine]);
  const active = categories.find((item) => item.id === category);
  const available = (active?.models || []).filter((item) => item.name.toLowerCase().includes(search.trim().toLowerCase()));
  const toggle = (item, label) => setMounted((current) => current.some((entry) => entry.value === item.value)
    ? current.filter((entry) => entry.value !== item.value)
    : [...current, { value: item.value, name: item.name, category: label, weight: 1, precision: 1, enabled: true }].slice(0, 16));
  return <div className="gallery-lora-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
    <section ref={dialogRef} className="gallery-lora-manager" role="dialog" aria-modal="true" aria-label="画廊卡片 LoRA 管理器" tabIndex="-1">
      <header><div><span>LORA CONFIGURATOR</span><h2>LoRA 组合编辑</h2><p>{engine} 引擎 · {mounted.length} / 16</p></div><button type="button" aria-label="关闭 LoRA 管理器" onClick={onClose}><X size={19} /></button></header>
      <div className="gallery-lora-body">
        <nav><button type="button" className={category === "mounted" ? "active" : ""} onClick={() => setCategory("mounted")}><span>已挂载</span><b>{mounted.length}</b></button>{categories.map((item) => <button type="button" className={category === item.id ? "active" : ""} key={item.id} onClick={() => setCategory(item.id)}><span>{item.label}</span><b>{item.models.length}</b></button>)}</nav>
        <main>
          {category === "mounted" ? <div className="gallery-mounted-loras">
            {mounted.length === 0 && <div className="gallery-lora-empty"><Layers3 size={28} /><span>从分类中挂载 LoRA</span></div>}
            {mounted.map((item, index) => <div className={item.enabled === false ? "disabled" : ""} key={item.value}>
              <button type="button" className={`mini-switch ${item.enabled !== false ? "active" : ""}`} role="switch" aria-label={`${item.enabled === false ? "启用" : "停用"} ${item.name}`} aria-checked={item.enabled !== false} onClick={() => setMounted((current) => current.map((entry) => entry.value === item.value ? { ...entry, enabled: entry.enabled === false } : entry))}><i /></button>
              <span><strong>{item.name}</strong><small>{item.category} · #{index + 1}</small></span>
              <input type="number" min="-5" max="5" step="0.01" value={item.weight} onChange={(event) => setMounted((current) => current.map((entry) => entry.value === item.value ? { ...entry, weight: Math.max(-5, Math.min(5, Number(event.target.value) || 0)), precision: 2 } : entry))} />
              <button type="button" aria-label={`移除 ${item.name}`} onClick={() => setMounted((current) => current.filter((entry) => entry.value !== item.value))}><Trash2 size={13} /></button>
            </div>)}
          </div> : <>
            <label className="gallery-lora-search"><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder={`搜索${active?.label || " LoRA"}`} /></label>
            <div className="gallery-lora-library">{available.map((item) => { const selected = mounted.some((entry) => entry.value === item.value); return <button type="button" className={selected ? "selected" : ""} key={item.value} onClick={() => toggle(item, active?.label || "other")}><span><strong>{item.name}</strong><small>{(item.size / 1024 / 1024).toFixed(1)} MB</small></span>{selected ? <Check size={15} /> : <Plus size={15} />}</button>; })}</div>
          </>}
          {loading && <div className="gallery-lora-empty"><RefreshCw className="spin" size={24} /><span>正在扫描 LoRA</span></div>}
          {error && <p className="gallery-form-error">{error}</p>}
        </main>
      </div>
      <footer><span>修改仅写入当前精选卡片，不会改变生图区域</span><button type="button" onClick={() => onApply(mounted)}>应用 LoRA 组合</button></footer>
    </section>
  </div>;
}

function GalleryCardEditor({ card, collectionId, collections, initialSettings, initialImages = [], onClose, onSaved }) {
  const [selectedCollection, setSelectedCollection] = useState(card?.collection_id || collectionId || collections[0]?.id || "");
  const [title, setTitle] = useState(card?.title || "");
  const [settings, setSettings] = useState(() => card
    ? normalizedSettings(card.settings, DEFAULT_SETTINGS, { hiresSourceKind: "persisted_card" })
    : normalizedSettings(undefined, initialSettings, { hiresSourceKind: "workspace_inheritance" }));
  const [images, setImages] = useState(() => card
    ? (card.images || []).map((image, index) => {
      const cardSettings = normalizedSettings(card.settings, DEFAULT_SETTINGS, { hiresSourceKind: "persisted_card" });
      const imageHires = selectedImageHiresSeed(cardSettings, index);
      const hasImageHires = Array.isArray(card.settings?.imageHiresSeedModes) && index < card.settings.imageHiresSeedModes.length;
      return { kind: "existing", id: image.id, name: image.name, url: image.url, seed: card.settings?.imageSeeds?.[index] || "", hiresSeedMode: hasImageHires ? imageHires.seedMode : undefined, hiresSeed: hasImageHires ? imageHires.seed : undefined };
    })
    : initialImages);
  const [checkpoints, setCheckpoints] = useState([]);
  const [splitAssets, setSplitAssets] = useState({ diffusion_model: [], text_encoder: [], vae: [] });
  const [loraOpen, setLoraOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [modelCatalogError, setModelCatalogError] = useState("");
  const dialogRef = useDialogLifecycle(!loraOpen, onClose);
  const fileInput = useRef(null);
  const setField = (key, value) => setSettings((current) => busy ? current : ({ ...current, [key]: value }));
  const setNested = (group, key, value) => setSettings((current) => busy ? current : ({ ...current, [group]: { ...current[group], [key]: value } }));
  const isAnima = settings.model === "Anima";
  const visibleSamplers = SAMPLERS;
  const visibleSchedulers = SCHEDULERS;
  const changeEngine = (nextModel) => setSettings((current) => ({
    ...current,
    model: nextModel,
    checkpoint: "",
    diffusionModel: "",
    textEncoder: "",
    vae: "",
    loras: [],
    sampler: SAMPLERS.includes(current.sampler) ? current.sampler : nextModel === "Anima" ? "euler" : "dpmpp_2m",
    scheduler: SCHEDULERS.includes(current.scheduler) ? current.scheduler : nextModel === "Anima" ? "simple" : "karras",
    guidance: nextModel !== "Anima" && current.guidance === "cfg_zero_star" ? "none" : current.guidance,
    processPreview: nextModel === "Anima" ? false : current.processPreview,
  }));
  useEffect(() => {
    const controller = new AbortController();
    fetch(`/api/models?engine=${encodeURIComponent(settings.model)}`, { signal: controller.signal, cache: "no-store" })
      .then(async (response) => response.ok ? response.json() : Promise.reject(new Error("模型目录不可用")))
      .then((payload) => {
        setModelCatalogError("");
        if (payload.model_type === "split") {
          setCheckpoints([]);
          setSplitAssets({
            diffusion_model: payload.assets?.diffusion_model?.models || [],
            text_encoder: payload.assets?.text_encoder?.models || [],
            vae: payload.assets?.vae?.models || [],
          });
        } else {
          setCheckpoints(payload.models || []);
          setSplitAssets({ diffusion_model: [], text_encoder: [], vae: [] });
        }
      })
      .catch((fetchError) => {
        if (fetchError.name === "AbortError") return;
        setCheckpoints([]);
        setSplitAssets({ diffusion_model: [], text_encoder: [], vae: [] });
        setModelCatalogError(fetchError.message || "模型目录不可用");
      });
    return () => controller.abort();
  }, [settings.model]);
  const addFiles = async (fileList) => {
    if (busy) return;
    setError("");
    try {
      const imported = await filesToNewImages(fileList);
      setImages((current) => [...current, ...imported]);
    } catch (readError) {
      setError(readError.message);
    }
  };
  const moveImage = (index, direction) => setImages((current) => {
    if (busy) return current;
    const target = index + direction;
    if (target < 0 || target >= current.length) return current;
    const next = [...current];
    [next[index], next[target]] = [next[target], next[index]];
    return next;
  });
  const moveStage = (stage, direction) => setSettings((current) => {
    if (busy) return current;
    const order = [...current.postprocessOrder];
    const index = order.indexOf(stage);
    const target = index + direction;
    if (index < 0 || target < 0 || target >= order.length) return current;
    [order[index], order[target]] = [order[target], order[index]];
    return { ...current, postprocessOrder: order };
  });
  const save = async (event) => {
    event.preventDefault();
    if (!selectedCollection || busy) return;
    if (modelCatalogError) {
      setError(`模型目录不可用：${modelCatalogError}`);
      return;
    }
    if (isAnima) {
      const required = [
        ["diffusionModel", "扩散模型", splitAssets.diffusion_model],
        ["textEncoder", "文本编码器", splitAssets.text_encoder],
        ["vae", "VAE", splitAssets.vae],
      ];
      const unavailable = required.find(([field, , catalog]) => !settings[field] || !catalog.some((item) => item.value === settings[field]));
      if (unavailable) {
        setError(`请选择当前目录中可用的 Anima ${unavailable[1]}`);
        return;
      }
    }
    if (!isAnima && settings.guidance === "cfg_zero_star") {
      setError("SD / iL 不支持 CFG-Zero*，请切换为“无”或 PAG");
      return;
    }
    setBusy(true);
    setError("");
    const normalizedHiresSeed = normalizeHiresSeed(settings.hires.seedMode, settings.hires.seed);
    const safeSettings = { ...settings, seed: normalizeUint64Seed(settings.seed, "0"), hires: { ...settings.hires, ...normalizedHiresSeed } };
    const savedSettings = isAnima ? {
      ...safeSettings,
      checkpoint: "",
      sampler: SAMPLERS.includes(settings.sampler) ? settings.sampler : "euler",
      scheduler: SCHEDULERS.includes(settings.scheduler) ? settings.scheduler : "simple",
      guidance: settings.guidance,
      processPreview: false,
    } : {
      ...safeSettings,
      diffusionModel: "",
      textEncoder: "",
      vae: "",
    };
    const payload = {
      collection_id: selectedCollection,
      title: title.trim() || null,
      settings: {
        ...savedSettings,
        imageSeeds: images.map((image) => normalizeUint64Seed(image.seed, "")),
        imageHiresSeedModes: images.map((image) => normalizeHiresSeed(image.hiresSeedMode ?? savedSettings.hires.seedMode, image.hiresSeed ?? savedSettings.hires.seed).seedMode),
        imageHiresSeeds: images.map((image) => normalizeHiresSeed(image.hiresSeedMode ?? savedSettings.hires.seedMode, image.hiresSeed ?? savedSettings.hires.seed).seed),
      },
      images: images.map((image) => image.kind === "existing" ? { gallery_image_id: image.id } : { data_url: image.dataUrl, name: image.name }),
    };
    try {
      const result = await galleryRequest(card ? `/cards/${card.id}` : "/cards", { method: card ? "PATCH" : "POST", body: JSON.stringify(payload) });
      onSaved(result);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  };
  return <div className="gallery-editor-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
    <form ref={dialogRef} className="gallery-card-editor" role="dialog" aria-modal="true" aria-labelledby="gallery-card-editor-title" tabIndex="-1" onSubmit={save}>
      <header><div><span>CURATED CARD EDITOR</span><h2 id="gallery-card-editor-title">{card ? "编辑精选卡片" : "添加精选卡片"}</h2><p>图片可选，参数可独立于当前生图工作区保存</p></div><button type="button" aria-label="关闭卡片编辑器" onClick={onClose}><X size={20} /></button></header>
      <fieldset className="gallery-editor-body" disabled={busy}>
        <section className="gallery-editor-media">
          <div className="gallery-editor-section-head"><div><span>01</span><strong>卡片与图片</strong></div><button type="button" onClick={() => fileInput.current?.click()}><Upload size={13} />添加图片</button><input ref={fileInput} type="file" hidden multiple accept="image/png,image/jpeg,image/webp,image/gif" onChange={(event) => { void addFiles(event.target.files); event.target.value = ""; }} /></div>
          <div className="gallery-editor-grid two"><label className="gallery-field"><span>收藏夹</span><select value={selectedCollection} onChange={(event) => setSelectedCollection(event.target.value)}>{collections.map((item) => <option value={item.id} key={item.id}>{item.id}</option>)}</select></label><label className="gallery-field"><span>概括标题（可选）</span><input value={title} maxLength={160} onChange={(event) => setTitle(event.target.value)} placeholder="留空时使用正向 Prompt 概括" /></label></div>
          <div className="gallery-editor-images">
            {images.map((image, index) => <article key={image.id || `${image.name}-${index}`}><img src={image.kind === "existing" ? thumbUrl(image) : image.url} alt="" loading="lazy" decoding="async" /><span>{image.name}<small>{index + 1} / {images.length}</small></span><div><button type="button" aria-label={`上移图片 ${index + 1}`} disabled={index === 0} onClick={() => moveImage(index, -1)}><ArrowUp size={12} /></button><button type="button" aria-label={`下移图片 ${index + 1}`} disabled={index === images.length - 1} onClick={() => moveImage(index, 1)}><ArrowDown size={12} /></button><button type="button" aria-label={`删除图片 ${index + 1}`} onClick={() => setImages((current) => current.filter((_, itemIndex) => itemIndex !== index))}><Trash2 size={12} /></button></div></article>)}
            {images.length === 0 && <button type="button" className="gallery-editor-drop" onClick={() => fileInput.current?.click()}><ImageIcon size={26} /><span>图片可不添加</span><small>{title.trim() || settings.positive.trim() ? "卡片会显示文字概括" : "未填写标题和 Prompt 时显示 No image"}</small></button>}
          </div>
        </section>

        <section>
          <div className="gallery-editor-section-head"><div><span>02</span><strong>模型组件与 LoRA</strong></div><button type="button" onClick={() => setLoraOpen(true)}><Layers3 size={13} />LoRA 管理器</button></div>
          <div className={`gallery-editor-grid ${isAnima ? "four" : "three"}`}>
            <label className="gallery-field"><span>模型引擎</span><select value={settings.model} onChange={(event) => changeEngine(event.target.value)}><option value="SD">SD</option><option value="iL">iL / SDXL</option><option value="Anima">Anima</option></select></label>
            {isAnima ? [["diffusionModel", "扩散模型", splitAssets.diffusion_model], ["textEncoder", "文本编码器", splitAssets.text_encoder], ["vae", "VAE", splitAssets.vae]].map(([key, label, catalog]) => <label className="gallery-field" key={key}><span>{label}</span><select value={settings[key]} onChange={(event) => setField(key, event.target.value)}><option value="">未指定</option>{settings[key] && !catalog.some((item) => item.value === settings[key]) && <option value={settings[key]}>{settings[key]}（未在当前目录发现）</option>}{catalog.map((item) => <option value={item.value} key={item.value}>{item.name}</option>)}</select></label>) : <label className="gallery-field wide"><span>底模选择</span><select value={settings.checkpoint} onChange={(event) => setField("checkpoint", event.target.value)}><option value="">未指定</option>{settings.checkpoint && !checkpoints.some((item) => item.value === settings.checkpoint) && <option value={settings.checkpoint}>{settings.checkpoint}（未在当前目录发现）</option>}{checkpoints.map((item) => <option value={item.value} key={item.value}>{item.name}</option>)}</select></label>}
          </div>
          {modelCatalogError && <p className="gallery-form-error">模型目录扫描失败：{modelCatalogError}</p>}
          <div className="gallery-editor-lora-summary">{settings.loras.length ? settings.loras.map((item, index) => <span className={item.enabled === false ? "disabled" : ""} key={`${item.value}-${index}`}><b>{item.name || compactName(item.value)}</b><small>{formatWeight(item.weight ?? 1)}</small></span>) : <p>未挂载 LoRA</p>}</div>
        </section>

        <section>
          <div className="gallery-editor-section-head"><div><span>03</span><strong>Prompt</strong></div></div>
          <p className="prompt-syntax-help gallery-prompt-syntax-help">权重语法：<code>(text)</code> = 1.1；<code>(text:1.25)</code> = 显式权重；<code>\(text\)</code> = 字面括号。</p>
          <label className="gallery-field prompt-positive"><span>正向 Prompt</span><textarea value={settings.positive} maxLength={8000} onChange={(event) => setField("positive", event.target.value)} placeholder="画面主体、风格、构图与光线" /></label>
          <label className="gallery-field prompt-negative"><span>负向 Prompt</span><textarea value={settings.negative} maxLength={8000} onChange={(event) => setField("negative", event.target.value)} placeholder="不希望出现的内容" /></label>
        </section>

        <details open>
          <summary><SlidersHorizontal size={14} />采样参数与画布</summary>
          <div className="gallery-details-content">
            <div className="gallery-editor-grid four"><NumericField label="采样步数" value={settings.steps} min={1} max={100} onChange={(value) => setField("steps", Math.round(value))} /><NumericField label="CFG" value={settings.cfg} min={0} max={30} step={0.1} onChange={(value) => setField("cfg", value)} /><NumericField label="降噪" value={settings.denoise} min={0} max={1} step={0.01} onChange={(value) => setField("denoise", value)} /><label className="gallery-field"><span>Seed</span><input value={settings.seed} onChange={(event) => setField("seed", event.target.value.replace(/\D/g, ""))} /></label></div>
            <div className="gallery-editor-grid three"><label className="gallery-field"><span>引导增强</span><select value={settings.guidance} aria-invalid={!isAnima && settings.guidance === "cfg_zero_star"} aria-describedby={!isAnima && settings.guidance === "cfg_zero_star" ? "gallery-guidance-note" : undefined} onChange={(event) => { const next = event.target.value; if (isAnima || next !== "cfg_zero_star") setField("guidance", next); }}>{GUIDANCE.map(([id, label]) => { const disabled = !isAnima && id === "cfg_zero_star"; return <option value={id} key={id} disabled={disabled}>{label}{disabled ? "（当前模型不可用）" : ""}</option>; })}</select></label><label className="gallery-field"><span>采样器</span><select value={settings.sampler} onChange={(event) => setField("sampler", event.target.value)}>{visibleSamplers.map((item) => <option value={item} key={item}>{item}</option>)}</select></label><label className="gallery-field"><span>调度器</span><select value={settings.scheduler} onChange={(event) => setField("scheduler", event.target.value)}>{visibleSchedulers.map((item) => <option value={item} key={item}>{item}</option>)}</select></label></div>
            {settings.guidance === "pag" && <><div className="gallery-editor-grid two"><NumericField label="PAG 强度" value={settings.pag.scale} min={0} max={5} step={0.01} onChange={(value) => setNested("pag", "scale", value)} /><label className="gallery-field"><span>PAG 作用层</span><select value={settings.pag.appliedLayers} onChange={(event) => setNested("pag", "appliedLayers", event.target.value)}><option value="mid">Mid（推荐）</option><option value="all">全部自注意力层（高风险）</option></select></label></div>{settings.pag.appliedLayers === "all" && <p className="gallery-guidance-note">全部层会明显放大对比、描边和色彩。动漫模型建议使用 Mid，并从强度 0.3 开始。</p>}</>}
            {settings.guidance === "cfg_zero_star" && !isAnima && <p id="gallery-guidance-note" className="gallery-guidance-note" role="status">SD / iL 不支持 CFG-Zero*；该引导可用于已就绪的 Anima Flow Matching 引擎。</p>}
            <div className="gallery-editor-grid two"><NumericField label="单批图片数" value={settings.imagesPerBatch} min={1} max={10} onChange={(value) => setField("imagesPerBatch", Math.round(value))} /><NumericField label="批次数" value={settings.batchCount} min={1} max={20} onChange={(value) => setField("batchCount", Math.round(value))} /></div>
            <div className="gallery-editor-grid four"><NumericField label="画布宽度" value={settings.size.width} min={64} max={2048} onChange={(value) => setField("size", { ...settings.size, width: Math.round(value / 64) * 64 })} /><NumericField label="画布高度" value={settings.size.height} min={64} max={2048} onChange={(value) => setField("size", { ...settings.size, height: Math.round(value / 64) * 64 })} /><label className="gallery-field"><span>Seed 模式</span><select value={settings.seedMode} onChange={(event) => setField("seedMode", event.target.value)}><option value="fixed">固定</option><option value="random">随机</option><option value="increment">递增</option><option value="decrement">递减</option></select></label><ToggleField label="过程预览" detail={isAnima ? "Anima 不支持" : "下一次生成时使用"} checked={settings.processPreview !== false} disabled={isAnima} onChange={(value) => setField("processPreview", value)} /></div>
            {isAnima && <p className="gallery-guidance-note">Anima 始终关闭过程预览；该限制不影响 LoRA 或后处理参数。</p>}
          </div>
        </details>

        <details>
          <summary><WandSparkles size={14} />Hires.fix</summary>
          <div className="gallery-hires-seed-editor">
            <div className="gallery-editor-grid two">
              <label className="gallery-field"><span>Hires Seed 模式</span><select value={settings.hires.seedMode} onChange={(event) => { const mode = event.target.value; const next = normalizeHiresSeed(mode, mode === "fixed" ? normalizeUint64Seed(settings.hires.seed, settings.seed) : null); setSettings((current) => ({ ...current, hires: { ...current.hires, ...next } })); }}><option value="inherit">继承首轮</option><option value="fixed">固定</option><option value="random">每张随机</option></select></label>
              {settings.hires.seedMode === "fixed" && <label className="gallery-field gallery-hires-seed-field"><span>固定 Hires Seed</span><i><input inputMode="numeric" maxLength="20" value={settings.hires.seed} onChange={(event) => setNested("hires", "seed", event.target.value.replace(/\D/g, ""))} onBlur={() => setNested("hires", "seed", normalizeUint64Seed(settings.hires.seed, "0"))} /><button type="button" title="生成固定 Hires Seed" onClick={() => setNested("hires", "seed", secureRandomUint64Seed())}><RefreshCw size={12} /></button></i></label>}
            </div>
            {images.length > 0 && <div className="gallery-image-hires-seeds"><strong>逐图 Hires Seed</strong>{images.map((image, index) => { const imageHires = normalizeHiresSeed(image.hiresSeedMode ?? settings.hires.seedMode, image.hiresSeed ?? settings.hires.seed); return <div key={image.id || `${image.name}-${index}`}><span>{index + 1}</span><select value={imageHires.seedMode} onChange={(event) => { const mode = event.target.value; const next = normalizeHiresSeed(mode, mode === "fixed" ? normalizeUint64Seed(imageHires.seed, settings.seed) : null); setImages((current) => current.map((item, itemIndex) => itemIndex === index ? { ...item, hiresSeedMode: next.seedMode, hiresSeed: next.seed } : item)); }}><option value="inherit">继承</option><option value="fixed">固定</option><option value="random">随机</option></select>{imageHires.seedMode === "fixed" ? <><input inputMode="numeric" maxLength="20" value={imageHires.seed} onChange={(event) => setImages((current) => current.map((item, itemIndex) => itemIndex === index ? { ...item, hiresSeedMode: "fixed", hiresSeed: event.target.value.replace(/\D/g, "") } : item))} onBlur={() => setImages((current) => current.map((item, itemIndex) => itemIndex === index ? { ...item, hiresSeed: normalizeUint64Seed(item.hiresSeed, "0") } : item))} /><button type="button" title={`生成图片 ${index + 1} 的固定 Hires Seed`} onClick={() => setImages((current) => current.map((item, itemIndex) => itemIndex === index ? { ...item, hiresSeedMode: "fixed", hiresSeed: secureRandomUint64Seed() } : item))}><RefreshCw size={12} /></button></> : <small>{imageHires.seedMode === "inherit" ? "依赖该图首轮 Seed" : "下次生成时重新随机"}</small>}</div>; })}</div>}
          </div>
          <div className="gallery-details-content"><ToggleField label="启用 Hires.fix" detail="应用时由生图区域验证引擎能力与超分模型" checked={settings.hires.enabled} onChange={(value) => setNested("hires", "enabled", value)} /><div className="gallery-editor-grid four"><label className="gallery-field"><span>超分模型</span><input value={settings.hires.model} onChange={(event) => setNested("hires", "model", event.target.value)} /></label><NumericField label="倍率" value={settings.hires.scale} min={1} max={4} step={0.1} onChange={(value) => setNested("hires", "scale", value)} /><NumericField label="重绘强度" value={settings.hires.denoise} min={0.05} max={1} step={0.01} onChange={(value) => setNested("hires", "denoise", value)} /><NumericField label="Hires 步数" value={settings.hires.steps} min={1} max={100} onChange={(value) => setNested("hires", "steps", Math.round(value))} /><NumericField label="Hires CFG" value={settings.hires.cfg} min={0} max={30} step={0.1} onChange={(value) => setNested("hires", "cfg", value)} /><NumericField label="像素放大分块" value={settings.hires.tileSize} min={32} max={2048} onChange={(value) => setNested("hires", "tileSize", Math.round(value))} /><NumericField label="像素放大分块重叠" value={settings.hires.tileOverlap} min={0} max={512} onChange={(value) => setNested("hires", "tileOverlap", Math.round(value))} /></div>{isAnima && <div className="gallery-editor-grid two"><label className="gallery-field"><span>重绘方式</span><select value={settings.hires.executionMode} onChange={(event) => setNested("hires", "executionMode", event.target.value)}><option value="usdu_tiled">USDU 分块重绘（推荐）</option><option value="full_frame">整图重绘（兼容）</option></select></label><label className="gallery-field"><span>Hires 采样器</span><select value={settings.hires.sampler || ""} onChange={(event) => setNested("hires", "sampler", event.target.value || null)}><option value="">跟随首轮</option>{SAMPLERS.map((item) => <option value={item} key={item}>{item}</option>)}</select></label><label className="gallery-field"><span>Hires 调度器</span><select value={settings.hires.scheduler || ""} onChange={(event) => setNested("hires", "scheduler", event.target.value || null)}><option value="">跟随首轮</option>{SCHEDULERS.map((item) => <option value={item} key={item}>{item}</option>)}</select></label></div>} {isAnima && settings.hires.executionMode === "usdu_tiled" && <p className="gallery-field-detail">扩散重绘分块：Auto 使用首轮图片宽高；padding 32；每 tile 按 Hires 步数执行；seam fix 目前为 None。</p>}</div>
        </details>

        <details>
          <summary><Sparkles size={14} />ADetailer</summary>
          <div className="gallery-details-content"><ToggleField label="启用 ADetailer" detail="应用时由生图区域验证引擎能力与检测模型" checked={settings.adetailer.enabled} onChange={(value) => setNested("adetailer", "enabled", value)} /><div className="gallery-editor-grid four"><label className="gallery-field wide"><span>检测模型</span><input value={settings.adetailer.detector} onChange={(event) => setNested("adetailer", "detector", event.target.value)} /></label><NumericField label="置信度" value={settings.adetailer.confidence} min={0.05} max={1} step={0.01} onChange={(value) => setNested("adetailer", "confidence", value)} /><NumericField label="区域数" value={settings.adetailer.maxDetections} min={1} max={8} onChange={(value) => setNested("adetailer", "maxDetections", Math.round(value))} /><NumericField label="最小比例" value={settings.adetailer.maskMinRatio} min={0} max={0.5} step={0.01} onChange={(value) => setNested("adetailer", "maskMinRatio", value)} /><NumericField label="最大比例" value={settings.adetailer.maskMaxRatio} min={0.05} max={1} step={0.01} onChange={(value) => setNested("adetailer", "maskMaxRatio", value)} /><NumericField label="膨胀/腐蚀" value={settings.adetailer.dilateErode} min={-128} max={128} onChange={(value) => setNested("adetailer", "dilateErode", Math.round(value))} /><NumericField label="蒙版模糊" value={settings.adetailer.maskBlur} min={0} max={64} onChange={(value) => setNested("adetailer", "maskBlur", Math.round(value))} /><NumericField label="边距" value={settings.adetailer.padding} min={0} max={256} onChange={(value) => setNested("adetailer", "padding", Math.round(value))} /><NumericField label="重绘强度" value={settings.adetailer.denoise} min={0.05} max={1} step={0.01} onChange={(value) => setNested("adetailer", "denoise", value)} /></div><div className="gallery-editor-grid two"><label className="gallery-field"><span>ADetailer 正向 Prompt</span><textarea value={settings.adetailer.prompt} onChange={(event) => setNested("adetailer", "prompt", event.target.value)} /></label><label className="gallery-field"><span>ADetailer 负向 Prompt</span><textarea value={settings.adetailer.negativePrompt} onChange={(event) => setNested("adetailer", "negativePrompt", event.target.value)} /></label></div></div>
        </details>

        <details>
          <summary><Zap size={14} />RTX VSR 与后处理</summary>
          <div className="gallery-details-content"><ToggleField label="启用 RTX VSR" detail="应用时由生图区域验证引擎能力与 RTX 运行时" checked={settings.rtx.enabled} onChange={(value) => setNested("rtx", "enabled", value)} /><div className="gallery-editor-grid two"><NumericField label="RTX 倍率" value={settings.rtx.scale} min={1} max={4} step={0.01} onChange={(value) => setNested("rtx", "scale", value)} /><label className="gallery-field"><span>RTX 质量</span><select value={settings.rtx.quality} onChange={(event) => setNested("rtx", "quality", event.target.value)}><option value="low">LOW</option><option value="medium">MEDIUM</option><option value="high">HIGH</option><option value="ultra">ULTRA</option></select></label></div><div className="gallery-stage-order">{settings.postprocessOrder.map((stage, index) => <div key={stage}><span>{index + 1}</span><strong>{{ hires: "Hires.fix", adetailer: "ADetailer", rtx: "RTX VSR" }[stage]}</strong><button type="button" disabled={index === 0} onClick={() => moveStage(stage, -1)}><ArrowUp size={12} /></button><button type="button" disabled={index === settings.postprocessOrder.length - 1} onClick={() => moveStage(stage, 1)}><ArrowDown size={12} /></button></div>)}</div></div>
        </details>
        {error && <p className="gallery-form-error">{error}</p>}
      </fieldset>
      <footer><span>{images.length ? `${images.length} 张图片` : "无图片卡片"} · {settings.loras.length} 个 LoRA · {settings.size.width} × {settings.size.height}</span><div><button type="button" onClick={onClose}>取消</button><button type="submit" className="primary" disabled={busy || !selectedCollection}>{busy ? <RefreshCw className="spin" size={14} /> : <Save size={14} />}保存卡片</button></div></footer>
    </form>
    {loraOpen && <GalleryLoraManager engine={settings.model} value={settings.loras} onClose={() => setLoraOpen(false)} onApply={(next) => { setField("loras", next); setLoraOpen(false); }} />}
  </div>;
}

function ApplySettingsDialog({ card, onClose, onApply }) {
  const dialogRef = useDialogLifecycle(true, onClose);
  // Defaults to Prompt only. Applying everything replaces the model, the whole
  // LoRA mount and every sampling parameter, which is a much larger action than
  // "I want this card's prompt" — the reason people open this dialog.
  const [selected, setSelected] = useState(() => new Set(APPLY_DEFAULT_GROUPS));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const toggle = (id) => setSelected((current) => { if (busy) return current; const next = new Set(current); if (next.has(id)) next.delete(id); else next.add(id); return next; });
  const apply = async (groups) => {
    setBusy(true);
    setError("");
    try {
      await onApply(card.settings, groups);
      onClose();
    } catch (applyError) {
      setError(applyError.message);
    } finally {
      setBusy(false);
    }
  };
  return <div className="gallery-dialog-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
    <section ref={dialogRef} className="gallery-apply-dialog" role="dialog" aria-modal="true" aria-labelledby="gallery-apply-dialog-title" tabIndex="-1">
      <header><div><span>APPLY PARAMETERS</span><h2 id="gallery-apply-dialog-title">应用到生图区域</h2><p title={card.title || card.settings?.positive}>{displayTitle(card)}</p></div><button type="button" aria-label="关闭参数应用窗口" onClick={onClose}><X size={18} /></button></header>
      <div className="gallery-apply-grid">{APPLY_GROUPS.map(([id, label, detail]) => <button type="button" className={selected.has(id) ? "selected" : ""} aria-pressed={selected.has(id)} disabled={busy} key={id} onClick={() => toggle(id)}><i>{selected.has(id) && <Check size={12} />}</i><span><strong>{label}</strong><small>{detail}</small></span></button>)}</div>
      {card.settings?.loraGroupPrompt && selected.has("prompts") && (
        // Applied prompts are the handwritten part only. The prefix came from
        // whichever combinations were enabled at the time, and those live in the
        // workspace rather than the card, so saying nothing would let someone
        // apply a prompt and quietly not reproduce the picture.
        <p className="gallery-apply-hint">该卡片的正向 Prompt 还包含 LoRA 组合的预设前缀，应用时不会带入。需要完全复现请在卡片详情中复制「实际提交的正向 Prompt」。</p>
      )}
      {error && <p className="gallery-form-error">{error}</p>}
      <footer><button type="button" disabled={busy} onClick={() => void apply(APPLY_GROUPS.map(([id]) => id))}>应用全部 {APPLY_GROUPS.length} 项</button><button type="button" className="primary" disabled={busy || selected.size === 0} onClick={() => void apply([...selected])}><WandSparkles size={14} />应用所选 {selected.size} 项</button></footer>
    </section>
  </div>;
}

function GalleryDetail({ card, onBack, onEdit, onDelete, onApply, onNotice }) {
  const [imageIndex, setImageIndex] = useState(0);
  const [applyOpen, setApplyOpen] = useState(false);
  useEffect(() => setImageIndex(0), [card.id]);
  const baseSettings = normalizedSettings({ ...card.settings, seed: card.settings?.imageSeeds?.[imageIndex] || card.settings?.seed });
  const settings = { ...baseSettings, hires: { ...baseSettings.hires, ...selectedImageHiresSeed(baseSettings, imageIndex) } };
  const image = card.images?.[imageIndex];
  const stageLabels = { hires: "Hires.fix", adetailer: "ADetailer", rtx: "RTX VSR" };
  const isAnima = settings.model === "Anima";
  return <section className="gallery-focus">
    <div className="gallery-focus-media">
      <header><button type="button" onClick={onBack}><ArrowLeft size={15} />返回卡片</button><span>{card.collection_id} / {card.image_count || 0} IMAGE</span></header>
      <div className="gallery-focus-stage">{image ? <FocusImage image={image} alt={displayTitle(card)} /> : <div><ImageIcon size={42} /><strong>{card.title?.trim() || "无图片精选卡片"}</strong><p>{settings.positive || "未填写 Prompt"}</p></div>}</div>
      {card.images?.length > 1 && <div className="gallery-focus-thumbs">{card.images.map((item, index) => <button type="button" className={index === imageIndex ? "active" : ""} key={item.id} onClick={() => setImageIndex(index)}><img src={thumbUrl(item)} alt={`图片 ${index + 1}`} loading="lazy" decoding="async" /><b>{index + 1}</b></button>)}</div>}
    </div>
    <aside className="gallery-inspector">
      <header><div><span>CURATED DETAIL</span><h1 title={card.title || settings.positive}>{displayTitle(card)}</h1><p>{card.collection_id} · 更新于 {formatDate(card.updated_at)}</p></div><div><button type="button" onClick={() => onEdit(card)}><Pencil size={14} />编辑</button><button type="button" className="danger" aria-label="删除精选卡片" onClick={() => onDelete(card)}><Trash2 size={14} /></button></div></header>
      <div className="gallery-inspector-scroll">
        {settings.loraGroupPrompt && (
          <PromptBlock tone="composed" label="实际提交的正向 Prompt" value={effectivePrompt(settings)} onNotice={onNotice} />
        )}
        <PromptBlock tone="positive" label={settings.loraGroupPrompt ? "正向 Prompt（手写部分）" : "正向 Prompt"} value={settings.positive} onNotice={onNotice} />
        <PromptBlock tone="negative" label="负向 Prompt" value={settings.negative} onNotice={onNotice} />
        {settings.loraGroups.length > 0 && (
          <section className="gallery-detail-section gallery-group-section">
            <header><span>LORA GROUPS</span><strong>生成时启用的组合</strong></header>
            <div className="gallery-group-list">
              {settings.loraGroups.map((group) => {
                const members = settings.loras.filter((item) => item.groupId === group.id);
                return (
                  <div className="gallery-group-entry" key={group.id}>
                    <header><i /><strong title={group.name}>{group.name}</strong><b>{members.length} 个 LoRA</b></header>
                    {group.presetPrompt
                      ? <p className="gallery-group-preset">{group.presetPrompt}</p>
                      : <p className="gallery-group-preset empty">该组合没有预设提示词</p>}
                    {members.length > 0 && (
                      <ul>{members.map((item) => (
                        <li className={item.enabled === false ? "disabled" : ""} key={item.value}>
                          <span title={item.name}>{item.name || compactName(item.value)}</span>
                          <b>{formatWeight(item.weight ?? 1)}</b>
                        </li>
                      ))}</ul>
                    )}
                  </div>
                );
              })}
            </div>
            <p className="gallery-detail-line">组合的预设提示词在提交生成时拼接到正向 Prompt 最前面，生图页的输入框不会被改写。</p>
          </section>
        )}
        <section className="gallery-detail-section"><header><span>MODEL</span><strong>模型组件与 LoRA</strong></header><dl><div><dt>引擎</dt><dd>{settings.model}</dd></div>{isAnima ? <><div><dt>扩散模型</dt><dd title={settings.diffusionModel}>{compactName(settings.diffusionModel)}</dd></div><div><dt>文本编码器</dt><dd title={settings.textEncoder}>{compactName(settings.textEncoder)}</dd></div><div><dt>VAE</dt><dd title={settings.vae}>{compactName(settings.vae)}</dd></div></> : <div><dt>底模</dt><dd title={settings.checkpoint}>{compactName(settings.checkpoint)}</dd></div>}</dl><div className="gallery-detail-loras">{settings.loras.length ? settings.loras.map((item, index) => { const owner = settings.loraGroups.find((group) => group.id === item.groupId); return <div className={item.enabled === false ? "disabled" : ""} key={`${item.value}-${index}`}><span>{String(index + 1).padStart(2, "0")}</span><strong>{item.name || compactName(item.value)}</strong>{owner && <em className="gallery-lora-group-tag" title={`来自组合 ${owner.name}`}>{owner.name}</em>}<b>{formatWeight(item.weight ?? 1)}</b><small>{item.enabled === false ? "关闭" : "启用"}</small></div>; }) : <p>未使用 LoRA</p>}</div></section>
      <section className="gallery-detail-section"><header><span>SAMPLING</span><strong>采样与画布</strong></header><div className="gallery-stat-grid"><div><span>STEP</span><b>{settings.steps}</b></div><div><span>CFG</span><b>{settings.cfg}</b></div><div><span>DENOISE</span><b>{settings.denoise}</b></div><div><span>SEED</span><b title={settings.seed}>{settings.seed}</b></div><div><span>SIZE</span><b>{settings.size.width} × {settings.size.height}</b></div><div><span>BATCH</span><b>{settings.imagesPerBatch} × {settings.batchCount}</b></div></div><p className="gallery-detail-line">{GUIDANCE.find(([id]) => id === settings.guidance)?.[1] || "无（None）"}{settings.guidance === "pag" ? ` ${settings.pag.scale} / ${settings.pag.appliedLayers === "mid" ? "Mid" : "全部层"}` : ""} · {settings.sampler} · {settings.scheduler} · 过程预览 {settings.processPreview === false ? "关闭" : "开启"}</p></section>
        <section className="gallery-detail-section"><header><span>POST PROCESS</span><strong>后处理开关与顺序</strong></header><div className="gallery-process-list">{settings.postprocessOrder.map((stage, index) => { const stageSettings = settings[stage]; return <div className={stageSettings?.enabled ? "enabled" : "disabled"} key={stage}><span>{index + 1}</span><strong>{stageLabels[stage]}</strong><b>{stageSettings?.enabled ? "ON" : "OFF"}</b><small>{stage === "hires" ? `${stageSettings.scale}x · ${compactName(stageSettings.model)}${isAnima ? ` · ${stageSettings.executionMode === "usdu_tiled" ? "USDU 分块重绘" : "整图重绘"}` : ""}` : stage === "adetailer" ? `${compactName(stageSettings.detector)} · ${stageSettings.confidence}` : `${stageSettings.scale}x · ${String(stageSettings.quality).toUpperCase()}`}</small></div>; })}</div><p className="gallery-detail-line">Hires Seed · {settings.hires.seedMode === "inherit" ? "继承当前图片首轮 Seed" : settings.hires.seedMode === "random" ? "每张安全随机" : settings.hires.seed}</p></section>
        <section className="gallery-detail-section"><header><span>ADETAILER DETAIL</span><strong>局部重绘参数</strong></header><dl><div><dt>Prompt</dt><dd>{settings.adetailer.prompt || "继承主 Prompt"}</dd></div><div><dt>负向</dt><dd>{settings.adetailer.negativePrompt || "继承主负向"}</dd></div><div><dt>蒙版</dt><dd>{settings.adetailer.maskMinRatio}–{settings.adetailer.maskMaxRatio} · blur {settings.adetailer.maskBlur}</dd></div><div><dt>重绘</dt><dd>{settings.adetailer.denoise} · {settings.adetailer.useSteps ? `${settings.adetailer.steps} steps` : "继承步数"}</dd></div></dl></section>
      </div>
      <footer><button type="button" onClick={() => setApplyOpen(true)}><SlidersHorizontal size={15} /><span><strong>应用参数</strong><small>选择性回填到生图区域</small></span><ChevronRight size={15} /></button></footer>
      {applyOpen && <ApplySettingsDialog card={{ ...card, settings }} onClose={() => setApplyOpen(false)} onApply={onApply} />}
    </aside>
  </section>;
}

export default function GalleryPage({ currentSettings, focus, onApplySettings, onApplyPrompt, onNotice }) {
  const [data, setData] = useState({ collections: [], cards: [] });
  const [activeSection, setActiveSection] = useState("gallery");
  const [selectedCollection, setSelectedCollection] = useState(null);
  const [selectedCard, setSelectedCard] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [collectionDialog, setCollectionDialog] = useState(null);
  const [editor, setEditor] = useState(null);
  const [dragState, setDragState] = useState(null);
  const [reorderBusy, setReorderBusy] = useState("");
  const [cardMenu, setCardMenu] = useState(null);
  const [promptEntries, setPromptEntries] = useState([]);
  const [promptLoading, setPromptLoading] = useState(true);
  const [promptError, setPromptError] = useState("");
  const [promptDialog, setPromptDialog] = useState(null);
  const dragSession = useRef(null);
  const refresh = async () => {
    setLoading(true);
    setError("");
    try {
      const payload = await galleryRequest();
      setData(payload);
      setSelectedCard((current) => current ? payload.cards.find((item) => item.id === current.id) || null : null);
      return payload;
    } catch (requestError) {
      setError(requestError.message);
      return null;
    } finally {
      setLoading(false);
    }
  };
  const refreshPrompts = async () => {
    setPromptLoading(true);
    setPromptError("");
    try {
      const payload = await galleryRequest("/prompts");
      setPromptEntries(payload.prompts || []);
      return payload;
    } catch (requestError) {
      setPromptError(requestError.message);
      return null;
    } finally {
      setPromptLoading(false);
    }
  };
  useEffect(() => { void refresh(); void refreshPrompts(); }, []);
  useEffect(() => {
    if (!focus || loading) return;
    if (focus.collectionId) { setActiveSection("gallery"); setSelectedCollection(focus.collectionId); }
    if (focus.cardId) setSelectedCard(data.cards.find((item) => item.id === focus.cardId) || null);
  }, [focus, loading, data.cards]);
  const cardsByCollection = new Map(data.collections.map((collection) => [collection.id, data.cards.filter((card) => card.collection_id === collection.id)]));
  const visibleCollections = selectedCollection === null ? data.collections : data.collections.filter((item) => item.id === selectedCollection);
  const deleteCard = async (card) => {
    if (!window.confirm(`删除精选卡片“${displayTitle(card)}”？精选副本会被删除，原始生成图片不受影响。`)) return;
    try {
      await galleryRequest(`/cards/${card.id}`, { method: "DELETE" });
      setSelectedCard(null);
      await refresh();
      onNotice("精选卡片已删除");
    } catch (requestError) {
      onNotice(requestError.message, true);
    }
  };
  const openCardMenu = (card, event) => {
    const menuWidth = 224;
    const menuHeight = 150;
    const margin = 8;
    setCardMenu({
      card,
      x: Math.max(margin, Math.min(event.clientX, window.innerWidth - menuWidth - margin)),
      y: Math.max(margin, Math.min(event.clientY, window.innerHeight - menuHeight - margin)),
    });
  };
  const editFromCardMenu = (card) => {
    setCardMenu(null);
    setEditor({ mode: "edit", card });
  };
  const deleteFromCardMenu = (card) => {
    setCardMenu(null);
    void deleteCard(card);
  };
  const startCardReorder = (card, point) => {
    if (reorderBusy) return false;
    const order = data.cards.filter((item) => item.collection_id === card.collection_id).map((item) => item.id);
    if (order.length < 2) {
      onNotice("当前收藏夹只有一张卡片，无需排序");
      return false;
    }
    dragSession.current = { collectionId: card.collection_id, original: [...order], order: [...order], changed: false };
    setDragState({ cardId: card.id, collectionId: card.collection_id, targetId: card.id, ...point });
    return true;
  };
  const moveCardDuringReorder = (card, targetId, point) => {
    const session = dragSession.current;
    if (!session || session.collectionId !== card.collection_id) return;
    setDragState((current) => current ? { ...current, targetId: targetId || current.targetId, ...point } : current);
    if (!targetId || card.id === targetId) return;
    const fromIndex = session.order.indexOf(card.id);
    const targetIndex = session.order.indexOf(targetId);
    if (fromIndex < 0 || targetIndex < 0 || fromIndex === targetIndex) return;
    const next = [...session.order];
    const [moved] = next.splice(fromIndex, 1);
    next.splice(targetIndex, 0, moved);
    session.order = next;
    session.changed = true;
    setData((current) => ({ ...current, cards: applyCollectionCardOrder(current.cards, session.collectionId, next) }));
    setDragState({ cardId: card.id, collectionId: card.collection_id, targetId, ...point });
  };
  const finishCardReorder = async () => {
    const session = dragSession.current;
    dragSession.current = null;
    setDragState(null);
    if (!session?.changed) return;
    setReorderBusy(session.collectionId);
    try {
      const payload = await galleryRequest(`/collections/${encodeURIComponent(session.collectionId)}/card-order`, {
        method: "PUT",
        body: JSON.stringify({ card_ids: session.order }),
      });
      const saved = new Map((payload.cards || []).map((card) => [card.id, card]));
      setData((current) => ({ ...current, cards: current.cards.map((card) => saved.get(card.id) || card) }));
      onNotice("卡片顺序已保存");
    } catch (requestError) {
      setData((current) => ({ ...current, cards: applyCollectionCardOrder(current.cards, session.collectionId, session.original) }));
      onNotice(requestError.message, true);
    } finally {
      setReorderBusy("");
    }
  };
  const saveCollection = async (collection) => {
    setCollectionDialog(null);
    setSelectedCollection(collection.id);
    await refresh();
    onNotice("收藏夹已保存");
  };
  const collectionDeleted = async () => {
    setCollectionDialog(null);
    setSelectedCollection(null);
    setSelectedCard(null);
    await refresh();
    onNotice("收藏夹已删除");
  };
  const saveCard = async (card) => {
    setEditor(null);
    setSelectedCollection(card.collection_id);
    await refresh();
    setSelectedCard(card);
    onNotice("精选卡片已保存");
  };
  const savePromptEntry = async () => {
    setPromptDialog(null);
    await refreshPrompts();
    onNotice("词条已保存");
  };
  const deletePromptEntry = async (entry) => {
    if (!window.confirm(`删除词条“${entry.title}”？`)) return;
    try {
      await galleryRequest(`/prompts/${entry.id}`, { method: "DELETE" });
      await refreshPrompts();
      onNotice("词条已删除");
    } catch (requestError) {
      onNotice(requestError.message, true);
    }
  };
  const applyPromptEntry = async (entry, target) => {
    try {
      await onApplyPrompt(entry, target);
      onNotice(`已将“${entry.title}”应用到${target === "image" ? "图生图" : "文生图"}`);
    } catch (applyError) {
      onNotice(applyError.message, true);
    }
  };
  return <section className="gallery-page">
    <aside className="gallery-collection-rail">
      <header><div><span>PERSONAL LIBRARY</span><h1>画廊</h1></div><button type="button" title="创建收藏夹" onClick={() => setCollectionDialog({ mode: "create" })}><FolderPlus size={17} /></button></header>
      <button type="button" className={`gallery-all-collection ${activeSection === "gallery" && selectedCollection === null ? "active" : ""}`} onClick={() => { setActiveSection("gallery"); setSelectedCollection(null); setSelectedCard(null); }}><span><Images size={15} /><b>全部收藏夹</b></span><small>{data.cards.length} 卡片 · {data.collections.reduce((sum, item) => sum + item.image_count, 0)} 图片</small></button>
      <button type="button" className={`gallery-prompt-library-nav ${activeSection === "prompts" ? "active" : ""}`} onClick={() => { setActiveSection("prompts"); setSelectedCard(null); }}><span><BookOpen size={15} /><b>词库</b></span><small>{promptEntries.length} 个 Prompt 词条</small></button>
      <nav>{data.collections.map((collection, index) => <button type="button" className={activeSection === "gallery" && selectedCollection === collection.id ? "active" : ""} key={collection.id} onClick={() => { setActiveSection("gallery"); setSelectedCollection(collection.id); setSelectedCard(null); }}><i>{String(index + 1).padStart(2, "0")}</i><span><strong>{collection.id}</strong><small>{collection.description || "无简介"}</small></span><b>{collection.card_count}</b></button>)}</nav>
      <footer>{activeSection === "gallery" && selectedCollection !== null && <button type="button" onClick={() => setCollectionDialog({ mode: "edit", collection: data.collections.find((item) => item.id === selectedCollection) })}><Pencil size={13} />编辑当前收藏夹</button>}<span>SQLite 本地索引 · 图片与 Prompt 独立保存</span></footer>
    </aside>
    <main className="gallery-main">
      {activeSection === "prompts" ? <PromptLibrary entries={promptEntries} loading={promptLoading} error={promptError} onRefresh={() => void refreshPrompts()} onAdd={() => setPromptDialog({})} onEdit={(entry) => setPromptDialog({ entry })} onDelete={(entry) => void deletePromptEntry(entry)} onApply={(entry, target) => void applyPromptEntry(entry, target)} onNotice={onNotice} /> : selectedCard ? <GalleryDetail card={selectedCard} onBack={() => setSelectedCard(null)} onEdit={(card) => setEditor({ mode: "edit", card })} onDelete={deleteCard} onApply={onApplySettings} onNotice={onNotice} /> : <>
        <header className="gallery-hero"><div><span>CURATED COLLECTION / {selectedCollection === null ? "ALL" : selectedCollection.toUpperCase()}</span><h2>{selectedCollection === null ? "全部精选" : selectedCollection}</h2><p>{selectedCollection === null ? "按收藏夹浏览全部精选卡片，点击卡片进入大图与完整参数视图。" : data.collections.find((item) => item.id === selectedCollection)?.description || "这个收藏夹还没有简介。"}</p></div><div><button type="button" className={selectedCollection === null ? "active" : ""} onClick={() => { setSelectedCollection(null); setSelectedCard(null); }}><Images size={15} />全部</button>{selectedCollection !== null && <button type="button" onClick={() => setCollectionDialog({ mode: "edit", collection: data.collections.find((item) => item.id === selectedCollection) })}><Pencil size={14} />管理</button>}<button type="button" onClick={() => void refresh()} disabled={loading}><RefreshCw className={loading ? "spin" : ""} size={15} />刷新</button></div></header>
        <div className="gallery-scroll">
          {error && <div className="gallery-page-error"><X size={22} /><strong>无法读取画廊</strong><p>{error}</p></div>}
          {!error && loading && <div className="gallery-loading"><RefreshCw className="spin" size={25} /><span>正在读取精选集</span></div>}
          {!error && !loading && data.collections.length === 0 && <div className="gallery-first-empty"><span><Sparkles size={33} /></span><strong>创建第一个精选收藏夹</strong><p>自定义收藏夹 ID 与简介，然后添加生成结果或手动卡片。</p><button type="button" onClick={() => setCollectionDialog({ mode: "create" })}><FolderPlus size={15} />创建收藏夹</button></div>}
          {!error && !loading && visibleCollections.map((collection) => {
            const collectionCards = cardsByCollection.get(collection.id) || [];
            return <section className="gallery-collection-section" key={collection.id}>
              <header><div><span>{collection.id}</span><small>{collection.description || "PERSONAL COLLECTION"}</small></div><b>{collection.card_count} CARDS / {collection.image_count} IMAGES</b></header>
              <div className={`gallery-card-grid ${dragState?.collectionId === collection.id ? "reordering" : ""}`}>
                {collectionCards.map((card, index) => <GalleryCardTile card={card} tileIndex={index} key={card.id} onOpen={setSelectedCard} onMenu={openCardMenu} dragState={dragState} reorderBusy={reorderBusy === collection.id} onReorderStart={startCardReorder} onReorderOver={moveCardDuringReorder} onReorderEnd={finishCardReorder} />)}
                <AddCardTile collectionId={collection.id} tileIndex={collectionCards.length} onAdd={(id, initialImages) => setEditor({ mode: "create", collectionId: id, initialImages })} onDropError={(message) => onNotice(message, true)} />
              </div>
            </section>;
          })}
        </div>
      </>}
    </main>
    {dragState && <div className="gallery-drag-ghost" style={{ left: dragState.x, top: dragState.y }}><Images size={14} /><span><strong>正在排序</strong><small>{displayTitle(data.cards.find((card) => card.id === dragState.cardId), 42)}</small></span></div>}
    {cardMenu && <GalleryCardMenu menu={cardMenu} onClose={() => setCardMenu(null)} onEdit={editFromCardMenu} onDelete={deleteFromCardMenu} />}
    {collectionDialog && <CollectionDialog collection={collectionDialog.collection} onClose={() => setCollectionDialog(null)} onSaved={saveCollection} onDeleted={collectionDeleted} />}
    {editor && <GalleryCardEditor card={editor.card} collectionId={editor.collectionId} collections={data.collections} initialSettings={currentSettings} initialImages={editor.initialImages} onClose={() => setEditor(null)} onSaved={saveCard} />}
    {promptDialog && <PromptLibraryDialog entry={promptDialog.entry} onClose={() => setPromptDialog(null)} onSaved={() => void savePromptEntry()} />}
  </section>;
}
