import { useEffect, useRef, useState } from "react";
import { LORA_PUBLICATION_EVENT } from "./lora-library-client.js";
import { createPortal } from "react-dom";
import {
  AlertTriangle,
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  BookOpen,
  Check,
  ChevronDown,
  ChevronRight,
  FileSearch,
  Copy,
  FolderArchive,
  FolderPlus,
  FolderOpen,
  ImageIcon,
  ImagePlus,
  Images,
  Layers3,
  Pencil,
  Plus,
  RefreshCw,
  Save,
  Search,
  SlidersHorizontal,
  Sparkles,
  Trash2,
  Upload,
  WandSparkles,
  X,
  Zap,
} from "lucide-react";
import { SAMPLER_NAMES as SAMPLERS, SCHEDULER_NAMES as SCHEDULERS } from "./sampling-options";
import { normalizeHiresSeed, normalizeUint64Seed, secureRandomUint64Seed, supportsUsduTiled } from "./hires-settings";
import { DEFAULT_SETTINGS, GUIDANCE, boundedGalleryImageIndex, clone, displayTitle, distributeGalleryCards, galleryImageSeed, galleryRequest, normalizedSettings, useDialogLifecycle, withGalleryLoraRevision } from "./gallery-core";
import { galleryAutoScrollStep, galleryDropAnchor, galleryOrderChanged, galleryPointerIsCoarse, galleryPressIntent, gallerySlotShifts, resolveDropIndex, spliceGalleryOrder, GALLERY_CLICK_SWALLOW_MS, GALLERY_DRAG_HOLD_MS } from "./gallery-drag.js";
import { composeGroupPrompt } from "./lora-groups";
import { formatWeight } from "./lora-weight";
import { copyText } from "./clipboard";
import { engineGuidance, isSplitEngine } from "./engine-settings.js";
import { MOSS_SAMPLERS, MOSS_SCHEDULERS, isMossEngine, mossSamplerFor, mossSchedulerFor, normalizeMossSampling } from "./moss-engine.js";
import { NATIVE_STEP_ENGINES, adetailerSummary, normalizeADetailerUnits } from "./adetailer-units.js";
import ResampleChainPanel from "./ResampleChainPanel.jsx";
import GalleryOutputDrawer from "./GalleryOutputDrawer.jsx";
import GalleryTransferDialog from "./GalleryTransferDialog.jsx";
import { GALLERY_IMAGE_INFO_READ_LIMIT, galleryCardFromImageInfo, imageInfoHasGenerationData, imageInfoRecordedSeed } from "./gallery-image-info.js";
import { extractPromptLoras, modelDisplayName } from "./image-metadata.js";
import { clipSkipRange, normalizeClipSkip } from "./clip-skip.js";
import { DEFAULT_CANVAS_MAX_EDGE, effectiveCanvasMaxEdge } from "./canvas-limit.js";
import {
  chainSizeLine,
  resampleChainIssue,
  resampleUnitSummary,
  resampleUnitTitle,
  resampleUnits,
  updateChainStage,
} from "./resample-chain.js";

const MAX_MANUAL_IMAGE_BYTES = 20 * 1024 * 1024;
const MAX_MANUAL_TOTAL_BYTES = 100 * 1024 * 1024;
const clampViewerValue = (value, minimum, maximum) => Math.max(minimum, Math.min(maximum, value));
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
  ["chain", "后处理链", "重采样单元、ADetailer、RTX VSR 与执行顺序"],
  ["auxiliary", "其他选项", "透明背景模型"],
];

/** The resample seed one saved image of a card ran with, falling back to the unit's own mode. */
function selectedImageHiresSeed(settings, imageIndex) {
  const mode = settings.imageHiresSeedModes?.[imageIndex];
  const seed = settings.imageHiresSeeds?.[imageIndex];
  const firstUnit = resampleUnits(settings.postprocessChain).find((unit) => unit.enabled && unit.denoise > 0) || null;
  return normalizeHiresSeed(mode ?? firstUnit?.seedMode, seed ?? firstUnit?.seed);
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

// A picture added in this editor session has no id yet; this key is what a read that finishes later
// uses to find it again after the strip has been reordered or trimmed. Never saved.
let imageLocalSequence = 0;
function imageLocalKey() {
  imageLocalSequence += 1;
  return `local-${imageLocalSequence}`;
}

async function filesToNewImages(fileList) {
  const files = [...fileList];
  if (!files.length) throw new Error("没有检测到可添加的图片文件");
  if (files.reduce((total, file) => total + file.size, 0) > MAX_MANUAL_TOTAL_BYTES) {
    throw new Error("单次添加图片总大小不能超过 100 MiB");
  }
  const imported = [];
  for (const file of files) imported.push(await fileToDataUrl(file));
  return imported.map((item) => ({ kind: "new", localKey: imageLocalKey(), dataUrl: item.dataUrl, name: item.name, url: item.dataUrl }));
}

// Only PNG carries the text chunks the reader understands, so nothing else is uploaded to be read.
function carriesPngRecord(image) {
  if (image.kind === "new") return /^data:image\/png[;,]/i.test(image.dataUrl || "");
  return image.kind === "output" && /\.png$/i.test(image.name || "");
}

async function imageDataUrl(image) {
  if (image.dataUrl) return image.dataUrl;
  const response = await fetch(image.url, { cache: "no-store" });
  if (!response.ok) throw new Error(`无法读取 ${image.name}`);
  const blob = await response.blob();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.addEventListener("load", () => resolve(String(reader.result || "")));
    reader.addEventListener("error", () => reject(new Error(`无法读取 ${image.name}`)));
    reader.readAsDataURL(blob);
  });
}

async function readImageRecord(image) {
  const response = await fetch("/api/image-info/read", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: await imageDataUrl(image), name: image.name }),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `读取 ${image.name} 的生成参数失败（HTTP ${response.status}）`);
  return payload.info || null;
}

// The installed ADetailer detectors: a recorded detector is only written when it names one of them,
// exactly as the image reader decides. An unreachable service means none, never a guess.
async function readDetectorCatalog() {
  try {
    const response = await fetch("/api/inference/adetailer/models", { cache: "no-store" });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) return { models: [], ready: false };
    return { models: Array.isArray(payload.models) ? payload.models : [], ready: payload.available === true };
  } catch {
    return { models: [], ready: false };
  }
}

function NumericField({ label, value, onChange, min, max, step = 1, disabled = false }) {
  return <label className="gallery-field"><span>{label}</span><input type="number" value={value} min={min} max={max} step={step} disabled={disabled} onChange={(event) => {
    const next = Number(event.target.value);
    if (Number.isFinite(next)) onChange(Math.max(min ?? next, Math.min(max ?? next, next)));
  }} /></label>;
}

function GalleryChainSlider({ label, value, min, max, step = 1, inputStep = step, fixed, suffix = "", disabled = false, onChange }) {
  const numeric = Number(value);
  const display = Number.isFinite(numeric) ? (Number.isInteger(fixed) ? numeric.toFixed(fixed) : numeric) : "--";
  const commit = (next) => {
    const parsed = Number(next);
    if (!Number.isFinite(parsed)) return;
    onChange(Math.max(min, Math.min(max, parsed)));
  };
  return <label className="gallery-chain-slider">
    <span><b>{label}</b><em>{display}{suffix}</em></span>
    <input type="range" value={Number.isFinite(numeric) ? numeric : min} min={min} max={max} step={step} disabled={disabled} aria-label={label} onChange={(event) => commit(event.target.value)} />
    <input type="number" value={Number.isFinite(numeric) ? numeric : min} min={min} max={max} step={inputStep} disabled={disabled} aria-label={`${label}数值`} onChange={(event) => commit(event.target.value)} />
  </label>;
}

function GalleryChainNumberInput({ ariaLabel, value, min, max, step = 1, integer = false, disabled = false, onChange }) {
  return <input
    type="number"
    className="bounded-number"
    aria-label={ariaLabel}
    value={value ?? ""}
    min={min}
    max={max}
    step={integer ? 1 : step}
    disabled={disabled}
    onChange={(event) => {
      const next = Number(event.target.value);
      if (!Number.isFinite(next)) return;
      onChange(Math.max(min, Math.min(max, integer ? Math.round(next) : next)));
    }}
  />;
}

function ToggleField({ label, detail, checked, onChange, disabled = false }) {
  const locked = disabled && !checked;
  return <label className={`gallery-toggle-field ${checked ? "active" : ""} ${disabled ? "disabled" : ""}`}><span><strong>{label}</strong><small>{detail}</small></span><input type="checkbox" checked={checked} disabled={locked} onChange={(event) => { if (!disabled || !event.target.checked) onChange(event.target.checked); }} /><i /></label>;
}

// How many left-out items the notice lists before it counts the rest.
const RECORD_NOTICE_ITEMS = 6;

/** What reading an added picture's record did to the card, and a way back. */
function GalleryRecordNotice({ record, onUndo, onDismiss }) {
  const noticeRef = useRef(null);
  // A picture can be added while the parameters are scrolled down to the chain; the outcome is
  // brought into view once it is known, and left where it is when it is already visible.
  useEffect(() => {
    if (record.state !== "reading") noticeRef.current?.scrollIntoView({ block: "nearest" });
  }, [record]);
  const dismiss = <button type="button" className="gallery-editor-record-dismiss" aria-label="关闭图片参数提示" title="关闭" onClick={onDismiss}><X size={12} /></button>;
  if (record.state === "reading") {
    return <div ref={noticeRef} className="gallery-editor-record reading" role="status" aria-live="polite">
      <RefreshCw className="spin" size={13} />
      <div><strong>正在读取图片中的生成参数</strong><small>{record.count > 1 ? `${record.name} 等 ${record.count} 张` : record.name}</small></div>
    </div>;
  }
  if (record.state === "none" || record.state === "error") {
    return <div ref={noticeRef} className={`gallery-editor-record ${record.state}`} role="status" aria-live="polite">
      {record.state === "error" ? <AlertTriangle size={13} /> : <FileSearch size={13} />}
      <div>
        <strong>{record.state === "error" ? "无法读取图片中的生成参数" : "图片中没有可读取的生成参数"}</strong>
        <small>{record.state === "error" ? record.message : `${record.count > 1 ? `${record.name} 等 ${record.count} 张` : record.name}：参数保持不变`}</small>
      </div>
      {dismiss}
    </div>;
  }
  // Everything the picture recorded that the card could not take, each with its reason.
  const left = [
    ...record.missing.map((item) => `${item.kind === "checkpoint" ? "底模" : "LoRA"} ${item.label || item.name}：${item.reason}`),
    ...record.unapplied,
    ...record.skipped.map((item) => `${item}：本工作区不支持`),
  ];
  const shown = left.slice(0, RECORD_NOTICE_ITEMS);
  return <div ref={noticeRef} className="gallery-editor-record filled" role="status" aria-live="polite">
    <Check size={13} />
    <div>
      <strong>已从 {record.name} 读取生成参数{record.source ? ` · ${record.source}` : ""}</strong>
      <small>已填入：{record.filled.length ? record.filled.join("、") : "无"}{record.engineChanged ? `；引擎切换为 ${record.engine}` : ""}</small>
      {record.notes.length > 0 && <small>{record.notes.join("；")}</small>}
      {shown.length > 0 && <ul aria-label="未填入的项目">
        {shown.map((item) => <li key={item}>{item}</li>)}
        {left.length > shown.length && <li>另有 {left.length - shown.length} 项未填入</li>}
      </ul>}
    </div>
    <button type="button" className="gallery-editor-record-undo" title="恢复读取图片之前的参数" onClick={onUndo}>撤销</button>
    {dismiss}
  </div>;
}

/**
 * One post-processing stage in the card editor: resample units, ADetailer and RTX VSR all fold the
 * same way. Collapsed, the header still says what the stage will do; the header itself folds the
 * card, while its switch, order arrows and delete keep their own clicks.
 */
function GalleryChainCard({ kind, title, detail, enabled, expanded, disabled = false, onToggleEnabled, onToggleExpanded, actions, children }) {
  const foldFromHeader = (event) => {
    if (event.target.closest("button, input, label, select, textarea, a")) return;
    onToggleExpanded();
  };
  return <section className={`postprocess-stage-card gallery-resample-card ${enabled ? "enabled" : ""} ${expanded ? "expanded" : ""}`}>
    <header className="gallery-resample-card-header" onClick={foldFromHeader}>
      <span>{kind}</span>
      <strong title={title}>{title}</strong>
      <small>{detail}</small>
      <ToggleField label="启用" detail="" checked={enabled} disabled={disabled} onChange={onToggleEnabled} />
      <button type="button" className="gallery-chain-card-fold" aria-label={`${expanded ? "收起" : "展开"}${title}`} title={expanded ? "收起" : "展开"} aria-expanded={expanded} onClick={onToggleExpanded}><ChevronDown size={13} /></button>
      {actions}
    </header>
    {expanded && children}
  </section>;
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
function FocusImage({ image, alt, style }) {
  const [loaded, setLoaded] = useState(false);
  const preview = thumbUrl(image);
  const hasPreview = preview && preview !== image.url;
  useEffect(() => setLoaded(false), [image.url]);
  return (
    <div className={`gallery-focus-frame ${loaded ? "ready" : ""}`} style={style}>
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

/**
 * The prompt as submitted. A card recorded since presets became switches carries the whole thing;
 * an older one is rebuilt from the only piece it kept, the enabled groups' prefix ahead of the text.
 */
function effectivePrompt(settings) {
  if (typeof settings.composedPrompt === "string" && settings.composedPrompt) return settings.composedPrompt;
  return settings.loraGroupPrompt
    ? composeGroupPrompt([{ id: "recorded", name: "", enabled: true, presetPrompt: settings.loraGroupPrompt, members: [] }], settings.positive)
    : settings.positive;
}

function effectiveNegativePrompt(settings) {
  return typeof settings.composedNegative === "string" && settings.composedNegative ? settings.composedNegative : settings.negative;
}

function PromptBlock({ tone, label, value, onNotice }) {
  return <section className={`gallery-prompt ${tone}`}>
    <header><span>{label}</span><button type="button" disabled={!value} onClick={async () => onNotice(await copyText(value) ? `${label}已复制` : "复制失败")}><Copy size={13} />复制</button></header>
    <p>{value || "未填写"}</p>
  </section>;
}

function PromptLibraryDialog({ entry, onClose, onSaved }) {
  const [title, setTitle] = useState(entry?.title || "");
  const [positivePrompt, setPositivePrompt] = useState(entry?.positive_prompt || "");
  const [negativePrompt, setNegativePrompt] = useState(entry?.negative_prompt || "");
  const [notes, setNotes] = useState(entry?.notes || "");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const dialogRef = useDialogLifecycle(true, onClose, "", !busy);
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
        <label className="gallery-field prompt-positive"><span>正向 Prompt</span><textarea value={positivePrompt} onChange={(event) => setPositivePrompt(event.target.value)} placeholder="主体、风格、构图、光线等" /></label>
        <label className="gallery-field prompt-negative"><span>负向 Prompt</span><textarea value={negativePrompt} onChange={(event) => setNegativePrompt(event.target.value)} placeholder="不希望出现的内容" /></label>
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

function GalleryCardTile({ card, tileIndex = 0, onOpen, onMenu, dragState, reorderBusy, onReorderStart, slotShift = 0 }) {
  const settings = normalizedSettings(card.settings);
  const first = card.images?.[0];
  // This component only decides whether a press is a click, a scroll or the start of a sort. Once
  // it is a sort the page owns the gesture on `window`, because the list re-flows on release and a
  // card that changes column is unmounted and remounted -- taking its pointer capture, and any
  // listener bound to it, out of the document mid-drag.
  const gesture = useRef(null);
  const swallowClick = useRef(false);
  const swallowTimer = useRef(0);
  const cardRef = useRef(null);
  const [holding, setHolding] = useState(false);
  const isDragging = dragState?.cardId === card.id;

  const swallowNextClick = () => {
    swallowClick.current = true;
    window.clearTimeout(swallowTimer.current);
    swallowTimer.current = window.setTimeout(() => { swallowClick.current = false; }, GALLERY_CLICK_SWALLOW_MS);
  };

  const endPress = (releaseCapture = true) => {
    const current = gesture.current;
    gesture.current = null;
    setHolding(false);
    if (!current) return;
    if (current.timer) window.clearTimeout(current.timer);
    if (releaseCapture && current.node.hasPointerCapture?.(current.pointerId)) current.node.releasePointerCapture(current.pointerId);
  };

  useEffect(() => () => {
    if (gesture.current?.timer) window.clearTimeout(gesture.current.timer);
    window.clearTimeout(swallowTimer.current);
  }, []);

  // `touch-action` is read when the gesture begins, long before the hold has decided this press is
  // a sort, so the page is held still here instead -- and only while a drag is actually running.
  useEffect(() => {
    const node = cardRef.current;
    if (!node) return undefined;
    const block = (event) => { if (dragState?.cardId === card.id) event.preventDefault(); };
    node.addEventListener("touchmove", block, { passive: false });
    return () => node.removeEventListener("touchmove", block);
  }, [dragState?.cardId, card.id]);

  const handOver = (current) => {
    if (gesture.current !== current || current.declined) return;
    swallowNextClick();
    // The page takes the pointer from here; this node may be re-rendered and must not be the thing
    // the gesture depends on.
    endPress();
    // A collection with one card refuses the sort and says so once, rather than once per move.
    if (onReorderStart(card, { x: current.x, y: current.y, pointerId: current.pointerId, pointerType: current.pointerType }) === false) {
      current.declined = true;
    }
  };

  const pointerDown = (event) => {
    if (event.button !== 0 || !event.isPrimary || reorderBusy) return;
    const node = event.currentTarget;
    const pointerId = event.pointerId;
    node.setPointerCapture(pointerId);
    const coarse = galleryPointerIsCoarse(event.pointerType);
    setHolding(coarse);
    const next = {
      node, pointerId, pointerType: event.pointerType,
      startX: event.clientX, startY: event.clientY, x: event.clientX, y: event.clientY,
      startedAt: Date.now(), declined: false, timer: 0,
    };
    // A hold can complete with the finger perfectly still, which produces no further events.
    if (coarse) next.timer = window.setTimeout(() => handOver(next), GALLERY_DRAG_HOLD_MS);
    gesture.current = next;
  };

  const pointerMove = (event) => {
    const current = gesture.current;
    if (!current || event.pointerId !== current.pointerId) return;
    current.x = event.clientX;
    current.y = event.clientY;
    const intent = galleryPressIntent({
      pointerType: current.pointerType,
      distance: Math.hypot(event.clientX - current.startX, event.clientY - current.startY),
      elapsed: Date.now() - current.startedAt,
    });
    if (intent === "drag") handOver(current);
    else if (intent === "release") {
      // A finger that wandered during the hold is scrolling. Give the gesture back to the page, and
      // make sure the press does not also open the card on the way out.
      swallowNextClick();
      endPress();
    }
  };

  const contextMenu = (event) => {
    if (event.button !== 2) return;
    event.preventDefault();
    endPress();
    if (!reorderBusy && !dragState) onMenu(card, event);
  };

  return <button
    ref={cardRef}
    type="button"
    className={`gallery-card ${holding ? "holding" : ""} ${isDragging ? "dragging" : ""} ${slotShift ? "yielding" : ""} ${reorderBusy ? "reorder-busy" : ""}`}
    data-gallery-card-id={card.id}
    data-gallery-collection-id={card.collection_id}
    style={{ "--gallery-card-index": Math.min(tileIndex, 12), "--gallery-slot-shift": `${slotShift}px` }}
    aria-grabbed={isDragging}
    title="单击查看；拖动排序（触摸屏先按住再拖）；右键删除"
    onPointerDown={pointerDown}
    onPointerMove={pointerMove}
    onPointerUp={() => endPress()}
    onPointerCancel={() => endPress()}
    onContextMenu={contextMenu}
    onClick={(event) => {
      // The click that ends a drag is consumed here rather than waiting out a timer, so a touch
      // click arriving late cannot slip through and open the card the user was moving.
      if (swallowClick.current || isDragging) {
        swallowClick.current = false;
        window.clearTimeout(swallowTimer.current);
        event.preventDefault();
        return;
      }
      onOpen(card);
    }}
  >
    <span className="gallery-card-visual">
      {first ? <img src={thumbUrl(first)} alt="" loading="lazy" decoding="async" /> : <span className="gallery-card-empty"><ImageIcon size={25} /><b>{displayTitle(card)}</b></span>}
      {card.image_count > 1 && <b className="gallery-image-count"><Images size={12} />{card.image_count}</b>}
      <i>{settings.model || "--"}</i>
      <span className="gallery-card-open-hint"><b>{String(tileIndex + 1).padStart(2, "0")}</b><span>查看预览<ChevronRight size={14} /></span></span>
    </span>
    <span className="gallery-card-copy">
      <strong>{displayTitle(card)}</strong>
      <small>{compactName(isSplitEngine(settings.model) ? settings.diffusionModel : settings.checkpoint)} · {settings.size?.width || "--"} × {settings.size?.height || "--"}</small>
      <span><b>{settings.steps || "--"} STEP</b><b>CFG {settings.cfg ?? "--"}</b><b>{settings.loras?.filter((item) => item.enabled !== false).length || 0} LoRA</b></span>
    </span>
  </button>;
}

function useGalleryColumnCount() {
  const getCount = () => typeof window === "undefined" ? 4 : window.innerWidth <= 680 ? 2 : window.innerWidth <= 980 ? 3 : 4;
  const [columnCount, setColumnCount] = useState(getCount);
  useEffect(() => {
    const update = () => setColumnCount(getCount());
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);
  return columnCount;
}

function GalleryCardColumns({ cards, dragState, reorderBusy, onOpen, onMenu, onReorderStart, slotShifts }) {
  const columnCount = useGalleryColumnCount();
  const columns = distributeGalleryCards(cards, columnCount);
  const cardIndexes = new Map(cards.map((card, index) => [card.id, index]));
  return <div className={`gallery-card-grid gallery-columns-${columnCount}${dragState ? " reordering" : ""}`}>
    {columns.map((column, columnIndex) => <div className="gallery-card-column" key={columnIndex}>{column.map((card) => {
      const index = cardIndexes.get(card.id) ?? 0;
      return <GalleryCardTile card={card} tileIndex={index} key={card.id} onOpen={onOpen} onMenu={onMenu} dragState={dragState} reorderBusy={reorderBusy} onReorderStart={onReorderStart} slotShift={slotShifts?.get(card.id) || 0} />;
    })}</div>)}
  </div>;
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
  const [id, setId] = useState(collection?.id || "");
  const [description, setDescription] = useState(collection?.description || "");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const dialogRef = useDialogLifecycle(true, onClose, "", !busy);
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
  return <div className="gallery-dialog-backdrop" onMouseDown={(event) => event.target === event.currentTarget && !busy && onClose()}>
    <form ref={dialogRef} className="gallery-small-dialog" role="dialog" aria-modal="true" aria-labelledby="gallery-collection-dialog-title" tabIndex="-1" onSubmit={save}>
      <header><div><span>COLLECTION</span><h2 id="gallery-collection-dialog-title">{collection ? "编辑收藏夹" : "创建收藏夹"}</h2></div><button type="button" aria-label="关闭收藏夹编辑" disabled={busy} onClick={onClose}><X size={18} /></button></header>
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

function GalleryCardEditor({
  card,
  collectionId,
  collections,
  initialSettings,
  initialImages = [],
  canvasMaxEdge = DEFAULT_CANVAS_MAX_EDGE,
  upscalers = null,
  upscalerRuntimeReady = true,
  upscalersRefreshing = false,
  onRefreshUpscalers = null,
  upscalerDownloadJob = null,
  upscalerDownloadBusy = false,
  upscalerDownloadError = "",
  onDownloadUpscaler = null,
  onRetryUpscaler = null,
  onClose,
  onSaved,
}) {
  const [selectedCollection, setSelectedCollection] = useState(card?.collection_id || collectionId || collections[0]?.id || "");
  const [title, setTitle] = useState(card?.title || "");
  const [settings, setSettings] = useState(() => card
    ? normalizedSettings(card.settings, DEFAULT_SETTINGS)
    : normalizedSettings(undefined, initialSettings));
  const [images, setImages] = useState(() => card
    ? (card.images || []).map((image, index) => {
      const cardSettings = normalizedSettings(card.settings, DEFAULT_SETTINGS);
      const imageHires = selectedImageHiresSeed(cardSettings, index);
      const hasImageHires = Array.isArray(card.settings?.imageHiresSeedModes) && index < card.settings.imageHiresSeedModes.length;
      return { kind: "existing", id: image.id, name: image.name, url: image.url, seed: card.settings?.imageSeeds?.[index] ?? "", hiresSeedMode: hasImageHires ? imageHires.seedMode : undefined, hiresSeed: hasImageHires ? imageHires.seed : undefined };
    })
    : initialImages);
  const [checkpoints, setCheckpoints] = useState([]);
  const [splitAssets, setSplitAssets] = useState({ diffusion_model: [], text_encoder: [], text_encoder_2: [], vae: [] });
  const [loraOpen, setLoraOpen] = useState(false);
  const [outputOpen, setOutputOpen] = useState(false);
  const [activeImage, setActiveImage] = useState(0);
  const [imageDrag, setImageDrag] = useState(false);
  const [imageReading, setImageReading] = useState(false);
  const [imageError, setImageError] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [modelCatalogError, setModelCatalogError] = useState("");
  // What reading the added pictures' generation records did: reading, filled (with an undo), found
  // nothing, or failed. The card's settings are only ever changed through this, never silently.
  const [record, setRecord] = useState(null);
  const dialogRef = useDialogLifecycle(!loraOpen, onClose, "", !busy);
  const fileInput = useRef(null);
  const imageDragDepth = useRef(0);
  const imagesRef = useRef(images);
  imagesRef.current = images;
  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  const busyRef = useRef(busy);
  busyRef.current = busy;
  const recordToken = useRef(0);
  const shownIndex = Math.min(activeImage, Math.max(0, images.length - 1));
  const shownImage = images[shownIndex] || null;
  const promptTagLoras = [...extractPromptLoras(settings.positive), ...extractPromptLoras(settings.negative)];
  const setField = (key, value) => setSettings((current) => busy ? current : ({ ...current, [key]: value }));
  const setNested = (group, key, value) => setSettings((current) => busy ? current : ({ ...current, [group]: { ...current[group], [key]: value } }));
  // Every engine that mounts separate component files -- Anima, both Flux generations and Krea 2 --
  // not only Anima, which was the only one when this editor was written.
  const isSplit = isSplitEngine(settings.model);
  const isFlux = settings.model === "Flux";
  // Guidance the selected engine refuses: distilled Flux takes none, CFG-Zero* needs a
  // flow-matching engine, Krea 2 has no PAG. One rule set, shared with the workspace.
  const guidanceBlocked = engineGuidance(settings.model, settings.guidance) !== settings.guidance;
  // MOSS offers ComfyUI's exact updates only; every other engine keeps the shared vocabulary.
  const visibleSamplers = isMossEngine(settings.model) ? MOSS_SAMPLERS : SAMPLERS;
  const visibleSchedulers = isMossEngine(settings.model) ? MOSS_SCHEDULERS : SCHEDULERS;
  const changeEngine = (nextModel) => setSettings((current) => {
    const next = {
      ...current,
      model: nextModel,
      checkpoint: "",
      diffusionModel: "",
      textEncoder: "",
      textEncoder2: "",
      vae: "",
      loras: [],
      sampler: isMossEngine(nextModel) ? mossSamplerFor(current.sampler) : SAMPLERS.includes(current.sampler) ? current.sampler : isSplitEngine(nextModel) ? "euler" : "dpmpp_2m",
      scheduler: isMossEngine(nextModel) ? mossSchedulerFor(current.scheduler) : SCHEDULERS.includes(current.scheduler) ? current.scheduler : isSplitEngine(nextModel) ? "simple" : "karras",
      guidance: engineGuidance(nextModel, current.guidance),
    };
    if (isMossEngine(nextModel)) next.mossSampling = normalizeMossSampling(current.mossSampling);
    else delete next.mossSampling;
    return next;
  });
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
            text_encoder_2: payload.assets?.text_encoder_2?.models || [],
            vae: payload.assets?.vae?.models || [],
          });
        } else {
          setCheckpoints(payload.models || []);
          setSplitAssets({ diffusion_model: [], text_encoder: [], text_encoder_2: [], vae: [] });
        }
      })
      .catch((fetchError) => {
        if (fetchError.name === "AbortError") return;
        setCheckpoints([]);
        setSplitAssets({ diffusion_model: [], text_encoder: [], text_encoder_2: [], vae: [] });
        setModelCatalogError(fetchError.message || "模型目录不可用");
      });
    return () => controller.abort();
  }, [settings.model]);
  const chainIssue = resampleChainIssue(settings.postprocessChain, {
    baseSteps: settings.steps,
    engine: settings.model,
    nativeEngines: NATIVE_STEP_ENGINES,
    upscalers: upscalerRuntimeReady ? upscalers : [],
    size: settings.size,
  });
  // A picture made by this program, SD-WebUI/Forge or ComfyUI says how it was made. The first added
  // picture with a readable record fills the card; every read picture keeps its own recorded seed.
  // The newest addition wins: a read still running for an earlier one stops writing.
  const readImageRecords = async (added) => {
    const readable = added.filter(carriesPngRecord).slice(0, GALLERY_IMAGE_INFO_READ_LIMIT);
    if (!readable.length) return;
    const token = ++recordToken.current;
    setRecord({ state: "reading", name: readable[0].name, count: readable.length });
    const seeds = new Map();
    let filled = false;
    let failure = "";
    let detectors = null;
    for (const image of readable) {
      let info = null;
      try {
        info = await readImageRecord(image);
      } catch (readError) {
        failure ||= readError.message;
      }
      if (token !== recordToken.current) return;
      if (!imageInfoHasGenerationData(info)) continue;
      const seed = imageInfoRecordedSeed(info);
      if (seed) seeds.set(image.localKey, seed);
      if (filled || busyRef.current) continue;
      detectors ||= await readDetectorCatalog();
      if (token !== recordToken.current) return;
      const previous = settingsRef.current;
      const result = galleryCardFromImageInfo(previous, info, {
        canvasMaxEdge,
        upscalers: upscalers?.models || [],
        upscalerRuntimeReady,
        detectors: detectors.models,
        detectorRuntimeReady: detectors.ready,
      });
      if (!result) continue;
      filled = true;
      setSettings(result.settings);
      setRecord({ ...result, state: "filled", name: image.name, previous });
    }
    if (seeds.size) {
      setImages((current) => current.map((image) => (image.localKey && seeds.has(image.localKey) && !image.seed ? { ...image, seed: seeds.get(image.localKey) } : image)));
    }
    if (!filled) {
      setRecord(failure
        ? { state: "error", message: failure }
        : { state: "none", name: readable[0].name, count: readable.length });
    }
  };
  const undoRecord = () => {
    if (busy || !record?.previous) return;
    setSettings(record.previous);
    setRecord(null);
  };
  // Pictures dropped on the add-card tile arrive with the editor; they are read like any other.
  useEffect(() => {
    if (!card && initialImages.length) void readImageRecords(initialImages);
    return () => { recordToken.current += 1; };
    // Once per editor: `initialImages` is the drop that opened it.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const addFiles = async (fileList) => {
    if (busy || imageReading) return;
    setImageError("");
    setImageReading(true);
    let imported = [];
    try {
      // The same import the add-card tile uses: PNG/JPEG/WebP/GIF, 20 MiB each, 100 MiB per batch.
      imported = await filesToNewImages(fileList);
      setActiveImage(imagesRef.current.length);
      setImages((current) => [...current, ...imported]);
    } catch (readError) {
      setImageError(`图片未加入：${readError.message}`);
    } finally {
      setImageReading(false);
    }
    if (imported.length) void readImageRecords(imported);
  };
  // Only a drag that carries files is an image drop. Text dragged over a prompt box, or a link
  // dragged in from a page, is left to the browser and never read as a picture.
  const carriesFiles = (event) => [...(event.dataTransfer?.types || [])].includes("Files");
  const imageDragEnter = (event) => {
    if (!carriesFiles(event) || busy) return;
    event.preventDefault();
    imageDragDepth.current += 1;
    setImageDrag(true);
  };
  const imageDragOver = (event) => {
    if (!carriesFiles(event) || busy) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
  };
  const imageDragLeave = (event) => {
    if (!carriesFiles(event)) return;
    imageDragDepth.current = Math.max(0, imageDragDepth.current - 1);
    if (!imageDragDepth.current) setImageDrag(false);
  };
  const imageDrop = (event) => {
    if (!carriesFiles(event) || busy) return;
    event.preventDefault();
    event.stopPropagation();
    imageDragDepth.current = 0;
    setImageDrag(false);
    void addFiles(event.dataTransfer.files);
  };
  const addOutputImages = (files) => {
    setOutputOpen(false);
    setImageError("");
    const current = imagesRef.current;
    const known = new Set(current.filter((image) => image.kind === "output").map((image) => image.assetId));
    const additions = files.filter((file) => !known.has(file.id)).map((file) => ({ kind: "output", localKey: imageLocalKey(), assetId: file.id, name: file.name, url: file.url }));
    if (!additions.length) return;
    setActiveImage(current.length);
    setImages([...current, ...additions]);
    void readImageRecords(additions);
  };
  const moveImage = (index, direction) => {
    const current = imagesRef.current;
    const target = index + direction;
    if (busy || target < 0 || target >= current.length) return;
    const next = [...current];
    [next[index], next[target]] = [next[target], next[index]];
    setImages(next);
    setActiveImage((selected) => (selected === index ? target : selected === target ? index : selected));
  };
  const removeImage = (index) => {
    const current = imagesRef.current;
    if (busy || index < 0 || index >= current.length) return;
    setImages(current.filter((_, itemIndex) => itemIndex !== index));
    setActiveImage((selected) => Math.max(0, Math.min(current.length - 2, selected > index ? selected - 1 : selected)));
  };
  const setStage = (id, updates) => setSettings((current) => (busy ? current : { ...current, postprocessChain: updateChainStage(current.postprocessChain, id, updates) }));
  const save = async (event) => {
    event.preventDefault();
    if (!selectedCollection || busy) return;
    if (modelCatalogError) {
      setError(`模型目录不可用：${modelCatalogError}`);
      return;
    }
    if (isSplit) {
      const required = [
        ["diffusionModel", "扩散模型", splitAssets.diffusion_model],
        ["textEncoder", isFlux ? "CLIP-L 文本编码器" : "文本编码器", splitAssets.text_encoder],
        ...(isFlux ? [["textEncoder2", "T5-XXL 文本编码器", splitAssets.text_encoder_2]] : []),
        ["vae", "VAE", splitAssets.vae],
      ];
      const unavailable = required.find(([field, , catalog]) => !settings[field] || !catalog.some((item) => item.value === settings[field]));
      if (unavailable) {
        setError(`请选择当前目录中可用的 ${settings.model} ${unavailable[1]}`);
        return;
      }
    }
    if (guidanceBlocked) {
      setError(`${settings.model} 不支持${GUIDANCE.find(([id]) => id === settings.guidance)?.[1] || "当前引导增强"}，请切换为可用选项`);
      return;
    }
    if (chainIssue) {
      setError(`后处理链不可保存：${chainIssue}`);
      return;
    }
    setBusy(true);
    setError("");
    // The first unit that redraws is the one an output's recorded seed belongs to.
    const firstRedrawUnit = resampleUnits(settings.postprocessChain).find((unit) => unit.enabled && unit.denoise > 0) || null;
    const safeSettings = {
      ...settings,
      seed: normalizeUint64Seed(settings.seed, "0"),
      postprocessChain: settings.postprocessChain.map((stage) => (stage.type === "resample"
        ? { ...stage, ...normalizeHiresSeed(stage.seedMode, stage.seed) }
        : stage)),
    };
    const savedSettings = isSplit ? {
      ...safeSettings,
      checkpoint: "",
      textEncoder2: isFlux ? settings.textEncoder2 || "" : "",
      sampler: SAMPLERS.includes(settings.sampler) ? settings.sampler : "euler",
      scheduler: SCHEDULERS.includes(settings.scheduler) ? settings.scheduler : "simple",
      guidance: settings.guidance,
    } : {
      ...safeSettings,
      diffusionModel: "",
      textEncoder: "",
      textEncoder2: "",
      vae: "",
    };
    const payload = {
      collection_id: selectedCollection,
      title: title.trim() || null,
      // An edited card keeps the revision it was read at; a new card is stamped with the current one.
      settings: withGalleryLoraRevision({
        ...savedSettings,
        imageSeeds: images.map((image) => normalizeUint64Seed(image.seed, "")),
        imageHiresSeedModes: images.map((image) => normalizeHiresSeed(image.hiresSeedMode ?? firstRedrawUnit?.seedMode, image.hiresSeed ?? firstRedrawUnit?.seed).seedMode),
        imageHiresSeeds: images.map((image) => normalizeHiresSeed(image.hiresSeedMode ?? firstRedrawUnit?.seedMode, image.hiresSeed ?? firstRedrawUnit?.seed).seed),
      }),
      // An output picture travels as its `asset_id` only; the server copies the file it names.
      images: images.map((image) => image.kind === "existing"
        ? { gallery_image_id: image.id }
        : image.kind === "output" ? { asset_id: image.assetId, name: image.name } : { data_url: image.dataUrl, name: image.name }),
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
  return <div className="gallery-editor-backdrop" onMouseDown={(event) => event.target === event.currentTarget && !busy && onClose()}>
    <form ref={dialogRef} className="gallery-card-editor gallery-card-editor-split" role="dialog" aria-modal="true" aria-labelledby="gallery-card-editor-title" tabIndex="-1" onSubmit={save}>
      <header><div><span>CURATED CARD EDITOR</span><h2 id="gallery-card-editor-title">{card ? "编辑精选卡片" : "添加精选卡片"}</h2><p>图片可选，参数可独立于当前生图工作区保存</p></div><button type="button" aria-label="关闭卡片编辑器" disabled={busy} onClick={onClose}><X size={20} /></button></header>
      <div className="gallery-editor-split">
        {/* The picture is the card, so it gets the fixed left half: whole, at any aspect ratio,
            with the configuration scrolling beside it rather than pushing it off screen. */}
        <section className="gallery-editor-stage" aria-label="卡片图片">
          <div
            className={`gallery-editor-preview ${imageDrag ? "drop-active" : ""} ${imageReading ? "drop-reading" : ""} ${shownImage ? "" : "empty"}`}
            role="button"
            tabIndex={busy ? -1 : 0}
            aria-label={shownImage ? `当前图片 ${shownImage.name}；点击添加图片，或拖入本地图片` : "点击选择本地图片，或把图片拖到这里"}
            aria-busy={imageReading}
            onClick={() => !busy && fileInput.current?.click()}
            onKeyDown={(event) => { if (!busy && (event.key === "Enter" || event.key === " ")) { event.preventDefault(); fileInput.current?.click(); } }}
            onDragEnter={imageDragEnter}
            onDragOver={imageDragOver}
            onDragLeave={imageDragLeave}
            onDrop={imageDrop}
          >
            {shownImage
              ? <img src={shownImage.kind === "new" || shownImage.kind === "output" ? shownImage.url : shownImage.url || thumbUrl(shownImage)} alt={shownImage.name || "卡片图片"} decoding="async" />
              : <div className="gallery-editor-preview-empty"><ImageIcon size={30} /><strong>拖入或点击添加图片</strong><small>{title.trim() || settings.positive.trim() ? "不添加图片时，卡片显示文字概括" : "图片可不添加；未填写标题和 Prompt 时显示 No image"}</small></div>}
            {(imageDrag || imageReading) && <div className="gallery-editor-drop-overlay">{imageReading ? <RefreshCw className="spin" size={24} /> : <Upload size={24} />}<strong>{imageReading ? "正在读取图片" : "释放以加入卡片"}</strong><small>PNG · JPEG · WebP · GIF · 单张 20 MiB · 单次 100 MiB</small></div>}
            {shownImage && images.length > 1 && <span className="gallery-editor-preview-count">{shownIndex + 1} / {images.length}</span>}
          </div>
          <input ref={fileInput} type="file" hidden multiple accept="image/png,image/jpeg,image/webp,image/gif" onChange={(event) => { void addFiles(event.target.files); event.target.value = ""; }} />
          <div className="gallery-editor-stage-actions">
            <button type="button" disabled={busy || imageReading} onClick={() => fileInput.current?.click()}><Upload size={13} />添加本地图片</button>
            <button type="button" disabled={busy} onClick={() => setOutputOpen(true)}><FolderOpen size={13} />从 output 选择</button>
          </div>
          {imageError && <p className="gallery-form-error" role="alert">{imageError}</p>}
          {images.length > 0 && <ol className="gallery-editor-strip" aria-label="卡片图片顺序">
            {images.map((image, index) => <li key={image.id || image.assetId || `${image.name}-${index}`} className={index === shownIndex ? "active" : ""}>
              <button type="button" className="gallery-editor-strip-thumb" aria-label={`查看第 ${index + 1} 张：${image.name}`} aria-current={index === shownIndex} onClick={() => setActiveImage(index)}>
                <img src={image.kind === "existing" ? thumbUrl(image) : image.url} alt="" loading="lazy" decoding="async" />
                {image.kind === "output" && <i>output</i>}
              </button>
              <div>
                <button type="button" aria-label={`前移图片 ${index + 1}`} disabled={busy || index === 0} onClick={() => moveImage(index, -1)}><ArrowLeft size={12} /></button>
                <button type="button" aria-label={`后移图片 ${index + 1}`} disabled={busy || index === images.length - 1} onClick={() => moveImage(index, 1)}><ArrowRight size={12} /></button>
                <button type="button" aria-label={`删除图片 ${index + 1}`} disabled={busy} onClick={() => removeImage(index)}><Trash2 size={12} /></button>
              </div>
            </li>)}
          </ol>}
        </section>
      <fieldset className="gallery-editor-body" disabled={busy}>
        {record && <GalleryRecordNotice record={record} onUndo={undoRecord} onDismiss={() => setRecord(null)} />}
        <section className="gallery-editor-media">
          <div className="gallery-editor-section-head"><div><span>01</span><strong>卡片信息</strong></div></div>
          <div className="gallery-editor-grid two"><label className="gallery-field"><span>收藏夹</span><select value={selectedCollection} onChange={(event) => setSelectedCollection(event.target.value)}>{collections.map((item) => <option value={item.id} key={item.id}>{item.id}</option>)}</select></label><label className="gallery-field"><span>概括标题（可选）</span><input value={title} maxLength={160} onChange={(event) => setTitle(event.target.value)} placeholder="留空时使用正向 Prompt 概括" /></label></div>
        </section>

        <section>
          <div className="gallery-editor-section-head"><div><span>02</span><strong>模型组件与 LoRA</strong></div><button type="button" onClick={() => setLoraOpen(true)}><Layers3 size={13} />LoRA 管理器</button></div>
          <div className={`gallery-editor-grid ${isSplit ? "four" : "three"}`}>
            <label className="gallery-field"><span>模型引擎</span><select value={settings.model} onChange={(event) => changeEngine(event.target.value)}><option value="SD">SD</option><option value="iL">iL / SDXL</option><option value="Anima">Anima</option><option value="Flux">FLUX.1</option><option value="Flux2">FLUX.2</option><option value="Krea2">Krea 2</option><option value="MOSS">MOSS</option></select></label>
            {isSplit ? [["diffusionModel", "扩散模型", splitAssets.diffusion_model], ["textEncoder", isFlux ? "CLIP-L 文本编码器" : "文本编码器", splitAssets.text_encoder], ...(isFlux ? [["textEncoder2", "T5-XXL 文本编码器", splitAssets.text_encoder_2]] : []), ["vae", "VAE", splitAssets.vae]].map(([key, label, catalog]) => <label className="gallery-field" key={key}><span>{label}</span><select value={settings[key]} onChange={(event) => setField(key, event.target.value)}><option value="">未指定</option>{settings[key] && !catalog.some((item) => item.value === settings[key]) && <option value={settings[key]}>{settings[key]}（未在当前目录发现）</option>}{catalog.map((item) => <option value={item.value} key={item.value}>{item.name}</option>)}</select></label>) : <label className="gallery-field wide"><span>底模选择</span><select value={settings.checkpoint} onChange={(event) => setField("checkpoint", event.target.value)}><option value="">未指定</option>{settings.checkpoint && !checkpoints.some((item) => item.value === settings.checkpoint) && <option value={settings.checkpoint}>{settings.checkpoint}（未在当前目录发现）</option>}{checkpoints.map((item) => <option value={item.value} key={item.value}>{item.name}</option>)}</select></label>}
          </div>
          {modelCatalogError && <p className="gallery-form-error">模型目录扫描失败：{modelCatalogError}</p>}
          <div className="gallery-editor-lora-summary">
            {settings.loras.map((item, index) => <span className={item.enabled === false ? "disabled" : ""} key={`${item.value}-${index}`}><b>{item.name || compactName(item.value)}</b><small>{formatWeight(item.weight ?? 1)}</small></span>)}
            {/* A `<lora:…>` tag in the prompt loads its LoRA when the card runs; it is shown here so the
                card's LoRAs read in one place, and never mounted as well, which the server refuses. */}
            {promptTagLoras.map((item, index) => <span className="prompt-tag" key={`tag-${item.name}-${index}`} title={`由提示词中的 <lora:${item.name}> 标签加载`}><b>{modelDisplayName(item.name)}</b><small>标签 {formatWeight(item.weight)}</small></span>)}
            {!settings.loras.length && !promptTagLoras.length && <p>未挂载 LoRA</p>}
          </div>
        </section>

        <section>
          <div className="gallery-editor-section-head"><div><span>03</span><strong>Prompt</strong></div></div>
          <p className="prompt-syntax-help gallery-prompt-syntax-help">权重语法：<code>(text)</code> = 1.1；<code>(text:1.25)</code> = 显式权重；<code>\(text\)</code> = 字面括号。</p>
          <label className="gallery-field prompt-positive"><span>正向 Prompt</span><textarea value={settings.positive} onChange={(event) => setField("positive", event.target.value)} placeholder="画面主体、风格、构图与光线" /></label>
          <label className="gallery-field prompt-negative"><span>负向 Prompt</span><textarea value={settings.negative} onChange={(event) => setField("negative", event.target.value)} placeholder="不希望出现的内容" /></label>
        </section>

        <details open>
          <summary><SlidersHorizontal size={14} />采样参数与画布</summary>
          <div className="gallery-details-content">
            <div className="gallery-editor-grid four"><NumericField label="采样步数" value={settings.steps} min={1} max={100} onChange={(value) => setField("steps", Math.round(value))} /><NumericField label="CFG" value={settings.cfg} min={0} max={30} step={0.1} onChange={(value) => setField("cfg", value)} /><NumericField label="降噪" value={settings.denoise} min={0} max={1} step={0.01} onChange={(value) => setField("denoise", value)} /><label className="gallery-field"><span>Seed</span><input value={settings.seed} onChange={(event) => setField("seed", event.target.value.replace(/\D/g, ""))} /></label></div>
            <div className="gallery-editor-grid three"><label className="gallery-field"><span>引导增强</span><select value={settings.guidance} aria-invalid={guidanceBlocked} aria-describedby={guidanceBlocked ? "gallery-guidance-note" : undefined} onChange={(event) => { const next = event.target.value; if (engineGuidance(settings.model, next) === next) setField("guidance", next); }}>{GUIDANCE.map(([id, label]) => { const disabled = engineGuidance(settings.model, id) !== id; return <option value={id} key={id} disabled={disabled}>{label}{disabled ? "（当前模型不可用）" : ""}</option>; })}</select></label><label className="gallery-field"><span>采样器</span><select value={settings.sampler} onChange={(event) => setField("sampler", event.target.value)}>{visibleSamplers.map((item) => <option value={item} key={item}>{item}</option>)}</select></label><label className="gallery-field"><span>调度器</span><select value={settings.scheduler} onChange={(event) => setField("scheduler", event.target.value)}>{visibleSchedulers.map((item) => <option value={item} key={item}>{item}</option>)}</select></label></div>
            {settings.guidance === "pag" && <><div className="gallery-editor-grid two"><NumericField label="PAG 强度" value={settings.pag.scale} min={0} max={5} step={0.01} onChange={(value) => setNested("pag", "scale", value)} /><label className="gallery-field"><span>PAG 作用层</span><select value={settings.pag.appliedLayers} onChange={(event) => setNested("pag", "appliedLayers", event.target.value)}><option value="mid">Mid（推荐）</option><option value="all">全部自注意力层（高风险）</option></select></label></div>{settings.pag.appliedLayers === "all" && <p className="gallery-guidance-note">全部层会明显放大对比、描边和色彩。动漫模型建议使用 Mid，并从强度 0.3 开始。</p>}</>}
            {guidanceBlocked && <p id="gallery-guidance-note" className="gallery-guidance-note" role="status">{settings.model} 不支持{GUIDANCE.find(([id]) => id === settings.guidance)?.[1] || "当前引导增强"}：FLUX 两代为蒸馏引导，只能选“无”；CFG-Zero* 需要 Anima 或 Krea 2 的 Flow Matching；Krea 2 不支持 PAG。</p>}
            <div className="gallery-editor-grid two"><NumericField label="单批图片数" value={settings.imagesPerBatch} min={1} max={10} onChange={(value) => setField("imagesPerBatch", Math.round(value))} /><NumericField label="批次数" value={settings.batchCount} min={1} max={20} onChange={(value) => setField("batchCount", Math.round(value))} /></div>
            {clipSkipRange(settings.model) && <div className="gallery-editor-grid two"><NumericField label={`CLIP skip（${clipSkipRange(settings.model).minimum}～${clipSkipRange(settings.model).maximum}）`} value={normalizeClipSkip(settings.model, settings.clipSkip)} min={clipSkipRange(settings.model).minimum} max={clipSkipRange(settings.model).maximum} onChange={(value) => setField("clipSkip", normalizeClipSkip(settings.model, value))} /></div>}
            <div className="gallery-editor-grid three"><NumericField label="画布宽度" value={settings.size.width} min={64} max={effectiveCanvasMaxEdge(canvasMaxEdge, settings.model)} onChange={(value) => setField("size", { ...settings.size, width: Math.round(value / 64) * 64 })} /><NumericField label="画布高度" value={settings.size.height} min={64} max={effectiveCanvasMaxEdge(canvasMaxEdge, settings.model)} onChange={(value) => setField("size", { ...settings.size, height: Math.round(value / 64) * 64 })} /><label className="gallery-field"><span>Seed 模式</span><select value={settings.seedMode} onChange={(event) => setField("seedMode", event.target.value)}><option value="fixed">固定</option><option value="random">随机</option><option value="increment">递增</option><option value="decrement">递减</option></select></label></div>
          </div>
        </details>

        <details>
          <summary><Zap size={14} />后处理链</summary>
          <div className="gallery-details-content">
            <ResampleChainPanel
              chain={settings.postprocessChain}
              onChange={(nextChain) => setSettings((current) => busy ? current : { ...current, postprocessChain: nextChain })}
              engine={settings.model}
              samplers={visibleSamplers}
              schedulers={visibleSchedulers}
              baseSize={settings.size}
              baseSteps={settings.steps}
              baseCfg={settings.cfg}
              upscalers={upscalers || { models: [], builtins: [] }}
              upscalerRuntimeReady={upscalerRuntimeReady}
              upscalersRefreshing={upscalersRefreshing}
              onRefreshUpscalers={onRefreshUpscalers}
              upscalerDownloadJob={upscalerDownloadJob}
              upscalerDownloadBusy={upscalerDownloadBusy}
              upscalerDownloadError={upscalerDownloadError}
              onDownloadUpscaler={onDownloadUpscaler}
              onRetryUpscaler={onRetryUpscaler}
              locked={busy}
              Slider={GalleryChainSlider}
               NumberInput={GalleryChainNumberInput}
               randomSeed={secureRandomUint64Seed}
               chainIssue={chainIssue}
               renderCard={({ title: stageTitle, detail, enabled, expanded, onToggleEnabled, onToggleExpanded, actions, children }) => (
                <GalleryChainCard kind="重采样" title={stageTitle} detail={detail} enabled={enabled} expanded={expanded} disabled={busy} onToggleEnabled={() => onToggleEnabled()} onToggleExpanded={onToggleExpanded} actions={actions}>{children}</GalleryChainCard>
              )}
              renderStage={(stage, index, actions) => {
                const toggleExpanded = () => setStage(stage.id, { expanded: !stage.expanded });
                if (stage.type === "adetailer") {
                  // Every slot exists; the ones switched on are the passes that run, in this order.
                  const units = normalizeADetailerUnits(stage);
                  const active = units.filter((unit) => unit.enabled);
                  const setUnit = (id, updates) => setStage(stage.id, { units: units.map((unit) => (unit.id === id ? { ...unit, ...updates } : unit)) });
                  return <GalleryChainCard kind="细节修复" title="ADetailer" detail={stage.enabled ? adetailerSummary(stage) : "关闭"} enabled={stage.enabled} expanded={stage.expanded} disabled={busy} onToggleEnabled={(value) => setStage(stage.id, { enabled: value })} onToggleExpanded={toggleExpanded} actions={actions}>
                    <div className="gallery-chain-card-body">
                      {active.length === 0 && <p className="gallery-field-detail">没有启用的检测单元</p>}
                      {active.map((unit, position) => <div className="gallery-ade-unit" key={unit.id}>
                        <span>{String(position + 1).padStart(2, "0")}</span>
                        <label className="gallery-field"><span>检测模型</span><input value={unit.detector} placeholder="如 face_yolov8n.pt" onChange={(event) => setUnit(unit.id, { detector: event.target.value })} /></label>
                        <NumericField label="置信度" value={unit.confidence} min={0.05} max={1} step={0.01} onChange={(value) => setUnit(unit.id, { confidence: value })} />
                        <NumericField label="重绘强度" value={unit.denoise} min={0.05} max={1} step={0.01} onChange={(value) => setUnit(unit.id, { denoise: value })} />
                      </div>)}
                    </div>
                  </GalleryChainCard>;
                }
                return <GalleryChainCard kind="RTX 放大" title="RTX VSR" detail={stage.enabled ? `${stage.scale}× · ${String(stage.quality).toUpperCase()}` : "关闭"} enabled={stage.enabled} expanded={stage.expanded} disabled={busy} onToggleEnabled={(value) => setStage(stage.id, { enabled: value })} onToggleExpanded={toggleExpanded} actions={actions}>
                  <div className="gallery-chain-card-body gallery-editor-grid two">
                    <NumericField label="RTX 倍率" value={stage.scale} min={1} max={4} step={0.01} onChange={(value) => setStage(stage.id, { scale: value })} />
                    <label className="gallery-field"><span>RTX 质量</span><select value={stage.quality} onChange={(event) => setStage(stage.id, { quality: event.target.value })}><option value="low">LOW</option><option value="medium">MEDIUM</option><option value="high">HIGH</option><option value="ultra">ULTRA</option></select></label>
                  </div>
                </GalleryChainCard>;
              }}
            />
          </div>
        </details>
        {error && <p className="gallery-form-error">{error}</p>}
      </fieldset>
      </div>
      <footer><span>{images.length ? `${images.length} 张图片` : "无图片卡片"} · {settings.loras.length} 个 LoRA · {settings.size.width} × {settings.size.height}</span><div><button type="button" onClick={onClose}>取消</button><button type="submit" className="primary" disabled={busy || !selectedCollection}>{busy ? <RefreshCw className="spin" size={14} /> : <Save size={14} />}保存卡片</button></div></footer>
    </form>
    {loraOpen && <GalleryLoraManager engine={settings.model} value={settings.loras} onClose={() => setLoraOpen(false)} onApply={(next) => { setField("loras", next); setLoraOpen(false); }} />}
    {outputOpen && createPortal(<GalleryOutputDrawer onClose={() => setOutputOpen(false)} onPick={addOutputImages} />, document.body)}
  </div>;
}

// Where a card's parameters land. Prompts, sampling, canvas and the chain belong to the chosen
// page; the model and the LoRA mount are shared by both, and image-to-image keeps its source.
const APPLY_TARGETS = [
  ["t2i", "文生图", "写入文生图页"],
  ["i2i", "图生图", "写入图生图页，来源图保持不变"],
];

function ApplySettingsDialog({ card, onClose, onApply }) {
  // Defaults to Prompt only. Applying everything replaces the model, the whole
  // LoRA mount and every sampling parameter, which is a much larger action than
  // "I want this card's prompt" — the reason people open this dialog.
  const [selected, setSelected] = useState(() => new Set(APPLY_DEFAULT_GROUPS));
  // A card saved from the image page opens on the image page; any other on text-to-image.
  const [target, setTarget] = useState(() => (card.settings?.page === "image" ? "i2i" : "t2i"));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const dialogRef = useDialogLifecycle(true, onClose, "", !busy);
  const toggle = (id) => setSelected((current) => { if (busy) return current; const next = new Set(current); if (next.has(id)) next.delete(id); else next.add(id); return next; });
  const apply = async (groups) => {
    setBusy(true);
    setError("");
    try {
      await onApply(card.settings, groups, { target, page: target === "i2i" ? "image" : "generate" });
      onClose();
    } catch (applyError) {
      setError(applyError.message);
    } finally {
      setBusy(false);
    }
  };
  return <div className="gallery-dialog-backdrop" onMouseDown={(event) => event.target === event.currentTarget && !busy && onClose()}>
    <section ref={dialogRef} className="gallery-apply-dialog" role="dialog" aria-modal="true" aria-labelledby="gallery-apply-dialog-title" tabIndex="-1">
      <header><div><span>APPLY PARAMETERS</span><h2 id="gallery-apply-dialog-title">应用到生图区域</h2><p title={card.title || card.settings?.positive}>{displayTitle(card)}</p></div><button type="button" aria-label="关闭参数应用窗口" disabled={busy} onClick={onClose}><X size={18} /></button></header>
      <div className="gallery-apply-target" role="radiogroup" aria-label="应用目标页面">
        {APPLY_TARGETS.map(([id, label, detail]) => <button type="button" key={id} role="radio" aria-checked={target === id} className={target === id ? "selected" : ""} disabled={busy} title={detail} onClick={() => setTarget(id)}><strong>{label}</strong><small>{detail}</small></button>)}
      </div>
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

function GalleryInspector({ card, settings, isSplit, onEdit, onDelete, onNotice, onApplyOpenChange }) {
  const tilesHires = supportsUsduTiled(settings.model);
  const stageLabels = { resample: "重采样", adetailer: "ADetailer", rtx: "RTX VSR" };
  return <section className="gallery-inspector" aria-label="生成参数">
    <header><div><span>CURATED DETAIL</span><h1 title={card.title || settings.positive}>{displayTitle(card)}</h1><p>{card.collection_id} · 更新于 {formatDate(card.updated_at)}</p></div><div><button type="button" onClick={() => onEdit(card)}><Pencil size={14} />编辑</button><button type="button" className="danger" aria-label="删除精选卡片" onClick={() => onDelete(card)}><Trash2 size={14} /></button></div></header>
    <div className="gallery-inspector-scroll">
      {effectivePrompt(settings) !== settings.positive && (
        <PromptBlock tone="composed" label="实际提交的正向 Prompt" value={effectivePrompt(settings)} onNotice={onNotice} />
      )}
      <PromptBlock tone="positive" label={effectivePrompt(settings) !== settings.positive ? "正向 Prompt（手写部分）" : "正向 Prompt"} value={settings.positive} onNotice={onNotice} />
      {effectiveNegativePrompt(settings) !== settings.negative && (
        <PromptBlock tone="composed" label="实际提交的负向 Prompt" value={effectiveNegativePrompt(settings)} onNotice={onNotice} />
      )}
      <PromptBlock tone="negative" label={effectiveNegativePrompt(settings) !== settings.negative ? "负向 Prompt（手写部分）" : "负向 Prompt"} value={settings.negative} onNotice={onNotice} />
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
      <section className="gallery-detail-section"><header><span>MODEL</span><strong>模型组件与 LoRA</strong></header><dl><div><dt>引擎</dt><dd>{settings.model}</dd></div>{isSplit ? <><div><dt>扩散模型</dt><dd title={settings.diffusionModel}>{compactName(settings.diffusionModel)}</dd></div><div><dt>文本编码器</dt><dd title={settings.textEncoder}>{compactName(settings.textEncoder)}</dd></div><div><dt>VAE</dt><dd title={settings.vae}>{compactName(settings.vae)}</dd></div></> : <div><dt>底模</dt><dd title={settings.checkpoint}>{compactName(settings.checkpoint)}</dd></div>}</dl><div className="gallery-detail-loras">{settings.loras.length ? settings.loras.map((item, index) => { const owner = settings.loraGroups.find((group) => group.id === item.groupId); return <div className={item.enabled === false ? "disabled" : ""} key={`${item.value}-${index}`}><span>{String(index + 1).padStart(2, "0")}</span><strong>{item.name || compactName(item.value)}</strong>{owner && <em className="gallery-lora-group-tag" title={`来自组合 ${owner.name}`}>{owner.name}</em>}<b>{formatWeight(item.weight ?? 1)}</b><small>{item.enabled === false ? "关闭" : "启用"}</small></div>; }) : <p>未使用 LoRA</p>}</div></section>
      <section className="gallery-detail-section"><header><span>SAMPLING</span><strong>采样与画布</strong></header><div className="gallery-stat-grid"><div><span>STEP</span><b>{settings.steps}</b></div><div><span>CFG</span><b>{settings.cfg}</b></div><div><span>DENOISE</span><b>{settings.denoise}</b></div><div><span>SEED</span><b title={settings.seed}>{settings.seed}</b></div><div><span>SIZE</span><b>{settings.size.width} × {settings.size.height}</b></div><div><span>BATCH</span><b>{settings.imagesPerBatch} × {settings.batchCount}</b></div></div><p className="gallery-detail-line">{GUIDANCE.find(([id]) => id === settings.guidance)?.[1] || "无（None）"}{settings.guidance === "pag" ? ` ${settings.pag.scale} / ${settings.pag.appliedLayers === "mid" ? "Mid" : "全部层"}` : ""} · {settings.sampler} · {settings.scheduler}</p></section>
      <section className="gallery-detail-section"><header><span>POST PROCESS</span><strong>后处理链</strong></header><div className="gallery-process-list">{settings.postprocessChain.map((stage, index) => <div className={stage.enabled ? "enabled" : "disabled"} key={stage.id}><span>{index + 1}</span><strong>{stage.type === "resample" ? resampleUnitTitle(settings.postprocessChain, stage) : stageLabels[stage.type]}</strong><b>{stage.enabled ? "ON" : "OFF"}</b><small>{stage.type === "resample" ? `${resampleUnitSummary(stage)}${stage.upscaler ? ` · ${compactName(stage.upscaler)}` : ""}${tilesHires && stage.denoise > 0 ? ` · ${stage.executionMode === "usdu_tiled" ? "USDU 分块重绘" : "整图重绘"}` : ""}` : stage.type === "adetailer" ? `${compactName(stage.units?.[0]?.detector)} · ${(stage.units || []).length} 个单元` : `${stage.scale}x · ${String(stage.quality).toUpperCase()}`}</small></div>)}</div><p className="gallery-detail-line">尺寸链 · {chainSizeLine(settings.size, settings.postprocessChain, settings.model)}</p></section>
      {(() => {
        const detail = settings.postprocessChain.find((stage) => stage.type === "adetailer");
        const unit = detail?.units?.[0];
        if (!unit) return null;
        return <section className="gallery-detail-section"><header><span>ADETAILER DETAIL</span><strong>局部重绘参数</strong></header><dl><div><dt>Prompt</dt><dd>{unit.prompt || "继承主 Prompt"}</dd></div><div><dt>负向</dt><dd>{unit.negativePrompt || "继承主负向"}</dd></div><div><dt>蒙版</dt><dd>{unit.maskMinRatio}–{unit.maskMaxRatio} · blur {unit.maskBlur}</dd></div><div><dt>重绘</dt><dd>{unit.denoise} · {unit.useSteps ? `${unit.steps} steps` : "继承步数"}</dd></div></dl></section>;
      })()}
    </div>
    <footer><button type="button" onClick={() => onApplyOpenChange(true)}><SlidersHorizontal size={15} /><span><strong>应用参数</strong><small>选择性回填到生图区域</small></span><ChevronRight size={15} /></button></footer>
  </section>;
}

function GalleryDetail({ card, cards, onBack, onNavigate, onEdit, onDelete, onApply, onNotice }) {
  const [imageIndex, setImageIndex] = useState(0);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [applyOpen, setApplyOpen] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [zoomOrigin, setZoomOrigin] = useState({ x: 50, y: 50 });
  const panSession = useRef(null);
  const closeViewer = () => {
    if (drawerOpen) {
      setDrawerOpen(false);
      return;
    }
    onBack();
  };
  const dialogRef = useDialogLifecycle(!applyOpen, closeViewer);
  const resetViewport = () => {
    panSession.current = null;
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setZoomOrigin({ x: 50, y: 50 });
  };
  const viewerCards = cards?.length ? cards : [card];
  const cardIndex = Math.max(0, viewerCards.findIndex((item) => item.id === card.id));
  const imageCount = card.images?.length || 0;
  const safeImageIndex = boundedGalleryImageIndex(imageIndex, imageCount);
  const viewerCardKey = viewerCards.map((item) => item.id).join("\u0000");
  useEffect(() => {
    setImageIndex(0);
    setDrawerOpen(false);
    resetViewport();
  }, [card.id]);
  const changeCard = (direction) => {
    if (viewerCards.length < 2) return;
    onNavigate(viewerCards[(cardIndex + direction + viewerCards.length) % viewerCards.length]);
  };
  useEffect(() => {
    if (applyOpen) return undefined;
    const onKeyDown = (event) => {
      if (event.defaultPrevented || ["INPUT", "TEXTAREA", "SELECT"].includes(event.target?.tagName)) return;
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        changeCard(-1);
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        changeCard(1);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [applyOpen, cardIndex, viewerCardKey]);
  const baseSettings = normalizedSettings({ ...card.settings, seed: galleryImageSeed(card.settings, safeImageIndex) });
  // Viewing one image of a grouped card pins the first redrawing unit to that image's own seed.
  const imageUnitSeed = selectedImageHiresSeed(baseSettings, safeImageIndex);
  const pinnedUnitId = resampleUnits(baseSettings.postprocessChain).find((unit) => unit.enabled && unit.denoise > 0)?.id;
  const settings = pinnedUnitId
    ? { ...baseSettings, postprocessChain: updateChainStage(baseSettings.postprocessChain, pinnedUnitId, imageUnitSeed) }
    : baseSettings;
  const image = card.images?.[safeImageIndex];
  const isSplit = isSplitEngine(settings.model);
  const zoomAtPointer = (event) => {
    event.preventDefault();
    const rect = event.currentTarget.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    setZoomOrigin({ x: clampViewerValue(((event.clientX - rect.left) / rect.width) * 100, 0, 100), y: clampViewerValue(((event.clientY - rect.top) / rect.height) * 100, 0, 100) });
    setZoom((current) => {
      const nextZoom = clampViewerValue(current * (event.deltaY < 0 ? 1.18 : 0.84), 1, 5);
      if (nextZoom === 1) setPan({ x: 0, y: 0 });
      return nextZoom;
    });
  };
  const beginPan = (event) => {
    if ((event.pointerType === "mouse" ? event.button !== 2 : !event.isPrimary) || zoom <= 1) return;
    event.preventDefault();
    panSession.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, startX: pan.x, startY: pan.y };
    event.currentTarget.setPointerCapture?.(event.pointerId);
  };
  const movePan = (event) => {
    const current = panSession.current;
    if (!current || current.pointerId !== event.pointerId) return;
    event.preventDefault();
    const rect = event.currentTarget.getBoundingClientRect();
    const limitX = rect.width * (zoom - 1) / 2;
    const limitY = rect.height * (zoom - 1) / 2;
    setPan({ x: clampViewerValue(current.startX + event.clientX - current.x, -limitX, limitX), y: clampViewerValue(current.startY + event.clientY - current.y, -limitY, limitY) });
  };
  const endPan = (event) => {
    if (!panSession.current || panSession.current.pointerId !== event.pointerId) return;
    panSession.current = null;
    if (event.currentTarget.hasPointerCapture?.(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  };
  return <div className={`gallery-focus${drawerOpen ? " inspector-open" : ""}`}>
    <div className="gallery-focus-backdrop" aria-hidden="true" />
    <section ref={dialogRef} className="gallery-focus-shell" role="dialog" aria-modal="true" aria-labelledby="gallery-viewer-title" tabIndex="-1">
      <header className="gallery-viewer-topline"><div><span>NOW VIEWING / {String(cardIndex + 1).padStart(2, "0")} OF {String(viewerCards.length).padStart(2, "0")}</span><h1 id="gallery-viewer-title" title={card.title || settings.positive}>{displayTitle(card)}</h1></div><div><button type="button" className={drawerOpen ? "active" : ""} aria-pressed={drawerOpen} onClick={() => setDrawerOpen((current) => !current)}><SlidersHorizontal size={15} /><span>参数</span></button><button type="button" className="gallery-viewer-reset-zoom" disabled={zoom === 1 && pan.x === 0 && pan.y === 0} title="重置缩放与平移" onClick={resetViewport}><RefreshCw size={14} /><span>{Math.round(zoom * 100)}%</span></button><button type="button" className="gallery-viewer-close" aria-label="关闭图片预览" onClick={onBack}><X size={18} /><span>关闭</span></button></div></header>
      <div className="gallery-viewer-content"><button type="button" className="gallery-viewer-nav previous" aria-label="上一张精选" disabled={viewerCards.length < 2} onClick={() => changeCard(-1)}><ArrowLeft size={18} /></button><figure className="gallery-viewer-figure"><div className={`gallery-viewer-image-viewport${zoom > 1 ? " zoomed" : ""}${panSession.current ? " panning" : ""}`} onWheel={zoomAtPointer} onDoubleClick={() => { if (zoom > 1) resetViewport(); else setZoom(2.5); }} onPointerDown={beginPan} onPointerMove={movePan} onPointerUp={endPan} onPointerCancel={endPan} onContextMenu={(event) => { if (zoom > 1) event.preventDefault(); }}>{image ? <FocusImage key={image.id || image.url} image={image} alt={displayTitle(card)} style={{ "--gallery-viewer-zoom": zoom, "--gallery-viewer-pan-x": `${pan.x}px`, "--gallery-viewer-pan-y": `${pan.y}px`, "--gallery-viewer-origin-x": `${zoomOrigin.x}%`, "--gallery-viewer-origin-y": `${zoomOrigin.y}%` }} /> : <div className="gallery-viewer-empty"><ImageIcon size={42} /><strong>{card.title?.trim() || "无图片精选卡片"}</strong><p>{settings.positive || "未填写 Prompt"}</p></div>}</div><figcaption><div><span>{card.collection_id} / {settings.model} / {settings.size.width} × {settings.size.height} / IMAGE {String(imageCount ? safeImageIndex + 1 : 0).padStart(2, "0")} OF {String(imageCount).padStart(2, "0")}</span><strong>{displayTitle(card)}</strong></div><p>滚轮或双击缩放 · 右键/触摸拖拽 · Esc 关闭</p></figcaption>{imageCount > 1 && <div className="gallery-viewer-thumbs">{card.images.map((item, index) => <button type="button" className={index === safeImageIndex ? "active" : ""} key={item.id} aria-label={`查看图片 ${index + 1}`} onClick={() => { setImageIndex(index); resetViewport(); }}><img src={thumbUrl(item)} alt="" loading="lazy" decoding="async" /><b>{String(index + 1).padStart(2, "0")}</b></button>)}</div>}</figure><button type="button" className="gallery-viewer-nav next" aria-label="下一张精选" disabled={viewerCards.length < 2} onClick={() => changeCard(1)}><ArrowRight size={18} /></button></div>
      <div className="gallery-viewer-drawer" aria-hidden={!drawerOpen}><button type="button" className="gallery-viewer-drawer-backdrop" tabIndex={drawerOpen ? 0 : -1} aria-label="关闭参数面板" onClick={() => setDrawerOpen(false)} /><aside className="gallery-viewer-drawer-sheet" inert={drawerOpen ? undefined : ""}><header><div><span>GENERATION RECORD</span><strong>参数与复现</strong></div><button type="button" aria-label="关闭参数面板" onClick={() => setDrawerOpen(false)}><X size={17} /></button></header><GalleryInspector card={card} settings={settings} isSplit={isSplit} onEdit={onEdit} onDelete={onDelete} onNotice={onNotice} onApplyOpenChange={setApplyOpen} /></aside></div>
    </section>
    {applyOpen && <ApplySettingsDialog card={{ ...card, settings }} onClose={() => setApplyOpen(false)} onApply={onApply} />}
  </div>;
}

export default function GalleryPage({
  currentSettings,
  canvasMaxEdge = DEFAULT_CANVAS_MAX_EDGE,
  focus,
  onApplySettings,
  onApplyPrompt,
  onNotice,
  upscalerCatalog = null,
  upscalerRuntimeReady = true,
  upscalersRefreshing = false,
  onRefreshUpscalers = null,
  upscalerDownloadJob = null,
  upscalerDownloadBusy = false,
  upscalerDownloadError = "",
  onDownloadUpscaler = null,
  onRetryUpscaler = null,
}) {
  const [data, setData] = useState({ collections: [], cards: [] });
  const [activeSection, setActiveSection] = useState("gallery");
  const [selectedCollection, setSelectedCollection] = useState(null);
  const [selectedCard, setSelectedCard] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [collectionDialog, setCollectionDialog] = useState(null);
  const [transferOpen, setTransferOpen] = useState(false);
  const [editor, setEditor] = useState(null);
  const [dragState, setDragState] = useState(null);
  const dragGhostRef = useRef(null);
  const galleryRoomRef = useRef(null);
  const libraryRef = useRef(null);
  const searchBoxRef = useRef(null);
  // The collections nav is pinned, so scrolling slides the search box up behind it. The arrow
  // appears at the moment the search box has gone entirely under the nav's bottom edge -- the point
  // where the tools are no longer reachable without scrolling back.
  const [backToTop, setBackToTop] = useState(false);
  const galleryColumnCount = useGalleryColumnCount();
  const galleryColumnCountRef = useRef(galleryColumnCount);
  galleryColumnCountRef.current = galleryColumnCount;
  const [reorderBusy, setReorderBusy] = useState("");
  const [cardMenu, setCardMenu] = useState(null);
  const [promptEntries, setPromptEntries] = useState([]);
  const [promptLoading, setPromptLoading] = useState(true);
  const [promptError, setPromptError] = useState("");
  const [promptDialog, setPromptDialog] = useState(null);
  const [gallerySearch, setGallerySearch] = useState("");
  const [galleryFilter, setGalleryFilter] = useState("all");
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
  useEffect(() => { const published = () => { void refresh(); }; window.addEventListener(LORA_PUBLICATION_EVENT, published); return () => window.removeEventListener(LORA_PUBLICATION_EVENT, published); }, []);
  useEffect(() => {
    if (!focus || loading) return;
    if (focus.collectionId) { setActiveSection("gallery"); setSelectedCollection(focus.collectionId); }
    if (focus.cardId) setSelectedCard(data.cards.find((item) => item.id === focus.cardId) || null);
  }, [focus, loading, data.cards]);
  const cardsByCollection = new Map(data.collections.map((collection) => [collection.id, data.cards.filter((card) => card.collection_id === collection.id)]));
  const selectedCollectionRecord = data.collections.find((item) => item.id === selectedCollection) || null;
  const searchQuery = gallerySearch.trim().toLocaleLowerCase();
  const filteredCards = data.cards.filter((card) => {
    if (selectedCollection !== null && card.collection_id !== selectedCollection) return false;
    const imageCount = card.images?.length || 0;
    if (galleryFilter === "single" && imageCount !== 1) return false;
    if (galleryFilter === "multi" && imageCount < 2) return false;
    if (galleryFilter === "text" && imageCount !== 0) return false;
    if (!searchQuery) return true;
    const settings = normalizedSettings(card.settings);
    return [card.title, card.collection_id, settings.positive, settings.negative, settings.model, settings.checkpoint, settings.diffusionModel, ...settings.loras.map((item) => item.name || item.value)]
      .some((value) => String(value || "").toLocaleLowerCase().includes(searchQuery));
  });
  const visibleImageCount = filteredCards.reduce((total, card) => total + (card.images?.length || 0), 0);
  const collectionScopes = [null, ...data.collections.map((item) => item.id)];
  const navigateCollection = (direction) => {
    if (collectionScopes.length < 2) return;
    const currentIndex = Math.max(0, collectionScopes.indexOf(selectedCollection));
    setActiveSection("gallery");
    setSelectedCollection(collectionScopes[(currentIndex + direction + collectionScopes.length) % collectionScopes.length]);
    setSelectedCard(null);
  };
  const viewerCards = selectedCard
    ? filteredCards.length ? filteredCards : selectedCollection === null ? data.cards : cardsByCollection.get(selectedCard.collection_id) || [selectedCard]
    : [];
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
  // The whole drag lives here, on `window`, rather than on the card. The list re-flows only on
  // release, but even so the gesture must not depend on a node: a card that changes column is
  // unmounted and remounted, and its pointer capture goes with it. Bound to the window, the drag
  // survives anything React does to the grid.
  const startCardReorder = (card, point) => {
    if (reorderBusy || dragSession.current) return false;
    const order = data.cards.filter((item) => item.collection_id === card.collection_id).map((item) => item.id);
    if (order.length < 2) {
      onNotice("当前收藏夹只有一张卡片，无需排序");
      return false;
    }
    const measure = () => [...document.querySelectorAll(`[data-gallery-card-id][data-gallery-collection-id="${CSS.escape(card.collection_id)}"]`)]
      .map((node) => ({ id: node.dataset.galleryCardId, rect: node.getBoundingClientRect() }));
    const session = {
      cardId: card.id,
      collectionId: card.collection_id,
      pointerId: point.pointerId,
      order: [...order],
      laid: measure(),
      measure,
      x: point.x,
      y: point.y,
      anchor: null,
      raf: 0,
      scrollRaf: 0,
    };
    dragSession.current = session;
    setDragState({ cardId: card.id, collectionId: card.collection_id, anchor: null, shifts: new Map() });
    runDragFrame();
    runDragAutoScroll();
    return true;
  };

  // The ghost follows the pointer by writing a transform straight onto the node once per frame.
  // Re-rendering the page for every pointer sample is what made the drag feel heavy.
  const runDragFrame = () => {
    const session = dragSession.current;
    if (!session) return;
    const ghost = dragGhostRef.current;
    if (ghost) ghost.style.transform = `translate3d(${session.x}px, ${session.y}px, 0)`;
    session.raf = window.requestAnimationFrame(runDragFrame);
  };

  const runDragAutoScroll = () => {
    const session = dragSession.current;
    if (!session) return;
    const scroll = document.querySelector(".gallery-room") || document.querySelector(".gallery-scroll");
    const step = scroll ? galleryAutoScrollStep(session.y, scroll.getBoundingClientRect()) : 0;
    if (step) {
      scroll.scrollBy({ top: step });
      // Scrolling moved every card, so the drop point has to be recomputed against fresh geometry.
      session.laid = session.measure();
      updateDropAnchor(session);
    }
    session.scrollRaf = window.requestAnimationFrame(runDragAutoScroll);
  };

  // Only the anchor and the yielded slot change while the pointer moves. The order is not touched
  // until release, so passing over a card can never carry it away mid-drag.
  const updateDropAnchor = (session) => {
    const anchor = galleryDropAnchor(session.laid, { x: session.x, y: session.y });
    const sameCard = anchor.type === session.anchor?.type && anchor.cardId === session.anchor?.cardId;
    if (sameCard) return;
    session.anchor = anchor;
    const shifts = gallerySlotShifts(session.laid, anchor, session.cardId, galleryColumnCountRef.current);
    setDragState((current) => current ? { ...current, anchor, shifts } : current);
  };

  const stopDragSession = () => {
    const session = dragSession.current;
    dragSession.current = null;
    if (session?.raf) window.cancelAnimationFrame(session.raf);
    if (session?.scrollRaf) window.cancelAnimationFrame(session.scrollRaf);
    setDragState(null);
    return session;
  };

  const cancelCardReorder = () => {
    // Nothing was reordered on the way here, so abandoning is just putting the visuals back.
    stopDragSession();
  };

  useEffect(() => {
    if (!dragState) return undefined;
    const owns = (event) => dragSession.current && (event.pointerId === undefined || event.pointerId === dragSession.current.pointerId);
    const move = (event) => {
      const session = dragSession.current;
      if (!owns(event)) return;
      event.preventDefault();
      session.x = event.clientX;
      session.y = event.clientY;
      updateDropAnchor(session);
    };
    const up = (event) => { if (owns(event)) void finishCardReorder(); };
    const cancel = (event) => { if (owns(event)) cancelCardReorder(); };
    const key = (event) => { if (event.key === "Escape") { event.preventDefault(); cancelCardReorder(); } };
    window.addEventListener("pointermove", move, { passive: false });
    window.addEventListener("pointerup", up);
    window.addEventListener("pointercancel", cancel);
    window.addEventListener("blur", cancelCardReorder);
    window.addEventListener("keydown", key);
    return () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      window.removeEventListener("pointercancel", cancel);
      window.removeEventListener("blur", cancelCardReorder);
      window.removeEventListener("keydown", key);
    };
  }, [dragState?.cardId]);

  const finishCardReorder = async () => {
    const session = stopDragSession();
    if (!session) return;
    // One splice, here, on release. The anchor names a card the user can see, so a filtered view
    // inserts where it looks like it will rather than sweeping hidden cards to one end.
    const index = resolveDropIndex(session.order, session.anchor, session.cardId);
    const next = spliceGalleryOrder(session.order, session.cardId, index);
    // Released where it started: nothing moved, so nothing is sent.
    if (!galleryOrderChanged(session.order, next)) return;
    const original = session.order;
    setData((current) => ({ ...current, cards: applyCollectionCardOrder(current.cards, session.collectionId, next) }));
    setReorderBusy(session.collectionId);
    try {
      const payload = await galleryRequest(`/collections/${encodeURIComponent(session.collectionId)}/card-order`, {
        method: "PUT",
        body: JSON.stringify({ card_ids: next }),
      });
      const saved = new Map((payload.cards || []).map((card) => [card.id, card]));
      setData((current) => ({ ...current, cards: current.cards.map((card) => saved.get(card.id) || card) }));
      onNotice("卡片顺序已保存");
    } catch (requestError) {
      setData((current) => ({ ...current, cards: applyCollectionCardOrder(current.cards, session.collectionId, original) }));
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
  const transferImported = async (result) => {
    const first = result.collections?.[0]?.id;
    await refresh();
    // The dialog's own receipt reports the result; a toast here would sit on its footer at phone width.
    if (first) { setActiveSection("gallery"); setSelectedCollection(first); setSelectedCard(null); }
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
  useEffect(() => {
    const room = galleryRoomRef.current;
    if (!room) return undefined;
    let frame = 0;
    const measure = () => {
      frame = 0;
      const nav = libraryRef.current;
      const search = searchBoxRef.current;
      // At the top there is nothing to go back to, whatever the geometry says.
      if (!nav || !search || room.scrollTop <= 0) return setBackToTop(false);
      setBackToTop(search.getBoundingClientRect().bottom < nav.getBoundingClientRect().bottom);
    };
    const schedule = () => { if (!frame) frame = window.requestAnimationFrame(measure); };
    measure();
    room.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    return () => {
      if (frame) window.cancelAnimationFrame(frame);
      room.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", schedule);
    };
  }, [activeSection, selectedCollection]);

  const scrollGalleryToTop = () => {
    galleryRoomRef.current?.scrollTo({ top: 0, behavior: "smooth" });
  };

  return <section className="gallery-page gallery-room" ref={galleryRoomRef}>
    <div className="gallery-room-grain" aria-hidden="true" />
    <div className="gallery-room-wash" aria-hidden="true" />
    <header className="gallery-room-header">
      <div className="gallery-room-wordmark"><span><i /><i /><i /></span><strong>XIRIA GALLERY</strong><em>/</em><small>LOCAL</small></div>
      <div className="gallery-room-status"><i />SQLITE CURATED ARCHIVE</div>
      <button type="button" onClick={() => { setActiveSection(activeSection === "prompts" ? "gallery" : "prompts"); setSelectedCard(null); }}><span>{activeSection === "prompts" ? "RETURN TO THE ROOM" : "OPEN PROMPT LIBRARY"}</span>{activeSection === "prompts" ? <ArrowLeft size={15} /> : <BookOpen size={15} />}</button>
    </header>
    <main className="gallery-main gallery-room-main">
      {activeSection === "prompts" ? <div className="gallery-room-prompts"><PromptLibrary entries={promptEntries} loading={promptLoading} error={promptError} onRefresh={() => void refreshPrompts()} onAdd={() => setPromptDialog({})} onEdit={(entry) => setPromptDialog({ entry })} onDelete={(entry) => void deletePromptEntry(entry)} onApply={(entry, target) => void applyPromptEntry(entry, target)} onNotice={onNotice} /></div> : <>
        <section className="gallery-room-archive">
          <nav className="gallery-room-library" aria-label="画廊收藏夹" ref={libraryRef}><button type="button" className={selectedCollection === null ? "active" : ""} onClick={() => { setSelectedCollection(null); setSelectedCard(null); }}><i />全部作品 <b>{data.cards.length}</b></button>{data.collections.map((collection) => <button type="button" className={selectedCollection === collection.id ? "active" : ""} key={collection.id} onClick={() => { setSelectedCollection(collection.id); setSelectedCard(null); }}><i />{collection.id} <b>{collection.card_count}</b></button>)}{backToTop && <button
            type="button"
            className="gallery-room-top"
            title="回到顶部"
            aria-label="回到顶部"
            onClick={scrollGalleryToTop}
          ><ArrowUp size={20} strokeWidth={2.4} /></button>}</nav>
          <div className="gallery-room-navigator"><button type="button" aria-label="上一个收藏夹" disabled={collectionScopes.length < 2} onClick={() => navigateCollection(-1)}><ArrowLeft size={17} /></button><div><span>{selectedCollection === null ? "ALL COLLECTIONS" : "CURRENT COLLECTION"}</span><strong>{selectedCollection === null ? "全部作品" : selectedCollection}</strong><small>{selectedCollectionRecord?.description || (selectedCollection === null ? "跨收藏夹浏览全部本地精选卡片" : "这个收藏夹还没有简介")}</small></div><button type="button" aria-label="下一个收藏夹" disabled={collectionScopes.length < 2} onClick={() => navigateCollection(1)}><ArrowRight size={17} /></button></div>
          <div className="gallery-room-tools">
            <label ref={searchBoxRef}><Search size={15} /><input type="search" value={gallerySearch} onChange={(event) => setGallerySearch(event.target.value)} placeholder="搜索标题、Prompt、模型或 LoRA" />{gallerySearch && <button type="button" aria-label="清空搜索" onClick={() => setGallerySearch("")}><X size={13} /></button>}</label>
            <div className="gallery-room-filters">{[["all", "全部"], ["single", "单图"], ["multi", "多图"], ["text", "无图卡片"]].map(([id, label]) => <button type="button" className={galleryFilter === id ? "active" : ""} key={id} onClick={() => setGalleryFilter(id)}>{label}</button>)}</div>
            <div className="gallery-room-actions"><button type="button" onClick={() => setCollectionDialog({ mode: "create" })}><FolderPlus size={14} />新建收藏夹</button>{selectedCollectionRecord && <AddCardTile collectionId={selectedCollectionRecord.id} onAdd={(id, initialImages) => setEditor({ mode: "create", collectionId: id, initialImages })} onDropError={(message) => onNotice(message, true)} />}{selectedCollectionRecord && <button type="button" onClick={() => setCollectionDialog({ mode: "edit", collection: selectedCollectionRecord })}><Pencil size={14} />管理</button>}<button type="button" className="gallery-transfer-entry" onClick={() => setTransferOpen(true)}><FolderArchive size={14} />导入 / 导出</button><button type="button" onClick={() => void refresh()} disabled={loading}><RefreshCw className={loading ? "spin" : ""} size={14} />刷新</button></div>
          </div>
          <div className="gallery-room-meta"><span>{filteredCards.length} STUDIES</span><i /><span>{visibleImageCount} IMAGES</span><b>按住卡片可拖拽排序 · 右键管理</b></div>
          <div className="gallery-scroll gallery-room-scroll">
            {error && <div className="gallery-page-error"><X size={22} /><strong>无法读取画廊</strong><p>{error}</p></div>}
            {!error && loading && <div className="gallery-loading"><RefreshCw className="spin" size={25} /><span>正在读取精选集</span></div>}
            {!error && !loading && data.collections.length === 0 && <div className="gallery-first-empty"><span><Sparkles size={33} /></span><strong>创建第一个精选收藏夹</strong><p>自定义收藏夹 ID 与简介，然后添加生成结果或手动卡片。</p><button type="button" onClick={() => setCollectionDialog({ mode: "create" })}><FolderPlus size={15} />创建收藏夹</button></div>}
            {!error && !loading && data.collections.length > 0 && filteredCards.length === 0 && <div className="gallery-first-empty"><span><FolderOpen size={33} /></span><strong>{gallerySearch || galleryFilter !== "all" ? "没有匹配的作品" : "这个收藏夹还是空的"}</strong><p>{gallerySearch || galleryFilter !== "all" ? "尝试清空搜索或切换筛选条件。" : "从生图结果加入画廊，或手动添加一张精选卡片。"}</p>{selectedCollectionRecord && <button type="button" onClick={() => setEditor({ mode: "create", collectionId: selectedCollectionRecord.id, initialImages: [] })}><ImagePlus size={15} />添加作品</button>}</div>}
            {!error && !loading && filteredCards.length > 0 && <GalleryCardColumns cards={filteredCards} dragState={dragState} reorderBusy={Boolean(reorderBusy)} onOpen={setSelectedCard} onMenu={openCardMenu} onReorderStart={startCardReorder} slotShifts={dragState?.shifts} />}
          </div>
        </section>
      </>}
      <footer className="gallery-room-footer"><span>XIRIA CANVAS AI / LOCAL GALLERY</span><span>SQLITE INDEX · INDEPENDENT COPIES</span><span>2026 / PERSONAL ARCHIVE</span></footer>
    </main>
    {selectedCard && createPortal(<GalleryDetail card={selectedCard} cards={viewerCards} onBack={() => setSelectedCard(null)} onNavigate={setSelectedCard} onEdit={(card) => { setSelectedCard(null); setEditor({ mode: "edit", card }); }} onDelete={deleteCard} onApply={onApplySettings} onNotice={onNotice} />, document.body)}
    {dragState && createPortal(<div ref={dragGhostRef} className="gallery-drag-ghost"><Images size={14} /><span><strong>正在排序</strong><small>{displayTitle(data.cards.find((card) => card.id === dragState.cardId), 42)}</small></span></div>, document.body)}
    {cardMenu && <GalleryCardMenu menu={cardMenu} onClose={() => setCardMenu(null)} onEdit={editFromCardMenu} onDelete={deleteFromCardMenu} />}
    {transferOpen && createPortal(<GalleryTransferDialog collections={data.collections} cards={data.cards} currentCollection={selectedCollection} onClose={() => setTransferOpen(false)} onImported={transferImported} />, document.body)}
    {collectionDialog && <CollectionDialog collection={collectionDialog.collection} onClose={() => setCollectionDialog(null)} onSaved={saveCollection} onDeleted={collectionDeleted} />}
    {/* Portalled like the detail viewer: the page is a stacking context under the sticky header,
        which otherwise covered the editor's top edge. */}
    {editor && createPortal(<GalleryCardEditor card={editor.card} collectionId={editor.collectionId} collections={data.collections} initialSettings={currentSettings} initialImages={editor.initialImages} canvasMaxEdge={canvasMaxEdge} upscalers={upscalerCatalog} upscalerRuntimeReady={upscalerRuntimeReady} upscalersRefreshing={upscalersRefreshing} onRefreshUpscalers={onRefreshUpscalers} upscalerDownloadJob={upscalerDownloadJob} upscalerDownloadBusy={upscalerDownloadBusy} upscalerDownloadError={upscalerDownloadError} onDownloadUpscaler={onDownloadUpscaler} onRetryUpscaler={onRetryUpscaler} onClose={() => setEditor(null)} onSaved={saveCard} />, document.body)}
    {promptDialog && <PromptLibraryDialog entry={promptDialog.entry} onClose={() => setPromptDialog(null)} onSaved={() => void savePromptEntry()} />}
  </section>;
}
