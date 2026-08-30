import { Suspense, lazy, useEffect, useRef, useState } from "react";
import { fitViewerZoom, inverseViewerHandleScale, intrinsicDimensions, viewerLayerBounds, viewerResizeGestureMove, viewerResizeDisableDecision, viewerZoomAtPoint, VIEWER_MAX_ZOOM, VIEWER_MIN_ZOOM } from "./viewer-geometry.js";
import { createViewerRafScheduler, ViewerAsyncSession, viewerOpenPlan } from "./viewer-async-session.js";
import { VIEWER_TOOLBAR_POPOVER_LAYOUT, VIEWER_TOOLBAR_POPOVER_TEMPLATES, viewerEscapeAction, viewerToolbarPopoverTransition } from "./viewer-toolbar.js";
import { pluginDiagnosticMessage, pluginRegistrySummary, pluginRemoveConfirmation, pluginStatePresentation, pluginToggleAvailable } from "./plugin-presentation.js";
import {
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  Blocks,
  Check,
  ChevronLeft,
  ChevronRight,
  ChevronDown,
  CircleHelp,
  Copy,
  Cpu,
  Download,
  ExternalLink,
  FileText,
  FolderOpen,
  ImageIcon,
  ImagePlus,
  LayoutTemplate,
  Layers3,
  Maximize2,
  Minimize2,
  Move,
  PanelLeft,
  Pause,
  Palette,
  Pencil,
  Play,
  Plus,
  RefreshCw,
  Search,
  Save,
  Send,
  Share2,
  Settings2,
  ShieldAlert,
  SlidersHorizontal,
  Sparkles,
  Square,
  Terminal,
  Trash2,
  Upload,
  WandSparkles,
  Wrench,
  ZoomIn,
  ZoomOut,
  X,
  Zap,
} from "lucide-react";
import {
  POSTPROCESS_STAGE_IDS,
  normalizePostprocessOrder,
  postprocessTargetSize,
} from "./postprocessing";
import { APP_VERSION } from "./app-version";
import { BrandLogo, LoadingLogo } from "./BrandLogo";
import { formatFileSize } from "./format-size";
import { DEFAULT_THEME, applyThemeToDocument, deriveLogoTheme, loadThemeState, normalizeHex } from "./theme";
import { AddToGalleryDialog } from "./Gallery";
import { useDialogLifecycle } from "./gallery-core";
import { generationHiresSeedSettings, hiresEffectiveSteps as hiresEffectiveStepCount, hiresSeedPayload, normalizeHiresSeed, normalizeUint64Seed, secureRandomUint64Seed } from "./hires-settings";
import LoraDetailsDialog from "./LoraDetailsDialog";
import { appLoraDragLocked } from "./lora-drag-handle";
import { formatWeight } from "./lora-weight";
import {
  DEFAULT_WORKSPACE_LAYOUT,
  isDefaultWorkspaceLayout,
  readWorkspaceLayout,
  resizeLeftPanel,
  steppedLeftPanel,
  toggleLeftPanel,
  workspaceLayoutClassName,
  workspaceLayoutStyle,
  writeWorkspaceLayout,
} from "./workspace-layout";
import {
  IMAGE_TO_IMAGE_DEFAULTS,
  imageToImageRequestBody,
  nextImageToImageSeed,
  normalizeImageToImageSettings,
  normalizeSourceSeed,
  outputSize,
} from "./image-to-image";
import SizeGrid from "./SizeGrid";
import WorkspaceSelect from "./WorkspaceSelect";
import { LORA_SYNC_CHANNEL, sameMountedLoras } from "./lora-state";
import {
  applyMountedLoraSync,
  canStartMountedLoraScan,
  emptyMountedLoraMap,
  frozenMountedLorasForScope,
  galleryMountedLorasForTarget,
  engineScopeKey,
  mountedLorasForScope,
  nextMountedLoraRevision,
  normalizeMountedLoraMap,
  reconcileMountedLoraScan,
  sameMountedLoraMap,
  shouldApplyMountedLoraScan,
  transitionMountedLoraScope,
  updateMountedLorasForScope,
  withMountedLorasForScope,
} from "./lora-model-scope";
import {
  composeGroupPrompt,
  disableUnmountedGroups,
  emptyLoraGroupMap,
  enabledLoraGroups,
  loraGroupsForScope,
  mountedEntryForGroups,
  syncMountedIntoGroups,
  normalizeLoraGroupMap,
  updateLoraGroupsForScope,
} from "./lora-groups";
import LoraMountPanel from "./LoraMountPanel.jsx";
import LoraGroupPanel from "./LoraGroupPanel.jsx";
import {
  createPromptPreset,
  deletePromptPreset,
  insertPromptPreset,
  normalizePromptPresetContainer,
  seededPromptPresetContainer,
  sortPromptPresetRecords,
  updatePromptPreset,
  validatePromptPresetDraft,
} from "./prompt-presets";
import { SAMPLER_NAMES as samplerNames, SCHEDULER_NAMES as schedulerNames } from "./sampling-options";
import {
  ADETAILER_UNIT_LIMIT,
  DISTILLED_GUIDANCE_ENGINES,
  activeADetailerUnits,
  adetailerPageIndex,
  adetailerPayload,
  adetailerStageIssue,
  adetailerStepUnitId,
  adetailerSummary,
  adetailerUnitLabel,
  adetailerUnitSteps,
  normalizeADetailerStage,
} from "./adetailer-units";
import { ASSISTANT_APPLY_PROMPT, ASSISTANT_CHANNEL_NAME, writePromptSnapshot } from "./ai-assistant-protocol";

// The gallery and the toolbox are whole pages behind their own nav tabs, so they load on first
// visit rather than riding along with the generate page every session. The toolbox splits again
// per tool, so opening it does not pull in every tool's code.
const GalleryPage = lazy(() => import("./GalleryPage"));
const ImageToImagePage = lazy(() => import("./ImageToImagePage"));
const ToolboxPage = lazy(() => import("./ToolboxPage"));
const SharedModelDirectories = lazy(() => import("./SharedModelDirectories"));
// The assistant is opened on demand and carries its own provider table and chat surface, so it stays
// out of the workspace chunk until the user actually asks for it.
const AiAssistantOverlay = lazy(() => import("./AiAssistantOverlay"));

const TRANSPARENT_BACKGROUND_TAG = "({Transparent background})";
const TRANSPARENT_BACKGROUND_PATTERN = /\(\{\s*transparent\s+background\s*\}\)/i;
const TRANSPARENT_BACKGROUND_PATTERN_ALL = /\(\{\s*transparent\s+background\s*\}\)/gi;

const models = [
  { name: "SD", detail: "Stable Diffusion", ready: true },
  { name: "iL", detail: "Illustrious", ready: true },
  { name: "Anima", detail: "原生 Flow Matching", ready: true },
  { name: "Flux", detail: "FLUX.1 蒸馏引导", ready: true },
  { name: "Flux2", detail: "FLUX.2 大模型引导", ready: true },
  { name: "Krea2", detail: "Krea 2 单流 DiT", ready: true },
];

const ANIMA_SAMPLERS = samplerNames;
const ANIMA_SCHEDULERS = schedulerNames;
// FLUX.1 is driven by the same ComfyUI KSampler node, so it offers the same names. The schedules
// behind them differ per engine; the vocabulary does not.
const FLUX_SAMPLERS = samplerNames;
const FLUX_SCHEDULERS = schedulerNames;
// FLUX.2 reuses FLUX.1's model sampling, so it reuses the vocabulary too; only where the shift
// comes from differs.
const FLUX2_SAMPLERS = samplerNames;
const FLUX2_SCHEDULERS = schedulerNames;
// Krea 2 runs the same ModelSamplingFlux table as both Flux engines, at the static shift its own
// model config declares, so it offers the same vocabulary as well.
const KREA2_SAMPLERS = samplerNames;
const KREA2_SCHEDULERS = schedulerNames;
// Engines that mount separate component files instead of one checkpoint.
const SPLIT_MODEL_ENGINES = ["Anima", "Flux", "Flux2", "Krea2"];
const READY_ENGINES = ["SD", "iL", "Anima", "Flux", "Flux2", "Krea2"];

const guidanceOptions = [
  { id: "none", label: "无（None）", detail: "标准 CFG 采样" },
  { id: "pag", label: "PAG（扰动注意力引导）", detail: "强度可调 · 默认仅作用于 Mid 层" },
  { id: "cfg_zero_star", label: "CFG-Zero*（零星 CFG）", detail: "优化缩放 · 前 4% 步零初始化", flowMatching: true },
];
const PAG_DEFAULTS = { scale: 0.3, appliedLayers: "mid" };

function pagAvailableForEngine(health, engine) {
  const pagHealth = health?.guidance?.pag;
  return pagHealth?.available === true
    && (pagHealth.engines || []).includes(engine)
    && health?.engines?.[engine]?.features?.pag === true;
}

const MAX_SEED = 0xFFFFFFFFFFFFFFFFn;
const DEFAULT_PERFORMANCE = {
  memory_mode: "auto",
  attention_backend: "auto",
  compute_dtype: "fp16",
  vae_mode: "auto",
  cuda_math: "balanced",
  keep_model_cached: true,
  allow_shared_memory: true,
  calculate_model_hash: false,
  staged_vae_decode: false,
  compile_transformer: false,
  vram_limit_gb: 0,
};
const makeGridSlots = (count, columns) => {
  const rows = Math.ceil(count / columns);
  return Array.from({ length: count }, (_, index) => ({
    x: (index % columns) / columns,
    y: Math.floor(index / columns) / rows,
    w: 1 / columns,
    h: 1 / rows,
  }));
};
const makeStripSlots = (count, vertical = false) => Array.from({ length: count }, (_, index) => vertical
  ? { x: 0, y: index / count, w: 1, h: 1 / count }
  : { x: index / count, y: 0, w: 1 / count, h: 1 });
const makeFeatureSlots = (count) => {
  const remainder = Math.max(1, count - 1);
  return [
    { x: 0, y: 0, w: .6, h: 1 },
    ...Array.from({ length: remainder }, (_, index) => ({ x: .6, y: index / remainder, w: .4, h: 1 / remainder })),
  ];
};
const collageTemplates = Array.from({ length: 8 }, (_, index) => index + 2).flatMap((count) => {
  const gridColumns = Math.ceil(Math.sqrt(count));
  return [
    { id: `${count}-compare`, count, label: "横向对比", slots: makeStripSlots(count) },
    { id: `${count}-grid`, count, label: `${gridColumns} 列网格`, slots: makeGridSlots(count, gridColumns) },
    { id: `${count}-feature`, count, label: "主图 + 组图", slots: makeFeatureSlots(count) },
  ];
});
const edgeLineStyles = [
  { id: "solid", label: "实线" },
  { id: "dashed", label: "短划线" },
  { id: "dotted", label: "点线" },
  { id: "double", label: "双线" },
  { id: "glow", label: "辉光" },
];

function assetRatio(asset) {
  const width = Number(asset?.width);
  const height = Number(asset?.height);
  return width > 0 && height > 0 ? width / height : 1;
}

function adaptiveCollageLayout(template, entries) {
  const ratios = Array.from({ length: template.count }, (_, index) => assetRatio(entries[index]));
  if (template.id.endsWith("-compare")) {
    const total = ratios.reduce((sum, ratio) => sum + ratio, 0);
    let x = 0;
    return { aspect: total, slots: ratios.map((ratio) => { const slot = { x, y: 0, w: ratio / total, h: 1 }; x += slot.w; return slot; }) };
  }
  if (template.id.endsWith("-grid")) {
    const columns = Math.ceil(Math.sqrt(template.count));
    const rows = [];
    for (let index = 0; index < template.count; index += columns) rows.push(ratios.slice(index, index + columns));
    const inverseWidths = rows.map((row) => 1 / row.reduce((sum, ratio) => sum + ratio, 0));
    const totalHeight = inverseWidths.reduce((sum, height) => sum + height, 0);
    let y = 0;
    const slots = [];
    rows.forEach((row, rowIndex) => {
      const rowWidth = row.reduce((sum, ratio) => sum + ratio, 0);
      const height = inverseWidths[rowIndex] / totalHeight;
      let x = 0;
      row.forEach((ratio) => { const width = ratio / rowWidth; slots.push({ x, y, w: width, h: height }); x += width; });
      y += height;
    });
    return { aspect: 1 / totalHeight, slots };
  }
  const sideCount = Math.max(1, template.count - 1);
  const mainWidth = ratios[0];
  const sideWidth = Math.max(...ratios.slice(1).map((ratio) => ratio / sideCount), 1 / sideCount);
  const totalWidth = mainWidth + sideWidth;
  return {
    aspect: totalWidth,
    slots: [
      { x: 0, y: 0, w: mainWidth / totalWidth, h: 1 },
      ...ratios.slice(1).map((_, index) => ({ x: mainWidth / totalWidth, y: index / sideCount, w: sideWidth / totalWidth, h: 1 / sideCount })),
    ],
  };
}

function drawEdgeLine(context, x, y, width, height, options, scale = 1, hiddenSides = []) {
  if (!options.enabled) return;
  const lineWidth = Math.max(1, options.width * scale);
  const hidden = new Set(hiddenSides);
  context.save();
  context.strokeStyle = options.color;
  context.lineWidth = options.style === "double" ? lineWidth * 3 : lineWidth;
  if (options.style === "dashed") context.setLineDash([lineWidth * 4, lineWidth * 2.5]);
  if (options.style === "dotted") { context.setLineDash([lineWidth, lineWidth * 2.4]); context.lineCap = "round"; }
  if (options.style === "glow") { context.shadowColor = options.color; context.shadowBlur = lineWidth * 5; }
  const strokeSides = (inset = context.lineWidth / 2) => {
    const left = x + inset;
    const top = y + inset;
    const right = x + Math.max(inset, width - inset);
    const bottom = y + Math.max(inset, height - inset);
    context.beginPath();
    if (!hidden.has("top")) { context.moveTo(left, top); context.lineTo(right, top); }
    if (!hidden.has("right")) { context.moveTo(right, top); context.lineTo(right, bottom); }
    if (!hidden.has("bottom")) { context.moveTo(right, bottom); context.lineTo(left, bottom); }
    if (!hidden.has("left")) { context.moveTo(left, bottom); context.lineTo(left, top); }
    context.stroke();
  };
  if (hidden.size) strokeSides();
  else context.strokeRect(x + context.lineWidth / 2, y + context.lineWidth / 2, Math.max(0, width - context.lineWidth), Math.max(0, height - context.lineWidth));
  if (options.style === "double") {
    context.lineWidth = lineWidth;
    if (hidden.size) strokeSides(lineWidth * 2.5);
    else context.strokeRect(x + lineWidth * 2.5, y + lineWidth * 2.5, Math.max(0, width - lineWidth * 5), Math.max(0, height - lineWidth * 5));
  }
  context.restore();
}
const performancePresets = [
  { id: "auto", label: "自适应推荐", range: "全部显卡", detail: "按模型、画布与实时空闲显存自动安排", settings: { ...DEFAULT_PERFORMANCE } },
  { id: "speed", label: "高速直通", range: "12 GB 以上", detail: "优先让模型完整驻留 GPU，适合连续生成", settings: { ...DEFAULT_PERFORMANCE, memory_mode: "high_vram", vae_mode: "full" } },
  { id: "large-model", label: "大型模型调度", range: "8 GB 以上", detail: "大型模型采样后转入内存，再单独执行 VAE 解码", settings: { ...DEFAULT_PERFORMANCE, memory_mode: "sdxl_balanced", staged_vae_decode: true } },
  { id: "balanced", label: "显存平衡", range: "4 GB 以上", detail: "按组件迁移，采样后释放显存并切片解码", settings: { ...DEFAULT_PERFORMANCE, memory_mode: "normal_vram", vae_mode: "sliced", staged_vae_decode: true } },
  { id: "low-memory", label: "低占用保护", range: "不足 4 GB", detail: "逐层卸载、分块解码，并在任务结束后释放模型", settings: { ...DEFAULT_PERFORMANCE, memory_mode: "low_vram", attention_backend: "sliced", vae_mode: "tiled", keep_model_cached: false, staged_vae_decode: true } },
  { id: "ultra-low", label: "极限省存 · Illustrious", range: "4 GB 显存 + 16 GB 内存", detail: "串行计算正负条件并关闭附加链路，目标是让 1024 画布尽量跑通", settings: { ...DEFAULT_PERFORMANCE, memory_mode: "ultra_low_vram", attention_backend: "sliced", compute_dtype: "fp16", vae_mode: "tiled", cuda_math: "strict", keep_model_cached: false, allow_shared_memory: false, staged_vae_decode: true } },
  { id: "large-canvas", label: "高分辨率稳态", range: "大画布", detail: "预留更多显存并使用空间分块，减少峰值波动", settings: { ...DEFAULT_PERFORMANCE, vae_mode: "tiled", allow_shared_memory: false, staged_vae_decode: true } },
];
const performanceChoices = {
  memory_mode: [
    ["auto", "动态预算", "按模型、画布与空闲显存实时选择"],
    ["high_vram", "GPU 常驻", "空间足够时优先吞吐，超预算会安全降级"],
    ["sdxl_balanced", "仅大型模型调度", "大型模型使用组件迁移，普通模型保持动态选择"],
    ["normal_vram", "组件调度", "在 GPU 与内存间按组件迁移"],
    ["low_vram", "逐层保护", "最小化峰值显存，生成速度较慢"],
    ["ultra_low_vram", "极限省存", "串行 CFG 与逐层卸载，专用于 4 GB 运行大型模型"],
  ],
  attention_backend: [
    ["auto", "原生优先", "优先当前 PyTorch SDPA，兼容新架构"],
    ["sdpa", "PyTorch SDPA", "使用框架原生缩放点积注意力"],
    ["xformers", "xformers", "仅在已安装且运行验证通过时启用"],
    ["sage", "SageAttention", "Anima 自注意力改用 INT8 量化内核，更快；同种子出图会改变"],
    ["sliced", "切片注意力", "节省显存，牺牲部分速度"],
  ],
  compute_dtype: [
    ["fp16", "FP16 标准", "兼容性与速度均衡"],
    ["bf16", "BF16 宽动态", "支持的 GPU 上降低数值溢出风险"],
  ],
  vae_mode: [
    ["auto", "跟随显存策略", "常驻 / 切片 / 分块自动联动"],
    ["full", "完整解码", "速度优先，峰值显存最高"],
    ["sliced", "批次切片", "降低多图解码占用"],
    ["tiled", "空间分块", "高分辨率与低显存优先"],
  ],
};
const seedModes = [
  { id: "random", label: "全随机" },
  { id: "fixed", label: "固定" },
  { id: "increment", label: "递增" },
  { id: "decrement", label: "递减" },
];
// The stage is an ordered list of detail passes; `adetailer-units.js` owns what
// a unit is, so this page and the image-to-image page cannot drift apart.
const ADETAILER_DEFAULTS = normalizeADetailerStage({ enabled: false, expanded: false });
const HIRES_DEFAULTS = {
  enabled: false,
  expanded: false,
  model: "",
  seedMode: "inherit",
  seed: "",
  scale: 1,
  denoise: 0.35,
  steps: 20,
  cfg: 7,
  tileSize: 192,
  tileOverlap: 16,
  executionMode: "full_frame",
  sampler: null,
  scheduler: null,
  tileWidth: "auto",
  tileHeight: "auto",
  padding: 32,
  maskBlur: 8,
  seamMode: "none",
  uniformTiles: true,
  tiledDecode: true,
};
const RTX_DEFAULTS = {
  enabled: false,
  expanded: false,
  scale: 2,
  quality: "ultra",
};
const RTX_QUALITY_LEVELS = ["low", "medium", "high", "ultra"];

function normalizeSeed(value) {
  const digits = String(value).replace(/\D/g, "") || "0";
  try {
    return BigInt(digits) > MAX_SEED ? MAX_SEED.toString() : BigInt(digits).toString();
  } catch {
    return "0";
  }
}

function randomSeed() {
  return secureRandomUint64Seed();
}

function loadBrowserImage(source) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = async () => { try { if (image.decode) await image.decode(); } catch {} resolve(image); };
    image.onerror = () => reject(new Error("图片读取失败"));
    image.src = source;
  });
}

function dataUrlBlob(dataUrl) {
  const [header, encoded] = dataUrl.split(",", 2);
  const mime = header.match(/data:([^;]+)/)?.[1] || "image/png";
  const bytes = atob(encoded);
  const array = new Uint8Array(bytes.length);
  for (let index = 0; index < bytes.length; index += 1) array[index] = bytes.charCodeAt(index);
  return new Blob([array], { type: mime });
}

function readImageFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("图片读取失败"));
    reader.readAsDataURL(file);
  });
}

async function imageAssetFromSource(asset) {
  const image = await loadBrowserImage(asset.url);
  const { naturalWidth, naturalHeight } = intrinsicDimensions(image.naturalWidth, image.naturalHeight);
  return { ...asset, naturalWidth, naturalHeight };
}

function isGifAsset(asset) {
  const source = asset?.url || "";
  const mime = asset?.mimeType || asset?.mime_type || "";
  return mime === "image/gif" || source.startsWith("data:image/gif") || /\.gif(?:$|[?#])/i.test(asset?.name || "");
}

async function renderAnimatedCollage(layers, width, height, edgeLine) {
  const response = await fetch("/api/inference/collages/animated", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ layers, width, height, edge_line: edgeLine }),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || "GIF 拼图合成失败");
  }
  return readImageFile(await response.blob());
}

async function imageSourceDataUrl(source) {
  if (source.startsWith("data:image/")) return source;
  const response = await fetch(source);
  if (!response.ok) throw new Error("无法读取 GIF 拼图源图");
  return readImageFile(await response.blob());
}

function isEditableTarget(target) {
  return target instanceof Element && Boolean(target.closest("input, textarea, select, [contenteditable='true']"));
}

function sharedEdgeLineSegments(rectangles, tolerance = 1) {
  const sideNames = ["top", "right", "bottom", "left"];
  const visible = new Map(rectangles.map((rect) => [rect.id, {
    top: [[rect.left, rect.right]],
    right: [[rect.top, rect.bottom]],
    bottom: [[rect.left, rect.right]],
    left: [[rect.top, rect.bottom]],
  }]));
  const subtract = (id, side, start, end) => {
    const segments = visible.get(id)?.[side] || [];
    visible.get(id)[side] = segments.flatMap(([segmentStart, segmentEnd]) => {
      if (end <= segmentStart + tolerance || start >= segmentEnd - tolerance) return [[segmentStart, segmentEnd]];
      const next = [];
      if (start > segmentStart + tolerance) next.push([segmentStart, Math.min(start, segmentEnd)]);
      if (end < segmentEnd - tolerance) next.push([Math.max(end, segmentStart), segmentEnd]);
      return next;
    });
  };
  const compare = (first, second, firstSide, secondSide) => {
    if (Math.abs(first[firstSide] - second[secondSide]) > tolerance) return;
    const vertical = firstSide === "left" || firstSide === "right";
    const firstStart = vertical ? first.top : first.left;
    const firstEnd = vertical ? first.bottom : first.right;
    const secondStart = vertical ? second.top : second.left;
    const secondEnd = vertical ? second.bottom : second.right;
    const start = Math.max(firstStart, secondStart);
    const end = Math.min(firstEnd, secondEnd);
    if (end - start <= tolerance) return;
    if (firstEnd - firstStart >= secondEnd - secondStart) subtract(second.id, secondSide, start, end);
    else subtract(first.id, firstSide, start, end);
  };
  for (let firstIndex = 0; firstIndex < rectangles.length; firstIndex += 1) {
    for (let secondIndex = firstIndex + 1; secondIndex < rectangles.length; secondIndex += 1) {
      const first = rectangles[firstIndex];
      const second = rectangles[secondIndex];
      compare(first, second, "right", "left");
      compare(first, second, "left", "right");
      compare(first, second, "bottom", "top");
      compare(first, second, "top", "bottom");
    }
  }
  return Object.fromEntries(rectangles.map((rect) => [rect.id, Object.fromEntries(sideNames.map((side) => {
    const horizontal = side === "top" || side === "bottom";
    const origin = horizontal ? rect.left : rect.top;
    const length = Math.max(1, horizontal ? rect.right - rect.left : rect.bottom - rect.top);
    return [side, (visible.get(rect.id)?.[side] || []).map(([start, end]) => ({
      start: (start - origin) / length * 100,
      span: (end - start) / length * 100,
    }))];
  }))]));
}

// Canvas export only needs to know which complete edges are covered by a neighbor.
function sharedEdgeHiddenSides(rectangles, tolerance = 1) {
  const segments = sharedEdgeLineSegments(rectangles, tolerance);
  return Object.fromEntries(Object.entries(segments).map(([id, sides]) => [id, Object.entries(sides)
    .filter(([, visible]) => visible.length === 0)
    .map(([side]) => side)]));
}

function loadWorkspaceState(saved) {
  try {
    if (!saved || typeof saved !== "object") return null;
    const savedModel = READY_ENGINES.includes(saved.model) ? saved.model : "SD";
    const isAnima = savedModel === "Anima";
    const isSplitModel = SPLIT_MODEL_ENGINES.includes(savedModel);
    const numberInRange = (value, fallback, minimum, maximum) => {
      const number = Number(value);
      return Number.isFinite(number) ? Math.max(minimum, Math.min(maximum, number)) : fallback;
    };
    const savedADetailer = saved.adetailer && typeof saved.adetailer === "object" ? saved.adetailer : {};
    const savedHires = saved.hires && typeof saved.hires === "object" ? saved.hires : {};
    const savedHiresSeed = normalizeHiresSeed(savedHires.seedMode, savedHires.seed);
    const savedRtx = saved.rtx && typeof saved.rtx === "object" ? saved.rtx : {};
    const savedPag = saved.pag && typeof saved.pag === "object" ? saved.pag : {};
    const hasPromptPresets = Object.prototype.hasOwnProperty.call(saved, "promptPresets");
    const promptPresetLibrary = normalizePromptPresetContainer(saved.promptPresets, { missing: !hasPromptPresets });
    const savedHiresTileSize = Math.round(numberInRange(savedHires.tileSize, 192, 32, 2048));
    const savedHiresTileOverlap = Math.min(
      Math.round(numberInRange(savedHires.tileOverlap, 16, 0, 512)),
      Math.floor(savedHiresTileSize / 2),
    );
    const hiresExecutionMode = isAnima && !["full_frame", "usdu_tiled"].includes(savedHires.executionMode)
      ? "usdu_tiled"
      : savedHires.executionMode === "usdu_tiled" && isAnima ? "usdu_tiled" : "full_frame";
    const normalizeHiresTileDimension = (value) => value === "auto" || value === undefined || value === null
      ? "auto"
      : Math.round(numberInRange(value, 0, 1, 8192));
    const modelIdentity = {
      model: savedModel,
      checkpoint: typeof saved.checkpoint === "string" ? saved.checkpoint : "",
      diffusionModel: typeof saved.diffusionModel === "string" ? saved.diffusionModel : "",
      textEncoder: typeof saved.textEncoder === "string" ? saved.textEncoder : "",
      textEncoder2: typeof saved.textEncoder2 === "string" ? saved.textEncoder2 : "",
      vae: typeof saved.vae === "string" ? saved.vae : "",
    };
    const activeLoraScopeKey = engineScopeKey(savedModel);
    const mountedLoraMap = normalizeMountedLoraMap(saved.mountedLorasByEngine, {
      fieldMissing: !Object.prototype.hasOwnProperty.call(saved, "mountedLorasByEngine"),
      legacyLoras: saved.loras,
      activeEngine: activeLoraScopeKey,
      activeModelIdentity: isSplitModel
        ? { engine: savedModel, assets: [modelIdentity.diffusionModel, modelIdentity.textEncoder, ...(savedModel === "Flux" ? [modelIdentity.textEncoder2] : []), modelIdentity.vae] }
        : { engine: savedModel, assets: [modelIdentity.checkpoint] },
    });
    const loraGroupMap = normalizeLoraGroupMap(saved.loraGroupsByEngine, {
      fieldMissing: !Object.prototype.hasOwnProperty.call(saved, "loraGroupsByEngine"),
    });
    return {
      model: savedModel,
      checkpoint: modelIdentity.checkpoint,
      diffusionModel: modelIdentity.diffusionModel,
      textEncoder: modelIdentity.textEncoder,
      textEncoder2: modelIdentity.textEncoder2,
      vae: modelIdentity.vae,
      positive: typeof saved.positive === "string" ? saved.positive : "",
      negative: typeof saved.negative === "string" ? saved.negative : "",
      promptPresets: promptPresetLibrary.container,
      promptPresetLibraryError: promptPresetLibrary.error,
      promptPresetLibraryWarning: promptPresetLibrary.warning,
      promptPresetLibraryRaw: promptPresetLibrary.fatal ? promptPresetLibrary.raw : null,
      shouldPersistPromptPresets: promptPresetLibrary.shouldPersist,
      steps: Math.round(numberInRange(saved.steps, 28, 1, 60)),
      cfg: numberInRange(saved.cfg, 6.5, 1, 15),
      denoise: numberInRange(saved.denoise, 1, 0, 1),
      imagesPerBatch: Math.round(numberInRange(saved.imagesPerBatch, 1, 1, 10)),
      batchCount: Math.round(numberInRange(saved.batchCount, 1, 1, 20)),
      seed: normalizeSeed(saved.seed ?? "847291"),
      seedMode: seedModes.some((item) => item.id === saved.seedMode) ? saved.seedMode : "random",
      sampler: isSplitModel ? ANIMA_SAMPLERS.includes(saved.sampler) ? saved.sampler : "euler" : samplerNames.includes(saved.sampler) ? saved.sampler : "dpmpp_2m",
      scheduler: isSplitModel ? ANIMA_SCHEDULERS.includes(saved.scheduler) ? saved.scheduler : "simple" : schedulerNames.includes(saved.scheduler) ? saved.scheduler : "karras",
      guidance: guidanceOptions.some((item) => item.id === saved.guidance) ? saved.guidance : "none",
      pag: {
        scale: Math.round(numberInRange(savedPag.scale, PAG_DEFAULTS.scale, 0, 5) * 100) / 100,
        appliedLayers: ["mid", "all"].includes(savedPag.appliedLayers) ? savedPag.appliedLayers : PAG_DEFAULTS.appliedLayers,
      },
      size: {
        width: Math.round(numberInRange(saved.size?.width, 1024, 0, 2048) / 64) * 64,
        height: Math.round(numberInRange(saved.size?.height, 1024, 0, 2048) / 64) * 64,
      },
      processPreview: isSplitModel ? false : saved.processPreview !== false,
      backgroundRemovalModel: typeof saved.backgroundRemovalModel === "string" ? saved.backgroundRemovalModel : "",
      samplingExpanded: saved.samplingExpanded !== false,
      hires: {
        enabled: savedHires.enabled === true,
        expanded: savedHires.expanded === true,
        model: typeof savedHires.model === "string" ? savedHires.model : "",
        ...savedHiresSeed,
        scale: Math.round(numberInRange(savedHires.scale, 1, 1, 4) * 10) / 10,
        denoise: numberInRange(savedHires.denoise, 0.35, 0.05, 1),
        steps: Math.round(numberInRange(savedHires.steps, 20, 1, 100)),
        cfg: numberInRange(savedHires.cfg, 7, 0, 30),
        tileSize: savedHiresTileSize,
        tileOverlap: savedHiresTileOverlap,
        executionMode: hiresExecutionMode,
        sampler: (isSplitModel ? ANIMA_SAMPLERS : samplerNames).includes(savedHires.sampler) ? savedHires.sampler : null,
        scheduler: (isSplitModel ? ANIMA_SCHEDULERS : schedulerNames).includes(savedHires.scheduler) ? savedHires.scheduler : null,
        tileWidth: normalizeHiresTileDimension(savedHires.tileWidth),
        tileHeight: normalizeHiresTileDimension(savedHires.tileHeight),
        padding: Math.round(numberInRange(savedHires.padding, 32, 0, 256)),
        maskBlur: Math.round(numberInRange(savedHires.maskBlur, 8, 0, 64)),
        seamMode: "none",
        uniformTiles: savedHires.uniformTiles !== false,
        tiledDecode: savedHires.tiledDecode !== false,
      },
      adetailer: normalizeADetailerStage(savedADetailer),
      rtx: {
        enabled: savedRtx.enabled === true,
        expanded: savedRtx.expanded === true,
        scale: Math.round(numberInRange(savedRtx.scale, 2, 1, 4) * 100) / 100,
        quality: RTX_QUALITY_LEVELS.includes(savedRtx.quality) ? savedRtx.quality : "ultra",
      },
      postprocessOrder: normalizePostprocessOrder(saved.postprocessOrder ?? saved.postprocess_order, savedHires.order),
      loraCategory: typeof saved.loraCategory === "string" && saved.loraCategory.length <= 300 ? saved.loraCategory : "character",
      loraSearch: typeof saved.loraSearch === "string" ? saved.loraSearch : "",
      // `loras` stays as a compatibility mirror only. The keyed map owns all
      // model-specific lists and must never fall back to a different identity.
       mountedLorasByEngine: mountedLoraMap.container,
      mountedLorasWarning: mountedLoraMap.warning,
      mountedLorasRaw: mountedLoraMap.raw || null,
      shouldPersistMountedLoras: !mountedLoraMap.fatal,
      activeLoraScopeKey,
      loras: mountedLoraMap.activeLoras,
      // Groups are a convenience over a mounted library that works without
      // them, so a damaged group file is reset rather than treated as fatal.
      loraGroupsByEngine: loraGroupMap.container,
      loraGroupsWarning: loraGroupMap.warning,
      // Image-to-image keeps its own prompt and sampling block rather than borrowing the
      // text-to-image ones: the two pages are used for different things in the same session, and
      // sharing them would mean editing one page's prompt from the other. The source picture is
      // deliberately absent — see `src/image-to-image.js`.
      imageToImage: normalizeImageToImageSettings(saved.imageToImage, {
        samplers: isSplitModel ? ANIMA_SAMPLERS : samplerNames,
        schedulers: isSplitModel ? ANIMA_SCHEDULERS : schedulerNames,
        defaultSampler: isSplitModel ? "euler" : "dpmpp_2m",
        defaultScheduler: isSplitModel ? "simple" : "karras",
      }),
    };
  } catch {
    return null;
  }
}

function gallerySettingsWithoutPromptPresets(settings) {
  if (!settings || typeof settings !== "object") return settings;
  // Libraries, not settings: the keyed mount map and the saved group
  // definitions describe the whole workspace, so copying them into every image
  // record would both bloat the gallery and let a restore rewrite combinations
  // the user never asked to change.
  const { promptPresets: _promptPresets, mountedLorasByEngine: _mountedLorasByEngine, loraGroupsByEngine: _loraGroupsByEngine, ...gallerySettings } = settings;
  return gallerySettings;
}

function reconcileModels(current, discovered) {
  const discoveredValues = new Set(discovered.map((item) => item.value));
  const currentValues = new Set(current.map((item) => item.value));
  const retained = current.filter((item) => discoveredValues.has(item.value));
  const additions = discovered.filter((item) => !currentValues.has(item.value));
  return { models: [...retained, ...additions], changed: retained.length !== current.length || additions.length > 0 };
}

function reconcileLoraLibrary(current, discovered) {
  const discoveredById = new Map(discovered.map((category) => [category.id, category]));
  const currentIds = new Set(current.map((category) => category.id));
  let changed = current.some((category) => !discoveredById.has(category.id));
  const categories = current.filter((category) => discoveredById.has(category.id)).map((category) => {
    const scannedCategory = discoveredById.get(category.id);
    const result = reconcileModels(category.models, scannedCategory.models);
    const categoryChanged = category.label !== scannedCategory.label || category.directory !== scannedCategory.directory;
    if (!result.changed && !categoryChanged) return category;
    changed = true;
    return { ...scannedCategory, models: result.models };
  });
  for (const category of discovered) {
    if (!currentIds.has(category.id)) {
      categories.push(category);
      changed = true;
    }
  }
  return { categories, changed };
}

function BoundedNumberInput({ value, min, max, step = 1, integer = false, fixed = null, normalize, onCommit, disabled = false, className = "", ariaLabel }) {
  const inputRef = useRef(null);
  const displayValue = (next) => fixed === null ? String(next) : Number(next).toFixed(fixed);
  const [typedValue, setTypedValue] = useState(displayValue(value));

  useEffect(() => {
    if (document.activeElement !== inputRef.current) setTypedValue(displayValue(value));
  }, [value]);

  const reset = () => setTypedValue(displayValue(value));
  const commit = () => {
    const parsed = Number(typedValue);
    if (!typedValue.trim() || !Number.isFinite(parsed)) {
      reset();
      return;
    }
    const clamped = Math.max(min, Math.min(max, integer ? Math.round(parsed) : parsed));
    const next = Math.max(min, Math.min(max, normalize ? normalize(clamped) : clamped));
    setTypedValue(displayValue(next));
    onCommit(next);
  };

  return <input
    ref={inputRef}
    className={`bounded-number${className ? ` ${className}` : ""}`}
    type="text"
    inputMode={integer ? "numeric" : "decimal"}
    value={typedValue}
    aria-label={ariaLabel}
    disabled={disabled}
    onChange={(event) => setTypedValue(event.target.value)}
    onKeyDown={(event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        commit();
        event.currentTarget.blur();
      } else if (event.key === "Escape") {
        event.preventDefault();
        reset();
        event.currentTarget.blur();
      }
    }}
    onBlur={reset}
  />;
}

function Slider({ label, value, min, max, step = 1, inputStep = step, integer = false, fixed = null, onChange, suffix = "", disabled = false }) {
  const progress = ((value - min) / (max - min)) * 100;
  const rangeValue = Math.max(min, Math.min(max, Math.round((value - min) / step) * step + min));

  const updateRange = (next) => onChange(integer ? Math.round(Number(next)) : Number(next));

  return (
    <label className="slider-field">
      <span>{label}</span>
      <span className="slider-value"><BoundedNumberInput value={value} min={min} max={max} step={inputStep} integer={integer} fixed={fixed} ariaLabel={`${label}数值`} disabled={disabled} onCommit={onChange} />{suffix}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={rangeValue}
        disabled={disabled}
        style={{ "--progress": `${progress}%` }}
        onChange={(event) => updateRange(event.target.value)}
      />
    </label>
  );
}

function CountField({ label, detail, value, min = 1, max, disabled = false, onChange }) {
  const commit = (next) => onChange(Math.max(min, Math.min(max, Math.round(Number(next) || min))));
  return (
    <label className="count-field">
      <span><strong>{label}</strong><small>{detail}</small></span>
      <span className="count-stepper">
        <button type="button" disabled={disabled || value <= min} onClick={() => commit(value - 1)}>-</button>
        <BoundedNumberInput value={value} min={min} max={max} integer ariaLabel={`${label}数值`} disabled={disabled} onCommit={commit} />
        <button type="button" disabled={disabled || value >= max} onClick={() => commit(value + 1)}>+</button>
      </span>
    </label>
  );
}

const promptPresetPositionLabels = { start: "开头", middle: "中间", end: "结尾" };

function PresetBox({ title, type, records, disabled, libraryError, libraryWarning, onSelect, onCreate, onEdit, onDelete }) {
  const [open, setOpen] = useState(false);
  const [menuId, setMenuId] = useState("");
  const listId = `prompt-preset-${type}-list`;
  const menuRef = useRef(null);
  useEffect(() => {
    if (!menuId) return undefined;
    const closeMenu = (event) => {
      if (!menuRef.current?.contains(event.target)) setMenuId("");
    };
    const closeOnEscape = (event) => {
      if (event.key === "Escape") {
        event.stopPropagation();
        setMenuId("");
      }
    };
    window.addEventListener("pointerdown", closeMenu);
    window.addEventListener("keydown", closeOnEscape);
    return () => {
      window.removeEventListener("pointerdown", closeMenu);
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [menuId]);
  useEffect(() => {
    if (libraryError) setOpen(true);
  }, [libraryError]);
  return (
    <div className={`preset-box ${open ? "open" : ""}`}>
      <div className="preset-head">
        <button type="button" aria-expanded={open} aria-controls={listId} onClick={() => { setOpen((current) => !current); setMenuId(""); }}>
          <span>{title}</span>
          <span className="preset-meta">{records.length} 个预设 <ChevronDown size={15} /></span>
        </button>
        <button type="button" className="prompt-preset-add" data-prompt-preset-focus-fallback={type} aria-label={`新增${type === "positive" ? "正向" : "负向"} Prompt 预设`} disabled={disabled || Boolean(libraryError)} onClick={onCreate}><Plus size={14} />新增</button>
      </div>
      {open && (
        <div className="preset-list" id={listId}>
          {libraryError && <p className="prompt-preset-error" role="alert">{libraryError}</p>}
          {!libraryError && libraryWarning && <p className="prompt-preset-warning" role="status">{libraryWarning}</p>}
          {!libraryError && records.length === 0 && <div className="prompt-preset-empty"><strong>暂无{type === "positive" ? "正向" : "负向"}预设</strong><p>可以新增一个常用 Prompt，并设置默认插入位置。</p><button type="button" disabled={disabled} onClick={onCreate}><Plus size={14} />新增预设</button></div>}
          {!libraryError && records.length > 0 && <div className="prompt-preset-grid">
            {records.map((record) => {
              const menuOpen = menuId === record.id;
              const menuControlId = `prompt-preset-menu-${type}-${record.id.replace(/[^a-zA-Z0-9_-]/g, "-")}`;
              return <article className="prompt-preset-card" key={record.id}>
                <button type="button" className="prompt-preset-main" aria-label={`插入${type === "positive" ? "正向" : "负向"}预设“${record.name}”到${promptPresetPositionLabels[record.position]}`} disabled={disabled} onClick={() => onSelect(record)}>
                  <span><strong>{record.name}</strong><b>{promptPresetPositionLabels[record.position]}</b></span>
                  <small>{record.content.replace(/\s+/g, " ")}</small>
                </button>
                <div className="prompt-preset-more-wrap" ref={menuOpen ? menuRef : undefined}>
                  <button type="button" className="prompt-preset-more" aria-label={`更多操作：${record.name}`} aria-haspopup="menu" aria-expanded={menuOpen} aria-controls={menuControlId} disabled={disabled} onClick={() => setMenuId((current) => current === record.id ? "" : record.id)}><span aria-hidden="true">•••</span></button>
                  {menuOpen && <div className="prompt-preset-menu" id={menuControlId} role="menu" aria-label={`${record.name}预设操作`}>
                    <button type="button" role="menuitem" disabled={disabled} onClick={(event) => { event.currentTarget.closest("article")?.querySelector(".prompt-preset-more")?.focus(); setMenuId(""); onEdit(record); }}><Pencil size={13} />编辑</button>
                    <button type="button" role="menuitem" className="danger" disabled={disabled} onClick={(event) => { event.currentTarget.closest("article")?.querySelector(".prompt-preset-more")?.focus(); setMenuId(""); onDelete(record); }}><Trash2 size={13} />删除</button>
                  </div>}
                </div>
              </article>;
            })}
          </div>}
        </div>
      )}
    </div>
  );
}

function PromptPresetDialog({ dialog, records, running, onSave, onRequestClose }) {
  const original = dialog.record ? {
    name: dialog.record.name,
    content: dialog.record.content,
    type: dialog.record.type,
    position: dialog.record.position,
  } : { name: "", content: "", type: dialog.type, position: "end" };
  const [draft, setDraft] = useState(original);
  const [operationError, setOperationError] = useState("");
  const validation = validatePromptPresetDraft(draft, records, dialog.record?.id || null);
  const dirty = JSON.stringify(draft) !== JSON.stringify(original);
  const requestClose = () => {
    if (dirty && !window.confirm("草稿尚未保存，确定要放弃更改吗？")) return;
    onRequestClose();
  };
  const dialogRef = useDialogLifecycle(true, requestClose, "[data-prompt-preset-focus-fallback]");
  const titleId = `prompt-preset-dialog-title-${dialog.record ? "edit" : "new"}`;
  const update = (key, value) => setDraft((current) => ({ ...current, [key]: value }));
  return <div className="prompt-preset-backdrop" onMouseDown={(event) => event.target === event.currentTarget && requestClose()}>
    <form ref={dialogRef} className="prompt-preset-dialog" role="dialog" aria-modal="true" aria-labelledby={titleId} tabIndex="-1" onSubmit={(event) => { event.preventDefault(); if (!running && validation.valid) setOperationError(onSave(draft) || ""); }}>
      <header><div><span>PROMPT PRESET LIBRARY</span><h2 id={titleId}>{dialog.record ? "编辑 Prompt 预设" : "新增 Prompt 预设"}</h2><p>{dialog.record ? `保留固定 ID · 保存后版本 ${dialog.record.version + 1}` : "保存常用内容并指定点击卡片时的插入位置"}</p></div><button type="button" aria-label="关闭 Prompt 预设窗口" onClick={requestClose}><X size={18} /></button></header>
      <div className="prompt-preset-dialog-body">
        {running && <p className="prompt-preset-readonly" role="status">生成任务运行期间只能查看；编辑项与保存操作已锁定。</p>}
        {operationError && <p className="prompt-preset-error" role="alert">{operationError}</p>}
        <label className={validation.errors.name ? "invalid" : ""}><span>预设名称 <small>{draft.name.length} / 48</small></span><input autoFocus data-dialog-autofocus value={draft.name} maxLength={48} disabled={running} aria-invalid={Boolean(validation.errors.name)} onChange={(event) => update("name", event.target.value)} />{validation.errors.name && <em>{validation.errors.name}</em>}</label>
        <label className={`prompt-preset-content-field ${validation.errors.content ? "invalid" : ""}`}><span>Prompt 内容 <small>{draft.content.length} / 2000</small></span><textarea value={draft.content} maxLength={2000} disabled={running} aria-invalid={Boolean(validation.errors.content)} onChange={(event) => update("content", event.target.value)} placeholder="保留权重、括号与内部换行" />{validation.errors.content && <em>{validation.errors.content}</em>}</label>
        <fieldset><legend>Prompt 类型</legend><div className="prompt-preset-segmented two">{[["positive", "正向"], ["negative", "负向"]].map(([value, label]) => <button type="button" className={draft.type === value ? "active" : ""} aria-pressed={draft.type === value} disabled={running} key={value} onClick={() => update("type", value)}>{label}</button>)}</div><small>切换类型后，预设会移动到目标分区末尾。</small></fieldset>
        <fieldset><legend>默认插入位置</legend><div className="prompt-preset-segmented three">{[["start", "开头"], ["middle", "中间"], ["end", "结尾"]].map(([value, label]) => <button type="button" className={draft.position === value ? "active" : ""} aria-pressed={draft.position === value} disabled={running} key={value} onClick={() => update("position", value)}>{label}</button>)}</div><small>中间优先替换当前选区或插入光标处；没有有效选区时使用文本逻辑中点。</small></fieldset>
      </div>
      <footer><span>{dialog.record ? `ID ${dialog.record.id}` : "ID 将由安全随机源生成"}</span><div><button type="button" onClick={requestClose}>取消</button><button type="submit" className="primary" disabled={running || !validation.valid}><Save size={14} />{running ? "生成中已锁定" : "保存预设"}</button></div></footer>
    </form>
  </div>;
}

function PromptPresetDeleteDialog({ record, running, onConfirm, onClose }) {
  const dialogRef = useDialogLifecycle(true, onClose, `[data-prompt-preset-focus-fallback="${record.type}"]`);
  return <div className="prompt-preset-backdrop prompt-preset-delete-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
    <section ref={dialogRef} className="prompt-preset-delete-dialog" role="dialog" aria-modal="true" aria-labelledby="prompt-preset-delete-title" tabIndex="-1">
      <Trash2 size={25} /><h2 id="prompt-preset-delete-title">删除“{record.name}”？</h2><p>预设删除后不会自动恢复，包括原有内置预设。此操作不会改写当前 Prompt 文本。</p>{running && <p className="prompt-preset-readonly" role="status">生成任务运行期间不能删除预设。</p>}
      <div><button type="button" autoFocus onClick={onClose}>取消</button><button type="button" className="danger" disabled={running} onClick={onConfirm}>确认删除</button></div>
    </section>
  </div>;
}

function ThemeColorField({ label, detail, value, onChange }) {
  const [text, setText] = useState(value);

  useEffect(() => setText(value), [value]);

  const commit = (next) => {
    setText(next);
    const normalized = normalizeHex(next);
    if (normalized) onChange(normalized);
  };

  return <label className="theme-color-field">
    <span><b>{label}</b><small>{detail}</small></span>
    <span className="theme-color-inputs">
      <input className="theme-swatch" type="color" value={value} onChange={(event) => commit(event.target.value)} aria-label={`${label}取色`} />
      <input className={`theme-hex ${normalizeHex(text) ? "" : "invalid"}`} value={text} maxLength={7} spellCheck="false" aria-label={`${label}颜色码`} onChange={(event) => commit(event.target.value)} onBlur={() => setText(value)} />
    </span>
  </label>;
}

function Sparkline({ data, color = "var(--lime)", height = 48, max, label }) {
  const [tooltip, setTooltip] = useState(null);
  if (!data || data.length < 2) return <div className="sparkline-placeholder" style={{ height }} />;
  const ceiling = max ?? Math.max(...data, 1);
  const points = data.map((value, index) => {
    const x = (index / (data.length - 1)) * 100;
    const y = 100 - (value / ceiling) * 100;
    return `${x},${y}`;
  }).join(" ");
  const ticks = [0, Math.round(ceiling / 2), Math.round(ceiling)];
  const formatVal = (value) => label ? label(value) : value >= 1000 ? `${(value / 1000).toFixed(1)}K` : String(value);

  const handleMouse = (event) => {
    const wrapRect = event.currentTarget.getBoundingClientRect();
    const chartRect = event.currentTarget.querySelector("svg").getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (event.clientX - chartRect.left) / chartRect.width));
    const index = Math.round(ratio * (data.length - 1));
    const value = data[Math.max(0, Math.min(data.length - 1, index))];
    setTooltip({ x: event.clientX - wrapRect.left, value: formatVal(value) });
  };

  return (
    <div className="sparkline-wrap" style={{ height }} onMouseMove={handleMouse} onMouseLeave={() => setTooltip(null)}>
      <div className="sparkline-yaxis">
        {ticks.map((v, i) => <span key={i} style={i === 0 ? { bottom: 0 } : i === ticks.length - 1 ? { top: 0 } : { top: "50%", transform: "translateY(-50%)" }}>{formatVal(v)}</span>)}
      </div>
      <svg className="sparkline-svg" viewBox="0 0 100 100" preserveAspectRatio="none">
        <polyline points={points} fill="none" stroke={color} strokeWidth="2" vectorEffect="non-scaling-stroke" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      {tooltip && <div className="sparkline-tooltip" style={{ left: tooltip.x }}><span>{tooltip.value}</span></div>}
    </div>
  );
}

function PageLoading({ label }) {
  return <section className="page-loading" role="status" aria-live="polite"><i /><span>{label}</span></section>;
}

function App() {
  const viewerSession = useRef(new ViewerAsyncSession());
  const viewerFitRaf = useRef(null);
  useEffect(() => {
    viewerSession.current.mount();
    return () => {
      viewerFitRaf.current?.cancel();
      viewerFitRaf.current = null;
      viewerSession.current.unmount();
    };
  }, []);
  const [activePage, setActivePage] = useState("generate");
  const [restoredWorkspace, setRestoredWorkspace] = useState(null);
  const [theme, setTheme] = useState(DEFAULT_THEME);
  const [motionEnabled, setMotionEnabled] = useState(() => {
    try {
      const saved = window.localStorage.getItem("xirai_motion_enabled");
      return saved === null ? !window.matchMedia("(prefers-reduced-motion: reduce)").matches : saved !== "false";
    } catch {
      return true;
    }
  });
  const [uiStateReady, setUiStateReady] = useState(false);
  const restorePending = useRef({ checkpoint: true, diffusionModel: true, textEncoder: true, textEncoder2: true, vae: true, loras: true });
  const [model, setModel] = useState("SD");
  const [loras, setLoras] = useState([]);
  const [mountedLorasByEngine, setMountedLorasByEngine] = useState(emptyMountedLoraMap);
  const [loraGroupsByEngine, setLoraGroupsByEngine] = useState(emptyLoraGroupMap);
  // Session-only rescan signal. It intentionally is not part of the workspace
  // snapshot, and scan-prune commits do not change it, preventing a response
  // from continuously scheduling another automatic request.
  const [mountedLoraRescanGeneration, setMountedLoraRescanGeneration] = useState(0);
  const [mountedLorasWarning, setMountedLorasWarning] = useState("");
  const [mountedLorasRaw, setMountedLorasRaw] = useState(null);
  const [shouldPersistMountedLoras, setShouldPersistMountedLoras] = useState(true);
  const [loraWorkspaceLocked, setLoraWorkspaceLocked] = useState(false);
  const [positive, setPositive] = useState("a quiet observatory above the clouds, lone astronomer, warm light, cinematic composition");
  const [negative, setNegative] = useState("low quality, blurry, malformed hands, oversaturated, watermark");
  const [promptPresets, setPromptPresets] = useState(seededPromptPresetContainer);
  const [promptPresetLibraryError, setPromptPresetLibraryError] = useState("");
  const [promptPresetLibraryWarning, setPromptPresetLibraryWarning] = useState("");
  const [shouldPersistPromptPresets, setShouldPersistPromptPresets] = useState(true);
  const [promptPresetLibraryRaw, setPromptPresetLibraryRaw] = useState(null);
  const [promptPresetDialog, setPromptPresetDialog] = useState(null);
  const [promptPresetDelete, setPromptPresetDelete] = useState(null);
  const [steps, setSteps] = useState(28);
  const [cfg, setCfg] = useState(6.5);
  const [denoise, setDenoise] = useState(1);
  const [imageToImage, setImageToImage] = useState(IMAGE_TO_IMAGE_DEFAULTS);
  // The source picture is session state, never workspace state: it is a multi-megabyte data URL and
  // `ui-state.json` is rewritten on every settings change.
  const [imageSource, setImageSource] = useState(null);
  const [imagesPerBatch, setImagesPerBatch] = useState(1);
  const [batchCount, setBatchCount] = useState(1);
  const [seed, setSeed] = useState("847291");
  const [seedMode, setSeedMode] = useState("random");
  const [sampler, setSampler] = useState("dpmpp_2m");
  const [scheduler, setScheduler] = useState("karras");
  const [guidance, setGuidance] = useState("none");
  const [pag, setPag] = useState(PAG_DEFAULTS);
  const [size, setSize] = useState({ width: 1024, height: 1024 });
  const [status, setStatus] = useState("idle");
  const [activeJobRecoveryPending, setActiveJobRecoveryPending] = useState(true);
  const [progress, setProgress] = useState(0);
  const [generationPhase, setGenerationPhase] = useState("");
  const [generationError, setGenerationError] = useState("");
  const [generationWarning, setGenerationWarning] = useState("");
  const [promptConditioning, setPromptConditioning] = useState(null);
  const [generatedImage, setGeneratedImage] = useState("");
  const [generatedName, setGeneratedName] = useState("");
  const [generatedOutputs, setGeneratedOutputs] = useState([]);
  const [selectedOutputIndex, setSelectedOutputIndex] = useState(0);
  const [generatedSettings, setGeneratedSettings] = useState(null);
  const [galleryAddOpen, setGalleryAddOpen] = useState(false);
  const [galleryFocus, setGalleryFocus] = useState(null);
  const [appNotice, setAppNotice] = useState(null);
  const [livePreview, setLivePreview] = useState("");
  const [generationStage, setGenerationStage] = useState("queued");
  const [generationStageStep, setGenerationStageStep] = useState(0);
  const [generationStageTotal, setGenerationStageTotal] = useState(0);
  const [previewKind, setPreviewKind] = useState("");
  const [generationDetail, setGenerationDetail] = useState(null);
  const [generationStep, setGenerationStep] = useState(0);
  const [generationTotal, setGenerationTotal] = useState(0);
  const [generationBatchIndex, setGenerationBatchIndex] = useState(0);
  const [generationBatchCount, setGenerationBatchCount] = useState(1);
  const [generationCompletedImages, setGenerationCompletedImages] = useState(0);
  const [generationTotalImages, setGenerationTotalImages] = useState(1);
  const [generationTaskStatus, setGenerationTaskStatus] = useState("idle");
  const [generationElapsed, setGenerationElapsed] = useState(0);
  const [generationPausedTime, setGenerationPausedTime] = useState(0);
  const [generationControlBusy, setGenerationControlBusy] = useState("");
  const [generationProgressCollapsed, setGenerationProgressCollapsed] = useState(true);
  const [processPreview, setProcessPreview] = useState(true);
  const [generationJob, setGenerationJob] = useState("");
  const [inferenceHealth, setInferenceHealth] = useState(null);
  const [assistantOpen, setAssistantOpen] = useState(false);
  const [checkpoints, setCheckpoints] = useState([]);
  const [checkpoint, setCheckpoint] = useState("");
  const [checkpointDirectory, setCheckpointDirectory] = useState("");
  const [checkpointMissing, setCheckpointMissing] = useState(false);
  const [diffusionModels, setDiffusionModels] = useState([]);
  const [diffusionModel, setDiffusionModel] = useState("");
  const [diffusionModelDirectory, setDiffusionModelDirectory] = useState("");
  const [diffusionModelMissing, setDiffusionModelMissing] = useState(false);
  const [textEncoders, setTextEncoders] = useState([]);
  const [textEncoder, setTextEncoder] = useState("");
  const [textEncoderDirectory, setTextEncoderDirectory] = useState("");
  const [textEncoderMissing, setTextEncoderMissing] = useState(false);
  // FLUX.1 only. Both encoders live in the same directory, so the second picker shares
  // `textEncoderDirectory` and differs only in which files the catalogue classified into it.
  const [textEncoders2, setTextEncoders2] = useState([]);
  const [textEncoder2, setTextEncoder2] = useState("");
  const [textEncoder2Missing, setTextEncoder2Missing] = useState(false);
  const [vaes, setVaes] = useState([]);
  const [vae, setVae] = useState("");
  const [vaeDirectory, setVaeDirectory] = useState("");
  const [vaeMissing, setVaeMissing] = useState(false);
  const [animaRuntime, setAnimaRuntime] = useState(null);
  const [modelLoading, setModelLoading] = useState(true);
  const [modelSwitching, setModelSwitching] = useState(false);
  const [modelsRefreshing, setModelsRefreshing] = useState(false);
  const [modelError, setModelError] = useState("");
  const [loraLibrary, setLoraLibrary] = useState([]);
  const [loraDirectory, setLoraDirectory] = useState("");
  const [loraLoading, setLoraLoading] = useState(true);
  const [lorasRefreshing, setLorasRefreshing] = useState(false);
  const [loraError, setLoraError] = useState("");
  const [loraManagerOpen, setLoraManagerOpen] = useState(false);
  const [loraManagerMaximized, setLoraManagerMaximized] = useState(false);
  const [loraCategory, setLoraCategory] = useState("character");
  const [loraSearch, setLoraSearch] = useState("");
  const [collapsedModalSharedFolders, setCollapsedModalSharedFolders] = useState({});
  const [loraLookups, setLoraLookups] = useState({});
  const [loraDetail, setLoraDetail] = useState(null);
  const [hardwareMonitorOpen, setHardwareMonitorOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  // Layout of the generate page, held apart from every creation parameter so that resetting it
  // cannot reach one. Read synchronously on the first render: applying a stored width a frame late
  // would show the default panel and then snap.
  const [workspaceLayout, setWorkspaceLayout] = useState(() => readWorkspaceLayout(typeof window === "undefined" ? null : window.localStorage));
  const [panelResizing, setPanelResizing] = useState(false);
  const leftPanelRef = useRef(null);
  const panelResizeRef = useRef(null);
  const [settingsTab, setSettingsTab] = useState("general");
  const [settingsError, setSettingsError] = useState("");
  const [onlineUpdate, setOnlineUpdate] = useState({ checking: false, error: "", release: null, checked: false });
  const [updateConfirmOpen, setUpdateConfirmOpen] = useState(false);
  const [reconfiguring, setReconfiguring] = useState(false);
  const [performanceConfig, setPerformanceConfig] = useState(DEFAULT_PERFORMANCE);
  const [performanceEditorMode, setPerformanceEditorMode] = useState("recommended");
  const [performanceActive, setPerformanceActive] = useState({});
  const [performanceCapabilities, setPerformanceCapabilities] = useState({});
  const [performanceLoading, setPerformanceLoading] = useState(false);
  const [performanceSaving, setPerformanceSaving] = useState(false);
  const [performanceError, setPerformanceError] = useState("");
  const [diagnosticLogs, setDiagnosticLogs] = useState([]);
  const [selectedDiagnosticLog, setSelectedDiagnosticLog] = useState("");
  const [diagnosticLogContent, setDiagnosticLogContent] = useState("");
  const [diagnosticLogsLoading, setDiagnosticLogsLoading] = useState(false);
  const [diagnosticLogReading, setDiagnosticLogReading] = useState(false);
  const [diagnosticLogsClearing, setDiagnosticLogsClearing] = useState(false);
  const [diagnosticLogsError, setDiagnosticLogsError] = useState("");
  const [pluginRegistry, setPluginRegistry] = useState(null);
  const [pluginsLoading, setPluginsLoading] = useState(false);
  const [pluginsError, setPluginsError] = useState("");
  const [pluginPendingId, setPluginPendingId] = useState("");
  const [samplingExpanded, setSamplingExpanded] = useState(true);
  const [hires, setHires] = useState(HIRES_DEFAULTS);
  const [upscalersRefreshing, setUpscalersRefreshing] = useState(false);
  const [adetailer, setADetailer] = useState(ADETAILER_DEFAULTS);
  // Which unit the pager is showing. View state, not configuration: it is never
  // persisted, because where the user last paged to says nothing about the render.
  const [adetailerPage, setADetailerPage] = useState("");
  const [rtx, setRtx] = useState(RTX_DEFAULTS);
  const [postprocessOrder, setPostprocessOrder] = useState(POSTPROCESS_STAGE_IDS);
  const [adetailerModels, setADetailerModels] = useState([]);
  const [adetailerInfo, setADetailerInfo] = useState({ loading: true, available: false, runtimeAvailable: false, directory: "", builtins: [], error: "" });
  const [adetailerDownload, setADetailerDownload] = useState(null);
  const [backgroundRemovalDownload, setBackgroundRemovalDownload] = useState(null);
  const [backgroundRemovalModel, setBackgroundRemovalModel] = useState("");
  const [backgroundRemovalPickerOpen, setBackgroundRemovalPickerOpen] = useState(false);
  const [hardwareStats, setHardwareStats] = useState(null);
  const [hardwareHistory, setHardwareHistory] = useState({ gpu: [], vram: [], cpu: [], ram: [] });
  const [imageViewerOpen, setImageViewerOpen] = useState(false);
  const [viewerZoom, setViewerZoom] = useState(1);
  const [viewerPan, setViewerPan] = useState({ x: 0, y: 0 });
  const [viewerExpanded, setViewerExpanded] = useState(false);
  const [viewerSidebarOpen, setViewerSidebarOpen] = useState(false);
  const [viewerHistory, setViewerHistory] = useState([]);
  const [viewerFolders, setViewerFolders] = useState([]);
  const [viewerDirectory, setViewerDirectory] = useState({ id: "", name: "输出目录", label: "/", parent_id: "", image_count: 0, folders: [] });
  const [viewerOutputRootId, setViewerOutputRootId] = useState("");
  const [viewerSelectedFolder, setViewerSelectedFolder] = useState("");
  const [viewerHistoryLoading, setViewerHistoryLoading] = useState(false);
  const [viewerHistoryBatch, setViewerHistoryBatch] = useState(null);
  const [viewerHistoryReturnId, setViewerHistoryReturnId] = useState("");
  const [hiddenHistoryAssets, setHiddenHistoryAssets] = useState([]);
  const [viewerLayers, setViewerLayers] = useState([]);
  const [viewerLayerResizeEnabled, setViewerLayerResizeEnabled] = useState(true);
  const [activeViewerLayer, setActiveViewerLayer] = useState("");
  const [viewerGridSize, setViewerGridSize] = useState(16);
  const [viewerGridEnabled, setViewerGridEnabled] = useState(true);
  const [viewerEdgeSnapEnabled, setViewerEdgeSnapEnabled] = useState(false);
  const [viewerAlignmentGuidesEnabled, setViewerAlignmentGuidesEnabled] = useState(true);
  const [viewerSnapGuide, setViewerSnapGuide] = useState(null);
  const [viewerSnappedLayers, setViewerSnappedLayers] = useState([]);
  const [viewerLayerEdges, setViewerLayerEdges] = useState({});
  const [viewerEdgePanelOpen, setViewerEdgePanelOpen] = useState(false);
  const [viewerEdgeLine, setViewerEdgeLine] = useState({ enabled: false, style: "solid", color: "#D6FF3F", width: 2 });
  const [viewerTemplatesOpen, setViewerTemplatesOpen] = useState(false);
  const [activeCollage, setActiveCollage] = useState(null);
  const [activeCollageSlot, setActiveCollageSlot] = useState(-1);
  const [collageResult, setCollageResult] = useState(null);
  const [viewerMenu, setViewerMenu] = useState(null);
  const [historyDelete, setHistoryDelete] = useState(null);
  const [viewerNotice, setViewerNotice] = useState("");
  const [viewerToolbarFocus, setViewerToolbarFocus] = useState({ focusTarget: null, focusReturn: null });
  const [viewerToolbarHeight, setViewerToolbarHeight] = useState(46);
  const [initialLoading, setInitialLoading] = useState(true);
  const [loaderLeaving, setLoaderLeaving] = useState(false);
  const [consoleOpen, setConsoleOpen] = useState(false);
  const [consoleHeight, setConsoleHeight] = useState(300);
  const [consoleEntries, setConsoleEntries] = useState([]);
  const [consoleCommand, setConsoleCommand] = useState("");
  const [consoleRunning, setConsoleRunning] = useState(false);
  const [consoleError, setConsoleError] = useState("");
  const viewerDrag = useRef(null);
  const viewerNudge = useRef(null);
  const viewerUndo = useRef([]);
  const viewerClipboard = useRef(null);
  const viewerCanvasRef = useRef(null);
  const viewerToolbarRef = useRef(null);
  const viewerEdgeTriggerRef = useRef(null);
  const viewerTemplateTriggerRef = useRef(null);
  const viewerEdgePanelRef = useRef(null);
  const viewerTemplatePanelRef = useRef(null);
  const viewerHistoryRef = useRef(null);
  const viewerHistoryScroll = useRef(0);
  const loaderStartedAt = useRef(Date.now());
  const generationLocked = useRef(false);
  const completedModelDownloadJob = useRef("");
  const inferenceHealthRef = useRef(inferenceHealth);
  const currentModel = useRef(model);
  const currentCheckpoint = useRef(checkpoint);
  const currentDiffusionModel = useRef(diffusionModel);
  const currentTextEncoder = useRef(textEncoder);
  const currentTextEncoder2 = useRef(textEncoder2);
  const currentVae = useRef(vae);
  const modelsRefreshInFlight = useRef(false);
  const loraLookupControllers = useRef(new Set());
  const loraSyncChannel = useRef(null);
  const loraSyncReceiving = useRef(false);
  const lorasRef = useRef(loras);
  const loraDragGateRef = useRef(null);
  const mountedLorasMapRef = useRef(mountedLorasByEngine);
  // Deliberately session-only: it protects a scan response without changing
  // the persisted mountedLorasByEngine v2 schema.
  const mountedLoraRevisionRef = useRef(0);
  const loraGroupsMapRef = useRef(emptyLoraGroupMap());
  const activeLoraGroupsRef = useRef([]);
  const activeLoraScopeRef = useRef(null);
  const loraScanToken = useRef(0);
  const loraScanGate = useRef(null);
  // Which model identity the browser view (category tab, search box, lookup
  // cache) was last reset for. The scan effect re-runs for far more reasons
  // than a model change, and those must not move the user's view.
  const loraViewScopeRef = useRef(null);
  const activeJobRecoveryToken = useRef(0);
  const consoleLatest = useRef(0);
  const consoleOutputRef = useRef(null);
  const backgroundRemovalPickerRef = useRef(null);
  const positivePromptRef = useRef(null);
  const negativePromptRef = useRef(null);
  const promptSelectionCache = useRef({ positive: null, negative: null });
  const promptTextRevision = useRef({
    positive: { text: positive, revision: 0 },
    negative: { text: negative, revision: 0 },
  });
  const promptFocusRequest = useRef(null);
  const promptFocusRaf = useRef(null);
  const promptFocusSession = useRef(0);
  const workspaceSnapshot = useRef(null);
  const uiStateSnapshot = useRef(null);
  inferenceHealthRef.current = inferenceHealth;
  currentModel.current = model;
  currentCheckpoint.current = checkpoint;
  currentDiffusionModel.current = diffusionModel;
  currentTextEncoder.current = textEncoder;
  currentTextEncoder2.current = textEncoder2;
  currentVae.current = vae;
  lorasRef.current = loras;
  loraDragGateRef.current = { status, modelSwitching, loraWorkspaceLocked, shouldPersistMountedLoras };
  loraScanGate.current = { activeJobRecoveryPending, status, modelSwitching, workspaceLocked: loraWorkspaceLocked, shouldPersist: shouldPersistMountedLoras, model };
  for (const [type, text] of [["positive", positive], ["negative", negative]]) {
    if (promptTextRevision.current[type].text !== text) {
      promptTextRevision.current[type] = { text, revision: promptTextRevision.current[type].revision + 1 };
    }
  }
  const persistedPromptPresets = promptPresetLibraryError && !shouldPersistPromptPresets ? promptPresetLibraryRaw : promptPresets;
  const activeLoraScopeKey = engineScopeKey(model);
  const activeScopedLoras = mountedLorasForScope(mountedLorasByEngine, activeLoraScopeKey);
  const activeLoraGroups = loraGroupsForScope(loraGroupsByEngine, activeLoraScopeKey);
  activeLoraGroupsRef.current = activeLoraGroups;
  /**
   * The group facts a card should record. A card built from a finished
   * generation keeps that run's frozen record; one built from the workspace —
   * the card editor, or adding to the gallery before anything was generated —
   * describes what is mounted right now. Both go through here so a new
   * card-producing path cannot silently omit them again.
   */
  const galleryCardSettings = (source, { record } = {}) => {
    const base = gallerySettingsWithoutPromptPresets(source);
    if (record) return base;
    const enabled = enabledLoraGroups(activeLoraGroups);
    return {
      ...base,
      loraGroups: enabled.map((group) => ({ id: group.id, name: group.name, presetPrompt: group.presetPrompt })),
      loraGroupPrompt: composeGroupPrompt(enabled, ""),
    };
  };
  const loraDragLocked = appLoraDragLocked({ status, modelSwitching, loraWorkspaceLocked, shouldPersistMountedLoras });
  workspaceSnapshot.current = {
    model, checkpoint, diffusionModel, textEncoder, textEncoder2, vae, positive, negative, steps, cfg, denoise, imagesPerBatch, batchCount, seed: normalizeSeed(seed), seedMode,
    promptPresets: persistedPromptPresets,
    sampler, scheduler, guidance, pag, size, processPreview, samplingExpanded, hires, adetailer, rtx, postprocessOrder, loraCategory, loraSearch,
    mountedLorasByEngine: shouldPersistMountedLoras ? mountedLorasByEngine : mountedLorasRaw,
    loraGroupsByEngine,
    // Compatibility mirror for existing consumers. It is always derived from
     // the active engine, never used as a cross-engine source.
    loras: activeScopedLoras,
    backgroundRemovalModel,
    imageToImage,
  };
  uiStateSnapshot.current = { theme, workspace: workspaceSnapshot.current };

  const persistUiState = (keepalive = false) => {
    if (!uiStateReady) return;
    if ((promptPresetLibraryError && !shouldPersistPromptPresets) || !shouldPersistMountedLoras) return;
    void fetch("/api/ui-state", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(uiStateSnapshot.current),
      keepalive,
      }).catch(() => {});
  };

  const commitMountedLoraMap = (container, { rescan = true } = {}) => {
    const previous = mountedLorasMapRef.current;
    const nextRevision = nextMountedLoraRevision(mountedLoraRevisionRef.current, previous, container);
    const changed = nextRevision !== mountedLoraRevisionRef.current;
    mountedLorasMapRef.current = container;
    if (changed) {
      mountedLoraRevisionRef.current = nextRevision;
      if (rescan) setMountedLoraRescanGeneration(nextRevision);
      // A semantic map mutation has invalidated any manual request that was
      // showing a spinner. Its completion is not allowed to update indicators.
      setLorasRefreshing(false);
      setLoraLoading(false);
    }
    setMountedLorasByEngine(container);
    return changed;
  };

  const commitMountedLoras = (updater, scopeKey = activeLoraScopeRef.current || activeLoraScopeKey) => {
    if (status === "running" || modelSwitching || loraWorkspaceLocked || !scopeKey || !shouldPersistMountedLoras) return false;
    const result = updateMountedLorasForScope(mountedLorasMapRef.current, scopeKey, updater);
    if (!result.changed) return false;
    commitMountedLoraMap(result.container);
    if (scopeKey === activeLoraScopeRef.current || scopeKey === activeLoraScopeKey) {
      lorasRef.current = result.loras;
      setLoras(result.loras);
    }
    return true;
  };

  // Groups follow the mounted library's locks: a combination cannot be switched
  // on while a job is running any more than a LoRA can be mounted.
  const commitLoraGroups = (updater, scopeKey = activeLoraScopeRef.current || activeLoraScopeKey) => {
    if (status === "running" || modelSwitching || loraWorkspaceLocked || !scopeKey) return false;
    const result = updateLoraGroupsForScope(loraGroupsMapRef.current, scopeKey, updater);
    if (!result.changed) return false;
    loraGroupsMapRef.current = result.container;
    setLoraGroupsByEngine(result.container);
    return true;
  };

    const transitionActiveLoraScope = (targetEngine, { targetLoras } = {}) => {
    const targetScopeKey = engineScopeKey(targetEngine);
    const transition = transitionMountedLoraScope(mountedLorasMapRef.current, {
      sourceEngine: activeLoraScopeRef.current,
      sourceLoras: lorasRef.current,
      targetEngine: targetScopeKey,
      accepted: true,
    });
    const container = targetScopeKey && Array.isArray(targetLoras)
      ? withMountedLorasForScope(transition.container, targetScopeKey, targetLoras)
      : transition.container;
    const nextLoras = targetScopeKey ? mountedLorasForScope(container, targetScopeKey) : [];
    activeLoraScopeRef.current = targetScopeKey;
    lorasRef.current = nextLoras;
    commitMountedLoraMap(container);
    setLoras(nextLoras);
    return { scopeKey: targetScopeKey, loras: nextLoras };
  };

  // Asset resolution can produce a complete key after an engine switch. This
  // effect only changes the active mirror; it never waits for (or is reset by)
  // a LoRA directory scan.
  useEffect(() => {
    const targetScopeKey = activeLoraScopeKey;
    if (targetScopeKey === activeLoraScopeRef.current) {
      const stored = mountedLorasForScope(mountedLorasMapRef.current, targetScopeKey);
      if (!sameMountedLoras(stored, lorasRef.current)) {
        lorasRef.current = stored;
        setLoras(stored);
      }
      return;
    }
    const transition = transitionMountedLoraScope(mountedLorasMapRef.current, {
      sourceEngine: activeLoraScopeRef.current,
      sourceLoras: lorasRef.current,
      targetEngine: targetScopeKey,
      accepted: true,
    });
    activeLoraScopeRef.current = transition.activeScopeKey;
    lorasRef.current = transition.activeLoras;
    commitMountedLoraMap(transition.container);
    setLoras(transition.activeLoras);
  }, [activeLoraScopeKey]);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/ui-state", { cache: "no-store" })
      .then(async (response) => {
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "无法读取项目页面状态");
        return payload.state;
      })
      .catch(() => null)
      .then((saved) => {
        if (cancelled) return;
        const workspace = loadWorkspaceState(saved?.workspace);
        if (workspace) {
          commitMountedLoraMap(workspace.mountedLorasByEngine);
          loraGroupsMapRef.current = workspace.loraGroupsByEngine;
          setLoraGroupsByEngine(workspace.loraGroupsByEngine);
          if (workspace.loraGroupsWarning) setAppNotice({ message: workspace.loraGroupsWarning, error: true });
          lorasRef.current = workspace.loras;
          activeLoraScopeRef.current = workspace.activeLoraScopeKey;
          setRestoredWorkspace(workspace);
          setModel(workspace.model);
          setLoras(workspace.loras);
          setMountedLorasWarning(workspace.mountedLorasWarning);
          setMountedLorasRaw(workspace.mountedLorasRaw);
          setShouldPersistMountedLoras(workspace.shouldPersistMountedLoras);
          activeLoraScopeRef.current = workspace.activeLoraScopeKey;
          setPositive(workspace.positive);
          setNegative(workspace.negative);
          setPromptPresets(workspace.promptPresets);
          setPromptPresetLibraryError(workspace.promptPresetLibraryError);
          setPromptPresetLibraryWarning(workspace.promptPresetLibraryWarning);
          setPromptPresetLibraryRaw(workspace.promptPresetLibraryRaw);
          setShouldPersistPromptPresets(workspace.shouldPersistPromptPresets);
          setSteps(workspace.steps);
          setCfg(workspace.cfg);
          setDenoise(workspace.denoise);
          setImagesPerBatch(workspace.imagesPerBatch);
          setBatchCount(workspace.batchCount);
          setSeed(workspace.seed);
          setSeedMode(workspace.seedMode);
          setSampler(workspace.sampler);
          setScheduler(workspace.scheduler);
          setGuidance(workspace.guidance);
          setPag(workspace.pag);
          setSize(workspace.size);
          setProcessPreview(workspace.processPreview);
          setBackgroundRemovalModel(workspace.backgroundRemovalModel);
          setCheckpoint(workspace.checkpoint);
          setDiffusionModel(workspace.diffusionModel);
          setTextEncoder(workspace.textEncoder);
          setTextEncoder2(workspace.textEncoder2);
          setVae(workspace.vae);
          setLoraCategory(workspace.loraCategory);
          setLoraSearch(workspace.loraSearch);
          setSamplingExpanded(workspace.samplingExpanded);
          setHires({ ...HIRES_DEFAULTS, ...workspace.hires });
          setADetailer({ ...ADETAILER_DEFAULTS, ...workspace.adetailer });
          setRtx({ ...RTX_DEFAULTS, ...workspace.rtx });
          setPostprocessOrder(workspace.postprocessOrder);
          setImageToImage(workspace.imageToImage);
        }
        setTheme(loadThemeState(saved?.theme));
        try {
          if (saved?.motionEnabled === false && window.localStorage.getItem("xirai_motion_enabled") === null) {
            window.localStorage.setItem("xirai_motion_enabled", "false");
            setMotionEnabled(false);
          }
        } catch {}
        setUiStateReady(true);
      });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    applyThemeToDocument(theme);
  }, [theme]);

  useEffect(() => {
    document.documentElement.dataset.motion = motionEnabled ? "on" : "off";
    if (!uiStateReady) return;
    try { window.localStorage.setItem("xirai_motion_enabled", String(motionEnabled)); } catch {}
  }, [motionEnabled, uiStateReady]);

  useEffect(() => {
    if (!consoleOpen) return undefined;
    let stopped = false;
    const poll = async () => {
      try {
        const response = await fetch(`/api/console?after=${consoleLatest.current}`, { cache: "no-store" });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "无法读取控制台输出");
        if (stopped) return;
        if (payload.entries?.length) {
          consoleLatest.current = payload.latest || consoleLatest.current;
          setConsoleEntries((current) => [...current, ...payload.entries].slice(-1200));
        }
        setConsoleRunning(payload.command_running === true);
        setConsoleError("");
      } catch (error) {
        if (!stopped) setConsoleError(error.message);
      }
    };
    void poll();
    const timer = window.setInterval(poll, 900);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [consoleOpen]);

  useEffect(() => {
    if (!consoleOpen || !consoleOutputRef.current) return;
    consoleOutputRef.current.scrollTop = consoleOutputRef.current.scrollHeight;
  }, [consoleOpen, consoleEntries]);

  useEffect(() => {
    if (!uiStateReady) return undefined;
    const timer = window.setTimeout(persistUiState, 300);
    return () => window.clearTimeout(timer);
  }, [theme, model, checkpoint, diffusionModel, textEncoder, textEncoder2, vae, positive, negative, promptPresets, promptPresetLibraryError, shouldPersistPromptPresets, mountedLorasByEngine, shouldPersistMountedLoras, steps, cfg, denoise, imagesPerBatch, batchCount, seed, seedMode, sampler, scheduler, guidance, pag, size, processPreview, samplingExpanded, hires, adetailer, rtx, postprocessOrder, loraCategory, loraSearch, loras, backgroundRemovalModel, imageToImage, uiStateReady]);

  useEffect(() => {
    const saveBeforeExit = () => persistUiState(true);
    window.addEventListener("pagehide", saveBeforeExit);
    return () => window.removeEventListener("pagehide", saveBeforeExit);
  }, [uiStateReady]);

  useEffect(() => {
    const pending = promptFocusRequest.current;
    if (!pending) return undefined;
    if (pending.text !== (pending.type === "positive" ? positive : negative)) {
      promptFocusRequest.current = null;
      promptFocusSession.current += 1;
      return undefined;
    }
    if (promptFocusRaf.current !== null) window.cancelAnimationFrame(promptFocusRaf.current);
    promptFocusRaf.current = window.requestAnimationFrame(() => {
      promptFocusRaf.current = null;
      if (pending.session !== promptFocusSession.current) return;
      const node = pending.type === "positive" ? positivePromptRef.current : negativePromptRef.current;
      if (!node?.isConnected || node.value !== pending.text) return;
      node.focus();
      node.setSelectionRange(pending.caret, pending.caret);
      recordPromptSelection(pending.type, { currentTarget: node });
    });
    return () => {
      if (promptFocusRaf.current !== null) window.cancelAnimationFrame(promptFocusRaf.current);
      promptFocusRaf.current = null;
    };
  }, [positive, negative]);

  useEffect(() => () => {
    promptFocusSession.current += 1;
    if (promptFocusRaf.current !== null) window.cancelAnimationFrame(promptFocusRaf.current);
    promptFocusRaf.current = null;
    promptFocusRequest.current = null;
  }, []);

  useEffect(() => {
    if (!uiStateReady) return undefined;
    const controller = new AbortController();
    setModelLoading(true);
    setModelError("");
    setCheckpoints([]);
    setDiffusionModels([]);
    setTextEncoders([]);
    setTextEncoders2([]);
    setVaes([]);
    setCheckpointDirectory("");
    setDiffusionModelDirectory("");
    setTextEncoderDirectory("");
    setVaeDirectory("");
    setAnimaRuntime(null);
    const restoringCheckpoint = restorePending.current.checkpoint && restoredWorkspace?.model === model;
    const restoringDiffusionModel = restorePending.current.diffusionModel && restoredWorkspace?.model === model;
    const restoringTextEncoder = restorePending.current.textEncoder && restoredWorkspace?.model === model;
    const restoringTextEncoder2 = restorePending.current.textEncoder2 && restoredWorkspace?.model === model;
    const restoringVae = restorePending.current.vae && restoredWorkspace?.model === model;
    if (!restoringCheckpoint) setCheckpoint("");
    if (!restoringDiffusionModel) setDiffusionModel("");
    if (!restoringTextEncoder) setTextEncoder("");
    if (!restoringTextEncoder2) setTextEncoder2("");
    if (!restoringVae) setVae("");
    setCheckpointMissing(false);
    setDiffusionModelMissing(false);
    setTextEncoderMissing(false);
    setTextEncoder2Missing(false);
    setVaeMissing(false);

    fetch(`/api/models?engine=${encodeURIComponent(model)}`, { signal: controller.signal })
      .then(async (response) => {
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "无法读取模型目录");
        return payload;
      })
      .then((payload) => {
        if (payload.model_type === "split") {
          const diffusionAsset = payload.assets?.diffusion_model || {};
          const textEncoderAsset = payload.assets?.text_encoder || {};
          const textEncoder2Asset = payload.assets?.text_encoder_2 || {};
          const vaeAsset = payload.assets?.vae || {};
          const nextDiffusionModels = diffusionAsset.models || [];
          const nextTextEncoders = textEncoderAsset.models || [];
          const nextTextEncoders2 = textEncoder2Asset.models || [];
          const nextVaes = vaeAsset.models || [];
          setDiffusionModels(nextDiffusionModels);
          setTextEncoders(nextTextEncoders);
          setTextEncoders2(nextTextEncoders2);
          setVaes(nextVaes);
          setDiffusionModelDirectory(diffusionAsset.directory || "");
          setTextEncoderDirectory(textEncoderAsset.directory || "");
          setVaeDirectory(vaeAsset.directory || "");
          setAnimaRuntime(payload.runtime || null);
          if (restoringDiffusionModel && restoredWorkspace.diffusionModel) {
            setDiffusionModel(restoredWorkspace.diffusionModel);
            setDiffusionModelMissing(!nextDiffusionModels.some((item) => item.value === restoredWorkspace.diffusionModel));
          } else setDiffusionModel(nextDiffusionModels[0]?.value ?? "");
          if (restoringTextEncoder && restoredWorkspace.textEncoder) {
            setTextEncoder(restoredWorkspace.textEncoder);
            setTextEncoderMissing(!nextTextEncoders.some((item) => item.value === restoredWorkspace.textEncoder));
          } else setTextEncoder(nextTextEncoders[0]?.value ?? "");
          if (restoringTextEncoder2 && restoredWorkspace.textEncoder2) {
            setTextEncoder2(restoredWorkspace.textEncoder2);
            setTextEncoder2Missing(!nextTextEncoders2.some((item) => item.value === restoredWorkspace.textEncoder2));
          } else setTextEncoder2(nextTextEncoders2[0]?.value ?? "");
          if (restoringVae && restoredWorkspace.vae) {
            setVae(restoredWorkspace.vae);
            setVaeMissing(!nextVaes.some((item) => item.value === restoredWorkspace.vae));
          } else setVae(nextVaes[0]?.value ?? "");
          setCheckpoint("");
        } else {
          const nextCheckpoints = payload.models || [];
          setCheckpoints(nextCheckpoints);
          setCheckpointDirectory(payload.directory || "");
          if (restoringCheckpoint && restoredWorkspace.checkpoint) {
            setCheckpoint(restoredWorkspace.checkpoint);
            setCheckpointMissing(!nextCheckpoints.some((item) => item.value === restoredWorkspace.checkpoint));
          } else setCheckpoint(nextCheckpoints[0]?.value ?? "");
          setDiffusionModel("");
          setTextEncoder("");
          setTextEncoder2("");
          setVae("");
        }
        restorePending.current = { ...restorePending.current, checkpoint: false, diffusionModel: false, textEncoder: false, textEncoder2: false, vae: false };
      })
      .catch((error) => {
        if (error.name !== "AbortError") {
          restorePending.current = { ...restorePending.current, checkpoint: false, diffusionModel: false, textEncoder: false, textEncoder2: false, vae: false };
          setModelError(error.message);
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setModelLoading(false);
      });

    return () => {
      controller.abort();
      for (const lookupController of loraLookupControllers.current) lookupController.abort();
      loraLookupControllers.current.clear();
    };
  }, [model, uiStateReady]);

  useEffect(() => {
    if (!canStartMountedLoraScan({ uiStateReady, ...loraScanGate.current })) {
      setLoraLoading(false);
      setLorasRefreshing(false);
      return undefined;
    }
    for (const lookupController of loraLookupControllers.current) lookupController.abort();
    loraLookupControllers.current.clear();
    setLoraLoading(true);
    setLorasRefreshing(false);
    setLoraError("");
    const restoringLoras = restorePending.current.loras && restoredWorkspace?.model === model;
    const scanScopeKey = activeLoraScopeKey;
    const scanToken = ++loraScanToken.current;
    const scanRevision = mountedLoraRevisionRef.current;
    // Only a change of model identity may move the browser view. This effect
    // also re-runs for a rescan, a lock change and every job transition, and
    // resetting then threw the user out of the tab they were working in — with
    // a weight edit that meant the mounted list unmounted under the pointer.
    const viewScope = `${model}::${activeLoraScopeKey || ""}`;
    if (loraViewScopeRef.current !== viewScope) {
      loraViewScopeRef.current = viewScope;
      if (!restoringLoras) {
        setLoraCategory("character");
        setLoraSearch("");
      }
      setLoraLookups({});
    }
    const controller = new AbortController();

    fetch(`/api/loras?engine=${encodeURIComponent(model)}`, { signal: controller.signal })
      .then(async (response) => {
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "无法读取 LoRA 目录");
        return payload;
      })
      .then((payload) => {
        const gate = loraScanGate.current;
        if (controller.signal.aborted || model !== currentModel.current) return;
        const scan = reconcileMountedLoraScan({
          getCurrentContainer: () => mountedLorasMapRef.current,
          categories: payload.categories,
          scanScopeKey,
          requestToken: scanToken,
          latestToken: loraScanToken.current,
          capturedRevision: scanRevision,
          latestRevision: mountedLoraRevisionRef.current,
          responseEngine: model,
          activeEngine: activeLoraScopeRef.current,
          ...gate,
        });
        if (!scan.applied) return;
        setLoraLibrary(payload.categories);
        setLoraDirectory(payload.directory);
        const selectedExists = payload.categories.some((category) => category.id === restoredWorkspace?.loraCategory);
        if (restoringLoras && !selectedExists) {
          setLoraCategory(payload.categories.find((category) => category.models.length)?.id || "character");
        }
        // The pure reconciliation reads the current map only after token,
        // engine, recovery/lock and revision admission. A stale listing may
        // therefore never prune a local or cross-window edit.
        if (scan.changed) {
          lorasRef.current = scan.loras;
          commitMountedLoraMap(scan.container, { rescan: false });
          setLoras(scan.loras);
        }
        restorePending.current.loras = false;
        setLoraError("");
      })
      .catch((error) => {
        if (error.name !== "AbortError" && shouldApplyMountedLoraScan({ requestToken: scanToken, latestToken: loraScanToken.current, capturedRevision: scanRevision, latestRevision: mountedLoraRevisionRef.current, responseEngine: model, activeEngine: activeLoraScopeRef.current, ...loraScanGate.current })) {
          restorePending.current.loras = false;
          setLoraError(error.message);
        }
      })
      .finally(() => {
        if (!controller.signal.aborted && shouldApplyMountedLoraScan({ requestToken: scanToken, latestToken: loraScanToken.current, capturedRevision: scanRevision, latestRevision: mountedLoraRevisionRef.current, responseEngine: model, activeEngine: activeLoraScopeRef.current, ...loraScanGate.current })) setLoraLoading(false);
      });

    return () => controller.abort();
  }, [model, activeLoraScopeKey, mountedLoraRescanGeneration, uiStateReady, activeJobRecoveryPending, status, modelSwitching, loraWorkspaceLocked, shouldPersistMountedLoras]);

  const refreshADetailerModels = async (signal) => {
    setADetailerInfo((current) => ({ ...current, loading: true, error: "" }));
    try {
      const response = await fetch("/api/inference/adetailer/models", { cache: "no-store", signal });
      const body = await response.text();
      let payload = {};
      if (body) {
        try {
          payload = JSON.parse(body);
        } catch {
          if (response.ok) throw new Error("ADetailer 检测服务返回了无效响应");
        }
      }
      if (!response.ok) throw new Error(payload.detail || payload.error || `ADetailer 检测服务暂时不可用（HTTP ${response.status}）`);
      const models = payload.models || [];
      setADetailerModels(models);
      setADetailerInfo({
        loading: false,
        available: payload.available === true,
        runtimeAvailable: payload.runtime_available === true,
        directory: payload.directory || "",
        builtins: payload.builtins || [],
        error: "",
      });
      setADetailer((current) => {
        const values = new Set(models.map((item) => item.value));
        // A unit whose detector vanished from the library falls back to the
        // default rather than silently pointing at a file that is gone.
        const units = current.units.map((unit) => (
          unit.detector && values.has(unit.detector) ? unit : { ...unit, detector: payload.default || "" }
        ));
        return units.some((unit, index) => unit !== current.units[index]) ? { ...current, units } : current;
      });
      return true;
    } catch (error) {
      if (error.name !== "AbortError") {
        setADetailerInfo({ loading: false, available: false, runtimeAvailable: false, directory: "", builtins: [], error: error.message });
      }
      return false;
    }
  };

  const refreshUpscalers = async (signal) => {
    setUpscalersRefreshing(true);
    try {
      const response = await fetch("/api/inference/upscalers", { cache: "no-store", signal });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "无法读取超分模型目录");
      setInferenceHealth((current) => current ? { ...current, upscalers: payload } : current);
      return payload;
    } catch (error) {
      if (error.name !== "AbortError") {
        setInferenceHealth((current) => current ? { ...current, upscalers: { runtime_available: false, available: false, directory: "", models: [], error: error.message } } : current);
      }
      return null;
    } finally {
      setUpscalersRefreshing(false);
    }
  };

  useEffect(() => {
    if (inferenceHealth?.status !== "ready") return undefined;
    const controller = new AbortController();
    let retryTimer;
    const refresh = async () => {
      const ready = await refreshADetailerModels(controller.signal);
      if (!ready && !controller.signal.aborted) retryTimer = window.setTimeout(refresh, 2000);
    };
    void refresh();
    return () => {
      controller.abort();
      window.clearTimeout(retryTimer);
    };
  }, [inferenceHealth?.status]);

  useEffect(() => {
    const inferenceSettled = ["ready", "error", "offline"].includes(inferenceHealth?.status);
    if (modelLoading || loraLoading || !inferenceSettled || !initialLoading) return undefined;

    const minimumDelay = Math.max(0, 1000 - (Date.now() - loaderStartedAt.current));
    const leaveTimer = window.setTimeout(() => setLoaderLeaving(true), minimumDelay);
    const removeTimer = window.setTimeout(() => setInitialLoading(false), minimumDelay + 480);
    return () => {
      window.clearTimeout(leaveTimer);
      window.clearTimeout(removeTimer);
    };
  }, [modelLoading, loraLoading, inferenceHealth?.status, initialLoading]);

  useEffect(() => {
    if (!loraManagerOpen) return undefined;
    const handleEscape = (event) => event.key === "Escape" && !loraDetail && setLoraManagerOpen(false);
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", handleEscape);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleEscape);
    };
  }, [loraManagerOpen, loraDetail]);

  useEffect(() => {
    if (!uiStateReady || typeof BroadcastChannel === "undefined") return undefined;
    const channel = new BroadcastChannel(LORA_SYNC_CHANNEL);
    loraSyncChannel.current = channel;
    channel.onmessage = (event) => {
      const payload = event.data;
      if (payload?.type === "request-workspace-loras") {
        const scopeKey = activeLoraScopeRef.current;
        channel.postMessage({ type: "workspace-loras", engine: scopeKey, loras: mountedLorasForScope(mountedLorasMapRef.current, scopeKey), locked: status === "running" || modelSwitching });
        return;
      }
      if (payload?.type === "workspace-lora-lock") {
        const locked = payload.locked === true;
        loraDragGateRef.current = { ...(loraDragGateRef.current || {}), loraWorkspaceLocked: locked };
        setLoraWorkspaceLocked(locked);
        return;
      }
      if (payload?.type !== "workspace-loras") return;
      const locked = payload.locked === true;
      loraDragGateRef.current = { ...(loraDragGateRef.current || {}), loraWorkspaceLocked: locked };
      setLoraWorkspaceLocked(locked);
      if (locked || status === "running" || modelSwitching || !shouldPersistMountedLoras) return;
      const synced = applyMountedLoraSync(mountedLorasMapRef.current, { engine: payload.engine || payload.scopeKey, loras: payload.loras }, { activeEngine: activeLoraScopeRef.current, locked: false });
      if (!synced.applied) return;
      const mapChanged = !sameMountedLoraMap(mountedLorasMapRef.current, synced.container);
      if (mapChanged) {
        commitMountedLoraMap(synced.container);
        loraSyncReceiving.current = true;
      }
      if ((payload.engine || payload.scopeKey) === activeLoraScopeRef.current) {
        if (!sameMountedLoras(synced.activeLoras, lorasRef.current)) {
          loraSyncReceiving.current = true;
          lorasRef.current = synced.activeLoras;
          setLoras(synced.activeLoras);
        }
      }
    };
    return () => {
      channel.close();
      if (loraSyncChannel.current === channel) loraSyncChannel.current = null;
    };
  }, [uiStateReady, status, modelSwitching, loraWorkspaceLocked, shouldPersistMountedLoras]);

  useEffect(() => {
    if (!uiStateReady || !loraSyncChannel.current) return;
    if (loraSyncReceiving.current) {
      loraSyncReceiving.current = false;
      return;
    }
    const scopeKey = activeLoraScopeRef.current;
    if (!scopeKey || status === "running" || modelSwitching || loraWorkspaceLocked) return;
    loraSyncChannel.current.postMessage({ type: "workspace-loras", engine: scopeKey, loras: mountedLorasForScope(mountedLorasMapRef.current, scopeKey), locked: false });
  }, [uiStateReady, activeLoraScopeKey, mountedLorasByEngine, loras, status, modelSwitching, loraWorkspaceLocked]);

  useEffect(() => {
    if (!uiStateReady || !loraSyncChannel.current) return;
    loraSyncChannel.current.postMessage({ type: "workspace-lora-lock", locked: status === "running" || modelSwitching });
  }, [uiStateReady, status, modelSwitching]);

  useEffect(() => {
    if (!appNotice) return undefined;
    const timer = window.setTimeout(() => setAppNotice(null), appNotice.error ? 6000 : 3200);
    return () => window.clearTimeout(timer);
  }, [appNotice]);

  useEffect(() => {
    if (mountedLorasWarning) setAppNotice({ message: mountedLorasWarning, error: true });
  }, [mountedLorasWarning]);

  useEffect(() => {
    if (!imageViewerOpen) return undefined;
    void refreshViewerHistory();
    const handleEscape = (event) => {
      if (event.key !== "Escape") return;
      const action = viewerEscapeAction({ historyDelete, contextMenu: viewerMenu, popover: viewerEdgePanelOpen ? VIEWER_TOOLBAR_POPOVER_LAYOUT : viewerTemplatesOpen ? VIEWER_TOOLBAR_POPOVER_TEMPLATES : "none", historyBatch: viewerHistoryBatch });
      if (action === "historyDelete") setHistoryDelete(null);
      else if (action === "contextMenu") setViewerMenu(null);
      else if (action === "layout" || action === "templates") closeViewerToolbarPopover();
      else if (action === "historyBatch") setViewerHistoryBatch(null);
      else closeImageViewer();
    };
    const closeMenu = (event) => {
      setViewerMenu(null);
      const target = event.target;
      if (viewerEdgePanelOpen && !viewerEdgePanelRef.current?.contains(target) && !viewerEdgeTriggerRef.current?.contains(target)) closeViewerToolbarPopover();
      if (viewerTemplatesOpen && !viewerTemplatePanelRef.current?.contains(target) && !viewerTemplateTriggerRef.current?.contains(target)) closeViewerToolbarPopover();
    };
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", handleEscape);
    window.addEventListener("pointerdown", closeMenu);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleEscape);
      window.removeEventListener("pointerdown", closeMenu);
    };
  }, [imageViewerOpen, historyDelete, viewerMenu, viewerEdgePanelOpen, viewerTemplatesOpen, viewerHistoryBatch]);

  useEffect(() => {
    if (!imageViewerOpen) return undefined;
    const panel = viewerEdgePanelOpen ? viewerEdgePanelRef.current : viewerTemplatesOpen ? viewerTemplatePanelRef.current : null;
    if (panel) {
      const frame = window.requestAnimationFrame(() => panel.querySelector("input, button, [tabindex]:not([tabindex='-1'])")?.focus());
      return () => window.cancelAnimationFrame(frame);
    }
    const trigger = viewerToolbarFocus.focusReturn === VIEWER_TOOLBAR_POPOVER_LAYOUT ? viewerEdgeTriggerRef.current : viewerToolbarFocus.focusReturn === VIEWER_TOOLBAR_POPOVER_TEMPLATES ? viewerTemplateTriggerRef.current : null;
    setViewerToolbarFocus({ focusTarget: null, focusReturn: null });
    trigger?.focus();
    return undefined;
  }, [imageViewerOpen, viewerEdgePanelOpen, viewerTemplatesOpen]);

  useEffect(() => {
    if (!imageViewerOpen || !viewerToolbarRef.current) return undefined;
    const updateHeight = () => setViewerToolbarHeight(Math.ceil(viewerToolbarRef.current?.getBoundingClientRect().height || 46));
    updateHeight();
    const observer = new ResizeObserver(updateHeight);
    observer.observe(viewerToolbarRef.current);
    return () => observer.disconnect();
  }, [imageViewerOpen]);

  useEffect(() => {
    if (!hardwareMonitorOpen) return undefined;
    const fetchHardware = async () => {
      try {
        const response = await fetch("/api/inference/hardware", { cache: "no-store" });
        if (!response.ok) return;
        const stats = await response.json();
        setHardwareStats(stats);
        setHardwareHistory((current) => ({
          gpu: [...current.gpu, stats.gpu_util ?? 0].slice(-30),
          vram: [...current.vram, (stats.vram_used_mb ?? 0) / 1024].slice(-30),
          cpu: [...current.cpu, stats.cpu_percent ?? 0].slice(-30),
          ram: [...current.ram, stats.ram_used_gb ?? 0].slice(-30),
        }));
      } catch {}
    };
    fetchHardware();
    const timer = window.setInterval(fetchHardware, 2000);
    return () => { window.clearInterval(timer); setHardwareStats(null); setHardwareHistory({ gpu: [], vram: [], cpu: [], ram: [] }); };
  }, [hardwareMonitorOpen]);

  useEffect(() => {
    if (!settingsOpen) return undefined;
    const handleEscape = (event) => event.key === "Escape" && !reconfiguring && setSettingsOpen(false);
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", handleEscape);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleEscape);
    };
  }, [settingsOpen, reconfiguring]);

  useEffect(() => {
    if (!settingsOpen || settingsTab !== "logs") return;
    void refreshDiagnosticLogs();
  }, [settingsOpen, settingsTab]);

  useEffect(() => {
    if (!settingsOpen || settingsTab !== "performance") return;
    void refreshPerformanceSettings();
  }, [settingsOpen, settingsTab]);

  useEffect(() => {
    if (!settingsOpen || settingsTab !== "plugins") return;
    void refreshPlugins();
  }, [settingsOpen, settingsTab]);

  useEffect(() => {
    if (!uiStateReady || inferenceHealth?.performance_settings?.memory_mode !== "ultra_low_vram") return;
    setProcessPreview(false);
    setHires((current) => current.enabled ? { ...current, enabled: false } : current);
    setADetailer((current) => current.enabled ? { ...current, enabled: false } : current);
    setRtx((current) => current.enabled ? { ...current, enabled: false } : current);
    commitMountedLoras(() => []);
  }, [uiStateReady, inferenceHealth?.performance_settings?.memory_mode]);

  useEffect(() => {
    const models = inferenceHealth?.upscalers?.models || [];
    const compatible = models.filter((item) => item.compatible);
    if (!compatible.length) return;
    setHires((current) => compatible.some((item) => item.id === current.model)
      ? current
      : { ...current, model: compatible[0].id });
  }, [inferenceHealth?.upscalers?.models]);

  useEffect(() => {
    const models = (inferenceHealth?.background_removal?.models || []).filter((item) => item.selectable !== false);
    if (!models.length) return;
    setBackgroundRemovalModel((current) => models.some((item) => item.id === current)
      ? current
      : (models.find((item) => item.installed)?.id || models[0].id));
  }, [inferenceHealth?.background_removal?.models]);

  useEffect(() => {
    if (!backgroundRemovalPickerOpen) return undefined;
    const close = (event) => {
      if (event.type === "keydown" && event.key !== "Escape") return;
      if (event.type === "pointerdown" && backgroundRemovalPickerRef.current?.contains(event.target)) return;
      setBackgroundRemovalPickerOpen(false);
    };
    window.addEventListener("pointerdown", close);
    window.addEventListener("keydown", close);
    return () => {
      window.removeEventListener("pointerdown", close);
      window.removeEventListener("keydown", close);
    };
  }, [backgroundRemovalPickerOpen]);

  useEffect(() => {
    const controller = new AbortController();
    let retryTimer;
    let attempts = 0;
    const checkHealth = async () => {
      try {
        const response = await fetch("/api/inference/health", { signal: controller.signal });
        if (!response.ok) throw new Error("推理服务未就绪");
        const health = await response.json();
        setInferenceHealth(health);
        attempts = health.status === "ready" ? 0 : attempts + 1;
        if (health.performance_settings?.memory_mode === "ultra_low_vram") {
          setProcessPreview(false);
          setHires((current) => current.enabled ? { ...current, enabled: false } : current);
          setADetailer((current) => current.enabled ? { ...current, enabled: false } : current);
          setRtx((current) => current.enabled ? { ...current, enabled: false } : current);
          commitMountedLoras(() => []);
        }
        retryTimer = window.setTimeout(checkHealth, health.status === "ready" ? 2000 : 1000);
      } catch (error) {
        if (error.name === "AbortError") return;
        attempts += 1;
        setInferenceHealth({ status: "offline", error: error.message });
        retryTimer = window.setTimeout(checkHealth, Math.min(5000, 1000 * attempts));
      }
    };
    checkHealth();
    return () => {
      controller.abort();
      window.clearTimeout(retryTimer);
    };
  }, []);

  useEffect(() => {
    let stopped = false;
    let timer;
    const poll = async () => {
      try {
        const response = await fetch("/api/background-removal/download", { cache: "no-store" });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "无法读取透明背景模型下载状态");
        if (stopped) return;
        if (payload.job) setBackgroundRemovalDownload(payload.job);
        if (payload.job?.active) timer = window.setTimeout(poll, 500);
      } catch {
        if (!stopped && backgroundRemovalDownload?.active) timer = window.setTimeout(poll, 2000);
      }
    };
    poll();
    return () => { stopped = true; window.clearTimeout(timer); };
  }, [backgroundRemovalDownload?.active]);

  useEffect(() => {
    if (activePage === "toolbox") return undefined;
    let stopped = false;
    let timer;
    const poll = async () => {
      try {
        const response = await fetch("/api/model-download/job", { cache: "no-store" });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "无法读取模型下载状态");
        if (stopped) return;
        const job = payload.job;
        if (job?.active) {
          timer = window.setTimeout(poll, 1500);
          return;
        }
        const completionKey = job?.jobId ? `${job.jobId}:${job.completedModels}:${job.failedModels}` : "";
        if (job?.jobId && job.completedModels > 0 && completedModelDownloadJob.current !== completionKey) {
          completedModelDownloadJob.current = completionKey;
          const targets = job.targets?.length ? job.targets : [{ kind: job.kind, engine: job.engine }];
          if (targets.some((target) => (target.kind === "checkpoint" && target.engine === model) || (SPLIT_MODEL_ENGINES.includes(model) && (target.engine === model || ["diffusion_model", "text_encoder", "vae", "config"].includes(target.kind))))) void refreshCheckpoints();
          if (targets.some((target) => target.kind === "lora" && target.engine === model)) void refreshLoras();
          if (targets.some((target) => target.kind === "yolo")) void refreshADetailerModels();
          if (targets.some((target) => target.kind === "upscaler")) void refreshUpscalers();
        }
      } catch {
        if (!stopped) timer = window.setTimeout(poll, 5000);
      }
    };
    void poll();
    return () => { stopped = true; window.clearTimeout(timer); };
  }, [activePage, model]);

  useEffect(() => {
    if (!activeJobRecoveryPending || inferenceHealth?.status !== "ready" || generationJob || status === "running") return undefined;
    const recoveryToken = ++activeJobRecoveryToken.current;
    let stopped = false;
    fetch("/api/inference/jobs/active", { cache: "no-store" })
      .then(async (response) => {
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "无法恢复生成任务");
        return payload.job;
      })
      .then((job) => {
        if (stopped || !job?.id) return;
        generationLocked.current = true;
        setGenerationJob(job.id);
        setStatus("running");
        setGenerationTaskStatus(job.status || "running");
        setGenerationPhase(job.phase || "正在恢复生成任务");
        setGenerationProgressCollapsed(false);
      })
      .catch(() => {})
      .finally(() => { if (activeJobRecoveryToken.current === recoveryToken) setActiveJobRecoveryPending(false); });
    return () => { stopped = true; };
  }, [activeJobRecoveryPending, inferenceHealth?.status, generationJob, status]);

  useEffect(() => {
    if (!generationJob || status !== "running") return undefined;
    let stopped = false;
    let failures = 0;
    const poll = async () => {
      try {
        const response = await fetch(`/api/inference/jobs/${generationJob}`, { cache: "no-store" });
        const job = await response.json();
        if (!response.ok) {
          const error = new Error(job.detail || "无法获取生成进度");
          error.terminal = response.status === 404;
          throw error;
        }
        if (stopped) return;
        failures = 0;
        setProgress(job.progress ?? 0);
        setGenerationTaskStatus(job.status || "running");
        setGenerationPhase(job.phase || "正在生成");
        setGenerationStage(job.stage || "");
        setGenerationStageStep(job.stage_step ?? 0);
        setGenerationStageTotal(job.stage_total ?? 0);
        setPreviewKind(job.preview_kind || "");
        setGenerationDetail(job.adetailer_state || null);
        setGenerationWarning(job.warning || "");
        if (job.prompt_tokens !== null && job.prompt_tokens !== undefined) {
          setPromptConditioning({
            tokens: job.prompt_tokens,
            blocks: job.prompt_blocks,
            weightedTokens: job.prompt_weighted_tokens,
            negativeTokens: job.negative_prompt_tokens,
            negativeBlocks: job.negative_prompt_blocks,
            negativeWeightedTokens: job.negative_prompt_weighted_tokens,
          });
        }
        setGenerationStep(job.step ?? 0);
        setGenerationTotal(job.total_steps ?? steps);
        setGenerationBatchIndex(job.batch_index ?? 0);
        setGenerationBatchCount(job.batch_count ?? batchCount);
        setGenerationCompletedImages(job.completed_images ?? 0);
        setGenerationTotalImages(job.total_images ?? imagesPerBatch * batchCount);
        setGenerationElapsed(job.elapsed_seconds ?? 0);
        setGenerationPausedTime(job.paused_seconds ?? 0);
        if (Object.prototype.hasOwnProperty.call(job, "model_cached")) {
          setInferenceHealth((current) => ({
            ...current,
            memory_mode: job.memory_mode,
            memory_label: job.memory_label,
            memory_reason: job.memory_reason,
            offload_mode: job.offload_mode,
            attention_backend: job.attention_backend,
            compute_dtype: job.compute_dtype,
            vae_mode: job.vae_mode,
            model_resident: job.model_resident,
            model_cached: job.model_cached,
            loaded_checkpoint: job.loaded_checkpoint,
            loaded_checkpoint_path: job.loaded_checkpoint_path,
            ...(Object.prototype.hasOwnProperty.call(job, "loaded_model_assets") ? { loaded_model_assets: job.loaded_model_assets } : {}),
            loaded_engine: job.loaded_engine,
          }));
        }
        if (job.preview_url && job.preview_version) {
          setLivePreview(`${job.preview_url}?v=${job.preview_version}`);
        }
        if (job.status === "complete") {
          generationLocked.current = false;
          const version = Date.now();
          const outputs = (job.outputs?.length ? job.outputs : [{
            index: 0,
            batch_index: 1,
            image_index: 1,
            seed: seed,
            output_name: job.output_name || `XirAI-${generationJob}.png`,
            image_url: job.image_url,
          }]).map((output) => ({ ...output, url: `${output.image_url}?v=${version}` }));
          setGeneratedOutputs(outputs);
          setSelectedOutputIndex(0);
          setGeneratedImage(outputs[0]?.url || "");
          setGeneratedName(outputs[0]?.output_name || `XirAI-${generationJob}.png`);
          setLivePreview("");
          setStatus("complete");
          setProgress(100);
          setGenerationPhase(`完成 · ${job.elapsed_seconds ?? 0}s`);
          setGenerationTaskStatus("complete");
          return;
        }
        if (job.status === "error") {
          generationLocked.current = false;
          setStatus("error");
          setGenerationError(job.error || "生成失败");
          setGenerationPhase("生成失败");
          setGenerationTaskStatus("error");
          return;
        }
        if (job.status === "cancelled") {
          generationLocked.current = false;
          setStatus("cancelled");
          setGenerationTaskStatus("cancelled");
          setGenerationPhase("生成已终止");
          setLivePreview("");
          return;
        }
        window.setTimeout(poll, 500);
      } catch (error) {
        if (stopped) return;
        failures += 1;
        if (error.terminal) {
          generationLocked.current = false;
          setStatus("error");
          setGenerationError("推理服务已重启，原生成任务无法恢复。请重新生成。");
          setGenerationPhase("生成任务已丢失");
          setGenerationTaskStatus("error");
          return;
        }
        setGenerationWarning(`推理服务连接中断，正在重试：${error.message}`);
        setGenerationPhase("正在重新连接推理服务");
        window.setTimeout(poll, Math.min(5000, 1000 * failures));
      }
    };
    poll();
    return () => { stopped = true; };
  }, [generationJob, status]);

  // Every run starts from the same cleared board, whichever page asked for it. Both pages share one
  // set of job state because the inference service runs one job at a time: two independent copies
  // would let a page display progress for a run it did not start.
  const beginGenerationRun = ({ totalSteps, batches, totalImages }) => {
    generationLocked.current = true;
    setProgress(0);
    setStatus("running");
    setGenerationPhase("正在提交任务");
    setGenerationError("");
    setGenerationWarning("");
    setPromptConditioning(null);
    setGeneratedImage("");
    setGeneratedName("");
    setGeneratedOutputs([]);
    setSelectedOutputIndex(0);
    setLivePreview("");
    setGenerationStage("queued");
    setGenerationStageStep(0);
    setGenerationStageTotal(0);
    setPreviewKind("");
    setGenerationDetail(null);
    setGenerationStep(0);
    setGenerationTotal(totalSteps);
    setGenerationBatchIndex(0);
    setGenerationBatchCount(batches);
    setGenerationCompletedImages(0);
    setGenerationTotalImages(totalImages);
    setGenerationTaskStatus("queued");
    setGenerationElapsed(0);
    setGenerationPausedTime(0);
    setGenerationControlBusy("");
    setGenerationProgressCollapsed(true);
    setGenerationJob("");
  };

  const generate = async () => {
    if (generationLocked.current || generationDisabledReason) return;
    const animaGeneration = model === "Anima";
    const fluxGeneration = model === "Flux";
    const flux2Generation = model === "Flux2";
    const krea2Generation = model === "Krea2";
    // Every native engine shares the ComfyUI sampler vocabulary, so one normalisation covers them.
    const nativeGeneration = animaGeneration || fluxGeneration || flux2Generation || krea2Generation;
    // Both Flux generations are guidance distilled: no negative branch, no enhancement.
    const distilledGeneration = DISTILLED_GUIDANCE_ENGINES.includes(model);
    beginGenerationRun({ totalSteps: steps, batches: batchCount, totalImages: imagesPerBatch * batchCount });
    const generationSeed = seedMode === "random" ? randomSeed() : normalizeSeed(seed);
    const generationLoras = frozenMountedLorasForScope(mountedLorasMapRef.current, activeLoraScopeRef.current);
    // Which combinations were on, and the exact prefix they contributed. Recorded
    // so a gallery card can show the prompt that actually produced the image; the
    // group *library* stays out of the card, only this run's facts go in.
    const generationGroups = enabledLoraGroups(loraGroupsForScope(loraGroupsMapRef.current, activeLoraScopeRef.current));
    const generationGroupPrompt = composeGroupPrompt(generationGroups, "");
    const generationHiresSeed = generationHiresSeedSettings(hires);
    const generationHires = { ...hires, ...generationHiresSeed };
    setSeed(generationSeed);
    setGeneratedSettings(JSON.parse(JSON.stringify({
      ...gallerySettingsWithoutPromptPresets(workspaceSnapshot.current),
      seed: generationSeed,
      sampler: nativeGeneration && !ANIMA_SAMPLERS.includes(sampler) ? "euler" : sampler,
      scheduler: nativeGeneration && !ANIMA_SCHEDULERS.includes(scheduler) ? "simple" : scheduler,
      guidance: distilledGeneration ? "none" : guidance,
      processPreview: nativeGeneration ? false : processPreview,
      hires: generationHires,
      adetailer,
      rtx,
      loras: generationLoras,
      loraGroups: generationGroups.map((group) => ({ id: group.id, name: group.name, presetPrompt: group.presetPrompt })),
      loraGroupPrompt: generationGroupPrompt,
    })));
    try {
      const response = await fetch("/api/inference/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          engine: model,
          ...(fluxGeneration
            ? { diffusion_model: diffusionModel, text_encoder: textEncoder, text_encoder_2: textEncoder2, vae }
            : nativeGeneration ? { diffusion_model: diffusionModel, text_encoder: textEncoder, vae } : { checkpoint }),
          // Enabled LoRA groups contribute their preset prompt ahead of what the
          // user typed. This is composed onto the request body only: the prompt
          // box is never rewritten, and `generatedSettings` keeps the user's own
          // text so restoring a gallery item cannot prepend the presets a second
          // time on the next run.
          prompt: composeGroupPrompt(generationGroups, positive),
          // Both Flux generations are guidance distilled and have no unconditional branch, so the
          // box keeps what the user typed for the other engines while the request carries nothing
          // to encode.
          negative_prompt: distilledGeneration ? "" : negative.trim(),
          width: size.width,
          height: size.height,
          steps,
          cfg,
          denoise,
          seed: generationSeed,
          images_per_batch: imagesPerBatch,
          batch_count: batchCount,
          sampler: nativeGeneration && !ANIMA_SAMPLERS.includes(sampler) ? "euler" : sampler,
          scheduler: nativeGeneration && !ANIMA_SCHEDULERS.includes(scheduler) ? "simple" : scheduler,
          guidance: distilledGeneration ? "none" : guidance,
          pag: { scale: pag.scale, applied_layers: pag.appliedLayers },
          preview_enabled: nativeGeneration ? false : processPreview,
          background_removal_model: transparentPromptEnabled ? backgroundRemovalModel : null,
          hires: {
            // Protocol 27 carries strict execution and lossless Hires seed contracts.
            enabled: hires.enabled,
            model: hires.model,
            ...hiresSeedPayload(generationHiresSeed),
            scale: hires.scale,
            denoise: hires.denoise,
            steps: hires.steps,
            cfg: hires.cfg,
            tile_size: hires.tileSize,
            tile_overlap: hires.tileOverlap,
            execution_mode: animaGeneration && hires.executionMode === "usdu_tiled" ? "usdu_tiled" : "full_frame",
            ...(hires.sampler ? { sampler: hires.sampler } : {}),
            ...(hires.scheduler ? { scheduler: hires.scheduler } : {}),
            tile_width: hires.tileWidth,
            tile_height: hires.tileHeight,
            padding: hires.padding,
            mask_blur: hires.maskBlur,
            seam_mode: "none",
            uniform_tiles: hires.uniformTiles !== false,
            tiled_decode: hires.tiledDecode !== false,
          },
          adetailer: adetailerPayload(adetailer, model),
          rtx: {
            enabled: rtx.enabled,
            scale: rtx.scale,
            quality: rtx.quality,
          },
          postprocess_order: normalizePostprocessOrder(postprocessOrder),
          loras: generationLoras.filter((lora) => lora.enabled !== false).map((lora) => ({ path: lora.value, weight: lora.weight })),
        }),
      });
      const job = await response.json();
      if (!response.ok) {
        const detail = Array.isArray(job.detail) ? job.detail.map((item) => item.msg || String(item)).join("；") : job.detail;
        throw new Error(detail || job.error || "任务提交失败");
      }
      setGenerationJob(job.id);
      setGenerationTaskStatus(job.status || "queued");
      setGenerationPhase(job.phase || "已进入队列");
      if (seedMode === "increment") {
        setSeed(((BigInt(generationSeed) + BigInt(imagesPerBatch * batchCount)) & MAX_SEED).toString());
      } else if (seedMode === "decrement") {
        const decrement = BigInt(imagesPerBatch * batchCount) % (MAX_SEED + 1n);
        setSeed(((BigInt(generationSeed) - decrement + MAX_SEED + 1n) & MAX_SEED).toString());
      }
    } catch (error) {
      generationLocked.current = false;
      setStatus("error");
      setGenerationError(error.message);
      setGenerationPhase("任务提交失败");
      logClientGenerationFailure("job-submission", error.message);
    }
  };

  const generateFromImage = async () => {
    if (generationLocked.current || status === "running" || !imageSource) return;
    const settings = normalizeImageToImageSettings(imageToImage, {
      samplers: activeSamplerNames,
      schedulers: activeSchedulerNames,
    });
    const runSeed = settings.seedMode === "random" ? randomSeed() : normalizeSourceSeed(settings.seed);
    const runHiresSeed = generationHiresSeedSettings(settings.hires);
    // The same LoRA mounts back this page, so the same combinations apply: a group contributes its
    // members *and* the trigger words that make them work. Composed onto the request body only —
    // the prompt box is never rewritten — exactly as on the text-to-image page.
    const runLoras = frozenMountedLorasForScope(mountedLorasMapRef.current, activeLoraScopeRef.current);
    const runGroups = enabledLoraGroups(loraGroupsForScope(loraGroupsMapRef.current, activeLoraScopeRef.current));
    beginGenerationRun({
      totalSteps: settings.steps,
      batches: settings.batchCount,
      totalImages: settings.imagesPerBatch * settings.batchCount,
    });
    setImageToImage({ ...settings, seed: runSeed });
    // What a gallery card records has to be what produced the image. Left alone, the dialog falls
    // back to the *workspace* snapshot — the text-to-image composer's prompt, steps and denoise —
    // and would attribute those to a picture they had nothing to do with. Restoring such a card
    // still lands on the text-to-image page without the source image, which is why the page it
    // came from is recorded alongside the parameters.
    setGeneratedSettings(JSON.parse(JSON.stringify({
      ...gallerySettingsWithoutPromptPresets(workspaceSnapshot.current),
      page: "image",
      positive: settings.positive,
      negative: settings.negative,
      steps: settings.steps,
      cfg: settings.cfg,
      denoise: settings.denoise,
      seed: runSeed,
      seedMode: settings.seedMode,
      sampler: settings.sampler,
      scheduler: settings.scheduler,
      guidance: "none",
      size: outputSize(imageSource, settings),
      imagesPerBatch: settings.imagesPerBatch,
      batchCount: settings.batchCount,
      processPreview: model !== "Anima",
      imageToImage: { ...settings, hires: { ...settings.hires, ...runHiresSeed } },
      hires: { ...settings.hires, ...runHiresSeed },
      adetailer: settings.adetailer,
      rtx: settings.rtx,
      postprocessOrder: settings.postprocessOrder,
      loraGroups: runGroups.map((group) => ({ id: group.id, name: group.name, presetPrompt: group.presetPrompt })),
      loraGroupPrompt: composeGroupPrompt(runGroups, ""),
    })));
    try {
      const response = await fetch("/api/inference/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(imageToImageRequestBody({
          engine: model,
          checkpoint,
          diffusionModel,
          textEncoder,
          textEncoder2,
          vae,
           source: imageSource,
           settings: { ...settings, positive: composeGroupPrompt(runGroups, settings.positive) },
           seed: runSeed,
           hiresSeed: runHiresSeed,
           samplers: activeSamplerNames,
           schedulers: activeSchedulerNames,
           loras: runLoras,
        })),
      });
      const job = await response.json();
      if (!response.ok) {
        const detail = Array.isArray(job.detail) ? job.detail.map((item) => item.msg || String(item)).join("；") : job.detail;
        throw new Error(detail || job.error || "任务提交失败");
      }
      setGenerationJob(job.id);
      setGenerationTaskStatus(job.status || "queued");
      setGenerationPhase(job.phase || "已进入队列");
      setImageToImage({ ...settings, seed: nextImageToImageSeed(settings, runSeed) });
    } catch (error) {
      generationLocked.current = false;
      setStatus("error");
      setGenerationError(error.message);
      setGenerationPhase("任务提交失败");
      logClientGenerationFailure("job-submission", error.message);
    }
  };

  const releaseLoadedModel = async () => {
    if (inferenceHealth?.status !== "ready") return true;
    // Nothing cached is nothing to release. The request is not free on the other side — it takes the
    // pipeline lock and runs a full collection — so asking for it anyway put a round trip in front of
    // every model change made while no model was loaded.
    if (!inferenceHealth?.model_cached) return true;
    try {
      const response = await fetch("/api/inference/model-cache", { method: "DELETE" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "底模缓存释放失败");
      if (payload.status !== "released" || payload.model_cached) throw new Error("底模缓存释放状态异常，请重试切换");
    } catch (error) {
      setGenerationWarning(error.message);
      return false;
    }
    setInferenceHealth((current) => current ? {
      ...current,
      loaded_checkpoint: null,
      loaded_checkpoint_path: null,
      loaded_model_assets: null,
      loaded_engine: null,
      memory_mode: null,
      memory_label: null,
      memory_reason: null,
      offload_mode: null,
      attention_backend: "none",
      model_resident: false,
      model_cached: false,
    } : current);
    return true;
  };

  const unloadLoadedModel = async () => {
    if (status === "running" || modelSwitching || !inferenceHealth?.model_cached) return;
    setModelSwitching(true);
    if (await releaseLoadedModel()) setGenerationWarning("");
    setModelSwitching(false);
  };

  const applyGallerySettings = async (savedSettings, groups, { page = "generate", label = "精选参数" } = {}) => {
    if (status === "running" || modelSwitching || modelLoading) throw new Error(`生成或模型切换期间不能应用${label}`);
    const selectedGroups = new Set(groups);
    const current = workspaceSnapshot.current;
    const source = savedSettings && typeof savedSettings === "object" ? { ...savedSettings } : {};
    delete source.promptPresets;
    delete source.mountedLorasByEngine;
    delete source.loraGroupsByEngine;
    const sourceModel = READY_ENGINES.includes(source.model) ? source.model : current.model;
    const targetModel = selectedGroups.has("model") ? sourceModel : current.model;
    let targetHealthPayload = targetModel === model ? inferenceHealth : null;
    const sourceModelFields = selectedGroups.has("model") ? {
      checkpoint: typeof source.checkpoint === "string" ? source.checkpoint : "",
      diffusionModel: typeof source.diffusionModel === "string" ? source.diffusionModel : "",
      textEncoder: typeof source.textEncoder === "string" ? source.textEncoder : "",
      textEncoder2: typeof source.textEncoder2 === "string" ? source.textEncoder2 : "",
      vae: typeof source.vae === "string" ? source.vae : "",
    } : {};
    const normalized = loadWorkspaceState({
      ...current,
      ...source,
      ...sourceModelFields,
      model: targetModel,
      size: { ...current.size, ...source.size },
      hires: { ...current.hires, ...source.hires },
      adetailer: { ...current.adetailer, ...source.adetailer },
      rtx: { ...current.rtx, ...source.rtx },
      loras: Array.isArray(source.loras) ? source.loras : current.loras,
    });
    if (!normalized) throw new Error("精选卡片参数格式无效");
    if (selectedGroups.has("loras") && !selectedGroups.has("model") && sourceModel !== model) {
      throw new Error("该 LoRA 组合属于其他模型引擎，请同时选择“底模”参数");
    }
    const engineChanged = selectedGroups.has("model") && normalized.model !== model;
    const checkpointChanged = selectedGroups.has("model") && normalized.checkpoint !== checkpoint;
    const splitAssetsChanged = selectedGroups.has("model") && (
      normalized.diffusionModel !== diffusionModel
      || normalized.textEncoder !== textEncoder
      || normalized.textEncoder2 !== textEncoder2
      || normalized.vae !== vae
    );
    if (selectedGroups.has("model")) {
      const [modelsResponse, healthResponse] = await Promise.all([
        fetch(`/api/models?engine=${encodeURIComponent(normalized.model)}`, { cache: "no-store" }),
        SPLIT_MODEL_ENGINES.includes(normalized.model) ? fetch("/api/inference/health", { cache: "no-store" }) : Promise.resolve(null),
      ]);
      const modelsPayload = await modelsResponse.json().catch(() => ({}));
      if (!modelsResponse.ok) throw new Error(modelsPayload.error || "无法验证精选卡片的模型目录");
      if (SPLIT_MODEL_ENGINES.includes(normalized.model)) {
        const required = [
          ["diffusionModel", "扩散模型", modelsPayload.assets?.diffusion_model?.models || []],
          ["textEncoder", normalized.model === "Flux" ? "CLIP-L 文本编码器" : "文本编码器", modelsPayload.assets?.text_encoder?.models || []],
          ...(normalized.model === "Flux" ? [["textEncoder2", "T5-XXL 文本编码器", modelsPayload.assets?.text_encoder_2?.models || []]] : []),
          ["vae", "VAE", modelsPayload.assets?.vae?.models || []],
        ];
        for (const [field, label, catalog] of required) {
          if (!normalized[field] || !catalog.some((item) => item.value === normalized[field])) {
            throw new Error(`精选卡片的 ${normalized.model} ${label}未安装或已移动，当前模型缓存保持不变`);
          }
        }
        const healthPayload = await healthResponse.json().catch(() => ({}));
        const engineHealth = healthPayload.engines?.[normalized.model];
        if (!healthResponse.ok || !engineHealth?.available) {
          throw new Error(engineHealth?.reason || `${normalized.model} 原生运行时尚未就绪，当前模型缓存保持不变`);
        }
        targetHealthPayload = healthPayload;
      } else if (!normalized.checkpoint || !modelsPayload.models?.some((item) => item.value === normalized.checkpoint)) {
        throw new Error("精选卡片的底模未安装或已移动，当前模型缓存保持不变");
      }
    }
    const targetLoraIdentity = selectedGroups.has("model")
      ? { model: normalized.model, checkpoint: normalized.checkpoint, diffusionModel: normalized.diffusionModel, textEncoder: normalized.textEncoder, textEncoder2: normalized.textEncoder2, vae: normalized.vae }
      : { model: current.model, checkpoint: current.checkpoint, diffusionModel: current.diffusionModel, textEncoder: current.textEncoder, textEncoder2: current.textEncoder2, vae: current.vae };
    const targetLoraScopeKey = engineScopeKey(targetLoraIdentity.model);
    if (selectedGroups.has("loras") && !targetLoraScopeKey) {
      throw new Error("LoRA 只能应用到已完整选择的模型组件，当前挂载库保持不变");
    }
    const galleryTarget = galleryMountedLorasForTarget(mountedLorasMapRef.current, {
      targetEngine: targetLoraScopeKey,
      sourceLoras: source.loras,
      applyLoras: selectedGroups.has("loras"),
    });
    const galleryTargetLoras = galleryTarget.loras;
    if (SPLIT_MODEL_ENGINES.includes(targetModel)) {
      const features = targetHealthPayload?.engines?.[targetModel]?.features || {};
      const effectiveLoras = selectedGroups.has("loras") ? galleryTargetLoras : (engineChanged || checkpointChanged || splitAssetsChanged) ? galleryTargetLoras : current.loras;
      const effectiveHires = selectedGroups.has("hires") ? normalized.hires : current.hires;
      const effectiveADetailer = selectedGroups.has("adetailer") ? normalized.adetailer : current.adetailer;
      const effectiveRtx = selectedGroups.has("rtx") ? normalized.rtx : current.rtx;
      const requestedFeatures = [
        [(engineChanged || selectedGroups.has("loras")) && effectiveLoras.some((item) => item.enabled !== false), "lora", "LoRA"],
        [(engineChanged || selectedGroups.has("hires")) && effectiveHires.enabled, "hires", "Hires.fix"],
        [(engineChanged || selectedGroups.has("adetailer")) && effectiveADetailer.enabled, "adetailer", "ADetailer"],
        [(engineChanged || selectedGroups.has("rtx")) && effectiveRtx.enabled, "rtx", "RTX VSR"],
      ];
      const unsupported = requestedFeatures.find(([enabled, feature]) => enabled && features[feature] !== true);
      if (unsupported) throw new Error(`当前推理服务未声明 ${targetModel} ${unsupported[2]} 能力，精选参数尚未应用`);
      const enabledLoras = effectiveLoras.filter((item) => item.enabled !== false);
      if ((engineChanged || selectedGroups.has("loras")) && enabledLoras.length) {
        const loraResponse = await fetch(`/api/loras?engine=${encodeURIComponent(targetModel)}`, { cache: "no-store" });
        const loraPayload = await loraResponse.json().catch(() => ({}));
        if (!loraResponse.ok) throw new Error(loraPayload.error || `无法验证精选卡片的 ${targetModel} LoRA 目录`);
        const available = new Set((loraPayload.categories || []).flatMap((category) => category.models || []).map((item) => item.value));
        if (enabledLoras.some((item) => !available.has(item.value))) throw new Error(`精选卡片的 ${targetModel} LoRA 文件未安装或已移动`);
      }
      if ((engineChanged || selectedGroups.has("hires")) && effectiveHires.enabled) {
        const upscalers = targetHealthPayload?.upscalers || {};
        const selected = (upscalers.models || []).find((item) => item.id === effectiveHires.model);
        if (!upscalers.runtime_available || !selected?.compatible) throw new Error("精选卡片的 Hires.fix 超分模型尚未就绪");
        if (Math.floor(effectiveHires.steps * effectiveHires.denoise) < 1 || effectiveHires.tileOverlap > Math.floor(effectiveHires.tileSize / 2)) {
          throw new Error("精选卡片的 Hires.fix 参数无效");
        }
      }
      if ((engineChanged || selectedGroups.has("adetailer")) && effectiveADetailer.enabled) {
        const detectorReady = adetailerInfo.available && adetailerModels.some((item) => item.value === effectiveADetailer.detector);
        const baseSteps = selectedGroups.has("sampling") ? normalized.steps : current.steps;
        const effectiveSteps = Math.floor((effectiveADetailer.useSteps ? effectiveADetailer.steps : baseSteps) * effectiveADetailer.denoise);
        if (!detectorReady || effectiveSteps < 1) throw new Error("精选卡片的 ADetailer 检测模型或重绘参数尚未就绪");
      }
      if ((engineChanged || selectedGroups.has("rtx")) && effectiveRtx.enabled && targetHealthPayload?.rtx_vsr?.available !== true) {
        throw new Error(targetHealthPayload?.rtx_vsr?.reason || "精选卡片需要的 RTX VSR 运行时尚未就绪");
      }
    }
    const modelIdentityChanged = engineChanged || checkpointChanged || splitAssetsChanged;
    if (modelIdentityChanged) {
      setModelSwitching(true);
      if (!(await releaseLoadedModel())) {
        setModelSwitching(false);
        throw new Error("当前底模缓存未能安全释放，参数尚未应用");
      }
    }

    if (selectedGroups.has("prompts")) {
      setPositive(normalized.positive);
      setNegative(normalized.negative);
      setADetailer((currentValue) => ({
        ...currentValue,
        units: currentValue.units.map((unit, index) => {
          const source = normalized.adetailer.units[index];
          return source ? { ...unit, prompt: source.prompt, negativePrompt: source.negativePrompt } : unit;
        }),
      }));
    }
    if (selectedGroups.has("sampling")) {
      setSteps(normalized.steps);
      setCfg(normalized.cfg);
      setDenoise(normalized.denoise);
      setImagesPerBatch(normalized.imagesPerBatch);
      setBatchCount(normalized.batchCount);
      setSeed(normalized.seed);
      setSeedMode(normalized.seedMode);
      setSampler(normalized.sampler);
      setScheduler(normalized.scheduler);
      setGuidance(normalized.guidance);
      setPag(normalized.pag);
    }
    if (selectedGroups.has("canvas")) setSize(normalized.size);
    if (selectedGroups.has("hires")) setHires({ ...HIRES_DEFAULTS, ...normalized.hires });
    if (selectedGroups.has("adetailer")) setADetailer((currentValue) => normalizeADetailerStage({
      ...normalized.adetailer,
      // Prompts are their own group: taking the card's units without its prompts
      // keeps whatever the user has typed here, unit by unit.
      units: normalized.adetailer.units.map((unit, index) => (selectedGroups.has("prompts") ? unit : {
        ...unit,
        prompt: currentValue.units[index]?.prompt ?? "",
        negativePrompt: currentValue.units[index]?.negativePrompt ?? "",
      })),
    }));
    if (selectedGroups.has("rtx")) {
      setRtx({ ...RTX_DEFAULTS, ...normalized.rtx });
      setPostprocessOrder(normalized.postprocessOrder);
    }
    if (selectedGroups.has("auxiliary")) {
      setProcessPreview(normalized.processPreview);
      setBackgroundRemovalModel(normalized.backgroundRemovalModel);
    }

    if (engineChanged) {
      const nextLoras = selectedGroups.has("loras") ? galleryTargetLoras : mountedLorasForScope(mountedLorasMapRef.current, targetLoraScopeKey);
      const restored = { ...normalized, loras: nextLoras };
      restorePending.current = { checkpoint: true, diffusionModel: true, textEncoder: true, textEncoder2: true, vae: true, loras: true };
      setRestoredWorkspace(restored);
      setCheckpoint(normalized.checkpoint);
      setDiffusionModel(normalized.diffusionModel);
      setTextEncoder(normalized.textEncoder);
      setTextEncoder2(normalized.textEncoder2);
      setVae(normalized.vae);
       transitionActiveLoraScope(normalized.model, {
        ...(selectedGroups.has("loras") ? { targetLoras: galleryTargetLoras } : {}),
      });
      if (normalized.model === "Anima") {
        if (!selectedGroups.has("sampling")) {
          setGuidance((currentGuidance) => currentGuidance === "pag" && !pagAvailableForEngine(targetHealthPayload, normalized.model) ? "none" : currentGuidance);
        }
        setProcessPreview(false);
      } else if (normalized.model === "Flux") {
        setGuidance("none");
        setProcessPreview(false);
      } else if (!selectedGroups.has("sampling")) {
        setGuidance((currentGuidance) => currentGuidance === "cfg_zero_star" ? "none" : currentGuidance);
      }
      setModel(normalized.model);
    } else {
      if (selectedGroups.has("model")) {
        setCheckpoint(normalized.checkpoint);
        setCheckpointMissing(Boolean(normalized.checkpoint) && !checkpoints.some((item) => item.value === normalized.checkpoint));
        setDiffusionModel(normalized.diffusionModel);
        setDiffusionModelMissing(Boolean(normalized.diffusionModel) && !diffusionModels.some((item) => item.value === normalized.diffusionModel));
        setTextEncoder(normalized.textEncoder);
        setTextEncoderMissing(Boolean(normalized.textEncoder) && !textEncoders.some((item) => item.value === normalized.textEncoder));
        setTextEncoder2(normalized.textEncoder2);
        setTextEncoder2Missing(Boolean(normalized.textEncoder2) && !textEncoders2.some((item) => item.value === normalized.textEncoder2));
        setVae(normalized.vae);
        setVaeMissing(Boolean(normalized.vae) && !vaes.some((item) => item.value === normalized.vae));
        transitionActiveLoraScope(normalized.model, {
          ...(selectedGroups.has("loras") ? { targetLoras: galleryTargetLoras } : {}),
        });
      }
      if (selectedGroups.has("loras") && !selectedGroups.has("model")) {
        commitMountedLoras(() => galleryTargetLoras, targetLoraScopeKey);
      }
    }
    if (selectedGroups.has("loras")) {
      // The applied list replaces the mounted one outright, so a combination
      // that was switched on is no longer represented in it. Left on, it would
      // keep prefixing its preset prompt onto every request while none of its
      // LoRAs were loaded, and the next mounted edit would sync it back to zero
      // members and empty the user's own definition.
      commitLoraGroups((current) => disableUnmountedGroups(current, galleryTargetLoras).groups, targetLoraScopeKey);
    }
    // A caller that has its own result surface — the image reader, which has to
    // keep its missing-model marks on screen — passes `page: null` to stay put.
    if (page) {
      setActivePage(page);
      setGalleryFocus(null);
    }
    setAppNotice({ message: `已应用 ${selectedGroups.size} 组${label}`, error: false });
    if (modelIdentityChanged) setModelSwitching(false);
  };

  /**
   * Write a picture's recorded settings into a generate page.
   *
   * Prompt, sampling and canvas belong to whichever page was chosen; the model
   * and the LoRA mount are one shared state, so those rows land in the same
   * place whichever target is picked — which is why the panel labels them.
   * Everything that touches the model goes through the gallery apply path
   * rather than a second copy of it, so engine switching, catalogue validation
   * and the missing-checkpoint mark stay in one implementation.
   */
  const applyImageInfoParameters = async (plan) => {
    const shared = new Set(["model", "loras"]);
    const workspaceGroups = plan.target === "i2i" ? plan.groups.filter((id) => shared.has(id)) : [...plan.groups];
    if (plan.target === "i2i" && Object.keys(plan.imageToImage).length) {
      setImageToImage((current) => normalizeImageToImageSettings({ ...current, ...plan.imageToImage }));
    }
    if (!workspaceGroups.length) return;
    const snapshot = gallerySettingsWithoutPromptPresets(workspaceSnapshot.current);
    const overlay = { ...plan.overlay };
    const targetEngine = overlay.model || snapshot.model;
    if (Array.isArray(overlay.loras)) {
      // A LoRA found under another engine's root cannot be mounted here: the
      // catalogue scan would prune it moments later. Dropping the whole group
      // rather than mounting an empty list matters — an empty list is an
      // instruction to unmount everything.
      overlay.loras = overlay.loras.filter((item) => !item.engine || item.engine === targetEngine);
      if (!overlay.loras.length) {
        delete overlay.loras;
        const index = workspaceGroups.indexOf("loras");
        if (index >= 0) workspaceGroups.splice(index, 1);
      }
    }
    if (!workspaceGroups.length) return;
    await applyGallerySettings({ ...snapshot, ...overlay }, workspaceGroups, { page: null, label: "图片参数" });
  };

  // Choosing a different engine, checkpoint or component is a change of *selection*. It used to tear
  // the loaded pipeline down first and wait for it, which put a multi-gigabyte teardown — and, on the
  // next run, a full reload from disk — in front of a click that only changes what the pickers say.
  // Nothing needs it: `load_pipeline` and `load_anima_pipeline` both call `clear_pipeline()` when the
  // request does not match what is loaded, so the old weights are gone before the new ones are read,
  // and a selection changed and changed back reuses the cache instead of re-reading it. Freeing on
  // demand is what 卸载模型 is for, and the health panel keeps saying what is actually resident.
  const selectModel = (nextModel) => {
    if (nextModel === model || status === "running" || modelSwitching) return;
    // Engine changes restore that engine's independent list; asset changes
    // below deliberately leave it untouched.
    transitionActiveLoraScope(nextModel);
    setCheckpoint("");
    setDiffusionModel("");
    setTextEncoder("");
    setTextEncoder2("");
    setVae("");
    if (nextModel === "Anima") {
      setSampler((current) => ANIMA_SAMPLERS.includes(current) ? current : "euler");
      setScheduler((current) => ANIMA_SCHEDULERS.includes(current) ? current : "simple");
      setGuidance((currentGuidance) => currentGuidance === "pag" && !pagAvailableForEngine(inferenceHealthRef.current, nextModel) ? "none" : currentGuidance);
      setProcessPreview(false);
    } else if (nextModel === "Flux") {
      setSampler((current) => FLUX_SAMPLERS.includes(current) ? current : "euler");
      setScheduler((current) => FLUX_SCHEDULERS.includes(current) ? current : "simple");
      // Guidance distillation removes the unconditional branch, so neither enhancement has
      // anything to work against and the process preview has no latent stream to draw from.
      setGuidance("none");
      setProcessPreview(false);
    } else if (nextModel === "Flux2") {
      setSampler((current) => FLUX2_SAMPLERS.includes(current) ? current : "euler");
      setScheduler((current) => FLUX2_SCHEDULERS.includes(current) ? current : "simple");
      setGuidance("none");
      setProcessPreview(false);
    } else if (nextModel === "Krea2") {
      setSampler((current) => KREA2_SAMPLERS.includes(current) ? current : "euler");
      setScheduler((current) => KREA2_SCHEDULERS.includes(current) ? current : "simple");
      // Krea 2 keeps its unconditional branch, so CFG-Zero* survives the switch. PAG does not:
      // this runtime installs no attention override for the single-stream blocks.
      setGuidance((currentGuidance) => currentGuidance === "pag" ? "none" : currentGuidance);
      setProcessPreview(false);
    } else {
      setSampler((current) => samplerNames.includes(current) ? current : "dpmpp_2m");
      setScheduler((current) => schedulerNames.includes(current) ? current : "karras");
      setGuidance((currentGuidance) => currentGuidance === "cfg_zero_star" ? "none" : currentGuidance);
    }
    setModel(nextModel);
  };

  const selectCheckpoint = (nextCheckpoint) => {
    if (nextCheckpoint === checkpoint || status === "running" || modelSwitching) return;
    setCheckpoint(nextCheckpoint);
    setCheckpointMissing(false);
  };

  const selectSplitModelAsset = (kind, nextValue) => {
    const currentValue = { diffusionModel, textEncoder, textEncoder2, vae }[kind];
    if (nextValue === currentValue || status === "running" || modelSwitching) return;
    if (kind === "diffusionModel") { setDiffusionModel(nextValue); setDiffusionModelMissing(false); }
    if (kind === "textEncoder") { setTextEncoder(nextValue); setTextEncoderMissing(false); }
    if (kind === "textEncoder2") { setTextEncoder2(nextValue); setTextEncoder2Missing(false); }
    if (kind === "vae") { setVae(nextValue); setVaeMissing(false); }
  };

  const refreshCheckpoints = async () => {
    if (modelsRefreshInFlight.current || modelsRefreshing || status === "running" || modelSwitching) return;
    const engine = model;
    modelsRefreshInFlight.current = true;
    setModelsRefreshing(true);
    try {
      const response = await fetch(`/api/models?engine=${encodeURIComponent(engine)}`, { cache: "no-store" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "无法读取模型目录");
      if (engine !== currentModel.current) return;
      if (payload.model_type === "split") {
        const refreshAsset = (asset, currentValue, setCatalog, setDirectory, setValue, setMissing) => {
          const nextModels = asset?.models || [];
          setCatalog((current) => {
            const result = reconcileModels(current, nextModels);
            return result.changed ? result.models : current;
          });
          setDirectory(asset?.directory || "");
          if (!currentValue && nextModels.length) {
            setValue(nextModels[0].value);
            setMissing(false);
          } else setMissing(Boolean(currentValue) && !nextModels.some((item) => item.value === currentValue));
        };
        refreshAsset(payload.assets?.diffusion_model, currentDiffusionModel.current, setDiffusionModels, setDiffusionModelDirectory, setDiffusionModel, setDiffusionModelMissing);
        refreshAsset(payload.assets?.text_encoder, currentTextEncoder.current, setTextEncoders, setTextEncoderDirectory, setTextEncoder, setTextEncoderMissing);
        refreshAsset(payload.assets?.text_encoder_2, currentTextEncoder2.current, setTextEncoders2, () => {}, setTextEncoder2, setTextEncoder2Missing);
        refreshAsset(payload.assets?.vae, currentVae.current, setVaes, setVaeDirectory, setVae, setVaeMissing);
        setAnimaRuntime(payload.runtime || null);
        setCheckpoints([]);
      } else {
        const nextModels = payload.models || [];
        setDiffusionModels([]);
        setTextEncoders([]);
        setTextEncoders2([]);
        setVaes([]);
        setDiffusionModelDirectory("");
        setTextEncoderDirectory("");
        setVaeDirectory("");
        setAnimaRuntime(null);
        setCheckpoints((current) => {
          const result = reconcileModels(current, nextModels);
          return result.changed ? result.models : current;
        });
        setCheckpointDirectory(payload.directory || "");
        const selected = currentCheckpoint.current;
        if (!selected && nextModels.length > 0) {
          setCheckpoint(nextModels[0].value);
          setCheckpointMissing(false);
        } else setCheckpointMissing(Boolean(selected) && !nextModels.some((item) => item.value === selected));
      }
      setModelError("");
    } catch (error) {
      if (engine === currentModel.current) setModelError(error.message);
    } finally {
      modelsRefreshInFlight.current = false;
      setModelsRefreshing(false);
    }
  };

  const refreshLoras = async () => {
    if (lorasRefreshing || !canStartMountedLoraScan(loraScanGate.current)) return;
    const engine = model;
    const scopeKey = activeLoraScopeRef.current;
    const requestToken = ++loraScanToken.current;
    const scanRevision = mountedLoraRevisionRef.current;
    setLorasRefreshing(true);
    try {
      const response = await fetch(`/api/loras?engine=${encodeURIComponent(engine)}`, { cache: "no-store" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "无法读取 LoRA 目录");
      if (engine !== currentModel.current) return;
      const scan = reconcileMountedLoraScan({
        getCurrentContainer: () => mountedLorasMapRef.current,
        categories: payload.categories,
        scanScopeKey: scopeKey,
        requestToken,
        latestToken: loraScanToken.current,
        capturedRevision: scanRevision,
        latestRevision: mountedLoraRevisionRef.current,
        responseEngine: engine,
        activeEngine: activeLoraScopeRef.current,
        ...loraScanGate.current,
      });
      if (!scan.applied) return;
      const knownValues = new Set(loraLibrary.flatMap((category) => category.models.map((item) => item.value)));
      const scannedValues = new Set(payload.categories.flatMap((category) => category.models.map((item) => item.value)));
      setLoraLibrary((current) => {
        const result = reconcileLoraLibrary(current, payload.categories);
        return result.changed ? result.categories : current;
      });
      if (loraCategory !== "mounted" && !payload.categories.some((category) => category.id === loraCategory)) {
        setLoraCategory(payload.categories.find((category) => category.models.length)?.id || "character");
      }
      if (loraDirectory !== payload.directory) setLoraDirectory(payload.directory);
      if ([...knownValues].some((value) => !scannedValues.has(value))) {
        setLoraLookups((current) => Object.fromEntries(Object.entries(current).filter(([value]) => scannedValues.has(value))));
      }
      if (scan.changed) {
        lorasRef.current = scan.loras;
        commitMountedLoraMap(scan.container, { rescan: false });
        setLoras(scan.loras);
      }
      setLoraError("");
    } catch (error) {
      if (engine === currentModel.current && shouldApplyMountedLoraScan({ requestToken, latestToken: loraScanToken.current, capturedRevision: scanRevision, latestRevision: mountedLoraRevisionRef.current, responseEngine: engine, activeEngine: activeLoraScopeRef.current, ...loraScanGate.current })) setLoraError(error.message);
    } finally {
      if (shouldApplyMountedLoraScan({ requestToken, latestToken: loraScanToken.current, capturedRevision: scanRevision, latestRevision: mountedLoraRevisionRef.current, responseEngine: engine, activeEngine: activeLoraScopeRef.current, ...loraScanGate.current })) {
        setLorasRefreshing(false);
        setLoraLoading(false);
      }
    }
  };

  const openLoraManagerPage = () => {
    persistUiState();
    window.open("/lora", "_blank", "noopener,noreferrer");
    setLoraManagerOpen(false);
  };

  const enterConfigurator = async () => {
    if (reconfiguring || status === "running") return;
    persistUiState();
    setSettingsError("");
    setReconfiguring(true);
    try {
      const response = await fetch("/api/system/reconfigure", { method: "POST" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "无法启动环境配置器");
      for (let attempt = 0; attempt < 120; attempt += 1) {
        await new Promise((resolve) => setTimeout(resolve, 700));
        try {
          const stateResponse = await fetch("/api/state", { cache: "no-store" });
          if (!stateResponse.ok) continue;
          const setupState = await stateResponse.json();
          if (setupState.returnToApp) {
            window.location.replace("/config");
            return;
          }
        } catch {}
      }
      throw new Error("环境配置器启动超时。请重新启动主程序后再试。");
    } catch (error) {
      setSettingsError(error.message);
      setReconfiguring(false);
    }
  };
  const enterManualUpdater = () => {
    if (reconfiguring || status === "running") return;
    persistUiState(true);
    window.location.assign("/update");
  };

  const checkForOnlineUpdate = async () => {
    if (onlineUpdate.checking || reconfiguring || status === "running") return;
    setOnlineUpdate({ checking: true, error: "", release: null, checked: false });
    try {
      const response = await fetch("/api/system/update/check", { method: "POST" });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.error || "检查更新失败");
      const release = payload.online_release;
      setOnlineUpdate({ checking: false, error: "", release, checked: true });
      // Only a release that is both newer and installable becomes a prompt; anything else is
      // answered in place so the user is never asked to confirm a no-op.
      if (release?.update_available) setUpdateConfirmOpen(true);
    } catch (error) {
      setOnlineUpdate({ checking: false, error: error.message, release: null, checked: true });
    }
  };

  const startOnlineUpdate = async () => {
    if (onlineUpdate.checking) return;
    setOnlineUpdate((current) => ({ ...current, checking: true, error: "" }));
    try {
      const response = await fetch("/api/system/update/download", { method: "POST" });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.error || "无法开始下载更新包");
      setUpdateConfirmOpen(false);
      // The download runs on the server; the updater page is where its progress, the apply step
      // and the restart already live, so the flow joins the offline one from here on.
      persistUiState(true);
      window.location.assign("/update");
    } catch (error) {
      setOnlineUpdate((current) => ({ ...current, checking: false, error: error.message }));
    }
  };

  const controlGeneration = async (action) => {
    if (!generationJob || generationControlBusy) return;
    setGenerationControlBusy(action);
    try {
      const response = await fetch(`/api/inference/jobs/${generationJob}/${action}`, { method: "POST" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "任务控制失败");
      setGenerationTaskStatus(payload.status);
      if (action === "pause") setGenerationPhase("当前步骤结束后暂停");
      if (action === "resume") setGenerationPhase("正在继续生成");
      if (action === "cancel") setGenerationPhase("当前步骤结束后终止");
    } catch (error) {
      setGenerationWarning(error.message);
    } finally {
      setGenerationControlBusy("");
    }
  };

  const fitViewerToCanvas = () => {
    const rect = viewerCanvasRef.current?.getBoundingClientRect();
    if (!rect || !viewerLayers.length) return;
    const zoom = fitViewerZoom(rect.width, rect.height, viewerLayers);
    setViewerZoom(zoom);
    setViewerPan({ x: 0, y: 0 });
  };

  const closeImageViewer = () => {
    viewerFitRaf.current?.cancel();
    viewerFitRaf.current = null;
    viewerSession.current.close();
    setImageViewerOpen(false);
    setViewerMenu(null);
    setViewerEdgePanelOpen(false);
    setViewerTemplatesOpen(false);
    setViewerToolbarFocus({ focusTarget: null, focusReturn: null });
    setViewerHistoryBatch(null);
  };

  const applyViewerToolbarPopoverTransition = (target, reason) => {
    const current = viewerEdgePanelOpen ? VIEWER_TOOLBAR_POPOVER_LAYOUT : viewerTemplatesOpen ? VIEWER_TOOLBAR_POPOVER_TEMPLATES : "none";
    const transition = viewerToolbarPopoverTransition(current, target, reason);
    setViewerEdgePanelOpen(transition.next === VIEWER_TOOLBAR_POPOVER_LAYOUT);
    setViewerTemplatesOpen(transition.next === VIEWER_TOOLBAR_POPOVER_TEMPLATES);
    setViewerToolbarFocus({ focusTarget: transition.focusTarget, focusReturn: transition.focusReturn });
  };

  const closeViewerToolbarPopover = (reason = "close") => {
    applyViewerToolbarPopoverTransition("none", reason);
  };

  const toggleViewerToolbarPopover = (action) => {
    const current = viewerEdgePanelOpen ? VIEWER_TOOLBAR_POPOVER_LAYOUT : viewerTemplatesOpen ? VIEWER_TOOLBAR_POPOVER_TEMPLATES : "none";
    applyViewerToolbarPopoverTransition(current === action ? "none" : action, "toggle");
    setViewerMenu(null);
  };

  const openViewerContextMenu = (menu) => {
    applyViewerToolbarPopoverTransition("none", "context-menu");
    setViewerMenu(menu);
  };

  const openImageViewer = async () => {
    viewerFitRaf.current?.cancel();
    viewerFitRaf.current = null;
    const session = viewerSession.current.beginSession();
    const token = viewerSession.current.request("open", { session, latest: true });
    setViewerZoom(1);
    setViewerLayerResizeEnabled(true);
    setViewerPan({ x: 0, y: 0 });
    viewerUndo.current = [];
    setViewerLayers([]);
    setActiveViewerLayer("");
    setViewerHistoryBatch(null);
    setViewerEdgePanelOpen(false);
    setViewerTemplatesOpen(false);
    setViewerToolbarFocus({ focusTarget: null, focusReturn: null });
    setViewerSidebarOpen(false);
    setActiveCollage(null);
    setCollageResult(null);
    setViewerMenu(null);
    setImageViewerOpen(true);
    const plan = viewerOpenPlan(generatedImage);
    if (plan.empty) return;
    const initialLayer = await imageAssetFromSource({
      id: `generated-${selectedOutput?.asset_id || selectedOutputIndex}`,
      assetId: selectedOutput?.asset_id || "",
      url: generatedImage,
      name: generatedName || "XirAI.png",
    }).then((asset) => ({ ...asset, x: 0, y: 0, scale: 1 })).catch((error) => {
      if (viewerSession.current.isCurrent(token)) setViewerNotice(error.message);
      return null;
    });
    if (!viewerSession.current.isCurrent(token) || !initialLayer) return;
    setViewerLayers([initialLayer]);
    setActiveViewerLayer(initialLayer.id);
    if (initialLayer) {
      const raf = viewerFitRaf.current || (viewerFitRaf.current = createViewerRafScheduler(globalThis));
      raf.schedule(() => {
        if (!viewerSession.current.isCurrent(token)) return;
        const rect = viewerCanvasRef.current?.getBoundingClientRect();
        if (rect) setViewerZoom(fitViewerZoom(rect.width, rect.height, [initialLayer]));
      });
    }
  };

  const refreshViewerHistory = async (folderId = viewerSelectedFolder) => {
    const token = viewerSession.current.request("history", { latest: true });
    setViewerHistoryLoading(true);
    try {
      const query = folderId ? `?folder=${encodeURIComponent(folderId)}` : "";
      const response = await fetch(`/api/inference/history${query}`, { cache: "no-store" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "无法读取图片历史");
      if (!viewerSession.current.isCurrent(token)) return null;
      setViewerHistory(payload.cards || []);
      const directory = payload.directory || { id: "", name: "输出目录", label: "/", parent_id: "", image_count: 0, folders: payload.folders || [] };
      setViewerDirectory(directory);
      setViewerFolders(directory.folders || []);
      if (directory.label === "/") setViewerOutputRootId(directory.id || "");
      return payload;
    } catch (error) {
      if (viewerSession.current.isCurrent(token)) setViewerNotice(error.message);
      return null;
    } finally {
      if (viewerSession.current.isCurrent(token)) setViewerHistoryLoading(false);
    }
  };

  const selectViewerFolder = (folderId) => {
    setViewerSelectedFolder(folderId);
    setViewerHistoryBatch(null);
    setViewerHistoryReturnId("");
    if (viewerHistoryRef.current) viewerHistoryRef.current.scrollTop = 0;
    void refreshViewerHistory(folderId);
  };

  const visibleHistoryFiles = (files) => files.filter((file) => !hiddenHistoryAssets.includes(file.id));
  const visibleHistoryCards = viewerHistory.map((card) => {
    const files = visibleHistoryFiles(card.files || []);
    return { ...card, files, count: files.length, preview: files.find((file) => file.image_index === 1) || files[0] };
  }).filter((card) => card.files.length > 0);

  const saveViewerUndo = (snapshot = { layers: viewerLayers, snappedLayers: viewerSnappedLayers, activeLayer: activeViewerLayer }) => {
    viewerUndo.current = [...viewerUndo.current.slice(-49), {
      layers: snapshot.layers.map((layer) => ({ ...layer })),
      snappedLayers: [...snapshot.snappedLayers],
      activeLayer: snapshot.activeLayer,
    }];
  };

  const undoViewerChange = () => {
    const snapshot = viewerUndo.current.pop();
    if (!snapshot) return;
    setViewerLayers(snapshot.layers);
    setViewerSnappedLayers(snapshot.snappedLayers);
    setActiveViewerLayer(snapshot.activeLayer);
    setViewerSnapGuide(null);
    setViewerNotice("已撤销上一步画布操作");
  };

  const viewerLayerFromAsset = (asset, position = {}) => ({
    id: `${asset.id || asset.asset_id || "viewer"}-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    assetId: asset.id || asset.asset_id || "",
    url: asset.url,
    originalUrl: asset.originalUrl || asset.url,
    naturalWidth: asset.naturalWidth || asset.width || 1,
    naturalHeight: asset.naturalHeight || asset.height || 1,
    name: asset.name || asset.output_name || "XirAI.png",
    x: position.x ?? 0,
    y: position.y ?? 0,
    scale: 1,
    isCollage: Boolean(asset.manual_layout || asset.manualLayout),
    manualLayout: asset.manual_layout || asset.manualLayout || null,
    mimeType: asset.mime_type || asset.mimeType || "",
  });

  const restoreManualCollage = async (layout) => {
    viewerFitRaf.current?.cancel();
    viewerFitRaf.current = null;
    const session = viewerSession.current.beginSession();
    const token = viewerSession.current.request("restore", { session, latest: true });
    const sourceLayers = layout?.layers?.filter((layer) => typeof layer?.url === "string" && layer.url) || [];
    if (!sourceLayers.length) {
      setViewerNotice("该拼图没有可恢复的原图布局");
      return false;
    }
    const loadedLayers = await Promise.all(sourceLayers.map((layer) => imageAssetFromSource(layer).catch((error) => {
      if (viewerSession.current.isCurrent(token)) setViewerNotice(error.message);
      return null;
    })));
    if (!viewerSession.current.isCurrent(token)) return false;
    const restored = loadedLayers.filter(Boolean).map((layer, index) => ({
      id: `restored-${Date.now()}-${index}-${Math.random().toString(16).slice(2)}`,
      assetId: layer.assetId || "",
      url: layer.url,
      originalUrl: layer.originalUrl || layer.url,
      naturalWidth: layer.naturalWidth,
      naturalHeight: layer.naturalHeight,
      name: layer.name || `图片 ${index + 1}`,
      x: Number.isFinite(layer.x) ? layer.x : 0,
      y: Number.isFinite(layer.y) ? layer.y : 0,
      scale: Math.max(.1, Math.min(8, Number(layer.scale) || 1)),
    }));
    viewerUndo.current = [];
    setViewerLayers(restored);
    setViewerSnappedLayers([]);
    setActiveViewerLayer(restored.at(-1).id);
    setViewerZoom(1);
    setViewerPan({ x: 0, y: 0 });
    setCollageResult(null);
    setViewerMenu(null);
    setViewerNotice("已恢复原图、位置和实际缩放，可继续无损排版");
    return true;
  };

  const focusViewerAsset = async (asset) => {
    const token = viewerSession.current.request("focus", { latest: true });
    const loaded = await imageAssetFromSource(asset).catch((error) => {
      if (viewerSession.current.isCurrent(token)) setViewerNotice(error.message);
      return null;
    });
    if (!viewerSession.current.isCurrent(token)) return;
    if (!loaded) return;
    const layer = viewerLayerFromAsset(loaded);
    viewerUndo.current = [];
    setViewerLayers([layer]);
    setActiveViewerLayer(layer.id);
    setViewerSnappedLayers([]);
    setGeneratedName(layer.name);
    setViewerZoom(1);
    setViewerPan({ x: 0, y: 0 });
    setViewerMenu(null);
  };

  const addViewerAsset = async (asset, position = {}) => {
    const token = viewerSession.current.request("add", { latest: false });
    const loaded = await imageAssetFromSource(asset).catch((error) => {
      if (viewerSession.current.isCurrent(token)) setViewerNotice(error.message);
      return null;
    });
    if (!viewerSession.current.isCurrent(token)) return;
    if (!loaded) return;
    const layer = viewerLayerFromAsset(loaded, position);
    saveViewerUndo();
    setViewerLayers((current) => [...current, layer]);
    setActiveViewerLayer(layer.id);
    setViewerMenu(null);
  };

  const openHistoryCard = (card) => {
    if (card.kind === "batch" && card.files.length > 1) {
      viewerHistoryScroll.current = viewerHistoryRef.current?.scrollTop || 0;
      setViewerHistoryReturnId(card.id);
      setViewerHistoryBatch(card);
      focusViewerAsset(card.preview);
      return;
    }
    focusViewerAsset(card.preview);
  };

  const returnToHistoryCards = () => {
    setViewerHistoryBatch(null);
    window.requestAnimationFrame(() => {
      if (viewerHistoryRef.current) viewerHistoryRef.current.scrollTop = viewerHistoryScroll.current;
    });
  };

  const historyDragStart = (event, asset) => {
    event.dataTransfer.effectAllowed = "copy";
    event.dataTransfer.setData("application/x-xirai-image", JSON.stringify(asset));
    event.dataTransfer.setData("text/plain", asset.url);
  };

  const viewerDrop = (event) => {
    event.preventDefault();
    if (activeCollage) return;
    const encoded = event.dataTransfer.getData("application/x-xirai-image");
    if (!encoded) {
      void addViewerFiles([...event.dataTransfer.files], event);
      return;
    }
    try {
      const asset = JSON.parse(encoded);
      const rect = event.currentTarget.getBoundingClientRect();
      addViewerAsset(asset, {
        x: (event.clientX - rect.left - rect.width / 2 - viewerPan.x) / viewerZoom,
        y: (event.clientY - rect.top - rect.height / 2 - viewerPan.y) / viewerZoom,
      });
    } catch {}
  };

  const addViewerFiles = async (files, event) => {
    const imageFiles = files.filter((file) => file.type.startsWith("image/"));
    if (activeCollage || !imageFiles.length) return;
    const token = viewerSession.current.request("drop", { latest: false });
    try {
      const assets = await Promise.all(imageFiles.map(async (file, index) => ({
        id: "",
        url: await readImageFile(file),
        name: file.name || `粘贴图片-${index + 1}.png`,
        mimeType: file.type,
      }))).then((assets) => Promise.all(assets.map((asset) => imageAssetFromSource(asset))));
      if (!(await viewerSession.current.waitForDropTurn(token))) return;
      const rect = viewerCanvasRef.current?.getBoundingClientRect();
      const point = rect && event ? {
        x: (event.clientX - rect.left - rect.width / 2 - viewerPan.x) / viewerZoom,
        y: (event.clientY - rect.top - rect.height / 2 - viewerPan.y) / viewerZoom,
      } : { x: -viewerPan.x / viewerZoom, y: -viewerPan.y / viewerZoom };
      saveViewerUndo();
      const layers = assets.map((asset, index) => viewerLayerFromAsset(asset, { x: point.x + index * 28, y: point.y + index * 28 }));
      setViewerLayers((current) => [...current, ...layers]);
      setActiveViewerLayer(layers.at(-1).id);
      setViewerNotice(`已添加 ${layers.length} 张本地图片`);
    } catch (error) {
      if (viewerSession.current.isCurrent(token)) setViewerNotice(error.message);
    } finally {
      viewerSession.current.releaseDrop(token);
    }
  };

  const updateViewerLayer = (id, updates) => setViewerLayers((current) => current.map((layer) => layer.id === id ? { ...layer, ...updates } : layer));

  const scaleViewerLayer = (id, factor) => {
    if (!viewerLayerResizeEnabled) return;
    saveViewerUndo();
    setViewerLayers((current) => current.map((layer) => layer.id === id
      ? { ...layer, scale: Math.max(.1, Math.min(8, layer.scale * factor)) }
      : layer));
  };

  const setViewerLayerScale = (id, percentage) => {
    if (!viewerLayerResizeEnabled) return;
    const value = Number(percentage);
    if (!Number.isFinite(value)) return;
    saveViewerUndo();
    updateViewerLayer(id, { scale: Math.max(.1, Math.min(8, value / 100)) });
  };

  const removeViewerLayer = (id) => {
    const removed = viewerLayers.find((layer) => layer.id === id);
    saveViewerUndo();
    setViewerLayers((current) => current.filter((layer) => layer.id !== id));
    setViewerSnappedLayers((current) => current.filter((layerId) => layerId !== id));
    setActiveViewerLayer((current) => current === id ? "" : current);
    if (removed?.isCollage) setCollageResult(null);
    setViewerMenu(null);
  };

  const copyViewerLayer = async (layer, includeLayerMarker = false) => {
    const token = viewerSession.current.request("copy", { latest: true });
    try {
      let blob;
      if (layer.url.startsWith("data:")) {
        blob = dataUrlBlob(layer.url);
      } else if (layer.assetId) {
        const response = await fetch("/api/inference/history/copy", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ asset_id: layer.assetId }),
        });
        if (!response.ok) throw new Error("无法读取干净图片");
        blob = await response.blob();
      } else {
        const image = await loadBrowserImage(layer.url);
        const canvas = document.createElement("canvas");
        canvas.width = image.naturalWidth;
        canvas.height = image.naturalHeight;
        canvas.getContext("2d").drawImage(image, 0, 0);
        blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
      }
      if (!viewerSession.current.isCurrent(token)) return;
      const mimeType = blob.type === "image/gif" ? "image/gif" : "image/png";
      const contents = { [mimeType]: blob };
      if (includeLayerMarker) contents["text/plain"] = new Blob([`XIRAI_LAYER:${layer.id}`], { type: "text/plain" });
      await navigator.clipboard.write([new ClipboardItem(contents)]);
      if (viewerSession.current.isCurrent(token)) setViewerNotice(mimeType === "image/gif" ? "已复制 GIF 动画" : "已复制无生成元数据的 PNG 图片");
    } catch (error) {
      if (viewerSession.current.isCurrent(token)) setViewerNotice(`复制失败：${error.message}`);
    }
    if (viewerSession.current.isCurrent(token)) setViewerMenu(null);
  };

  const requestHistoryDelete = (files, label) => {
    const assetIds = files.map((file) => file.id).filter(Boolean);
    if (!assetIds.length) return;
    setHistoryDelete({ assetIds, label, count: assetIds.length });
    setViewerMenu(null);
  };

  const finishHistoryDelete = async (deleteSource) => {
    if (!historyDelete) return;
    const assetIds = historyDelete.assetIds;
    const token = viewerSession.current.request("history-delete", { latest: true });
    if (deleteSource) {
      try {
        const response = await fetch("/api/inference/history", {
          method: "DELETE",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ asset_ids: assetIds, delete_source: true }),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "源文件删除失败");
        if (!viewerSession.current.isCurrent(token)) return;
        await refreshViewerHistory(viewerSelectedFolder);
        if (!viewerSession.current.isCurrent(token)) return;
        setViewerLayers((current) => current.filter((layer) => !assetIds.includes(layer.assetId)));
        const remaining = generatedOutputs.filter((output) => !assetIds.includes(output.asset_id));
        if (remaining.length !== generatedOutputs.length) {
          setGeneratedOutputs(remaining);
          setSelectedOutputIndex(0);
          setGeneratedImage(remaining[0]?.url || "");
          setGeneratedName(remaining[0]?.output_name || "");
          if (!remaining.length) setStatus("idle");
        }
        setViewerNotice(`已删除 ${assetIds.length} 个源文件`);
      } catch (error) {
        if (viewerSession.current.isCurrent(token)) setViewerNotice(error.message);
        if (viewerSession.current.isCurrent(token)) setHistoryDelete(null);
        return;
      }
    } else {
      if (!viewerSession.current.isCurrent(token)) return;
      setViewerNotice(`已从本次启动的左侧历史中隐藏 ${assetIds.length} 张图片`);
    }
    if (!viewerSession.current.isCurrent(token)) return;
    setHiddenHistoryAssets((current) => [...new Set([...current, ...assetIds])]);
    if (viewerHistoryBatch && assetIds.some((id) => viewerHistoryBatch.files.some((file) => file.id === id))) {
      const files = viewerHistoryBatch.files.filter((file) => !assetIds.includes(file.id));
      setViewerHistoryBatch(files.length ? { ...viewerHistoryBatch, files, count: files.length, preview: files[0] } : null);
    }
    if (!viewerSession.current.isCurrent(token)) return;
    setHistoryDelete(null);
  };

  const chooseCollageTemplate = (template) => {
    setActiveCollage({ templateId: template.id, slots: Array(template.count).fill(null) });
    setActiveCollageSlot(-1);
    setCollageResult(null);
    setViewerTemplatesOpen(false);
    setViewerZoom(1);
    setViewerPan({ x: 0, y: 0 });
    setViewerNotice(`已启用 ${template.count} 图 · ${template.label}，拖入图片填充区块`);
  };

  const setCollageSlot = (index, asset, existingToken = null) => {
    const token = existingToken || viewerSession.current.request("slot", { latest: true, key: `slot:${index}` });
    if (!viewerSession.current.isCurrent(token)) return;
    const entry = { asset, scale: 1, alignX: .5, alignY: .5 };
    setActiveCollage((current) => current
      ? { ...current, slots: current.slots.map((item, slotIndex) => slotIndex === index ? entry : item) }
      : current);
    void loadBrowserImage(asset.url).then((image) => {
      if (!viewerSession.current.isCurrent(token)) return;
      setActiveCollage((current) => current
        ? { ...current, slots: current.slots.map((item, slotIndex) => slotIndex === index && item?.asset.url === asset.url ? { ...item, asset: { ...item.asset, width: image.naturalWidth, height: image.naturalHeight } } : item) }
        : current);
    }).catch((error) => { if (viewerSession.current.isCurrent(token)) setViewerNotice(error.message); });
  };

  const dropCollageSlot = (event, index) => {
    event.preventDefault();
    event.stopPropagation();
    const encoded = event.dataTransfer.getData("application/x-xirai-image");
    if (encoded) {
      try { setCollageSlot(index, JSON.parse(encoded)); setActiveCollageSlot(index); } catch {}
      return;
    }
    const file = [...event.dataTransfer.files].find((item) => item.type.startsWith("image/"));
    if (!file) return;
    const token = viewerSession.current.request("slot", { latest: true, key: `slot:${index}` });
    void readImageFile(file).then((url) => {
      if (!viewerSession.current.isCurrent(token)) return;
      setCollageSlot(index, { id: "", url, name: file.name || `外部图片-${index + 1}.png`, mimeType: file.type }, token);
      if (!viewerSession.current.isCurrent(token)) return;
      setActiveCollageSlot(index);
      setViewerNotice(`已将外部图片填入区块 ${index + 1}${event.dataTransfer.files.length > 1 ? "（仅使用第一张）" : ""}`);
    }).catch((error) => { if (viewerSession.current.isCurrent(token)) setViewerNotice(error.message); });
  };

  const updateCollageSlot = (index, updates) => setActiveCollage((current) => current
    ? { ...current, slots: current.slots.map((item, slotIndex) => slotIndex === index && item ? { ...item, ...updates } : item) }
    : current);

  const toggleEdgeLines = () => {
    updateViewerEdgeLine({ enabled: !viewerEdgeLine.enabled });
  };

  const updateViewerEdgeLine = (updates) => {
    const next = { ...viewerEdgeLine, ...updates };
    setViewerEdgeLine(next);
    if (!collageResult || collageResult.saved) return;
    if (collageResult.mode === "manual") {
      restoreManualCollage(collageResult.manualLayout);
      setViewerNotice("边缘线已更新，已恢复原图布局，请再次点击一键拼图");
      return;
    }
    setActiveCollage({ templateId: collageResult.templateId, slots: collageResult.slots });
    setActiveCollageSlot(-1);
    setViewerLayers([]);
    setCollageResult(null);
    setViewerNotice("边缘线已更新，请再次确认拼图后保存");
  };

  const pickEdgeColor = async () => {
    const token = viewerSession.current.request("eyedropper", { latest: true });
    try {
      if (!globalThis.EyeDropper) throw new Error("当前浏览器不支持颜色提取器");
      const result = await new globalThis.EyeDropper().open();
      if (!viewerSession.current.isCurrent(token)) return;
      updateViewerEdgeLine({ color: result.sRGBHex });
    } catch (error) {
      if (error.name !== "AbortError" && viewerSession.current.isCurrent(token)) setViewerNotice(error.message);
    }
  };

  const imageLayoutForSlot = (image, slot, slotState) => {
    const baseScale = Math.min(slot.width / image.naturalWidth, slot.height / image.naturalHeight);
    const scale = baseScale * (slotState?.scale || 1);
    const width = image.naturalWidth * scale;
    const height = image.naturalHeight * scale;
    return {
      width,
      height,
      x: slot.x + (slot.width - width) * (slotState?.alignX ?? .5),
      y: slot.y + (slot.height - height) * (slotState?.alignY ?? .5),
    };
  };

  const confirmCollage = async () => {
    const token = viewerSession.current.request("confirm", { latest: true });
    const template = collageTemplates.find((item) => item.id === activeCollage?.templateId);
    if (!template || activeCollage.slots.some((slot) => !slot)) {
      setViewerNotice("请先填满全部拼图区块");
      return;
    }
    try {
      const entries = activeCollage.slots;
      const images = await Promise.all(entries.map((entry) => loadBrowserImage(entry.asset.url)));
      if (!viewerSession.current.isCurrent(token)) return;
      const layout = adaptiveCollageLayout(template, entries.map((entry) => entry.asset));
      const width = Math.min(24576, Math.ceil(Math.max(...images.map((image, index) => image.naturalWidth / layout.slots[index].w))));
      const height = Math.min(24576, Math.ceil(width / layout.aspect));
      const animated = entries.some((entry) => isGifAsset(entry.asset));
      if (animated) {
        const animatedLayers = await Promise.all(entries.map(async (entry, index) => {
          const normalized = layout.slots[index];
          const slot = { x: normalized.x * width, y: normalized.y * height, width: normalized.w * width, height: normalized.h * height };
          const drawing = imageLayoutForSlot(images[index], slot, entry);
          return {
            url: await imageSourceDataUrl(entry.asset.url),
            x: drawing.x,
            y: drawing.y,
            width: drawing.width,
            height: drawing.height,
            clip: slot,
          };
        }));
        const animationScale = Math.min(1, 4096 / width, 4096 / height);
        const animatedWidth = Math.max(1, Math.round(width * animationScale));
        const animatedHeight = Math.max(1, Math.round(height * animationScale));
        const dataUrl = await renderAnimatedCollage(animatedLayers.map((layer) => ({
          ...layer,
          x: layer.x * animationScale,
          y: layer.y * animationScale,
          width: layer.width * animationScale,
          height: layer.height * animationScale,
          clip: layer.clip && Object.fromEntries(Object.entries(layer.clip).map(([key, value]) => [key, value * animationScale])),
        })), animatedWidth, animatedHeight, viewerEdgeLine);
        if (!viewerSession.current.isCurrent(token)) return;
        const name = `XirAI-collage-${new Date().toISOString().replace(/[:T]/g, "-").slice(0, 16)}.gif`;
        const result = { dataUrl, name, width: animatedWidth, height: animatedHeight, templateId: template.id, slots: activeCollage.slots, edgeLine: { ...viewerEdgeLine }, isGif: true, needsConfirmation: false };
        const layer = { id: `collage-${Date.now()}`, url: dataUrl, originalUrl: dataUrl, naturalWidth: animatedWidth, naturalHeight: animatedHeight, name, x: 0, y: 0, scale: 1, isCollage: true, mimeType: "image/gif" };
        setCollageResult(result);
        setActiveCollage(null);
        setViewerLayers([layer]);
        setActiveViewerLayer(layer.id);
        setViewerNotice(`GIF 拼图已合成为 ${animatedWidth} × ${animatedHeight}`);
        return;
      }
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const context = canvas.getContext("2d", { alpha: true });
      const slotEdges = sharedEdgeHiddenSides(layout.slots.map((slot, index) => ({ id: String(index), left: slot.x, right: slot.x + slot.w, top: slot.y, bottom: slot.y + slot.h })), .0001);
      images.forEach((image, index) => {
        const normalized = layout.slots[index];
        const slot = { x: normalized.x * width, y: normalized.y * height, width: normalized.w * width, height: normalized.h * height };
        const drawing = imageLayoutForSlot(image, slot, entries[index]);
        context.save();
        context.beginPath();
        context.rect(slot.x, slot.y, slot.width, slot.height);
        context.clip();
        context.drawImage(image, drawing.x, drawing.y, drawing.width, drawing.height);
        context.restore();
        drawEdgeLine(context, slot.x, slot.y, slot.width, slot.height, viewerEdgeLine, 1, slotEdges[index] || []);
      });
      const dataUrl = canvas.toDataURL("image/png");
      if (!viewerSession.current.isCurrent(token)) return;
      const name = `XirAI-collage-${new Date().toISOString().replace(/[:T]/g, "-").slice(0, 16)}.png`;
      const result = { dataUrl, name, width, height, templateId: template.id, slots: activeCollage.slots, edgeLine: { ...viewerEdgeLine }, needsConfirmation: false };
      const layer = { id: `collage-${Date.now()}`, url: dataUrl, originalUrl: dataUrl, naturalWidth: width, naturalHeight: height, name, x: 0, y: 0, scale: 1, isCollage: true };
      setCollageResult(result);
      setActiveCollage(null);
      setViewerLayers([layer]);
      setActiveViewerLayer(layer.id);
      setViewerNotice(`拼图已合成为 ${width} × ${height} 无损 PNG`);
    } catch (error) {
      if (viewerSession.current.isCurrent(token)) setViewerNotice(`拼图失败：${error.message}`);
    }
  };

  const editCollage = (target = null) => {
    const manualLayout = target?.manualLayout || (collageResult?.mode === "manual" ? collageResult.manualLayout : null);
    if (manualLayout) {
      restoreManualCollage(manualLayout);
      return;
    }
    if (!collageResult) {
      setViewerNotice("该拼图没有可恢复的编辑布局");
      return;
    }
    setActiveCollage({ templateId: collageResult.templateId, slots: collageResult.slots });
    setViewerEdgeLine(collageResult.edgeLine || viewerEdgeLine);
    setActiveCollageSlot(-1);
    setViewerLayers([]);
    setCollageResult(null);
    setViewerMenu(null);
  };

  const discardCollage = () => {
    setActiveCollage(null);
    setCollageResult(null);
    setViewerLayers([]);
    setViewerNotice("拼图已清空，未保存源文件");
  };

  const cancelCollageDraft = () => {
    setActiveCollage(null);
    setViewerMenu(null);
    setViewerNotice("已取消未完成的拼图，预览窗口图片已保留");
  };

  const saveCollage = async () => {
    if (!collageResult) return;
    const token = viewerSession.current.request("save", { latest: true });
    try {
      const response = await fetch("/api/inference/collages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_data: collageResult.dataUrl, name: collageResult.name, manual_layout: collageResult.persistedManualLayout }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "拼图保存失败");
      if (!viewerSession.current.isCurrent(token)) return;
      setViewerLayers((current) => current.map((layer) => layer.isCollage ? { ...layer, assetId: payload.id, url: payload.url, name: payload.name, manualLayout: collageResult.manualLayout || layer.manualLayout } : layer));
      setCollageResult((current) => ({ ...current, saved: true, assetId: payload.id, url: payload.url, name: payload.name }));
      await refreshViewerHistory(viewerSelectedFolder);
      if (viewerSession.current.isCurrent(token)) setViewerNotice(`已保存到当日 outputs：${payload.name}`);
    } catch (error) {
      if (viewerSession.current.isCurrent(token)) setViewerNotice(error.message);
    }
  };

  const createManualCollage = async () => {
    if (viewerLayers.length < 2) {
      setViewerNotice("至少需要两张图片才能一键拼图");
      return;
    }
    const token = viewerSession.current.request("manual-collage", { latest: true });
    try {
       const nodes = viewerLayers.map((layer) => ({
          layer,
       }));
       const images = await Promise.all(nodes.map((item) => loadBrowserImage(item.layer.url)));
       if (!viewerSession.current.isCurrent(token)) return;
       const rectangles = nodes.map((item) => viewerLayerBounds(item.layer));
      const left = Math.min(...rectangles.map((rect) => rect.left));
      const top = Math.min(...rectangles.map((rect) => rect.top));
      const right = Math.max(...rectangles.map((rect) => rect.right));
      const bottom = Math.max(...rectangles.map((rect) => rect.bottom));
      // Use each image's unscaled display size as the output-resolution reference.
      // A 10% handle resize must keep the image small in the exported collage.
       const outputScale = 1;
       const width = Math.ceil(right - left);
       const height = Math.ceil(bottom - top);
       if (width > 24576 || height > 24576) throw new Error(`导出尺寸 ${width} × ${height} 超过 24576 像素上限，请缩小图层后重试`);
      const animated = nodes.some((item) => isGifAsset(item.layer));
      if (animated) {
        const animatedLayers = await Promise.all(nodes.map(async (item, index) => {
          const rect = rectangles[index];
          return {
            url: await imageSourceDataUrl(item.layer.url),
            x: (rect.left - left) * outputScale,
            y: (rect.top - top) * outputScale,
            width: rect.width * outputScale,
            height: rect.height * outputScale,
          };
        }));
        const animationScale = Math.min(1, 4096 / width, 4096 / height);
        const animatedWidth = Math.max(1, Math.round(width * animationScale));
        const animatedHeight = Math.max(1, Math.round(height * animationScale));
        const dataUrl = await renderAnimatedCollage(animatedLayers.map((layer) => ({
          ...layer,
          x: layer.x * animationScale,
          y: layer.y * animationScale,
          width: layer.width * animationScale,
          height: layer.height * animationScale,
         })), animatedWidth, animatedHeight, viewerEdgeLine);
         if (!viewerSession.current.isCurrent(token)) return;
        const name = `XirAI-manual-collage-${new Date().toISOString().replace(/[:T]/g, "-").slice(0, 16)}.gif`;
        const manualLayout = {
          version: 1,
          layers: viewerLayers.map(({ assetId, url, name: layerName, x, y, scale }) => ({ assetId, url, name: layerName, x, y, scale })),
        };
        const persistedManualLayout = manualLayout.layers.every((layer) => layer.assetId && layer.url.startsWith("/api/inference/history/assets/")) ? manualLayout : null;
        const layer = { id: `collage-${Date.now()}`, url: dataUrl, originalUrl: dataUrl, naturalWidth: animatedWidth, naturalHeight: animatedHeight, name, x: 0, y: 0, scale: 1, isCollage: true, mimeType: "image/gif", manualLayout };
        setCollageResult({ mode: "manual", dataUrl, name, width: animatedWidth, height: animatedHeight, manualLayout, persistedManualLayout, edgeLine: { ...viewerEdgeLine }, isGif: true, needsConfirmation: false });
        setViewerLayers([layer]);
        setActiveViewerLayer(layer.id);
        setViewerNotice(`已将手动排版合成为 ${animatedWidth} × ${animatedHeight} GIF`);
        return;
      }
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const context = canvas.getContext("2d", { alpha: true });
      const layerEdges = sharedEdgeHiddenSides(rectangles.map((rect, index) => ({ id: nodes[index].layer.id, left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom })), Math.max(1, viewerEdgeLine.width));
      images.forEach((image, index) => {
        const rect = rectangles[index];
        const x = (rect.left - left) * outputScale;
        const y = (rect.top - top) * outputScale;
        const drawWidth = rect.width * outputScale;
        const drawHeight = rect.height * outputScale;
        context.drawImage(image, x, y, drawWidth, drawHeight);
        drawEdgeLine(context, x, y, drawWidth, drawHeight, viewerEdgeLine, outputScale, layerEdges[nodes[index].layer.id] || []);
      });
      const name = `XirAI-manual-collage-${new Date().toISOString().replace(/[:T]/g, "-").slice(0, 16)}.png`;
       const dataUrl = canvas.toDataURL("image/png");
       if (!viewerSession.current.isCurrent(token)) return;
      const manualLayout = {
        version: 1,
        layers: viewerLayers.map(({ assetId, url, name: layerName, x, y, scale }) => ({ assetId, url, name: layerName, x, y, scale })),
      };
      const persistedManualLayout = manualLayout.layers.every((layer) => layer.assetId && layer.url.startsWith("/api/inference/history/assets/")) ? manualLayout : null;
      const layer = { id: `collage-${Date.now()}`, url: dataUrl, originalUrl: dataUrl, naturalWidth: width, naturalHeight: height, name, x: 0, y: 0, scale: 1, isCollage: true, manualLayout };
      setCollageResult({ mode: "manual", dataUrl, name, width, height, manualLayout, persistedManualLayout, edgeLine: { ...viewerEdgeLine }, needsConfirmation: false });
      setViewerLayers([layer]);
      setActiveViewerLayer(layer.id);
      setViewerNotice(`已将手动排版合成为 ${width} × ${height} PNG`);
     } catch (error) {
       if (viewerSession.current.isCurrent(token)) setViewerNotice(`一键拼图失败：${error.message}`);
    }
  };

  const selectGeneratedOutput = (index) => {
    const output = generatedOutputs[index];
    if (!output) return;
    setSelectedOutputIndex(index);
    setGeneratedImage(output.url);
    setGeneratedName(output.output_name);
    setViewerZoom(1);
    setViewerPan({ x: 0, y: 0 });
  };

  const zoomImageAt = (event) => {
    event.preventDefault();
    const rect = event.currentTarget.getBoundingClientRect();
    const pointerX = event.clientX - rect.left - rect.width / 2;
    const pointerY = event.clientY - rect.top - rect.height / 2;
    const result = viewerZoomAtPoint(viewerZoom, viewerPan, { x: pointerX, y: pointerY }, event.deltaY < 0 ? 1.15 : 1 / 1.15);
    setViewerPan(result.pan);
    setViewerZoom(result.zoom);
  };

  const startViewerDrag = (event) => {
    if (event.target.closest?.(".viewer-image-layer, .collage-slot")) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    viewerDrag.current = { kind: "canvas", x: event.clientX, y: event.clientY, panX: viewerPan.x, panY: viewerPan.y };
  };

  const moveViewerImage = (event) => {
    if (!viewerDrag.current || viewerDrag.current.kind !== "canvas" || !event.currentTarget.hasPointerCapture(event.pointerId)) return;
    setViewerPan({
      x: viewerDrag.current.panX + event.clientX - viewerDrag.current.x,
      y: viewerDrag.current.panY + event.clientY - viewerDrag.current.y,
    });
  };

  const startViewerLayerDrag = (event, layer) => {
    event.stopPropagation();
    const resizeHandle = event.target.closest?.(".layer-corner");
    if (resizeHandle && !viewerLayerResizeEnabled) return;
    setActiveViewerLayer(layer.id);
    const snapToken = viewerSession.current.request("snap", { latest: true });
    window.requestAnimationFrame(() => { if (viewerSession.current.isCurrent(snapToken)) refreshViewerSnapGuide(layer.id); });
    event.currentTarget.setPointerCapture(event.pointerId);
    const undoSnapshot = { layers: viewerLayers, snappedLayers: viewerSnappedLayers, activeLayer: activeViewerLayer };
    if (resizeHandle) {
      const rect = event.currentTarget.getBoundingClientRect();
      const handle = ["tl", "tr", "bl", "br", "top", "right", "bottom", "left"].find((name) => event.target.classList.contains(name));
      const isSideHandle = ["top", "right", "bottom", "left"].includes(handle);
      let originX = handle?.includes("r") ? 0 : 1;
      let originY = handle?.includes("b") ? 0 : 1;
      if (handle === "top") { originX = .5; originY = 1; }
      if (handle === "bottom") { originX = .5; originY = 0; }
      if (handle === "left") { originX = 1; originY = .5; }
      if (handle === "right") { originX = 0; originY = .5; }
      if (isSideHandle && viewerAlignmentGuidesEnabled) {
        const others = [...viewerCanvasRef.current.querySelectorAll("[data-viewer-layer-id]")]
          .filter((node) => node.dataset.viewerLayerId !== layer.id)
          .map((node) => node.getBoundingClientRect());
        const aligned = (value, targets) => others.some((other) => targets(other).some((target) => Math.abs(value - target) <= 8));
        const xPositions = { left: rect.left, center: (rect.left + rect.right) / 2, right: rect.right };
        const yPositions = { top: rect.top, center: (rect.top + rect.bottom) / 2, bottom: rect.bottom };
        const chooseAnchor = (positions, candidates, targets, fallback) => candidates.find((name) => aligned(positions[name], targets)) || fallback;
        if (handle === "top" || handle === "bottom") {
          const opposite = handle === "top" ? "bottom" : "top";
          const yAnchor = chooseAnchor(yPositions, [opposite, "center"], (other) => [other.top, (other.top + other.bottom) / 2, other.bottom], opposite);
          const xAnchor = chooseAnchor(xPositions, ["left", "right", "center"], (other) => [other.left, (other.left + other.right) / 2, other.right], "center");
          originX = { left: 0, center: .5, right: 1 }[xAnchor];
          originY = { top: 0, center: .5, bottom: 1 }[yAnchor];
        } else {
          const opposite = handle === "left" ? "right" : "left";
          const xAnchor = chooseAnchor(xPositions, [opposite, "center"], (other) => [other.left, (other.left + other.right) / 2, other.right], opposite);
          const yAnchor = chooseAnchor(yPositions, ["top", "bottom", "center"], (other) => [other.top, (other.top + other.bottom) / 2, other.bottom], "center");
          originX = { left: 0, center: .5, right: 1 }[xAnchor];
          originY = { top: 0, center: .5, bottom: 1 }[yAnchor];
        }
      }
      const anchorX = rect.left + rect.width * originX;
      const anchorY = rect.top + rect.height * originY;
      viewerDrag.current = {
        kind: "resize", id: layer.id, handle, originX, originY, anchorX, anchorY,
        width: rect.width, height: rect.height, scale: layer.scale, layerX: layer.x, layerY: layer.y,
        startX: event.clientX, startY: event.clientY, undoSnapshot, changed: false, unsnapped: false, sizeMagnet: null,
        pointerId: event.pointerId, pointerTarget: event.currentTarget,
        resizeGesture: { active: true, initialTransform: { scale: layer.scale, x: layer.x, y: layer.y }, currentTransform: { scale: layer.scale, x: layer.x, y: layer.y }, changed: false, pointerId: event.pointerId },
      };
      return;
    }
    const rect = event.currentTarget.getBoundingClientRect();
    viewerDrag.current = {
      kind: "layer",
      id: layer.id,
      x: event.clientX,
      y: event.clientY,
      layerX: layer.x,
      layerY: layer.y,
      rect: { left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom },
      magnetX: null,
      magnetY: null,
      guideMagnetX: null,
      guideMagnetY: null,
      undoSnapshot,
      changed: false,
      unsnapped: false,
    };
  };

  const moveViewerLayer = (event) => {
    if (!viewerDrag.current || !["layer", "resize"].includes(viewerDrag.current.kind) || !event.currentTarget.hasPointerCapture(event.pointerId)) return;
    if (viewerDrag.current.kind === "resize" && !viewerLayerResizeEnabled) return;
    if (viewerDrag.current.kind === "resize") {
      const drag = viewerDrag.current;
      const ratio = drag.handle === "top" || drag.handle === "bottom"
        ? (event.clientY - drag.anchorY) / (drag.startY - drag.anchorY)
        : drag.handle === "left" || drag.handle === "right"
          ? (event.clientX - drag.anchorX) / (drag.startX - drag.anchorX)
          : ((event.clientX - drag.anchorX) * (drag.originX ? -drag.width : drag.width) + (event.clientY - drag.anchorY) * (drag.originY ? -drag.height : drag.height)) / (drag.width ** 2 + drag.height ** 2);
      let scale = Math.max(.1, Math.min(8, drag.scale * ratio));
      const canvasRect = viewerCanvasRef.current?.getBoundingClientRect();
      const resizeRect = (nextScale) => {
        const factor = nextScale / drag.scale;
        const width = drag.width * factor;
        const height = drag.height * factor;
        return {
          left: drag.anchorX - width * drag.originX,
          right: drag.anchorX + width * (1 - drag.originX),
          top: drag.anchorY - height * drag.originY,
          bottom: drag.anchorY + height * (1 - drag.originY),
          width,
          height,
        };
      };
      if (viewerAlignmentGuidesEnabled && canvasRect) {
        const rawRect = resizeRect(scale);
        const guideThreshold = 20;
        const snapThreshold = 12;
        const releaseThreshold = 24;
        const others = [...viewerCanvasRef.current.querySelectorAll("[data-viewer-layer-id]")]
          .filter((node) => node.dataset.viewerLayerId !== drag.id)
          .map((node) => {
            const rect = node.getBoundingClientRect();
            return { id: node.dataset.viewerLayerId, left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom, width: rect.width, height: rect.height };
          });
        const candidates = [];
        others.forEach((other) => {
          const sharesHorizontalEdge = [rawRect.top, rawRect.bottom].some((source) => [other.top, other.bottom].some((target) => Math.abs(target - source) <= guideThreshold));
          const sharesVerticalEdge = [rawRect.left, rawRect.right].some((source) => [other.left, other.right].some((target) => Math.abs(target - source) <= guideThreshold));
          const addSizeCandidates = (dimension, targetSize, currentSize) => {
            const scale = drag.scale * targetSize / currentSize;
            if (scale < .1 || scale > 8) return;
            candidates.push({ layerId: other.id, dimension, match: targetSize === (dimension === "height" ? other.height : other.width) ? "full" : "center", scale, distance: Math.abs(targetSize - (dimension === "height" ? rawRect.height : rawRect.width)), other });
          };
          if (sharesHorizontalEdge) {
            addSizeCandidates("height", other.height, drag.height);
            addSizeCandidates("height", other.height / 2, drag.height);
          }
          if (sharesVerticalEdge) {
            addSizeCandidates("width", other.width, drag.width);
            addSizeCandidates("width", other.width / 2, drag.width);
          }
        });
        const held = drag.sizeMagnet && candidates.find((candidate) => candidate.layerId === drag.sizeMagnet.layerId && candidate.dimension === drag.sizeMagnet.dimension && candidate.match === drag.sizeMagnet.match && candidate.distance <= releaseThreshold);
        const nearest = candidates.filter((candidate) => candidate.distance <= snapThreshold).sort((first, second) => first.distance - second.distance)[0];
        const sizeMatch = held || nearest;
        drag.sizeMagnet = sizeMatch ? { layerId: sizeMatch.layerId, dimension: sizeMatch.dimension, match: sizeMatch.match } : null;
        if (sizeMatch) scale = sizeMatch.scale;

        const snappedRect = resizeRect(scale);
        const nearestGuide = (axis) => {
          const sources = axis === "x" ? [snappedRect.left, (snappedRect.left + snappedRect.right) / 2, snappedRect.right] : [snappedRect.top, (snappedRect.top + snappedRect.bottom) / 2, snappedRect.bottom];
          let best = null;
          others.forEach((other) => {
            const targets = axis === "x" ? [other.left, (other.left + other.right) / 2, other.right] : [other.top, (other.top + other.bottom) / 2, other.bottom];
            sources.forEach((source) => targets.forEach((target) => {
              const distance = target - source;
              if (Math.abs(distance) <= guideThreshold && (!best || Math.abs(distance) < Math.abs(best.distance))) best = { target, distance };
            }));
          });
          return best;
        };
        const nearestX = nearestGuide("x");
        const nearestY = nearestGuide("y");
        const matchingEdges = (axis) => {
          if (!sizeMatch) return [];
          const sources = axis === "x" ? [snappedRect.left, snappedRect.right] : [snappedRect.top, snappedRect.bottom];
          const targets = axis === "x"
            ? [sizeMatch.other.left, (sizeMatch.other.left + sizeMatch.other.right) / 2, sizeMatch.other.right]
            : [sizeMatch.other.top, (sizeMatch.other.top + sizeMatch.other.bottom) / 2, sizeMatch.other.bottom];
          return targets.filter((target) => sources.some((source) => Math.abs(target - source) <= 1));
        };
        const verticalMatches = sizeMatch?.dimension === "width" ? matchingEdges("x") : [];
        const horizontalMatches = sizeMatch?.dimension === "height" ? matchingEdges("y") : [];
        const vertical = [...new Set([nearestX?.target, ...verticalMatches].filter(Number.isFinite))].map((position) => position - canvasRect.left);
        const horizontal = [...new Set([nearestY?.target, ...horizontalMatches].filter(Number.isFinite))].map((position) => position - canvasRect.top);
        const nextGuide = vertical.length || horizontal.length ? { x: vertical[0] ?? null, y: horizontal[0] ?? null, x2: vertical[1] ?? null, y2: horizontal[1] ?? null } : null;
        setViewerSnapGuide((current) => current?.x === nextGuide?.x && current?.y === nextGuide?.y && current?.x2 === nextGuide?.x2 && current?.y2 === nextGuide?.y2 ? current : nextGuide);
      } else {
        drag.sizeMagnet = null;
        setViewerSnapGuide(null);
      }
      const resized = scale / drag.scale;
      const x = drag.layerX + (0.5 - drag.originX) * drag.width * (resized - 1) / viewerZoom;
      const y = drag.layerY + (0.5 - drag.originY) * drag.height * (resized - 1) / viewerZoom;
      const nextTransform = { scale, x, y };
      drag.resizeGesture = viewerResizeGestureMove(drag.resizeGesture, nextTransform);
      drag.changed = drag.resizeGesture.changed;
      if (drag.resizeGesture.shouldApply) {
        if (!drag.unsnapped) {
          setViewerSnappedLayers((current) => current.filter((id) => id !== drag.id));
          drag.unsnapped = true;
        }
        updateViewerLayer(drag.id, nextTransform);
      }
      return;
    }
    const drag = viewerDrag.current;
    const rawX = drag.layerX + (event.clientX - drag.x) / viewerZoom;
    const rawY = drag.layerY + (event.clientY - drag.y) / viewerZoom;
    let x = rawX;
    let y = rawY;
    const canvasRect = viewerCanvasRef.current?.getBoundingClientRect();
    if (canvasRect) {
      const deltaX = (rawX - drag.layerX) * viewerZoom;
      const deltaY = (rawY - drag.layerY) * viewerZoom;
      const candidate = { left: drag.rect.left + deltaX, right: drag.rect.right + deltaX, top: drag.rect.top + deltaY, bottom: drag.rect.bottom + deltaY };
      const snapThreshold = 11;
      const guideThreshold = 20;
      const releaseThreshold = 24;
      const chooseAxis = (axis, threshold, magnet = null, requireOverlap = false) => {
        const sources = axis === "x" ? [candidate.left, (candidate.left + candidate.right) / 2, candidate.right] : [candidate.top, (candidate.top + candidate.bottom) / 2, candidate.bottom];
        if (magnet) {
          const distance = magnet.target - sources[magnet.sourceIndex];
          if (Math.abs(distance) <= releaseThreshold) return { ...magnet, distance };
        }
        let best = null;
        viewerCanvasRef.current.querySelectorAll("[data-viewer-layer-id]").forEach((node) => {
          if (node.dataset.viewerLayerId === drag.id) return;
          const other = node.getBoundingClientRect();
          if (requireOverlap) {
            const overlap = axis === "x" ? Math.min(candidate.bottom, other.bottom) - Math.max(candidate.top, other.top) : Math.min(candidate.right, other.right) - Math.max(candidate.left, other.left);
            if (overlap < 8) return;
          }
          const targets = axis === "x" ? [other.left, (other.left + other.right) / 2, other.right] : [other.top, (other.top + other.bottom) / 2, other.bottom];
          sources.forEach((source, sourceIndex) => targets.forEach((target) => {
            const distance = target - source;
            if (Math.abs(distance) <= threshold && (!best || Math.abs(distance) < Math.abs(best.distance))) best = { target, sourceIndex, distance, layerId: node.dataset.viewerLayerId };
          }));
        });
        return best;
      };
      const guideX = viewerAlignmentGuidesEnabled ? chooseAxis("x", guideThreshold, drag.guideMagnetX) : null;
      const guideY = viewerAlignmentGuidesEnabled ? chooseAxis("y", guideThreshold, drag.guideMagnetY) : null;
      drag.guideMagnetX = guideX;
      drag.guideMagnetY = guideY;
      const edgeX = !guideX && viewerEdgeSnapEnabled ? chooseAxis("x", snapThreshold, drag.magnetX, true) : null;
      const edgeY = !guideY && viewerEdgeSnapEnabled ? chooseAxis("y", snapThreshold, drag.magnetY, true) : null;
      drag.magnetX = edgeX;
      drag.magnetY = edgeY;
      const bestX = guideX || edgeX;
      const bestY = guideY || edgeY;
      if (bestX) x = rawX + bestX.distance / viewerZoom;
      else if (viewerGridEnabled) x = Math.round(rawX / viewerGridSize) * viewerGridSize;
      if (bestY) y = rawY + bestY.distance / viewerZoom;
      else if (viewerGridEnabled) y = Math.round(rawY / viewerGridSize) * viewerGridSize;
      drag.snappedWith = [edgeX?.layerId, edgeY?.layerId].filter(Boolean);
      setViewerSnapGuide(guideX || guideY ? { x: guideX ? guideX.target - canvasRect.left : null, y: guideY ? guideY.target - canvasRect.top : null } : null);
    } else if (viewerGridEnabled) {
      x = Math.round(rawX / viewerGridSize) * viewerGridSize;
      y = Math.round(rawY / viewerGridSize) * viewerGridSize;
    } else {
      setViewerSnapGuide(null);
    }
    if (!drag.unsnapped && (x !== drag.layerX || y !== drag.layerY)) {
      setViewerSnappedLayers((current) => current.filter((id) => id !== drag.id));
      drag.unsnapped = true;
    }
    if (x !== drag.layerX || y !== drag.layerY) drag.changed = true;
    updateViewerLayer(viewerDrag.current.id, { x, y });
  };

  const finishViewerPointer = () => {
    const drag = viewerDrag.current;
    if (drag?.changed) saveViewerUndo(drag.undoSnapshot);
    if (drag?.kind === "layer" && drag.snappedWith?.length) {
      setViewerSnappedLayers((current) => [...new Set([...current, drag.id, ...drag.snappedWith])]);
    }
    viewerDrag.current = null;
    if (!viewerAlignmentGuidesEnabled) setViewerSnapGuide(null);
    else if (drag?.kind === "resize") {
      const snapToken = viewerSession.current.request("snap", { latest: true });
      window.requestAnimationFrame(() => { if (viewerSession.current.isCurrent(snapToken)) refreshViewerSnapGuide(drag.id); });
    }
  };

  const finishViewerResizeForDisable = () => {
    const drag = viewerDrag.current;
    const decision = viewerResizeDisableDecision(drag);
    if (!decision.shouldFinish) return;
    if (decision.shouldReleasePointer && drag.pointerTarget?.hasPointerCapture?.(drag.pointerId)) {
      drag.pointerTarget.releasePointerCapture?.(drag.pointerId);
    }
    if (decision.shouldSaveUndo) saveViewerUndo(drag.undoSnapshot);
    viewerDrag.current = null;
    setViewerSnapGuide(null);
  };

  const toggleViewerLayerResize = () => {
    if (viewerLayerResizeEnabled) finishViewerResizeForDisable();
    setViewerLayerResizeEnabled((current) => !current);
  };

  const refreshViewerSnapGuide = (layerId = activeViewerLayer) => {
    if (!viewerAlignmentGuidesEnabled || activeCollage || !layerId || !viewerCanvasRef.current) {
      setViewerSnapGuide(null);
      return;
    }
    if (viewerDrag.current?.kind === "resize") return;
    const activeNode = viewerCanvasRef.current.querySelector(`[data-viewer-layer-id="${layerId}"]`);
    const canvasRect = viewerCanvasRef.current.getBoundingClientRect();
    if (!activeNode || !canvasRect.width || !canvasRect.height) {
      setViewerSnapGuide(null);
      return;
    }
    const activeRect = activeNode.getBoundingClientRect();
    const guideThreshold = 20;
    const nearestAxis = (axis) => {
      const sources = axis === "x"
        ? [activeRect.left, (activeRect.left + activeRect.right) / 2, activeRect.right]
        : [activeRect.top, (activeRect.top + activeRect.bottom) / 2, activeRect.bottom];
      let best = null;
      viewerCanvasRef.current.querySelectorAll("[data-viewer-layer-id]").forEach((node) => {
        if (node === activeNode) return;
        const other = node.getBoundingClientRect();
        const targets = axis === "x"
          ? [other.left, (other.left + other.right) / 2, other.right]
          : [other.top, (other.top + other.bottom) / 2, other.bottom];
        sources.forEach((source) => targets.forEach((target) => {
          const distance = target - source;
          if (Math.abs(distance) <= guideThreshold && (!best || Math.abs(distance) < Math.abs(best.distance))) best = { target, distance };
        }));
      });
      return best;
    };
    const bestX = nearestAxis("x");
    const bestY = nearestAxis("y");
    const next = bestX || bestY ? { x: bestX ? bestX.target - canvasRect.left : null, y: bestY ? bestY.target - canvasRect.top : null } : null;
    setViewerSnapGuide((current) => current?.x === next?.x && current?.y === next?.y ? current : next);
  };

  useEffect(() => {
    if (!imageViewerOpen || activeCollage || !viewerAlignmentGuidesEnabled) {
      setViewerSnapGuide(null);
      return undefined;
    }
    const frame = window.requestAnimationFrame(() => refreshViewerSnapGuide());
    return () => window.cancelAnimationFrame(frame);
  }, [activeCollage, activeViewerLayer, imageViewerOpen, viewerAlignmentGuidesEnabled, viewerLayers, viewerZoom]);

  const copyActiveViewerLayer = () => {
    if (!activeViewerLayerItem || activeCollage) return;
    viewerClipboard.current = { layer: { ...activeViewerLayerItem }, copiedAt: Date.now() };
    void copyViewerLayer(activeViewerLayerItem, true);
  };

  const pasteViewerLayer = (event) => {
    if (!imageViewerOpen || activeCollage || isEditableTarget(event.target)) return;
    const copied = viewerClipboard.current;
    const marker = event.clipboardData?.getData("text/plain");
    if (copied && marker === `XIRAI_LAYER:${copied.layer.id}` && Date.now() - copied.copiedAt <= 300000) {
      event.preventDefault();
      const layer = { ...copied.layer, id: `viewer-copy-${Date.now()}-${Math.random().toString(16).slice(2)}`, x: copied.layer.x + 28, y: copied.layer.y + 28 };
      saveViewerUndo();
      setViewerLayers((current) => [...current, layer]);
      setActiveViewerLayer(layer.id);
      setViewerNotice("已粘贴画布图片");
      return;
    }
    const clipboardItems = [...(event.clipboardData?.items || [])];
    const files = [...new Map([...event.clipboardData?.files || [], ...clipboardItems.map((item) => item.kind === "file" ? item.getAsFile() : null)]
      .filter((file) => file?.type.startsWith("image/"))
      .map((file) => [`${file.name}:${file.size}:${file.lastModified}`, file])).values()];
    if (files.length) {
      event.preventDefault();
      viewerClipboard.current = null;
      void addViewerFiles(files);
      return;
    }
    if (!copied || Date.now() - copied.copiedAt > 300000) return;
    event.preventDefault();
    const layer = { ...copied.layer, id: `viewer-copy-${Date.now()}-${Math.random().toString(16).slice(2)}`, x: copied.layer.x + 28, y: copied.layer.y + 28 };
    saveViewerUndo();
    setViewerLayers((current) => [...current, layer]);
    setActiveViewerLayer(layer.id);
    setViewerNotice("已粘贴画布图片");
  };

  useEffect(() => {
    if (!imageViewerOpen || activeCollage || !viewerLayers.length) {
      setViewerLayerEdges({});
      return undefined;
    }
    const frame = window.requestAnimationFrame(() => {
      const rectangles = viewerLayers.map((layer) => {
        const node = viewerCanvasRef.current?.querySelector(`[data-viewer-layer-id="${layer.id}"]`);
        const rect = node?.getBoundingClientRect();
        return rect && { id: layer.id, left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom };
      }).filter(Boolean);
      const edges = sharedEdgeHiddenSides(rectangles, Math.max(1, viewerEdgeLine.width));
      setViewerLayerEdges((current) => JSON.stringify(current) === JSON.stringify(edges) ? current : edges);
    });
    return () => window.cancelAnimationFrame(frame);
  }, [activeCollage, imageViewerOpen, viewerEdgeLine.width, viewerLayers, viewerZoom]);

  // Writing a prompt the assistant produced has to go through the same revision bookkeeping as a
  // keystroke, otherwise the preset inserter would splice into a caret offset from the old text.
  // Left-panel geometry. The width is measured at pointer-down rather than read from state,
  // because until the panel has been resized once its width belongs to the stylesheet; every
  // position is then computed from that starting rect, so a clamped drag cannot drift.
  const measuredLeftPanelWidth = () => leftPanelRef.current?.getBoundingClientRect().width || 0;

  const commitWorkspaceLayout = (next) => {
    setWorkspaceLayout((current) => writeWorkspaceLayout(window.localStorage, typeof next === "function" ? next(current) : next));
  };

  const beginPanelResize = (event) => {
    if (event.button !== 0) return;
    panelResizeRef.current = { pointerId: event.pointerId, startX: event.clientX, startWidth: measuredLeftPanelWidth() };
    event.currentTarget.setPointerCapture?.(event.pointerId);
    setPanelResizing(true);
  };

  const continuePanelResize = (event) => {
    const session = panelResizeRef.current;
    if (!session || session.pointerId !== event.pointerId) return;
    setWorkspaceLayout((current) => resizeLeftPanel(current, session.startWidth, event.clientX - session.startX));
  };

  // Persisted once at the end of the gesture rather than on every move: a drag is one decision,
  // and writing each frame would put a few hundred entries through storage for one of them.
  const endPanelResize = (event) => {
    const session = panelResizeRef.current;
    if (!session || session.pointerId !== event.pointerId) return;
    panelResizeRef.current = null;
    setPanelResizing(false);
    event.currentTarget.releasePointerCapture?.(event.pointerId);
    commitWorkspaceLayout((current) => current);
  };

  const panelResizeKeyDown = (event) => {
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault();
      commitWorkspaceLayout((current) => steppedLeftPanel(current, event.key === "ArrowRight" ? 1 : -1, measuredLeftPanelWidth()));
      return;
    }
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      commitWorkspaceLayout(toggleLeftPanel);
      return;
    }
    // The same reset the settings dialog offers, reachable without leaving the handle.
    if (event.key === "Home") {
      event.preventDefault();
      commitWorkspaceLayout({ ...DEFAULT_WORKSPACE_LAYOUT });
    }
  };

  // Layout only. Creation parameters live in `workspace` state and a different store entirely, so
  // there is nothing here that could reach them.
  const resetWorkspaceLayout = () => commitWorkspaceLayout({ ...DEFAULT_WORKSPACE_LAYOUT });

  const applyAssistantPrompt = ({ positive: nextPositive, negative: nextNegative } = {}, { allowEmpty = false, trim = true } = {}) => {
    const commit = (type, value) => {
      const revision = promptTextRevision.current[type].revision + 1;
      promptTextRevision.current[type] = { text: value, revision };
      promptSelectionCache.current[type] = { start: value.length, end: value.length, revision };
      if (type === "positive") setPositive(value);
      else setNegative(value);
    };
    const apply = (type, source) => {
      if (typeof source !== "string") return;
      const value = trim ? source.trim() : source;
      if (allowEmpty || value.trim()) commit(type, value);
    };
    apply("positive", nextPositive);
    apply("negative", nextNegative);
  };

  const applyPromptLibraryEntry = async (entry, target) => {
    if (status === "running") throw new Error("生成期间不能应用词库 Prompt");
    const nextPositive = typeof entry?.positive_prompt === "string" ? entry.positive_prompt : "";
    const nextNegative = typeof entry?.negative_prompt === "string" ? entry.negative_prompt : "";
    if (target === "image") {
      setImageToImage((current) => ({ ...current, positive: nextPositive, negative: nextNegative }));
      setActivePage("image");
      return;
    }
    applyAssistantPrompt({ positive: nextPositive, negative: nextNegative }, { allowEmpty: true, trim: false });
    setActivePage("generate");
  };

  // The popped-out tab has no prompt boxes, so it applies its result by broadcasting to this one.
  useEffect(() => {
    if (typeof window.BroadcastChannel !== "function") return undefined;
    const channel = new BroadcastChannel(ASSISTANT_CHANNEL_NAME);
    channel.onmessage = (event) => {
      if (event.data?.type !== ASSISTANT_APPLY_PROMPT) return;
      applyAssistantPrompt(event.data);
      setAppNotice({ message: "已应用 AI 助手发送的提示词", error: false });
    };
    return () => channel.close();
  }, []);

  // Mirrors the prompt boxes for the standalone assistant page. Debounced because it runs on every
  // keystroke and the popped-out tab only needs the text when the user asks it to optimise.
  useEffect(() => {
    const timer = window.setTimeout(() => writePromptSnapshot(window.localStorage, { positive, negative }), 400);
    return () => window.clearTimeout(timer);
  }, [negative, positive]);

  const recordPromptSelection = (type, event) => {
    const node = event?.currentTarget || (type === "positive" ? positivePromptRef.current : negativePromptRef.current);
    const current = promptTextRevision.current[type];
    if (!node || node.value !== current.text || !Number.isInteger(node.selectionStart) || !Number.isInteger(node.selectionEnd)) {
      promptSelectionCache.current[type] = null;
      return;
    }
    promptSelectionCache.current[type] = {
      start: node.selectionStart,
      end: node.selectionEnd,
      revision: current.revision,
    };
  };

  const changePromptText = (type, event) => {
    const value = event.currentTarget.value;
    const revision = promptTextRevision.current[type].revision + 1;
    promptTextRevision.current[type] = { text: value, revision };
    promptSelectionCache.current[type] = {
      start: event.currentTarget.selectionStart,
      end: event.currentTarget.selectionEnd,
      revision,
    };
    if (type === "positive") setPositive(value);
    else setNegative(value);
  };

  const applyPreset = (requestedRecord) => {
    if (status === "running" || promptPresetLibraryError) return;
    const record = promptPresets.records.find((candidate) => candidate.id === requestedRecord?.id);
    if (!record) return;
    const type = record.type;
    const state = promptTextRevision.current[type];
    const node = type === "positive" ? positivePromptRef.current : negativePromptRef.current;
    const liveSelection = document.activeElement === node && node?.value === state.text
      ? { start: node.selectionStart, end: node.selectionEnd }
      : null;
    const cached = promptSelectionCache.current[type];
    const cachedSelection = cached?.revision === state.revision ? { start: cached.start, end: cached.end } : null;
    const selection = liveSelection || cachedSelection;
    const result = insertPromptPreset(state.text, record.content, record.position, selection);
    const revision = state.revision + 1;
    promptTextRevision.current[type] = { text: result.text, revision };
    promptSelectionCache.current[type] = { start: result.caret, end: result.caret, revision };
    const session = ++promptFocusSession.current;
    promptFocusRequest.current = { type, text: result.text, caret: result.caret, session };
    if (type === "positive") setPositive((current) => current === state.text ? result.text : insertPromptPreset(current, record.content, record.position).text);
    else setNegative((current) => current === state.text ? result.text : insertPromptPreset(current, record.content, record.position).text);
  };

  const openPromptPresetDialog = (type, record = null) => {
    if (status === "running" || promptPresetLibraryError) return;
    setPromptPresetDialog({ type, record });
  };

  const savePromptPreset = (draft) => {
    if (status === "running" || promptPresetLibraryError) return "生成任务运行期间不能保存 Prompt 预设";
    try {
      const result = promptPresetDialog?.record
        ? updatePromptPreset(promptPresets, promptPresetDialog.record.id, draft)
        : createPromptPreset(promptPresets, draft);
      if (!result.validation.valid) return Object.values(result.validation.errors)[0] || "Prompt 预设格式无效";
      setPromptPresets(result.container);
      setShouldPersistPromptPresets(true);
      setPromptPresetDialog(null);
      return "";
    } catch (error) {
      return error.message || "无法保存 Prompt 预设";
    }
  };

  const requestDeletePromptPreset = (record) => {
    if (status === "running" || promptPresetLibraryError) return;
    setPromptPresetDelete(record);
  };

  const confirmDeletePromptPreset = () => {
    if (status === "running" || promptPresetLibraryError || !promptPresetDelete) return;
    setPromptPresets((current) => deletePromptPreset(current, promptPresetDelete.id));
    setShouldPersistPromptPresets(true);
    setPromptPresetDelete(null);
  };

  const updateADetailer = (updates) => setADetailer((current) => ({ ...current, ...updates }));
  // Units are addressed by id rather than index: removing one shifts every index
  // after it, and a half-typed number field would then belong to a different unit.
  const updateADetailerUnit = (id, updates) => setADetailer((current) => ({
    ...current,
    units: current.units.map((unit) => (unit.id === id ? { ...unit, ...updates } : unit)),
  }));
  const updateHires = (updates) => setHires((current) => ({ ...current, ...updates }));
  const updateRtx = (updates) => setRtx((current) => ({ ...current, ...updates }));
  const movePostprocessStage = (stage, direction) => {
    if (status === "running") return;
    setPostprocessOrder((current) => {
      const normalized = normalizePostprocessOrder(current);
      const from = normalized.indexOf(stage);
      const to = from + direction;
      if (from < 0 || to < 0 || to >= normalized.length) return normalized;
      const next = [...normalized];
      [next[from], next[to]] = [next[to], next[from]];
      return next;
    });
  };
  const updateOverallTheme = (overall) => setTheme((current) => ({ ...current, overall, ...deriveLogoTheme(overall) }));
  const updateLogoTheme = (key, value) => setTheme((current) => ({ ...current, [key]: value }));
  const resetTheme = () => setTheme({ ...DEFAULT_THEME });
  const toggleLightTheme = () => setTheme((current) => ({ ...current, light: !current.light }));
  const resizeConsole = (event) => {
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    const startY = event.clientY;
    const startHeight = consoleHeight;
    const move = (moveEvent) => setConsoleHeight(Math.max(180, Math.min(Math.floor(window.innerHeight * 2 / 3), startHeight + startY - moveEvent.clientY)));
    const stop = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop, { once: true });
  };
  const submitConsoleCommand = async (event) => {
    event.preventDefault();
    const command = consoleCommand.trim();
    if (!command || consoleRunning) return;
    setConsoleError("");
    try {
      const response = await fetch("/api/console/commands", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ command }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "无法执行控制台命令");
      setConsoleCommand("");
      setConsoleRunning(true);
    } catch (error) {
      setConsoleError(error.message);
    }
  };
  const clearConsole = () => {
    setConsoleEntries([]);
    setConsoleError("");
  };
  const logClientGenerationFailure = (stage, message) => {
    void fetch("/api/logs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        stage,
        message,
        details: {
          engine: model,
          checkpoint,
          diffusion_model: diffusionModel,
          text_encoder: textEncoder,
          vae,
          width: size.width,
          height: size.height,
          steps,
          cfg,
          denoise,
          sampler,
          scheduler,
          guidance,
          pag,
          lora_count: loras.filter((lora) => lora.enabled !== false).length,
          hires_enabled: hires.enabled,
          adetailer_enabled: adetailer.enabled,
          adetailer_units: activeADetailerUnits(adetailer).length,
          rtx_enabled: rtx.enabled,
          rtx_scale: rtx.scale,
          rtx_quality: rtx.quality,
          postprocess_order: normalizePostprocessOrder(postprocessOrder),
        },
      }),
    }).catch(() => {});
  };
  const applyPerformancePayload = (payload) => {
    setPerformanceConfig({ ...DEFAULT_PERFORMANCE, ...payload.settings });
    setPerformanceActive(payload.active || {});
    setPerformanceCapabilities(payload.capabilities || {});
  };
  const refreshPerformanceSettings = async () => {
    setPerformanceLoading(true);
    setPerformanceError("");
    try {
      const response = await fetch("/api/inference/performance", { cache: "no-store" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "无法读取性能配置");
      applyPerformancePayload(payload);
    } catch (error) {
      setPerformanceError(error.message);
    } finally {
      setPerformanceLoading(false);
    }
  };
  const savePerformanceSettings = async () => {
    if (performanceSaving || status === "running") return;
    setPerformanceSaving(true);
    setPerformanceError("");
    try {
      const response = await fetch("/api/inference/performance", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(performanceConfig),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "无法保存性能配置");
      applyPerformancePayload(payload);
      if (payload.settings?.memory_mode === "ultra_low_vram") {
        setProcessPreview(false);
        setHires((current) => ({ ...current, enabled: false }));
        setADetailer((current) => ({ ...current, enabled: false }));
        setRtx((current) => ({ ...current, enabled: false }));
        commitMountedLoras(() => []);
      }
      setInferenceHealth((current) => current ? {
        ...current,
        model_cached: payload.active?.model_cached === true,
        model_resident: false,
        loaded_checkpoint: null,
        loaded_checkpoint_path: null,
        loaded_engine: null,
        memory_mode: payload.active?.memory_mode || null,
        memory_label: null,
        memory_reason: null,
        offload_mode: null,
        attention_backend: payload.active?.attention_backend || "none",
        compute_dtype: payload.active?.compute_dtype || "none",
        vae_mode: payload.active?.vae_mode || "none",
        memory_mode_request: payload.settings?.memory_mode || "auto",
        performance_settings: payload.settings,
      } : current);
      setGenerationWarning("");
    } catch (error) {
      setPerformanceError(error.message);
    } finally {
      setPerformanceSaving(false);
    }
  };
  const refreshDiagnosticLogs = async () => {
    setDiagnosticLogsLoading(true);
    setDiagnosticLogsError("");
    try {
      const response = await fetch("/api/logs", { cache: "no-store" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "无法读取日志目录");
      const logs = payload.logs || [];
      setDiagnosticLogs(logs);
      if (selectedDiagnosticLog && !logs.some((log) => log.name === selectedDiagnosticLog)) {
        setSelectedDiagnosticLog("");
        setDiagnosticLogContent("");
      }
    } catch (error) {
      setDiagnosticLogsError(error.message);
    } finally {
      setDiagnosticLogsLoading(false);
    }
  };
  const selectDiagnosticLog = async (name) => {
    if (diagnosticLogReading || name === selectedDiagnosticLog) return;
    setSelectedDiagnosticLog(name);
    setDiagnosticLogContent("");
    setDiagnosticLogReading(true);
    setDiagnosticLogsError("");
    try {
      const response = await fetch(`/api/logs?name=${encodeURIComponent(name)}`, { cache: "no-store" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "无法读取日志文件");
      setDiagnosticLogContent(payload.content || "");
    } catch (error) {
      setDiagnosticLogsError(error.message);
    } finally {
      setDiagnosticLogReading(false);
    }
  };
  const refreshPlugins = async () => {
    setPluginsLoading(true);
    setPluginsError("");
    try {
      const response = await fetch("/api/plugins", { cache: "no-store" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "无法读取插件目录");
      setPluginRegistry(payload);
    } catch (error) {
      setPluginsError(error.message);
    } finally {
      setPluginsLoading(false);
    }
  };
  const runPluginAction = async (id, { request, failure, adoptSnapshot = true }) => {
    if (pluginPendingId) return;
    setPluginPendingId(id);
    setPluginsError("");
    try {
      const response = await request();
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || failure);
      // Mutating routes answer with a fresh snapshot, so the list never drifts from the server.
      if (adoptSnapshot) setPluginRegistry(payload);
    } catch (error) {
      setPluginsError(error.message);
    } finally {
      setPluginPendingId("");
    }
  };
  const setPluginEnabled = (id, enabled) => runPluginAction(id, {
    failure: "无法保存插件启用状态",
    request: () => fetch(`/api/plugins/${encodeURIComponent(id)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled }),
    }),
  });
  const revealPluginFolder = (id) => runPluginAction(id, {
    failure: "无法打开插件文件夹",
    adoptSnapshot: false,
    request: () => fetch(`/api/plugins/${encodeURIComponent(id)}/reveal`, { method: "POST" }),
  });
  const removePlugin = (plugin) => {
    // Deleting a folder is irreversible, so it never happens without an explicit confirmation.
    if (!window.confirm(pluginRemoveConfirmation(plugin))) return;
    return runPluginAction(plugin.id, {
      failure: "无法移除插件",
      request: () => fetch(`/api/plugins/${encodeURIComponent(plugin.id)}`, { method: "DELETE" }),
    });
  };
  const clearDiagnosticLogs = async () => {
    if (diagnosticLogsClearing || diagnosticLogs.length === 0) return;
    if (!window.confirm(`确定清空 logs 目录中的 ${diagnosticLogs.length} 个日志文件吗？此操作不可恢复。`)) return;
    setDiagnosticLogsClearing(true);
    setDiagnosticLogsError("");
    try {
      const response = await fetch("/api/logs", { method: "DELETE" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "无法清空日志目录");
      setDiagnosticLogs([]);
      setSelectedDiagnosticLog("");
      setDiagnosticLogContent("");
    } catch (error) {
      setDiagnosticLogsError(error.message);
    } finally {
      setDiagnosticLogsClearing(false);
    }
  };

  const downloadRecommendedYoloModels = async () => {
    if (status === "running" || adetailerDownload?.status === "downloading") return;
    setADetailerDownload({ label: "基础推荐模型", status: "downloading", modelIndex: 0, totalModels: adetailerInfo.builtins.length, currentBytes: 0, totalBytes: 0, speedBps: 0, route: "正在测速", connections: 0, message: "正在测速下载线路..." });
    let completed = false;
    try {
      const response = await fetch("/api/yolo/download", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.error || "无法启动 YOLO 模型下载");
      }
      if (!response.body) throw new Error("浏览器不支持下载进度流");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let pending = "";
      while (true) {
        const { done, value } = await reader.read();
        pending += decoder.decode(value || new Uint8Array(), { stream: !done });
        const lines = pending.split("\n");
        pending = lines.pop() || "";
        for (const line of lines) {
          if (!line.trim()) continue;
          const event = JSON.parse(line);
          if (event.type === "starting") {
            setADetailerDownload((current) => ({ ...current, totalModels: event.total_models || current?.totalModels || 5, message: "正在测速下载线路..." }));
          } else if (event.type === "model") {
            setADetailerDownload((current) => ({ ...current, name: event.name, modelLabel: event.label, modelIndex: event.index, totalModels: event.total_models, currentBytes: 0, totalBytes: 0, speedBps: 0, route: "正在测速", connections: 0, message: `正在准备 ${event.index} / ${event.total_models}` }));
          } else if (event.type === "route") {
            setADetailerDownload((current) => ({ ...current, route: event.label, connections: event.connections || 0, message: event.cached ? `模型 ${event.model_index} / ${current?.totalModels || 5} 已在本地` : `模型 ${event.model_index} / ${current?.totalModels || 5} · 已选择 ${event.label}` }));
          } else if (event.type === "progress") {
            setADetailerDownload((current) => ({
              ...current,
              currentBytes: event.current_bytes || 0,
              totalBytes: event.total_bytes || 0,
              speedBps: event.speed_bps || 0,
              route: event.route || current?.route,
              connections: event.connections || 0,
              message: event.cached ? "本地模型已可用" : "正在下载...",
            }));
          } else if (event.type === "warning") {
            setADetailerDownload((current) => ({ ...current, message: event.message }));
          } else if (event.type === "complete") {
            completed = true;
            setADetailerDownload((current) => ({ ...current, status: "complete", modelIndex: event.total_models, totalModels: event.total_models, message: event.cached_models === event.total_models ? "5 个模型均已在本地" : "5 个基础推荐模型下载完成" }));
          } else if (event.type === "error") {
            setADetailerDownload((current) => ({ ...current, status: "error", error: event.error, message: event.error }));
          }
        }
        if (done) break;
      }
      if (completed) await refreshADetailerModels();
    } catch (error) {
      setADetailerDownload((current) => ({ ...current, status: "error", error: error.message, message: error.message }));
    }
  };

  const downloadBackgroundRemovalModel = async (modelId) => {
    if (status === "running" || backgroundRemovalDownload?.active) return;
    const selected = inferenceHealth?.background_removal?.models?.find((item) => item.id === modelId);
    if (!selected || selected.local) return;
    setBackgroundRemovalModel(modelId);
    setBackgroundRemovalDownload({ status: "queued", active: true, modelId, label: selected.label, currentBytes: 0, totalBytes: selected.size || 0, speedBps: 0, route: "正在测速", connections: 8, message: `正在准备 ${selected.label}` });
    try {
      const response = await fetch("/api/background-removal/download", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: modelId }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "无法启动透明背景模型下载");
      setBackgroundRemovalDownload(payload.job);
    } catch (error) {
      setBackgroundRemovalDownload((current) => ({ ...current, status: "error", message: error.message }));
    }
  };

  const toggleLora = (lora, category) => {
    let next = null;
    commitMountedLoras((current) => {
      next = current.some((item) => item.value === lora.value)
        ? current.filter((item) => item.value !== lora.value)
        // A collecting group claims new mounts, so picking LoRAs from a category
        // builds that combination directly instead of landing them unsorted.
        : current.length >= 16 ? current : [...current, mountedEntryForGroups({ ...lora, category: category.label, weight: 1, precision: 1, enabled: true }, activeLoraGroupsRef.current)];
      return next;
    });
    // Mounting from the library bypasses the mount panel, so the write-back has
    // to happen here too or a collected LoRA would never reach its definition.
    if (next) commitLoraGroups((current) => syncMountedIntoGroups(current, next));
  };

  const lookupLora = async (lora, refresh = false, categoryId = "") => {
    if (loraLookups[lora.value]?.loading) return;
    const requestModel = model;
    const controller = new AbortController();
    loraLookupControllers.current.add(controller);
    setLoraLookups((current) => ({ ...current, [lora.value]: { loading: true, error: "" } }));
    try {
      const response = await fetch("/api/lora-lookup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ engine: model, path: lora.value, refresh, category: categoryId }),
        signal: controller.signal,
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "查询失败");
      if (controller.signal.aborted || currentModel.current !== requestModel) return;
      setLoraLibrary((current) => current.map((category) => ({
        ...category,
        models: category.models.map((item) => item.value === lora.value ? { ...item, metadata: payload.metadata } : item),
      })));
      setLoraLookups((current) => ({ ...current, [lora.value]: { loading: false, error: payload.metadata.status === "not_found" ? "未找到来源" : "" } }));
      return payload.metadata;
    } catch (error) {
      if (error.name !== "AbortError" && currentModel.current === requestModel) {
        setLoraLookups((current) => ({ ...current, [lora.value]: { loading: false, error: error.message } }));
      }
    } finally {
      loraLookupControllers.current.delete(controller);
    }
    return null;
  };

  const openLoraDetails = (item, categoryId, categoryLabel) => {
    setLoraDetail({ value: item.value, name: item.name, categoryId, categoryLabel });
    if (item.metadata?.detailSchema !== 1 || item.metadata?.triggerReviewSchema !== 4) void lookupLora(item, false, categoryId);
  };

  useEffect(() => {
    const handleShortcut = (event) => {
      if (imageViewerOpen && !activeCollage && activeViewerLayerItem && !isEditableTarget(event.target)) {
        const adjustment = { ArrowUp: [0, -1], ArrowDown: [0, 1], ArrowLeft: [-1, 0], ArrowRight: [1, 0] }[event.key];
        if (adjustment) {
          event.preventDefault();
          if (!viewerNudge.current || viewerNudge.current.key !== event.key || viewerNudge.current.id !== activeViewerLayerItem.id) {
            viewerNudge.current = { key: event.key, id: activeViewerLayerItem.id };
            saveViewerUndo({ layers: viewerLayers, snappedLayers: viewerSnappedLayers, activeLayer: activeViewerLayer });
          }
          setViewerLayers((current) => current.map((layer) => layer.id === activeViewerLayerItem.id
            ? { ...layer, x: layer.x + adjustment[0], y: layer.y + adjustment[1] }
            : layer));
          return;
        }
      }
      if (!(event.ctrlKey || event.metaKey)) return;
      if (imageViewerOpen && !isEditableTarget(event.target)) {
        if (event.key.toLowerCase() === "z") {
          if (event.repeat) return;
          event.preventDefault();
          undoViewerChange();
          return;
        }
        if (event.key.toLowerCase() === "c" && activeViewerLayerItem && !activeCollage) {
          event.preventDefault();
          copyActiveViewerLayer();
          return;
        }
      }
      if (event.key.toLowerCase() === "k") {
        event.preventDefault();
        setAssistantOpen(true);
      } else if (event.key === "Enter") {
        event.preventDefault();
        // The run shortcut belongs to whichever page is on screen. On 图生图 it must start that
        // page's run rather than silently launching a text-to-image one with the composer's
        // prompt, and on a page with no run at all it does nothing.
        if (activePage === "image") void generateFromImage();
        else if (activePage === "generate") void generate();
      }
    };
    const releaseViewerNudge = (event) => {
      if (!event || viewerNudge.current?.key === event.key) viewerNudge.current = null;
    };
    window.addEventListener("keydown", handleShortcut);
    window.addEventListener("keyup", releaseViewerNudge);
    window.addEventListener("blur", releaseViewerNudge);
    window.addEventListener("paste", pasteViewerLayer);
    return () => {
      window.removeEventListener("keydown", handleShortcut);
      window.removeEventListener("keyup", releaseViewerNudge);
      window.removeEventListener("blur", releaseViewerNudge);
      window.removeEventListener("paste", pasteViewerLayer);
    };
  });

  const activeLoraCategory = loraCategory === "mounted" || loraCategory === "groups" ? { id: loraCategory, label: loraCategory === "groups" ? "组合" : "已挂载", directory: "", models: [] } : loraLibrary.find((category) => category.id === loraCategory);
  // A mounted entry carries no metadata of its own; the preview belongs to the
  // library listing, so an asset whose source was never looked up simply has no
  // image and the row says so rather than showing a broken frame.
  const loraPreviewUrlFor = (item) => {
    const metadata = loraLibrary.flatMap((category) => category.models).find((entry) => entry.value === item.value)?.metadata;
    return metadata?.status === "found"
      ? `/api/lora-preview?engine=${encodeURIComponent(model)}&path=${encodeURIComponent(item.value)}&v=${encodeURIComponent(metadata.queriedAt)}`
      : "";
  };
  const matchesLoraSearch = (item) => item.name.toLowerCase().includes(loraSearch.trim().toLowerCase());
  const visibleLoras = loraCategory === "mounted" ? [] : activeLoraCategory?.models.filter(matchesLoraSearch) ?? [];
  const displayedGenerationProgress = progress;
  const selectedOutput = generatedOutputs[selectedOutputIndex] || null;
  const transparentPromptEnabled = TRANSPARENT_BACKGROUND_PATTERN.test(positive);
  const transparentPromptSubject = positive.replace(TRANSPARENT_BACKGROUND_PATTERN_ALL, "").replace(/^\s*,\s*|\s*,\s*$/g, "").trim();
  const backgroundRemovalModels = (inferenceHealth?.background_removal?.models || []).filter((item) => item.selectable !== false);
  const recommendedBackgroundRemovalModels = backgroundRemovalModels.filter((item) => !item.local);
  const localBackgroundRemovalModels = backgroundRemovalModels.filter((item) => item.local);
  const selectedBackgroundRemovalModel = backgroundRemovalModels.find((item) => item.id === backgroundRemovalModel) || null;
  const selectedBackgroundRemovalReady = Boolean(selectedBackgroundRemovalModel?.installed && inferenceHealth?.background_removal?.runtime_available);
  const backgroundRemovalDownloadForSelection = backgroundRemovalDownload?.modelId === backgroundRemovalModel ? backgroundRemovalDownload : null;
  const backgroundRemovalProgress = backgroundRemovalDownloadForSelection?.totalBytes > 0
    ? Math.min(100, backgroundRemovalDownloadForSelection.currentBytes / backgroundRemovalDownloadForSelection.totalBytes * 100)
    : 0;
  const toggleTransparentBackground = () => setPositive((current) => TRANSPARENT_BACKGROUND_PATTERN.test(current)
    ? current.replace(TRANSPARENT_BACKGROUND_PATTERN_ALL, "").replace(/\s*,\s*,+/g, ", ").replace(/^\s*,\s*|\s*,\s*$/g, "").trim()
    : `${current.trim()}${current.trim() ? ", " : ""}${TRANSPARENT_BACKGROUND_TAG}`);
  const activeViewerLayerItem = viewerLayers.find((layer) => layer.id === activeViewerLayer) || viewerLayers.at(-1) || null;
  const activeCollageTemplate = collageTemplates.find((template) => template.id === activeCollage?.templateId) || null;
  const activeCollageLayout = activeCollageTemplate && activeCollage
    ? adaptiveCollageLayout(activeCollageTemplate, activeCollage.slots.map((entry) => entry?.asset))
    : null;
  const activeCollageEdges = activeCollageLayout
    ? sharedEdgeHiddenSides(activeCollageLayout.slots
      .map((slot, index) => activeCollage.slots[index] && ({ id: String(index), left: slot.x, right: slot.x + slot.w, top: slot.y, bottom: slot.y + slot.h }))
      .filter(Boolean), .0001)
    : {};
  const activeCollageSlotItem = activeCollage?.slots[activeCollageSlot] || null;
  const outputBatches = [...new Set(generatedOutputs.map((output) => output.batch_index))];
  const selectedBatchOutputs = generatedOutputs.filter((output) => output.batch_index === selectedOutput?.batch_index);
  const activeDetail = generationDetail?.detections?.find((item) => item.status === "active")
    || generationDetail?.detections?.[Math.max(0, (generationDetail?.region_index || 1) - 1)];
  const stageProgress = generationStageTotal > 0 ? Math.min(100, Math.round(generationStageStep / generationStageTotal * 100)) : 0;
  const stageLabel = generationStage === "adetailer_detect"
    ? "YOLO DETECTION"
    : generationStage === "background_remove"
      ? "BACKGROUND REMOVAL"
      : generationStage === "adetailer_inpaint"
      ? "ADETAILER INPAINT"
      : generationStage === "rtx_upscale"
        ? "RTX VSR UPSCALE"
      : generationStage === "hires_upscale"
        ? "HIRES UPSCALE"
        : generationStage === "hires_sampling"
          ? "HIRES DIFFUSION"
      : generationStage === "base_sampling"
        ? "SAMPLING"
        : generationStage === "postprocess_source"
          ? "SOURCE READY"
        : generationStage === "model_load"
          ? "MODEL LOADING"
          : generationStage === "prompt_encode"
            ? "PROMPT ENCODING"
            : generationStage === "save"
              ? "SAVING"
              : generationStage === "sampler_offload"
                ? "MODEL TO RAM"
                : generationStage === "vae_decode"
                  ? "VAE DECODING"
                  : generationStage === "model_restore"
                    ? "MODEL RESTORE"
                    : generationStage.toUpperCase();
  const canvasSizeValid = size.width >= 64 && size.height >= 64
    && size.width <= 2048 && size.height <= 2048
    && size.width % 64 === 0 && size.height % 64 === 0;
  const isAnima = model === "Anima";
  const isFlux = model === "Flux";
  const isFlux2 = model === "Flux2";
  const isKrea2 = model === "Krea2";
  // Both Flux generations are guidance distilled, and the label is what the notes below name.
  const isDistilledGuidance = DISTILLED_GUIDANCE_ENGINES.includes(model);
  const distilledEngineLabel = isFlux2 ? "FLUX.2" : "FLUX.1";
  // "Split model" is the shape of the picker — several component files instead of one checkpoint —
  // and it is what most of the engine-conditional UI below actually branches on.
  const isSplitModel = SPLIT_MODEL_ENGINES.includes(model);
  const engineLabel = model === "SD" ? "Stable Diffusion" : model === "iL" ? "Illustrious" : model;
  const selectedEngineHealth = inferenceHealth?.engines?.[model] || {};
  const selectedEngineFeatures = selectedEngineHealth.features || {};
  const nativeSamplerNames = isFlux ? FLUX_SAMPLERS : isFlux2 ? FLUX2_SAMPLERS : isKrea2 ? KREA2_SAMPLERS : ANIMA_SAMPLERS;
  const nativeSchedulerNames = isFlux ? FLUX_SCHEDULERS : isFlux2 ? FLUX2_SCHEDULERS : isKrea2 ? KREA2_SCHEDULERS : ANIMA_SCHEDULERS;
  const activeSamplerNames = isSplitModel ? (selectedEngineHealth.samplers || nativeSamplerNames).filter((name) => nativeSamplerNames.includes(name)) : samplerNames;
  const activeSchedulerNames = isSplitModel ? (selectedEngineHealth.schedulers || nativeSchedulerNames).filter((name) => nativeSchedulerNames.includes(name)) : schedulerNames;
  const engineAllowsLora = isSplitModel ? selectedEngineFeatures.lora === true : selectedEngineFeatures.lora !== false;
  const engineAllowsHires = isSplitModel ? selectedEngineFeatures.hires === true : selectedEngineFeatures.hires !== false;
  const engineAllowsADetailer = isSplitModel ? selectedEngineFeatures.adetailer === true : selectedEngineFeatures.adetailer !== false;
  const engineAllowsRtx = isSplitModel ? selectedEngineFeatures.rtx === true : selectedEngineFeatures.rtx !== false;
  const engineAllowsProcessPreview = !isSplitModel && selectedEngineFeatures.process_preview !== false;
  // Both Flux generations steer with a distilled guidance embedding instead of classifier-free
  // guidance, so the CFG control means something different and the negative prompt has nothing to
  // encode into.
  const engineAllowsNegativePrompt = !isDistilledGuidance;
  const ultraLowMode = inferenceHealth?.performance_settings?.memory_mode === "ultra_low_vram";
  const guidanceFlowCompatible = inferenceHealth?.guidance?.cfg_zero_star?.available === true
    && (inferenceHealth.guidance.cfg_zero_star.engines || []).includes(model);
  const guidancePagCompatible = pagAvailableForEngine(inferenceHealth, model);
  const guidanceReady = guidance === "pag" ? guidancePagCompatible : guidance !== "cfg_zero_star" || guidanceFlowCompatible;
  const enabledLoras = loras.filter((lora) => lora.enabled !== false);
  const loraReady = enabledLoras.length === 0 || engineAllowsLora;
  const availableLoraValues = new Set(loraLibrary.flatMap((category) => category.models.map((item) => item.value)));
  const loraResourcesReady = enabledLoras.length === 0 || (!loraLoading && !loraError && enabledLoras.every((lora) => availableLoraValues.has(lora.value)));
  const adetailerUnits = adetailer.units;
  const adetailerActiveUnits = activeADetailerUnits(adetailer);
  const adetailerEffectiveSteps = Math.min(...adetailerActiveUnits.map((unit) => adetailerUnitSteps(unit, steps, model)), Infinity);
  const adetailerIssue = adetailer.enabled
    ? adetailerStageIssue(adetailer, steps, (detector) => adetailerModels.some((item) => item.value === detector), model)
    : "";
  const adetailerReady = !adetailer.enabled || (adetailerInfo.available && !adetailerIssue);
  const adetailerLocked = status === "running";
  const upscalerModels = inferenceHealth?.upscalers?.models || [];
  const compatibleUpscalers = upscalerModels.filter((item) => item.compatible);
  const selectedUpscaler = upscalerModels.find((item) => item.id === hires.model) || null;
  const hiresEffectiveSteps = hiresEffectiveStepCount(hires, model);
  const hiresConfigurationReady = hiresEffectiveSteps >= 1 && hires.tileOverlap <= Math.floor(hires.tileSize / 2);
  const hiresSeedReady = hires.seedMode !== "fixed" || normalizeUint64Seed(hires.seed) !== null;
  const hiresReady = !hires.enabled || Boolean(inferenceHealth?.upscalers?.runtime_available && selectedUpscaler?.compatible && hiresConfigurationReady && hiresSeedReady);
  const hiresControlsLocked = !hires.enabled || status === "running";
  const normalizedPostprocessOrder = normalizePostprocessOrder(postprocessOrder);
  const postprocessSettings = { hires, adetailer, rtx };
  const postprocessDimensions = postprocessTargetSize(size, normalizedPostprocessOrder, postprocessSettings);
  const hiresPreviewDimensions = postprocessTargetSize(size, normalizedPostprocessOrder, {
    ...postprocessSettings,
    hires: { ...hires, enabled: true },
  });
  const hiresStageTrace = hiresPreviewDimensions.trace.find((entry) => entry.stage === "hires");
  const hiresTargetSize = hiresStageTrace?.output || size;
  const rtxPreviewDimensions = postprocessTargetSize(size, normalizedPostprocessOrder, {
    ...postprocessSettings,
    rtx: { ...rtx, enabled: true },
  });
  const rtxStageTrace = rtxPreviewDimensions.trace.find((entry) => entry.stage === "rtx");
  const rtxHealth = inferenceHealth?.rtx_vsr || {};
  const rtxReady = !rtx.enabled || rtxHealth.available === true;
  const rtxStatusReason = rtxHealth.probing
    ? "正在探测 NVIDIA RTX VSR 运行时，请稍候"
    : rtxHealth.reason || (rtxHealth.available ? "RTX VSR 已就绪" : "RTX VSR 运行时不可用");
  const animaRuntimeHealth = selectedEngineHealth.runtime || {};
  const animaTokenizerMissing = isAnima ? [
    !(animaRuntimeHealth.qwen_tokenizer?.installed ?? animaRuntime?.qwen_tokenizer?.installed) && "Qwen Tokenizer",
    !(animaRuntimeHealth.qwen_tokenizer_config?.installed ?? animaRuntime?.qwen_tokenizer_config?.installed) && "Qwen Tokenizer Config",
    !(animaRuntimeHealth.t5_tokenizer?.installed ?? animaRuntime?.t5_tokenizer?.installed) && "T5 Tokenizer",
  ].filter(Boolean) : isFlux ? [
    // FLUX.1 reads the same bundled T5 tokenizer; its CLIP-L tokenizer comes from the Stable
    // Diffusion runtime config, which only the backend can see, so that half is reported through
    // the engine's own health rather than the model catalogue.
    !(animaRuntimeHealth.t5_tokenizer?.installed ?? animaRuntime?.t5_tokenizer?.installed) && "T5 Tokenizer",
    animaRuntimeHealth.clip_tokenizer?.installed === false && "CLIP-L Tokenizer",
  ].filter(Boolean) : isFlux2 || isKrea2 ? [
    // [klein] reads the same bundled Qwen tokenizer Anima does, and so does Krea 2's Qwen3-VL —
    // ComfyUI points every Qwen3 encoder at the Qwen2.5 table. [dev]'s Mistral tokenizer is not an
    // installed resource at all, so there is nothing here that can be missing for it.
    !(animaRuntimeHealth.qwen_tokenizer?.installed ?? animaRuntime?.qwen_tokenizer?.installed) && "Qwen Tokenizer",
  ].filter(Boolean) : [];
  const animaSettingsUnsupported = isSplitModel && (
    !activeSamplerNames.includes(sampler)
    || !activeSchedulerNames.includes(scheduler)
    || processPreview
    || (isDistilledGuidance && guidance !== "none")
  );
  const splitModelAssetsReady = isFlux
    ? Boolean(diffusionModel && !diffusionModelMissing && textEncoder && !textEncoderMissing
      && textEncoder2 && !textEncoder2Missing && vae && !vaeMissing)
    : Boolean(diffusionModel && !diffusionModelMissing && textEncoder && !textEncoderMissing && vae && !vaeMissing);
  const pipelineConfigReady = isSplitModel
    ? selectedEngineHealth.available === true
    : inferenceHealth?.pipeline_configs?.[model === "iL" ? "sdxl" : "sd"] === true;
  // The part of the generate page's readiness chain that image-to-image shares: a usable engine and
  // model files on disk. Everything after that in `generationDisabledReason` is about the
  // text-to-image composer — its prompt, its canvas, its post-processing stages — and the
  // image-to-image page owns its own equivalents.
  const imageEngineReady = !modelSwitching && !modelLoading && !modelError && loraResourcesReady && (isSplitModel
    ? Boolean(splitModelAssetsReady && !animaTokenizerMissing.length && !ultraLowMode && loraReady)
    : Boolean(checkpoint && !checkpointMissing));
  const generationDisabledReason = status === "running" ? "正在生成"
    : modelSwitching ? "正在切换或释放模型"
      : modelLoading ? "正在读取模型目录"
        : modelError ? `模型目录不可用：${modelError}`
          : isSplitModel && !diffusionModel ? `请先选择 ${engineLabel} 扩散模型`
          : isSplitModel && diffusionModelMissing ? `当前 ${engineLabel} 扩散模型文件已不存在`
            : isSplitModel && !textEncoder ? `请先选择 ${engineLabel} ${isFlux ? "CLIP-L " : ""}文本编码器`
              : isSplitModel && textEncoderMissing ? `当前 ${engineLabel} ${isFlux ? "CLIP-L " : ""}文本编码器文件已不存在`
                : isFlux && !textEncoder2 ? "请先选择 Flux T5-XXL 文本编码器"
                  : isFlux && textEncoder2Missing ? "当前 Flux T5-XXL 文本编码器文件已不存在"
                : isSplitModel && !vae ? `请先选择 ${engineLabel} VAE`
                  : isSplitModel && vaeMissing ? `当前 ${engineLabel} VAE 文件已不存在`
                     : isSplitModel && animaTokenizerMissing.length ? `${animaTokenizerMissing.join(" / ")} 运行资源缺失或损坏，请修复或重新运行环境配置器`
                       : isSplitModel && ultraLowMode ? `${engineLabel} 原生运行时暂不支持极限省存模式`
                        : isSplitModel && animaSettingsUnsupported ? `当前 ${engineLabel} 参数包含不受支持的采样、引导或过程预览设置`
                         : !loraReady ? `当前推理服务未声明 ${engineLabel} LoRA 能力`
                           : !loraResourcesReady ? "已启用的 LoRA 文件未安装、已移动或目录尚未就绪"
                             : hires.enabled && !engineAllowsHires ? `当前推理服务未声明 ${engineLabel} Hires.fix 能力`
                             : adetailer.enabled && !engineAllowsADetailer ? `当前推理服务未声明 ${engineLabel} ADetailer 能力`
                               : rtx.enabled && !engineAllowsRtx ? `当前推理服务未声明 ${engineLabel} RTX VSR 能力`
                         : !isSplitModel && !checkpoint ? "请先选择可用底模"
                          : !isSplitModel && checkpointMissing ? "当前底模文件已不存在"
      : !positive.trim() ? "请输入正向提示词"
          : transparentPromptEnabled && !transparentPromptSubject ? "透明背景标签之外还需要填写主体提示词"
          : transparentPromptEnabled && !selectedBackgroundRemovalModel ? "请选择透明背景模型"
          : transparentPromptEnabled && !selectedBackgroundRemovalModel?.installed ? "所选透明背景模型尚未下载"
          : transparentPromptEnabled && !inferenceHealth?.background_removal?.runtime_available ? "ONNX Runtime 尚未就绪，请重新运行环境配置器"
          : !canvasSizeValid ? "画布尺寸必须为 64 的倍数且不超过 2048"
            : ultraLowMode && (hires.enabled || adetailer.enabled || rtx.enabled) ? "极限省存模式正在关闭附加后处理阶段"
              : ultraLowMode && guidance !== "none" ? "极限省存模式不支持引导增强"
                : !guidanceReady ? `当前 ${model} 推理运行时未声明 ${guidance === "pag" ? "PAG" : "CFG-Zero*"} 可用`
               : !hiresReady ? !inferenceHealth?.upscalers?.runtime_available ? "Hires.fix 运行环境尚未配置，请重新运行环境配置器" : !selectedUpscaler?.compatible ? "Hires.fix 已启用，但没有可用的兼容超分模型" : !hiresSeedReady ? "固定 Hires Seed 必须是 0 到 18446744073709551615" : hiresEffectiveSteps < 1 ? "Hires 步数乘以重绘强度至少需要产生 1 个有效步骤" : "Hires 分块重叠不能超过分块尺寸的一半"
              : !adetailerReady ? adetailerIssue || "ADetailer 已启用，但检测模型尚未就绪"
                : !rtxReady ? rtxStatusReason
                  : !postprocessDimensions.valid ? "RTX VSR 中间目标超过 8192 边长或 32 MP 安全限制"
                  : rtx.enabled && rtxHealth.probing ? rtxStatusReason
                    : inferenceHealth?.status !== "ready" ? inferenceHealth?.status === "error" || inferenceHealth?.status === "offline"
                ? `推理服务不可用：${inferenceHealth.error || "正在自动重启"}`
                : "推理服务正在启动"
                : !inferenceHealth?.cuda ? "CUDA 当前不可用"
                  : !pipelineConfigReady ? isSplitModel ? selectedEngineHealth.reason || `${engineLabel} 推理运行时尚未就绪` : "缺少当前底模的本地运行配置，请联网重新运行环境配置器" : "";
  const loaderProgress = (modelLoading ? 0 : 32)
    + (loraLoading ? 0 : 28)
    + (["ready", "error", "offline"].includes(inferenceHealth?.status) ? 40 : inferenceHealth ? 18 : 0);
  const loaderStatus = modelLoading
    ? "正在读取模型目录"
    : loraLoading
      ? "正在扫描 LoRA 资源"
      : inferenceHealth?.status === "ready"
        ? "工作区准备完成"
        : inferenceHealth?.status === "error" || inferenceHealth?.status === "offline"
          ? "工作区已加载，推理服务不可用"
           : "正在启动 CUDA 推理服务";
  const environmentItems = [
    ["GPU", inferenceHealth?.device || "未检测"],
    ["显存", inferenceHealth?.vram_bytes ? `${(inferenceHealth.vram_bytes / 1024 ** 3).toFixed(1)} GB` : "--"],
    ["Python", inferenceHealth?.python_version || "--"],
    ["PyTorch", inferenceHealth?.torch_version || "--"],
    ["CUDA Runtime", inferenceHealth?.cuda_runtime ? `CUDA ${inferenceHealth.cuda_runtime}` : "--"],
    ["cuDNN", inferenceHealth?.cudnn_version || "--"],
    ["Diffusers", inferenceHealth?.diffusers_version || "--"],
    ["Transformers", inferenceHealth?.transformers_version || "--"],
    ["xformers", inferenceHealth?.xformers_version || "未安装"],
    ["Node.js", inferenceHealth?.node_version || "--"],
    ["显存策略", inferenceHealth?.memory_mode_request?.toUpperCase() || "AUTO"],
    ["服务端口", `${inferenceHealth?.web_port || 7709} / ${inferenceHealth?.inference_port || 8718}`],
  ];
  const vramGb = (performanceCapabilities.vram_bytes || inferenceHealth?.vram_bytes || 0) / 1024 ** 3;
  const ramGb = (performanceCapabilities.ram_bytes || 0) / 1024 ** 3;
  const vramLimitInfo = performanceCapabilities.vram_limit || {};
  const vramLimitMinGb = Math.max(0.5, Number(vramLimitInfo.minimum_bytes || 0) / 1024 ** 3);
  const vramLimitMaxGb = Math.max(vramLimitMinGb, Number(vramLimitInfo.maximum_bytes || performanceCapabilities.vram_bytes || 0) / 1024 ** 3);
  const vramLimitAutomatic = Number(performanceConfig.vram_limit_gb || 0) <= 0;
  const vramLimitValue = vramLimitAutomatic
    ? vramLimitMaxGb
    : Math.min(vramLimitMaxGb, Math.max(vramLimitMinGb, Number(performanceConfig.vram_limit_gb)));
  const vramLimitPercent = vramLimitMaxGb > vramLimitMinGb
    ? ((vramLimitValue - vramLimitMinGb) / (vramLimitMaxGb - vramLimitMinGb)) * 100
    : 100;
  const recommendedPerformancePreset = vramGb >= 11.5 ? "speed" : vramGb >= 7.5 ? "large-model" : vramGb >= 5.5 ? "balanced" : ramGb >= 15 ? "ultra-low" : "low-memory";
  const activePerformancePreset = performancePresets.find((preset) => Object.entries(preset.settings)
    .filter(([key]) => !["allow_shared_memory", "calculate_model_hash", "vram_limit_gb"].includes(key))
    .every(([key, value]) => performanceConfig[key] === value))?.id;
  const pluginList = pluginRegistry?.plugins || [];
  const pluginRegistryDiagnostics = (pluginRegistry?.diagnostics || []).filter((diagnostic) => !diagnostic.id);
  const pluginSummary = pluginRegistrySummary(pluginRegistry);
  const configuredPerformanceLabel = (key) => performanceChoices[key]?.find(([value]) => value === performanceConfig[key])?.[1] || "自动";
  const actualPerformanceValue = (...values) => values.find((value) => value && String(value).toLowerCase() !== "none");
  const runtimeMemoryMode = actualPerformanceValue(performanceActive.memory_mode, inferenceHealth?.memory_mode);
  const runtimeAttentionBackend = actualPerformanceValue(performanceActive.attention_backend, inferenceHealth?.attention_backend);
  const runtimeComputeDtype = actualPerformanceValue(performanceActive.compute_dtype, inferenceHealth?.compute_dtype);
  const renderModalLoraCard = (item, categoryLabel) => {
    const mounted = loras.some((lora) => lora.value === item.value);
    const lookup = loraLookups[item.value];
    const metadata = item.metadata;
    const hasPreview = metadata?.status === "found";
    const previewUrl = hasPreview ? `/api/lora-preview?engine=${encodeURIComponent(model)}&path=${encodeURIComponent(item.value)}&v=${encodeURIComponent(metadata.queriedAt)}` : "";
    return <article
      aria-disabled={status === "running" || modelSwitching || loraWorkspaceLocked || !shouldPersistMountedLoras}
      className={`lora-library-card ${mounted ? "mounted" : ""} ${status === "running" || modelSwitching || loraWorkspaceLocked || !shouldPersistMountedLoras ? "locked" : ""}`}
      key={item.value}
      role="button"
      tabIndex="0"
      onClick={() => toggleLora(item, activeLoraCategory)}
      onKeyDown={(event) => {
        if (event.target !== event.currentTarget) return;
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          toggleLora(item, activeLoraCategory);
        }
      }}
    >
      <div className="lora-card-preview">
        <span>No image</span>
        {hasPreview && <img src={previewUrl} alt={`${metadata.modelName} 预览`} onError={(event) => event.currentTarget.remove()} />}
        <button className="lora-card-query" disabled={lookup?.loading} title="使用文件 SHA-256 在 Civitai.com 和 Civitai.red 查询" onClick={(event) => { event.stopPropagation(); lookupLora(item, false, activeLoraCategory.id); }}>
          {lookup?.loading ? <RefreshCw className="spin" size={13} /> : <Search size={13} />}{lookup?.loading ? "查询中" : "查询来源"}
        </button>
        <span className="lora-card-state">{mounted ? <><Check size={14} />已挂载</> : <><Plus size={14} />挂载</>}</span>
      </div>
      <div className="lora-card-copy">
        <strong title={item.name}>{item.name}</strong>
        <small>{(item.size / 1024 / 1024).toFixed(1)} MB · {categoryLabel}</small>
        {metadata?.status === "found" && <div className="lora-card-metadata"><span title={`${metadata.modelName} · ${metadata.versionName}`}>{metadata.modelName} · {metadata.versionName}</span><small>{metadata.baseModel}{metadata.triggerGroups?.length ? ` · ${metadata.triggerGroups.flatMap((group) => group.words).slice(0, 3).join(", ")}` : ""}</small><a href={metadata.sourceUrl} target="_blank" rel="noreferrer" onClick={(event) => event.stopPropagation()}>{metadata.sourceDomain}<ExternalLink size={11} /></a></div>}
        {(metadata?.status === "not_found" || lookup?.error) && <span className="lora-card-error">{lookup?.error || "未找到来源"}</span>}
        <button type="button" className="lora-card-detail" onClick={(event) => { event.stopPropagation(); openLoraDetails(item, activeLoraCategory.id, activeLoraCategory.label); }}><FileText size={12} />查看详细信息</button>
      </div>
    </article>;
  };

  return (
    <main className={`app-shell ${theme.light ? "theme-light" : ""}`}>
      {initialLoading && <div className={`app-loader ${loaderLeaving ? "leaving" : ""} ${theme.light ? "theme-light" : ""}`} role="status" aria-live="polite" aria-label={loaderStatus}>
         <div className="loader-grid" />
         <div className="loader-content">
           <div className="loader-emblem">
             <LoadingLogo />
           </div>
          <div className="loader-brand"><span>XiriaCanvas</span><strong>AI</strong></div>
          <p className="loader-status"><i />{loaderStatus}</p>
          <div className="loader-track"><i style={{ width: `${loaderProgress}%` }} /></div>
          <div className="loader-readout"><span>LOCAL WORKSPACE</span><b>{String(loaderProgress).padStart(3, "0")}%</b><span>{inferenceHealth?.device || "SYSTEM INIT"}</span></div>
        </div>
        <span className="loader-corner loader-corner-tl" /><span className="loader-corner loader-corner-tr" /><span className="loader-corner loader-corner-bl" /><span className="loader-corner loader-corner-br" />
      </div>}
      <header className="topbar">
        <div className="brand"><BrandLogo className="topbar-logo" /><strong>XiriaCanvas</strong><span>AI</span></div>
        <nav className="main-nav" aria-label="主导航">
          <button type="button" className={activePage === "generate" ? "active" : ""} aria-current={activePage === "generate" ? "page" : undefined} onClick={() => setActivePage("generate")}><WandSparkles size={15} />文生图</button>
          <button type="button" className={activePage === "image" ? "active" : ""} aria-current={activePage === "image" ? "page" : undefined} onClick={() => { setActivePage("image"); closeImageViewer(); }}><ImagePlus size={15} />图生图</button>
          <button type="button" className={activePage === "gallery" ? "active" : ""} aria-current={activePage === "gallery" ? "page" : undefined} onClick={() => { setActivePage("gallery"); setLoraManagerOpen(false); closeImageViewer(); }}><ImageIcon size={15} />画廊</button>
          <button type="button" className={activePage === "toolbox" ? "active" : ""} aria-current={activePage === "toolbox" ? "page" : undefined} onClick={() => { setActivePage("toolbox"); setLoraManagerOpen(false); closeImageViewer(); }}><Wrench size={15} />工具箱</button>
        </nav>
        <div className="top-actions">
          <button className="icon-button" title="帮助"><CircleHelp size={18} /></button>
          <button className={`icon-button ${settingsOpen ? "active" : ""}`} title="设置" onClick={() => { setHardwareMonitorOpen(false); setSettingsError(""); setSettingsOpen(true); }}><Settings2 size={18} /></button>
          <div className="device" onClick={() => setHardwareMonitorOpen((current) => !current)} role="button" tabIndex="0" title="硬件性能检测" onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); setHardwareMonitorOpen((current) => !current); } }}><Cpu size={15} /><span>LOCAL</span><i /></div>
        </div>
      </header>

      {activePage === "generate" ? <section
        className={`${workspaceLayoutClassName(workspaceLayout)}${panelResizing ? " panel-resizing" : ""}`}
        style={workspaceLayoutStyle(workspaceLayout)}
      >
        <aside className="left-panel panel" ref={leftPanelRef}>
          <div className="panel-scroll">
            <div className="section-heading"><span>01</span><h2>模型引擎</h2></div>
            <div className="model-segment">
              {models.map((item) => (
                <button
                  key={item.name}
                  disabled={!item.ready || status === "running" || modelSwitching}
                  title={`${item.name} · ${item.detail}`}
                  className={`model-seg ${model === item.name ? "active" : ""}`}
                  onClick={() => selectModel(item.name)}
                >
                  <span className="seg-name">{item.name}</span>
                  <span className="seg-detail">{item.ready ? item.detail : "即将支持"}</span>
                </button>
              ))}
            </div>
            {isSplitModel ? <div className={`checkpoint-picker split-model-picker ${modelError || diffusionModelMissing || textEncoderMissing || (isFlux && textEncoder2Missing) || vaeMissing ? "error" : ""}`}>
              <span className="checkpoint-label"><b>模型组件</b><span className="asset-list-actions"><small>{modelLoading ? "扫描中" : `${diffusionModels.length + textEncoders.length + (isFlux ? textEncoders2.length : 0) + vaes.length} 个资源`}</small><button type="button" className="asset-refresh" title={`刷新 ${engineLabel} 模型组件`} aria-label={`刷新 ${engineLabel} 模型组件`} disabled={modelLoading || modelsRefreshing || status === "running" || modelSwitching} onClick={(event) => { event.preventDefault(); refreshCheckpoints(); }}><RefreshCw className={modelsRefreshing ? "spin" : ""} size={13} /></button></span></span>
              {[
                ["diffusionModel", "扩散模型", diffusionModel, diffusionModels, diffusionModelDirectory, diffusionModelMissing],
                ["textEncoder", isFlux ? "文本编码器 · CLIP-L" : isFlux2 || isKrea2 ? "文本编码器 · 大语言模型" : "文本编码器", textEncoder, textEncoders, textEncoderDirectory, textEncoderMissing],
                ...(isFlux ? [["textEncoder2", "文本编码器 · T5-XXL", textEncoder2, textEncoders2, textEncoderDirectory, textEncoder2Missing]] : []),
                ["vae", "VAE", vae, vaes, vaeDirectory, vaeMissing],
              ].map(([kind, label, value, catalog, directory, missing]) => <label className="split-model-field" key={kind}>
                <span>{label}<small>{modelLoading ? "扫描中" : `${catalog.length} 个`}</small></span>
                <span className="checkpoint-select"><WorkspaceSelect ariaLabel={label} value={value} disabled={status === "running" || modelSwitching || modelLoading || catalog.length === 0} onChange={(nextValue) => selectSplitModelAsset(kind, nextValue)} options={[
                  ...(modelLoading ? [{ value: "", label: "正在加载模型目录...", disabled: true }] : []),
                  ...(!modelLoading && catalog.length === 0 ? [{ value: "", label: "未检测到可用资源", disabled: true }] : []),
                  ...(missing && value ? [{ value, label: `${value}（文件已删除）` }] : []),
                  ...catalog.map((item) => ({ value: item.value, label: item.name })),
                ]} /></span>
                <span className="checkpoint-path" title={modelError || (missing ? `${label}文件已删除，请选择其他资源` : directory)}>{modelError || (missing ? `${label}文件已删除，请选择其他资源` : directory || "正在解析模型路径...")}</span>
              </label>)}
            </div> : <div className={`checkpoint-picker ${modelError || checkpointMissing ? "error" : ""}`}>
              <span className="checkpoint-label"><b>底模选择</b><span className="asset-list-actions"><small>{modelLoading ? "扫描中" : `${checkpoints.length} 个模型`}</small><button type="button" className="asset-refresh" title="刷新底模列表" aria-label="刷新底模列表" disabled={modelLoading || modelsRefreshing || status === "running" || modelSwitching} onClick={(event) => { event.preventDefault(); refreshCheckpoints(); }}><RefreshCw className={modelsRefreshing ? "spin" : ""} size={13} /></button></span></span>
              <span className="checkpoint-select"><WorkspaceSelect ariaLabel="底模选择" value={checkpoint} disabled={status === "running" || modelSwitching || modelLoading || checkpoints.length === 0} onChange={selectCheckpoint} options={[
                ...(modelLoading ? [{ value: "", label: "正在加载模型目录...", disabled: true }] : []),
                ...(!modelLoading && checkpoints.length === 0 ? [{ value: "", label: "未检测到可用底模", disabled: true }] : []),
                ...(checkpointMissing && checkpoint ? [{ value: checkpoint, label: `${checkpoint}（文件已删除）` }] : []),
                ...checkpoints.map((item) => ({ value: item.value, label: item.name })),
              ]} /></span>
              <span className="checkpoint-path" title={modelError || (checkpointMissing ? "当前所选底模文件已删除，请选择其他模型" : checkpointDirectory)}>{modelError || (checkpointMissing ? "当前所选底模文件已删除，请选择其他模型" : checkpointDirectory || "正在解析模型路径...")}</span>
            </div>}

            <div className="section-heading lora-title"><span>02</span><h2>LoRA 挂载</h2><small className="lora-count">已挂载 {loras.length}</small><button onClick={() => { setLoraCategory("mounted"); setLoraManagerOpen(true); }}><Layers3 size={14} />管理</button></div>
            {loras.length === 0 ? <button className="empty-lora" onClick={() => setLoraManagerOpen(true)}><Layers3 size={16} /><span>尚未挂载 LoRA<small>点击管理，从分类库中选择</small></span></button> : <div className="lora-summary">
              <div className="lora-summary-table">
                {loras.map((lora) => (
                  <div key={lora.value} className={`lora-summary-row ${lora.enabled === false ? "disabled" : ""}`} title={`${lora.name} · 分类 ${lora.category}`}>
                    <span className="lora-summary-name">{lora.enabled === false && <Square size={8} />}{lora.name}</span>
                    <b>{formatWeight(lora.weight)}</b>
                  </div>
                ))}
              </div>
            </div>}
            {isSplitModel && !engineAllowsLora && <p className="guidance-unavailable" role="status">当前推理服务未声明 {engineLabel} LoRA 能力；可管理现有组合，但启用后不能生成。</p>}

            <label className="guidance-select"><span><b>引导增强（Guidance）</b><small>{isDistilledGuidance ? `${distilledEngineLabel} 蒸馏引导，无无条件分支` : guidanceOptions.find((item) => item.id === guidance)?.detail}</small></span><WorkspaceSelect ariaLabel="引导增强" value={guidance} disabled={status === "running" || isDistilledGuidance} ariaInvalid={!guidanceReady} ariaDescribedBy={!guidanceReady ? "guidance-compatibility-note" : undefined} onChange={setGuidance} options={(isDistilledGuidance ? guidanceOptions.filter((item) => item.id === "none") : guidanceOptions).map((item) => ({ value: item.id, label: item.label }))} /></label>
            {isDistilledGuidance && <p className="guidance-unavailable" role="status">{distilledEngineLabel} 经过引导蒸馏，每一步只跑一次前向，没有可供 PAG 或 CFG-Zero* 作用的无条件分支。</p>}
            {guidance === "pag" && <div className="guidance-settings">
              <Slider label="PAG 强度" value={pag.scale} min={0} max={5} step={0.1} inputStep={0.01} fixed={2} disabled={status === "running"} onChange={(value) => setPag((current) => ({ ...current, scale: value }))} />
              <label><span>PAG 作用层</span><WorkspaceSelect ariaLabel="PAG 作用层" value={pag.appliedLayers} disabled={status === "running"} onChange={(appliedLayers) => setPag((current) => ({ ...current, appliedLayers }))} options={[{ value: "mid", label: "Mid（推荐）" }, { value: "all", label: "全部自注意力层（高风险）" }]} /></label>
              <small>建议从 0.3 开始；全部层会显著放大对比、描边和色彩，动漫模型尤其容易失真。</small>
            </div>}
            {guidance === "pag" && !guidancePagCompatible && <p id="guidance-compatibility-note" className="guidance-unavailable" role="status" aria-live="polite">当前 {model} 推理运行时未声明 PAG 可用，请切换为“无”后生成。</p>}
            {guidance === "cfg_zero_star" && !guidanceFlowCompatible && <p id="guidance-compatibility-note" className="guidance-unavailable" role="status" aria-live="polite">当前 {model} 推理运行时未声明 CFG-Zero* 可用，请切换为“无”后生成。</p>}

            <button type="button" className={`section-heading parameter-title parameter-toggle ${samplingExpanded ? "expanded" : ""}`} aria-expanded={samplingExpanded} onClick={() => setSamplingExpanded((current) => !current)}><span>03</span><h2>采样参数</h2><small>{sampler} · {steps} STEP · {imagesPerBatch} 张 × {batchCount} 批</small><SlidersHorizontal size={15} /><ChevronDown className="parameter-chevron" size={15} /></button>
            {samplingExpanded && <div className="sampling-parameters">
              <div className="sampler-row">
                <label>采样器<WorkspaceSelect ariaLabel="采样器" value={sampler} onChange={setSampler} options={activeSamplerNames.map((name) => ({ value: name, label: name }))} /></label>
                <label>调度器<WorkspaceSelect ariaLabel="调度器" value={scheduler} onChange={setScheduler} options={activeSchedulerNames.map((name) => ({ value: name, label: name }))} /></label>
              </div>
              <div className="sliders">
                <Slider label="采样步数" value={steps} min={1} max={60} integer onChange={setSteps} />
                <Slider label={isDistilledGuidance ? "蒸馏引导（Guidance）" : "CFG 引导"} value={cfg} min={1} max={15} step={0.5} inputStep={0.1} onChange={setCfg} />
                <Slider label="降噪强度" value={denoise} min={0} max={1} step={0.05} inputStep={0.01} onChange={setDenoise} />
              </div>
              <div className="generation-count-grid">
                <CountField label="单批图片数" detail="一次批量推理同时生成" value={imagesPerBatch} max={10} disabled={status === "running"} onChange={setImagesPerBatch} />
                <CountField label="生成批次数" detail="批次之间按顺序排队" value={batchCount} max={20} disabled={status === "running"} onChange={setBatchCount} />
              </div>
              <p className="generation-count-summary">本任务共生成 <b>{imagesPerBatch * batchCount}</b> 张 · 每批一次推理，最多同时生成 10 张</p>
              <label className="seed-field"><span>随机种子</span><div><input inputMode="numeric" maxLength="20" value={seed} onChange={(event) => setSeed(event.target.value.replace(/\D/g, ""))} onBlur={() => setSeed(normalizeSeed(seed))} /><button title="生成随机种子" onClick={() => setSeed(randomSeed())}><RefreshCw size={15} /></button></div></label>
              <div className="seed-modes" aria-label="种子生成模式">
                {seedModes.map((mode) => <button key={mode.id} className={seedMode === mode.id ? "active" : ""} onClick={() => setSeedMode(mode.id)}>{mode.label}</button>)}
              </div>
              <p className="seed-range">范围 0 ～ 18446744073709551615</p>
            </div>}

            <button type="button" className={`section-heading parameter-title parameter-toggle ${hires.expanded ? "expanded" : ""}`} aria-expanded={hires.expanded} onClick={() => updateHires({ expanded: !hires.expanded })}><span>04</span><h2>Hires.fix</h2><small>{hires.enabled ? `${hires.scale.toFixed(1)}x · ${hiresTargetSize.width} × ${hiresTargetSize.height}` : "关闭"}</small><ZoomIn size={15} /><ChevronDown className="parameter-chevron" size={15} /></button>
            {hires.expanded && <div className={`hires-parameters ${hires.enabled ? "enabled" : ""}`}>
              <div className="hires-enable">
                <div><strong>超分放大 + 扩散精修</strong><small>像素模型预放大后，使用当前底模、LoRA 和提示词进行第二次 img2img</small></div>
                <button type="button" role="switch" aria-label="启用 Hires.fix" aria-checked={hires.enabled} className={hires.enabled ? "active" : ""} disabled={status === "running" || (!engineAllowsHires && !hires.enabled) || ultraLowMode || (!inferenceHealth?.upscalers?.runtime_available && !hires.enabled)} onClick={() => (engineAllowsHires || hires.enabled) && updateHires({ enabled: !hires.enabled })}><i /></button>
              </div>
              {!engineAllowsHires && <p className="hires-unavailable">当前推理服务未声明 {model} Hires.fix 能力。</p>}
              {ultraLowMode && <p className="hires-unavailable">极限省存模式会关闭全部附加后处理阶段。</p>}
              {engineAllowsHires && !inferenceHealth?.upscalers?.runtime_available && <p className="hires-unavailable">超分运行库尚未安装，请重新运行 Setup-XirAI 配置环境。</p>}
              {engineAllowsHires && inferenceHealth?.upscalers?.runtime_available && compatibleUpscalers.length === 0 && <p className="hires-unavailable">尚无兼容超分模型。请前往模型下载器下载推荐模型，或手动放入 models/upscalers 后刷新。</p>}
              <label className="hires-model">超分模型<span><WorkspaceSelect ariaLabel="超分模型" value={hires.model} disabled={!hires.enabled || status === "running" || compatibleUpscalers.length === 0} onChange={(modelValue) => updateHires({ model: modelValue })} options={compatibleUpscalers.length ? compatibleUpscalers.map((item) => ({ value: item.id, label: `${item.label} · ${item.architecture} · ${item.scale}x` })) : [{ value: "", label: "暂无兼容模型", disabled: true }]} /><button type="button" title="重新扫描超分模型" disabled={upscalersRefreshing || status === "running"} onClick={() => void refreshUpscalers()}><RefreshCw className={upscalersRefreshing ? "spin" : ""} size={12} /></button></span></label>
              <div className="hires-scale"><Slider label="放大倍率" value={hires.scale} min={1} max={4} step={0.1} inputStep={0.1} fixed={1} disabled={hiresControlsLocked} onChange={(scale) => updateHires({ scale: Math.round(scale * 10) / 10 })} suffix="x" /><p><span>INPUT {hiresStageTrace?.input.width || size.width} × {hiresStageTrace?.input.height || size.height}</span><b>OUTPUT {hiresTargetSize.width} × {hiresTargetSize.height}</b></p></div>
              <div className="hires-sliders">
                <Slider label="二次重绘强度" value={hires.denoise} min={0.05} max={1} step={0.05} inputStep={0.01} disabled={hiresControlsLocked} onChange={(value) => updateHires({ denoise: value })} />
                <Slider label="Hires 步数" value={hires.steps} min={1} max={100} integer disabled={hiresControlsLocked} onChange={(value) => updateHires({ steps: value })} />
                <Slider label="Hires CFG" value={hires.cfg} min={0} max={30} step={0.5} inputStep={0.1} disabled={hiresControlsLocked} onChange={(value) => updateHires({ cfg: value })} />
               </div>
               <div className="hires-seed-settings">
                 <label>Hires Seed 模式<WorkspaceSelect ariaLabel="Hires Seed 模式" value={hires.seedMode} disabled={hiresControlsLocked} onChange={(seedModeValue) => updateHires({ seedMode: seedModeValue, seed: seedModeValue === "fixed" ? normalizeUint64Seed(hires.seed, normalizeSeed(seed)) : "" })} options={[{ value: "inherit", label: "继承每张首轮 Seed" }, { value: "fixed", label: "固定 Hires Seed" }, { value: "random", label: "每张安全随机" }]} /></label>
                 {hires.seedMode === "fixed" && <label className="hires-seed-field">Hires Seed<span><input inputMode="numeric" maxLength="20" value={hires.seed} disabled={hiresControlsLocked} onChange={(event) => updateHires({ seed: event.target.value.replace(/\D/g, "") })} onBlur={() => updateHires({ seed: normalizeUint64Seed(hires.seed, "0") })} /><button type="button" title="生成固定 Hires Seed" disabled={hiresControlsLocked} onClick={() => updateHires({ seed: randomSeed() })}><RefreshCw size={13} /></button></span></label>}
                 <small>{hires.seedMode === "inherit" ? "每张结果继承该张首轮 Seed" : hires.seedMode === "fixed" ? "所有结果使用同一个无损 uint64 Hires Seed" : "每张结果在后端独立解析一次安全 uint64 Hires Seed"}</small>
               </div>
               {hires.enabled && hiresEffectiveSteps < 1 && <p className="hires-unavailable">当前步数与重绘强度不会产生有效扩散步骤，请提高其中一项。</p>}
               {model === "Anima" && <label className="hires-model">重绘方式<WorkspaceSelect ariaLabel="Hires 重绘方式" value={hires.executionMode === "usdu_tiled" ? "usdu_tiled" : "full_frame"} disabled={hiresControlsLocked} onChange={(executionMode) => updateHires({ executionMode })} options={[{ value: "usdu_tiled", label: "USDU 分块重绘（推荐）" }, { value: "full_frame", label: "整图重绘（兼容）" }]} /></label>}
               <div className="hires-tile-grid">
                 <label>Hires 采样器<WorkspaceSelect ariaLabel="Hires 采样器" value={hires.sampler || ""} disabled={hiresControlsLocked} onChange={(samplerValue) => updateHires({ sampler: samplerValue || null })} options={[{ value: "", label: "跟随首轮" }, ...(model === "Anima" ? ANIMA_SAMPLERS : samplerNames).map((item) => ({ value: item, label: item }))]} /></label>
                 <label>Hires 调度器<WorkspaceSelect ariaLabel="Hires 调度器" value={hires.scheduler || ""} disabled={hiresControlsLocked} onChange={(schedulerValue) => updateHires({ scheduler: schedulerValue || null })} options={[{ value: "", label: "跟随首轮" }, ...(model === "Anima" ? ANIMA_SCHEDULERS : schedulerNames).map((item) => ({ value: item, label: item }))]} /></label>
               </div>
              {model === "Anima" && hires.executionMode === "usdu_tiled" && <div className="hires-tile-grid"><label>扩散重绘分块宽度<output>Auto（只读）</output></label><label>扩散重绘分块高度<output>Auto（只读）</output></label></div>}
              {model === "Anima" && hires.executionMode === "usdu_tiled" && <p className="hires-incompatible">Auto = 首轮源图宽高（当前 {size.width} × {size.height}），本轮 2x 会形成 2 × 2；padding 32；mask blur 8；uniform tiles；per-tile VAE tiled decode；seam None；每 tile 执行 Hires steps。</p>}
              <div className="hires-tile-grid">
                <label>RealESRGAN / SR 像素放大分块<BoundedNumberInput value={hires.tileSize} min={32} max={2048} integer disabled={hiresControlsLocked} onCommit={(tileSize) => updateHires({ tileSize, tileOverlap: Math.min(hires.tileOverlap, Math.floor(tileSize / 2)) })} /></label>
                <label>RealESRGAN / SR 像素放大分块重叠<BoundedNumberInput value={hires.tileOverlap} min={0} max={Math.min(512, Math.floor(hires.tileSize / 2))} integer disabled={hiresControlsLocked} onCommit={(tileOverlap) => updateHires({ tileOverlap })} /></label>
               </div>
              {upscalerModels.some((item) => !item.compatible) && <p className="hires-incompatible">检测到 {upscalerModels.filter((item) => !item.compatible).length} 个不兼容文件；已从选择列表隐藏。</p>}
              <p className="hires-path" title={inferenceHealth?.upscalers?.directory}>模型目录 · {inferenceHealth?.upscalers?.directory || "models/upscalers"} · 支持 PTH / PT / CKPT 权重字典及 Safetensors，不加载 TorchScript</p>
            </div>}

            <button type="button" className={`section-heading parameter-title parameter-toggle ${adetailer.expanded ? "expanded" : ""}`} aria-expanded={adetailer.expanded} onClick={() => updateADetailer({ expanded: !adetailer.expanded })}><span>05</span><h2>ADetailer</h2><small>{adetailer.enabled ? adetailerSummary(adetailer) : "关闭"}</small><Sparkles size={15} /><ChevronDown className="parameter-chevron" size={15} /></button>
            {adetailer.expanded && <div className={`adetailer-parameters ${adetailer.enabled ? "enabled" : ""}`}>
              <div className="adetailer-enable">
                <div><strong>检测后局部重绘</strong><small>按所选后处理顺序修复面部、手部或人物区域</small></div>
                <button type="button" role="switch" aria-label="启用 ADetailer" aria-checked={adetailer.enabled} className={adetailer.enabled ? "active" : ""} disabled={adetailerLocked || (!engineAllowsADetailer && !adetailer.enabled) || ultraLowMode || (!adetailerInfo.runtimeAvailable && !adetailer.enabled)} onClick={() => (engineAllowsADetailer || adetailer.enabled) && updateADetailer({ enabled: !adetailer.enabled })}><i /></button>
              </div>
              {!engineAllowsADetailer && <p className="adetailer-unavailable">当前推理服务未声明 {model} ADetailer 能力。</p>}
              {ultraLowMode && <p className="adetailer-unavailable">极限省存模式会关闭全部附加后处理阶段。</p>}
              {engineAllowsADetailer && !adetailerInfo.available && <p className="adetailer-unavailable">{adetailerInfo.loading ? "正在读取检测模型..." : adetailerInfo.error || (adetailerInfo.runtimeAvailable ? "尚未安装 YOLO 检测模型，请从下方模型库下载或手动放入 models/yolo" : "项目 YOLO 运行环境不可用，请重新运行环境配置")}</p>}
              <div className="adetailer-model-library">
                <div className="adetailer-library-heading"><strong>基础推荐模型</strong><span><small>{adetailerInfo.builtins.filter((item) => item.installed).length} / {adetailerInfo.builtins.length} 已就绪</small><button type="button" title="刷新 YOLO 模型列表" aria-label="刷新 YOLO 模型列表" disabled={adetailerInfo.loading || status === "running"} onClick={() => { void refreshADetailerModels(); }}><RefreshCw className={adetailerInfo.loading ? "spin" : ""} size={11} /></button></span></div>
                <div className="adetailer-recommended-model">
                  <span><b>ADetailer 基础模型包</b><small>包含面部、手部和人物分割检测所需的 5 个官方模型</small></span>
                  {adetailerInfo.builtins.length > 0 && adetailerInfo.builtins.every((item) => item.installed)
                    ? <em><Check size={12} />已安装</em>
                    : <button type="button" disabled={status === "running" || adetailerDownload?.status === "downloading"} onClick={downloadRecommendedYoloModels}><Download size={12} />一键下载全部（5）</button>}
                </div>
              </div>
              {adetailerDownload && <div className={`adetailer-download-status ${adetailerDownload.status}`} aria-live="polite">
                <div><b>{adetailerDownload.label}{adetailerDownload.modelIndex ? ` · ${adetailerDownload.modelIndex} / ${adetailerDownload.totalModels}` : ""}</b><span>{adetailerDownload.message}</span></div>
                {adetailerDownload.totalBytes > 0 && <i><i style={{ width: `${Math.min(100, adetailerDownload.currentBytes / adetailerDownload.totalBytes * 100)}%` }} /></i>}
                <small>{adetailerDownload.currentBytes > 0 ? `${formatFileSize(adetailerDownload.currentBytes)} / ${formatFileSize(adetailerDownload.totalBytes)}` : ""}{adetailerDownload.speedBps > 0 ? ` · ${formatFileSize(adetailerDownload.speedBps)}/s` : ""}{adetailerDownload.connections > 1 ? ` · ${adetailerDownload.connections} 路分片` : ""}</small>
              </div>}
              {(() => {
                const unitLocked = !adetailer.enabled || adetailerLocked;
                const index = adetailerPageIndex(adetailerUnits, adetailerPage);
                const unit = adetailerUnits[index];
                if (!unit) return null;
                const unitSteps = adetailerUnitSteps(unit, steps, model);
                return <div className="adetailer-units">
                  {adetailerUnits.length > 1 && <nav className="adetailer-pager" aria-label="ADetailer 检测单元">
                    <button type="button" aria-label="上一个检测单元" disabled={index < 1} onClick={() => setADetailerPage(adetailerStepUnitId(adetailerUnits, adetailerPage, -1))}><ChevronLeft size={14} /></button>
                    <ol>
                      {adetailerUnits.map((entry, position) => <li key={entry.id}>
                        <button
                          type="button"
                          className={`${position === index ? "current" : ""} ${entry.enabled ? "on" : ""}`}
                          aria-current={position === index ? "true" : undefined}
                          aria-label={`第 ${position + 1} 个检测单元${entry.enabled ? "" : "（已关闭）"}`}
                          onClick={() => setADetailerPage(entry.id)}
                        >{position + 1}</button>
                      </li>)}
                    </ol>
                    <button type="button" aria-label="下一个检测单元" disabled={index >= adetailerUnits.length - 1} onClick={() => setADetailerPage(adetailerStepUnitId(adetailerUnits, adetailerPage, 1))}><ChevronRight size={14} /></button>
                  </nav>}
                  {/* Keyed by the unit so paging *replaces* the controls rather than
                      reusing them: `BoundedNumberInput` holds an uncommitted draft of
                      its own, and reused controls would carry one unit's half-typed
                      value onto the next unit's field. */}
                  <section className={`adetailer-unit ${unit.enabled ? "on" : ""}`} key={unit.id}>
                    <header>
                      <span>{String(index + 1).padStart(2, "0")}</span>
                      <strong>{adetailerUnitLabel(index, adetailerUnits.length)}</strong>
                      <small>{unit.detector ? unit.detector.split("/").pop() : "未选择模型"}</small>
                      <button
                        type="button"
                        role="switch"
                        aria-label={`启用第 ${index + 1} 个 ADetailer 单元`}
                        aria-checked={unit.enabled}
                        className={unit.enabled ? "active" : ""}
                        disabled={unitLocked}
                        onClick={() => updateADetailerUnit(unit.id, { enabled: !unit.enabled })}
                      ><i /></button>
                    </header>
                    <div className="adetailer-unit-body">
                      <label className="adetailer-model">检测模型<WorkspaceSelect ariaLabel={`第 ${index + 1} 个 ADetailer 检测模型`} value={unit.detector} disabled={unitLocked} onChange={(detector) => updateADetailerUnit(unit.id, { detector })} options={adetailerModels.length ? adetailerModels.map((item) => ({ value: item.value, label: `${item.label || item.name}${item.source === "community" ? " · 社区" : ""}` })) : [{ value: "", label: "暂无模型", disabled: true }]} /></label>
                      <div className="adetailer-sliders">
                        <Slider label="检测置信度" value={unit.confidence} min={0.05} max={1} step={0.05} inputStep={0.01} disabled={unitLocked} onChange={(value) => updateADetailerUnit(unit.id, { confidence: value })} />
                        <Slider label="最多处理区域" value={unit.maxDetections} min={1} max={8} integer disabled={unitLocked} onChange={(value) => updateADetailerUnit(unit.id, { maxDetections: value })} />
                        <Slider label="最小区域比例" value={unit.maskMinRatio} min={0} max={0.5} step={0.01} disabled={unitLocked} onChange={(value) => updateADetailerUnit(unit.id, { maskMinRatio: value, maskMaxRatio: Math.max(value, unit.maskMaxRatio) })} />
                        <Slider label="最大区域比例" value={unit.maskMaxRatio} min={0.05} max={1} step={0.05} disabled={unitLocked} onChange={(value) => updateADetailerUnit(unit.id, { maskMaxRatio: value, maskMinRatio: Math.min(value, unit.maskMinRatio) })} />
                      </div>
                      <div className="adetailer-mask-grid">
                        <label>膨胀 / 腐蚀<BoundedNumberInput value={unit.dilateErode} min={-128} max={128} integer disabled={unitLocked} onCommit={(dilateErode) => updateADetailerUnit(unit.id, { dilateErode })} /></label>
                        <label>蒙版模糊<BoundedNumberInput value={unit.maskBlur} min={0} max={64} integer disabled={unitLocked} onCommit={(maskBlur) => updateADetailerUnit(unit.id, { maskBlur })} /></label>
                        <label>局部边距<BoundedNumberInput value={unit.padding} min={0} max={256} integer disabled={unitLocked} onCommit={(padding) => updateADetailerUnit(unit.id, { padding })} /></label>
                      </div>
                      <div className="adetailer-denoise"><Slider label="重绘强度" value={unit.denoise} min={0.05} max={1} step={0.05} inputStep={0.01} disabled={unitLocked} onChange={(value) => updateADetailerUnit(unit.id, { denoise: value })} /></div>
                      <div className="adetailer-overrides">
                        <label><input type="checkbox" checked={unit.useSteps} disabled={unitLocked} onChange={(event) => updateADetailerUnit(unit.id, { useSteps: event.target.checked })} /><span>独立步数</span><BoundedNumberInput value={unit.steps} min={1} max={100} integer disabled={unitLocked || !unit.useSteps} onCommit={(unitStepCount) => updateADetailerUnit(unit.id, { steps: unitStepCount })} /></label>
                        <label><input type="checkbox" checked={unit.useCfg} disabled={unitLocked} onChange={(event) => updateADetailerUnit(unit.id, { useCfg: event.target.checked })} /><span>独立 CFG</span><BoundedNumberInput value={unit.cfg} min={0} max={30} step={0.1} disabled={unitLocked || !unit.useCfg} onCommit={(cfg) => updateADetailerUnit(unit.id, { cfg })} /></label>
                      </div>
                      <div className="adetailer-prompts">
                        <label>正向提示词<textarea value={unit.prompt} disabled={unitLocked} onChange={(event) => updateADetailerUnit(unit.id, { prompt: event.target.value })} placeholder="留空继承主提示词；[PROMPT] 插入主提示词" /></label>
                        <label>负向提示词<textarea value={unit.negativePrompt} disabled={unitLocked || !engineAllowsNegativePrompt} onChange={(event) => updateADetailerUnit(unit.id, { negativePrompt: event.target.value })} placeholder={engineAllowsNegativePrompt ? "留空继承主负向提示词" : "FLUX.1 没有无条件分支，负向提示词不会参与生成"} /></label>
                      </div>
                      {unit.enabled && unitSteps < 1 && <p className="adetailer-unavailable">该单元有效步数为 0，请提高主采样或独立步数，或提高重绘强度。</p>}
                    </div>
                  </section>
                </div>;
              })()}
              <div className="adetailer-unit-add">
                <small>{ADETAILER_UNIT_LIMIT} 个检测单元 · 已启用 {activeADetailerUnits(adetailer).length} 个 · 按编号顺序依次执行，后一个在前一个的结果上继续修复</small>
              </div>
              <p className="adetailer-path" title={adetailerInfo.directory}>CPU 检测 · 重绘继承当前采样器、调度器、底模与 LoRA</p>
            </div>}

            <button type="button" className={`section-heading parameter-title parameter-toggle ${rtx.expanded ? "expanded" : ""}`} aria-expanded={rtx.expanded} onClick={() => updateRtx({ expanded: !rtx.expanded })}><span>06</span><h2>RTX VSR</h2><small>{rtx.enabled ? `${rtx.scale.toFixed(2)}x · ${rtxStageTrace?.output.width || size.width} × ${rtxStageTrace?.output.height || size.height}` : "关闭"}</small><Zap size={15} /><ChevronDown className="parameter-chevron" size={15} /></button>
            {rtx.expanded && <div className={`rtx-parameters ${rtx.enabled ? "enabled" : ""}`}>
              <div className="rtx-enable">
                <div><strong>NVIDIA RTX 超分</strong><small>执行最终或中间像素超分</small></div>
                <button type="button" role="switch" aria-label="启用 RTX VSR" aria-checked={rtx.enabled} className={rtx.enabled ? "active" : ""} disabled={status === "running" || (!engineAllowsRtx && !rtx.enabled) || ultraLowMode || ((!rtxHealth.available || !rtxPreviewDimensions.valid) && !rtx.enabled)} onClick={() => (engineAllowsRtx || rtx.enabled) && updateRtx({ enabled: !rtx.enabled })}><i /></button>
              </div>
              {!engineAllowsRtx && <p className="rtx-unavailable">当前推理服务未声明 {model} RTX VSR 能力。</p>}
              {ultraLowMode && <p className="rtx-unavailable">极限省存模式会关闭全部附加后处理阶段。</p>}
              {engineAllowsRtx && !rtxHealth.available && <p className={`rtx-unavailable ${rtxHealth.probing ? "probing" : ""}`} role="status" aria-live="polite">{rtxStatusReason}</p>}
              {engineAllowsRtx && rtxHealth.available && rtxHealth.warning && <p className="rtx-unavailable">{rtxHealth.warning}</p>}
              {engineAllowsRtx && rtxHealth.available && !rtxPreviewDimensions.valid && <p className="rtx-unavailable">当前顺序中的 RTX 中间目标超过 8192 边长或 32 MP 安全限制，请降低倍率、Hires 倍率或调整顺序。</p>}
              <div className="rtx-runtime-grid" aria-label="RTX VSR 运行环境">
                <div><span>状态</span><strong>{rtxHealth.probing ? "PROBING" : rtxHealth.available ? "READY" : rtxHealth.supported ? "RUNTIME REQUIRED" : "UNSUPPORTED"}</strong></div>
                <div><span>运行时</span><strong>{rtxHealth.runtime_version || "--"}</strong></div>
                <div><span>设备</span><strong title={rtxHealth.device}>{rtxHealth.device || "--"}</strong></div>
                <div><span>CUDA 能力</span><strong>{rtxHealth.compute_capability || "--"}</strong></div>
                <div><span>驱动</span><strong>{rtxHealth.driver_version || "--"}</strong></div>
              </div>
              <div className="rtx-scale"><Slider label="放大倍率" value={rtx.scale} min={1} max={4} step={0.01} inputStep={0.01} fixed={2} disabled={status === "running"} onChange={(scale) => updateRtx({ scale: Math.round(scale * 100) / 100 })} suffix="x" /><p><span>INPUT {rtxStageTrace?.input.width || size.width} × {rtxStageTrace?.input.height || size.height}</span><b>OUTPUT {rtxStageTrace?.output.width || size.width} × {rtxStageTrace?.output.height || size.height}</b></p><p><span>ORDERED FINAL</span><b>{rtxPreviewDimensions.width} × {rtxPreviewDimensions.height}</b></p></div>
              <label className="rtx-quality">处理质量<WorkspaceSelect ariaLabel="RTX VSR 处理质量" value={rtx.quality} disabled={status === "running"} onChange={(quality) => updateRtx({ quality })} options={[{ value: "low", label: "LOW · 低" }, { value: "medium", label: "MEDIUM · 中" }, { value: "high", label: "HIGH · 高" }, { value: "ultra", label: "ULTRA · 极致" }]} /></label>
              <p className="rtx-proprietary">RTX VSR 依赖 NVIDIA 专有 Video Effects 运行时，仅在受支持的 x64 Windows / Linux NVIDIA RTX 设备上可用；生成内容仍在本机处理。</p>
            </div>}

            <section className="postprocess-order" aria-labelledby="postprocess-order-title">
              <header><div><span>07</span><strong id="postprocess-order-title">后处理顺序</strong><small>禁用阶段仍可排序；生成期间锁定</small></div><b>FINAL {postprocessDimensions.width} × {postprocessDimensions.height}</b></header>
              <div className="postprocess-order-list">
                {normalizedPostprocessOrder.map((stage, index) => {
                  const stageSettings = postprocessSettings[stage];
                  const labels = { hires: ["Hires.fix", "扩散精修"], adetailer: ["ADetailer", "局部重绘"], rtx: ["RTX VSR", "NVIDIA 超分"] };
                  return <div className={stageSettings.enabled ? "enabled" : "disabled"} key={stage}><span>{String(index + 1).padStart(2, "0")}</span><div><strong>{labels[stage][0]}</strong><small>{labels[stage][1]} · {stageSettings.enabled ? "启用" : "关闭"}</small></div><div className="postprocess-order-actions"><button type="button" aria-label={`上移 ${labels[stage][0]}`} title={`上移 ${labels[stage][0]}`} disabled={status === "running" || index === 0} onClick={() => movePostprocessStage(stage, -1)}><ArrowUp size={13} /></button><button type="button" aria-label={`下移 ${labels[stage][0]}`} title={`下移 ${labels[stage][0]}`} disabled={status === "running" || index === normalizedPostprocessOrder.length - 1} onClick={() => movePostprocessStage(stage, 1)}><ArrowDown size={13} /></button></div></div>;
                })}
              </div>
              {!postprocessDimensions.valid && <p>RTX VSR 中间目标超过 8192 边长或 32 MP 安全限制。</p>}
            </section>
          </div>
        </aside>

        {/* The gutter between the configuration panel and the composer is the whole control: drag
            to resize, double-click to collapse or restore. No tab sits on it — the seam is already
            the thing you reach for, and a button parked in a 7px strip was chrome for a gesture
            that reads fine without one. The keyboard gets the same two actions, because a
            pointer-only affordance would put the panel out of reach of anyone not using one. */}
        <div
          className="panel-resizer"
          role="separator"
          aria-orientation="vertical"
          aria-expanded={!workspaceLayout.leftCollapsed}
          aria-label={workspaceLayout.leftCollapsed ? "展开左侧配置面板" : "调整左侧配置面板宽度"}
          tabIndex={0}
          onPointerDown={beginPanelResize}
          onPointerMove={continuePanelResize}
          onPointerUp={endPanelResize}
          onPointerCancel={endPanelResize}
          onKeyDown={panelResizeKeyDown}
          onDoubleClick={() => commitWorkspaceLayout(toggleLeftPanel)}
          title={workspaceLayout.leftCollapsed ? "双击或向右拖动展开配置面板" : "拖动调整宽度 · 双击折叠"}
        />

        <section className="center-panel panel">
          <div className="prompt-header">
            <div><span className="eyebrow">PROMPT COMPOSER</span><p className="prompt-syntax-help">权重语法：<code>(text)</code> = 1.1；<code>(text:1.25)</code> = 显式权重；<code>\(text\)</code> = 字面括号。</p></div>
            <button className="optimize" onClick={() => setAssistantOpen(true)} aria-expanded={assistantOpen}><WandSparkles size={16} />AI 助手<kbd>Ctrl/⌘ K</kbd></button>
          </div>

          <label className={`prompt-field positive-field ${transparentPromptEnabled ? "transparent-enabled" : ""}`}>
            <div><span>正向提示词{transparentPromptEnabled && <b className="special-tag-mark">TRANSPARENT PNG</b>}</span><small>{positive.length} / 2000</small></div>
            <textarea ref={positivePromptRef} value={positive} onChange={(event) => changePromptText("positive", event)} onSelect={(event) => recordPromptSelection("positive", event)} onClick={(event) => recordPromptSelection("positive", event)} onKeyUp={(event) => recordPromptSelection("positive", event)} onFocus={(event) => recordPromptSelection("positive", event)} placeholder="描述画面主体、环境、光线与风格..." spellCheck={false} />
            <Sparkles className="field-watermark" size={46} />
          </label>
          <div className={`transparent-tag-control ${transparentPromptEnabled ? "active" : ""}`}>
            <div className="transparent-model-picker" ref={backgroundRemovalPickerRef}>
              <button type="button" className="transparent-model-select" aria-haspopup="listbox" aria-expanded={backgroundRemovalPickerOpen} disabled={status === "running"} onClick={() => setBackgroundRemovalPickerOpen((current) => !current)}>
                <span><b>{selectedBackgroundRemovalModel?.label || "选择透明背景模型"}</b><small>{selectedBackgroundRemovalModel ? `${selectedBackgroundRemovalModel.installed ? "已安装" : formatFileSize(selectedBackgroundRemovalModel.size)}${selectedBackgroundRemovalModel.local ? " · 本地模型" : ""}` : "推荐模型与本地 ONNX"}</small></span>
                <ChevronDown size={14} />
              </button>
              {backgroundRemovalPickerOpen && <div className="transparent-model-menu" role="listbox" aria-label="透明背景模型">
                <header><span>推荐模型</span><small>固定哈希 · 下载默认 8 路</small></header>
                {recommendedBackgroundRemovalModels.map((item) => <div className={`transparent-model-option ${backgroundRemovalModel === item.id ? "selected" : ""}`} role="option" aria-selected={backgroundRemovalModel === item.id} key={item.id}>
                  <button type="button" className="transparent-model-choice" onClick={() => { setBackgroundRemovalModel(item.id); if (item.installed) setBackgroundRemovalPickerOpen(false); }}><i /><span><b>{item.label}</b><small>{item.description}</small><em>{formatFileSize(item.size)} · {item.license || "本地使用"}</em></span></button>
                  {item.installed ? <strong><Check size={12} />已安装</strong> : <button type="button" className="transparent-model-download" disabled={status === "running" || backgroundRemovalDownload?.active} onClick={() => downloadBackgroundRemovalModel(item.id)}>{backgroundRemovalDownload?.active && backgroundRemovalDownload.modelId === item.id ? <RefreshCw className="spin" size={12} /> : <Download size={12} />}{backgroundRemovalDownload?.active && backgroundRemovalDownload.modelId === item.id ? "下载中" : "下载"}</button>}
                </div>)}
                <header><span>本地模型</span><small title={inferenceHealth?.background_removal?.directory}>放入 models/background-removal</small></header>
                {localBackgroundRemovalModels.length ? localBackgroundRemovalModels.map((item) => <div className={`transparent-model-option ${backgroundRemovalModel === item.id ? "selected" : ""}`} role="option" aria-selected={backgroundRemovalModel === item.id} key={item.id}><button type="button" className="transparent-model-choice" onClick={() => { setBackgroundRemovalModel(item.id); setBackgroundRemovalPickerOpen(false); }}><i /><span><b>{item.label}</b><small>{item.description}</small><em>{formatFileSize(item.size)} · 本地 ONNX</em></span></button><strong><Check size={12} />已发现</strong></div>) : <p>未发现额外 ONNX 模型</p>}
              </div>}
            </div>
            <div className={`transparent-mode-copy ${backgroundRemovalDownloadForSelection?.status || ""}`} aria-live="polite"><strong>透明背景模式</strong><span>{backgroundRemovalDownloadForSelection?.status === "error" ? backgroundRemovalDownloadForSelection.message : backgroundRemovalDownloadForSelection?.active ? backgroundRemovalDownloadForSelection.message : selectedBackgroundRemovalReady ? `${selectedBackgroundRemovalModel.label} · 已就绪` : selectedBackgroundRemovalModel?.installed ? "模型已安装，等待 ONNX Runtime" : "选择模型后可在下拉栏内下载"}</span>{backgroundRemovalDownloadForSelection?.active && <><i><i style={{ width: `${backgroundRemovalProgress}%` }} /></i><small>{formatFileSize(backgroundRemovalDownloadForSelection.currentBytes || 0)} / {formatFileSize(backgroundRemovalDownloadForSelection.totalBytes || selectedBackgroundRemovalModel?.size || 0)}{backgroundRemovalDownloadForSelection.speedBps > 0 ? ` · ${formatFileSize(backgroundRemovalDownloadForSelection.speedBps)}/s` : ""} · {backgroundRemovalDownloadForSelection.route || "正在测速"} · {backgroundRemovalDownloadForSelection.connections || 8} 路</small></>}</div>
            <button type="button" className={`transparent-mode-switch ${transparentPromptEnabled ? "active" : ""}`} role="switch" aria-checked={transparentPromptEnabled} disabled={status === "running"} onClick={toggleTransparentBackground}><i /><span>{transparentPromptEnabled ? "已启用" : "未启用"}</span></button>
          </div>
          <PresetBox title="预设正向 Prompt" type="positive" records={sortPromptPresetRecords(promptPresets.records, "positive")} disabled={status === "running"} libraryError={promptPresetLibraryError} libraryWarning={promptPresetLibraryWarning} onSelect={applyPreset} onCreate={() => openPromptPresetDialog("positive")} onEdit={(record) => openPromptPresetDialog("positive", record)} onDelete={requestDeletePromptPreset} />

          <label className="prompt-field negative-field">
            <div><span>负向提示词</span><small>{engineAllowsNegativePrompt ? `${negative.length} / 1000` : "当前引擎不使用"}</small></div>
            <textarea ref={negativePromptRef} value={negative} disabled={!engineAllowsNegativePrompt} onChange={(event) => changePromptText("negative", event)} onSelect={(event) => recordPromptSelection("negative", event)} onClick={(event) => recordPromptSelection("negative", event)} onKeyUp={(event) => recordPromptSelection("negative", event)} onFocus={(event) => recordPromptSelection("negative", event)} placeholder={engineAllowsNegativePrompt ? "描述需要避免的内容..." : "FLUX.1 没有无条件分支，负向提示词不会参与生成"} spellCheck={false} />
          </label>
          {/* The text is kept, not cleared: switching back to another engine should find it intact. */}
          {!engineAllowsNegativePrompt && negative.trim() && <p className="guidance-unavailable" role="status">已保留负向提示词，但本次 FLUX.1 生成不会使用它。</p>}
          <PresetBox title="预设负向 Prompt" type="negative" records={sortPromptPresetRecords(promptPresets.records, "negative")} disabled={status === "running"} libraryError={promptPresetLibraryError} libraryWarning={promptPresetLibraryWarning} onSelect={applyPreset} onCreate={() => openPromptPresetDialog("negative")} onEdit={(record) => openPromptPresetDialog("negative", record)} onDelete={requestDeletePromptPreset} />

          <div className="canvas-settings">
            <div className="section-heading"><span>08</span><h2>画布尺寸</h2><div className="dimension"><BoundedNumberInput value={size.width} min={0} max={2048} integer normalize={(value) => Math.round(value / 64) * 64} onCommit={(width) => setSize((current) => ({ ...current, width }))} ariaLabel="画布宽度" /><i>×</i><BoundedNumberInput value={size.height} min={0} max={2048} integer normalize={(value) => Math.round(value / 64) * 64} onCommit={(height) => setSize((current) => ({ ...current, height }))} ariaLabel="画布高度" /><b>PX</b></div></div>
            <SizeGrid width={size.width} height={size.height} onChange={(width, height) => setSize({ width, height })} />
            <div className="size-presets">
              {[{ label: "512", w: 512, h: 512 }, { label: "1:1", w: 1024, h: 1024 }, { label: "4:3", w: 1152, h: 896 }, { label: "3:4", w: 896, h: 1152 }, { label: "16:9", w: 1344, h: 768 }, { label: "9:16", w: 768, h: 1344 }].map((item) => (
                <button key={item.label} className={size.width === item.w && size.height === item.h ? "active" : ""} onClick={() => setSize({ width: item.w, height: item.h })}>{item.label}</button>
              ))}
              <span>范围 0 × 0 ～ 2048 × 2048</span>
            </div>
            {!canvasSizeValid && <p className="canvas-size-warning">0 × 0 可用于网格定位；生成尺寸至少为 64 × 64</p>}
          </div>
        </section>

        <aside className="preview-panel panel">
          <div className="preview-head"><div><span className="eyebrow">{generatedOutputs.length ? `OUTPUT ${String(selectedOutputIndex + 1).padStart(2, "0")}${generatedOutputs.length > 1 ? ` / ${String(generatedOutputs.length).padStart(2, "0")}` : ""}` : "OUTPUT WORKSPACE"}</span><div className="preview-title-row"><h2>生成预览</h2><button className={`preview-toggle ${processPreview ? "active" : ""}`} role="switch" aria-checked={processPreview} disabled={status === "running" || !engineAllowsProcessPreview} title={!engineAllowsProcessPreview ? `${model} 不支持过程预览` : ""} onClick={() => engineAllowsProcessPreview && setProcessPreview((current) => !current)}><i /><span>过程预览</span></button></div></div><div><button type="button" className="unload-model-button" title="释放已加载底模占用的 GPU/内存" disabled={status === "running" || modelSwitching || !inferenceHealth?.model_cached} onClick={unloadLoadedModel}><Trash2 size={13} /><span>{modelSwitching ? "正在卸载" : "卸载模型"}</span></button><button type="button" className="unload-model-button gallery-preview-add" title="把一张或多张生成结果加入画廊" disabled={!generatedOutputs.some((output) => output.asset_id)} onClick={() => setGalleryAddOpen(true)}><ImagePlus size={13} /><span>加入画廊</span></button><button className="icon-button" title="打开图片预览与拼图工作区" onClick={openImageViewer}><Maximize2 size={17} /></button><button className="icon-button" disabled={!generatedImage} onClick={() => { if (!generatedImage) return; const link = document.createElement("a"); link.href = generatedImage; link.download = generatedName || `XirAI-${generationJob}.png`; link.click(); }}><Download size={17} /></button></div></div>
          <div className={`preview-stage ${status}`}>
            <div className="preview-grid" />
            {status === "idle" && <div className="empty-preview"><div className="empty-orbit"><ImageIcon size={29} /></div><strong>等待生成</strong><p>也可以直接打开 outputs 图片<br />进入预览与拼图工作区</p><button type="button" onClick={openImageViewer}><FolderOpen size={14} />打开预览工作区</button></div>}
            {status === "running" && <div className={`generating-preview ${previewKind || generationStage}`}>
              <div className="step-preview-empty">{livePreview ? null : <><ImageIcon size={28} /><span>{processPreview ? "等待首帧异步预览" : "过程预览已关闭，完成后显示"}</span></>}</div>
              {livePreview && <img className="step-preview-image" src={livePreview} alt={`${stageLabel} 实时预览`} />}
              {previewKind === "adetailer_detection" && <div className="preview-stage-badge"><span>YOLO</span><b>检测完成 · {generationDetail?.selected_count || 0} 个区域</b></div>}
              {generationStage === "adetailer_inpaint" && <div className="preview-stage-badge detail"><span>局部放大</span><b>{activeDetail?.class_name || "区域"} · {(activeDetail?.confidence * 100 || 0).toFixed(0)}%</b></div>}
            </div>}
            {status === "complete" && <button className={`completed-preview ${selectedOutput?.transparent_background ? "transparent-preview" : ""}`} onClick={openImageViewer} title="点击放大预览"><img className="generated-image" src={generatedImage} alt={`第 ${selectedOutputIndex + 1} 张 AI 生成结果`} />{selectedOutput?.transparent_background && <div className="transparent-output-badge">RGBA · {selectedOutput.background_removal?.method === "birefnet-lite-fp16" ? "BIREFNET LITE" : selectedOutput.background_removal?.method === "bria-rmbg-2-fp16" ? "RMBG 2.0" : selectedOutput.background_removal?.method === "u2netp-onnx" ? "U-2-NETP" : selectedOutput.background_removal?.method?.startsWith("local:") ? "LOCAL ONNX" : "ALGORITHM"}</div>}<div className="image-caption"><span>{selectedOutput?.width || size.width} × {selectedOutput?.height || size.height} · 批 {selectedOutput?.batch_index || 1} / 图 {selectedOutput?.image_index || 1}</span><span>Seed {selectedOutput?.seed ?? seed} · {generationElapsed.toFixed(1)} 秒</span></div></button>}
            {status === "cancelled" && <div className="generation-cancelled"><Square size={30} /><strong>生成已终止</strong><p>已安全停止任务，没有保存未完成图片</p><button onClick={() => setStatus("idle")}>返回调整参数</button></div>}
            {status === "error" && <div className="generation-error"><X size={32} /><strong>生成失败</strong><p>{generationError}</p><button onClick={() => setStatus("idle")}>返回调整参数</button></div>}
            <div className="corner corner-tl" /><div className="corner corner-tr" /><div className="corner corner-bl" /><div className="corner corner-br" />
          </div>
          {status === "running" && <div className={`generation-progress-dock ${generationProgressCollapsed ? "collapsed" : ""}`}>
            <button type="button" className="generation-progress-summary" aria-expanded={!generationProgressCollapsed} onClick={() => setGenerationProgressCollapsed((current) => !current)}>
              <span className="generation-progress-main"><i /><b>{stageLabel}</b><strong>{generationPhase || "正在生成"}</strong></span>
              <span className="generation-progress-percent">{displayedGenerationProgress}%</span>
              <ChevronDown size={14} />
            </button>
            <div className="generation-progress-line"><i style={{ width: `${displayedGenerationProgress}%` }} /></div>
            <div className="generation-progress-detail"><span>{generationStageTotal > 0 ? `${generationStageStep} / ${generationStageTotal}` : generationStage === "adetailer_detect" ? "ANALYZING" : "WAIT"}</span><span>{generationStage === "adetailer_inpaint" ? `REGION ${generationDetail?.region_index || 1} / ${generationDetail?.region_total || 1}` : generationStage === "rtx_upscale" ? "NVIDIA VFX" : generationStep > 0 ? `STEP ${generationStep} / ${generationTotal || steps}` : "MODEL LOADING"} · BATCH {generationBatchIndex || 1}/{generationBatchCount} · {generationCompletedImages}/{generationTotalImages} IMAGES</span></div>
          </div>}
          {status === "complete" && generatedOutputs.length > 1 && <div className="output-selector" aria-label="生成结果选择器">
            <div className="output-selector-head"><span>结果浏览</span><b>第 {selectedOutputIndex + 1} / {generatedOutputs.length} 张</b></div>
            {outputBatches.length > 1 && <div className="output-batch-tabs">{outputBatches.map((batch) => <button type="button" key={batch} className={selectedOutput?.batch_index === batch ? "active" : ""} onClick={() => selectGeneratedOutput(generatedOutputs.findIndex((output) => output.batch_index === batch))}>批次 {batch}</button>)}</div>}
            <div className="output-image-tabs">{selectedBatchOutputs.map((output) => <button type="button" key={output.index} className={`${selectedOutputIndex === output.index ? "active" : ""} ${output.transparent_background ? "transparent-preview" : ""}`} title={`批次 ${output.batch_index} · 图片 ${output.image_index} · Seed ${output.seed}`} onClick={() => selectGeneratedOutput(output.index)}><img src={output.url} alt="" /><span>{output.image_index}</span></button>)}</div>
          </div>}
          {status === "running" && <div className="generation-controls"><button className="pause-control" disabled={generationControlBusy === "cancel" || generationTaskStatus === "cancelling" || generationTaskStatus === "pausing"} onClick={() => controlGeneration(generationTaskStatus === "paused" ? "resume" : "pause")}>{generationTaskStatus === "paused" ? <Play size={15} /> : <Pause size={15} />}<span>{generationTaskStatus === "paused" ? "继续生成" : generationTaskStatus === "pausing" ? "正在暂停" : "暂停生成"}</span></button><button className="cancel-control" disabled={generationControlBusy === "cancel" || generationTaskStatus === "cancelling"} onClick={() => controlGeneration("cancel")}><Square size={14} /><span>{generationTaskStatus === "cancelling" ? "正在终止" : "终止生成"}</span></button></div>}
          <div className="generation-info">
            <div title={isSplitModel ? [diffusionModel, textEncoder, textEncoder2, vae].filter(Boolean).join(" · ") : checkpoint}><span>当前模型</span><strong>{isSplitModel ? `${engineLabel} · ${diffusionModel ? diffusionModel.split(/[\\/]/).pop() : "未选择扩散模型"}` : engineLabel}</strong>{isSplitModel && <small>{textEncoder ? textEncoder.split(/[\\/]/).pop() : "未选择编码器"} · {vae ? vae.split(/[\\/]/).pop() : "未选择 VAE"}</small>}</div>
            <div title={inferenceHealth?.memory_reason || "首次生成加载模型时自动评估"}><span>显存档位</span><strong>{inferenceHealth?.memory_label || "AUTO 待评估"}</strong></div>
            <div><span>底模缓存</span><strong>{inferenceHealth?.model_resident ? "GPU 常驻" : inferenceHealth?.model_cached ? "内存缓存" : "未加载"}</strong></div>
          </div>
          {promptConditioning && <p className="conditioning-info">{isAnima ? "Qwen3 + T5 Tokenizer 条件" : isFlux ? "T5-XXL + CLIP-L 条件" : isFlux2 ? "大语言模型三层取样条件" : isKrea2 ? "Qwen3-VL 十二层取样条件" : "CLIP 条件"}：正向 {promptConditioning.tokens} tokens / {promptConditioning.blocks} blocks{promptConditioning.weightedTokens ? ` / ${promptConditioning.weightedTokens} 加权` : ""}；负向 {promptConditioning.negativeTokens} tokens / {promptConditioning.negativeBlocks} blocks{promptConditioning.negativeWeightedTokens ? ` / ${promptConditioning.negativeWeightedTokens} 加权` : ""}</p>}
          {generationWarning && <p className="generation-warning">{generationWarning}</p>}
          <button className="generate-button" title={generationDisabledReason || "开始生成"} onClick={generate} disabled={status === "running" || Boolean(generationDisabledReason)}>
            <span className="generate-icon">{status === "running" ? <RefreshCw className="spin" size={18} /> : <Zap size={18} />}</span>
            <span><strong>{status === "running" ? "正在生成" : generationDisabledReason ? "暂时无法生成" : "开始生成"}</strong><small>{status === "running" ? `批次 ${generationBatchIndex || 1}/${generationBatchCount} · ${generationCompletedImages}/${generationTotalImages} 张` : generationDisabledReason || `${imagesPerBatch} 张 × ${batchCount} 批 · 共 ${imagesPerBatch * batchCount} 张`}</small></span>
            <kbd>Ctrl/⌘ ↵</kbd>
          </button>
          <p className={`backend-note ${inferenceHealth?.status === "ready" && inferenceHealth.cuda ? "online" : ""}`}><span />项目内推理服务 · <b>{inferenceHealth?.status === "ready" ? inferenceHealth.cuda ? inferenceHealth.device : "CUDA 不可用" : inferenceHealth?.status === "error" ? `启动失败：${inferenceHealth.error}` : inferenceHealth?.status === "offline" ? "离线" : "正在启动"}</b></p>
        </aside>
      </section> : activePage === "image" ? <Suspense fallback={<PageLoading label="正在加载图生图" />}><ImageToImagePage
        engine={{
          name: model,
          label: isSplitModel ? `${engineLabel} · ${diffusionModel ? diffusionModel.split(/[\\/]/).pop() : "未选择扩散模型"}` : engineLabel,
          detail: isSplitModel ? [diffusionModel, textEncoder, textEncoder2, vae].filter(Boolean).join(" · ") : checkpoint,
          samplers: activeSamplerNames,
           schedulers: activeSchedulerNames,
           loras,
           // Resolved capability, not the raw health block. Anima has to *declare* each stage while
           // SD/iL only have to not deny it, and `engineAllowsHires` and its siblings are where that
           // asymmetry is decided — re-reading `features` on the page would be a second copy of the
           // rule, free to drift from the one the generate page enforces.
           features: { hires: engineAllowsHires, adetailer: engineAllowsADetailer, rtx: engineAllowsRtx },
           ready: imageEngineReady,
          serviceReady: inferenceHealth?.status === "ready" && Boolean(inferenceHealth.cuda),
          serviceLabel: inferenceHealth?.status === "ready" ? inferenceHealth.cuda ? inferenceHealth.device : "CUDA 不可用" : inferenceHealth?.status === "error" ? `启动失败：${inferenceHealth.error}` : inferenceHealth?.status === "offline" ? "离线" : "正在启动",
        }}
        modelPicker={{
          engines: models,
          switching: modelSwitching,
          loading: modelLoading,
          refreshing: modelsRefreshing,
          error: modelError,
          checkpoints,
          checkpoint,
          checkpointDirectory,
          checkpointMissing,
          assets: [
            { kind: "diffusionModel", label: "扩散模型", value: diffusionModel, options: diffusionModels, directory: diffusionModelDirectory, missing: diffusionModelMissing },
            { kind: "textEncoder", label: isFlux ? "文本编码器 · CLIP-L" : isFlux2 || isKrea2 ? "文本编码器 · 大语言模型" : "文本编码器", value: textEncoder, options: textEncoders, directory: textEncoderDirectory, missing: textEncoderMissing },
            ...(isFlux ? [{ kind: "textEncoder2", label: "文本编码器 · T5-XXL", value: textEncoder2, options: textEncoders2, directory: textEncoderDirectory, missing: textEncoder2Missing }] : []),
            { kind: "vae", label: "VAE", value: vae, options: vaes, directory: vaeDirectory, missing: vaeMissing },
          ],
          onSelectEngine: selectModel,
          onSelectCheckpoint: selectCheckpoint,
          onSelectAsset: selectSplitModelAsset,
          onRefresh: refreshCheckpoints,
        }}
        job={{
          status,
          progress: displayedGenerationProgress,
          phase: generationPhase,
          error: generationError,
          warning: generationWarning,
          livePreview,
          outputs: generatedOutputs,
          selectedIndex: selectedOutputIndex,
          step: generationStep,
          totalSteps: generationTotal,
          batchIndex: generationBatchIndex,
          batchCount: generationBatchCount,
          completedImages: generationCompletedImages,
          totalImages: generationTotalImages,
          elapsed: generationElapsed,
          taskStatus: generationTaskStatus,
          controlBusy: generationControlBusy,
        }}
        settings={imageToImage}
        source={imageSource}
        onSettingsChange={setImageToImage}
        onSourceChange={setImageSource}
        onGenerate={generateFromImage}
        onControl={(action) => action === "reset" ? setStatus("idle") : controlGeneration(action)}
        onSelectOutput={selectGeneratedOutput}
        onOpenViewer={openImageViewer}
        onOpenLoraManager={() => { setLoraCategory("mounted"); setLoraManagerOpen(true); }}
         onAddToGallery={() => setGalleryAddOpen(true)}
         onNotice={(message, error = false) => setAppNotice({ message, error })}
         postprocess={{
           upscalers: compatibleUpscalers,
           upscalerRuntime: inferenceHealth?.upscalers?.runtime_available === true,
           adetailerModels,
           adetailerInfo,
           rtxHealth,
           ultraLow: ultraLowMode,
         }}
       /></Suspense> : activePage === "gallery" ? <Suspense fallback={<PageLoading label="正在加载画廊" />}><GalleryPage
        currentSettings={galleryCardSettings(workspaceSnapshot.current)}
        focus={galleryFocus}
         onApplySettings={applyGallerySettings}
         onApplyPrompt={applyPromptLibraryEntry}
        onNotice={(message, error = false) => setAppNotice({ message, error })}
      /></Suspense> : <Suspense fallback={<PageLoading label="正在加载工具箱" />}><ToolboxPage onApplyImageParameters={applyImageInfoParameters} onDownloaded={({ kind, engine, targets: completedTargets }) => {
        const targets = completedTargets?.length ? completedTargets : [{ kind, engine }];
        if (targets.some((target) => (target.kind === "checkpoint" && target.engine === model) || (SPLIT_MODEL_ENGINES.includes(model) && (target.engine === model || ["diffusion_model", "text_encoder", "vae", "config"].includes(target.kind))))) void refreshCheckpoints();
        if (targets.some((target) => target.kind === "lora" && target.engine === model)) void refreshLoras();
        if (targets.some((target) => target.kind === "yolo")) void refreshADetailerModels();
        if (targets.some((target) => target.kind === "upscaler")) void refreshUpscalers();
      }} /></Suspense>}
      {galleryAddOpen && <AddToGalleryDialog
        outputs={generatedOutputs}
        selectedOutputIndex={selectedOutputIndex}
        settings={generatedSettings ? galleryCardSettings(generatedSettings, { record: true }) : galleryCardSettings(workspaceSnapshot.current)}
        onClose={() => setGalleryAddOpen(false)}
        onSaved={({ collectionId, cardId, count, imageCount }) => {
          setGalleryAddOpen(false);
          setGalleryFocus({ collectionId, cardId, nonce: Date.now() });
          setAppNotice({ message: `已将 ${imageCount} 张图片加入精选集${count > 1 ? `，创建 ${count} 张卡片` : ""}`, error: false });
        }}
      />}
      {promptPresetDialog && <PromptPresetDialog dialog={promptPresetDialog} records={promptPresets.records} running={status === "running"} onSave={savePromptPreset} onRequestClose={() => setPromptPresetDialog(null)} />}
      {promptPresetDelete && <PromptPresetDeleteDialog record={promptPresetDelete} running={status === "running"} onConfirm={confirmDeletePromptPreset} onClose={() => setPromptPresetDelete(null)} />}
      {loraManagerOpen && (
        <div className="lora-modal-backdrop" onMouseDown={(event) => event.target === event.currentTarget && setLoraManagerOpen(false)}>
          <section className={`lora-modal ${loraManagerMaximized ? "maximized" : ""}`} role="dialog" aria-modal="true" aria-labelledby="lora-manager-title">
            <header className="lora-modal-head">
              <div><span className="eyebrow">MODEL ASSET LIBRARY</span><h2 id="lora-manager-title">LoRA 管理</h2><p>{model} 引擎 · 已挂载 {loras.length} 个</p></div>
              <div className="lora-modal-actions">
                <button type="button" className="asset-refresh" title="刷新 LoRA 列表" aria-label="刷新 LoRA 列表" disabled={status === "running" || lorasRefreshing || modelSwitching || loraWorkspaceLocked || !shouldPersistMountedLoras} onClick={refreshLoras}><RefreshCw className={lorasRefreshing ? "spin" : ""} size={16} /></button>
                <button type="button" className="modal-close" title={loraManagerMaximized ? "还原窗口" : "最大化窗口"} aria-label={loraManagerMaximized ? "还原 LoRA 管理器窗口" : "最大化 LoRA 管理器窗口"} onClick={() => setLoraManagerMaximized((current) => !current)}>{loraManagerMaximized ? <Minimize2 size={18} /> : <Maximize2 size={18} />}</button>
                <button type="button" className="modal-close" title="在新标签页打开完整 LoRA 资产页" aria-label="在新标签页打开 LoRA 管理器" onClick={openLoraManagerPage}><ExternalLink size={18} /></button>
                <button type="button" className="modal-close" aria-label="关闭 LoRA 管理器" onClick={() => setLoraManagerOpen(false)}><X size={20} /></button>
              </div>
            </header>
            <div className="lora-modal-body">
              <nav className="lora-categories" aria-label="LoRA 分类">
                <button className={`lora-mounted-nav ${loraCategory === "mounted" ? "active" : ""}`} onClick={() => setLoraCategory("mounted")}>
                  <span>已挂载</span><b>{loras.length}</b>
                </button>
                <button className={`lora-mounted-nav ${loraCategory === "groups" ? "active" : ""}`} onClick={() => setLoraCategory("groups")}>
                  <span>组合</span><b>{activeLoraGroups.length}</b>
                </button>
                <span>分类</span>
                {loraLibrary.map((category) => (
                  <button className={loraCategory === category.id ? "active" : ""} key={category.id} onClick={() => setLoraCategory(category.id)}>
                    <span>{category.label}</span><b>{category.models.length}</b>
                  </button>
                ))}
                {loraLoading && <p>正在扫描目录...</p>}
                <div className="lora-directory" title={loraDirectory}><span>模型路径</span><p>{loraDirectory || "正在解析..."}</p></div>
              </nav>
              <div className="lora-browser">
                {loraCategory === "mounted" ? <div className="lora-mounted-manager">
                    <div className="lora-browser-head">
                      <div><h3>已挂载 LoRA 管理</h3><p>调整权重与启用状态 · 按住 Shift 可精确到 0.01 · 已启用的组合以独立分块显示</p></div>
                  </div>
                    <LoraMountPanel
                      variant="modal"
                      loras={loras}
                      groups={activeLoraGroups}
                      locked={loraDragLocked}
                      previewUrlFor={loraPreviewUrlFor}
                      onUpdateLoras={commitMountedLoras}
                      onUpdateGroups={commitLoraGroups}
                    />
                </div> : loraCategory === "groups" ? <div className="lora-mounted-manager">
                    <LoraGroupPanel
                      groups={activeLoraGroups}
                      loras={loras}
                      locked={loraDragLocked}
                      onUpdateGroups={commitLoraGroups}
                      onUpdateLoras={commitMountedLoras}
                      onNotice={(message) => setAppNotice({ message, error: false })}
                    />
                </div> : <>
                <div className="lora-browser-head">
                  <div><h3>{activeLoraCategory?.label || "LoRA"}</h3><p>{activeLoraCategory?.directory || loraError || "选择分类浏览模型"}</p></div>
                  <label className="lora-search"><Search size={15} /><input value={loraSearch} onChange={(event) => setLoraSearch(event.target.value)} placeholder="搜索 LoRA" /></label>
                </div>
                {activeLoraCategory?.shared ? <div className="lora-modal-shared-view">
                  {(activeLoraCategory.roots || []).map((root) => {
                    const folders = (root.folders || []).map((folder) => ({ ...folder, models: folder.models.filter(matchesLoraSearch) })).filter((folder) => folder.models.length);
                    return <section className="lora-modal-shared-root" key={root.id}>
                      <header><div><Share2 size={15} /><strong>{root.label}</strong><code title={root.path}>{root.path}</code></div><span>{root.files} 个文件 · {formatFileSize(root.bytes)}</span></header>
                      {folders.length ? folders.map((folder) => {
                        const folderKey = `${root.id}/${folder.name}`;
                        const collapsed = collapsedModalSharedFolders[folderKey] !== false;
                        return <div className={`lora-modal-shared-folder ${collapsed ? "collapsed" : ""}`} key={folderKey}>
                          <button type="button" aria-expanded={!collapsed} onClick={() => setCollapsedModalSharedFolders((current) => ({ ...current, [folderKey]: !collapsed }))}><ChevronDown size={14} /><FolderOpen size={15} /><strong>{folder.label}</strong><b>{folder.models.length}</b></button>
                          {!collapsed && <div className="lora-library-grid">{folder.models.map((item) => renderModalLoraCard(item, folder.label))}</div>}
                        </div>;
                      }) : <p className="lora-modal-shared-empty">{loraSearch ? "该共享目录中没有匹配的文件" : "该共享目录为空"}</p>}
                    </section>;
                  })}
                </div> : <div className="lora-library-grid">
                  {visibleLoras.map((item) => renderModalLoraCard(item, activeLoraCategory.label))}
                  {!visibleLoras.length && <div className="lora-library-empty"><Layers3 size={30} /><strong>{loraSearch ? "没有匹配的 LoRA" : "该分类暂无 LoRA"}</strong><p>{loraSearch ? "尝试其他关键词" : `将模型文件放入 ${activeLoraCategory?.directory || "对应分类目录"}`}</p></div>}
                  {loraError && <div className="lora-library-empty error"><X size={30} /><strong>LoRA 目录读取失败</strong><p>{loraError}</p></div>}
                </div>}
                </>}
              </div>
            </div>
            <footer className="lora-modal-foot"><span>在已挂载标签管理权重和状态；在分类标签浏览并挂载或卸载 LoRA</span><button onClick={() => setLoraManagerOpen(false)}>完成 · {loras.length} 已挂载</button></footer>
          </section>
        </div>
      )}
      {loraDetail && (() => {
        const item = loraLibrary.flatMap((category) => category.models).find((modelItem) => modelItem.value === loraDetail.value) || loraDetail;
        const lookup = loraLookups[loraDetail.value];
        return <LoraDetailsDialog item={item} metadata={item.metadata} loading={lookup?.loading} error={lookup?.error} onRefresh={() => void lookupLora(item, true, loraDetail.categoryId)} onClose={() => setLoraDetail(null)} />;
      })()}
      {settingsOpen && <div className="settings-backdrop" onMouseDown={(event) => event.target === event.currentTarget && !reconfiguring && setSettingsOpen(false)}>
        <section className="settings-modal" role="dialog" aria-modal="true" aria-labelledby="settings-title">
          <header className="settings-head">
            <div><span className="eyebrow">WORKSPACE CONTROL</span><h2 id="settings-title">设置</h2><p>本地工作区、运行环境与应用信息</p></div>
            <button className="modal-close" disabled={reconfiguring} onClick={() => setSettingsOpen(false)}><X size={20} /></button>
          </header>
          <div className="settings-body">
            <nav className="settings-nav" aria-label="设置分类">
              <button className={settingsTab === "general" ? "active" : ""} onClick={() => setSettingsTab("general")}><Settings2 size={16} /><span>常规设置<small>工作区和界面</small></span></button>
              <button className={settingsTab === "models" ? "active" : ""} onClick={() => setSettingsTab("models")}><FolderOpen size={16} /><span>模型目录<small>本地路径 / 共享来源</small></span></button>
              <button className={settingsTab === "performance" ? "active" : ""} onClick={() => setSettingsTab("performance")}><SlidersHorizontal size={16} /><span>性能调度<small>显存 / 精度 / 内核</small></span></button>
              <button className={settingsTab === "theme" ? "active" : ""} onClick={() => setSettingsTab("theme")}><Palette size={16} /><span>主题设置<small>界面与 Logo 调色板</small></span></button>
              <button className={settingsTab === "plugins" ? "active" : ""} onClick={() => setSettingsTab("plugins")}><Blocks size={16} /><span>插件扩展<small>发现与启用管理</small></span></button>
              <button className={settingsTab === "environment" ? "active" : ""} onClick={() => setSettingsTab("environment")}><Cpu size={16} /><span>运行环境<small>Python / CUDA</small></span></button>
              <button className={settingsTab === "logs" ? "active" : ""} onClick={() => setSettingsTab("logs")}><FileText size={16} /><span>运行日志<small>故障诊断文件</small></span></button>
              <button className={settingsTab === "about" ? "active" : ""} onClick={() => setSettingsTab("about")}><BrandLogo className="settings-nav-logo" /><span>关于<small>XiriaCanvas AI</small></span></button>
              <div className="settings-save-state"><i /><span>工作区已自动保存<small>刷新或重配后自动恢复</small></span></div>
            </nav>
              <div className="settings-content">
              {settingsTab === "general" && <section className="settings-section">
                <div className="settings-section-title"><span>GENERAL</span><h3>常规设置</h3><p>这些设置和全部创作参数会自动保存在当前项目中。</p></div>
                <div className="settings-option-list">
                <div className="settings-option"><div><strong>工作区自动保存</strong><p>保存 Prompt、底模、采样参数、ADetailer、画布和全部 LoRA 挂载状态。</p></div><span className="settings-status enabled">始终启用</span></div>
                <div className="settings-option"><div><strong>界面布局</strong><p>左侧配置面板的宽度与折叠状态，保存在当前浏览器。恢复默认只会还原面板布局，不会改动任何创作参数或其他设置。</p></div><button type="button" className="settings-layout-reset" disabled={isDefaultWorkspaceLayout(workspaceLayout)} onClick={resetWorkspaceLayout}><PanelLeft size={13} />{isDefaultWorkspaceLayout(workspaceLayout) ? "已是默认布局" : "恢复默认布局"}</button></div>
                <div className="settings-option"><div><strong>界面动态效果</strong><p>控制画廊卡片入场、弹窗、提示条以及界面过渡。偏好保存在当前浏览器，关闭后仍保留功能反馈。</p></div><button type="button" role="switch" aria-label="启用界面动态效果" aria-checked={motionEnabled} className={`settings-switch ${motionEnabled ? "active" : ""}`} onClick={() => setMotionEnabled((current) => !current)}><i /></button></div>
                </div>
                <div className="settings-workspace-summary"><span>当前创作状态</span><div><b>{model}</b><b>{isSplitModel ? diffusionModel || "未选择扩散模型" : checkpoint || "未选择底模"}</b><b>{loras.length} 个 LoRA</b><b>{size.width} × {size.height}</b></div></div>
              </section>}
              {settingsTab === "models" && <section className="settings-section shared-model-settings-section">
                <div className="settings-section-title"><span>MODEL SOURCES</span><h3>模型目录</h3><p>管理当前项目以只读方式发现的外部模型来源。模型下载器仍只写入本项目的本地模型目录。</p></div>
                <Suspense fallback={<div className="settings-inline-loading"><RefreshCw className="spin" size={16} />正在加载模型目录</div>}>
                  <SharedModelDirectories onChanged={() => { void refreshCheckpoints(); void refreshLoras(); }} />
                </Suspense>
              </section>}
              {settingsTab === "performance" && <section className="settings-section performance-section">
                <div className="settings-section-title"><span>RUNTIME MATRIX</span><h3>性能调度</h3><p>结合动态显存预算与现代 PyTorch / Diffusers 内核。应用设置会卸载当前缓存底模，下次生成时按新策略构建。</p></div>
                <div className="performance-mode-switch"><button type="button" className={performanceEditorMode === "recommended" ? "active" : ""} onClick={() => setPerformanceEditorMode("recommended")}><Sparkles size={14} /><span><strong>新手推荐设置</strong><small>按显存和使用场景一键选择</small></span></button><button type="button" className={performanceEditorMode === "manual" ? "active" : ""} onClick={() => setPerformanceEditorMode("manual")}><SlidersHorizontal size={14} /><span><strong>手动选择</strong><small>逐项控制显存、内核和精度</small></span></button></div>
                <div className="performance-live-strip">
                  <div><span>GPU</span><strong>{performanceCapabilities.device || inferenceHealth?.device || "正在检测"}</strong></div>
                  <div><span>当前显存路径</span><strong>{runtimeMemoryMode || `计划 · ${configuredPerformanceLabel("memory_mode")}`}</strong></div>
                  <div><span>当前注意力</span><strong>{runtimeAttentionBackend || `计划 · ${configuredPerformanceLabel("attention_backend")}`}</strong></div>
                  <div><span>计算精度</span><strong>{runtimeComputeDtype || `计划 · ${configuredPerformanceLabel("compute_dtype")}`}</strong></div>
                </div>
                <div className="vram-wall-card">
                  <div className="vram-wall-head">
                    <div><span>VRAM HARD WALL</span><strong>显存占用上限</strong><p>按当前显卡生成安全范围。固定上限同时约束任务预算和 CUDA allocator；触墙后自动切换分组卸载，不降低模型精度、步数或图片尺寸。</p></div>
                    <div className="vram-wall-mode"><b>{vramLimitAutomatic ? "AUTO" : `${vramLimitValue.toFixed(1)} GB`}</b><button type="button" role="switch" aria-label="自动管理显存占用上限" aria-checked={vramLimitAutomatic} className={`settings-switch ${vramLimitAutomatic ? "active" : ""}`} onClick={() => setPerformanceConfig((current) => ({ ...current, vram_limit_gb: Number(current.vram_limit_gb || 0) > 0 ? 0 : Number(vramLimitMaxGb.toFixed(1)) }))}><i /></button><small>自动管理</small></div>
                  </div>
                  <div className={`vram-wall-control ${vramLimitAutomatic ? "automatic" : ""}`}>
                    <div className="vram-wall-scale"><span>{vramLimitMinGb.toFixed(1)} GB</span><strong>{vramLimitAutomatic ? `自适应 · 最高 ${vramLimitMaxGb.toFixed(1)} GB` : `${vramLimitValue.toFixed(1)} GB 上限墙`}</strong><span>{vramLimitMaxGb.toFixed(1)} GB</span></div>
                    <input type="range" min={vramLimitMinGb} max={vramLimitMaxGb} step="0.1" value={vramLimitValue} disabled={vramLimitAutomatic || !performanceCapabilities.cuda} aria-label="显存占用上限" aria-valuetext={vramLimitAutomatic ? "自动管理" : `${vramLimitValue.toFixed(1)} GB`} style={{ "--vram-wall-progress": `${vramLimitPercent}%` }} onChange={(event) => setPerformanceConfig((current) => ({ ...current, vram_limit_gb: Number(Number(event.target.value).toFixed(1)) }))} />
                    <div className="vram-wall-readout"><span>物理显存 {vramGb ? `${vramGb.toFixed(1)} GB` : "检测中"}</span><span>系统保留 {vramLimitInfo.platform_reserve_bytes ? `${(vramLimitInfo.platform_reserve_bytes / 1024 ** 3).toFixed(1)} GB` : "自动"}</span><span>{vramLimitAutomatic ? "每次任务按实时余量选路" : "超过墙值禁止继续分配"}</span></div>
                  </div>
                </div>
                {performanceEditorMode === "recommended" && <div className="performance-preset-grid">
                  {performancePresets.map((preset) => <button type="button" key={preset.id} className={activePerformancePreset === preset.id ? "active" : ""} disabled={performanceLoading} onClick={() => setPerformanceConfig((current) => ({ ...preset.settings, vram_limit_gb: current.vram_limit_gb }))}>
                    <span className="preset-index">{String(performancePresets.indexOf(preset) + 1).padStart(2, "0")}</span>
                    <span><strong>{preset.label}{recommendedPerformancePreset === preset.id && <em>当前显卡建议</em>}</strong><small>{preset.range}</small><p>{preset.detail}</p></span>
                    <i />
                  </button>)}
                </div>}
                {performanceEditorMode === "recommended" && activePerformancePreset === "ultra-low" && <div className="ultra-low-notice"><strong>1024 大型模型生存模式</strong><p>应用后会关闭过程预览、Hires.fix、ADetailer、RTX VSR 和已挂载 LoRA；正负条件改为串行计算，单张耗时可能明显增加。建议系统分页文件至少保留 24 GB 可用空间。</p></div>}
                {performanceEditorMode === "manual" && <div className="performance-manual-panel">{[
                  ["memory_mode", "显存调度", "按任务峰值预算决定模型驻留和 CPU 卸载方式"],
                  ["attention_backend", "注意力内核", "新显卡优先使用当前 PyTorch 原生 SDPA，不套用旧架构白名单"],
                  ["compute_dtype", "权重计算精度", "BF16 仅在硬件报告支持时实际启用，否则安全回退 FP16"],
                  ["vae_mode", "VAE 解码路径", "控制最终图像解码的峰值显存与速度"],
                ].map(([key, title, detail]) => <div className="performance-group" key={key}>
                  <div className="performance-group-head"><span><strong>{title}</strong><small>{detail}</small></span><b>{performanceChoices[key].find(([value]) => value === performanceConfig[key])?.[1]}</b></div>
                  <div className={`performance-choice-grid choice-grid-${key}`}>
                    {performanceChoices[key].map(([value, label, description]) => {
                      const unavailable = (key === "attention_backend" && value === "xformers" && performanceCapabilities.xformers === false)
                        || (key === "attention_backend" && value === "sage" && performanceCapabilities.sage === false)
                        || (key === "compute_dtype" && value === "bf16" && performanceCapabilities.bf16 === false);
                      return <button type="button" key={value} className={performanceConfig[key] === value ? "active" : ""} disabled={performanceLoading || unavailable} onClick={() => setPerformanceConfig((current) => ({ ...current, [key]: value }))}><i /><span><strong>{label}</strong><small>{unavailable ? "当前环境不可用" : description}</small></span></button>;
                    })}
                  </div>
                </div>)}</div>}
                {performanceEditorMode === "manual" && <div className="performance-toggle-row">
                  <div><strong>CUDA 数学策略</strong><p>平衡模式允许 TF32 与 cuDNN 自适应选核；严格模式关闭这些加速以减少跨设备数值差异。</p></div>
                  <div className="performance-segment"><button type="button" className={performanceConfig.cuda_math === "balanced" ? "active" : ""} onClick={() => setPerformanceConfig((current) => ({ ...current, cuda_math: "balanced" }))}>平衡加速</button><button type="button" className={performanceConfig.cuda_math === "strict" ? "active" : ""} onClick={() => setPerformanceConfig((current) => ({ ...current, cuda_math: "strict" }))}>严格计算</button></div>
                </div>}
                <div className="performance-toggle-row">
                  <div><strong>任务后保留底模</strong><p>开启可加快连续生成；关闭则每个任务结束后释放 GPU 与内存中的模型缓存。</p></div>
                  <button type="button" role="switch" aria-checked={performanceConfig.keep_model_cached} className={`settings-switch ${performanceConfig.keep_model_cached ? "active" : ""}`} onClick={() => setPerformanceConfig((current) => ({ ...current, keep_model_cached: !current.keep_model_cached }))}><i /></button>
                </div>
                <div className="performance-toggle-row">
                  <div><strong>使用共享显存</strong><p>开启时采用常规显存余量。关闭后软件会主动扩大安全余量，降低驱动触发系统内存接管的概率；该选项不能关闭操作系统的共享显存机制。</p></div>
                  <button type="button" role="switch" aria-checked={performanceConfig.allow_shared_memory} className={`settings-switch ${performanceConfig.allow_shared_memory ? "active" : ""}`} onClick={() => setPerformanceConfig((current) => ({ ...current, allow_shared_memory: !current.allow_shared_memory }))}><i /></button>
                </div>
                <div className="performance-toggle-row">
                  <div><strong>模型哈希计算</strong><p>计算底模 SHA-256 并写入生成图片元数据，便于识别模型；关闭可节省首次加载时的大文件读取时间。</p></div>
                  <button type="button" role="switch" aria-checked={performanceConfig.calculate_model_hash} className={`settings-switch ${performanceConfig.calculate_model_hash ? "active" : ""}`} onClick={() => setPerformanceConfig((current) => ({ ...current, calculate_model_hash: !current.calculate_model_hash }))}><i /></button>
                </div>
                <div className="performance-toggle-row">
                  <div><strong>采样后卸载再解码</strong><p>开启后先将采样模型转入系统内存并释放显存，再仅加载 VAE 完成解码。可明显降低采样与解码叠加的显存峰值，但下次生成需要重新恢复模型。</p></div>
                  <button type="button" role="switch" aria-checked={performanceConfig.staged_vae_decode} className={`settings-switch ${performanceConfig.staged_vae_decode ? "active" : ""}`} onClick={() => setPerformanceConfig((current) => ({ ...current, staged_vae_decode: !current.staged_vae_decode }))}><i /></button>
                </div>
                <div className="performance-toggle-row">
                  <div><strong>Transformer 编译加速</strong><p>{performanceCapabilities.triton === false ? "需要在配置器中安装 Triton；未安装时保持关闭。" : "把 Anima 的 28 个 Transformer 块交给 Inductor 编译，融合逐元素算子。每种画布尺寸首次生成需要一次编译（约 10 秒），之后每步更快；同种子出图会改变。仅对 Anima 生效。"}</p></div>
                  <button type="button" role="switch" aria-checked={performanceConfig.compile_transformer} disabled={performanceCapabilities.triton === false} className={`settings-switch ${performanceConfig.compile_transformer ? "active" : ""}`} onClick={() => setPerformanceConfig((current) => ({ ...current, compile_transformer: !current.compile_transformer }))}><i /></button>
                </div>
                <div className={`nvfp4-status ${performanceCapabilities.nvfp4?.runtime_ready ? "detected" : ""}`}>
                  <div><span>BLACKWELL PATH</span><strong>NVFP4 实验能力</strong><p>{performanceCapabilities.nvfp4?.hardware_supported ? performanceCapabilities.nvfp4?.runtime_ready ? "已检测到 50 系硬件和量化运行库；为保护单文件 SDXL、LoRA 与 CPU offload 兼容性，本版本仅检测不启用。" : "显卡支持 NVFP4，但项目缺少经过验证的 TorchAO / ModelOpt 量化运行链。" : "当前显卡不提供 NVFP4 Tensor Core 路径；不影响 FP16 / BF16 与 SDPA 优化。"}</p></div>
                  <b>{performanceCapabilities.nvfp4?.runtime_ready ? "RUNTIME DETECTED" : performanceCapabilities.nvfp4?.hardware_supported ? "RUNTIME REQUIRED" : "NOT APPLICABLE"}</b>
                </div>
                <div className="performance-actions"><button type="button" onClick={() => setPerformanceConfig((current) => ({ ...DEFAULT_PERFORMANCE, vram_limit_gb: current.vram_limit_gb }))} disabled={performanceSaving}><RefreshCw size={14} />恢复建议值</button><button type="button" className="performance-apply" onClick={savePerformanceSettings} disabled={performanceLoading || performanceSaving || status === "running"}>{performanceSaving ? <RefreshCw className="spin" size={14} /> : <Check size={14} />}{performanceSaving ? "正在应用" : "应用性能方案"}</button></div>
                {status === "running" && <p className="settings-notice">生成任务运行期间不能修改性能方案。</p>}
                {performanceError && <p className="settings-error">{performanceError}</p>}
              </section>}
              {settingsTab === "theme" && <section className="settings-section theme-section">
                <div className="settings-section-title"><span>THEME PALETTE</span><h3>主题设置</h3><p>主题会立即应用并自动保存，重新打开应用和加载界面时继续使用。</p></div>
                <div className="theme-toolbar">
                  <div><BrandLogo className="theme-preview-logo" /><span><strong>实时主题预览</strong><small>{theme.light ? "白色背景主题已启用" : `${theme.overall} · 暗色背景主题`}</small></span></div>
                  <div className="theme-toolbar-actions"><button type="button" role="switch" aria-checked={theme.light} className={`theme-mode-toggle ${theme.light ? "active" : ""}`} onClick={toggleLightTheme}><i /><span>黑 / 白主题</span></button><button type="button" className="theme-reset" onClick={resetTheme}><RefreshCw size={13} />恢复默认主题</button></div>
                </div>
                <div className="theme-group-label"><span>整体主题调色板</span><small>影响整个工具的强调色，并自动派生 Logo 配色</small></div>
                <ThemeColorField label="整体主题" detail="界面强调色 / 自动生成完整 Logo 配色" value={theme.overall} onChange={updateOverallTheme} />
                <div className="theme-group-label logo-palette-label"><span>Logo 主题调色板</span><small>以下四项可在自动结果上继续手动微调</small></div>
                <div className="theme-palette-grid">
                  <ThemeColorField label="主题色" detail="外环 / 主题角块" value={theme.accent} onChange={(value) => updateLogoTheme("accent", value)} />
                  <ThemeColorField label="主体色" detail="深色角块 / 中央 X" value={theme.ink} onChange={(value) => updateLogoTheme("ink", value)} />
                  <ThemeColorField label="圆盘色" detail="Logo 内部背景" value={theme.disc} onChange={(value) => updateLogoTheme("disc", value)} />
                  <ThemeColorField label="辅助线" detail="细环 / 第三轨道" value={theme.guide} onChange={(value) => updateLogoTheme("guide", value)} />
                </div>
                {theme.light && <p className="theme-mode-note">白色主题只切换界面背景、面板和文字明暗；整体主题色与 Logo 调色板保持不变。</p>}
              </section>}
              {settingsTab === "plugins" && <section className="settings-section plugins-section">
                <div className="settings-section-title"><span>EXTENSION PORT</span><h3>插件扩展</h3><p>把插件文件夹放进项目根目录的 <code>plugins</code> 后在这里管理。宿主只读取 <code>plugins/&lt;id&gt;/plugin.json</code>。</p></div>
                <div className="plugins-execution-note">
                  <ShieldAlert size={16} />
                  <div><strong>本版本不执行任何插件代码</strong><p>插件不会被导入、运行、打包或通过 HTTP 提供访问；<code>/plugins/*</code> 一律返回 404。这里的启用开关只记录你的选择并保存在 <code>state-cache/plugins.json</code>，它不是执行授权，也不会授予文件、网络、模型或 GPU 权限。新放入的插件默认关闭，需要你显式启用。</p></div>
                </div>
                <div className="plugins-toolbar">
                  <span><b>plugins/</b><small>{pluginsLoading ? "正在扫描..." : `${pluginSummary.total} 个插件 · 已启用 ${pluginSummary.enabled}${pluginSummary.needsAttention ? ` · ${pluginSummary.needsAttention} 个需要处理` : ""}`}</small></span>
                  <button type="button" onClick={refreshPlugins} disabled={pluginsLoading || Boolean(pluginPendingId)}><RefreshCw className={pluginsLoading ? "spin" : ""} size={14} />重新扫描</button>
                </div>
                <div className="plugins-list" aria-label="插件列表">
                  {!pluginsLoading && pluginList.length === 0 && <p className="plugins-empty">当前没有插件。在 <code>plugins/</code> 下新建一个文件夹，文件夹名与 <code>plugin.json</code> 的 <code>id</code> 保持一致即可被发现。</p>}
                  {pluginList.map((plugin) => {
                    const presentation = pluginStatePresentation(plugin.state);
                    const toggleAvailable = pluginToggleAvailable(plugin);
                    return <article key={plugin.id} className={`plugin-card ${plugin.enabled ? "enabled" : ""}`}>
                      <div className="plugin-card-main">
                        <div className="plugin-card-identity">
                          <strong>{plugin.name}</strong>
                          <small><code>{plugin.id}</code>{plugin.version && <em>v{plugin.version}</em>}</small>
                        </div>
                        <span className={`plugin-state ${presentation.tone}`}>{presentation.label}</span>
                      </div>
                      {plugin.description && <p className="plugin-card-description">{plugin.description}</p>}
                      <p className="plugin-card-hint">{presentation.hint}</p>
                      {plugin.diagnostics.length > 0 && <ul className="plugin-diagnostics">
                        {plugin.diagnostics.map((code) => <li key={code}><code>{code}</code>{pluginDiagnosticMessage(code)}</li>)}
                      </ul>}
                      <div className="plugin-card-foot">
                        <span>{plugin.enabled ? "已启用（仍不会执行）" : toggleAvailable ? "已关闭" : "不可启用"}</span>
                        <div className="plugin-card-actions">
                          <button type="button" disabled={Boolean(pluginPendingId)} onClick={() => revealPluginFolder(plugin.id)}><FolderOpen size={13} />打开文件夹</button>
                          <button type="button" className="plugin-remove" disabled={Boolean(pluginPendingId)} onClick={() => removePlugin(plugin)}><Trash2 size={13} />移除</button>
                          <button
                            type="button"
                            role="switch"
                            aria-label={`启用插件 ${plugin.name}`}
                            aria-checked={plugin.enabled}
                            disabled={!toggleAvailable || Boolean(pluginPendingId)}
                            className={`settings-switch ${plugin.enabled ? "active" : ""}`}
                            onClick={() => setPluginEnabled(plugin.id, !plugin.enabled)}
                          ><i /></button>
                        </div>
                      </div>
                    </article>;
                  })}
                </div>
                {pluginRegistryDiagnostics.length > 0 && <ul className="plugin-diagnostics registry">
                  {pluginRegistryDiagnostics.map((diagnostic) => <li key={diagnostic.code}><code>{diagnostic.code}</code>{pluginDiagnosticMessage(diagnostic.code)}</li>)}
                </ul>}
                <p className="plugins-doc-note">「打开文件夹」在系统文件管理器中显示该插件目录，不会运行其中任何文件；「移除」会永久删除该目录及其全部内容。编写规范见 <code>.Structure/13-PLUGIN-DEVELOPMENT-GUIDELINES-ZH.md</code>；架构与 API 见 <code>.Structure/12-PLUGIN-ARCHITECTURE-API.md</code>。</p>
                {pluginsError && <p className="settings-error">{pluginsError}</p>}
              </section>}
                {settingsTab === "environment" && <section className="settings-section">
                <div className="settings-section-title"><span>ENVIRONMENT</span><h3>运行环境</h3><p>当前 WebUI 与项目隔离 Python 推理环境。</p></div>
                <div className="environment-grid">{environmentItems.map(([label, value]) => <div key={label}><span>{label}</span><strong title={String(value)}>{value}</strong></div>)}</div>
                <div className="environment-path"><span>Python 可执行文件</span><code>{inferenceHealth?.python_executable || "正在读取..."}</code></div>
                <div className="reconfigure-card">
                  <div><strong>重新配置环境</strong><p>进入环境配置器以重新选择 PyTorch、CUDA Runtime 与 xformers。进入前会保存当前工作区；不做改动时可直接返回并恢复。</p></div>
                  <button disabled={reconfiguring || status === "running"} onClick={enterConfigurator}>{reconfiguring ? <><RefreshCw className="spin" size={15} />正在切换</> : <><Settings2 size={15} />进入环境配置</>}</button>
                </div>
                {status === "running" && <p className="settings-notice">生成任务运行期间不能切换环境配置。</p>}
                {settingsError && <p className="settings-error">{settingsError}</p>}
              </section>}
              {settingsTab === "logs" && <section className="settings-section logs-section">
                <div className="settings-section-title"><span>LOCAL DIAGNOSTICS</span><h3>运行日志</h3><p>环境配置失败、CPU 模式、推理启动异常，以及包含显存优化路径在内的生成失败都会自动写入项目根目录的 <code>logs</code> 文件夹。</p></div>
                <div className="logs-toolbar"><span><b>logs/</b><small>{diagnosticLogsLoading ? "正在读取..." : `${diagnosticLogs.length} 个日志文件`}</small></span><div><button type="button" onClick={refreshDiagnosticLogs} disabled={diagnosticLogsLoading || diagnosticLogsClearing}><RefreshCw className={diagnosticLogsLoading ? "spin" : ""} size={14} />刷新</button><button type="button" className="logs-clear" onClick={clearDiagnosticLogs} disabled={diagnosticLogs.length === 0 || diagnosticLogsClearing}>{diagnosticLogsClearing ? <RefreshCw className="spin" size={14} /> : <Trash2 size={14} />}{diagnosticLogsClearing ? "清空中" : "清空日志"}</button></div></div>
                <div className="logs-browser">
                  <div className="logs-list" aria-label="日志文件列表">
                    {diagnosticLogs.length === 0 && !diagnosticLogsLoading && <p>当前没有诊断日志。</p>}
                    {diagnosticLogs.map((log) => <button key={log.name} type="button" className={selectedDiagnosticLog === log.name ? "active" : ""} onClick={() => selectDiagnosticLog(log.name)}><FileText size={14} /><span><b>{log.name}</b><small>{new Date(log.modified_at).toLocaleString("zh-CN", { hour12: false })} · {(log.bytes / 1024).toFixed(log.bytes < 1024 ? 1 : 0)} KB</small></span></button>)}
                  </div>
                  <pre className="logs-content">{diagnosticLogReading ? "正在读取日志..." : diagnosticLogContent || "选择左侧日志文件以查看故障详情。"}</pre>
                </div>
                {diagnosticLogsError && <p className="settings-error">{diagnosticLogsError}</p>}
              </section>}
              {settingsTab === "about" && <section className="settings-section about-section">
                <div className="about-mark"><BrandLogo className="about-logo" /><div><span>XIRIA CANVAS</span><strong>XiriaCanvas AI</strong><small>LOCAL GENERATIVE WORKSPACE · {APP_VERSION}</small></div></div>
                <p className="about-copy">面向本地 Stable Diffusion 与 Illustrious 工作流的创作界面。模型、Prompt 和生成结果均由当前设备处理。</p>
                <div className="about-facts"><div><span>运行模式</span><strong>Local First</strong></div><div><span>推理框架</span><strong>PyTorch + Diffusers</strong></div><div><span>工作区标识</span><strong title={inferenceHealth?.workspace_id}>{inferenceHealth?.workspace_id?.slice(0, 12) || "--"}</strong></div><div><span>数据处理</span><strong>仅在本地设备</strong></div></div>
                <div className="about-note"><strong>数据与隐私</strong><p>工作区参数与性能方案保存在项目的 <code>state-cache</code> 目录。底模、LoRA 与输出文件保留在项目配置目录中，不会因进入设置或环境配置器而清除。</p></div>
                <div className="manual-update-card"><div><span>ONLINE PROGRAM UPDATE</span><strong>在线更新程序</strong><p>联网检查是否有新版本。发现新版本后会先征求确认，再下载官方发布的更新包并按手动更新的同一流程校验、替换与回滚。当前版本 {APP_VERSION}。</p>
                  {onlineUpdate.error && <em className="update-check-note error">{onlineUpdate.error}</em>}
                  {!onlineUpdate.error && onlineUpdate.checked && !onlineUpdate.release?.update_available && <em className="update-check-note">已是最新版本，无需更新。</em>}
                </div><button type="button" disabled={onlineUpdate.checking || reconfiguring || status === "running"} onClick={checkForOnlineUpdate}><RefreshCw size={15} className={onlineUpdate.checking ? "spin" : ""} />{onlineUpdate.checking ? "正在检查" : "检查更新"}</button></div>
                <div className="manual-update-card"><div><span>OFFLINE PROGRAM UPDATE</span><strong>手动更新程序</strong><p>导入新的干净项目 ZIP、7Z 等归档。更新器通过命令安全解压并替换程序文件，保留现有环境、模型和创作数据。</p></div><button type="button" disabled={reconfiguring || status === "running"} onClick={enterManualUpdater}><Upload size={15} />进入手动更新</button></div>
                <div className="about-platform"><span>系统</span><code>{inferenceHealth?.platform || "正在读取..."}</code></div>
              </section>}
            </div>
          </div>
          <footer className="settings-foot"><span>{inferenceHealth?.status === "ready" ? "推理服务在线" : "推理服务不可用"} · 协议 {inferenceHealth?.protocol || "--"}</span><button disabled={reconfiguring} onClick={() => setSettingsOpen(false)}>完成</button></footer>
        </section>
      </div>}
      {updateConfirmOpen && onlineUpdate.release && <div className="update-confirm-backdrop" onMouseDown={(event) => event.target === event.currentTarget && !onlineUpdate.checking && setUpdateConfirmOpen(false)}>
        <div className="update-confirm" role="dialog" aria-label="发现新版本">
          <header><span className="eyebrow">UPDATE AVAILABLE</span><strong>发现新版本</strong></header>
          <div className="update-confirm-versions">
            <div><span>当前版本</span><strong>{APP_VERSION}</strong></div>
            <ArrowRight size={16} />
            <div><span>可更新到</span><strong>{onlineUpdate.release.version}{onlineUpdate.release.prerelease ? " · 预览版" : ""}</strong></div>
          </div>
          <div className="update-confirm-facts">
            <div><span>更新包</span><strong>{onlineUpdate.release.asset_name}</strong></div>
            {onlineUpdate.release.asset_bytes > 0 && <div><span>大小</span><strong>{formatFileSize(onlineUpdate.release.asset_bytes)}</strong></div>}
            {onlineUpdate.release.published_at && <div><span>发布时间</span><strong>{new Date(onlineUpdate.release.published_at).toLocaleDateString()}</strong></div>}
            <div><span>校验</span><strong>{onlineUpdate.release.verified ? "SHA-256 校验" : "仅官方直连"}</strong></div>
          </div>
          {onlineUpdate.release.notes && <div className="update-confirm-notes"><span>更新内容</span><pre>{onlineUpdate.release.notes}</pre></div>}
          <p className="update-confirm-hint">更新会替换程序文件，并保留现有环境、模型、输出与创作数据。失败时自动回滚到当前版本。</p>
          {onlineUpdate.error && <p className="update-confirm-error">{onlineUpdate.error}</p>}
          <footer>
            <button type="button" className="ghost" disabled={onlineUpdate.checking} onClick={() => setUpdateConfirmOpen(false)}>稍后再说</button>
            <button type="button" disabled={onlineUpdate.checking} onClick={startOnlineUpdate}><Download size={15} />{onlineUpdate.checking ? "正在开始" : "立即更新"}</button>
          </footer>
        </div>
      </div>}
      {hardwareMonitorOpen && <div className="hw-monitor-backdrop" onMouseDown={(event) => event.target === event.currentTarget && setHardwareMonitorOpen(false)}>
        <div className="hw-monitor" role="dialog" aria-label="硬件性能检测">
          <header><span className="eyebrow">HARDWARE MONITOR</span><div><span className="hw-live-dot" />实时</div><button onClick={() => setHardwareMonitorOpen(false)}><X size={16} /></button></header>
          <div className="hw-body">
            <div className="hw-section">
              <div className="hw-section-title">GPU · {hardwareStats?.gpu_name || inferenceHealth?.device || "N/A"}</div>
              <div className="hw-row">
                <div className="hw-metric">
                  <span>温度</span>
                  <strong>{hardwareStats?.gpu_temp != null ? `${hardwareStats.gpu_temp} °C` : "--"}</strong>
                </div>
                <div className="hw-metric">
                  <span>利用率</span>
                  <strong>{hardwareStats?.gpu_util != null ? `${hardwareStats.gpu_util}%` : "--"}</strong>
                </div>
                {hardwareStats?.power_w != null && <div className="hw-metric">
                  <span>功耗</span><strong>{hardwareStats.power_w} W</strong>
                </div>}
                {hardwareStats?.fan_speed != null && <div className="hw-metric">
                  <span>风扇</span><strong>{hardwareStats.fan_speed}%</strong>
                </div>}
                <div className="hw-metric" title={hardwareStats?.memory_reason || inferenceHealth?.memory_reason || "首次生成加载模型时自动评估"}>
                  <span>显存档位</span><strong>{hardwareStats?.memory_label || inferenceHealth?.memory_label || "AUTO 待评估"}</strong>
                </div>
                <div className="hw-metric">
                  <span>注意力 / 缓存</span><strong>{hardwareStats?.attention_backend || inferenceHealth?.attention_backend || "--"} · {(hardwareStats?.model_resident ?? inferenceHealth?.model_resident) ? "GPU 常驻" : (hardwareStats?.model_cached ?? inferenceHealth?.model_cached) ? "内存" : "无"}</strong>
                </div>
              </div>
              <div className="hw-metric hw-vram">
                <span>显存</span>
                <strong>{hardwareStats?.vram_used_mb != null ? hardwareStats.vram_used_mb < 1024 ? `${hardwareStats.vram_used_mb} MB` : `${(hardwareStats.vram_used_mb / 1024).toFixed(1)}G` : "--"} / {hardwareStats?.vram_total_mb != null ? hardwareStats.vram_total_mb < 1024 ? `${hardwareStats.vram_total_mb} MB` : `${(hardwareStats.vram_total_mb / 1024).toFixed(1)}G` : "--"}</strong>
                <div className="hw-bar"><i style={{ width: `${Math.min(100, ((hardwareStats?.vram_used_mb || 0) / (hardwareStats?.vram_total_mb || 1)) * 100)}%` }} /></div>
              </div>
              <div className="hw-chart-label">GPU 利用率曲线 (%)</div>
              <Sparkline data={hardwareHistory.gpu} max={100} />
            </div>
            <div className="hw-section">
              <div className="hw-section-title">显存占用曲线 (GB)</div>
              <Sparkline data={hardwareHistory.vram} max={hardwareStats?.vram_total_mb ? hardwareStats.vram_total_mb / 1024 : undefined} label={(v) => `${Number(v).toFixed(1)}G`} />
            </div>
            <div className="hw-section">
              <div className="hw-section-title">系统</div>
              <div className="hw-row">
                <div className="hw-metric">
                  <span>CPU</span>
                  <strong>{hardwareStats?.cpu_percent != null ? `${hardwareStats.cpu_percent}%` : "--"}</strong>
                </div>
                <div className="hw-metric">
                  <span>内存</span>
                  <strong>{hardwareStats?.ram_used_gb != null ? `${hardwareStats.ram_used_gb}G / ${hardwareStats.ram_total_gb}G` : "--"}</strong>
                </div>
              </div>
              <div className="hw-chart-label">CPU 利用率曲线 (%)</div>
              <Sparkline data={hardwareHistory.cpu} max={100} />
              <div className="hw-chart-label">内存占用曲线 (GB)</div>
              <Sparkline data={hardwareHistory.ram} max={hardwareStats?.ram_total_gb || undefined} label={(value) => `${Number(value).toFixed(1)}G`} />
            </div>
          </div>
          <footer><i />每 2 秒自动刷新 · 点击遮罩关闭</footer>
        </div>
      </div>}
      {imageViewerOpen && (
        <div className={`image-viewer-backdrop ${viewerExpanded ? "expanded" : ""}`} onMouseDown={(event) => event.target === event.currentTarget && closeImageViewer()}>
          <section className={`image-viewer ${viewerExpanded ? "expanded" : ""} ${viewerSidebarOpen ? "sidebar-open" : ""}`} role="dialog" aria-modal="true" aria-label="图片预览工作区">
            <header className="image-viewer-head">
              <div className="viewer-title"><strong>{activeViewerLayerItem?.name || generatedName || "图片预览工作区"}</strong><span>{Math.round(viewerZoom * 100)}% · 滚轮仅缩放预览 · 拖动角点改变实际图片尺寸 · Ctrl+C / Ctrl+V 复制 · Ctrl+Z 撤销</span></div>
              <div className="viewer-head-actions">
                <button className={viewerSidebarOpen ? "active" : ""} onClick={() => setViewerSidebarOpen((current) => !current)} title="展开启动以来的图片历史"><PanelLeft size={15} />图片栏</button>
                 <button onClick={() => { setViewerZoom(1); setViewerPan({ x: 0, y: 0 }); }} title="100%：一个源像素对应一个 CSS 像素"><RefreshCw size={15} />100%</button><button onClick={fitViewerToCanvas}>适应窗口</button>
                <button onClick={() => setViewerExpanded((current) => !current)} title="扩展到浏览器窗口，不进入 F11 全屏">{viewerExpanded ? <Minimize2 size={15} /> : <Maximize2 size={15} />}{viewerExpanded ? "还原" : "全屏"}</button>
                <button className="viewer-close" onClick={closeImageViewer}><X size={18} /></button>
              </div>
            </header>
            <div className="image-viewer-body">
              <aside className="viewer-history" aria-hidden={!viewerSidebarOpen} ref={viewerHistoryRef}>
                <header>
                  {viewerHistoryBatch ? <button className="history-back" onClick={returnToHistoryCards}><ChevronLeft size={15} />返回当前文件夹</button> : <div><strong>{viewerSelectedFolder ? viewerDirectory.label || "输出目录" : "本次启动图片"}</strong><span>{visibleHistoryCards.reduce((total, card) => total + card.files.length, 0)} 张 · {viewerFolders.length} 个子文件夹</span></div>}
                  <button title="刷新当前文件夹" disabled={viewerHistoryLoading} onClick={() => refreshViewerHistory()}><RefreshCw className={viewerHistoryLoading ? "spin" : ""} size={14} /></button>
                </header>
                <div className="viewer-folder-browser">
                  <button type="button" className={!viewerSelectedFolder ? "active" : ""} disabled={viewerHistoryLoading} onClick={() => selectViewerFolder("")}><ImageIcon size={13} />本次启动</button>
                  <button type="button" className={viewerSelectedFolder === viewerOutputRootId ? "active" : ""} disabled={viewerHistoryLoading || !viewerOutputRootId} onClick={() => selectViewerFolder(viewerOutputRootId)}><FolderOpen size={13} />输出目录</button>
                  <button type="button" disabled={viewerHistoryLoading || !viewerSelectedFolder || !viewerDirectory.parent_id} onClick={() => selectViewerFolder(viewerDirectory.parent_id)}><ChevronLeft size={14} />上一级</button>
                  <span title={viewerSelectedFolder ? viewerDirectory.label : "本次启动生成"}><FolderOpen size={13} />{viewerSelectedFolder ? viewerDirectory.label : "输出目录 / 本次启动"}</span>
                </div>
                <div className={`viewer-history-grid ${viewerHistoryBatch ? "batch-detail" : ""}`}>
                  {!viewerHistoryBatch && viewerFolders.map((folder) => <button type="button" className="viewer-history-folder" key={folder.id} onClick={() => selectViewerFolder(folder.id)}>
                    <FolderOpen size={24} />
                    <span><strong>{folder.name}</strong><small>{folder.count} 张图片 · {folder.folder_count} 个文件夹</small></span>
                  </button>)}
                  {!viewerHistoryBatch && visibleHistoryCards.map((card) => <button
                    type="button"
                    key={card.id}
                    data-history-card={card.id}
                    className={`viewer-history-card ${card.preview.transparent_background ? "transparent-preview" : ""}`}
                    draggable
                    onDragStart={(event) => historyDragStart(event, card.preview)}
                    onClick={() => openHistoryCard(card)}
                    onContextMenu={(event) => { event.preventDefault(); event.stopPropagation(); openViewerContextMenu({ x: event.clientX, y: event.clientY, kind: "history", asset: card.preview, files: card.files, label: card.kind === "batch" ? `批次 ${card.batch_index} 的 ${card.count} 张图片` : card.preview.name }); }}
                  >
                    <img src={card.preview.url} alt="" />
                    <span>{card.preview.name}</span>
                    {card.kind === "batch" && <b>{card.count}</b>}
                  </button>)}
                  {viewerHistoryBatch && visibleHistoryFiles(viewerHistoryBatch.files).map((file) => <button
                    type="button"
                    key={file.id}
                    className={`viewer-history-card ${file.transparent_background ? "transparent-preview" : ""}`}
                    draggable
                    onDragStart={(event) => historyDragStart(event, file)}
                    onClick={() => focusViewerAsset(file)}
                    onContextMenu={(event) => { event.preventDefault(); event.stopPropagation(); openViewerContextMenu({ x: event.clientX, y: event.clientY, kind: "history", asset: file, files: [file], label: file.name }); }}
                  ><img src={file.url} alt="" /><span>图片 {file.image_index}</span></button>)}
                  {!viewerHistoryBatch && visibleHistoryCards.length === 0 && viewerFolders.length === 0 && <div className="history-empty">{viewerHistoryLoading ? <RefreshCw className="spin" size={24} /> : <ImageIcon size={24} />}<span>{viewerHistoryLoading ? "正在读取输出目录" : viewerSelectedFolder ? "这个文件夹没有图片或子文件夹" : "本次启动尚未生成图片，输出目录也没有可浏览的文件夹"}</span></div>}
                </div>
              </aside>
              <button className="viewer-sidebar-handle" type="button" aria-label={viewerSidebarOpen ? "收起图片栏" : "展开图片栏"} title={viewerSidebarOpen ? "收起图片栏" : "展开图片栏"} onClick={() => setViewerSidebarOpen((current) => !current)}>{viewerSidebarOpen ? <ChevronLeft size={17} /> : <PanelLeft size={17} />}<span>{viewerSidebarOpen ? "收起" : "图片"}</span></button>
              <section className="viewer-workspace" style={{ "--viewer-toolbar-height": `${viewerToolbarHeight}px` }}>
                  <div className="viewer-toolbar" ref={viewerToolbarRef}>
                    <div className="viewer-toolbar-group viewer-tool-group" aria-label="视图"><button title="缩小视图" aria-label="缩小视图" onClick={() => setViewerZoom((current) => Math.max(VIEWER_MIN_ZOOM, current / 1.15))}><ZoomOut size={14} /></button><output>{Math.round(viewerZoom * 100)}%</output><button title="放大视图" aria-label="放大视图" onClick={() => setViewerZoom((current) => Math.min(VIEWER_MAX_ZOOM, current * 1.15))}><ZoomIn size={14} /></button></div>
                    <div className="viewer-toolbar-group" aria-label="布局"><button ref={viewerEdgeTriggerRef} className={viewerEdgePanelOpen ? "active" : ""} aria-expanded={viewerEdgePanelOpen} aria-controls="viewer-alignment-panel" aria-haspopup="dialog" title="对齐与线条" onClick={() => toggleViewerToolbarPopover(VIEWER_TOOLBAR_POPOVER_LAYOUT)}><SlidersHorizontal size={14} /><span>对齐与线条</span></button><button ref={viewerTemplateTriggerRef} className={viewerTemplatesOpen ? "active" : ""} aria-expanded={viewerTemplatesOpen} aria-controls="viewer-template-panel" aria-haspopup="dialog" title="拼图模板" onClick={() => toggleViewerToolbarPopover(VIEWER_TOOLBAR_POPOVER_TEMPLATES)}><LayoutTemplate size={14} /><span>拼图模板</span></button>{!activeCollage && viewerLayers.length > 1 && <button title="一键拼图" onClick={createManualCollage}><Layers3 size={14} /><span>一键拼图</span></button>}</div>
                    <div className="viewer-toolbar-group" aria-label="图层"><button className={viewerLayerResizeEnabled ? "active" : ""} aria-pressed={viewerLayerResizeEnabled} title={viewerLayerResizeEnabled ? "关闭图片尺寸调整（不影响相机缩放或拖动）" : "开启图片尺寸调整"} onClick={toggleViewerLayerResize}><Move size={14} /><span>图片尺寸调整</span></button>{!activeCollage && activeViewerLayerItem && <div className="viewer-layer-scale"><Move size={13} /><span>选中图片</span><button disabled={!viewerLayerResizeEnabled} onClick={() => scaleViewerLayer(activeViewerLayerItem.id, 1 / 1.1)}>-</button><BoundedNumberInput value={Math.round(activeViewerLayerItem.scale * 100)} min={10} max={800} integer disabled={!viewerLayerResizeEnabled} onCommit={(percentage) => setViewerLayerScale(activeViewerLayerItem.id, percentage)} ariaLabel="选中图片缩放比例" /><em>%</em><button disabled={!viewerLayerResizeEnabled} onClick={() => scaleViewerLayer(activeViewerLayerItem.id, 1.1)}>+</button></div>}</div>
                    {activeCollageSlotItem && <div className="viewer-toolbar-group collage-slot-adjust" aria-label={`拼图区块 ${activeCollageSlot + 1}`}><div className="viewer-toolbar-subgroup viewer-layer-scale"><Move size={13} /><span>区块 {activeCollageSlot + 1}</span><button onClick={() => updateCollageSlot(activeCollageSlot, { scale: Math.max(.1, activeCollageSlotItem.scale / 1.1) })}>-</button><output>{Math.round(activeCollageSlotItem.scale * 100)}%</output><button onClick={() => updateCollageSlot(activeCollageSlot, { scale: Math.min(4, activeCollageSlotItem.scale * 1.1) })}>+</button></div><div className="viewer-toolbar-subgroup viewer-tool-group"><button title="左边缘对齐" onClick={() => updateCollageSlot(activeCollageSlot, { alignX: 0 })}>L</button><button title="水平居中" onClick={() => updateCollageSlot(activeCollageSlot, { alignX: .5 })}>C</button><button title="右边缘对齐" onClick={() => updateCollageSlot(activeCollageSlot, { alignX: 1 })}>R</button></div><div className="viewer-toolbar-subgroup viewer-tool-group"><button title="顶边缘对齐" onClick={() => updateCollageSlot(activeCollageSlot, { alignY: 0 })}>T</button><button title="垂直居中" onClick={() => updateCollageSlot(activeCollageSlot, { alignY: .5 })}>M</button><button title="底边缘对齐" onClick={() => updateCollageSlot(activeCollageSlot, { alignY: 1 })}>B</button></div></div>}
                    {(activeCollage || collageResult) && <div className="viewer-toolbar-group viewer-toolbar-results" aria-label="结果">{activeCollage && <button className="viewer-confirm" onClick={confirmCollage}><Check size={14} /><span>确认拼图</span></button>}{collageResult && <><button className="viewer-confirm" disabled={collageResult.saved} onClick={saveCollage}><Save size={14} /><span>{collageResult.saved ? "已保存" : "保存拼图"}</span></button><button onClick={editCollage}><LayoutTemplate size={14} /><span>重新拼图</span></button><button className="viewer-danger" onClick={discardCollage}><Trash2 size={14} /><span>删除拼图</span></button></>}</div>}
                 </div>
                 {(viewerTemplatesOpen || viewerEdgePanelOpen) && <div className="viewer-toolbar-popover-backdrop" style={{ top: viewerToolbarHeight }} aria-hidden="true" onPointerDown={() => closeViewerToolbarPopover("backdrop")} />}
                 {viewerTemplatesOpen && <div ref={viewerTemplatePanelRef} id="viewer-template-panel" className="collage-template-panel" style={{ top: viewerToolbarHeight + 1 }} role="dialog" aria-label="拼图模板">
                   <header><div><strong>拼图模板</strong><span>2–9 图 · 每种数量提供 3 种推荐排版</span></div><button aria-label="关闭拼图模板" onClick={closeViewerToolbarPopover}><X size={15} /></button></header>
                  <div className="collage-template-groups">{Array.from({ length: 8 }, (_, index) => index + 2).map((count) => <section key={count}><h4>{count} 图</h4><div>{collageTemplates.filter((template) => template.count === count).map((template) => <button key={template.id} onClick={() => chooseCollageTemplate(template)}><span className="template-miniature">{template.slots.map((slot, slotIndex) => <i key={slotIndex} style={{ left: `${slot.x * 100}%`, top: `${slot.y * 100}%`, width: `${slot.w * 100}%`, height: `${slot.h * 100}%` }} />)}</span><b>{template.label}</b></button>)}</div></section>)}</div>
                </div>}
                 {viewerEdgePanelOpen && <div ref={viewerEdgePanelRef} id="viewer-alignment-panel" className="viewer-edge-panel" role="dialog" aria-label="对齐与线条"><header><div><strong>对齐与线条</strong><span>吸附、辅助线与拼图边缘样式</span></div><button aria-label="关闭对齐与线条" onClick={closeViewerToolbarPopover}><X size={14} /></button></header><div className="viewer-alignment-controls"><label className="viewer-grid-control"><input type="checkbox" checked={viewerGridEnabled} onChange={(event) => setViewerGridEnabled(event.target.checked)} /><span>网格吸附</span><BoundedNumberInput value={viewerGridSize} min={4} max={256} integer onCommit={setViewerGridSize} ariaLabel="网格吸附像素尺寸" /><small>px / 格</small></label><label className="viewer-grid-control"><input type="checkbox" checked={viewerEdgeSnapEnabled} onChange={(event) => setViewerEdgeSnapEnabled(event.target.checked)} /><span>边缘吸附</span></label><label className="viewer-grid-control"><input type="checkbox" checked={viewerAlignmentGuidesEnabled} onChange={(event) => { setViewerAlignmentGuidesEnabled(event.target.checked); if (!event.target.checked) setViewerSnapGuide(null); }} /><span>辅助对齐线</span></label><label className="viewer-grid-control"><input type="checkbox" checked={viewerEdgeLine.enabled} onChange={toggleEdgeLines} /><span>显示边缘线</span></label></div><div className="edge-style-options">{edgeLineStyles.map((style) => <button key={style.id} className={viewerEdgeLine.style === style.id ? "active" : ""} onClick={() => updateViewerEdgeLine({ style: style.id })}><i className={style.id} style={{ "--edge-color": viewerEdgeLine.color }} />{style.label}</button>)}</div><div className="edge-color-row"><label><span>颜色</span><input type="color" value={viewerEdgeLine.color} onChange={(event) => updateViewerEdgeLine({ color: event.target.value })} /></label><button onClick={pickEdgeColor}><Palette size={14} />提取屏幕颜色</button><label><span>线宽</span><input type="range" min="1" max="12" value={viewerEdgeLine.width} onChange={(event) => updateViewerEdgeLine({ width: Number(event.target.value) })} /><b>{viewerEdgeLine.width}px</b></label></div></div>}
                <div
                  className={`image-viewer-canvas ${activeCollage ? "collage-active" : ""}`}
                  ref={viewerCanvasRef}
                  style={{ "--viewer-grid-size": `${viewerGridSize * viewerZoom}px`, "--edge-color": viewerEdgeLine.color, "--edge-width": `${viewerEdgeLine.width}px` }}
                  onWheel={zoomImageAt}
                  onPointerDown={startViewerDrag}
                  onPointerMove={moveViewerImage}
                  onPointerUp={finishViewerPointer}
                  onPointerCancel={finishViewerPointer}
                  onDragOver={(event) => event.preventDefault()}
                  onDrop={viewerDrop}
                  onContextMenu={(event) => { if (event.target === event.currentTarget) event.preventDefault(); }}
                >
                  <div className="viewer-scene" style={{ transform: `translate(${viewerPan.x}px, ${viewerPan.y}px) scale(${viewerZoom})` }}>
                    {!activeCollage && viewerLayers.map((layer) => <div
                      className={`viewer-image-layer ${activeViewerLayer === layer.id ? "active" : ""} ${viewerEdgeLine.enabled ? `has-edge edge-${viewerEdgeLine.style}` : ""}`}
                      key={layer.id}
                      data-viewer-layer-id={layer.id}
                       style={{ width: `${layer.naturalWidth}px`, height: `${layer.naturalHeight}px`, transform: `translate(${layer.x}px, ${layer.y}px) scale(${layer.scale})` }}
                      onPointerDown={(event) => startViewerLayerDrag(event, layer)}
                      onPointerMove={moveViewerLayer}
                      onPointerUp={finishViewerPointer}
                      onPointerCancel={finishViewerPointer}
                       onContextMenu={(event) => { event.preventDefault(); event.stopPropagation(); setActiveViewerLayer(layer.id); openViewerContextMenu({ x: event.clientX, y: event.clientY, kind: "layer", layer }); }}
                     ><img src={layer.url} alt={layer.name} draggable="false" />{viewerEdgeLine.enabled && ["top", "right", "bottom", "left"].filter((side) => !viewerLayerEdges[layer.id]?.includes(side)).map((side) => <i className={`layer-edge ${side}`} key={side} />)}{viewerLayerResizeEnabled && activeViewerLayer === layer.id && ["tl", "tr", "bl", "br", "top", "right", "bottom", "left"].map((handle) => <i className={`layer-corner-anchor ${handle}`} key={handle} style={{ "--viewer-handle-inverse": inverseViewerHandleScale(viewerZoom, layer.scale) }}><i className={`layer-corner ${handle}`} /></i>)}</div>)}
                    {activeCollage && activeCollageTemplate && activeCollageLayout && <div className={`collage-board ${activeCollageLayout.aspect > 1.35 ? "wide" : "square"}`} style={{ "--collage-aspect": activeCollageLayout.aspect }} onContextMenu={(event) => { event.preventDefault(); event.stopPropagation(); openViewerContextMenu({ x: event.clientX, y: event.clientY, kind: "collage-draft" }); }}>
                      {activeCollageLayout.slots.map((slot, index) => <div
                        className={`collage-slot ${activeCollage.slots[index] ? "filled" : ""} ${activeCollageSlot === index ? "active" : ""} ${activeCollage.slots[index] && viewerEdgeLine.enabled ? `has-edge edge-${viewerEdgeLine.style}` : ""}`}
                        key={index}
                        style={{ left: `${slot.x * 100}%`, top: `${slot.y * 100}%`, width: `${slot.w * 100}%`, height: `${slot.h * 100}%` }}
                        onClick={() => activeCollage.slots[index] && setActiveCollageSlot(index)}
                        onDragOver={(event) => event.preventDefault()}
                        onDrop={(event) => dropCollageSlot(event, index)}
                      >{activeCollage.slots[index] ? <><img src={activeCollage.slots[index].asset.url} alt="" draggable="true" onDragStart={(event) => historyDragStart(event, activeCollage.slots[index].asset)} style={{ transform: `scale(${activeCollage.slots[index].scale})`, transformOrigin: `${activeCollage.slots[index].alignX * 100}% ${activeCollage.slots[index].alignY * 100}%` }} />{viewerEdgeLine.enabled && ["top", "right", "bottom", "left"].filter((side) => !activeCollageEdges[index]?.includes(side)).map((side) => <i className={`layer-edge ${side}`} key={side} />)}</> : <span><ImagePlus size={20} />拖入图片<br />区块 {index + 1}</span>}</div>)}
                    </div>}
                  </div>
                  {!activeCollage && viewerLayers.length === 0 && <div className="viewer-canvas-empty"><ImagePlus size={31} /><strong>拖入或粘贴图片开始排版</strong><span>支持左侧图片栏、本地文件拖入和系统剪贴板图片</span></div>}
                  {viewerSnapGuide && <><i className="viewer-snap-guide vertical" style={{ left: viewerSnapGuide.x ?? -100 }} /><i className="viewer-snap-guide vertical" style={{ left: viewerSnapGuide.x2 ?? -100 }} /><i className="viewer-snap-guide horizontal" style={{ top: viewerSnapGuide.y ?? -100 }} /><i className="viewer-snap-guide horizontal" style={{ top: viewerSnapGuide.y2 ?? -100 }} /></>}
                  {activeCollage && viewerLayers.length > 0 && <div className="collage-source-tray"><span>预览窗口图片 · 拖入区块</span><div>{viewerLayers.map((layer) => <button type="button" key={layer.id} draggable onDragStart={(event) => historyDragStart(event, { id: layer.assetId, url: layer.url, name: layer.name })} title={`拖入拼图：${layer.name}`}><img src={layer.url} alt="" /><small>{layer.name}</small></button>)}</div></div>}
                  {viewerNotice && <button className="viewer-notice" onClick={() => setViewerNotice("")}>{viewerNotice}<X size={12} /></button>}
                </div>
              </section>
            </div>
          </section>
          {viewerMenu && <div className="viewer-context-menu" style={{ left: Math.min(viewerMenu.x, window.innerWidth - 230), top: Math.min(viewerMenu.y, window.innerHeight - 180) }} onPointerDown={(event) => event.stopPropagation()}>
            {viewerMenu.kind === "history" && <><button onClick={() => focusViewerAsset(viewerMenu.asset)}><ImageIcon size={14} />切换当前预览</button><button onClick={() => addViewerAsset(viewerMenu.asset)}><ImagePlus size={14} />添加到预览窗口</button>{viewerMenu.asset.manual_layout && <button onClick={() => editCollage({ manualLayout: viewerMenu.asset.manual_layout })}><LayoutTemplate size={14} />重新拼图</button>}<button className="danger" onClick={() => requestHistoryDelete(viewerMenu.files, viewerMenu.label)}><Trash2 size={14} />删除{viewerMenu.files.length > 1 ? "本批次" : "图片"}</button></>}
            {viewerMenu.kind === "layer" && <><button onClick={() => copyViewerLayer(viewerMenu.layer)}><Copy size={14} />复制干净 PNG</button>{viewerMenu.layer.isCollage && <button onClick={() => editCollage(viewerMenu.layer)}><LayoutTemplate size={14} />重新拼图</button>}<button className="danger" onClick={() => removeViewerLayer(viewerMenu.layer.id)}><Trash2 size={14} />删除当前预览图片</button></>}
            {viewerMenu.kind === "collage-draft" && <button className="danger" onClick={cancelCollageDraft}><X size={14} />取消拼图</button>}
          </div>}
          {historyDelete && <div className="viewer-confirm-backdrop" onMouseDown={(event) => event.target === event.currentTarget && setHistoryDelete(null)}><section className="viewer-delete-dialog"><Trash2 size={23} /><strong>删除{historyDelete.count > 1 ? "整组批次" : "图片"}？</strong><p>{historyDelete.label}<br />请选择仅从本次启动的左侧历史中隐藏，或同时永久删除 outputs 中的 {historyDelete.count} 个 PNG 源文件。</p><div><button onClick={() => finishHistoryDelete(false)}>只删除预览卡片</button><button className="danger" onClick={() => finishHistoryDelete(true)}>同时删除源文件</button><button onClick={() => setHistoryDelete(null)}>取消</button></div></section></div>}
        </div>
      )}
      {/* No Suspense backdrop: the assistant is a non-modal floating window, so its chunk must load
          without dimming or blocking the workspace behind it. */}
      {assistantOpen && <Suspense fallback={null}><AiAssistantOverlay
        open={assistantOpen}
        onClose={() => setAssistantOpen(false)}
        onApplyPrompt={applyAssistantPrompt}
        promptSnapshot={() => ({ positive: promptTextRevision.current.positive.text, negative: promptTextRevision.current.negative.text })}
      /></Suspense>}
      {appNotice && <button type="button" className={`app-notice ${appNotice.error ? "error" : "success"}`} role="status" onClick={() => setAppNotice(null)}>{appNotice.error ? <X size={16} /> : <Check size={16} />}<span>{appNotice.message}</span></button>}
      <button className={`console-toggle ${consoleOpen ? "active" : ""}`} type="button" aria-expanded={consoleOpen} onClick={() => setConsoleOpen((current) => !current)}><Terminal size={15} /><span>控制台</span>{consoleRunning && <i />}</button>
      {consoleOpen && <section className="console-drawer" style={{ height: `${consoleHeight}px` }} aria-label="本地控制台">
        <div className="console-resize" onPointerDown={resizeConsole} title="拖动调整控制台高度" />
        <header className="console-head"><div><Terminal size={15} /><span>LOCAL CONSOLE</span><small>推理日志 · 错误输出 · 本机命令</small></div><div><button type="button" title="清空当前视图" onClick={clearConsole}><Trash2 size={14} />清空</button><button type="button" className="console-hide" title="隐藏控制台" onClick={() => setConsoleOpen(false)}><X size={14} />隐藏</button></div></header>
        <div className="console-output" ref={consoleOutputRef} aria-live="polite">
          {consoleEntries.length === 0 && <p className="console-empty">等待本地推理日志。输入命令可在项目根目录执行。</p>}
          {consoleEntries.map((entry) => <div className={`console-entry ${entry.stream}`} key={entry.id}><time>{new Date(entry.at).toLocaleTimeString("zh-CN", { hour12: false })}</time><b>{entry.source === "inference" ? "INFERENCE" : "TERMINAL"}</b><pre>{entry.message}</pre></div>)}
        </div>
        {consoleError && <p className="console-error">{consoleError}</p>}
        <form className="console-command" onSubmit={submitConsoleCommand}><span>&gt;</span><input value={consoleCommand} onChange={(event) => setConsoleCommand(event.target.value)} placeholder="输入本机终端命令，例如：dir /a" spellCheck="false" disabled={consoleRunning} /><button type="submit" disabled={!consoleCommand.trim() || consoleRunning}>{consoleRunning ? <RefreshCw className="spin" size={15} /> : <Send size={15} />}{consoleRunning ? "执行中" : "执行"}</button></form>
      </section>}
    </main>
  );
}

export default App;
