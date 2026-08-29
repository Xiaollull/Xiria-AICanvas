import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const readSource = (file) => readFile(new URL(`../${file}`, import.meta.url), "utf8");

// The gallery is three modules: the card contract shared by both entry points, the lazily loaded
// curation page, and the add-to-gallery dialog that ships with the workspace. Assertions anchor on
// whichever module owns the code, so a future move shows up as a failure rather than a silent pass.
const readGalleryCore = () => readSource("src/gallery-core.js");
const readGalleryPage = () => readSource("src/GalleryPage.jsx");
// The model downloader is its own route chunk, no longer part of App.jsx.
const readDownloader = () => readSource("src/ModelDownloader.jsx");

function sourceBetween(source, start, end) {
  const startIndex = source.indexOf(start);
  assert.notEqual(startIndex, -1, `missing source anchor: ${start}`);
  const endIndex = source.indexOf(end, startIndex + start.length);
  assert.notEqual(endIndex, -1, `missing source anchor: ${end}`);
  return source.slice(startIndex, endIndex);
}

test("Anima workspace restore and generation preserve supported feature settings", async () => {
  const app = await readSource("src/App.jsx");
  const restore = sourceBetween(app, "function loadWorkspaceState(saved)", "function reconcileModels");
  const generate = sourceBetween(app, "  const generate = async () => {", "  const releaseLoadedModel = async () => {");

  assert.match(restore, /enabled: savedHires\.enabled === true/);
  // The stage is a list of units now, and one shared normaliser owns its shape.
  assert.match(restore, /adetailer: normalizeADetailerStage\(savedADetailer\)/);
  assert.match(restore, /enabled: savedRtx\.enabled === true/);
  assert.match(restore, /const mountedLoraMap = normalizeMountedLoraMap\(saved\.mountedLorasByEngine/);
  assert.match(restore, /mountedLorasByEngine: mountedLoraMap\.container/);
  assert.match(restore, /loras: mountedLoraMap\.activeLoras/);
  // Neither native engine streams a decoded latent, so the split is by picker shape, not by name.
  assert.match(restore, /processPreview: isSplitModel \? false : saved\.processPreview !== false/);
  assert.match(restore, /guidance: guidanceOptions\.some\(\(item\) => item\.id === saved\.guidance\) \? saved\.guidance : "none"/);
  assert.doesNotMatch(restore, /enabled: !isAnima/);
  assert.doesNotMatch(restore, /loras: isAnima \? \[\]/);
  assert.doesNotMatch(restore, /saved\.guidance !== "pag"/);

  assert.match(generate, /setGeneratedSettings\([\s\S]*?guidance,\s+processPreview/);
  assert.match(generate, /hires: \{\s+(?:\/\/[^\n]*\n\s+)?enabled: hires\.enabled,/);
  assert.match(generate, /adetailer: adetailerPayload\(adetailer, model\)/);
  assert.match(generate, /rtx: \{\s+enabled: rtx\.enabled,/);
  assert.match(generate, /scheduler: nativeGeneration && !ANIMA_SCHEDULERS\.includes\(scheduler\) \? "simple" : scheduler,\s+guidance: distilledGeneration \? "none" : guidance,\s+pag:/);
  assert.match(app, /import \{ SAMPLER_NAMES as samplerNames, SCHEDULER_NAMES as schedulerNames \} from "\.\/sampling-options"/);
  assert.match(app, /const ANIMA_SAMPLERS = samplerNames/);
  assert.match(app, /const ANIMA_SCHEDULERS = schedulerNames/);
  assert.match(generate, /const generationLoras = frozenMountedLorasForScope\(mountedLorasMapRef\.current, activeLoraScopeRef\.current\)/);
  assert.match(generate, /loras: generationLoras\.filter\(\(lora\) => lora\.enabled !== false\)/);
  assert.doesNotMatch(generate, /animaGeneration \? \{ \.\.\.(?:hires|adetailer|rtx), enabled: false \}/);
  assert.doesNotMatch(generate, /animaGeneration \? \[\] : loras/);
  assert.doesNotMatch(generate, /animaGeneration && guidance === "pag"/);
});

test("Hires migrates execution mode by engine and sends the independent tiled-refinement contract", async () => {
  const [app, galleryCore, galleryPage] = await Promise.all([readSource("src/App.jsx"), readGalleryCore(), readGalleryPage()]);
  const restore = sourceBetween(app, "function loadWorkspaceState(saved)", "function reconcileModels");
  const generate = sourceBetween(app, "  const generate = async () => {", "  const releaseLoadedModel = async () => {");

  assert.match(restore, /isAnima && !\["full_frame", "usdu_tiled"\]\.includes\(savedHires\.executionMode\)[\s\S]*?"usdu_tiled"/);
  assert.match(restore, /savedHires\.executionMode === "usdu_tiled" && isAnima \? "usdu_tiled" : "full_frame"/);
  assert.match(restore, /sampler: \(isSplitModel \? ANIMA_SAMPLERS : samplerNames\)\.includes\(savedHires\.sampler\) \? savedHires\.sampler : null/);
  assert.match(restore, /tileWidth: normalizeHiresTileDimension\(savedHires\.tileWidth\)/);
  assert.match(restore, /padding: Math\.round\(numberInRange\(savedHires\.padding, 32, 0, 256\)\)/);
  assert.match(restore, /maskBlur: Math\.round\(numberInRange\(savedHires\.maskBlur, 8, 0, 64\)\)/);

  assert.match(generate, /execution_mode: animaGeneration && hires\.executionMode === "usdu_tiled" \? "usdu_tiled" : "full_frame"/);
  assert.match(generate, /\.\.\.\(hires\.sampler \? \{ sampler: hires\.sampler \} : \{\}\)/);
  assert.match(generate, /\.\.\.\(hires\.scheduler \? \{ scheduler: hires\.scheduler \} : \{\}\)/);
  for (const field of ["tile_width: hires.tileWidth", "tile_height: hires.tileHeight", "padding: hires.padding", "mask_blur: hires.maskBlur", "seam_mode: \"none\"", "uniform_tiles: hires.uniformTiles !== false", "tiled_decode: hires.tiledDecode !== false"]) assert.match(generate, new RegExp(field));
  assert.match(generate, /tile_size: hires\.tileSize,[\s\S]*?tile_overlap: hires\.tileOverlap,[\s\S]*?execution_mode/);
  assert.doesNotMatch(app, /tileWidth:\s*512|tileHeight:\s*512|tile_width:\s*512|tile_height:\s*512/);

  assert.match(app, /USDU 分块重绘（推荐）/);
  assert.match(app, /整图重绘（兼容）/);
  assert.match(app, /像素放大分块/);
  assert.match(app, /扩散重绘分块宽度/);
  assert.match(app, /跟随首轮/);
  assert.match(app, /Auto = 首轮源图宽高/);
  // Delegation to the shared Hires normalizer, not a reimplementation. The
  // specifier now carries its extension so the module also loads under plain
  // Node, which is what makes gallery-core unit-testable rather than only
  // checkable as source text.
  assert.match(galleryCore, /import \{[^}]*\bnormalizeGalleryHires\b[^}]*\} from "\.\/hires-settings(?:\.js)?"/);
  assert.match(galleryCore, /hires: normalizeGalleryHires\(model, source\.hires, base\.hires/);
  assert.match(galleryPage, /USDU 分块重绘（推荐）/);
  assert.match(galleryPage, /像素放大分块/);
  for (const source of [galleryCore, galleryPage]) assert.doesNotMatch(source, /tileWidth:\s*512|tileHeight:\s*512/);
  assert.doesNotMatch(app, /Protocol 25 backend integration may return 422/);
  assert.match(app, /const hiresControlsLocked = !hires\.enabled \|\| status === "running"/);
  assert.match(app, /RealESRGAN \/ SR 像素放大分块[\s\S]*?disabled=\{hiresControlsLocked\}/);
  assert.match(app, /Auto = 首轮源图宽高[\s\S]*?mask blur 8[\s\S]*?uniform tiles[\s\S]*?per-tile VAE tiled decode[\s\S]*?seam None[\s\S]*?每 tile 执行 Hires steps/);
  assert.match(galleryPage, /aria-pressed=\{selected\.has\(id\)\} disabled=\{busy\}/);
  assert.match(galleryPage, /USDU 分块重绘/);
});

test("ADetailer request controls are locked while generation runs without locking task controls", async () => {
  const app = await readSource("src/App.jsx");
  const units = await readSource("src/adetailer-units.js");
  const generate = sourceBetween(app, "  const generate = async () => {", "  const releaseLoadedModel = async () => {");
  const adetailer = sourceBetween(app, "            <button type=\"button\" className={`section-heading parameter-title parameter-toggle ${adetailer.expanded", "            <button type=\"button\" className={`section-heading parameter-title parameter-toggle ${rtx.expanded");

  assert.match(app, /const adetailerLocked = status === "running"/);
  // Every field still reaches the request; the payload is built in one place so
  // the two generate surfaces cannot send different shapes.
  assert.match(generate, /adetailer: adetailerPayload\(adetailer, model\)/);
  assert.match(units, /export function adetailerPayload/);
  for (const field of ["detector", "confidence", "max_detections", "mask_min_ratio", "mask_max_ratio", "dilate_erode", "mask_blur", "padding", "denoise", "use_steps", "steps", "use_cfg", "cfg", "prompt"]) {
    assert.match(units, new RegExp(`${field}: unit\\.`), `${field} is missing from the ADetailer request payload`);
  }
  // `negative_prompt` is the one conditional field: an engine with no unconditional branch cannot
  // encode one, so the payload drops it while the editor keeps whatever the user typed.
  assert.match(units, /negative_prompt: carriesNegativePrompt \? unit\.negativePrompt\.trim\(\) : ""/);
  assert.match(units, /const enabled = stage\?\.enabled === true;/);
  // And a stage that is switched off carries no units at all: the wire is the run plan.
  assert.match(units, /return \{ enabled, units: enabled \? adetailerUnitsPayload\(stage, engine\) : \[\] \};/);

  // Every per-unit control is gated on the same lock, through `unitLocked`.
  assert.match(adetailer, /const unitLocked = !adetailer\.enabled \|\| adetailerLocked;/);
  for (const control of [
    "aria-label=\"启用 ADetailer\"[\\s\\S]*?disabled=\\{adetailerLocked",
    "value=\\{unit\\.detector\\} disabled=\\{unitLocked\\}",
    "label=\"检测置信度\"[\\s\\S]*?disabled=\\{unitLocked\\}",
    "label=\"最多处理区域\"[\\s\\S]*?disabled=\\{unitLocked\\}",
    "label=\"最小区域比例\"[\\s\\S]*?disabled=\\{unitLocked\\}",
    "label=\"最大区域比例\"[\\s\\S]*?disabled=\\{unitLocked\\}",
    "value=\\{unit\\.dilateErode\\}[\\s\\S]*?disabled=\\{unitLocked\\}",
    "value=\\{unit\\.maskBlur\\}[\\s\\S]*?disabled=\\{unitLocked\\}",
    "value=\\{unit\\.padding\\}[\\s\\S]*?disabled=\\{unitLocked\\}",
    "label=\"重绘强度\"[\\s\\S]*?disabled=\\{unitLocked\\}",
    "checked=\\{unit\\.useSteps\\} disabled=\\{unitLocked\\}",
    "value=\\{unit\\.steps\\}[\\s\\S]*?unitLocked",
    "checked=\\{unit\\.useCfg\\} disabled=\\{unitLocked\\}",
    "value=\\{unit\\.cfg\\}[\\s\\S]*?unitLocked",
    "value=\\{unit\\.prompt\\} disabled=\\{unitLocked\\}",
    // The negative prompt carries a second gate: a guidance-distilled engine has no branch to
    // encode one into. The run lock is still the first term.
    "value=\\{unit\\.negativePrompt\\} disabled=\\{unitLocked \\|\\| !engineAllowsNegativePrompt\\}",
    // A unit's own switch is a request control too. There is nothing to add or
    // remove: every slot exists from the start, so the switch is the whole gate.
    "aria-label=\\{`启用第 \\$\\{index \\+ 1\\} 个 ADetailer 单元`\\}[\\s\\S]*?disabled=\\{unitLocked\\}",
  ]) assert.match(adetailer, new RegExp(control));
  assert.doesNotMatch(adetailer, /添加检测单元|删除第/);
  assert.doesNotMatch(adetailer, /cancel|取消任务/i);
  assert.match(app, /RealESRGAN \/ SR 像素放大分块[\s\S]*?disabled=\{hiresControlsLocked\}/);
});

test("performance settings expose a hardware-derived VRAM wall slider", async () => {
  const app = await readSource("src/App.jsx");
  const performance = sourceBetween(app, "{settingsTab === \"performance\" && <section", "{settingsTab === \"theme\"");
  assert.match(app, /vram_limit_gb: 0/);
  assert.match(app, /vramLimitInfo = performanceCapabilities\.vram_limit/);
  assert.match(performance, /VRAM HARD WALL/);
  assert.match(performance, /type=\"range\"/);
  assert.match(performance, /aria-label=\"显存占用上限\"/);
  assert.match(performance, /vram_limit_gb: Number\(Number\(event\.target\.value\)\.toFixed\(1\)\)/);
  assert.match(performance, /自动管理显存占用上限/);
});

test("Anima controls and PAG fail closed from per-engine health without intrinsic rejection", async () => {
  const app = await readSource("src/App.jsx");
  const pagHealth = sourceBetween(app, "function pagAvailableForEngine", "const MAX_SEED");
  const readiness = sourceBetween(app, "  const selectedEngineHealth =", "  const loaderProgress =");

  assert.match(pagHealth, /pagHealth\?\.available === true/);
  assert.match(pagHealth, /\(pagHealth\.engines \|\| \[\]\)\.includes\(engine\)/);
  assert.match(pagHealth, /health\?\.engines\?\.\[engine\]\?\.features\?\.pag === true/);

  // A split-model engine must be told a capability is present; a checkpoint engine only has to not
  // be told it is absent. That asymmetry is the fail-closed rule, and it now covers Flux as well.
  assert.match(readiness, /engineAllowsLora = isSplitModel \? selectedEngineFeatures\.lora === true/);
  assert.match(readiness, /engineAllowsHires = isSplitModel \? selectedEngineFeatures\.hires === true/);
  assert.match(readiness, /engineAllowsADetailer = isSplitModel \? selectedEngineFeatures\.adetailer === true/);
  assert.match(readiness, /engineAllowsRtx = isSplitModel \? selectedEngineFeatures\.rtx === true/);
  assert.match(readiness, /loraReady = enabledLoras\.length === 0 \|\| engineAllowsLora/);
  // The engine names itself in the message now that two engines can produce it.
  assert.match(readiness, /当前推理服务未声明 \$\{engineLabel\} LoRA 能力/);
  assert.match(readiness, /当前推理服务未声明 \$\{engineLabel\} Hires\.fix 能力/);
  assert.match(readiness, /当前推理服务未声明 \$\{engineLabel\} ADetailer 能力/);
  assert.match(readiness, /当前推理服务未声明 \$\{engineLabel\} RTX VSR 能力/);
  assert.match(readiness, /guidancePagCompatible = pagAvailableForEngine\(inferenceHealth, model\)/);
  assert.match(readiness, /guidanceReady = guidance === "pag" \? guidancePagCompatible : guidance !== "cfg_zero_star" \|\| guidanceFlowCompatible/);
  assert.match(readiness, /当前 \$\{model\} 推理运行时未声明 \$\{guidance === "pag" \? "PAG" : "CFG-Zero\*"\} 可用/);

  const unsupported = sourceBetween(readiness, "  const animaSettingsUnsupported", "  const pipelineConfigReady");
  assert.match(unsupported, /processPreview/);
  assert.doesNotMatch(unsupported, /guidance === "pag"|hires\.enabled|adetailer\.enabled|rtx\.enabled|loras\.length/);

  assert.match(app, /\(!engineAllowsHires && !hires\.enabled\)/);
  assert.match(app, /\(!engineAllowsADetailer && !adetailer\.enabled\)/);
  assert.match(app, /\(!engineAllowsRtx && !rtx\.enabled\)/);
  assert.match(app, /guidance === "pag" && !guidancePagCompatible/);
  assert.doesNotMatch(app, /disabled=\{isAnima && item\.id === "pag"\}/);
  assert.doesNotMatch(app, /setGuidance\(isAnima && next === "pag" \? "none" : next\)/);
  assert.doesNotMatch(app, /Anima 首版暂不支持 (?:LoRA|Hires\.fix|ADetailer|RTX VSR)/);
});

test("Anima LoRA discovery, gallery apply, and downloader avoid engine-specific clearing", async () => {
  const [app, downloader] = await Promise.all([readSource("src/App.jsx"), readDownloader()]);
  const applyGallery = sourceBetween(app, "  const applyGallerySettings = async", "  // Choosing a different engine");
  const selectModel = sourceBetween(app, "  const selectModel = (nextModel)", "  const selectCheckpoint = (nextCheckpoint)");
  const refreshLoras = sourceBetween(app, "  const refreshLoras = async", "  const openLoraManagerPage =");

  assert.match(app, /fetch\(`\/api\/loras\?engine=\$\{encodeURIComponent\(model\)\}`/);
  assert.doesNotMatch(refreshLoras, /engine === "Anima"/);
  assert.doesNotMatch(selectModel, /setLoras\(\[\]\)|setHires\([\s\S]*?enabled: false|setADetailer\([\s\S]*?enabled: false|setRtx\([\s\S]*?enabled: false/);
  assert.match(app, /inferenceHealthRef\.current = inferenceHealth/);
  assert.match(selectModel, /setGuidance\(\(currentGuidance\) => currentGuidance === "pag" && !pagAvailableForEngine\(inferenceHealthRef\.current, nextModel\) \? "none" : currentGuidance\)/);
  assert.match(selectModel, /setGuidance\(\(currentGuidance\) => currentGuidance === "cfg_zero_star" \? "none" : currentGuidance\)/);
  assert.match(applyGallery, /const targetLoraScopeKey = engineScopeKey\(targetLoraIdentity\.model\)/);
  assert.match(applyGallery, /const galleryTarget = galleryMountedLorasForTarget\(mountedLorasMapRef\.current/);
  assert.match(applyGallery, /applyLoras: selectedGroups\.has\("loras"\)/);
  assert.match(applyGallery, /const nextLoras = selectedGroups\.has\("loras"\) \? galleryTargetLoras : mountedLorasForScope\(mountedLorasMapRef\.current, targetLoraScopeKey\)/);
  assert.match(applyGallery, /features\[feature\] !== true/);
  assert.match(applyGallery, /fetch\(`\/api\/loras\?engine=\$\{encodeURIComponent\(targetModel\)\}`/);
  assert.doesNotMatch(applyGallery, /normalized\.model === "Anima" \? \[\]/);
  assert.doesNotMatch(applyGallery, /set(?:Hires|ADetailer|Rtx)\([\s\S]*?enabled: false/);

  assert.match(downloader, /engine === "Anima" \? "anima" : engine === "Flux" \? "flux" : engine === "Flux2" \? "flux2" : engine === "Krea2" \? "krea2" : engine === "iL" \? "illustrious" : "sd"/);
  assert.match(downloader, /kind === "lora" && <option value="Anima">Anima<\/option>/);
  assert.match(downloader, /modelPaths\?\.loras\?\.\[enginePathKey\]/);
});

test("Gallery preserves, edits, and displays Anima PAG with other supported settings", async () => {
  const [galleryCore, gallery] = await Promise.all([readGalleryCore(), readGalleryPage()]);
  const editor = sourceBetween(gallery, "function GalleryCardEditor", "function ApplySettingsDialog");
  const save = sourceBetween(editor, "  const save = async", "  return <div className=\"gallery-editor-backdrop\"");

  assert.match(galleryCore, /loras: \(Array\.isArray\(source\.loras\)[\s\S]*?\)\.slice\(0, 16\)/);
  assert.match(editor, /const setNested = \(group, key, value\) => setSettings\(\(current\) => busy \? current : \(\{ \.\.\.current/);
  assert.doesNotMatch(editor, /current\.model === "Anima" && key === "enabled"/);
  assert.match(save, /processPreview: false/);
  assert.match(save, /guidance: settings\.guidance/);
  assert.match(save, /!isAnima && settings\.guidance === "cfg_zero_star"/);
  assert.doesNotMatch(save, /loras: \[\]|(?:hires|adetailer|rtx): \{ \.\.\.settings\.(?:hires|adetailer|rtx), enabled: false \}/);
  assert.doesNotMatch(save, /settings\.guidance === "pag" \? "none"/);
  assert.match(editor, /guidance: nextModel !== "Anima" && current\.guidance === "cfg_zero_star" \? "none" : current\.guidance/);
  assert.match(editor, /const disabled = !isAnima && id === "cfg_zero_star"/);
  assert.doesNotMatch(editor, /isAnima && (?:id|settings\.guidance|next) === "pag"/);
  assert.doesNotMatch(editor, /Anima 不支持 PAG/);
  assert.match(editor, /label="启用 Hires\.fix"[\s\S]*?checked=\{settings\.hires\.enabled\} onChange=/);
  assert.match(editor, /label="启用 ADetailer"[\s\S]*?checked=\{settings\.adetailer\.enabled\} onChange=/);
  assert.match(editor, /label="启用 RTX VSR"[\s\S]*?checked=\{settings\.rtx\.enabled\} onChange=/);
  assert.doesNotMatch(editor, /label="启用 (?:Hires\.fix|ADetailer|RTX VSR)"[^>]*disabled=\{isAnima\}/);
  assert.doesNotMatch(gallery, /settings\.model === "Anima" \? 0 : settings\.loras/);
  assert.doesNotMatch(gallery, /Anima 首版不支持 (?:LoRA|Hires\.fix|ADetailer|RTX VSR)/);
});

test("manual Gallery card model selections persist through edit and workspace apply", async () => {
  const [app, gallery] = await Promise.all([readSource("src/App.jsx"), readGalleryPage()]);
  const editor = sourceBetween(gallery, "function GalleryCardEditor", "function ApplySettingsDialog");
  const save = sourceBetween(editor, "  const save = async", "  return <div className=\"gallery-editor-backdrop\"");
  const applyGallery = sourceBetween(app, "  const applyGallerySettings = async", "  // Choosing a different engine");

  assert.match(editor, /card\s*\? normalizedSettings\(card\.settings, DEFAULT_SETTINGS, \{ hiresSourceKind: "persisted_card" \}\)\s*: normalizedSettings\(undefined, initialSettings, \{ hiresSourceKind: "workspace_inheritance" \}\)/);
  assert.match(editor, /value=\{settings\.checkpoint\} onChange=\{\(event\) => setField\("checkpoint", event\.target\.value\)\}/);
  assert.match(editor, /value=\{settings\[key\]\} onChange=\{\(event\) => setField\(key, event\.target\.value\)\}/);
  assert.match(save, /const savedSettings = isAnima \? \{[\s\S]*?checkpoint: "",[\s\S]*?\} : \{[\s\S]*?diffusionModel: "",[\s\S]*?textEncoder: "",[\s\S]*?vae: "",/);
  assert.match(applyGallery, /checkpoint: typeof source\.checkpoint === "string" \? source\.checkpoint : "",/);
  assert.match(applyGallery, /diffusionModel: typeof source\.diffusionModel === "string" \? source\.diffusionModel : "",/);
  assert.match(applyGallery, /setRestoredWorkspace\(restored\);[\s\S]*?setCheckpoint\(normalized\.checkpoint\);[\s\S]*?setDiffusionModel\(normalized\.diffusionModel\);[\s\S]*?setTextEncoder\(normalized\.textEncoder\);[\s\S]*?setVae\(normalized\.vae\);/);
  assert.match(applyGallery, /if \(selectedGroups\.has\("model"\)\) \{[\s\S]*?setCheckpoint\(normalized\.checkpoint\);[\s\S]*?setDiffusionModel\(normalized\.diffusionModel\);[\s\S]*?setTextEncoder\(normalized\.textEncoder\);[\s\S]*?setVae\(normalized\.vae\);/);
});

test("main and Gallery Prompt composers document only the supported weight grammar", async () => {
  const [app, gallery] = await Promise.all([readSource("src/App.jsx"), readGalleryPage()]);
  for (const source of [app, gallery]) {
    assert.match(source, /\(text\)<\/code> = 1\.1/);
    assert.match(source, /\(text:1\.25\)<\/code> = 显式权重/);
    assert.match(source, /\\\(text\\\)<\/code> = 字面括号/);
    assert.doesNotMatch(source, /prompt-syntax-help[^\n]*(?:square|方括号|BREAK|embedding)/i);
  }
});

test("changing the selection does not tear the loaded pipeline down", async () => {
  const app = await readSource("src/App.jsx");
  const selectModel = sourceBetween(app, "  const selectModel = (nextModel)", "  const selectCheckpoint = (nextCheckpoint)");
  const selectCheckpoint = sourceBetween(app, "  const selectCheckpoint = (nextCheckpoint)", "  const selectSplitModelAsset =");
  const selectAsset = sourceBetween(app, "  const selectSplitModelAsset =", "  const refreshCheckpoints = async");
  const release = sourceBetween(app, "  const releaseLoadedModel = async", "  const unloadLoadedModel = async");

  // A picker change is a change of selection. Releasing first put a multi-gigabyte teardown — and a
  // full reload from disk on the next run — in front of a click, for a release both loaders perform
  // themselves when the request does not match what is loaded.
  for (const [name, body] of [["selectModel", selectModel], ["selectCheckpoint", selectCheckpoint], ["selectSplitModelAsset", selectAsset]]) {
    assert.doesNotMatch(body, /releaseLoadedModel/, `${name} must not release the loaded pipeline`);
    assert.doesNotMatch(body, /await /, `${name} has nothing to wait for`);
  }
  // Switching away and back must therefore find the cache still there.
  assert.match(selectModel, /if \(nextModel === model \|\| status === "running" \|\| modelSwitching\) return;/);
  // Freeing on demand is what the explicit control is for, and it is the one place that still asks.
  const unload = sourceBetween(app, "  const unloadLoadedModel = async", "  const applyGallerySettings = async");
  assert.match(unload, /releaseLoadedModel/);
  assert.match(unload, /!inferenceHealth\?\.model_cached/);
  // Nothing cached is nothing to release: the request is not free on the other side.
  assert.match(release, /if \(!inferenceHealth\?\.model_cached\) return true;/);
});
