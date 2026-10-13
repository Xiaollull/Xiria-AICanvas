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

  // The three blocks are one chain now, and one shared migration owns its shape.
  assert.match(restore, /postprocessChain: migrateResampleChain\(\{/);
  assert.match(restore, /hires: \{ \.\.\.savedHires, \.\.\.savedHiresSeed \}/);
  assert.match(restore, /adetailer: savedADetailer/);
  assert.match(restore, /rtx: savedRtx/);
  assert.match(restore, /const mountedLoraMap = normalizeMountedLoraMap\(saved\.mountedLorasByEngine/);
  assert.match(restore, /mountedLorasByEngine: mountedLoraMap\.container/);
  assert.match(restore, /loras: mountedLoraMap\.activeLoras/);
  // The latent process preview was withdrawn, so the workspace no longer restores a switch for it.
  assert.doesNotMatch(restore, /processPreview/);
  assert.match(restore, /guidance: guidanceOptions\.some\(\(item\) => item\.id === saved\.guidance\) \? saved\.guidance : "none"/);
  assert.doesNotMatch(restore, /enabled: !isAnima/);
  assert.doesNotMatch(restore, /loras: isAnima \? \[\]/);
  assert.doesNotMatch(restore, /saved\.guidance !== "pag"/);

  assert.match(generate, /const submittedSettings = JSON\.parse\(JSON\.stringify\(\{[\s\S]*?guidance: distilledGeneration \? "none" : guidance,\s+postprocessChain,/);
  assert.match(generate, /postprocess_chain: resampleChainPayload\(postprocessChain, \{ engine: model \}\)/);
  assert.match(generate, /scheduler: nativeGeneration && !ANIMA_SCHEDULERS\.includes\(scheduler\) \? "simple" : scheduler,\s+guidance: distilledGeneration \? "none" : guidance,\s+pag:/);
  assert.match(app, /import \{ SAMPLER_NAMES as samplerNames, SCHEDULER_NAMES as schedulerNames \} from "\.\/sampling-options"/);
  assert.match(app, /const ANIMA_SAMPLERS = samplerNames/);
  assert.match(app, /const ANIMA_SCHEDULERS = schedulerNames/);
  assert.match(generate, /const generationLoras = frozenMountedLorasForScope\(mountedLorasMapRef\.current, activeLoraScopeRef\.current\)/);
  assert.match(generate, /loras: generationLoras\.filter\(\(lora\) => lora\.enabled !== false\)/);
  assert.doesNotMatch(generate, /animaGeneration \? \{ \.\.\.(?:postprocessChain|adetailer|rtx), enabled: false \}/);
  assert.doesNotMatch(generate, /animaGeneration \? \[\] : loras/);
  assert.doesNotMatch(generate, /animaGeneration && guidance === "pag"/);
});

test("Hires migrates execution mode by engine and sends the independent tiled-refinement contract", async () => {
  const [app, galleryCore, galleryPage] = await Promise.all([readSource("src/App.jsx"), readGalleryCore(), readGalleryPage()]);
  const restore = sourceBetween(app, "function loadWorkspaceState(saved)", "function reconcileModels");
  const generate = sourceBetween(app, "  const generate = async () => {", "  const releaseLoadedModel = async () => {");

  // The per-unit execution mode and every tiling field now live in the chain module, which the
  // workspace, the gallery and image-to-image all normalise through.
  const chain = await readSource("src/resample-chain.js");
  // The stored mode is the user's: an engine switch reports an enabled unit it cannot tile as a run
  // blocker instead of rewriting it, and a disabled unit is carried through untouched.
  assert.match(chain, /const executionMode = source\.executionMode === "usdu_tiled" \? "usdu_tiled" : "full_frame";/);
  assert.match(chain, /if \(unit\.executionMode === "usdu_tiled" && !supportsUsduTiled\(engine\)\) return "USDU 分块重绘仅支持 Anima \/ Krea2 引擎";/);
  assert.match(chain, /sampler: samplers\.includes\(source\.sampler\) \? source\.sampler : null/);
  assert.match(chain, /padding: Math\.round\(inRange\(source\.padding, 32, 0, 256\)\)/);
  assert.match(chain, /maskBlur: Math\.round\(inRange\(source\.maskBlur, 8, 0, 64\)\)/);
  assert.match(chain, /execution_mode: stage\.executionMode === "usdu_tiled" \? "usdu_tiled" : "full_frame"/);
  for (const field of ["tile_width: \"auto\"", "tile_height: \"auto\"", "padding: stage.padding", "mask_blur: stage.maskBlur", "seam_mode: \"none\"", "uniform_tiles: stage.uniformTiles !== false", "tiled_decode: stage.tiledDecode !== false"]) assert.match(chain, new RegExp(field));
  assert.match(chain, /tile_size: stage\.tileSize,[\s\S]*?tile_overlap: stage\.tileOverlap,[\s\S]*?execution_mode/);
  assert.doesNotMatch(app, /tileWidth:\s*512|tileHeight:\s*512|tile_width:\s*512|tile_height:\s*512/);

  const panel = await readSource("src/ResampleChainPanel.jsx");
  assert.match(panel, /USDU 分块重绘（推荐）/);
  assert.match(panel, /整图重绘（兼容）/);
  assert.match(panel, /像素放大分块/);
  assert.match(panel, /跟随首轮/);
  assert.match(panel, /Auto = 进入本单元的画布宽高/);
  // Delegation to the shared Hires normalizer, not a reimplementation. The
  // specifier now carries its extension so the module also loads under plain
  // Node, which is what makes gallery-core unit-testable rather than only
  // checkable as source text.
  assert.match(galleryCore, /import \{ migrateResampleChain \} from "\.\/resample-chain\.js"/);
  assert.match(galleryCore, /postprocessChain: migrateResampleChain\(/);
  // The gallery editor renders the same panel as the workspace, so it offers the same modes.
  assert.match(galleryPage, /import ResampleChainPanel from "\.\/ResampleChainPanel\.jsx"/);
  assert.match(galleryPage, /<ResampleChainPanel\s+chain=\{settings\.postprocessChain\}/);
  for (const source of [galleryCore, galleryPage]) assert.doesNotMatch(source, /tileWidth:\s*512|tileHeight:\s*512/);
  assert.doesNotMatch(app, /Protocol 25 backend integration may return 422/);
  assert.match(app, /const hiresControlsLocked = modelSwitching;/);
  assert.match(panel, /像素放大分块[\s\S]*?disabled=\{locked\}/);
  assert.match(panel, /Auto = 进入本单元的画布宽高[\s\S]*?mask blur 8[\s\S]*?uniform tiles[\s\S]*?per-tile VAE tiled decode[\s\S]*?seam None[\s\S]*?每 tile 执行本单元步数/);
  assert.match(galleryPage, /aria-pressed=\{selected\.has\(id\)\} disabled=\{busy\}/);
  assert.match(galleryPage, /USDU 分块重绘/);
});

test("ADetailer request controls stay editable for the next queued task", async () => {
  const app = await readSource("src/App.jsx");
  const units = await readSource("src/adetailer-units.js");
  const generate = sourceBetween(app, "  const generate = async () => {", "  const releaseLoadedModel = async () => {");
  const adetailer = sourceBetween(app, "              <section className={`postprocess-stage-card ${adetailer.enabled", "              <section className={`postprocess-stage-card ${rtx.enabled");

  assert.match(app, /const adetailerLocked = modelSwitching/);
  // Every field still reaches the request; the payload is built in one place so
  // the two generate surfaces cannot send different shapes.
  // The stage's own payload is unchanged; only where it sits in the request moved.
  const chainModule = await readSource("src/resample-chain.js");
  assert.match(chainModule, /return \{ type: "adetailer", \.\.\.adetailerPayload\(stage, engine\) \};/);
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

  // A disabled stage is still configurable before it is enabled. Every per-unit
  // control therefore follows only an actual model-switch transaction through `unitLocked`.
  assert.match(adetailer, /const unitLocked = adetailerLocked;/);
  assert.doesNotMatch(adetailer, /const unitLocked = !adetailer\.enabled/);
  // The unit's fields are drawn by the component Basic shares with the XYZ chain owner; Basic hands
  // it the same lock and negative-prompt gate, and keeps the stage and unit switches itself.
  assert.match(adetailer, /<ADetailerUnitControls unit=\{unit\} unitLocked=\{unitLocked\} models=\{adetailerModels\} engineAllowsNegativePrompt=\{engineAllowsNegativePrompt\}/);
  for (const control of [
    "switchDisabled=\\{adetailerLocked",
    // A unit's own switch is a request control too. There is nothing to add or
    // remove: every slot exists from the start, so the switch is the whole gate.
    "aria-label=\\{`启用第 \\$\\{index \\+ 1\\} 个 ADetailer 单元`\\}[\\s\\S]*?disabled=\\{unitLocked\\}",
  ]) assert.match(adetailer, new RegExp(control));
  const unitControls = sourceBetween(await readSource("src/PostprocessAuxControls.jsx"), "export function ADetailerUnitControls", "export function RTXControls");
  for (const control of [
    "value=\\{unit\\.detector\\} disabled=\\{unitLocked\\}",
    "label=\"检测置信度\"[\\s\\S]*?disabled=\\{unitLocked\\}",
    "label=\"最多处理区域\"[\\s\\S]*?disabled=\\{unitLocked\\}",
    "label=\"最小区域比例\"[\\s\\S]*?disabled=\\{unitLocked\\}",
    "label=\"最大区域比例\"[\\s\\S]*?disabled=\\{unitLocked\\}",
    // Dilate/erode, mask blur and padding are one mapped row of number inputs.
    "\\[\"dilateErode\", \"膨胀 / 腐蚀\", -128, 128\\], \\[\"maskBlur\", \"蒙版模糊\", 0, 64\\], \\[\"padding\", \"局部边距\", 0, 256\\]\\][\\s\\S]*?value=\\{unit\\[field\\]\\}[\\s\\S]*?disabled=\\{unitLocked\\}",
    "label=\"重绘强度\"[\\s\\S]*?disabled=\\{unitLocked\\}",
    "checked=\\{unit\\.useSteps\\} disabled=\\{unitLocked\\}",
    "value=\\{unit\\.steps\\}[\\s\\S]*?unitLocked",
    "checked=\\{unit\\.useCfg\\} disabled=\\{unitLocked\\}",
    "value=\\{unit\\.cfg\\}[\\s\\S]*?unitLocked",
    "value=\\{unit\\.prompt\\} disabled=\\{unitLocked\\}",
    // The negative prompt carries a second gate: a guidance-distilled engine has no branch to
    // encode one into. The model-switch lock is still the first term.
    "value=\\{unit\\.negativePrompt\\} disabled=\\{unitLocked \\|\\| !engineAllowsNegativePrompt\\}",
  ]) assert.match(unitControls, new RegExp(control));
  assert.doesNotMatch(adetailer, /添加检测单元|删除第/);
  assert.doesNotMatch(adetailer, /cancel|取消任务/i);
  // The resample unit's own controls follow the same lock, from the shared panel.
  assert.match(await readSource("src/ResampleChainPanel.jsx"), /像素放大分块[\s\S]*?disabled=\{locked\}/);
  assert.match(app, /locked=\{hiresControlsLocked\}/);
});

test("performance settings expose a hardware-derived VRAM wall slider", async () => {
  const app = await readSource("src/App.jsx");
  const performance = sourceBetween(app, "{settingsTab === \"performance\" && <section", "{settingsTab === \"theme\"");
  // The baseline lives in its own module now, because one of its entries follows the installed card.
  assert.match(await readSource("src/performance-defaults.js"), /vram_limit_gb: 0/);
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

  // Basic and XYZ read PAG through one helper; only an explicit "available" enables it.
  assert.match(pagHealth, /return runtimeGuidanceHealthStatus\(health, engine, "pag"\)\.status === "available";/);
  const runtime = sourceBetween(await readSource("src/xyz-axis-editor.js"), "export function runtimeGuidanceHealthStatus", "function guidanceHealthStatus");
  assert.match(runtime, /if \(!capability \|\| typeof capability\.available !== "boolean"\) \{\s*return \{ status: "unknown"/);
  assert.match(runtime, /if \(capability\.available === false\) return \{ status: "unavailable"/);
  assert.match(runtime, /if \(!Array\.isArray\(capability\.engines\)\) return \{ status: "unknown"/);
  assert.match(runtime, /if \(!capability\.engines\.includes\(engine\)\) return \{ status: "unavailable"/);
  // The engine must still prove PAG: its own feature flag or the runtime's named implementation.
  assert.match(runtime, /const feature = health\?\.engines\?\.\[engine\]\?\.features\?\.pag;/);
  assert.match(runtime, /if \(feature === false\) return \{ status: "unavailable"/);
  assert.match(runtime, /if \(feature === true \|\| capability\.implementations\?\.\[engine\] === PAG_IMPLEMENTATIONS\[engine\]\) return \{ status: "available"/);
  assert.match(runtime, /return \{ status: "unknown", reason: `当前引擎 \$\{engine\} 缺少明确 PAG 实现或特性证明` \};/);

  // A split-model engine must be told a capability is present; a checkpoint engine only has to not
  // be told it is absent. That asymmetry is the fail-closed rule, and it now covers Flux as well.
  assert.match(readiness, /engineAllowsLora = isSplitModel \? selectedEngineFeatures\.lora === true/);
  assert.match(readiness, /engineAllowsHires = isSplitModel \? selectedEngineFeatures\.hires === true/);
  assert.match(readiness, /engineAllowsADetailer = isSplitModel \? selectedEngineFeatures\.adetailer === true/);
  assert.match(readiness, /engineAllowsRtx = isSplitModel \? selectedEngineFeatures\.rtx === true/);
  assert.match(readiness, /loraReady = enabledLoras\.length === 0 \|\| engineAllowsLora/);
  // The engine names itself in the message now that two engines can produce it.
  assert.match(readiness, /当前推理服务未声明 \$\{engineLabel\} LoRA 能力/);
  assert.match(readiness, /当前推理服务未声明 \$\{engineLabel\} 重采样能力/);
  assert.match(readiness, /当前推理服务未声明 \$\{engineLabel\} ADetailer 能力/);
  assert.match(readiness, /当前推理服务未声明 \$\{engineLabel\} RTX VSR 能力/);
  assert.match(readiness, /guidancePagCompatible = pagAvailableForEngine\(inferenceHealth, model\)/);
  assert.match(readiness, /guidanceReady = guidance === "pag" \? guidancePagCompatible : guidance !== "cfg_zero_star" \|\| guidanceFlowCompatible/);
  assert.match(readiness, /当前 \$\{model\} 推理运行时未声明 \$\{guidance === "pag" \? "PAG" : "CFG-Zero\*"\} 可用/);

  const unsupported = sourceBetween(readiness, "  const animaSettingsUnsupported", "  const pipelineConfigReady");
  // Sampling and guidance are all that can still be unsupported here; the preview switch that
  // used to be the third case is gone.
  assert.match(unsupported, /activeSamplerNames\.includes\(sampler\)/);
  assert.doesNotMatch(unsupported, /processPreview/);
  assert.doesNotMatch(unsupported, /guidance === "pag"|hires\.enabled|adetailer\.enabled|rtx\.enabled|loras\.length/);

  assert.match(app, /\(!engineAllowsHires && !enabled\)/);
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
  // PAG availability is the one guidance rule the per-engine store cannot decide, because it
  // depends on what the installed runtime reports rather than on the engine alone, so it is still
  // applied here after the engine's own record has been restored. The static rules moved into
  // `engine-settings.js` and are covered by `engine-settings.test.mjs`.
  assert.match(selectModel, /restored\.guidance === "pag" && !pagAvailableForEngine\(inferenceHealthRef\.current, nextModel\)/);
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
  assert.doesNotMatch(save, /processPreview/);
  assert.match(save, /guidance: settings\.guidance/);
  // Guidance is refused by the engine's own rules (engine-settings), not by an Anima-only check.
  assert.match(save, /if \(guidanceBlocked\) \{/);
  assert.doesNotMatch(save, /loras: \[\]|(?:hires|adetailer|rtx): \{ \.\.\.settings\.(?:hires|adetailer|rtx), enabled: false \}/);
  assert.doesNotMatch(save, /settings\.guidance === "pag" \? "none"/);
  assert.match(editor, /guidance: engineGuidance\(nextModel, current\.guidance\)/);
  // Each option is enabled by the selected engine's own guidance rules, shared with the workspace.
  assert.match(editor, /const disabled = engineGuidance\(settings\.model, id\) !== id;/);
  assert.doesNotMatch(editor, /isAnima && (?:id|settings\.guidance|next) === "pag"/);
  assert.doesNotMatch(editor, /Anima 不支持 PAG/);
  assert.match(editor, /<summary><Zap size=\{14\} \/>后处理链<\/summary>/);
  assert.match(editor, /<ResampleChainPanel\s+chain=\{settings\.postprocessChain\}/);
  // Every stage in the chain is switched the same way, whatever its type: one foldable card whose
  // switch is only held while the editor is busy, never by the engine.
  const chainCard = sourceBetween(gallery, "function GalleryChainCard", "\n}\n");
  assert.match(chainCard, /<ToggleField label="启用" detail="" checked=\{enabled\} disabled=\{disabled\} onChange=\{onToggleEnabled\} \/>/);
  const cards = editor.match(/<GalleryChainCard [^>]*>/g) || [];
  assert.equal(cards.length, 3);
  for (const card of cards) assert.match(card, / disabled=\{busy\} /);
  assert.equal((editor.match(/onToggleEnabled=\{\(value\) => setStage\(stage\.id, \{ enabled: value \}\)\}/g) || []).length, 2);
  assert.doesNotMatch(editor, /label="启用 (?:Hires\.fix|ADetailer|RTX VSR)"[^>]*disabled=\{isAnima\}/);
  assert.doesNotMatch(gallery, /settings\.model === "Anima" \? 0 : settings\.loras/);
  assert.doesNotMatch(gallery, /Anima 首版不支持 (?:LoRA|Hires\.fix|ADetailer|RTX VSR)/);
});

test("manual Gallery card model selections persist through edit and workspace apply", async () => {
  const [app, gallery] = await Promise.all([readSource("src/App.jsx"), readGalleryPage()]);
  const editor = sourceBetween(gallery, "function GalleryCardEditor", "function ApplySettingsDialog");
  const save = sourceBetween(editor, "  const save = async", "  return <div className=\"gallery-editor-backdrop\"");
  const applyGallery = sourceBetween(app, "  const applyGallerySettings = async", "  // Choosing a different engine");

  assert.match(editor, /card\s*\? normalizedSettings\(card\.settings, DEFAULT_SETTINGS\)\s*: normalizedSettings\(undefined, initialSettings\)/);
  assert.match(editor, /value=\{settings\.checkpoint\} onChange=\{\(event\) => setField\("checkpoint", event\.target\.value\)\}/);
  assert.match(editor, /value=\{settings\[key\]\} onChange=\{\(event\) => setField\(key, event\.target\.value\)\}/);
  assert.match(save, /const savedSettings = isSplit \? \{[\s\S]*?checkpoint: "",[\s\S]*?\} : \{[\s\S]*?diffusionModel: "",[\s\S]*?textEncoder: "",[\s\S]*?vae: "",/);
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
  const selectModel = sourceBetween(app, "  const selectModel = (nextModel)", "  const loraEngineSwitchBlocker = ");
  // The LoRA manager's engine bar switches through selectModel and inherits the same rule.
  const loraSwitch = sourceBetween(app, "  const switchEngineFromLoraManager = (engine)", "  loraEngineSwitchRef.current = ");
  const selectCheckpoint = sourceBetween(app, "  const selectCheckpoint = (nextCheckpoint)", "  const selectSplitModelAsset =");
  const selectAsset = sourceBetween(app, "  const selectSplitModelAsset =", "  const refreshCheckpoints = async");
  const release = sourceBetween(app, "  const releaseLoadedModel = async", "  const unloadLoadedModel = async");

  // A picker change is a change of selection. Releasing first put a multi-gigabyte teardown — and a
  // full reload from disk on the next run — in front of a click, for a release both loaders perform
  // themselves when the request does not match what is loaded.
  assert.match(loraSwitch, /selectModel\(engine\);/);
  for (const [name, body] of [["selectModel", selectModel], ["switchEngineFromLoraManager", loraSwitch], ["selectCheckpoint", selectCheckpoint], ["selectSplitModelAsset", selectAsset]]) {
    assert.doesNotMatch(body, /releaseLoadedModel/, `${name} must not release the loaded pipeline`);
    assert.doesNotMatch(body, /await /, `${name} has nothing to wait for`);
  }
  // Switching away and back must therefore find the cache still there.
  assert.match(selectModel, /if \(nextModel === model \|\| modelSwitching\) return;/);
  // Freeing on demand is what the explicit control is for, and it is the one place that still asks.
  const unload = sourceBetween(app, "  const unloadLoadedModel = async", "  const applyGallerySettings = async");
  assert.match(unload, /releaseLoadedModel/);
  assert.match(unload, /!inferenceHealth\?\.model_cached/);
  // Nothing cached is nothing to release: the request is not free on the other side.
  assert.match(release, /if \(!inferenceHealth\?\.model_cached\) return true;/);
});
