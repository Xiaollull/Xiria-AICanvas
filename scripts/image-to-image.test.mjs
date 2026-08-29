import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  IMAGE_TO_IMAGE_DEFAULTS,
  IMAGE_TO_IMAGE_MODES,
  MAXIMUM_EDGE,
  RESIZE_MODES,
  SIZE_MODES,
  fitSourceSize,
  imageToImageBlockers,
  imageToImageRequestBody,
  nextImageToImageSeed,
  normalizeImageToImageSettings,
  normalizeSourceSeed,
  outputSize,
  postprocessSourceIssue,
  resizeModeMatters,
  snapEdge,
  sourceImageSummary,
  validateSourceFile,
} from "../src/image-to-image.js";

const readSource = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");
const source = (width, height, extra = {}) => ({ width, height, dataUrl: "data:image/png;base64,AAAA", name: "a.png", type: "image/png", size: 1024, ...extra });

test("a damaged settings block falls back to defaults instead of failing", () => {
  for (const value of [null, undefined, "settings", [], 7]) {
    assert.deepEqual(normalizeImageToImageSettings(value), normalizeImageToImageSettings(IMAGE_TO_IMAGE_DEFAULTS));
  }
  const repaired = normalizeImageToImageSettings({
    denoise: 9, steps: -4, cfg: 900, imagesPerBatch: 40, batchCount: 0,
    sizeMode: "elsewhere", resizeMode: "tile", seedMode: "spiral", seed: "-12abc",
  });
  assert.equal(repaired.denoise, 1);
  assert.equal(repaired.steps, 1);
  assert.equal(repaired.cfg, 30);
  assert.equal(repaired.imagesPerBatch, 10);
  assert.equal(repaired.batchCount, 1);
  assert.equal(repaired.sizeMode, IMAGE_TO_IMAGE_DEFAULTS.sizeMode);
  assert.equal(repaired.resizeMode, IMAGE_TO_IMAGE_DEFAULTS.resizeMode);
  assert.equal(repaired.seedMode, IMAGE_TO_IMAGE_DEFAULTS.seedMode);
  assert.equal(repaired.seed, "12");
});

test("denoise is never normalised to zero, because zero would ask the sampler to change nothing", () => {
  assert.equal(normalizeImageToImageSettings({ denoise: 0 }).denoise, 0.05);
  assert.equal(normalizeImageToImageSettings({ denoise: -3 }).denoise, 0.05);
});

test("sampler and scheduler fall back inside the engine's own catalogue", () => {
  const catalogues = { samplers: ["euler", "dpmpp_2m"], schedulers: ["simple", "karras"], defaultSampler: "dpmpp_2m", defaultScheduler: "karras" };
  assert.equal(normalizeImageToImageSettings({ sampler: "euler" }, catalogues).sampler, "euler");
  assert.equal(normalizeImageToImageSettings({ sampler: "ddim" }, catalogues).sampler, "dpmpp_2m");
  assert.equal(normalizeImageToImageSettings({ scheduler: "beta" }, catalogues).scheduler, "karras");
  // An empty catalogue means the engine list has not loaded yet; keeping the stored value beats
  // rewriting it to "" and losing the user's choice on the next save.
  assert.equal(normalizeImageToImageSettings({ sampler: "ddim" }).sampler, "ddim");
});

test("an oversized source is scaled to fit before it is snapped, so its framing survives", () => {
  // Clamping 4000 to the 2048 maximum while snapping 3000 on its own would turn a 4:3 picture into
  // a 2:3 one without saying so.
  const fitted = fitSourceSize(4000, 3000);
  assert.ok(Math.max(fitted.width, fitted.height) <= MAXIMUM_EDGE);
  assert.ok(Math.abs(fitted.width / fitted.height - 4 / 3) < 0.03, `${fitted.width}x${fitted.height}`);
  assert.equal(fitted.width % 64, 0);
  assert.equal(fitted.height % 64, 0);
});

test("every canvas edge is a multiple of 64 inside the server's own bounds", () => {
  assert.equal(snapEdge(1), 64);
  assert.equal(snapEdge(0), 64);
  assert.equal(snapEdge(-800), 64);
  assert.equal(snapEdge(99999), 2048);
  assert.equal(snapEdge(1080), 1088);
  assert.equal(snapEdge("1920"), 1920);
});

test("the size mode decides the canvas, and a missing source falls back to the stored one", () => {
  const settings = { sizeMode: "source", scale: 1, width: 512, height: 768 };
  assert.deepEqual(outputSize(source(1920, 1080), settings), { width: 1920, height: 1088 });
  assert.deepEqual(outputSize(source(1920, 1080), { ...settings, sizeMode: "scale", scale: 0.5 }), { width: 960, height: 512 });
  assert.deepEqual(outputSize(source(1920, 1080), { ...settings, sizeMode: "custom" }), { width: 512, height: 768 });
  assert.deepEqual(outputSize(null, settings), { width: 512, height: 768 });
});

test("the resize mode is only claimed to matter when the aspect ratios differ", () => {
  assert.equal(resizeModeMatters(source(1000, 500), { width: 1024, height: 512 }), false);
  assert.equal(resizeModeMatters(source(1000, 500), { width: 512, height: 512 }), true);
  assert.equal(resizeModeMatters(null, { width: 512, height: 512 }), false);
});

test("a source file is refused for type and for size, with a readable reason", () => {
  assert.equal(validateSourceFile({ type: "image/png", size: 1024 }), "");
  assert.match(validateSourceFile({ type: "image/gif", size: 1024 }), /PNG/);
  assert.match(validateSourceFile({ type: "image/png", size: 64 * 1024 * 1024 }), /上限/);
  assert.match(validateSourceFile(null), /选择/);
  assert.match(sourceImageSummary(source(800, 600)), /800 × 600/);
});

test("the run blocker names the first thing the user can act on", () => {
  const ready = { source: source(512, 512), settings: { positive: "a lantern" }, engineReady: true, serviceReady: true, running: false };
  assert.equal(imageToImageBlockers(ready), "");
  assert.match(imageToImageBlockers({ ...ready, source: null }), /来源图片/);
  assert.match(imageToImageBlockers({ ...ready, settings: { positive: "   " } }), /提示词/);
  assert.match(imageToImageBlockers({ ...ready, engineReady: false }), /模型/);
  assert.match(imageToImageBlockers({ ...ready, serviceReady: false }), /推理服务/);
  assert.match(imageToImageBlockers({ ...ready, running: true }), /正在生成/);
});

test("the request body carries the picture, the strength and nothing the page does not own", () => {
  const body = imageToImageRequestBody({
    engine: "SD",
    checkpoint: "model.safetensors",
    source: source(1024, 1024, { dataUrl: "data:image/png;base64,PIXELS", name: "cat.png" }),
    settings: { positive: "  a lantern  ", negative: " blurry ", denoise: 0.45, steps: 24, cfg: 7, sampler: "dpmpp_2m", scheduler: "karras", resizeMode: "contain" },
    seed: "99",
    loras: [{ value: "a.safetensors", weight: 0.8 }, { value: "b.safetensors", weight: 1, enabled: false }],
  });
  assert.equal(body.engine, "SD");
  assert.equal(body.checkpoint, "model.safetensors");
  assert.equal(body.prompt, "a lantern");
  assert.equal(body.negative_prompt, "blurry");
  assert.equal(body.denoise, 0.45);
  assert.equal(body.seed, "99");
  assert.equal(body.preview_enabled, true);
  assert.deepEqual(body.source_image, {
    enabled: true,
    image_data: "data:image/png;base64,PIXELS",
    resize_mode: "contain",
    name: "cat.png",
  });
  assert.deepEqual(body.postprocess_order, ["hires", "adetailer", "rtx"]);
  assert.deepEqual(body.loras, [{ path: "a.safetensors", weight: 0.8 }]);
  assert.equal(body.hires.enabled, false);
  assert.equal(body.adetailer.enabled, false);
  assert.equal(body.rtx.enabled, false);
  for (const key of ["diffusion_model", "text_encoder", "vae"]) assert.equal(key in body, false, `${key} must not be sent`);
});

test("i2i owns independent postprocess settings and emits the existing stage contract", () => {
  const settings = normalizeImageToImageSettings({
    positive: "portrait",
    hires: { enabled: true, model: "anime-upscaler", scale: 2, denoise: 0.2, steps: 12, seedMode: "fixed", seed: "123" },
    adetailer: { enabled: true, detector: "face_yolo", confidence: 0.8 },
    rtx: { enabled: true, scale: 1.5, quality: "high" },
    postprocessOrder: ["rtx", "hires", "adetailer"],
  });
  assert.equal(settings.hires.enabled, true);
  assert.equal(settings.adetailer.enabled, true);
  assert.equal(settings.rtx.quality, "high");
  assert.deepEqual(settings.postprocessOrder, ["rtx", "hires", "adetailer"]);
  const body = imageToImageRequestBody({ engine: "SD", checkpoint: "model.safetensors", source: source(1024, 1024), settings, seed: "99" });
  assert.equal(body.hires.enabled, true);
  assert.equal(body.hires.scale, 2);
  assert.equal(body.hires.mode, "fixed");
  assert.equal(body.hires.seed, "123");
  // The stage sends a run plan: one entry per unit that will actually run.
  assert.equal(body.adetailer.units.length, 1);
  assert.equal(body.adetailer.units[0].detector, "face_yolo");
  assert.equal(body.rtx.scale, 1.5);
  assert.deepEqual(body.postprocess_order, ["rtx", "hires", "adetailer"]);
});

test("i2i exposes and sends the full text-to-image Hires and ADetailer parameter sets", async () => {
  const settings = normalizeImageToImageSettings({
    positive: "portrait",
    hires: { enabled: true, model: "upscaler", cfg: 9.5, sampler: "euler", scheduler: "simple", tileSize: 128, tileOverlap: 999 },
    adetailer: { enabled: true, detector: "face", confidence: 0.65, maxDetections: 6, maskMinRatio: 0.08, maskMaxRatio: 0.72, dilateErode: -3, maskBlur: 9, padding: 48, denoise: 0.55, useSteps: true, steps: 33, useCfg: true, cfg: 8.5, prompt: " detailed face ", negativePrompt: " blur " },
  }, { samplers: ["euler"], schedulers: ["simple"] });
  assert.equal(settings.hires.tileOverlap, 64, "overlap is clamped to half the selected tile size");
  const body = imageToImageRequestBody({ engine: "SD", checkpoint: "model.safetensors", source: source(1024, 1024), settings, seed: "99", samplers: ["euler"], schedulers: ["simple"] });
  assert.deepEqual({ cfg: body.hires.cfg, sampler: body.hires.sampler, scheduler: body.hires.scheduler, tile_size: body.hires.tile_size, tile_overlap: body.hires.tile_overlap }, { cfg: 9.5, sampler: "euler", scheduler: "simple", tile_size: 128, tile_overlap: 64 });
  const unit = body.adetailer.units[0];
  assert.deepEqual({ max_detections: unit.max_detections, mask_min_ratio: unit.mask_min_ratio, mask_max_ratio: unit.mask_max_ratio, dilate_erode: unit.dilate_erode, mask_blur: unit.mask_blur, padding: unit.padding, use_steps: unit.use_steps, steps: unit.steps, use_cfg: unit.use_cfg, cfg: unit.cfg, prompt: unit.prompt, negative_prompt: unit.negative_prompt }, { max_detections: 6, mask_min_ratio: 0.08, mask_max_ratio: 0.72, dilate_erode: -3, mask_blur: 9, padding: 48, use_steps: true, steps: 33, use_cfg: true, cfg: 8.5, prompt: "detailed face", negative_prompt: "blur" });
  const page = await readSource("src/ImageToImagePage.jsx");
  for (const label of ["Hires CFG", "Hires 采样器", "Hires 调度器", "像素放大分块", "最多处理区域", "最小区域比例", "膨胀 / 腐蚀", "蒙版模糊", "局部边距", "独立步数", "独立 CFG", "正向提示词", "负向提示词"]) assert.ok(page.includes(label), `${label} control is required`);

  // Several detail passes in one run, each with its own settings: a face model
  // and a hand model are the case this exists for.
  const pair = normalizeImageToImageSettings({
    adetailer: { enabled: true, units: [
      { detector: "face.pt", denoise: 0.45, prompt: "face" },
      { detector: "hand.pt", denoise: 0.3, prompt: "hand" },
      { detector: "eyes.pt", enabled: false },
    ] },
  });
  const pairBody = imageToImageRequestBody({ engine: "SD", checkpoint: "model.safetensors", source: source(1024, 1024), settings: pair, seed: "1" });
  // Only the units that will run are sent, in the order they will run.
  assert.deepEqual(pairBody.adetailer.units.map((entry) => entry.detector), ["face.pt", "hand.pt"]);
  assert.deepEqual(pairBody.adetailer.units.map((entry) => entry.denoise), [0.45, 0.3]);
  assert.deepEqual(pairBody.adetailer.units.map((entry) => entry.prompt), ["face", "hand"]);
  // Units are addressed by a stable id, and normalising is idempotent.
  assert.deepEqual(pair.adetailer.units.map((entry) => entry.id), normalizeImageToImageSettings(pair).adetailer.units.map((entry) => entry.id));
});

test("both generate surfaces page through ADetailer units instead of stacking cards", async () => {
  const app = await readSource("src/App.jsx");
  const page = await readSource("src/ImageToImagePage.jsx");
  const styles = await readSource("src/styles.css");

  for (const [name, source, prefix] of [["App.jsx", app, "adetailer"], ["ImageToImagePage.jsx", page, "i2i-adetailer"]]) {
    // One unit is on screen; the rest are pages behind it. A `.map` over the
    // units into unit sections would be the stacked layout coming back.
    assert.doesNotMatch(source, /adetailerUnits\.map\(\((?:unit|entry), index\)/, `${name} must render one unit, not every unit`);
    // Nothing declares a page count: it is however many units exist.
    assert.match(source, new RegExp(`adetailerUnits\\.length > 1 && <nav className="${prefix}-pager"`), `${name} needs an automatic pager`);
    assert.match(source, /adetailerUnits\.map\(\(entry, position\) =>[\s\S]*?onClick=\{\(\) => setADetailerPage\(entry\.id\)\}/, `${name} needs one page control per unit`);
    // The arrows stop at the ends rather than wrapping past them.
    assert.match(source, /aria-label="上一个检测单元" disabled=\{(?:adetailerIndex|index) < 1\}/, `${name} must disable the left arrow on the first page`);
    assert.match(source, /aria-label="下一个检测单元" disabled=\{(?:adetailerIndex|index) >= adetailerUnits\.length - 1\}/, `${name} must disable the right arrow on the last page`);
    // A closed page still has to report whether its unit will run.
    assert.match(source, /className=\{`\$\{position === (?:adetailerIndex|index) \? "current" : ""\} \$\{entry\.enabled \? "on" : ""\}`\}/, `${name} page controls must show each unit's switch state`);
    // The page is resolved from an id, not a slot number.
    assert.match(source, /adetailerPageIndex\(adetailerUnits, adetailerPage\)/, `${name} must resolve the page by id`);
    // Every slot exists from the start, so there is nothing to add and nothing to
    // delete — a second pass is a switch to turn on.
    assert.doesNotMatch(source, /添加检测单元|删除第/, `${name} must not offer add or remove`);
    // The number inputs hold an uncommitted draft of their own, so the unit has
    // to key its controls: reused ones would show one unit's half-typed value on
    // the next unit's field.
    assert.match(source, new RegExp(`<section className=\\{\`${prefix}-unit \\$\\{unit\\.enabled \\? "on" : ""\\}\`\\} key=\\{unit\\.id\\}>`), `${name} must key the unit section by unit id`);
  }
  // The footer states how the stage runs rather than offering an add control.
  for (const source of [app, page]) assert.match(source, /个检测单元 · 已启用 \{activeADetailerUnits\([^)]*\)\.length\} 个 · 按编号顺序依次执行/);
  // Which page is open is never persisted: it says nothing about the render.
  assert.doesNotMatch(app, /adetailerPage:/);
  assert.doesNotMatch(page, /adetailerPage:/);

  for (const rule of [".adetailer-pager", ".i2i-adetailer-pager", ".adetailer-pager ol button.current", ".i2i-adetailer-pager ol button.current"]) {
    assert.ok(styles.includes(rule), `${rule} must be styled`);
  }
  assert.match(styles, /html\[data-theme-mode="light"\] \.adetailer-pager,/);
});

test("an Anima request sends split model assets and refuses process previews", () => {
  const body = imageToImageRequestBody({
    engine: "Anima",
    diffusionModel: "d.safetensors",
    textEncoder: "t.safetensors",
    vae: "v.safetensors",
    source: source(1024, 1024),
    settings: { positive: "a lantern" },
    seed: "1",
  });
  assert.equal(body.diffusion_model, "d.safetensors");
  assert.equal(body.preview_enabled, false);
  assert.equal("checkpoint" in body, false);
});

test("the seed walks the same way it does on the text-to-image page", () => {
  const settings = { imagesPerBatch: 2, batchCount: 3, seedMode: "increment" };
  assert.equal(nextImageToImageSeed(settings, "10"), "16");
  assert.equal(nextImageToImageSeed({ ...settings, seedMode: "decrement" }, "2"), "18446744073709551612");
  assert.equal(nextImageToImageSeed({ ...settings, seedMode: "random" }, "10"), "10");
  assert.equal(normalizeSourceSeed("99999999999999999999999"), "18446744073709551615");
});

test("the source picture is session state and never reaches the persisted workspace", async () => {
  // `ui-state.json` is rewritten on every settings change; a multi-megabyte data URL in the
  // workspace snapshot would make each autosave proportional to the picture.
  const app = await readSource("src/App.jsx");
  const snapshot = app.slice(app.indexOf("workspaceSnapshot.current = {"), app.indexOf("uiStateSnapshot.current"));
  assert.match(snapshot, /\bimageToImage,/);
  assert.equal(/imageSource/.test(snapshot), false);
  assert.match(app, /const \[imageSource, setImageSource\] = useState\(null\)/);
});

test("the image-to-image entry point sits immediately to the right of text-to-image", async () => {
  const app = await readSource("src/App.jsx");
  const nav = app.slice(app.indexOf('<nav className="main-nav"'), app.indexOf("</nav>"));
  const labels = [...nav.matchAll(/>([一-龥]+)<\/button>/g)].map((match) => match[1]);
  assert.deepEqual(labels.slice(0, 2), ["文生图", "图生图"]);
  assert.match(nav, /activePage === "image"/);
});

test("the page is a lazily loaded route, not part of the workspace chunk", async () => {
  const app = await readSource("src/App.jsx");
  assert.match(app, /const ImageToImagePage = lazy\(\(\) => import\("\.\/ImageToImagePage"\)\);/);
  assert.match(app, /activePage === "image" \? <Suspense/);
});

test("image-to-image can switch the shared engine and model assets", async () => {
  const [app, page] = await Promise.all([readSource("src/App.jsx"), readSource("src/ImageToImagePage.jsx")]);
  assert.match(page, /modelPicker\.onSelectEngine\(item\.name\)/);
  assert.match(page, /modelPicker\.onSelectCheckpoint/);
  assert.match(page, /modelPicker\.onSelectAsset\(asset\.kind, value\)/);
  assert.match(page, /ariaLabel="图生图底模选择"/);
  const props = app.slice(app.indexOf('activePage === "image" ? <Suspense'), app.indexOf('activePage === "gallery" ? <Suspense'));
  for (const callback of ["selectModel", "selectCheckpoint", "selectSplitModelAsset", "refreshCheckpoints"]) assert.ok(props.includes(callback), `${callback} is passed to image-to-image`);
});

test("the page is a stage plus a control rail, and every class it uses is styled", async () => {
  const css = await readSource("src/styles.css");
  const page = await readSource("src/ImageToImagePage.jsx");
  assert.match(css, /^\.image-workspace \{[^}]*display: grid;/m);
  assert.match(page, /className="i2i-stage-panel panel"/);
  assert.match(page, /className="i2i-controls-panel panel"/);
  for (const selector of [".i2i-compare", ".i2i-compare-pane", ".i2i-dropzone", ".i2i-denoise", ".i2i-mode-row", ".i2i-canvas-readout"]) {
    assert.ok(css.includes(`${selector} {`), `${selector} must be styled`);
    assert.ok(page.includes(selector.slice(1)), `${selector} must be used`);
  }
});

test("the LoRA block is the text-to-image one, not a second spelling of it", async () => {
  // Both pages read one mounted library. Two renderings of it would drift, and a user moving
  // between the pages would see the same mounts described two different ways.
  const [app, page] = await Promise.all([readSource("src/App.jsx"), readSource("src/ImageToImagePage.jsx")]);
  // Two substitutions, and only two: the array each page reads the mounts from, and how each opens
  // the manager — a prop here, App's own setters there. Everything that decides how a mount *looks*
  // has to survive them and match byte for byte.
  const block = (source, mounts) => source
    .slice(source.indexOf("<h2>LoRA 挂载</h2>"), source.indexOf("</div>}", source.indexOf("<h2>LoRA 挂载</h2>")))
    .replace(new RegExp(mounts, "g"), "MOUNTS")
    .replace(/onClick=\{(?:onOpenLoraManager|\(\) => \{? ?set[^}]*\}?)\}/g, "onClick={OPEN}")
    .replace(/\s+/g, " ")
    .trim();
  assert.equal(block(page, "engine\\.loras"), block(app, "loras"));
  assert.match(page, /className="section-heading lora-title"/);
  assert.match(page, /className="lora-summary-table"/);
});

test("the drop stage stays keyboard reachable instead of being a clickable div", async () => {
  const page = await readSource("src/ImageToImagePage.jsx");
  const stage = page.slice(page.indexOf("i2i-compare-figure i2i-dropzone"), page.indexOf("i2i-compare-foot"));
  assert.match(stage, /<button type="button" className="i2i-dropzone-hit"/);
  assert.match(stage, /aria-label="选择来源图片"/);
});

test("the shared job state is reset through one path for both pages", async () => {
  // Two independent copies of the run state would let a page report progress for a job it did not
  // start; the inference service runs one job at a time.
  const app = await readSource("src/App.jsx");
  assert.match(app, /const beginGenerationRun = \(\{ totalSteps, batches, totalImages \}\) => \{/);
  assert.equal((app.match(/beginGenerationRun\(\{/g) || []).length, 2);
  assert.match(app, /const generateFromImage = async \(\) => \{/);
});

test("a gallery card records this page's parameters, not the text-to-image composer's", async () => {
  // Without an explicit record the dialog falls back to the workspace snapshot, which would put the
  // other page's prompt, steps and denoise on a picture they had nothing to do with.
  const app = await readSource("src/App.jsx");
  const run = app.slice(app.indexOf("const generateFromImage = async"), app.indexOf("const releaseLoadedModel"));
  assert.match(run, /setGeneratedSettings\(JSON\.parse\(JSON\.stringify\(\{/);
  assert.match(run, /page: "image"/);
  for (const field of ["positive", "negative", "steps", "cfg", "denoise", "sampler", "scheduler"]) {
    assert.match(run, new RegExp(`${field}: settings\\.${field}`), `${field} must come from this page`);
  }
  for (const stage of ["hires", "adetailer", "rtx"]) assert.match(run, new RegExp(`${stage}: (?:settings\\.${stage}|\\{ \\.\\.\\.settings\\.${stage})`), `${stage} must be recorded from i2i settings`);
});

test("the run shortcut starts the run of the page that is on screen", async () => {
  // Ctrl/Cmd+Enter used to call `generate()` unconditionally; on 图生图 that would have launched a
  // text-to-image run with the other page's prompt.
  const app = await readSource("src/App.jsx");
  const shortcut = app.slice(app.indexOf('} else if (event.key === "Enter") {'), app.indexOf('const releaseViewerNudge'));
  assert.match(shortcut, /activePage === "image"\) void generateFromImage\(\)/);
  assert.match(shortcut, /activePage === "generate"\) void generate\(\)/);
});

test("a fixed Hires seed survives being chosen and being cleared mid-edit", () => {
  // `normalizeHiresSeed` collapses "fixed with no number" to `inherit`. This normaliser runs on
  // every render, so applying that repair here made 固定 Hires Seed a dead control: the mode snapped
  // back before a digit could be typed, and again the moment the field was cleared.
  const chosen = normalizeImageToImageSettings({ positive: "x", hires: { enabled: true, model: "m", seedMode: "fixed", seed: "" } });
  assert.equal(chosen.hires.seedMode, "fixed");
  assert.equal(chosen.hires.seed, "");
  assert.equal(normalizeImageToImageSettings({ hires: { seedMode: "fixed", seed: "42" } }).hires.seed, "42");
  assert.equal(normalizeImageToImageSettings({ hires: { seedMode: "spiral" } }).hires.seedMode, "inherit");
  // The unfinished edit is what the run blocker is for — the same guard the text-to-image page
  // carries. `generationHiresSeedSettings` would otherwise submit a silent 0, which is a legal seed
  // and therefore worse than a refusal: the run would succeed with a number nobody chose.
  assert.match(imageToImageBlockers({
    source: source(512, 512), settings: chosen, postprocess: { hiresReady: true },
  }), /固定 Hires Seed/);
  const body = imageToImageRequestBody({ engine: "SD", checkpoint: "c", source: source(512, 512), settings: chosen, seed: "1" });
  assert.match(body.hires.seed, /^[0-9]+$/, "whatever is sent is still a legal unsigned seed");
});

test("the page refuses a post-processing stage the run would refuse", () => {
  const enabled = normalizeImageToImageSettings({
    positive: "x",
    hires: { enabled: true, model: "m" },
  });
  const ready = { source: source(512, 512), settings: enabled, postprocess: { hiresReady: true } };
  assert.equal(imageToImageBlockers(ready), "");
  // The server answers 422 for any stage in ultra-low memory mode, so the page says so instead of
  // submitting into it.
  assert.match(imageToImageBlockers({ ...ready, postprocess: { ...ready.postprocess, ultraLow: true } }), /极限省存/);
  assert.match(imageToImageBlockers({ ...ready, postprocess: { hiresReady: false, hiresReason: "超分运行环境尚未配置" } }), /超分运行环境/);
  assert.match(imageToImageBlockers({
    source: source(512, 512),
    settings: normalizeImageToImageSettings({ positive: "x", rtx: { enabled: true, scale: 4 } }),
    postprocess: { rtxReady: false, rtxReason: "RTX VSR 运行时不可用" },
  }), /RTX VSR/);
});

test("stage capability is resolved once, by the page that owns the asymmetry", async () => {
  // Anima must *declare* each stage while SD/iL only have to not deny it. Reading `features` raw on
  // the image page would be a second copy of that rule, free to drift from `engineAllowsHires`.
  const app = await readSource("src/App.jsx");
  assert.match(app, /features: \{ hires: engineAllowsHires, adetailer: engineAllowsADetailer, rtx: engineAllowsRtx \}/);
  assert.match(app, /ultraLow: ultraLowMode/);
  const page = await readSource("src/ImageToImagePage.jsx");
  assert.match(page, /const stageLocked = \(stage\) => running \|\| postprocess\.ultraLow === true \|\| engine\.features\?\.\[stage\] === false;/);
  assert.match(page, /const stageReason = \(stage\)/, "a switch that cannot move says why");
});

test("the mode sits with the button it changes, and hides the controls it makes meaningless", async () => {
  const page = await readSource("src/ImageToImagePage.jsx");
  // It decides whether the run samples at all, so it belongs to the run bar rather than to the
  // numbered parameter stack on the rail — where it would also have renumbered every section.
  assert.match(page, /<div className="i2i-run-bar">[\s\S]{0,600}?className="i2i-mode-switch"/);
  assert.match(page, /const postprocessOnly = config\.mode === "postprocess";/);
  // Strength drives a base pass that post-processing does not run, and the canvas is the picture's
  // own. Both are left out rather than shown as controls with no effect.
  assert.match(page, /\{!postprocessOnly && <div className="i2i-denoise">/);
  assert.match(page, /\{postprocessOnly \? <>[\s\S]{0,900}?i2i-canvas-readout/);
  const css = await readSource("src/styles.css");
  assert.match(css, /^\.i2i-mode-switch \{[^}]*display: grid;/m);
  assert.match(css, /html\[data-theme-mode="light"\] \.i2i-mode-switch button \{/,
    "both themes paint it, like every other control on this page");
});

test("the vocabularies the backend validates against are the ones the page offers", () => {
  assert.deepEqual(RESIZE_MODES.map((mode) => mode.id), ["cover", "contain", "stretch"]);
  assert.deepEqual(SIZE_MODES.map((mode) => mode.id), ["source", "scale", "custom"]);
  assert.deepEqual(IMAGE_TO_IMAGE_MODES.map((mode) => mode.id), ["transform", "postprocess"]);
});

test("post-processing mode keeps the picture at its own size instead of choosing a canvas", () => {
  const postprocess = normalizeImageToImageSettings({ mode: "postprocess", sizeMode: "custom", width: 512, height: 512 });
  // 1080 is not a multiple of 64 and neither edge is negotiable: nothing resamples the source, so
  // snapping the canvas here would only mispredict what the first stage receives.
  assert.deepEqual(outputSize(source(1920, 1080), postprocess), { width: 1920, height: 1080 });
  // Same settings without the mode: the canvas rules are back and the source is fitted onto one.
  assert.deepEqual(outputSize(source(1920, 1080), { ...postprocess, mode: "transform" }), { width: 512, height: 512 });
  // No picture yet — the readout has to fall back to something rather than crash.
  assert.deepEqual(outputSize(null, postprocess), { width: 512, height: 512 });
});

test("an unusable mode falls back rather than reaching the request body", () => {
  assert.equal(normalizeImageToImageSettings({}).mode, "transform");
  for (const value of ["upscale", "", null, 7, ["postprocess"]]) {
    assert.equal(normalizeImageToImageSettings({ mode: value }).mode, "transform");
  }
  assert.equal(normalizeImageToImageSettings({ mode: "postprocess" }).mode, "postprocess");
});

test("the request body tells the run which of the two it is", () => {
  const settings = { positive: "x", mode: "postprocess", rtx: { enabled: true, scale: 2 } };
  const body = imageToImageRequestBody({ engine: "SD", checkpoint: "c", source: source(1920, 1080), settings, seed: "1" });
  assert.equal(body.postprocess_only, true);
  // The declared canvas is the source, because that is what the backend admits memory against.
  assert.deepEqual([body.width, body.height], [1920, 1080]);
  assert.equal(body.source_image.enabled, true);
  const transform = imageToImageRequestBody({ engine: "SD", checkpoint: "c", source: source(1920, 1080), settings: { ...settings, mode: "transform" }, seed: "1" });
  assert.equal(transform.postprocess_only, false);
  assert.deepEqual([transform.width, transform.height], [1920, 1088]);
});

test("post-processing refuses a run with nothing to do, and a picture the stages cannot take", () => {
  const withStage = normalizeImageToImageSettings({ mode: "postprocess", positive: "x", rtx: { enabled: true, scale: 2 } });
  assert.equal(imageToImageBlockers({ source: source(1024, 1024), settings: withStage, postprocess: { rtxReady: true } }), "");
  // Every stage off is a run that would load a model, sample nothing and save its own input.
  assert.match(imageToImageBlockers({
    source: source(1024, 1024),
    settings: normalizeImageToImageSettings({ mode: "postprocess", positive: "x" }),
  }), /至少启用/);
  // The same settings in transform mode are a perfectly ordinary run.
  assert.equal(imageToImageBlockers({
    source: source(1024, 1024),
    settings: normalizeImageToImageSettings({ mode: "transform", positive: "x" }),
  }), "");
  assert.match(imageToImageBlockers({ source: source(9000, 100), settings: withStage }), /8192 边/);
  assert.match(imageToImageBlockers({ source: source(32, 32), settings: withStage }), /至少 64 像素/);
});

test("the source envelope is reported the same way the run would report it", () => {
  assert.equal(postprocessSourceIssue(source(64, 64)), "");
  assert.equal(postprocessSourceIssue(source(8192, 4096)), "");
  assert.match(postprocessSourceIssue(source(8256, 64)), /8192 边/);
  assert.match(postprocessSourceIssue(source(8192, 8192)), /32MP/);
  assert.match(postprocessSourceIssue(source(63, 512)), /至少 64 像素/);
  // Nothing to judge yet is not a problem to report.
  assert.equal(postprocessSourceIssue(null), "");
});
