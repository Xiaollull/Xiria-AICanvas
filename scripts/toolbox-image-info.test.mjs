import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const readSource = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

const app = await readSource("src/App.jsx");
const toolbox = await readSource("src/ToolboxPage.jsx");
const reader = await readSource("src/ImageInfoReader.jsx");
const details = await readSource("src/ImageInfoDetails.jsx");
const fields = await readSource("src/ImageInfoFields.jsx");
const explorer = await readSource("src/metadata-explorer.js");
const viteConfig = await readSource("vite.config.js");
const styles = await readSource("src/styles.css");

function functionBody(source, signature) {
  const start = source.indexOf(signature);
  assert.notEqual(start, -1, `${signature} is gone from vite.config.js`);
  let depth = 0;
  let index = source.indexOf("{", start);
  const open = index;
  for (; index < source.length; index += 1) {
    if (source[index] === "{") depth += 1;
    else if (source[index] === "}" && (depth -= 1) === 0) return source.slice(open, index + 1);
  }
  throw new Error(`${signature} body is unbalanced`);
}

test("the toolbox replaced the downloader tab and lands on its first tool", () => {
  assert.match(app, /activePage === "toolbox"/);
  assert.match(app, /<Wrench size=\{15\} \/>工具箱/);
  // The downloader is no longer reachable as a top-level tab.
  assert.doesNotMatch(app, /activePage === "downloader"/);
  assert.doesNotMatch(app, /模型下载器<\/button>/);

  // Opening the toolbox opens a tool, not a launcher screen.
  assert.match(toolbox, /useState\(TOOLBOX_TOOLS\[0\]\.id\)/);
  assert.match(toolbox, /\{ id: "downloader"/);
  assert.match(toolbox, /\{ id: "image-info"/);
  assert.ok(toolbox.indexOf('id: "downloader"') < toolbox.indexOf('id: "image-info"'), "the downloader stays the default landing tool");
  assert.match(toolbox, /className="toolbox-rail-head"/);
  assert.match(toolbox, /className="toolbox-rail-foot"/);
  assert.match(toolbox, /className="toolbox-tool-icon"/);
});

test("each tool is its own chunk, so opening the toolbox loads one of them", () => {
  assert.match(toolbox, /const ModelDownloader = lazy\(\(\) => import\("\.\/ModelDownloader"\)\)/);
  assert.match(toolbox, /const ImageInfoReader = lazy\(\(\) => import\("\.\/ImageInfoReader"\)\)/);
  assert.match(app, /const ToolboxPage = lazy\(\(\) => import\("\.\/ToolboxPage"\)\)/);
  // App must not import a tool directly or the split is defeated.
  assert.doesNotMatch(app, /import\("\.\/ModelDownloader"\)/);
  assert.doesNotMatch(app, /import\("\.\/ImageInfoReader"\)/);

  // The rail and the tool own separate scrollports.
  assert.match(styles, /\.toolbox-page \{[^}]*overflow: hidden/);
  assert.match(styles, /\.toolbox-rail \{[^}]*overflow-y: auto/);
});

test("the downloader keeps working through the toolbox's refresh callback", () => {
  assert.match(toolbox, /<ModelDownloader onDownloaded=\{onDownloaded\} \/>/);
  assert.match(app, /<ToolboxPage onApplyImageParameters=\{applyImageInfoParameters\} onDownloaded=\{\(\{ kind, engine, targets: completedTargets \}\) =>/);
});

test("the reader shows the picture beside the facts and offers both input modes", () => {
  assert.match(reader, /className="image-info-body"/);
  assert.match(reader, /className=\{`image-info-canvas \$\{!imageUrl \? "empty" : ""\} \$\{dragActive \? "drag-active" : ""\}`\}/);
  assert.match(reader, /className="image-info-panel"/);
  // Left picture, right facts.
  assert.ok(reader.indexOf('className="image-info-canvas"') < reader.indexOf('className="image-info-panel"'));
  assert.match(styles, /\.image-info-body \{[^}]*grid-template-columns: minmax\(0, 1fr\) minmax\(360px, 480px\)/);

  // Single upload and a directory for batch.
  assert.match(reader, /type="file"/);
  assert.match(reader, /const scanDirectory = async/);
  assert.match(reader, /\/api\/image-info\/scan/);
  assert.match(reader, /\/api\/image-info\/read/);
});

test("a tall picture scales to fit instead of being cropped by the frame", () => {
  // The row must be declared. An implicit `auto` row sizes to the picture's
  // intrinsic height, which pushed a portrait image past the body it sits in
  // and let `overflow: hidden` cut the bottom off.
  assert.match(styles, /\.image-info-body \{[^}]*grid-template-rows: minmax\(0, 1fr\)/);
  // A bare `1fr` would keep an automatic min-content floor — for a replaced
  // element that is its intrinsic height, i.e. the same overflow again.
  assert.doesNotMatch(styles, /\.image-info-body \{[^}]*grid-template-rows: 1fr/);
  // Same class of bug one level up: no tool's content may stretch the stage
  // past the viewport.
  assert.match(styles, /\.toolbox-page \{[^}]*grid-template-rows: minmax\(0, 1fr\)/);

  // The whole chain has to stay definite or the percentages below go indefinite.
  assert.match(styles, /\.toolbox-page \{[^}]*height: calc\(100dvh - 62px\)/);
  assert.match(styles, /\.toolbox-stage > \* \{ height: 100%; \}/);
  assert.match(styles, /\.image-info-tool \{[^}]*height: 100%/);
  assert.match(styles, /\.image-info-body \{[^}]*min-height: 0/);

  // The display box owns a fixed ratio and is sized to the largest such box
  // that fits the frame on BOTH axes, so the picture never decides how tall
  // this area is.
  const stage = styles.match(/\.image-info-image-stage \{[^}]*\}/)[0];
  assert.match(stage, /aspect-ratio: 4 \/ 3/);
  assert.match(stage, /width: min\(100cqw, calc\(100cqh \* 4 \/ 3\)\)/);
  // `container-type: size` supplies those cq units and, just as importantly,
  // contains the frame's size so its contents cannot inflate it.
  assert.match(styles, /\.image-info-canvas \{[^}]*container-type: size/);
  assert.match(styles, /\.image-info-canvas \{[^}]*height: 100%/);

  // Absolutely positioned: the picture contributes nothing to any ancestor's
  // height, which is the loop that cropped it in the first place.
  const image = styles.match(/\.image-info-image-stage > img \{[^}]*\}/)[0];
  assert.match(image, /position: absolute/);
  assert.match(image, /inset: 0/);
  assert.match(image, /object-fit: contain/);
  assert.doesNotMatch(image, /object-fit: cover/);
  // Exactly one rule decides how the picture is presented.
  assert.doesNotMatch(styles, /\.image-info-canvas img \{/);
});

test("batch navigation is manual and bounded", () => {
  assert.match(reader, /const goTo = \(nextIndex\) => \{/);
  assert.match(reader, /Math\.min\(files\.length - 1, Math\.max\(0, nextIndex\)\)/);
  assert.match(reader, /上一张/);
  assert.match(reader, /下一张/);
  assert.match(reader, /event\.key === "ArrowLeft"/);
  assert.match(reader, /event\.key === "ArrowRight"/);
  // A stale in-flight read must not overwrite the picture the user moved to.
  assert.match(reader, /const token = \+\+requestRef\.current/);
  assert.match(reader, /token === requestRef\.current/);
});

test("a non-original image says so, and a non-PNG says something different", () => {
  // The requested wording for an image that carries nothing.
  assert.match(reader, /<strong>无元数据<\/strong>/);
  assert.match(reader, /info\.status === "empty"/);
  // Reporting "no metadata" for a JPEG would imply it was checked and found
  // bare, which is not what happened.
  assert.match(reader, /info\.status === "unsupported"/);
  assert.match(reader, /<strong>非 PNG 文件<\/strong>/);
  assert.match(functionBody(viteConfig, "async function readImageInfoBuffer("), /status: "unsupported"/);
});

test("the right panel carries every field the tool promises", () => {
  for (const field of ["来源", "引擎", "底模", "正向提示词", "负向提示词", "生成参数", "LoRA", "功能标记"]) {
    assert.ok(reader.includes(field), `${field} is missing from the info panel`);
  }
  // Parameters are handed to the shared module renderer; the rest map inline.
  assert.match(reader, /<ParameterModules parameters=\{info\.parameters\} \/>/);
  assert.match(reader, /info\.loras\.map/);
  assert.match(reader, /info\.flags\.map/);
});

test("a model found nowhere is red, warned about, and not signalled by colour alone", () => {
  // The verdict is rendered once, in the piece the panel and the dialog share.
  assert.match(fields, /match\.status === "missing"/);
  assert.match(fields, /className=\{`model-name-missing/);
  assert.match(fields, /本地模型目录与共享目录中都没有这个模型/);
  assert.match(reader, /className="info-missing-warning"/);
  assert.match(reader, /在本地模型目录和共享目录中都找不到/);
  assert.match(styles, /\.model-name-missing \{[^}]*color: #ff6b5a/);
  // The icon and the warning line repeat the verdict for anyone who cannot
  // separate these hues.
  assert.match(fields, /<AlertTriangle size=\{12\} \/>\{name\}/);
  // Both surfaces import it rather than keeping a copy that can drift.
  assert.match(reader, /import \{[^}]*\bModelName\b[^}]*\} from "\.\/ImageInfoFields"/);
  assert.match(details, /import \{[^}]*\bModelName\b[^}]*\} from "\.\/ImageInfoFields"/);
  assert.doesNotMatch(reader, /function ModelName\(/);
  assert.doesNotMatch(details, /function ModelName\(/);
});

test("a settings line long enough to be a wall is grouped into its modules", () => {
  // A1111 writes one flat line, so a run with four ADetailer passes arrives as
  // thirty-odd cells that are nine tenths repeated prefix.
  assert.match(explorer, /export function groupParameters/);
  assert.match(fields, /export function ParameterModules/);
  // Rendered from the shared component on both surfaces, never as a raw grid.
  assert.match(reader, /<ParameterModules parameters=\{info\.parameters\} \/>/);
  assert.match(details, /<ParameterModules parameters=\{info\.parameters\} \/>/);
  assert.doesNotMatch(reader, /info\.parameters\.map/);
  assert.doesNotMatch(details, /info\.parameters\.map/);
  // A record with nothing to group keeps the plain grid: one heading over one
  // grid is a line that says nothing.
  assert.match(fields, /if \(groups\.length === 1\) return <ParameterCells/);
  assert.match(styles, /\.info-parameter-modules > section \{/);
});

test("matching consults the project directories before the shared ones", () => {
  const body = functionBody(viteConfig, "async function buildModelCatalogs(");
  const local = body.indexOf('catalogEntry("local"');
  const shared = body.indexOf('catalogEntry("shared"');
  assert.ok(local > 0 && shared > local, "local roots must be collected before shared ones");
  assert.match(body, /getConfiguredDirectory\(engine, "checkpoints"\)/);
  assert.match(body, /findSharedModels\(root, "checkpoints"/);
  assert.match(body, /if \(!root\.enabled\) continue/);
  // A broken engine root or shared config costs that half of matching only.
  assert.ok(body.match(/catch \{/g).length >= 3);

  const resolve = functionBody(viteConfig, "async function resolveImageInfoModels(");
  assert.match(resolve, /checkpointMatch: info\.checkpoint \? matchModelName/);
  assert.match(resolve, /loras: \(info\.loras \|\| \[\]\)\.map/);
});

test("a typed directory is read-only, bounded, and cannot be escaped", () => {
  const directory = functionBody(viteConfig, "async function resolveImageDirectory(");
  assert.match(directory, /path\.isAbsolute/);
  assert.match(directory, /realpath/);
  assert.match(directory, /磁盘根目录/);

  const file = functionBody(viteConfig, "async function resolveImageFile(");
  // A name is one entry in the listed folder; a separator means it is leaving.
  assert.match(file, /path\.basename\(filename\) !== filename/);
  assert.match(file, /isPathInside\(root, resolved\)/);
  assert.match(file, /imageInfoExtensions\.has/);

  assert.match(viteConfig, /IMAGE_INFO_MAX_DIRECTORY_FILES = \d+/);
  assert.match(viteConfig, /IMAGE_INFO_MAX_UPLOAD_BYTES = \d+/);
  // The scan reads one folder, not a whole output tree.
  assert.doesNotMatch(functionBody(viteConfig, "async function imageInfoApi("), /recursive: true/);

  const apiPaths = viteConfig.match(/const apiPaths = \[[^\]]+\]/)[0];
  for (const route of ["scan", "read", "preview"]) {
    assert.ok(apiPaths.includes(`"/api/image-info/${route}"`), `/api/image-info/${route} is not routed`);
  }
});

test("an uploaded picture is displayed from the browser's own copy", () => {
  // Only the bytes needed for parsing are posted; the preview never round-trips.
  assert.match(reader, /URL\.createObjectURL\(file\)/);
  assert.match(reader, /URL\.revokeObjectURL/);
  assert.match(reader, /const releaseObjectUrl = \(\) => \{/);
  assert.match(reader, /useEffect\(\(\) => releaseObjectUrl, \[\]\)/);
  // The stage owns the box and the box owns its ratio; the image fills that box
  // with `contain`, so its intrinsic ratio is preserved and never cropped.
  // (It no longer sizes itself with auto + max-*: letting the picture have any
  // say in layout is what let a tall one overflow the frame.)
  assert.match(reader, /className="image-info-image-stage"/);
  assert.match(styles, /\.image-info-image-stage > img \{[^}]*position: absolute[^}]*inset: 0[^}]*object-fit: contain/);
});

test("the preview frame accepts image drops and clipboard screenshots", () => {
  assert.match(reader, /const handleDragEnter = \(event\) => \{/);
  assert.match(reader, /const handleDragLeave = \(event\) => \{/);
  assert.match(reader, /const handleDrop = \(event\) => \{/);
  assert.match(reader, /const handlePaste = \(event\) => \{/);
  assert.match(reader, /event\.clipboardData\?\.items/);
  assert.match(reader, /className=\{`image-info-canvas \$\{!imageUrl \? "empty" : ""\} \$\{dragActive \? "drag-active" : ""\}`\}/);
  assert.match(reader, /aria-label="图片预览区，支持拖拽或粘贴图片"/);
  assert.match(reader, /image-info-drop-overlay/);
  assert.match(reader, /image-info-empty-dropzone/);
  assert.match(reader, /拖入、粘贴或点击选择图片/);
  assert.match(reader, /uploadInputRef\.current\?\.click/);
  assert.match(styles, /\.image-info-canvas\.empty \{/);
  assert.match(styles, /\.image-info-canvas\.drag-active \{/);
  assert.match(styles, /\.image-info-empty-dropzone \{/);
  assert.match(styles, /\.image-info-drop-overlay \{/);
});

test("the details dialog presents the complete record in sections", () => {
  // The trigger lives on the facts panel and only lights up once a file has
  // actually been read.
  assert.match(reader, /const \[detailsOpen, setDetailsOpen\] = useState\(false\)/);
  assert.match(reader, /className="image-info-panel-head"/);
  assert.match(reader, /<Maximize2 size=\{12\} \/>详细信息/);
  assert.match(reader, /disabled=\{!info\}/);
  assert.match(reader, /<ImageInfoDetails info=\{info\} onClose=\{\(\) => setDetailsOpen\(false\)\} \/>/);

  // One fixed dialog with a real dialog role, an Escape path and a
  // backdrop-click path.
  assert.match(details, /className="image-info-details-backdrop"/);
  assert.match(details, /role="dialog" aria-modal="true" aria-labelledby="image-info-details-title"/);
  assert.match(details, /event\.key === "Escape"/);
  assert.match(details, /event\.target === event\.currentTarget/);

  // Complete block coverage: identity, model, prompts, parameters, LoRA,
  // flags and the raw PNG text chunks the summary never shows.
  for (const section of ["文件与来源", "底模", "提示词", "生成参数", "LoRA", "功能标记", "原始数据块"]) {
    assert.ok(details.includes(section), `${section} section is missing from the details dialog`);
  }

  assert.match(styles, /\.image-info-details-backdrop \{/);
  assert.match(styles, /\.image-info-details-dialog \{/);
  assert.match(styles, /\.image-info-details-section \{/);
  // Header, tab strip and one scrollport, all declared. An implicit `auto` row
  // here sizes to a 67-node graph and scrolls the page instead of the dialog.
  assert.match(styles, /\.image-info-details-dialog \{[^}]*grid-template-rows: auto auto minmax\(0, 1fr\)/);
});

test("an expanded record is cut into blocks rather than printed as one payload", () => {
  // The three views of the same record: the summary, one card per graph node,
  // and the chunks themselves.
  for (const tab of ["摘要", "工作流节点", "原始数据块"]) {
    assert.ok(details.includes(tab), `${tab} tab is missing from the details dialog`);
  }
  assert.match(details, /describeChunks\(info\.raw\)/);
  assert.match(details, /buildGraphModules\(graphChunk\.graph, comfyNodeRoles\(graphChunk\.graph\)\)/);

  // A ComfyUI export carries a 40 KB graph and a 120 KB workflow. Neither may
  // reach a <pre>: a graph becomes node cards, any other JSON becomes a fold
  // per `{}`, and only genuinely unstructured text is printed whole.
  assert.match(details, /chunk\.kind === "text" \? null : jsonBranch\(chunk\.graph \|\| chunk\.json\)/);
  assert.match(details, /branch\s*\?[\s\S]{0,400}<pre>\{chunk\.text\}<\/pre>/);
  assert.doesNotMatch(details, /<pre>\{String\(value\)\}<\/pre>/);
  // Children are described on open, so a collapsed branch costs one row.
  assert.match(details, /\{open && \(branch\.size/);

  // Every input is its own row, and a link is a jump rather than a printed id.
  assert.match(explorer, /export function buildGraphModules/);
  assert.match(details, /className="metadata-field-link"/);
  assert.match(details, /onJump\(field\.target\)/);
  assert.match(details, /setCategory\("all"\);\s*\n\s*setQuery\(""\);\s*\n\s*setFocusId\(id\)/);

  // Cards are a grid of blocks, not a column, and a long widget cannot decide
  // how tall one is.
  assert.match(styles, /\.metadata-node-grid \{[^}]*grid-template-columns: repeat\(auto-fill, minmax\(290px, 1fr\)\)/);
  assert.match(styles, /\.metadata-field-text p \{[^}]*max-height: 96px/);
  assert.match(styles, /\.metadata-json-children \{/);
  assert.match(styles, /\.image-info-details-tabs button\.on \{/);
});
