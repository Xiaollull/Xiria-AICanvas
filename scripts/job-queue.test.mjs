import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const readSource = (file) => readFile(new URL(`../${file}`, import.meta.url), "utf8");
const sourceBetween = (source, start, end) => {
  const from = source.indexOf(start);
  assert.notEqual(from, -1, `missing anchor: ${start}`);
  const to = source.indexOf(end, from + start.length);
  assert.notEqual(to, -1, `missing anchor: ${end}`);
  return source.slice(from, to);
};

test("submitting no longer waits for the running job to finish", async () => {
  const app = await readSource("src/App.jsx");
  const generate = sourceBetween(app, "  const generate = async () => {", "  const generateFromImage = async () => {");
  const imageGenerate = sourceBetween(app, "  const generateFromImage = async () => {", "  const releaseLoadedModel = async () => {");

  // The guard is the request in flight, not a job in progress: a double click must send one job,
  // but a second deliberate submission has to be allowed to queue.
  assert.match(generate, /if \(submissionInFlight\.current \|\| generationDisabledReason\) return;/);
  assert.doesNotMatch(generate, /if \(generationLocked\.current \|\| generationDisabledReason\) return;/);

  // The panel is only cleared when it is free. Clearing it while a job runs would throw away the
  // progress view of a run that is still going.
  assert.match(generate, /if \(!watching\) \{[\s\S]{0,500}beginGenerationRun\(/);
  assert.match(generate, /const watching = queueBusy;/);
  assert.match(imageGenerate, /if \(submissionInFlight\.current \|\| !imageSource\) return;/);
  assert.match(imageGenerate, /const watching = queueBusy;/);
  assert.doesNotMatch(imageGenerate, /generationLocked\.current \|\| status === "running"/);
});

test("the submit control queues instead of going dead while a job runs", async () => {
  const app = await readSource("src/App.jsx");
  const button = sourceBetween(app, '<button className={`generate-button', "</button>");
  // Only a real blocker disables it now — a missing model or an unreachable service.
  assert.match(button, /disabled=\{Boolean\(generationDisabledReason\)\}/);
  assert.doesNotMatch(button, /disabled=\{status === "running"/);
  assert.match(button, /加入队列/);
  const blockers = sourceBetween(app, "  const generationDisabledReason =", "  const loaderProgress =");
  assert.doesNotMatch(blockers, /status === "running"/);
});

test("the queue is read from the service rather than assembled in the page", async () => {
  const app = await readSource("src/App.jsx");
  assert.match(app, /fetch\("\/api\/inference\/jobs", \{ cache: "no-store" \}\)/);
  // Polling depends on a derived flag. Depending on the array restarts the effect on every
  // refresh, and each restart polls immediately, which is a request loop rather than an interval.
  assert.match(app, /const queueOutstanding = queue\.some\(\(job\) => job\.active\);/);
  assert.match(app, /\}, \[queueOutstanding, queueOpen, refreshQueue\]\);/);
  assert.doesNotMatch(app, /\}, \[queue, queueOpen, generationJob, refreshQueue\]\);/);
});

test("the panel follows the job the service is working on", async () => {
  const app = await readSource("src/App.jsx");
  // When the watched job ends the next one is picked up, or the queue would drain unobserved.
  // An XYZ experiment is watched by its own workspace, not by this preview panel.
  assert.match(app, /const next = queue\.find\(\(job\) => job\.active && job\.id !== generationJob && job\.kind !== "xyz_plot"\);/);
  assert.match(app, /setStagePreviews\(\[\]\);/);
});

test("the queue list names every state a job can be in", async () => {
  const app = await readSource("src/App.jsx");
  const table = sourceBetween(app, "const QUEUE_STATUS_TEXT = {", "};");
  for (const status of ["queued", "running", "paused", "cancelling", "complete", "error", "cancelled"]) {
    assert.match(table, new RegExp(`${status}:`), `queue list must name ${status}`);
  }
  // A finished job's row identifies the image it produced and how long it took.
  assert.match(app, /function queueDuration\(job\) \{/);
  assert.match(app, /job\.output\?\.output_name/);
  assert.match(app, /const selectable = \(finished && Boolean\(job\.output\?\.image_url\)\) \|\| live;/);
  assert.match(app, /if \(!selectable\) return;\s+onSelect\(job\);/);
  assert.match(app, /onSelect=\{selectQueueJob\}/);
  assert.match(app, /setStagePreviews\(job\.stage_preview \? \[job\.stage_preview\] : \[\]\);/);
});

test("model and render controls remain editable while an earlier request runs", async () => {
  const app = await readSource("src/App.jsx");
  assert.match(app, /if \(nextModel === model \|\| modelSwitching\) return;/);
  assert.match(app, /if \(nextCheckpoint === checkpoint \|\| modelSwitching\) return;/);
  assert.match(app, /const adetailerLocked = modelSwitching;/);
  assert.match(app, /const hiresControlsLocked = modelSwitching;/);
  assert.match(app, /const releaseIdleModel = modelIdentityChanged && !queueBusy;/);
  assert.doesNotMatch(app, /if \(status === "running" \|\| modelSwitching \|\| modelLoading\) throw new Error/);
  assert.doesNotMatch(app, /disabled=\{!item\.ready \|\| status === "running" \|\| modelSwitching\}/);
});

test("a finished stage is what the panel shows while the next stage runs", async () => {
  const [app, shared] = await Promise.all([readSource("src/App.jsx"), readSource("src/StagePreviews.jsx")]);
  assert.match(app, /const latestStagePreview = stagePreviews\.length \? stagePreviews\[stagePreviews\.length - 1\] : null;/);
  // The markup is shared with the image-to-image page now, but the rule is unchanged: the stage
  // picture shows while the next stage runs, and it is not a ternary -- it used to fall back to the
  // latent stream, and that stream has been withdrawn.
  assert.match(app, /<StagePreviewLive stage=\{latestStagePreview\} stageLabel=\{stageLabel\} onEnlarge=\{setEnlargedStage\} held=\{heldPreview\} \/>/);
  assert.match(shared, /\{stage && <button\s*\n\s*type="button"\s*\n\s*className="stage-preview-image"/);
  assert.doesNotMatch(app, /livePreview/);
  assert.doesNotMatch(shared, /livePreview/);
  // Every stage stays reachable and can be opened larger; the dialog itself stays in App.
  assert.match(shared, /className="stage-strip-item"/);
  assert.match(app, /<StageStrip stages=\{stagePreviews\} onEnlarge=\{setEnlargedStage\} \/>/);
  assert.match(app, /className="stage-enlarge"/);
});

test("stage pictures come from the job record, not from a second guess", async () => {
  const app = await readSource("src/App.jsx");
  assert.match(app, /if \(Array\.isArray\(job\.stage_previews\)\) \{\s*\n\s*setStagePreviews\(job\.stage_previews\);/);
});

test("the queue panel exposes its visual and interaction states", async () => {
  const [app, css] = await Promise.all([readSource("src/App.jsx"), readSource("src/styles.css")]);
  assert.match(app, /className={`queue-item \$\{status\} \$\{job\.id === watching \? "watching" : ""\}`}/);
  assert.match(app, /className="queue-progress-line"/);
  assert.match(app, /className="queue-thumb"/);
  assert.match(css, /\.queue-dropdown \{[^}]*border-radius: 12px;[^}]*box-shadow:/);
  assert.match(css, /\.queue-item\.watching \{[^}]*border-color: rgb\(var\(--accent-rgb\)/);
  assert.match(css, /\.queue-item-main:focus-visible/);
  assert.match(css, /@media \(prefers-reduced-motion: reduce\).*\.queue-dot\.running/s);
});

test("the run information stays attached to the watched job while the draft model changes", async () => {
  const app = await readSource("src/App.jsx");
  assert.match(app, /function generationJobIdentity\(job\) \{/);
  assert.match(app, /setGenerationRunIdentity\(generationJobIdentity\(job\)\);/);
  assert.match(app, /const displayedRunEngine = displayedRunIdentity\?\.engine \|\| model;/);
  assert.match(app, /<span>当前任务模型<\/span>/);
});

test("gallery additions use the completed job snapshot instead of the current draft", async () => {
  const app = await readSource("src/App.jsx");
  assert.match(app, /gallery_settings: gallerySettings/);
  assert.match(app, /gallerySettings,/);
  assert.match(app, /const loadQueueJobDetail = async/);
  assert.match(app, /"X-XirAI-Gallery-Access": galleryAccessToken/);
  assert.match(app, /sessionStorage\.setItem\(`\$\{GALLERY_JOB_ACCESS_STORAGE_PREFIX\}\$\{jobId\}`, token\)/);
  assert.match(app, /const gallerySettings = galleryJobSnapshotPayload\(submittedSettings\);/);
  assert.match(app, /const canAddGeneratedToGallery = Boolean\(generatedSettings\)/);
  assert.match(app, /settings=\{galleryAddSource \? galleryCardSettings\(galleryAddSource\.settings, \{ record: true \}\) : generatedSettings \? galleryCardSettings\(generatedSettings, \{ record: true \}\) : null\}/);
});
