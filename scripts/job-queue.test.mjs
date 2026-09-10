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

  // The guard is the request in flight, not a job in progress: a double click must send one job,
  // but a second deliberate submission has to be allowed to queue.
  assert.match(generate, /if \(submissionInFlight\.current \|\| generationDisabledReason\) return;/);
  assert.doesNotMatch(generate, /if \(generationLocked\.current \|\| generationDisabledReason\) return;/);

  // The panel is only cleared when it is free. Clearing it while a job runs would throw away the
  // progress view of a run that is still going.
  assert.match(generate, /if \(!watching\) \{\s*\n\s*beginGenerationRun\(/);
  assert.match(generate, /const watching = generationJob && status === "running";/);
});

test("the submit control queues instead of going dead while a job runs", async () => {
  const app = await readSource("src/App.jsx");
  const button = sourceBetween(app, '<button className={`generate-button', "</button>");
  // Only a real blocker disables it now — a missing model or an unreachable service.
  assert.match(button, /disabled=\{Boolean\(generationDisabledReason\)\}/);
  assert.doesNotMatch(button, /disabled=\{status === "running"/);
  assert.match(button, /加入队列/);
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
  assert.match(app, /const next = queue\.find\(\(job\) => job\.active && job\.id !== generationJob\);/);
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
  assert.match(app, /job\.outputs\?\.\[0\]\?\.output_name/);
});

test("a finished stage is what the panel shows while the next stage runs", async () => {
  const app = await readSource("src/App.jsx");
  assert.match(app, /const latestStagePreview = stagePreviews\.length \? stagePreviews\[stagePreviews\.length - 1\] : null;/);
  // The stage picture shows while the next stage runs. It is no longer a ternary: it used to fall
  // back to the latent stream, and that stream has been withdrawn.
  assert.match(app, /latestStagePreview\s*\n?\s*&& <button type="button" className="stage-preview-image"/);
  assert.doesNotMatch(app, /livePreview/);
  // Every stage stays reachable and can be opened larger.
  assert.match(app, /className="stage-strip-item"/);
  assert.match(app, /setEnlargedStage\(stage\)/);
  assert.match(app, /className="stage-enlarge"/);
});

test("stage pictures come from the job record, not from a second guess", async () => {
  const app = await readSource("src/App.jsx");
  assert.match(app, /if \(Array\.isArray\(job\.stage_previews\)\) \{\s*\n\s*setStagePreviews\(job\.stage_previews\);/);
});
