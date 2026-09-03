import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { collapseConsoleEntries, parseProgressLine, progressFigures, readProgressEntry } from "../src/console-progress.js";

const read = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");

// Lines copied from the backend formatter rather than retyped, so a change to either side breaks
// this file instead of silently leaving the drawer unable to read what the terminal prints.
const OPENING = "Batch 1/2 · Sampling 2 images           0%|                    |      0/30 [00:00<--:--, ?s/it]";
const MIDWAY = "Batch 1/2 · Sampling 2 images          66%|█████████████       |     20/30 [00:56<00:28, 2.81s/it]  job 01:08";
const DETAIL = "face_yolov8n · Inpainting 1/2          25%|█████               |      3/12 [00:07<00:21, 2.40s/it]  job 33/54, 02:39";
const TILE = "Hires.fix · Upscaling 2176 x 2816      50%|██████████          |       2/4 [00:25<00:25, 12.70s/tile]";
const LONG = "Sampling                               50%|██████████          |  900/1800 [1:02:05<1:02:05, 4.14s/it]";

const entry = (id, message, overrides = {}) => ({
  id, at: "2026-09-03T02:00:00.000Z", source: "inference", stream: "stdout", message, ...overrides,
});

test("a progress line yields the figures the drawer needs to redraw it", () => {
  assert.deepEqual(parseProgressLine(MIDWAY), {
    phase: "Batch 1/2 · Sampling 2 images",
    percent: 66,
    step: 20,
    total: 30,
    elapsed: "00:56",
    remaining: "00:28",
    rate: "2.81",
    unit: "it",
    jobStep: null,
    jobTotal: null,
    jobElapsed: "01:08",
  });
});

test("the opening line of a run reports no rate rather than a zero one", () => {
  const parsed = parseProgressLine(OPENING);
  assert.equal(parsed.step, 0);
  assert.equal(parsed.percent, 0);
  assert.equal(parsed.rate, null, "an unmeasured rate must not read as instant");
  assert.equal(parsed.remaining, "--:--");
});

test("the job-wide counter and clock are read when the line carries them", () => {
  const parsed = parseProgressLine(DETAIL);
  assert.equal(parsed.phase, "face_yolov8n · Inpainting 1/2");
  assert.deepEqual([parsed.jobStep, parsed.jobTotal, parsed.jobElapsed], [33, 54, "02:39"]);
});

test("tiled stages keep their own unit", () => {
  const parsed = parseProgressLine(TILE);
  assert.equal(parsed.unit, "tile");
  assert.equal(parsed.rate, "12.70");
  assert.equal(parsed.jobElapsed, null);
});

test("a run past an hour still parses", () => {
  const parsed = parseProgressLine(LONG);
  assert.deepEqual([parsed.elapsed, parsed.remaining], ["1:02:05", "1:02:05"]);
  assert.equal(parsed.step, 900);
});

test("ordinary log output is not mistaken for progress", () => {
  for (const line of [
    "",
    "Loading checkpoint anima_v5.safetensors",
    "INFO:     127.0.0.1:52413 - \"GET /api/inference/jobs/abc HTTP/1.1\" 200 OK",
    "Complete                              2 images · 54 steps · 03:01 elapsed · 3.36s/it average",
    "Downloading local runtime config from huggingface: black-forest-labs/FLUX.1-dev",
  ]) {
    assert.equal(parseProgressLine(line), null, line);
  }
});

test("the mounted-adapter block stays whole rather than being read as a bar", () => {
  // It arrives as one multi-line chunk on purpose — the adapters are one fact about the run — and
  // it carries no bar, so the drawer must render it as written instead of folding it into a row.
  const block = [
    "LoRA · 2 mounted                       0.75 · kazutake-hazano_v2_epoch28.safetensors · 145 MB",
    "                                       0.25 · ashima_v4.safetensors · 210 MB",
  ].join("\n");
  assert.equal(parseProgressLine(block.split("\n")[0]), null);
  assert.equal(readProgressEntry(entry(1, block)), null);
  const collapsed = collapseConsoleEntries([entry(1, MIDWAY), entry(2, block), entry(3, MIDWAY)]);
  assert.equal(collapsed.length, 3, "and it separates the runs on either side of it");
});

test("only the inference service's own stdout is read as progress", () => {
  assert.notEqual(readProgressEntry(entry(1, MIDWAY)), null);
  assert.equal(readProgressEntry(entry(1, MIDWAY, { source: "terminal" })), null);
  assert.equal(readProgressEntry(entry(1, MIDWAY, { stream: "stderr" })), null);
});

test("a chunk holding several progress lines collapses to its newest", () => {
  // The pipe coalesces writes whenever steps come faster than it drains.
  const parsed = readProgressEntry(entry(1, `${OPENING}\n${MIDWAY}\n`));
  assert.equal(parsed.step, 20);
});

test("a chunk that mixes progress with anything else is left whole", () => {
  // The warning is the part worth reading; folding it into a bar would hide it.
  assert.equal(readProgressEntry(entry(1, `${MIDWAY}\nxFormers is not available, falling back\n`)), null);
});

test("consecutive updates of one run occupy a single row", () => {
  const collapsed = collapseConsoleEntries([
    entry(1, "Loading checkpoint anima_v5.safetensors\n"),
    entry(2, OPENING),
    entry(3, MIDWAY),
    entry(4, MIDWAY.replace("20/30", "30/30")),
  ]);
  assert.equal(collapsed.length, 2);
  assert.equal(collapsed[1].progress.step, 30, "the row shows the newest update");
  assert.equal(collapsed[1].id, 2, "and keeps the id it was first drawn under, so it updates in place");
});

// A tile redraws in place, so its updates arrive opening with a carriage return and without a
// trailing newline — exactly the bytes the terminal needs to paint over the previous draw.
const tileRedraw = (id, tileIndex, step) => entry(
  id,
  `\rHires.fix · Tile ${tileIndex}/4                   ${String(Math.floor(100 * step / 15)).padStart(3)}%|██████████          | ${String(step).padStart(2)}/15 [00:24<00:28, 3.50s/it]`,
);

test("an in-place redraw is read despite the carriage return that carries it", () => {
  const parsed = parseProgressLine(tileRedraw(1, 2, 7).message);
  assert.equal(parsed.phase, "Hires.fix · Tile 2/4");
  assert.deepEqual([parsed.step, parsed.total], [7, 15]);
});

test("several redraws coalesced into one chunk collapse to the newest", () => {
  // They are separated only by the return that was meant to overwrite them, never by a newline.
  const chunk = entry(1, `${tileRedraw(1, 1, 6).message}${tileRedraw(1, 1, 7).message}`);
  assert.equal(readProgressEntry(chunk).step, 7);
});

test("each tile occupies one row, matching the line the terminal commits", () => {
  const entries = [];
  let id = 0;
  for (const tileIndex of [1, 2, 3, 4]) {
    for (const step of [5, 10, 15]) entries.push(tileRedraw((id += 1), tileIndex, step));
  }
  const collapsed = collapseConsoleEntries(entries);
  assert.equal(collapsed.length, 4, "four tiles, four rows");
  assert.deepEqual(collapsed.map((item) => item.progress.phase), [
    "Hires.fix · Tile 1/4", "Hires.fix · Tile 2/4", "Hires.fix · Tile 3/4", "Hires.fix · Tile 4/4",
  ]);
  assert.ok(collapsed.every((item) => item.progress.step === 15), "each row rests on its tile's last step");
});

test("a counter that restarts opens a new row even at the same size", () => {
  // Batch two of two: same stage, same step count, different run.
  const first = entry(1, MIDWAY.replace("20/30", "30/30").replace("66%", "100%"));
  const second = entry(2, MIDWAY.replace("20/30", " 2/30").replace("66%", "  6%"));
  assert.equal(collapseConsoleEntries([first, second]).length, 2);
});

test("each run gets its own row", () => {
  const collapsed = collapseConsoleEntries([entry(1, MIDWAY), entry(2, DETAIL), entry(3, TILE)]);
  assert.deepEqual(collapsed.map((item) => item.progress.phase), [
    "Batch 1/2 · Sampling 2 images",
    "face_yolov8n · Inpainting 1/2",
    "Hires.fix · Upscaling 2176 x 2816",
  ]);
});

test("a run interrupted by real output resumes on a new row", () => {
  // The interruption is the point: burying it under a bar that reaches back over it would lose
  // the ordering between what the service said and where the run was when it said it.
  const collapsed = collapseConsoleEntries([
    entry(1, OPENING),
    entry(2, "Model unloaded to system memory\n"),
    entry(3, MIDWAY),
  ]);
  assert.equal(collapsed.length, 3);
  assert.equal(collapsed[2].progress.step, 20);
});

test("entries that are not progress pass through untouched", () => {
  const plain = entry(1, "Starting inference service.\n", { stream: "system" });
  const collapsed = collapseConsoleEntries([plain]);
  assert.equal(collapsed[0], plain);
  assert.equal(collapsed[0].progress, undefined);
  assert.deepEqual(collapseConsoleEntries([]), []);
});

test("the figures spell out what the bar cannot", () => {
  assert.equal(
    progressFigures(parseProgressLine(MIDWAY)),
    "20/30 · 00:56 → 00:28 · 2.81s/it · 累计 01:08",
  );
  assert.equal(
    progressFigures(parseProgressLine(DETAIL)),
    "3/12 · 00:07 → 00:21 · 2.40s/it · 任务 33/54 · 累计 02:39",
  );
  // An unmeasured rate is shown as absent, not as zero seconds a step.
  assert.equal(progressFigures(parseProgressLine(OPENING)), "0/30 · 00:00 → --:-- · --s/it");
});

test("the backend is told to write UTF-8, because this end decodes it as UTF-8", async () => {
  // Piped, Python encodes stdout with the system code page — gbk on a Chinese Windows — while
  // both consoles read the chunk as UTF-8. Every block and every "·" then arrived as replacement
  // characters, and because a mangled block occupied two columns instead of one, the bar's closing
  // edge slid further right the fuller it got.
  const [vite, standalone] = await Promise.all([read("vite.config.js"), read("scripts/run-inference.mjs")]);
  assert.match(vite, /PYTHONIOENCODING: "utf-8"/);
  assert.match(standalone, /PYTHONIOENCODING \|\|= "utf-8"/);
});

test("a redraw reaches the terminal unprefixed and the drawer without the control character", async () => {
  const vite = await read("vite.config.js");
  // The terminal needs the carriage return to land at column zero, so that chunk is forwarded as
  // the backend wrote it rather than behind an "[inference] " label the bar would paint over.
  assert.match(vite, /text\.startsWith\("\\r"\) \? text : `\[inference\] \$\{text\}`/);
  // The drawer has no cursor to move, and a lone return inside a <pre> costs a blank line.
  assert.match(vite, /replace\(\/\\r\\n\?\/g, "\\n"\)/);
});

test("the drawer renders the parsed run rather than the raw ASCII bar", async () => {
  const [app, styles] = await Promise.all([read("src/App.jsx"), read("src/styles.css")]);
  assert.match(app, /from "\.\/console-progress\.js"/);
  // Collapsed as entries arrive, so progress cannot crowd real output out of the stored window.
  assert.match(app, /collapseConsoleEntries\(\[\.\.\.current, \.\.\.payload\.entries\]\)\.slice\(-1200\)/);
  assert.match(app, /className="console-progress-track"/);
  assert.match(app, /progressFigures\(entry\.progress\)/);
  assert.match(styles, /\.console-entry\.progress/);
  assert.match(styles, /\.console-progress-track/);
});
