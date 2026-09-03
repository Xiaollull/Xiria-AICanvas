// The backend prints one whole line per progress update, because its stdout is a pipe feeding two
// consoles at once and a `\r`-redrawn bar survives neither. That is right for the terminal, where
// the lines scroll past, and wrong for the drawer, where a two-minute batch would stack thirty
// near-identical rows and push every real log line out of view.
//
// So the drawer reads those lines back. Consecutive updates belonging to the same run collapse
// into one row that rewrites itself, and the parsed figures let it draw an actual bar instead of
// the ASCII one — the same numbers the terminal shows, in the form a GUI can show them.

const CLOCK = "(\\d{1,2}:\\d{2}(?::\\d{2})?|--:--)";

// The bar is full blocks against spaces, and the phase column is space-padded to a fixed width, so
// the delimiting pipes are what tell the two apart — the bar is matched between them, never by
// its content alone.
const PROGRESS_LINE = new RegExp(
  `^(.+?)\\s{2,}(\\d{1,3})%\\|(\\u2588* *)\\|\\s*(\\d+)/(\\d+) ` +
  `\\[${CLOCK}<${CLOCK}, (\\?|\\d+(?:\\.\\d+)?)s/([a-z]+)\\]` +
  `(?:\\s{2}job (.+))?$`,
);

const JOB_STEPS = /^(\d+)\/(\d+)/;
const JOB_CLOCK = new RegExp(`(?:^|, )${CLOCK}$`);

export function parseProgressLine(line) {
  // A tile redraws itself in place, so its line arrives opening with a carriage return and without
  // a newline. The drawer has no cursor to move: it replaces the row instead, and the control
  // character is only in the way.
  const match = PROGRESS_LINE.exec(String(line ?? "").replace(/^[\r\n]+/, "").trimEnd());
  if (!match) return null;
  const [, phase, percent, , step, total, elapsed, remaining, rate, unit, job] = match;
  const jobSteps = job ? JOB_STEPS.exec(job) : null;
  const jobClock = job ? JOB_CLOCK.exec(job) : null;
  return {
    phase: phase.trim(),
    percent: Number(percent),
    step: Number(step),
    total: Number(total),
    elapsed,
    remaining,
    rate: rate === "?" ? null : rate,
    unit,
    jobStep: jobSteps ? Number(jobSteps[1]) : null,
    jobTotal: jobSteps ? Number(jobSteps[2]) : null,
    jobElapsed: jobClock ? jobClock[1] : null,
  };
}

// A stdout chunk is whatever the pipe delivered, so a single entry can hold several lines. Only an
// entry that is progress and nothing else may collapse; one that also carries a warning or a
// traceback stays whole, because the part that matters is the part that is not progress.
export function readProgressEntry(entry) {
  if (!entry || entry.source !== "inference" || entry.stream !== "stdout") return null;
  // Split on carriage returns as well as newlines: when redraws come faster than the pipe drains,
  // several land in one chunk separated only by the return that was meant to overwrite them.
  const lines = String(entry.message ?? "").split(/[\r\n]+/).filter((line) => line.trim());
  if (lines.length === 0) return null;
  const parsed = lines.map(parseProgressLine);
  if (parsed.some((line) => line === null)) return null;
  return parsed[parsed.length - 1];
}

// The same figures the terminal prints, in the drawer's own words. The bar carries the percentage,
// so the text carries what the bar cannot: where the run is, how long it has taken, how fast.
export function progressFigures(progress) {
  const parts = [
    `${progress.step}/${progress.total}`,
    `${progress.elapsed} → ${progress.remaining}`,
    `${progress.rate ?? "--"}s/${progress.unit}`,
  ];
  if (progress.jobStep !== null) parts.push(`任务 ${progress.jobStep}/${progress.jobTotal}`);
  if (progress.jobElapsed) parts.push(`累计 ${progress.jobElapsed}`);
  return parts.join(" · ");
}

// The same rule the backend uses to decide where one run ends: a counter that restarts is a new
// run, a counter that keeps climbing is the same one. The label is deliberately not part of it —
// the tiled Hires.fix pass renames itself on every step while staying a single run.
const sameRun = (previous, next) => previous.total === next.total && next.step >= previous.step;

export function collapseConsoleEntries(entries) {
  const collapsed = [];
  for (const entry of entries) {
    const progress = readProgressEntry(entry);
    if (!progress) {
      collapsed.push(entry);
      continue;
    }
    const previous = collapsed[collapsed.length - 1];
    if (previous?.progress && sameRun(previous.progress, progress)) {
      // The row keeps the id it was first rendered under so React updates it in place: the bar
      // grows rather than the row being torn down and rebuilt under the reader's cursor.
      collapsed[collapsed.length - 1] = { ...entry, id: previous.id, progress };
      continue;
    }
    collapsed.push({ ...entry, progress });
  }
  return collapsed;
}
