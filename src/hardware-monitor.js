// Presentation rules for the hardware monitor panel.
//
// These live outside `App.jsx` because the panel's old inaccuracies were arithmetic, not layout:
// a missing reading charted as 0, one series read in MiB and another in rounded GB, and a fixed
// interval that assumed every reply arrived before the next request went out. Keeping the
// arithmetic here makes each of those a testable rule instead of an expression buried in JSX.

export const HARDWARE_POLL_MS = 2000;
export const HARDWARE_HISTORY_SAMPLES = 30;
export const EMPTY_HARDWARE_HISTORY = Object.freeze({ gpu: [], vram: [], cpu: [], ram: [] });

// The backend reports every memory figure in MiB, so one formatter keeps the panel from
// converting the same quantity two different ways in two different places.
export function formatMib(value) {
  if (!Number.isFinite(value)) return "--";
  return value < 1024 ? `${Math.round(value)} MB` : `${(value / 1024).toFixed(1)} GB`;
}

// A reading the backend could not take is `null`, and it stays `null` all the way to the chart.
// Substituting 0 is what made a machine with no `nvidia-smi` look like a permanently idle GPU.
export function appendHardwareSample(history, stats) {
  const push = (series, value) => [...(series || []), Number.isFinite(value) ? value : null].slice(-HARDWARE_HISTORY_SAMPLES);
  return {
    gpu: push(history?.gpu, stats?.gpu_util),
    vram: push(history?.vram, stats?.vram_used_mb / 1024),
    cpu: push(history?.cpu, stats?.cpu_percent),
    ram: push(history?.ram, stats?.ram_used_mb / 1024),
  };
}

// Gaps break the line rather than dragging it to the floor and back.
export function sparklineSegments(data, ceiling) {
  const segments = [];
  const span = Math.max(1, data.length - 1);
  let run = [];
  data.forEach((value, index) => {
    if (!Number.isFinite(value)) {
      if (run.length) segments.push(run);
      run = [];
      return;
    }
    const x = (index / span) * 100;
    const y = 100 - Math.min(1, Math.max(0, value / ceiling)) * 100;
    run.push(`${x},${y}`);
  });
  if (run.length) segments.push(run);
  // A lone sample between two gaps has no line to draw; doubling the point lets the round cap
  // show it as a dot instead of dropping it without trace.
  return segments.map((points) => (points.length === 1 ? [points[0], points[0]] : points).join(" "));
}

// Where the allocator wall sits along the VRAM bar. Without the mark, a capped card looks like a
// bar that stops short of full for no stated reason.
export function vramWallPercent(stats) {
  const limit = stats?.vram_limit_mb;
  const total = stats?.vram_total_mb;
  if (!Number.isFinite(limit) || !Number.isFinite(total) || total <= 0 || limit >= total) return null;
  return Math.min(100, Math.max(0, (limit / total) * 100));
}

// Temperature, utilisation, power and fan come from `nvidia-smi` on its own cadence, so the panel
// states how old they are instead of implying they were read with everything else.
export function sensorAgeLabel(stats) {
  const age = stats?.gpu_sensor_age_ms;
  if (!Number.isFinite(age)) return null;
  return age < 1000 ? "刚刚" : `${(age / 1000).toFixed(1)} 秒前`;
}
