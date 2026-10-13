// QPlan2/P2 Phase B: exact LoRA mix-search rules. Pure, integer-only and deterministic.
// A weight is an integer count of 0.0001 ("units"); nothing here rounds, clamps or draws randomness.
// Raw text is always kept by the caller so a partial "-" or "0." is an error, never a default.

export const UNIT = 10000;
export const MAX_UNITS = 5 * UNIT;
export const PREFERENCES = ["balanced", "min", "max"];
export const PREFERENCE_LABELS = { balanced: "均衡", min: "偏向最小", max: "偏向最大" };
export const DEFAULT_STEP = "0.1";
const DECIMAL = /^(-?)(\d+(?:\.\d+)?|\.\d+)$/;

/** Exact decimal text -> units. Over-precision, non-finite text and |w| > 5 are refused, never rounded. */
export function parseUnits(text) {
  if (typeof text === "number") text = Number.isFinite(text) ? String(text) : "";
  if (typeof text !== "string") return { ok: false, reason: "format", message: "需要十进制数" };
  const raw = text.trim();
  if (!raw) return { ok: false, reason: "empty", message: "不能为空" };
  if (/e/i.test(raw) || !DECIMAL.test(raw)) return { ok: false, reason: "format", message: "需要普通十进制数（例如 0.6 或 -0.25）" };
  const [, sign, body] = DECIMAL.exec(raw);
  const [whole = "", fraction = ""] = body.split(".");
  if (fraction.length > 4) return { ok: false, reason: "precision", message: "最多 4 位小数；不会四舍五入" };
  const wholeValue = Number(whole.replace(/^0+(?=\d)/, "") || "0");
  const units = wholeValue > 5 ? Infinity : wholeValue * UNIT + Number(fraction.padEnd(4, "0") || "0");
  if (units > MAX_UNITS) return { ok: false, reason: "range", message: "权重需在 −5 到 5 之间" };
  return { ok: true, units: sign && units ? -units : 0 + units };
}

/** Canonical, trimmed decimal text for a unit count (-0 is 0). */
export function formatUnits(units) {
  if (!Number.isInteger(units)) return "";
  const sign = units < 0 ? "-" : "", absolute = Math.abs(units);
  const whole = Math.trunc(absolute / UNIT), fraction = String(absolute % UNIT).padStart(4, "0").replace(/0+$/, "");
  return `${sign}${whole}${fraction ? `.${fraction}` : ""}`;
}

export function newRule(base = "1") {
  return { locked: false, base: String(base), min: "", max: "", preference: "balanced", step: DEFAULT_STEP };
}

/**
 * Validate one member's raw rule. Locked members only need a valid base; unlocked members need
 * min <= base <= max and a strictly positive step. Errors are keyed by field so the UI can place them.
 */
export function inspectRule(rule) {
  const errors = {}, parsed = {};
  const read = (field, label) => {
    const value = parseUnits(rule?.[field]);
    if (value.ok) parsed[field] = value.units; else errors[field] = `${label}${value.message}`;
  };
  read("base", "基础权重");
  if (rule?.locked === true) return { errors, parsed: Object.keys(errors).length ? null : { base: parsed.base, locked: true }, locked: true };
  read("min", "最小值"); read("max", "最大值"); read("step", "步长");
  if (!PREFERENCES.includes(rule?.preference)) errors.preference = "偏好需为 均衡 / 偏向最小 / 偏向最大";
  if (!errors.step && parsed.step <= 0) errors.step = "步长必须大于 0";
  if (!errors.min && !errors.max && parsed.min > parsed.max) errors.max = "最小值不能大于最大值";
  if (!errors.base && !errors.min && !errors.max && !errors.order) {
    if (parsed.base < parsed.min || parsed.base > parsed.max) errors.base = "基础权重必须位于最小值与最大值之间";
  }
  return { errors, parsed: Object.keys(errors).length ? null : { ...parsed, locked: false }, locked: false };
}

/**
 * The explicit candidate sequence of an unlocked member: min, min+step, ... while <= max, then the base
 * as a deduplicated Keep Baseline option. No off-grid maximum is appended and the base is never rounded.
 */
export function sequenceOf({ base, min, max, step }) {
  const gridCount = Math.floor((max - min) / step) + 1;
  const onGrid = (base - min) % step === 0;
  const baseSlot = onGrid ? (base - min) / step : Math.ceil((base - min) / step);
  const size = gridCount + (onGrid ? 0 : 1);
  const at = (index) => {
    if (!Number.isInteger(index) || index < 0 || index >= size) throw new RangeError("sequence index out of range");
    if (onGrid || index < baseSlot) return min + index * step;
    return index === baseSlot ? base : min + (index - 1) * step;
  };
  const indexOf = (units) => {
    if (units === base) return baseSlot;
    if (units < min || units > max || (units - min) % step !== 0) return -1;
    const grid = (units - min) / step;
    return onGrid || grid < baseSlot ? grid : grid + 1;
  };
  const first = min, last = Math.max(min + (gridCount - 1) * step, base);
  return { size, at, indexOf, baseIndex: baseSlot, baseOnGrid: onGrid, first, last, base, min, max, step, maxDistance: Math.max(Math.abs(first - base), Math.abs(last - base)) };
}

/** Per-member analysis of an ordered member list ({ rule }). Never throws; errors stay with their row. */
export function analyzeMembers(members) {
  const rows = members.map((member, index) => {
    const result = inspectRule(member.rule);
    return { index, locked: result.locked, errors: result.errors, parsed: result.parsed, preference: member.rule?.preference || "balanced",
      sequence: result.parsed && !result.locked ? sequenceOf(result.parsed) : null };
  });
  const free = rows.filter((row) => row.sequence);
  const valid = rows.length > 0 && rows.every((row) => !Object.keys(row.errors).length);
  // Reachable loose upper bound: each member's farthest reachable endpoint from its own base.
  const dmax = valid ? free.reduce((total, row) => total + row.sequence.maxDistance, 0) : null;
  const space = valid ? free.reduce((total, row) => total * BigInt(row.sequence.size), 1n) : null;
  return { rows, valid, free, dmax, space };
}

/** Parse the X3 maximum displacement. Out-of-range input is an error, never clamped. */
export function inspectMaxDisplacement(text, dmax) {
  const value = parseUnits(text);
  if (!value.ok) return { ok: false, message: `最大总位移${value.message}` };
  if (value.units < 0) return { ok: false, message: "最大总位移不能为负" };
  if (dmax !== null && dmax !== undefined && value.units > dmax) return { ok: false, message: `最大总位移超过可达上限 ${formatUnits(dmax)}；不会自动截断` };
  return { ok: true, units: value.units };
}

/** Fresh drafts may start at min(0.6, allowed maximum); an existing out-of-range value is left alone. */
export function defaultMaxDisplacement(dmax) {
  return formatUnits(Math.min(6000, dmax ?? 6000));
}

/** Sum of absolute distances from each unlocked member's fixed base. Locked members never contribute. */
export function displacementOf(free, vector) {
  let total = 0;
  for (let i = 0; i < free.length; i += 1) total += Math.abs(vector[i] - free[i].sequence.base);
  return total;
}

export const vectorKey = (vector) => vector.join(",");
export const baselineVector = (free) => free.map((row) => row.sequence.base);

/** Is the vector a legal point of the space (on every member's sequence) and within the displacement bound? */
export function inspectVector(free, vector, limit) {
  if (!Array.isArray(vector) || vector.length !== free.length) return { ok: false, reason: "length", message: "向量长度与解锁成员不一致" };
  for (let i = 0; i < free.length; i += 1) {
    if (!Number.isInteger(vector[i])) return { ok: false, reason: "value", index: i, message: "存在无效权重" };
    if (free[i].sequence.indexOf(vector[i]) < 0) return { ok: false, reason: "sequence", index: i, message: "不在该成员的候选序列中（未取整、未截断）" };
  }
  const displacement = displacementOf(free, vector);
  if (limit !== null && limit !== undefined && displacement > limit) return { ok: false, reason: "displacement", displacement, message: `总位移 ${formatUnits(displacement)} 超过上限 ${formatUnits(limit)}` };
  return { ok: true, displacement };
}

export const DEFAULT_WORK_LIMIT = 4_000_000;

/**
 * Feasible vector count under displacement <= limit.
 *  - exact/unconstrained: product of sequence sizes (BigInt);
 *  - exact/constrained: displacement-count convolution while the work stays bounded;
 *  - unknown: too large to count exactly. Never a materialised list and never a lossy Number.
 */
export function countSpace(free, limit, { workLimit = DEFAULT_WORK_LIMIT } = {}) {
  if (!free.length) return { kind: "exact", count: 1n, constrained: false };
  const space = free.reduce((total, row) => total * BigInt(row.sequence.size), 1n);
  const dmax = free.reduce((total, row) => total + row.sequence.maxDistance, 0);
  if (limit === null || limit === undefined || limit >= dmax) return { kind: "exact", count: space, constrained: false };
  const tables = free.map((row) => {
    const table = new Map();
    for (let i = 0; i < row.sequence.size; i += 1) {
      const distance = Math.abs(row.sequence.at(i) - row.sequence.base);
      if (distance <= limit) table.set(distance, (table.get(distance) || 0n) + 1n);
    }
    return [...table.entries()];
  });
  const work = tables.reduce((total, table) => total + table.length, 0) * (limit + 1);
  if (work > workLimit) return { kind: "unknown", constrained: true, reason: "过大，无法在有限工作量内精确计数" };
  let ways = new Array(limit + 1).fill(0n);
  ways[0] = 1n;
  for (const table of tables) {
    const next = new Array(limit + 1).fill(0n);
    for (let used = 0; used <= limit; used += 1) {
      if (!ways[used]) continue;
      for (const [distance, count] of table) { if (used + distance > limit) continue; next[used + distance] += ways[used] * count; }
    }
    ways = next;
  }
  return { kind: "exact", count: ways.reduce((total, value) => total + value, 0n), constrained: true };
}
