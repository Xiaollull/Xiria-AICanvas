// QPlan2/P2 Phase B: reproducible, bounded candidate selection over the exact integer space of
// lora-mix-rules.js. Preference is a soft tendency, not an aesthetic score; bounded work that finds
// only part of a large space is never reported as exhausting it.
import { baselineVector, countSpace, displacementOf, vectorKey } from "./lora-mix-rules.js";

export const SAMPLER_ALGORITHM = "lora-mix-stratified-fps-v1";
export const ENUMERATE_LIMIT = 6000n;
const MASK = (1n << 64n) - 1n, GOLDEN = 0x9E3779B97F4A7C15n;
export const validSamplingSeed = (text) => typeof text === "string" && /^\d{1,20}$/.test(text) && BigInt(text) <= MASK;

/** SplitMix64. The same (seed, stream) always yields the same sequence; nothing reads a clock. */
export function createRng(seedText, stream = 0) {
  if (!validSamplingSeed(seedText)) throw new Error("采样种子必须是 uint64 十进制文本");
  let state = (BigInt(seedText) + BigInt(stream) * GOLDEN) & MASK;
  const next = () => {
    state = (state + GOLDEN) & MASK;
    let z = state;
    z = ((z ^ (z >> 30n)) * 0xBF58476D1CE4E5B9n) & MASK;
    z = ((z ^ (z >> 27n)) * 0x94D049BB133111EBn) & MASK;
    return z ^ (z >> 31n);
  };
  const float = () => Number(next() >> 11n) / 9007199254740992;
  return { float, int: (bound) => Math.min(bound - 1, Math.floor(float() * bound)), shuffle(list) {
    for (let i = list.length - 1; i > 0; i -= 1) { const j = Math.floor(float() * (i + 1)); [list[i], list[j]] = [list[j], list[i]]; }
    return list;
  } };
}

const SIGN = { balanced: 0, min: -1, max: 1 };
const strataOf = (size) => Math.min(3, size);
const strataBounds = (size, strata, k) => [Math.floor(k * size / strata), Math.floor((k + 1) * size / strata) - 1];

/** Soft stratum weights: a preference tilts the mass, but every stratum keeps a real share. */
function strataWeights(strata, sign) {
  if (strata === 1) return [1];
  const raw = Array.from({ length: strata }, (_, k) => 1 + 1.2 * sign * (k - (strata - 1) / 2) / (strata - 1));
  const total = raw.reduce((sum, value) => sum + value, 0);
  return raw.map((value) => value / total);
}

/** Largest-remainder allocation so the pool's strata proportions are exact, then a seeded shuffle. */
function strataAssignment(count, weights, rng) {
  const exact = weights.map((weight) => weight * count), whole = exact.map(Math.floor);
  let left = count - whole.reduce((sum, value) => sum + value, 0);
  [...exact.keys()].sort((a, b) => (exact[b] - whole[b]) - (exact[a] - whole[a]) || a - b).forEach((index) => { if (left > 0) { whole[index] += 1; left -= 1; } });
  return rng.shuffle(whole.flatMap((amount, stratum) => Array(amount).fill(stratum)));
}

function enumerate(free, limit, cap) {
  const vectors = [], current = new Array(free.length);
  let complete = true;
  const visit = (position, used) => {
    if (!complete) return;
    if (position === free.length) {
      if (vectors.length >= cap) { complete = false; return; }
      vectors.push(current.slice()); return;
    }
    const { sequence } = free[position];
    for (let i = 0; i < sequence.size; i += 1) {
      const value = sequence.at(i), distance = Math.abs(value - sequence.base);
      if (limit !== null && used + distance > limit) continue;
      current[position] = value;
      visit(position + 1, used + distance);
      if (!complete) return;
    }
  };
  visit(0, 0);
  return { vectors, complete };
}

/** Nearest legal value to a real target, ties toward the base. Sequences are sorted: binary search. */
function nearestValue(sequence, target) {
  let low = 0, high = sequence.size - 1;
  while (low < high) { const mid = (low + high) >> 1; if (sequence.at(mid) < target) low = mid + 1; else high = mid; }
  let best = low, bestDistance = Math.abs(sequence.at(low) - target);
  for (const index of [low - 1, low + 1]) {
    if (index < 0 || index >= sequence.size) continue;
    const distance = Math.abs(sequence.at(index) - target);
    if (distance < bestDistance || (distance === bestDistance && Math.abs(index - sequence.baseIndex) < Math.abs(best - sequence.baseIndex))) { best = index; bestDistance = distance; }
  }
  return sequence.at(best);
}

/** Move members toward their bases until the displacement bound holds; bases are always legal values. */
function repair(free, vector, limit) {
  let total = displacementOf(free, vector);
  if (limit === null || total <= limit) return vector;
  const factor = limit / total;
  const out = vector.map((value, i) => { const { sequence } = free[i]; return nearestValue(sequence, sequence.base + (value - sequence.base) * factor); });
  total = displacementOf(free, out);
  while (total > limit) {
    let pick = -1, widest = 0;
    for (let i = 0; i < out.length; i += 1) { const distance = Math.abs(out[i] - free[i].sequence.base); if (distance > widest) { widest = distance; pick = i; } }
    const { sequence } = free[pick], here = sequence.indexOf(out[pick]);
    out[pick] = sequence.at(here > sequence.baseIndex ? here - 1 : here + 1);
    total = displacementOf(free, out);
  }
  return out;
}

const indexVector = (free, vector) => vector.map((value, i) => free[i].sequence.indexOf(value));

function preferenceScore(free, indices) {
  let total = 0, counted = 0;
  for (let i = 0; i < free.length; i += 1) {
    const sign = SIGN[free[i].preference] ?? 0;
    if (!sign || free[i].sequence.size < 2) continue;
    total += sign * (2 * indices[i] / (free[i].sequence.size - 1) - 1); counted += 1;
  }
  return counted ? total / counted : 0;
}

function distance(free, a, b) {
  let total = 0, counted = 0;
  for (let i = 0; i < free.length; i += 1) {
    const span = free[i].sequence.size - 1;
    if (span < 1) continue;
    total += Math.abs(a[i] - b[i]) / span; counted += 1;
  }
  return counted ? total / counted : 0;
}

/**
 * Select up to `count` distinct legal vectors. `free` rows are analyzeMembers().free rows, each
 * optionally carrying `preference`. `exclude` is a set of vector keys already tested/reserved in the
 * same context; `pinned` are kept vectors that count toward `count` and spread the rest away from them.
 */
export function proposeCandidates({ free, limit, count, seed, cursor = 0, includeBaseline = true, exclude = new Set(), pinned = [], workLimit }) {
  const base = baselineVector(free), baseKey = vectorKey(base), taken = new Set(exclude);
  const selected = [];
  const push = (vector, source) => { const key = vectorKey(vector); if (taken.has(key)) return false; taken.add(key); selected.push({ units: vector.slice(), key, displacement: displacementOf(free, vector), source }); return true; };
  // The baseline leads when requested (a kept baseline row counts as pinned, never twice); pinned rows follow.
  if (includeBaseline && selected.length < count) push(base, pinned.some((vector) => vectorKey(vector) === baseKey) ? "pinned" : "baseline");
  for (const vector of pinned) push(vector, "pinned");
  const result = { algorithm: SAMPLER_ALGORITHM, cursor, vectors: selected, enumerated: false, exhausted: false, workLimited: false, shortfall: 0, poolSize: 0 };
  const rng = createRng(seed, cursor);
  if (!free.length || selected.length >= count) { result.exhausted = !free.length && selected.length < count; result.shortfall = Math.max(0, count - selected.length); return result; }

  const space = countSpace(free, limit, { workLimit });
  let pool = [], enumeratedAll = false;
  if (space.kind === "exact" && space.count <= ENUMERATE_LIMIT) {
    const all = enumerate(free, limit, Number(ENUMERATE_LIMIT) + 1);
    enumeratedAll = all.complete;
    pool = all.vectors.filter((vector) => !taken.has(vectorKey(vector)));
    result.enumerated = enumeratedAll;
  } else {
    const target = Math.min(3000, Math.max(300, 30 * count)), seen = new Set(taken);
    const assignments = free.map((row) => strataAssignment(target, strataWeights(strataOf(row.sequence.size), SIGN[row.preference] ?? 0), rng));
    for (let j = 0; j < target; j += 1) {
      let vector = free.map((row, i) => {
        const { sequence } = row, strata = strataOf(sequence.size), [low, high] = strataBounds(sequence.size, strata, assignments[i][j]);
        return sequence.at(low + rng.int(high - low + 1));
      });
      // A third of the pool shrinks toward the baseline, so selection spans small and large changes.
      if (rng.float() < 1 / 3) {
        const shrink = 0.25 + 0.75 * rng.float();
        vector = vector.map((value, i) => { const { sequence } = free[i]; return nearestValue(sequence, sequence.base + (value - sequence.base) * shrink); });
      }
      vector = repair(free, vector, limit);
      const key = vectorKey(vector);
      if (seen.has(key)) continue;
      seen.add(key); pool.push(vector);
    }
    result.workLimited = true;
  }
  result.poolSize = pool.length;
  const poolIndices = pool.map((vector) => indexVector(free, vector));
  const reference = selected.map((entry) => indexVector(free, entry.units));
  const nearest = poolIndices.map((candidate) => reference.reduce((best, other) => Math.min(best, distance(free, candidate, other)), reference.length ? Infinity : 1));
  const alive = new Array(pool.length).fill(true);
  while (selected.length < count) {
    let best = -Infinity;
    const scores = new Array(pool.length).fill(-Infinity);
    for (let j = 0; j < pool.length; j += 1) {
      if (!alive[j]) continue;
      scores[j] = nearest[j] + 0.15 * preferenceScore(free, poolIndices[j]);
      if (scores[j] > best) best = scores[j];
    }
    if (best === -Infinity) break;
    // The seed chooses among near-best points, so a re-draw varies while a repeat of the same inputs is identical.
    const near = [];
    for (let j = 0; j < pool.length; j += 1) if (alive[j] && scores[j] >= best - 0.03) near.push(j);
    const pick = near[rng.int(near.length)];
    alive[pick] = false;
    push(pool[pick], "sampled");
    for (let j = 0; j < pool.length; j += 1) if (alive[j]) nearest[j] = Math.min(nearest[j], distance(free, poolIndices[j], poolIndices[pick]));
  }
  result.shortfall = Math.max(0, count - selected.length);
  // Only a complete enumeration proves the space is used up. A bounded pool proves nothing.
  result.exhausted = enumeratedAll && result.shortfall > 0;
  return result;
}
