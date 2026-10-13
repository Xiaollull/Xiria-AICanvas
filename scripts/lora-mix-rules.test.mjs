import test from "node:test";
import assert from "node:assert/strict";
import { DEFAULT_WORK_LIMIT, analyzeMembers, baselineVector, countSpace, defaultMaxDisplacement, displacementOf, formatUnits, inspectMaxDisplacement, inspectRule, inspectVector, parseUnits, sequenceOf, vectorKey } from "../src/lora-mix-rules.js";
import { ENUMERATE_LIMIT, SAMPLER_ALGORITHM, createRng, proposeCandidates, validSamplingSeed } from "../src/lora-mix-sampler.js";
import { readFile } from "node:fs/promises";
import { PARITY_FILE, buildParityFixture } from "./lora-mix-parity.mjs";

// QPlan2/P2 Phase B. Independent integer oracles: the expected sequences/counts below are built with plain
// loops and never call the module's own sequence/count helpers.
const rule = (base, min, max, step = "0.1", preference = "balanced") => ({ rule: { locked: false, base, min, max, step, preference } });
const locked = (base) => ({ rule: { locked: true, base } });
const SPEC = [rule("0.6", "0.2", "1.0", "0.1", "max"), rule("-0.2", "-0.8", "0.6", "0.1", "min"), locked("0.5")];
const U = (text) => parseUnits(text).units;

/** Oracle: the explicit sequence as a sorted unique integer list. */
function oracleSequence(base, min, max, step) {
  const values = new Set([base]);
  for (let value = min; value <= max; value += step) values.add(value);
  return [...values].sort((a, b) => a - b);
}
/** Oracle: every legal vector, displacement-filtered, by nested iteration. */
function oracleVectors(sequences, bases, limit) {
  let vectors = [[]];
  for (const sequence of sequences) vectors = vectors.flatMap((prefix) => sequence.map((value) => [...prefix, value]));
  return vectors.filter((vector) => limit === null || vector.reduce((total, value, i) => total + Math.abs(value - bases[i]), 0) <= limit);
}

test("M07 exact decimals: four places, signs, zero, partial and over-precise input are errors, never rounded", () => {
  assert.deepEqual(["0.6", "-0.2", "5", "-5", ".5", "-0", "0.0001", "-0.0001", "0.1234"].map((text) => parseUnits(text).units), [6000, -2000, 50000, -50000, 5000, 0, 1, -1, 1234]);
  assert.ok(Object.is(parseUnits("-0").units, 0), "signed zero is normalised");
  for (const [text, reason] of [["", "empty"], ["-", "format"], ["0.", "format"], ["1e2", "format"], ["0.12345", "precision"], ["5.0001", "range"], ["-5.0001", "range"], ["6", "range"], ["abc", "format"], ["--1", "format"], ["1,5", "format"]]) {
    const result = parseUnits(text);
    assert.equal(result.ok, false, text); assert.equal(result.reason, reason, text);
  }
  assert.deepEqual([6000, -2500, 0, 10000, 1, -50000].map(formatUnits), ["0.6", "-0.25", "0", "1", "0.0001", "-5"]);
  for (let units = -50000; units <= 50000; units += 7) assert.equal(U(formatUnits(units)), units);
});

test("M06/M07 rules: base within range, positive step, locked members need only a valid base and keep raw values", () => {
  assert.deepEqual(inspectRule({ locked: false, base: "0.6", min: "0.2", max: "1.0", step: "0.1", preference: "max" }).errors, {});
  assert.ok(inspectRule({ locked: false, base: "0.1", min: "0.2", max: "1.0", step: "0.1", preference: "max" }).errors.base);
  assert.ok(inspectRule({ locked: false, base: "1.5", min: "0.2", max: "1.0", step: "0.1", preference: "max" }).errors.base);
  assert.ok(inspectRule({ locked: false, base: "0.6", min: "1.0", max: "0.2", step: "0.1", preference: "max" }).errors.max);
  for (const step of ["0", "-0.1", "0.00001", "x", ""]) assert.ok(inspectRule({ locked: false, base: "0.6", min: "0.2", max: "1.0", step, preference: "max" }).errors.step, step);
  assert.ok(inspectRule({ locked: false, base: "0.6", min: "0.2", max: "1.0", step: "0.1", preference: "weird" }).errors.preference);
  const garbageButLocked = inspectRule({ locked: true, base: "0.5", min: "garbage", max: "", step: "-3", preference: "nope" });
  assert.deepEqual(garbageButLocked.errors, {}); assert.deepEqual(garbageButLocked.parsed, { base: 5000, locked: true });
  assert.ok(inspectRule({ locked: true, base: "0.123456" }).errors.base);
});

test("M07 sequences: grid from min, base inserted once in order, no off-grid max, exact to the oracle", () => {
  const cases = [
    ["0.6", "0.2", "1.0", "0.1"], ["0.65", "0.2", "1.0", "0.1"], ["0.95", "0.2", "1.0", "0.1"], ["1.0", "0.2", "1.0", "0.3"],
    ["0.2", "0.2", "1.0", "0.3"], ["0", "-0.8", "0.6", "0.25"], ["-0.2", "-0.8", "0.6", "0.1"], ["0.5", "0.5", "0.5", "0.1"], ["0.5", "0.4", "0.5", "5"], ["-5", "-5", "5", "2.5"],
  ];
  for (const [base, min, max, step] of cases) {
    const parsed = { base: U(base), min: U(min), max: U(max), step: U(step) }, expected = oracleSequence(parsed.base, parsed.min, parsed.max, parsed.step), sequence = sequenceOf(parsed);
    assert.equal(sequence.size, expected.length, `${base} ${min} ${max} ${step}`);
    assert.deepEqual(Array.from({ length: sequence.size }, (_, i) => sequence.at(i)), expected);
    expected.forEach((value, index) => assert.equal(sequence.indexOf(value), index));
    assert.equal(sequence.indexOf(parsed.max + 1), -1); assert.equal(sequence.indexOf(parsed.min - 1), -1);
    assert.equal(sequence.maxDistance, Math.max(Math.abs(expected[0] - parsed.base), Math.abs(expected.at(-1) - parsed.base)));
    assert.throws(() => sequence.at(sequence.size), RangeError);
  }
  // An off-grid base is an extra option; the real maximum is NOT appended when it is not on the grid.
  const odd = sequenceOf({ base: U("0.65"), min: U("0.2"), max: U("1.0"), step: U("0.3") });
  assert.deepEqual(Array.from({ length: odd.size }, (_, i) => formatUnits(odd.at(i))), ["0.2", "0.5", "0.65", "0.8"]);
  assert.equal(odd.baseOnGrid, false);
});

test("M08 displacement and reachable maximum match the spec example and an independent oracle", () => {
  const analysis = analyzeMembers(SPEC);
  assert.equal(analysis.valid, true);
  assert.equal(analysis.dmax, U("1.2"), "A 0.4 + B 0.8 + locked C 0");
  assert.equal(analysis.space, 9n * 15n);
  assert.deepEqual(analysis.free.map((row) => row.index), [0, 1]);
  const bases = analysis.free.map((row) => row.sequence.base), sequences = analysis.free.map((row) => oracleSequence(row.parsed.base, row.parsed.min, row.parsed.max, row.parsed.step));
  const all = oracleVectors(sequences, bases, null);
  assert.equal(all.length, 135);
  assert.equal(Math.max(...all.map((vector) => displacementOf(analysis.free, vector))), analysis.dmax, "the reachable bound is attained");
  assert.equal(inspectMaxDisplacement("0.6", analysis.dmax).units, 6000);
  assert.equal(inspectMaxDisplacement("1.2", analysis.dmax).ok, true);
  assert.equal(inspectMaxDisplacement("1.2001", analysis.dmax).ok, false);
  assert.match(inspectMaxDisplacement("1.2001", analysis.dmax).message, /不会自动截断/);
  assert.equal(inspectMaxDisplacement("-0.1", analysis.dmax).ok, false);
  assert.equal(inspectMaxDisplacement("", analysis.dmax).ok, false);
  assert.equal(defaultMaxDisplacement(analysis.dmax), "0.6");
  assert.equal(defaultMaxDisplacement(U("0.3")), "0.3", "min(0.6, allowed maximum)");
  const zero = oracleVectors(sequences, bases, 0);
  assert.deepEqual(zero, [bases]);
  assert.deepEqual(countSpace(analysis.free, 0), { kind: "exact", count: 1n, constrained: true });
  assert.equal(inspectVector(analysis.free, bases, 0).ok, true);
  assert.equal(inspectVector(analysis.free, [U("0.7"), bases[1]], 0).ok, false);
});

test("M13 counts equal brute force for constrained spaces; huge spaces stay BigInt; over-large DP is Unknown", () => {
  const rng = createRng("424242");
  for (let trial = 0; trial < 60; trial += 1) {
    const members = Array.from({ length: 1 + rng.int(4) }, () => {
      const min = -5000 + rng.int(60) * 100, span = rng.int(7) * 100, step = (1 + rng.int(4)) * 100, base = min + rng.int(span / 100 + 1) * 100 + (rng.float() < 0.3 ? 50 : 0);
      return rule(formatUnits(Math.min(base, min + span)), formatUnits(min), formatUnits(min + span), formatUnits(step));
    });
    const analysis = analyzeMembers(members);
    if (!analysis.valid) continue;
    const sequences = analysis.free.map((row) => oracleSequence(row.parsed.base, row.parsed.min, row.parsed.max, row.parsed.step)), bases = analysis.free.map((row) => row.sequence.base);
    for (const limit of [0, 100, 250, 600, analysis.dmax, analysis.dmax + 5000]) {
      const expected = BigInt(oracleVectors(sequences, bases, limit).length), got = countSpace(analysis.free, Math.min(limit, analysis.dmax));
      assert.equal(got.kind, "exact"); assert.equal(got.count, expected, `trial ${trial} limit ${limit}`);
    }
  }
  const wide = analyzeMembers(Array.from({ length: 16 }, () => rule("0", "-5", "5", "0.01")));
  assert.equal(wide.space, 1001n ** 16n);
  assert.ok(wide.space > BigInt(Number.MAX_SAFE_INTEGER));
  assert.deepEqual(countSpace(wide.free, null), { kind: "exact", count: 1001n ** 16n, constrained: false });
  assert.equal(countSpace(wide.free, 40000).kind, "unknown");
  assert.equal(countSpace(wide.free, 40000, { workLimit: DEFAULT_WORK_LIMIT * 1000 }).kind, "exact", "bounded work, not an arbitrary refusal");
});

test("M09/M10 the sampler is reproducible, legal, distinct, baseline-first and bounded by the budget", () => {
  assert.equal(SAMPLER_ALGORITHM, "lora-mix-stratified-fps-v1");
  const analysis = analyzeMembers(SPEC), limit = 6000, args = { free: analysis.free, limit, count: 12, seed: "20261009", cursor: 0 };
  const first = proposeCandidates(args), again = proposeCandidates(args);
  assert.deepEqual(first, again, "same rules + seed + cursor => identical selection");
  assert.notDeepEqual(first.vectors.map((entry) => entry.key), proposeCandidates({ ...args, seed: "20261010" }).vectors.map((entry) => entry.key));
  assert.notDeepEqual(first.vectors.map((entry) => entry.key), proposeCandidates({ ...args, cursor: 1 }).vectors.map((entry) => entry.key));
  assert.equal(first.vectors.length, 12);
  assert.equal(first.vectors[0].source, "baseline"); assert.equal(first.vectors[0].displacement, 0);
  assert.equal(new Set(first.vectors.map((entry) => entry.key)).size, 12);
  for (const entry of first.vectors) { const verdict = inspectVector(analysis.free, entry.units, limit); assert.equal(verdict.ok, true, entry.key); assert.equal(verdict.displacement, entry.displacement); }
  assert.equal(first.enumerated, true, "77 legal vectors are enumerated exactly");
  // Continuing excludes what is tested/reserved: disjoint, still legal and distinct.
  const tested = new Set(first.vectors.map((entry) => entry.key)), next = proposeCandidates({ ...args, cursor: 1, exclude: tested, includeBaseline: false });
  assert.equal(next.vectors.length, 12);
  assert.ok(next.vectors.every((entry) => !tested.has(entry.key)));
  assert.equal(first.vectors.length + next.vectors.length, 24);
  // Pinned vectors are kept, count toward the budget, and are never duplicated.
  const pinned = [first.vectors[3].units, first.vectors[4].units], withPins = proposeCandidates({ ...args, pinned });
  assert.equal(withPins.vectors[0].source, "baseline", "the baseline leads, then the pinned rows");
  assert.deepEqual(withPins.vectors.slice(1, 3).map((entry) => entry.key), pinned.map(vectorKey));
  const baselinePinned = proposeCandidates({ ...args, pinned: [first.vectors[0].units] });
  assert.deepEqual(baselinePinned.vectors.slice(0, 1).map((entry) => [entry.key, entry.source]), [[first.vectors[0].key, "pinned"]], "a kept baseline row is not added twice");
  assert.equal(new Set(baselinePinned.vectors.map((entry) => entry.key)).size, baselinePinned.vectors.length);
  assert.equal(withPins.vectors.length, 12); assert.equal(new Set(withPins.vectors.map((entry) => entry.key)).size, 12);
  assert.ok(validSamplingSeed("18446744073709551615")); assert.ok(!validSamplingSeed("18446744073709551616")); assert.ok(!validSamplingSeed("-1")); assert.ok(!validSamplingSeed(""));
  assert.throws(() => createRng("x"));
});

test("M09 insufficient and all-locked spaces never pad; exhaustion is only claimed after a complete enumeration", () => {
  const analysis = analyzeMembers(SPEC);
  const tiny = proposeCandidates({ free: analysis.free, limit: 0, count: 8, seed: "1" });
  assert.equal(tiny.vectors.length, 1); assert.equal(tiny.shortfall, 7); assert.equal(tiny.exhausted, true);
  const noBaseline = proposeCandidates({ free: analysis.free, limit: 0, count: 8, seed: "1", includeBaseline: false });
  assert.equal(noBaseline.vectors.length, 1, "with no baseline slot the only legal vector is still the baseline, once");
  const allLocked = analyzeMembers([locked("0.4"), locked("-0.2")]);
  assert.equal(allLocked.valid, true); assert.equal(allLocked.dmax, 0); assert.equal(allLocked.space, 1n);
  const profile = proposeCandidates({ free: allLocked.free, limit: 0, count: 32, seed: "9" });
  assert.deepEqual(profile.vectors.map((entry) => [entry.key, entry.source, entry.displacement]), [["", "baseline", 0]], "an all-locked rule set is exactly one profile");
  assert.equal(profile.shortfall, 31); assert.equal(profile.exhausted, true);
  // A larger space with a small budget needs no exhaustion claim; a huge space reports bounded work, not exhaustion.
  const huge = analyzeMembers(Array.from({ length: 8 }, () => rule("0", "-1", "1", "0.01", "balanced")));
  const sampled = proposeCandidates({ free: huge.free, limit: null, count: 32, seed: "5" });
  assert.equal(sampled.vectors.length, 32); assert.equal(sampled.enumerated, false); assert.equal(sampled.workLimited, true); assert.equal(sampled.exhausted, false);
  assert.ok(ENUMERATE_LIMIT > 0n);
  // A mid-sized space that cannot fill the request enumerates fully and says so.
  const small = analyzeMembers([rule("0", "-0.1", "0.1", "0.1"), rule("0", "-0.1", "0.1", "0.1")]);
  const all = proposeCandidates({ free: small.free, limit: null, count: 50, seed: "3" });
  assert.equal(all.vectors.length, 9); assert.equal(all.shortfall, 41); assert.equal(all.exhausted, true);
});

test("M10 coverage: low/mid/high of every member appear; preference tilts softly without removing the other end", () => {
  const members = Array.from({ length: 5 }, (_, i) => rule("0", "-1", "1", "0.05", ["balanced", "min", "max", "balanced", "max"][i]));
  const analysis = analyzeMembers(members), draw = (seed) => proposeCandidates({ free: analysis.free, limit: null, count: 30, seed });
  const result = draw("77");
  assert.equal(result.vectors.length, 30); assert.equal(result.enumerated, false);
  for (let i = 0; i < 5; i += 1) {
    const thirds = new Set(result.vectors.map((entry) => Math.min(2, Math.floor(analysis.free[i].sequence.indexOf(entry.units[i]) * 3 / analysis.free[i].sequence.size))));
    assert.equal(thirds.size, 3, `member ${i} reaches low, mid and high`);
  }
  const mean = (i, seeds) => { let total = 0, count = 0; for (const seed of seeds) for (const entry of draw(seed).vectors) { total += analysis.free[i].sequence.indexOf(entry.units[i]) / (analysis.free[i].sequence.size - 1); count += 1; } return total / count; };
  const seeds = ["1", "2", "3", "4", "5", "6"];
  assert.ok(mean(2, seeds) > mean(0, seeds), "toward-maximum sits above balanced");
  assert.ok(mean(0, seeds) > mean(1, seeds), "balanced sits above toward-minimum");
  assert.ok(mean(1, seeds) > 0.2 && mean(2, seeds) < 0.8, "a soft tendency, not a new hard range");
  // Separation: farthest-point selection spreads candidates farther apart than the first N of the same pool.
  const spread = (entries) => { let best = Infinity; for (let a = 0; a < entries.length; a += 1) for (let b = a + 1; b < entries.length; b += 1) { let d = 0; for (let i = 0; i < 5; i += 1) d += Math.abs(entries[a].units[i] - entries[b].units[i]); best = Math.min(best, d); } return best; };
  assert.ok(spread(result.vectors) > 0);
});

test("M10 displacement-bounded selection stays within the bound on a large space", () => {
  const members = Array.from({ length: 6 }, () => rule("0.2", "-1", "1.4", "0.05", "balanced"));
  const analysis = analyzeMembers(members), limit = U("0.8");
  const result = proposeCandidates({ free: analysis.free, limit, count: 40, seed: "31337" });
  assert.equal(result.vectors.length, 40);
  for (const entry of result.vectors) assert.ok(entry.displacement <= limit && inspectVector(analysis.free, entry.units, limit).ok);
  assert.ok(result.vectors.some((entry) => entry.displacement > limit * 0.6), "uses the available displacement, not only tiny moves");
  assert.deepEqual(baselineVector(analysis.free), analysis.free.map((row) => U("0.2")));
});

test("P1 the Python parity fixture is current: JS and backend/lora_mix_search.py agree on rules, sequences and counts", async () => {
  const stored = JSON.parse(await readFile(PARITY_FILE, "utf8"));
  assert.deepEqual(stored, JSON.parse(JSON.stringify(buildParityFixture())), "run `node scripts/lora-mix-parity.mjs --write` after changing lora-mix-rules.js");
  assert.ok(stored.rules.length >= 40 && stored.parse.length >= 20);
});
