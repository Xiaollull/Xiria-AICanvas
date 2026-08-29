import assert from "node:assert/strict";
import test from "node:test";
import {
  MAX_RTX_OUTPUT_EDGE,
  MAX_RTX_OUTPUT_PIXELS,
  bankersRound,
  enabledPostprocessStages,
  normalizePostprocessOrder,
  postprocessTargetSize,
} from "../src/postprocessing.js";

const permutations = [
  ["hires", "adetailer", "rtx"],
  ["hires", "rtx", "adetailer"],
  ["adetailer", "hires", "rtx"],
  ["adetailer", "rtx", "hires"],
  ["rtx", "hires", "adetailer"],
  ["rtx", "adetailer", "hires"],
];

const settings = {
  hires: { enabled: true, scale: 1.5 },
  adetailer: { enabled: true },
  rtx: { enabled: true, scale: 2 },
};

test("postprocess order normalizes every permutation and malformed persisted state", () => {
  for (const permutation of permutations) {
    assert.deepEqual(normalizePostprocessOrder(permutation), permutation);
  }
  assert.deepEqual(normalizePostprocessOrder(["rtx", "rtx", "unknown", "hires"]), ["rtx", "hires", "adetailer"]);
  assert.deepEqual(normalizePostprocessOrder(["adetailer"]), ["adetailer", "hires", "rtx"]);
});

test("legacy Hires order migrates to an exact three-stage permutation", () => {
  assert.deepEqual(normalizePostprocessOrder(undefined, "before_adetailer"), ["hires", "adetailer", "rtx"]);
  assert.deepEqual(normalizePostprocessOrder(undefined, "after_adetailer"), ["adetailer", "hires", "rtx"]);
  assert.deepEqual(normalizePostprocessOrder(undefined, "invalid"), ["hires", "adetailer", "rtx"]);
});

test("enabled stages preserve order for all permutations and enabled subsets", () => {
  for (const permutation of permutations) {
    for (let mask = 0; mask < 8; mask += 1) {
      const subsetSettings = Object.fromEntries(["hires", "adetailer", "rtx"].map((stage, index) => [stage, { enabled: Boolean(mask & (1 << index)) }]));
      assert.deepEqual(
        enabledPostprocessStages(permutation, subsetSettings),
        permutation.filter((stage) => subsetSettings[stage].enabled),
      );
    }
  }
});

test("target dimensions and stage traces follow all six permutations", () => {
  for (const order of permutations) {
    const result = postprocessTargetSize({ width: 1000, height: 700 }, order, settings);
    const hiresFirst = order.indexOf("hires") < order.indexOf("rtx");
    assert.deepEqual(
      { width: result.width, height: result.height },
      hiresFirst ? { width: 3072, height: 2176 } : { width: 3008, height: 2112 },
    );
    assert.deepEqual(result.trace.map((entry) => entry.stage), order);
    assert.deepEqual(result.trace[0].input, { width: 1000, height: 700 });
    for (let index = 1; index < result.trace.length; index += 1) {
      assert.deepEqual(result.trace[index].input, result.trace[index - 1].output);
    }
    assert.deepEqual(result.trace.at(-1).output, { width: result.width, height: result.height });
    assert.equal(result.valid, true);
  }
});

test("target traces contain only enabled stages for every subset", () => {
  for (const permutation of permutations) {
    for (let mask = 0; mask < 8; mask += 1) {
      const subsetSettings = Object.fromEntries(Object.entries(settings).map(([stage, value], index) => [stage, { ...value, enabled: Boolean(mask & (1 << index)) }]));
      const result = postprocessTargetSize({ width: 1024, height: 768 }, permutation, subsetSettings);
      assert.deepEqual(result.trace.map((entry) => entry.stage), enabledPostprocessStages(permutation, subsetSettings));
      assert.equal(result.valid, true);
    }
  }
});

test("RTX target accepts exact limits and rejects unsafe intermediate dimensions", () => {
  const rtxOnly = { hires: { enabled: false, scale: 1 }, adetailer: { enabled: false }, rtx: { enabled: true, scale: 2 } };
  const exact = postprocessTargetSize({ width: 4096, height: 2048 }, ["rtx", "hires", "adetailer"], rtxOnly);
  assert.deepEqual({ width: exact.width, height: exact.height }, { width: MAX_RTX_OUTPUT_EDGE, height: 4096 });
  assert.equal(exact.width * exact.height, MAX_RTX_OUTPUT_PIXELS);
  assert.equal(exact.valid, true);

  const edge = postprocessTargetSize({ width: 4100, height: 1024 }, ["rtx", "hires", "adetailer"], rtxOnly);
  assert.equal(edge.valid, false);
  assert.equal(edge.trace[0].valid, false);
  assert.match(edge.reason, /8192-edge/);

  const pixels = postprocessTargetSize({ width: 3000, height: 3000 }, ["rtx", "hires", "adetailer"], rtxOnly);
  assert.equal(pixels.valid, false);
  assert.ok(pixels.width * pixels.height > MAX_RTX_OUTPUT_PIXELS);

  const afterHires = postprocessTargetSize({ width: 2048, height: 2048 }, ["hires", "rtx", "adetailer"], {
    hires: { enabled: true, scale: 2 }, adetailer: { enabled: false }, rtx: { enabled: true, scale: 2 },
  });
  assert.equal(afterHires.trace.find((entry) => entry.stage === "rtx").valid, false);
});

test("banker rounding matches Python ties-to-even behavior", () => {
  assert.equal(bankersRound(0.5), 0);
  assert.equal(bankersRound(-0.5), 0);
  assert.equal(bankersRound(2.5), 2);
  assert.equal(bankersRound(3.5), 4);
  assert.equal(bankersRound(-2.5), -2);
  assert.equal(bankersRound(-3.5), -4);
  assert.equal(bankersRound(2.500001), 3);
  assert.equal(bankersRound(2.499999), 2);

  const ties = postprocessTargetSize({ width: 10, height: 14 }, ["rtx", "hires", "adetailer"], {
    hires: { enabled: false, scale: 1 }, adetailer: { enabled: false }, rtx: { enabled: true, scale: 2 },
  });
  assert.deepEqual({ width: ties.width, height: ties.height }, { width: 16, height: 32 });

  const fractional = postprocessTargetSize({ width: 512, height: 512 }, ["rtx", "hires", "adetailer"], {
    hires: { enabled: false, scale: 1 }, adetailer: { enabled: false }, rtx: { enabled: true, scale: 1.04 },
  });
  assert.deepEqual({ width: fractional.width, height: fractional.height }, { width: 528, height: 528 });
});
