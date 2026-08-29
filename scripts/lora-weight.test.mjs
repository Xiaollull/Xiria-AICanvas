import assert from "node:assert/strict";
import test from "node:test";

import { readFile } from "node:fs/promises";

import {
  WEIGHT_MAXIMUM,
  WEIGHT_MINIMUM,
  clampWeight,
  formatWeight,
  parseWeightInput,
  sliderWeight,
  steppedWeight,
  weightPrecisionLabel,
} from "../src/lora-weight.js";

test("weights are clamped to the mountable range and never NaN", () => {
  assert.equal(clampWeight(9), WEIGHT_MAXIMUM);
  assert.equal(clampWeight(-9), WEIGHT_MINIMUM);
  assert.equal(clampWeight("1.5"), 1.5);
  assert.equal(clampWeight("abc"), 0);
  assert.equal(clampWeight(undefined), 0);
  assert.equal(clampWeight(Infinity), 0);
});

test("a plain slider drag moves in tenths and keeps the hundredth the user set", () => {
  // The control advertises "步进 0.1"; a drag must not silently discard a 1.25.
  assert.deepEqual(sliderWeight({ weight: 1.25, precision: 2 }, 1.73), { weight: 1.75, precision: 2 });
  assert.deepEqual(sliderWeight({ weight: 1, precision: 1 }, 1.73), { weight: 1.7, precision: 1 });
  // Shift is the fine gesture: hundredths, and it records that precision.
  assert.deepEqual(sliderWeight({ weight: 1, precision: 1 }, 1.734, true), { weight: 1.73, precision: 2 });
  // Negative weights keep their sign rather than folding through zero.
  assert.equal(sliderWeight({ weight: -1, precision: 1 }, -2.31).weight, -2.3);
  assert.equal(sliderWeight({ weight: 0, precision: 1 }, 99).weight, WEIGHT_MAXIMUM);
});

test("a stepper click moves exactly 0.1 and snaps a hand-typed value onto the grid", () => {
  // 1 / 0.1 is 9.999999999999998, so a naive grid test makes this a no-op.
  assert.equal(steppedWeight({ weight: 1 }, 1).weight, 1.1);
  assert.equal(steppedWeight({ weight: 1 }, -1).weight, 0.9);
  assert.equal(steppedWeight({ weight: 0 }, 1).weight, 0.1);
  assert.equal(steppedWeight({ weight: -1 }, -1).weight, -1.1);
  assert.equal(steppedWeight({ weight: -1 }, 1).weight, -0.9);
  // Off-grid values land on the neighbouring grid point, which is the point of
  // preferring the button over the text field.
  assert.equal(steppedWeight({ weight: 1.234 }, 1).weight, 1.3);
  assert.equal(steppedWeight({ weight: 1.234 }, -1).weight, 1.2);
  assert.equal(steppedWeight({ weight: -0.05 }, 1).weight, 0);
  assert.equal(steppedWeight({ weight: -0.05 }, -1).weight, -0.1);
  // Stepping past the range stops at the boundary instead of wrapping.
  assert.equal(steppedWeight({ weight: 5 }, 1).weight, WEIGHT_MAXIMUM);
  assert.equal(steppedWeight({ weight: -5 }, -1).weight, WEIGHT_MINIMUM);
  // Repeated clicks must not accumulate float drift.
  let weight = 0;
  for (let index = 0; index < 10; index += 1) weight = steppedWeight({ weight }, 1).weight;
  assert.equal(weight, 1);
});

test("the weight field accepts what a user has to type on the way to a number", () => {
  // These are the states that made the old input impossible to edit: it parsed
  // them as invalid, wrote the previous value back, and fought every keystroke.
  for (const partial of ["", " ", "-", ".", "-.", "1.", "-0."]) {
    assert.equal(parseWeightInput(partial).state, "incomplete", `${JSON.stringify(partial)} must be editable`);
  }
  assert.deepEqual(parseWeightInput("1.25"), { state: "valid", weight: 1.25 });
  assert.deepEqual(parseWeightInput("-2"), { state: "valid", weight: -2 });
  assert.deepEqual(parseWeightInput(".5"), { state: "valid", weight: 0.5 });
  // Out of range commits at the boundary rather than being rejected outright.
  assert.deepEqual(parseWeightInput("99"), { state: "valid", weight: WEIGHT_MAXIMUM });
  assert.equal(parseWeightInput("1.23456").weight, 1.2346);
  for (const junk of ["abc", "1e5", "--1", "1.2.3", "0x10", "1,5"]) {
    assert.equal(parseWeightInput(junk).state, "invalid", `${junk} is not a weight`);
  }
});

test("the precision hint names the gesture that last set the weight", () => {
  assert.equal(weightPrecisionLabel({ precision: 1 }, true), "精调 0.01");
  assert.equal(weightPrecisionLabel({ precision: 1 }, false), "步进 0.1");
  assert.equal(weightPrecisionLabel({ precision: 2 }, false), "百分位已锁定");
  assert.equal(weightPrecisionLabel({ precision: 4 }, false), "手动精度 0.0001");
});

test("every surface writes the weight that will actually be sent", async () => {
  // `precision` says how the weight was last set — tenths grid, hundredths
  // locked, hand-typed — which is a fact about the input, not the number.
  // Formatting with it showed 0.75 as "0.8" on the generate page while the
  // manager showed 0.75, and the rounding was not even consistent: `toFixed`
  // reads the binary value, so 0.25 rose and 0.15 fell.
  assert.equal(formatWeight(0.75), "0.75");
  assert.equal(formatWeight(0.15), "0.15");
  assert.equal(formatWeight(0.25), "0.25");
  assert.equal((0.25).toFixed(1), "0.3");
  assert.equal((0.15).toFixed(1), "0.1");

  // A whole number reads as one, and a stored value keeps every digit it has.
  assert.equal(formatWeight(1), "1");
  assert.equal(formatWeight(0), "0");
  assert.equal(formatWeight(-0), "0");
  assert.equal(formatWeight(-0.85), "-0.85");
  assert.equal(formatWeight(1.2345), "1.2345");
  // Representation noise is removed without inventing precision.
  assert.equal(formatWeight(0.30000000000000004), "0.3");
  assert.equal(formatWeight(0.1 + 0.2), "0.3");
  // The range is the same one every other weight path enforces.
  assert.equal(formatWeight(99), "5");
  assert.equal(formatWeight(-99), "-5");
  assert.equal(formatWeight("abc"), "0");

  // The formatter is the only way a weight reaches the screen, on every surface
  // that shows one — the manager's own field included, which is what makes the
  // generate page and the manager agree by construction rather than by review.
  const sources = await Promise.all(["src/App.jsx", "src/GalleryPage.jsx", "src/ImageToImagePage.jsx", "src/LoraMountPanel.jsx"]
    .map((path) => readFile(new URL(`../${path}`, import.meta.url), "utf8")));
  for (const [index, source] of sources.entries()) {
    assert.doesNotMatch(source, /toFixed\((?:lora|item)\.precision/, `surface ${index} must not format a weight by its input precision`);
    assert.match(source, /formatWeight\(/, `surface ${index} must render weights through the shared formatter`);
  }
  assert.match(sources[3], /const text = draft === null \? formatWeight\(item\.weight\) : draft;/);
});
