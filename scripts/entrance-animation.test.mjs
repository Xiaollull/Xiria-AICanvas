import assert from "node:assert/strict";
import test from "node:test";

import {
  countUp,
  EASE_OUT_CUBIC,
  EASE_OUT_QUART,
  easeOutCubic,
  enter,
  riseIn,
  segmentEnd,
} from "../src/entrance-animation.js";

const fakeNode = () => {
  const calls = [];
  return { calls, animate: (keyframes, options) => { calls.push({ keyframes, options }); return { cancel() { calls.push("cancelled"); } }; } };
};

test("a staggered run reports when its last element finishes", () => {
  // GSAP composed one timeline and read this off it; the replacement has to compute it, so the
  // arithmetic that positions every later run is worth pinning down.
  assert.equal(segmentEnd(0, 4, 0.42, 0.06), 0.42 + 0.18);
  assert.equal(segmentEnd(1.5, 1, 0.42, 0.06), 1.92, "a single element has no stagger to add");
  assert.equal(segmentEnd(2, 0, 0.42, 0.06), 2, "an empty run does not advance the clock");
});

test("the overview sequence keeps the overlaps the timeline had", () => {
  // Command row of 3, then cards .2s before it ends, then panels .25s before the cards end.
  const command = segmentEnd(0, 3, 0.42, 0.06);
  assert.equal(Number(command.toFixed(3)), 0.54);
  const cards = segmentEnd(command - 0.2, 4, 0.42, 0.05);
  assert.equal(Number(cards.toFixed(3)), 0.91);
  assert.equal(Number((cards - 0.25).toFixed(3)), 0.66, "panels start before the cards have finished");
});

test("each element is held at its start state through its own delay", () => {
  const nodes = [fakeNode(), fakeNode(), fakeNode()];
  enter(nodes, { keyframes: riseIn(14), duration: 0.42, stagger: 0.06, delay: 0.1 });
  const delays = nodes.map((node) => node.calls[0].options.delay);
  assert.deepEqual(delays, [100, 160, 220]);
  for (const node of nodes) {
    const { options, keyframes } = node.calls[0];
    assert.equal(options.duration, 420);
    // Without a backwards fill a staggered element paints in its final position and then jumps
    // back to animate in, which is exactly what the delay is there to avoid.
    assert.equal(options.fill, "backwards");
    assert.equal(options.easing, EASE_OUT_QUART);
    assert.deepEqual(keyframes[0], { opacity: 0, transform: "translateY(14px)" });
    assert.deepEqual(keyframes[1], { opacity: 1, transform: "translateY(0)" });
  }
});

test("a run whose cursor lands before zero still starts immediately", () => {
  // `cursor - .2` is negative when the preceding run is short; a negative delay would be ignored
  // by one engine and treated as a seek by another.
  const node = fakeNode();
  enter([node], { keyframes: riseIn(16), duration: 0.48, delay: -0.15 });
  assert.equal(node.calls[0].options.delay, 0);
});

test("the eased counter matches the curve it replaced", () => {
  assert.equal(easeOutCubic(0), 0);
  assert.equal(easeOutCubic(1), 1);
  assert.equal(easeOutCubic(0.5), 0.875);
  assert.equal(easeOutCubic(-1), 0, "a frame before the start does not read negative");
  assert.equal(easeOutCubic(2), 1, "a late frame does not overshoot the target");
  assert.equal(EASE_OUT_CUBIC, "cubic-bezier(.215,.61,.355,1)");
});

test("counters run to their rendered value and stop there", () => {
  const nodes = [{ dataset: { count: "1200" }, textContent: "" }, { dataset: { count: "7" }, textContent: "" }];
  const frames = [];
  let clock = 0;
  const stop = countUp(nodes, {
    duration: 900,
    now: () => clock,
    schedule: (callback) => { frames.push(callback); return frames.length; },
    cancel: () => frames.push("cancelled"),
  });
  clock = 450;
  frames[0](450);
  assert.equal(nodes[0].textContent, (Math.round(1200 * easeOutCubic(0.5))).toLocaleString());
  frames[1](900);
  assert.equal(nodes[0].textContent, "1,200");
  assert.equal(nodes[1].textContent, "7");
  stop();
  // Stopping mid-flight must not leave a half-counted number on screen: the text was written
  // around React, so nothing else will paint over it.
  assert.equal(nodes[0].textContent, "1,200");
});

test("an element without a countable value is left alone", () => {
  const nodes = [{ dataset: { count: "not a number" }, textContent: "keep me" }];
  const stop = countUp(nodes, { schedule: () => 1, cancel: () => {} });
  stop();
  assert.equal(nodes[0].textContent, "keep me");
});
