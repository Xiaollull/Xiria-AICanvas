// Entrance animations for the LoRA manager, on the Web Animations API.
//
// These replace a GSAP timeline. GSAP's Standard License is free of charge but is not free
// software and cannot be conveyed under the AGPL, so the dependency had to go; `element.animate`
// covers everything the timeline was doing except the numeric counters, which ease text rather
// than a style and get a rAF pass instead.
//
// The arithmetic lives here rather than inline in the component because it is the part that can
// silently drift: GSAP composed a timeline and these are independent animations that have to be
// given the delays that timeline would have produced.

// GSAP's power3 and power2 are quartic and cubic; these are the standard bezier equivalents of
// their `.out` variants.
export const EASE_OUT_QUART = "cubic-bezier(.165,.84,.44,1)";
export const EASE_OUT_CUBIC = "cubic-bezier(.215,.61,.355,1)";

/** When a staggered run of `count` elements has finished, in seconds from `start`. */
export function segmentEnd(start, count, duration, stagger) {
  if (count <= 0) return start;
  return start + duration + stagger * (count - 1);
}

/** GSAP's power2.out, for the one value that is eased by hand. */
export function easeOutCubic(progress) {
  const clamped = Math.min(1, Math.max(0, progress));
  return 1 - (1 - clamped) ** 3;
}

/**
 * Play one staggered entrance and return its animations.
 *
 * `element.animate` has no equivalent of GSAP's `from()`, so each entrance names both ends. The
 * `backwards` fill is what holds an element at its start state through its stagger delay —
 * without it a staggered element paints in its final position and then jumps back to animate in.
 * Nothing is written inline, so cancelling is all the cleanup there is.
 */
export function enter(nodes, { keyframes, duration, stagger = 0, delay = 0, easing = EASE_OUT_QUART }) {
  return Array.from(nodes, (node, index) => node.animate(keyframes, {
    duration: duration * 1000,
    delay: Math.max(0, delay + stagger * index) * 1000,
    easing,
    fill: "backwards",
  }));
}

export const riseIn = (distance) => [
  { opacity: 0, transform: `translateY(${distance}px)` },
  { opacity: 1, transform: "translateY(0)" },
];

/**
 * Count every `[data-count]` element up to its rendered value, and return a stop function.
 *
 * One rAF pass drives the whole page rather than one per element. Stopping writes the final
 * value: a half-counted number left on screen would outlive the animation, because the text was
 * written around React and React has no reason to paint over it.
 */
export function countUp(nodes, { duration = 900, now = () => performance.now(), schedule = requestAnimationFrame, cancel = cancelAnimationFrame } = {}) {
  const targets = Array.from(nodes, (node) => ({ node, value: Number(node.dataset?.count) }))
    .filter(({ value }) => Number.isFinite(value));
  if (!targets.length) return () => {};
  const started = now();
  const paint = (fraction) => {
    for (const { node, value } of targets) node.textContent = Math.round(value * fraction).toLocaleString();
  };
  let frame = schedule(function step(timestamp) {
    const progress = duration > 0 ? (timestamp - started) / duration : 1;
    paint(easeOutCubic(progress));
    if (progress < 1) frame = schedule(step);
  });
  return () => {
    cancel(frame);
    paint(1);
  };
}
