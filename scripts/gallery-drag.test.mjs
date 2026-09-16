import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import {
  galleryAutoScrollStep,
  galleryPointerIsCoarse,
  galleryPressIntent,
  GALLERY_AUTOSCROLL_EDGE,
  GALLERY_AUTOSCROLL_MAX_STEP,
  GALLERY_CLICK_SWALLOW_MS,
  GALLERY_DRAG_HOLD_MS,
  GALLERY_DRAG_MOUSE_THRESHOLD,
  GALLERY_DRAG_TOUCH_TOLERANCE,
  GALLERY_DROP_HEAD,
  GALLERY_DROP_TAIL,
  galleryDropAnchor,
  galleryOrderChanged,
  gallerySlotShifts,
  resolveDropIndex,
  spliceGalleryOrder,
} from "../src/gallery-drag.js";

test("a mouse drags as soon as it moves; it does not wait to be held", () => {
  // Waiting a third of a second before a mouse drag would start is what made desktop sorting feel
  // broken: the card sat still under a pointer that was already moving.
  assert.equal(galleryPressIntent({ pointerType: "mouse", distance: 0, elapsed: 0 }), "wait");
  assert.equal(galleryPressIntent({ pointerType: "mouse", distance: GALLERY_DRAG_MOUSE_THRESHOLD, elapsed: 0 }), "wait");
  assert.equal(galleryPressIntent({ pointerType: "mouse", distance: GALLERY_DRAG_MOUSE_THRESHOLD + 1, elapsed: 0 }), "drag");
  // A press that never moves stays a click however long it is held down.
  assert.equal(galleryPressIntent({ pointerType: "mouse", distance: 2, elapsed: 5000 }), "wait");
});

test("a finger must be held still first, and is released to the page if it wanders", () => {
  const touch = (distance, elapsed) => galleryPressIntent({ pointerType: "touch", distance, elapsed });
  assert.equal(touch(0, 0), "wait");
  assert.equal(touch(0, GALLERY_DRAG_HOLD_MS - 1), "wait");
  assert.equal(touch(0, GALLERY_DRAG_HOLD_MS), "drag");
  // A finger wobbles where a mouse does not, so small movement during the hold is tolerated.
  assert.equal(touch(GALLERY_DRAG_TOUCH_TOLERANCE, GALLERY_DRAG_HOLD_MS), "drag");
  // Past the tolerance the press is a scroll: the page gets it back rather than the card keeping it.
  assert.equal(touch(GALLERY_DRAG_TOUCH_TOLERANCE + 1, 10), "release");
  assert.equal(touch(60, GALLERY_DRAG_HOLD_MS), "release");
  // Once the drag is running, nothing takes it back.
  assert.equal(galleryPressIntent({ pointerType: "touch", distance: 400, elapsed: 9000, held: true }), "drag");
  assert.equal(galleryPointerIsCoarse("touch"), true);
  assert.equal(galleryPointerIsCoarse("pen"), true);
  assert.equal(galleryPointerIsCoarse("mouse"), false);
});

test("auto-scroll accelerates into the edge and stops away from it", () => {
  const bounds = { top: 100, bottom: 700 };
  assert.equal(galleryAutoScrollStep(400, bounds), 0, "the middle of the list does not scroll");
  // Deeper into the band is faster, and the direction follows the edge being approached.
  const shallow = galleryAutoScrollStep(100 + GALLERY_AUTOSCROLL_EDGE - 8, bounds);
  const deep = galleryAutoScrollStep(100, bounds);
  assert.ok(shallow < 0 && deep < 0);
  assert.ok(Math.abs(deep) > Math.abs(shallow));
  assert.equal(deep, -GALLERY_AUTOSCROLL_MAX_STEP);
  assert.equal(galleryAutoScrollStep(700, bounds), GALLERY_AUTOSCROLL_MAX_STEP);
  assert.ok(galleryAutoScrollStep(700 - GALLERY_AUTOSCROLL_EDGE + 8, bounds) > 0);
  // Past the edge entirely still scrolls, at the cap rather than faster.
  assert.equal(galleryAutoScrollStep(40, bounds), -GALLERY_AUTOSCROLL_MAX_STEP);
  assert.equal(galleryAutoScrollStep(NaN, bounds), 0);
  assert.equal(galleryAutoScrollStep(400, null), 0);
});

test("a card opened by a stray click is the bug this suppression exists for", async () => {
  const page = await readFile(new URL("../src/GalleryPage.jsx", import.meta.url), "utf8");
  const css = await readFile(new URL("../src/styles.css", import.meta.url), "utf8");
  // The click after a touch can lag the release; a timer shorter than that lag is what let the
  // card open instead of moving. The click consumes the flag itself, so it cannot expire early.
  assert.match(page, /if \(swallowClick\.current \|\| isDragging\) \{\s*\n\s*swallowClick\.current = false;/);
  assert.ok(GALLERY_CLICK_SWALLOW_MS > 300, "the swallow must outlive the touch click delay");
  assert.doesNotMatch(page, /suppressClick/, "the old 120 ms timer is gone");
  // Touch scrolling is what a finger does by default, so the browser only gives up the gesture once
  // the hold has decided otherwise -- in the stylesheet and, because that is read too early, in a
  // non-passive touchmove handler as well.
  assert.match(css, /\.gallery-card \{[^}]*touch-action: pan-y;/);
  assert.match(css, /\.gallery-card\.holding, \.gallery-card\.dragging \{ touch-action: none; \}/);
  assert.match(page, /node\.addEventListener\("touchmove", block, \{ passive: false \}\)/);
  assert.match(page, /const block = \(event\) => \{ if \(dragState\?\.cardId === card\.id\) event\.preventDefault\(\); \}/);
  // A finger that turns out to be scrolling must not leave a click behind either.
  assert.match(page, /intent === "release"[\s\S]{0,320}swallowNextClick\(\);\s*\n\s*endPress\(\);/);
  // The hold affordance is for a finger only; a mouse drags immediately and would flash it.
  assert.match(page, /const coarse = galleryPointerIsCoarse\(event\.pointerType\);\s*\n[\s\S]{0,140}setHolding\(coarse\);/);
  assert.match(page, /if \(coarse\) next\.timer = window\.setTimeout\(\(\) => handOver\(next\), GALLERY_DRAG_HOLD_MS\);/);
  // A collection with one card refuses the sort. It must be asked once, not once per pointer move.
  assert.match(page, /current\.declined = true;/);
  assert.match(page, /if \(gesture\.current !== current \|\| current\.declined\) return;/);
  // Auto-scroll runs on its own frames, so a finger held at the edge keeps scrolling, and it
  // re-measures afterwards because scrolling moved every card the drop point is computed against.
  assert.match(page, /window\.requestAnimationFrame\(runDragAutoScroll\)/);
  assert.match(page, /scroll\.scrollBy\(\{ top: step \}\);\s*\n[\s\S]{0,140}session\.laid = session\.measure\(\);/);
  assert.doesNotMatch(page, /scrollBy\(\{ top: -14 \}\)/, "the fixed per-event nudge is gone");
});

test("the stylesheet's hold animation lasts exactly as long as the hold", async () => {
  const css = await readFile(new URL("../src/styles.css", import.meta.url), "utf8");
  const match = css.match(/\.gallery-card\.holding::before \{ animation: gallery-hold-progress ([\d.]+)s/);
  assert.ok(match, "the hold progress animation must exist");
  assert.equal(Math.round(Number(match[1]) * 1000), GALLERY_DRAG_HOLD_MS, "a ring that finishes early or late lies about when the drag starts");
});

const card = (id, left, top, height = 80) => ({ id, rect: { left, top, width: 100, height } });
// Two columns, dealt round-robin: a c in the left column, b d in the right, as the grid lays out.
const twoColumns = [card("a", 0, 0), card("b", 120, 0), card("c", 0, 100), card("d", 120, 100)];

test("the drop point is read from the pointer the way a caret is read between characters", () => {
  // Above a card's middle lands before it; below lands before whatever follows it.
  assert.deepEqual(galleryDropAnchor(twoColumns, { x: 40, y: 10 }), { type: "before", cardId: "a" });
  assert.deepEqual(galleryDropAnchor(twoColumns, { x: 40, y: 60 }), { type: "before", cardId: "c" });
  assert.deepEqual(galleryDropAnchor(twoColumns, { x: 160, y: 10 }), { type: "before", cardId: "b" });
  // Past the bottom of a column is the position after its last card, in visual order.
  assert.deepEqual(galleryDropAnchor(twoColumns, { x: 40, y: 900 }), { type: "before", cardId: "d" });
  // Past the bottom of the last column is the tail of the list.
  assert.deepEqual(galleryDropAnchor(twoColumns, { x: 160, y: 900 }), { type: GALLERY_DROP_TAIL });
  // The nearest column wins even when the pointer is in the gutter between two of them.
  assert.equal(galleryDropAnchor(twoColumns, { x: 112, y: 10 }).cardId, "b", "just past the gutter midpoint the right column is nearer");
  assert.equal(galleryDropAnchor(twoColumns, { x: 108, y: 10 }).cardId, "a", "and just before it, the left one still is");
  assert.deepEqual(galleryDropAnchor([], { x: 0, y: 0 }), { type: GALLERY_DROP_TAIL });
  assert.deepEqual(galleryDropAnchor(twoColumns, null), { type: GALLERY_DROP_TAIL });
});

test("a single column reads top to bottom", () => {
  const column = [card("a", 0, 0), card("b", 0, 100), card("c", 0, 200)];
  assert.deepEqual(galleryDropAnchor(column, { x: 50, y: 10 }), { type: "before", cardId: "a" });
  assert.deepEqual(galleryDropAnchor(column, { x: 50, y: 150 }), { type: "before", cardId: "c" });
  assert.deepEqual(galleryDropAnchor(column, { x: 50, y: 400 }), { type: GALLERY_DROP_TAIL });
});

test("an anchor names a visible card, so a filtered view inserts where it looks like it will", () => {
  // The collection holds six; the view shows three. Dropping before the visible "e" must land
  // before "e" in the full order, leaving the hidden cards around it where they are.
  const full = ["a", "b", "c", "d", "e", "f"];
  assert.equal(resolveDropIndex(full, { type: "before", cardId: "e" }, "a"), 3);
  const landed = spliceGalleryOrder(full, "a", resolveDropIndex(full, { type: "before", cardId: "e" }, "a"));
  assert.deepEqual(landed, ["b", "c", "d", "a", "e", "f"]);
  assert.equal(resolveDropIndex(full, { type: GALLERY_DROP_HEAD }, "f"), 0);
  assert.equal(resolveDropIndex(full, { type: GALLERY_DROP_TAIL }, "a"), 5);
  // An anchor naming a card that has since gone falls to the tail rather than throwing.
  assert.equal(resolveDropIndex(full, { type: "before", cardId: "ghost" }, "a"), 5);
});

test("releasing where it started changes nothing and sends nothing", () => {
  const order = ["a", "b", "c", "d"];
  const index = resolveDropIndex(order, { type: "before", cardId: "b" }, "a");
  assert.deepEqual(spliceGalleryOrder(order, "a", index), order);
  assert.equal(galleryOrderChanged(order, spliceGalleryOrder(order, "a", index)), false);
  // Its own position and the one just after it are the same place once the card is lifted out.
  assert.equal(galleryOrderChanged(order, spliceGalleryOrder(order, "b", resolveDropIndex(order, { type: "before", cardId: "c" }, "b"))), false);
  assert.equal(galleryOrderChanged(order, spliceGalleryOrder(order, "a", 2)), true);
  assert.deepEqual(spliceGalleryOrder(order, "missing", 0), order);
  assert.deepEqual(spliceGalleryOrder(order, "a", 99), ["b", "c", "d", "a"]);
});

test("only the column that receives the card yields, and only below the drop point", () => {
  // Landing before "c" (left column, visual index 1 once "a" is lifted out): "c" slides down by one
  // card, "b" and "d" in the other column do not move.
  const shifts = gallerySlotShifts(twoColumns, { type: "before", cardId: "c" }, "a", 2);
  assert.deepEqual([...shifts.keys()], ["c"]);
  assert.equal(shifts.get("c"), 80);
  // Nothing yields for a drop at the tail: there is nothing after it to move.
  assert.equal(gallerySlotShifts(twoColumns, { type: GALLERY_DROP_TAIL }, "a", 2).size, 0);
  assert.equal(gallerySlotShifts([], { type: "before", cardId: "c" }, "a", 2).size, 0);
});

test("the list is frozen until release: a pass over a card cannot carry it away", async () => {
  const page = await readFile(new URL("../src/GalleryPage.jsx", import.meta.url), "utf8");
  // The grid is dealt into columns round-robin, so a reorder mid-drag moves the dragged card to a
  // different parent. React unmounts and remounts it, the browser releases its pointer capture with
  // the node, and the gesture is orphaned -- the drag never ends and the ghost never leaves. So the
  // order is touched exactly once, on release.
  const dragBlock = page.slice(page.indexOf("const startCardReorder ="), page.indexOf("const finishCardReorder ="));
  assert.doesNotMatch(dragBlock, /setData\(/, "nothing may reorder the data while the pointer is moving");
  assert.doesNotMatch(page, /moveCardDuringReorder/, "the hover-reorder path is gone");
  assert.match(page, /const updateDropAnchor = \(session\) => \{/);
  const finishAt = page.indexOf("const finishCardReorder =");
  const landing = page.slice(finishAt, page.indexOf("const saveCollection =", finishAt));
  assert.equal((landing.match(/applyCollectionCardOrder/g) || []).length, 2, "one splice applied, one rollback path");
  assert.match(landing, /if \(!galleryOrderChanged\(session\.order, next\)\) return;/, "a release in place sends no request");

  // The gesture is owned by the window, so nothing React does to the grid can take it away.
  for (const event of ["pointermove", "pointerup", "pointercancel", "blur", "keydown"]) {
    assert.match(page, new RegExp(`window\\.addEventListener\\("${event}"`), `${event} is listened for on the window`);
    assert.match(page, new RegExp(`window\\.removeEventListener\\("${event}"`), `${event} is removed again`);
  }
  assert.match(page, /event\.pointerId === dragSession\.current\.pointerId/, "another pointer cannot drive this drag");
  assert.match(page, /event\.key === "Escape"[\s\S]{0,80}cancelCardReorder\(\)/);
  // The card hands the gesture over rather than keeping it on a node that may be rebuilt.
  assert.match(page, /const handOver = \(current\) => \{[\s\S]{0,400}endPress\(\);/);

  // The ghost is written straight to the node once per frame instead of re-rendering the page.
  assert.match(page, /ghost\.style\.transform = `translate3d\(\$\{session\.x\}px, \$\{session\.y\}px, 0\)`/);
  assert.match(page, /createPortal\(<div ref=\{dragGhostRef\} className="gallery-drag-ghost"/);
  assert.doesNotMatch(page, /className="gallery-drag-ghost" style=\{\{ left:/, "the ghost is no longer positioned by a React render");
});

test("the dragged card stays put as a placeholder and the slot marks where it lands", async () => {
  const css = await readFile(new URL("../src/styles.css", import.meta.url), "utf8");
  // It used to shrink and rotate, which read as "this card is now elsewhere" while it had not moved.
  assert.match(css, /\.gallery-card\.dragging \{[^}]*transform: none;[^}]*pointer-events: none;/);
  assert.match(css, /\.gallery-card\.yielding \{ transform: translateY\(var\(--gallery-slot-shift, 0px\)\);/);
  assert.match(css, /\.gallery-card\.yielding::after \{[^}]*background: var\(--lime\);/);
  // The hover lift would fight the inline shift, so it is disabled while reordering.
  assert.match(css, /\.gallery-card-grid\.reordering \.gallery-card:hover \{ transform: translateY\(var\(--gallery-slot-shift, 0px\)\);/);
  // The old "replace this card" pulse is gone, keyframes included.
  assert.doesNotMatch(css, /gallery-drop-target/);
  // The slot must still open when motion is reduced, or there is no way to see the landing point.
  assert.match(css, /@media \(prefers-reduced-motion: reduce\) \{[\s\S]*\.gallery-card\.yielding \{ transition: none; \}/);
});

test("the back-to-top arrow appears only once the search box has gone behind the pinned nav", async () => {
  const [page, css] = await Promise.all([
    readFile(new URL("../src/GalleryPage.jsx", import.meta.url), "utf8"),
    readFile(new URL("../src/styles.css", import.meta.url), "utf8"),
  ]);
  // The nav is pinned, which is what lets the search box slide up behind it -- without that the two
  // scroll together and their edges never cross.
  assert.match(css, /\.gallery-room-library \{ position: sticky; top: 0;[^}]*background: var\(--gallery-background\);/);
  // The trigger is exactly that crossing, and the top of the list is never a place to go back from.
  assert.match(page, /if \(!nav \|\| !search \|\| room\.scrollTop <= 0\) return setBackToTop\(false\);/);
  assert.match(page, /setBackToTop\(search\.getBoundingClientRect\(\)\.bottom < nav\.getBoundingClientRect\(\)\.bottom\);/);
  // Measured on a frame rather than on every scroll event, and torn down with the page.
  assert.match(page, /room\.addEventListener\("scroll", schedule, \{ passive: true \}\)/);
  assert.match(page, /window\.addEventListener\("resize", schedule\)/);
  for (const listener of ["scroll", "resize"]) assert.match(page, new RegExp(`removeEventListener\\("${listener}", schedule\\)`));
  assert.match(page, /if \(frame\) window\.cancelAnimationFrame\(frame\)/);

  // It rides the nav's bottom edge, and it is absent rather than disabled when there is no journey.
  assert.match(page, /\{backToTop && <button/);
  assert.match(page, /className="gallery-room-top"[\s\S]{0,160}onClick=\{scrollGalleryToTop\}/);
  assert.match(page, /aria-label="回到顶部"/);
  assert.match(page, /galleryRoomRef\.current\?\.scrollTo\(\{ top: 0, behavior: "smooth" \}\)/);
  assert.match(css, /\.gallery-room-top \{ position: absolute; right: 0; bottom: 0;[^}]*translate: 0 50%;/);
  assert.match(css, /animation: gallery-top-enter [\d.]+s/);
  assert.match(css, /@keyframes gallery-top-enter \{ from \{ opacity: 0;/);

  // Both overrides have to follow the rule they override: one class deep each, so source order is
  // the whole of the cascade here.
  const base = css.indexOf(".gallery-room-top { position: absolute");
  assert.ok(css.indexOf(".gallery-room-top { animation: none; }") > base, "the reduced-motion override must come after the base rule");
  assert.ok(css.indexOf(".gallery-room-top { position: fixed") > base, "the narrow-screen override must come after the base rule");
  // A horizontally scrolling nav clips its own children, so on narrow screens the arrow leaves it.
  assert.match(css, /\.gallery-room-top \{ position: fixed; right: 16px; bottom: calc\(16px \+ env\(safe-area-inset-bottom, 0px\)\)/);
  // The narrow rule must not restate the animation: it sits after the reduced-motion override, so
  // naming an animation there would switch the entry movement back on for exactly the people who
  // asked for less of it.
  assert.doesNotMatch(css, /\.gallery-room-top \{ position: fixed[^}]*animation/);
  // One keyframe serves both positions because it animates opacity and scale only, never the
  // `translate` each breakpoint rests at.
  assert.doesNotMatch(css, /gallery-top-enter-fixed/);
  assert.match(css, /@keyframes gallery-top-enter \{ from \{ opacity: 0; scale: [\d.]+; \} to \{ opacity: 1; scale: 1; \} \}/);
});

test("the arrow is an accent control rather than an outline, so it reads at a glance", async () => {
  const [page, css] = await Promise.all([
    readFile(new URL("../src/GalleryPage.jsx", import.meta.url), "utf8"),
    readFile(new URL("../src/styles.css", import.meta.url), "utf8"),
  ]);
  const base = css.match(/\.gallery-room-top \{ position: absolute[^}]*\}/)[0];
  // Filled in the accent with a contrasting glyph: the previous outline circle in body ink sat at
  // the same weight as the collection buttons beside it.
  assert.match(base, /background: var\(--gallery-amber\)/);
  assert.match(base, /color: var\(--accent-contrast\)/);
  assert.match(base, /border: 0/);
  // The button straddles the nav's edge, so its lower half sits over whatever card is scrolling
  // underneath -- including a white photo, against which the accent fill is only about 1.95:1. The
  // edge is carried by a dark hairline with a light one outside it: whichever the backdrop
  // swallows, the other stands out. A translucent accent halo cannot do that job on white.
  assert.match(base, /box-shadow: 0 0 0 2px rgba\(9,10,12,\.92\), 0 0 0 3\.5px rgba\(255,255,255,\.72\)/);
  assert.doesNotMatch(base, /box-shadow: 0 0 0 4px rgb\(var\(--accent-rgb\)/, "the first ring must be opaque, not a translucent accent");
  const hover = css.match(/\.gallery-room-top:hover \{[^}]*\}/)[0];
  assert.match(hover, /0 0 0 2px rgba\(9,10,12,\.96\), 0 0 0 4px #fff/, "hover keeps both hairlines rather than dropping to a glow");
  const size = Number(base.match(/width: (\d+)px/)[1]);
  assert.ok(size >= 44, `the control is ${size}px; it was 34px and easy to miss`);
  assert.equal(size, Number(base.match(/height: (\d+)px/)[1]), "it stays a circle");
  // The glyph grows with the control instead of sitting small inside it.
  assert.match(page, /<ArrowUp size=\{20\} strokeWidth=\{2\.4\} \/>/);
  // Touch targets are bigger than pointer ones, and a phone has room for it.
  assert.ok(Number(css.match(/\.gallery-room-top \{ position: fixed[^}]*width: (\d+)px/)[1]) > size);
  // Focus is visible against a filled accent, which an accent-coloured outline would not be.
  assert.match(css, /\.gallery-room-top:focus-visible \{ outline: 2px solid var\(--gallery-ink\)/);
});
