import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

// The stylesheet used to hold 884 distinct greys expressing about thirty tones, and every ordinary
// dialog picked its own backdrop, blur, radius, gutter and z-index. Nothing looked wrong on its
// own; nothing lined up either, and a new dialog had no value to copy that was obviously right.
//
// These tests hold the two layers that replaced that: a neutral ramp of value tokens, and a
// semantic contract for dialogs that resolves to different steps of the ramp per theme.

const styles = await readFile(new URL("../src/styles.css", import.meta.url), "utf8");

/** Declaration bodies, so a token *definition* is never mistaken for a use of a raw colour. */
function rulesOutsideRoot() {
  const rules = [];
  for (const match of styles.matchAll(/([^{}\n]+)\{([^{}]*)\}/g)) {
    const selector = match[1].trim();
    if (selector === ":root" || /^html\[data-theme-mode="[a-z]+"\]$/.test(selector)) continue;
    rules.push({ selector, body: match[2] });
  }
  return rules;
}

test("the neutral ramp is defined once, in order, and every step is used", () => {
  const root = styles.slice(styles.indexOf(":root {"), styles.indexOf("* { box-sizing"));
  const ramp = [...root.matchAll(/--n-(\d\d): (#[0-9a-f]{6});/g)];
  assert.ok(ramp.length >= 30, `expected a full ramp, found ${ramp.length} steps`);

  const luminance = (hex) => [1, 3, 5].reduce((total, index) => total + parseInt(hex.slice(index, index + 2), 16), 0) / 3;
  let previous = -1;
  for (const [, index, value] of ramp) {
    const current = luminance(value);
    assert.ok(current > previous, `--n-${index} (${value}) is not lighter than the step before it`);
    previous = current;
    assert.ok(styles.includes(`var(--n-${index})`), `--n-${index} is defined but never used`);
  }
});

test("no rule outside the token definitions reaches for a raw neutral grey", () => {
  // Chromatic colour is still written literally on purpose: the red error surfaces, the assistant
  // window's cool ramp and the accent tints are deliberate colour, not neutral.
  const isNeutral = (hex) => {
    const [r, g, b] = [1, 3, 5].map((index) => parseInt(hex.slice(index, index + 2), 16));
    const spread = Math.max(r, g, b) - Math.min(r, g, b);
    if (spread <= 3) return true;
    if (spread > 12) return false;
    const max = Math.max(r, g, b) / 255;
    const min = Math.min(r, g, b) / 255;
    const delta = max - min;
    let hue = 0;
    if (max === r / 255) hue = ((g - b) / 255 / delta) % 6;
    else if (max === g / 255) hue = (b - r) / 255 / delta + 2;
    else hue = (r - g) / 255 / delta + 4;
    hue = (hue * 60 + 360) % 360;
    return hue >= 45 && hue <= 150;
  };

  const offenders = [];
  for (const { selector, body } of rulesOutsideRoot()) {
    for (const [hex] of body.matchAll(/#[0-9a-f]{6}\b/gi)) {
      if (isNeutral(hex.toLowerCase())) offenders.push(`${selector.slice(0, 60)} -> ${hex}`);
    }
  }
  assert.deepEqual(offenders, [], "these greys belong on the ramp");
});

test("every ordinary dialog draws its backdrop from one contract", () => {
  const backdrops = [
    "key-manager-backdrop", "recommended-picker-backdrop", "recommended-zone-backdrop",
    "prompt-preset-backdrop", "lora-modal-backdrop", "lora-detail-backdrop", "settings-backdrop",
    "update-confirm-backdrop", "image-info-details-backdrop", "lora-card-editor-backdrop",
    "lora-picker-backdrop", "gallery-dialog-backdrop",
  ];
  for (const name of backdrops) {
    const rule = styles.match(new RegExp(`\\.${name}[^{]*\\{([^}]*position: fixed[^}]*)\\}`));
    assert.ok(rule, `${name} has no positioned rule`);
    assert.match(rule[1], /background: var\(--modal-backdrop\)/, `${name} sets its own backdrop colour`);
    assert.match(rule[1], /backdrop-filter: blur\(var\(--modal-blur\)\)/, `${name} sets its own blur`);
    assert.match(rule[1], /padding: var\(--modal-gutter\)/, `${name} sets its own gutter`);
  }
});

test("no ordinary dialog picks its own stacking number", () => {
  // Twenty-three hand-picked z-indexes between 40 and 365 meant the order two dialogs appeared in
  // was decided by whichever number someone happened to choose. The exceptions are by name, not by
  // number: each one builds a local stacking context inside a surface that is not an ordinary
  // dialog, so a number on the shared scale would mean nothing to it.
  const localStackingContexts = new Set([
    ".transparent-model-menu",              // anchored inside a workspace panel
    ".viewer-notice",                       // inside the image viewer
    ".viewer-context-menu",                 // inside the image viewer
    ".viewer-toolbar-popover-backdrop",     // inside the image viewer's toolbar
    ".gallery-focus > .gallery-dialog-backdrop", // dialogs inside the Gallery focus view
    ".lora-page-notice",                    // the standalone LoRA page, which has no dialogs
    ".lora-group-create-menu",              // anchored inside the LoRA group panel
  ]);
  const loose = [];
  for (const { selector, body } of rulesOutsideRoot()) {
    const found = body.match(/z-index:\s*(\d+)/);
    if (!found) continue;
    if (!/backdrop|dialog|modal|-menu|notice|drag-ghost|hover-preview/.test(selector)) continue;
    if (localStackingContexts.has(selector)) continue;
    loose.push(`${selector.slice(0, 60)} -> ${found[1]}`);
  }
  assert.deepEqual(loose, [], "these surfaces should be on the --z-* scale");
});

test("the dialog contract is redefined for light, so a dialog is legible without its own rule", () => {
  const light = styles.slice(styles.indexOf('html[data-theme-mode="light"] {\n  --modal-backdrop'));
  for (const token of ["--modal-backdrop", "--modal-surface", "--modal-header", "--modal-footer", "--modal-line", "--modal-shadow"]) {
    assert.ok(light.slice(0, 600).includes(token), `${token} has no light-theme value`);
  }
});

test("dialog surfaces share one border, radius and shadow", () => {
  for (const name of ["settings-modal", "lora-modal", "prompt-preset-dialog", "image-info-details-dialog", "lora-card-editor", "lora-picker"]) {
    const rule = styles.match(new RegExp(`\\.${name} \\{([^}]*width:[^}]*)\\}`));
    assert.ok(rule, `${name} has no sizing rule`);
    assert.match(rule[1], /border: 1px solid var\(--modal-line\)/, `${name} sets its own border colour`);
    assert.match(rule[1], /border-radius: var\(--modal-radius\)/, `${name} sets its own corner radius`);
    assert.match(rule[1], /box-shadow: var\(--modal-shadow\)/, `${name} sets its own shadow`);
  }
});

test("close buttons are one size everywhere", () => {
  // They were 38px, 35px and 34px with three different borders and three radii.
  const closers = [...styles.matchAll(/\.[a-z-]*(?:modal-close|dialog > header > button)[^{]*\{([^}]*width:[^}]*)\}/g)];
  assert.ok(closers.length >= 3, "the close buttons are not where this test expects them");
  for (const [, body] of closers) {
    assert.match(body, /width: var\(--modal-close-size\)/);
    assert.match(body, /height: var\(--modal-close-size\)/);
  }
});

test("the run diagnostics can be hidden, and the choice is project state", async () => {
  // Model, VRAM tier, cache state, tokenizer conditioning and the engine's sampling notice sit
  // between the preview and the generate button. They are worth reading while a run is being set
  // up and are in the way once it is settled, so whether they are on screen is the user's call.
  const app = await readFile(new URL("../src/App.jsx", import.meta.url), "utf8");
  assert.match(app, /aria-controls="generation-run-info"/);
  assert.match(app, /<div id="generation-run-info" hidden=\{!runInfoVisible\}>/);

  // The control lives with the other controls rather than in the title row. Put beside the title
  // it competed for width with the eyebrow and the heading, in a column narrow enough that all
  // three wrapped onto two lines each.
  assert.match(app, /className=\{`icon-button \$\{runInfoVisible \? "active" : ""\}`\}/);
  assert.doesNotMatch(app, /preview-toggle/);

  // Persisted through the workspace document, not browser storage, and shown by default.
  assert.match(app, /runInfoVisible: saved\.runInfoVisible !== false/);
  assert.match(app, /setRunInfoVisible\(workspace\.runInfoVisible\)/);
  assert.match(app, /size, samplingExpanded, runInfoVisible, hires/,
    "the preference is missing from the workspace snapshot, so it would not survive a reload");

  // `[hidden]` alone loses to any later `display` rule matching the same element.
  assert.match(styles, /#generation-run-info\[hidden\] \{ display: none; \}/);
});
