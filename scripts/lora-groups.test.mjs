import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  MAXIMUM_GROUPS,
  applyGroupEnabled,
  composeGroupPrompt,
  collectingGroupId,
  createLoraGroup,
  disableUnmountedGroups,
  emptyLoraGroupMap,
  enabledLoraGroups,
  groupIdOfMountedLora,
  loraGroupsForScope,
  mountedEntryForGroups,
  normalizeLoraGroupList,
  normalizeLoraGroupMap,
  partitionMountedByGroup,
  sameLoraGroups,
  setCollectingGroup,
  syncMountedIntoGroups,
  updateLoraGroupsForScope,
  withLoraGroupsForScope,
} from "../src/lora-groups.js";
import { normalizeMountedLoras } from "../src/lora-state.js";
import { nextMountedLoraRevision, withMountedLorasForScope } from "../src/lora-model-scope.js";

const readSource = (path) => readFile(new URL(`../${path}`, import.meta.url), "utf8");
const lora = (value, extra = {}) => ({ value, name: value, category: "character", weight: 1, precision: 1, enabled: true, ...extra });
const group = (id, extra = {}) => createLoraGroup({ id, name: id, members: [], ...extra });

test("a weight change no longer invalidates the directory scan", async () => {
  // This is the bug: the revision keyed on full entry equality, so every slider
  // tick restarted the scan effect, which reset the category tab to "character"
  // and tore the mounted list out from under the pointer mid-drag.
  const base = withMountedLorasForScope(undefined, "SD", [lora("a.safetensors"), lora("b.safetensors")]);
  const revision = 7;
  for (const [label, mutation] of [
    ["weight", [lora("a.safetensors", { weight: 0.4 }), lora("b.safetensors")]],
    ["precision", [lora("a.safetensors", { precision: 4 }), lora("b.safetensors")]],
    ["enabled", [lora("a.safetensors", { enabled: false }), lora("b.safetensors")]],
    ["order", [lora("b.safetensors"), lora("a.safetensors")]],
    ["group tag", [lora("a.safetensors", { groupId: "g-1" }), lora("b.safetensors")]],
  ]) {
    const next = withMountedLorasForScope(base, "SD", mutation);
    assert.equal(nextMountedLoraRevision(revision, base, next), revision, `${label} must not trigger a rescan`);
  }
  // What the scan is actually for — pruning a path that left the directory —
  // still invalidates it.
  const mounted = withMountedLorasForScope(base, "SD", [lora("a.safetensors")]);
  assert.equal(nextMountedLoraRevision(revision, base, mounted), revision + 1, "a mount set change still rescans");
  const added = withMountedLorasForScope(base, "SD", [lora("a.safetensors"), lora("b.safetensors"), lora("c.safetensors")]);
  assert.equal(nextMountedLoraRevision(revision, base, added), revision + 1);
  // A change in another engine's list counts too; the scan covers all three.
  assert.equal(nextMountedLoraRevision(revision, base, withMountedLorasForScope(base, "iL", [lora("x.safetensors")])), revision + 1);
});

test("the mounted entry schema carries a group tag without disturbing untagged entries", () => {
  const [tagged] = normalizeMountedLoras([lora("a", { groupId: "g-1" })]);
  assert.equal(tagged.groupId, "g-1");
  // An untagged entry must serialise exactly as before, or every existing
  // workspace would look "changed" on first read and trigger a spurious write.
  const [plain] = normalizeMountedLoras([lora("a")]);
  assert.ok(!("groupId" in plain), "an ungrouped mount gains no field");
  // A tag that cannot be a group id degrades to a standalone mount rather than
  // a dangling reference.
  for (const junk of ["../escape", "G-1", "a".repeat(65), 5, null, {}]) {
    assert.ok(!("groupId" in normalizeMountedLoras([lora("a", { groupId: junk })])[0]), `${JSON.stringify(junk)} is not a group id`);
  }
});

test("a damaged group file is reset rather than treated as fatal", () => {
  // Unlike the mounted map, groups are a convenience layered over a library that
  // works without them, so generation must never be blocked by a bad group file.
  for (const junk of ["nope", 5, [], { schemaVersion: 99, byEngine: {} }, { byEngine: {} }, { schemaVersion: 1 }]) {
    const result = normalizeLoraGroupMap(junk);
    assert.deepEqual(result.container, emptyLoraGroupMap());
    assert.ok(result.warning, "the user is told the groups were reset");
  }
  assert.equal(normalizeLoraGroupMap(undefined, { fieldMissing: true }).warning, "", "a first run is not an error");

  // Individual bad groups are isolated; the readable ones survive.
  const mixed = normalizeLoraGroupMap({
    schemaVersion: 1,
    byEngine: { SD: [group("g-1"), { id: "" }, null, group("g-1")], iL: [], Anima: [] },
  });
  assert.equal(mixed.container.byEngine.SD.length, 1, "duplicate and malformed groups are dropped");
  assert.match(mixed.warning, /已隔离/);
});

test("groups are engine scoped, so an SD combination cannot be enabled under Anima", () => {
  const map = withLoraGroupsForScope(emptyLoraGroupMap(), "SD", [group("g-1", { members: [lora("sd.safetensors")] })]);
  assert.equal(loraGroupsForScope(map, "SD").length, 1);
  assert.equal(loraGroupsForScope(map, "Anima").length, 0);
  assert.equal(loraGroupsForScope(map, "nonsense").length, 0);
  const updated = updateLoraGroupsForScope(map, "SD", (current) => current.map((entry) => ({ ...entry, name: "renamed" })));
  assert.equal(updated.changed, true);
  assert.equal(loraGroupsForScope(updated.container, "SD")[0].name, "renamed");
  assert.equal(loraGroupsForScope(updated.container, "iL").length, 0, "other engines are untouched");
  assert.equal(updateLoraGroupsForScope(map, "SD", (current) => current).changed, false, "a no-op is not an edit");
  assert.ok(sameLoraGroups(loraGroupsForScope(map, "SD"), loraGroupsForScope(map, "SD")));
});

test("enabled groups prepend their preset prompt, in order, ahead of the user's text", () => {
  const groups = [
    group("g-1", { enabled: true, presetPrompt: "masterpiece, best quality" }),
    group("g-2", { enabled: false, presetPrompt: "should not appear" }),
    group("g-3", { enabled: true, presetPrompt: "cinematic lighting" }),
  ];
  assert.equal(composeGroupPrompt(groups, "a cat"), "masterpiece, best quality, cinematic lighting, a cat");
  assert.equal(enabledLoraGroups(groups).length, 2);
  // A preset that already ends in a comma must not produce ", ,".
  assert.equal(composeGroupPrompt([group("g-1", { enabled: true, presetPrompt: "solo, " })], ", a cat"), "solo, a cat");
  // Every empty combination degrades to exactly the user's own prompt.
  assert.equal(composeGroupPrompt([], "a cat"), "a cat");
  assert.equal(composeGroupPrompt([group("g-1", { enabled: true })], "a cat"), "a cat");
  assert.equal(composeGroupPrompt(groups, ""), "masterpiece, best quality, cinematic lighting");
  assert.equal(composeGroupPrompt(null, "  a cat  "), "a cat");
});

test("enabling a group mounts its members and disabling removes exactly those", () => {
  const combination = group("g-1", { members: [lora("a"), lora("b", { weight: 0.6 })] });
  const standalone = [lora("z")];

  const on = applyGroupEnabled(standalone, combination, true);
  assert.equal(on.changed, true);
  assert.deepEqual(on.loras.map((item) => item.value), ["z", "a", "b"]);
  assert.equal(on.loras.find((item) => item.value === "b").weight, 0.6, "member weights come from the group");
  assert.equal(on.loras.find((item) => item.value === "z").groupId, undefined, "a standalone mount is not adopted");

  const off = applyGroupEnabled(on.loras, combination, false);
  assert.deepEqual(off.loras.map((item) => item.value), ["z"], "only the group's own entries leave");

  // A path already mounted standalone is adopted rather than duplicated: the
  // mounted list is keyed by path and a second copy would be dropped silently.
  const overlapping = applyGroupEnabled([lora("a", { weight: 2 })], combination, true);
  assert.deepEqual(overlapping.loras.map((item) => item.value), ["a", "b"]);
  assert.equal(overlapping.loras.filter((item) => item.value === "a").length, 1);

  // The 16-entry mount cap is refused rather than silently truncated.
  const full = Array.from({ length: 12 }, (_, index) => lora(`m${index}`));
  const big = group("g-2", { members: Array.from({ length: 8 }, (_, index) => lora(`n${index}`)) });
  const overflow = applyGroupEnabled(full, big, true);
  assert.equal(overflow.overflow, true);
  assert.equal(overflow.changed, false);
  assert.deepEqual(overflow.loras.map((item) => item.value), full.map((item) => item.value), "nothing is mounted on refusal");
});

test("the mounted queue splits into group blocks plus the standalone remainder", () => {
  const groups = [group("g-1", { enabled: true }), group("g-2", { enabled: true })];
  const mounted = [
    lora("a", { groupId: "g-1" }),
    lora("loose"),
    lora("b", { groupId: "g-2" }),
    lora("c", { groupId: "g-1" }),
    lora("orphan", { groupId: "deleted-group" }),
  ];
  const { blocks, standalone } = partitionMountedByGroup(mounted, groups);
  assert.deepEqual(blocks.map((block) => block.group.id), ["g-1", "g-2"]);
  assert.deepEqual(blocks[0].items.map((item) => item.value), ["a", "c"], "mounted order is preserved inside a block");
  // A tag pointing at a deleted group is shown as standalone, not hidden.
  assert.deepEqual(standalone.map((item) => item.value), ["loose", "orphan"]);
  assert.equal(groupIdOfMountedLora(lora("orphan", { groupId: "deleted-group" }), groups), "");
  // A group with no mounted member renders no empty block.
  assert.equal(partitionMountedByGroup([lora("loose")], groups).blocks.length, 0);
});

test("editing a grouped LoRA writes back to its group; standalone mounts do not", () => {
  const groups = [
    group("g-1", { enabled: true, members: [lora("a"), lora("b")] }),
    group("g-2", { enabled: false, members: [lora("kept")] }),
  ];
  const edited = [
    lora("a", { groupId: "g-1", weight: 0.35, precision: 4 }),
    lora("b", { groupId: "g-1", enabled: false }),
    lora("loose", { weight: 2 }),
  ];
  const [first, second] = syncMountedIntoGroups(groups, edited);
  assert.equal(first.members.find((member) => member.value === "a").weight, 0.35);
  assert.equal(first.members.find((member) => member.value === "b").enabled, false);
  assert.ok(!first.members.some((member) => member.value === "loose"), "a standalone mount never joins a group");
  assert.ok(!first.members.some((member) => member.groupId), "stored members carry no redundant tag");

  // Unmounting a grouped LoRA removes it from the group, which is what the
  // mounted area being a live view of the group means.
  const afterUnmount = syncMountedIntoGroups(groups, [lora("a", { groupId: "g-1" })]);
  assert.deepEqual(afterUnmount[0].members.map((member) => member.value), ["a"]);

  // A disabled group has no mounted representation at all, so an empty bucket
  // must not be read as "the user emptied it".
  assert.deepEqual(second.members.map((member) => member.value), ["kept"], "a disabled group is never emptied by a sync");
});

test("group storage is bounded and names are always usable", () => {
  const many = Array.from({ length: MAXIMUM_GROUPS + 5 }, (_, index) => group(`g-${index}`));
  assert.equal(normalizeLoraGroupMap({ schemaVersion: 1, byEngine: { SD: many, iL: [], Anima: [] } }).container.byEngine.SD.length, MAXIMUM_GROUPS);
  assert.equal(createLoraGroup({ name: "   " }, 3).name, "LoRA 组 4", "a blank name falls back to a numbered one");
  assert.equal(createLoraGroup({ name: "  spaced   out  " }).name, "spaced out");
  assert.equal(createLoraGroup({ name: "x".repeat(200) }).name.length, 60);
  assert.equal(createLoraGroup({ presetPrompt: "a\n\n\n\n\nb" }).presetPrompt, "a\n\nb");
  assert.equal(createLoraGroup({ presetPrompt: "y".repeat(5000) }).presetPrompt.length, 2000);
  // Members reuse the mounted-entry normalizer, so a path that could not be
  // mounted can never be stored in a group either.
  assert.equal(createLoraGroup({ members: [{ value: "" }, { value: "ok" }] }).members.length, 1);
  assert.ok(createLoraGroup({}).id.startsWith("g-"));
});

test("the preset prompt reaches the request body and never the prompt box", async () => {
  const app = await readSource("src/App.jsx");
  // The request body carries the composed prompt; the groups are resolved once
  // per generation and reused for the gallery record.
  assert.match(app, /prompt: composeGroupPrompt\(generationGroups, positive\)/);
  // Rewriting the box, or storing the composed text as the generated settings,
  // would prepend the presets a second time when the item is restored.
  assert.ok(!/setPositive\([^)]*composeGroupPrompt/.test(app), "the prompt box is never rewritten");
  // Checked per record rather than file-wide: the image-to-image request body legitimately composes
  // onto `settings.positive` on its way to the server, and only the stored record must stay clean.
  const records = [...app.matchAll(/setGeneratedSettings\(JSON\.parse\(JSON\.stringify\(\{[\s\S]*?\}\)\)\);/g)].map((match) => match[0]);
  assert.equal(records.length, 2, "text-to-image and image-to-image each freeze one record");
  for (const record of records) {
    assert.ok(!/positive: composeGroupPrompt/.test(record), "generated settings keep the user's own text");
  }
  assert.match(app, /loraGroupsByEngine,/, "groups are part of the persisted workspace");
  // Group definitions describe the workspace, not one image. Copying them into
  // every gallery record would bloat it and let a restore silently rewrite
  // combinations the user never asked to change.
  assert.match(app, /loraGroupsByEngine: _loraGroupsByEngine, \.\.\.gallerySettings/);
  assert.match(app, /delete source\.loraGroupsByEngine;/);
});

test("both mounted surfaces render one shared panel and one shared group editor", async () => {
  const [app, page, panel] = await Promise.all([
    readSource("src/App.jsx"),
    readSource("src/LoraManagerPage.jsx"),
    readSource("src/LoraMountPanel.jsx"),
  ]);
  for (const [name, source] of [["App.jsx", app], ["LoraManagerPage.jsx", page]]) {
    assert.match(source, /<LoraMountPanel[\s\S]*groups=\{activeLoraGroups\}/, `${name} passes groups to the shared panel`);
    assert.match(source, /<LoraGroupPanel/, `${name} exposes the group editor`);
    assert.match(source, /previewUrlFor=\{loraPreviewUrlFor\}/, `${name} supplies its own preview resolver`);
  }
  // Every mounted edit routes through one place, so a change to a grouped entry
  // always reaches its group definition.
  assert.match(panel, /const editMounted = /);
  assert.match(panel, /onUpdateGroups\?\.\(\(current\) => syncMountedIntoGroups\(current, next\)\)/);
  // The preview toggle is a view preference and defaults to off.
  assert.match(panel, /getItem\(PREVIEW_STORAGE_KEY\) === "on"/);
  assert.match(panel, /无图片/);
  // The preset box in the mounted area starts collapsed.
  assert.match(panel, /function GroupPromptBox[\s\S]*useState\(false\)/);
});

test("a group can collect what is mounted next, and only one can at a time", () => {
  const groups = [group("g-1"), group("g-2")];
  assert.equal(collectingGroupId(groups), "", "nothing collects until asked");

  const collecting = setCollectingGroup(groups, "g-2");
  assert.equal(collectingGroupId(collecting), "g-2");
  // A collector must be mounted for its block to exist, so pointing at one
  // enables it rather than leaving an invisible target.
  assert.equal(collecting.find((entry) => entry.id === "g-2").enabled, true);
  assert.equal(collecting.find((entry) => entry.id === "g-1").collecting, false);

  // Two collectors would make "where does this mount go" ambiguous.
  const contested = normalizeLoraGroupList([
    { ...group("g-1"), enabled: true, collecting: true },
    { ...group("g-2"), enabled: true, collecting: true },
  ]).groups;
  assert.deepEqual(contested.map((entry) => entry.collecting), [true, false]);

  // A disabled group has no mounted block, so it cannot be the target.
  const disabled = normalizeLoraGroupList([{ ...group("g-1"), enabled: false, collecting: true }]).groups;
  assert.equal(disabled[0].collecting, false);

  assert.equal(collectingGroupId(setCollectingGroup(collecting, "")), "", "collecting can be switched off");
});

test("a new mount joins the collecting group, and lands standalone without one", () => {
  const entry = { value: "a.safetensors", name: "a", category: "style", weight: 1, precision: 1, enabled: true };
  assert.equal(mountedEntryForGroups(entry, []).groupId, undefined);
  assert.equal(mountedEntryForGroups(entry, [group("g-1")]).groupId, undefined, "an idle group does not claim mounts");
  const tagged = mountedEntryForGroups(entry, setCollectingGroup([group("g-1")], "g-1"));
  assert.equal(tagged.groupId, "g-1");
  // Tagging must not disturb anything else about the entry.
  assert.deepEqual({ ...tagged, groupId: undefined }, { ...entry, groupId: undefined });

  // Collecting, then mounting, then writing back is the whole loop: the group
  // definition ends up holding what the user actually mounted.
  const collecting = setCollectingGroup([group("g-1")], "g-1");
  const mounted = [mountedEntryForGroups(entry, collecting)];
  const [saved] = syncMountedIntoGroups(collecting, mounted);
  assert.deepEqual(saved.members.map((member) => member.value), ["a.safetensors"]);
  assert.equal(saved.collecting, true, "collecting survives the write-back");
});

test("group creation offers both intents instead of assuming one", async () => {
  const panel = await readSource("src/LoraGroupPanel.jsx");
  // The old single button silently took whatever happened to be mounted.
  assert.ok(!panel.includes("createFromMounted"), "the presumptuous single action is gone");
  assert.match(panel, /const createGroup = \(\{ fromMounted \}\)/);
  assert.match(panel, /createGroup\(\{ fromMounted: true \}\)/);
  assert.match(panel, /createGroup\(\{ fromMounted: false \}\)/);
  // Capturing produces a finished group; an empty one is a workbench.
  assert.match(panel, /enabled: fromMounted,\s*\n\s*collecting: !fromMounted,/);
  // Capture has to tag the entries that are already mounted, or the group would
  // be a detached copy that the next write-back empties.
  assert.match(panel, /values\.has\(item\.value\) \? \{ \.\.\.item, groupId: group\.id \} : item/);
  // The choice surface is dismissible by the two gestures users expect.
  assert.match(panel, /window\.addEventListener\("pointerdown", dismiss\)/);
  assert.match(panel, /event\.key === "Escape"/);

  // Both mount surfaces route new mounts through the collector, and both write
  // the result back — mounting from the library bypasses the mount panel.
  for (const name of ["src/App.jsx", "src/LoraManagerPage.jsx"]) {
    const source = await readSource(name);
    assert.match(source, /mountedEntryForGroups\(\{ \.\.\.(?:lora|item),/, `${name} must honour the collector`);
    assert.match(source, /if \(next\) (?:commitLoraGroups|updateLoraGroups)\(\(current\) => syncMountedIntoGroups\(current, next\)\)/,
      `${name} must write a collected mount back into its group`);
  }

  // An enabled collector with nothing in it yet still needs a visible block.
  const mount = await readSource("src/LoraMountPanel.jsx");
  assert.match(mount, /collecting && !partitioned\.blocks\.some/);
  assert.match(mount, /lora-mount-group-waiting/);
});

test("a gallery card records the combinations its image was generated with", async () => {
  const app = await readSource("src/App.jsx");
  // The card must be able to show the prompt that actually produced the image,
  // not only the part the user typed.
  assert.match(app, /const generationGroups = enabledLoraGroups\(loraGroupsForScope\(loraGroupsMapRef\.current, activeLoraScopeRef\.current\)\)/);
  assert.match(app, /loraGroups: generationGroups\.map\(\(group\) => \(\{ id: group\.id, name: group\.name, presetPrompt: group\.presetPrompt \}\)\)/);
  assert.match(app, /loraGroupPrompt: generationGroupPrompt/);
  // One composition feeds both the request and the record, so a card can never
  // disagree with what was submitted.
  assert.match(app, /prompt: composeGroupPrompt\(generationGroups, positive\)/);
  // Exactly five sites, so a sixth cannot appear unnoticed: each page's request body, the frozen
  // record beside it, and the workspace-built card's derived prefix.
  assert.equal([...app.matchAll(/composeGroupPrompt\(/g)].length, 5);
  assert.match(app, /loraGroupPrompt: composeGroupPrompt\(enabled, ""\)/);
  // The record is this run's facts; the group library still stays out.
  assert.match(app, /delete source\.loraGroupsByEngine;/);

  const core = await readSource("src/gallery-core.js");
  assert.match(core, /loraGroups: \[\],/);
  assert.match(core, /loraGroupPrompt: "",/);
  assert.match(core, /export function normalizeCardLoraGroups/);
  // A card can never carry a group library back in and rewrite saved combinations.
  assert.match(core, /delete normalized\.loraGroupsByEngine;/);
});

test("card detail shows the groups, and apply defaults to the prompt alone", async () => {
  const gallery = await readSource("src/GalleryPage.jsx");
  // The submitted prompt is reconstructed with the real join rule rather than a
  // second implementation that could drift from it.
  assert.match(gallery, /function effectivePrompt\(settings\)/);
  assert.match(gallery, /composeGroupPrompt\(\[\{ id: "recorded"/);
  assert.match(gallery, /label="实际提交的正向 Prompt"/);
  assert.match(gallery, /gallery-group-entry/);
  assert.match(gallery, /gallery-lora-group-tag/, "a grouped LoRA is identifiable in the flat list too");

  // Prompt-only default, and Prompt listed first.
  assert.match(gallery, /const APPLY_DEFAULT_GROUPS = \["prompts"\]/);
  assert.match(gallery, /useState\(\(\) => new Set\(APPLY_DEFAULT_GROUPS\)\)/);
  assert.match(gallery, /const APPLY_GROUPS = \[\s*\n\s*\["prompts"/, "the default should be the first choice offered");
  // Applying only the prompt cannot reproduce a card whose prompt had a group
  // prefix, so that is said rather than left to surprise the user.
  assert.match(gallery, /gallery-apply-hint/);
  assert.match(gallery, /card\.settings\?\.loraGroupPrompt && selected\.has\("prompts"\)/);

  const styles = await readFile(new URL("../src/styles.css", import.meta.url), "utf8");
  for (const selector of [".gallery-group-entry {", ".gallery-apply-hint {", ".gallery-prompt.composed {"]) {
    assert.ok(styles.includes(selector), `${selector} is missing`);
  }
});

test("card settings round-trip the group record, and omit it when nothing is grouped", async () => {
  // Runtime rather than a text contract: the first version of this feature
  // recorded the fields only on the generation path, and a source-text check
  // could not see that a card built from the workspace never got them.
  const { normalizedSettings, normalizeCardLoraGroups } = await import("../src/gallery-core.js");

  const grouped = normalizedSettings({
    positive: "a cat",
    loras: [{ value: "a.safetensors", name: "a", groupId: "g-1", weight: 0.8 }],
    loraGroups: [{ id: "g-1", name: "写实风", presetPrompt: "masterpiece, best quality" }],
    loraGroupPrompt: "masterpiece, best quality",
  });
  assert.deepEqual(grouped.loraGroups, [{ id: "g-1", name: "写实风", presetPrompt: "masterpiece, best quality" }]);
  assert.equal(grouped.loraGroupPrompt, "masterpiece, best quality");
  assert.equal(grouped.loras[0].groupId, "g-1", "the tag that links an entry to its group survives");

  // A standalone-only card carries no title and no preset, so the detail view's
  // guards hide both sections rather than rendering empty ones.
  const standalone = normalizedSettings({ positive: "a cat", loras: [{ value: "b.safetensors", name: "b" }] });
  assert.deepEqual(standalone.loraGroups, []);
  assert.equal(standalone.loraGroupPrompt, "");

  // A card can never carry the workspace group library back in.
  assert.equal(normalizedSettings({ loraGroupsByEngine: { schemaVersion: 1 } }).loraGroupsByEngine, undefined);

  // The record is bounded and self-healing.
  assert.deepEqual(normalizeCardLoraGroups("nope"), []);
  assert.deepEqual(normalizeCardLoraGroups([null, { id: "BAD" }, { id: "g-1" }, { id: "g-1", name: "dup" }]),
    [{ id: "g-1", name: "g-1", presetPrompt: "" }]);
  assert.equal(normalizeCardLoraGroups([{ id: "g-1", name: "  spaced   out  " }])[0].name, "spaced out");
  assert.equal(normalizeCardLoraGroups(Array.from({ length: 40 }, (_, index) => ({ id: `g-${index}` }))).length, 32);
});

test("every path that builds a card carries the group record", async () => {
  const app = await readSource("src/App.jsx");
  // A card can be built three ways — from a finished generation, from the card
  // editor, and from add-to-gallery before anything was generated. Recording the
  // fields only on the first left the other two blank.
  assert.match(app, /const galleryCardSettings = \(source, \{ record \} = \{\}\)/);
  assert.match(app, /currentSettings=\{galleryCardSettings\(workspaceSnapshot\.current\)\}/);
  assert.match(app, /settings=\{generatedSettings \? galleryCardSettings\(generatedSettings, \{ record: true \}\) : galleryCardSettings\(workspaceSnapshot\.current\)\}/,
    "a generated card keeps its own frozen record instead of today's groups");
  // No card-producing path may reach for the raw helper again.
  assert.equal([...app.matchAll(/gallerySettingsWithoutPromptPresets\(/g)].length, 5,
    "declaration, the two generation records (text-to-image and image-to-image), the one call inside galleryCardSettings, and the image-reader apply source — which builds an apply overlay, not a card");
  // Image-to-image shares the mounts, so it must also carry the combinations they belong to.
  const imageRun = app.slice(app.indexOf("const generateFromImage = async"), app.indexOf("const releaseLoadedModel"));
  assert.match(imageRun, /composeGroupPrompt\(runGroups, settings\.positive\)/,
    "an enabled group contributes its trigger words to an image-to-image run too");
  assert.match(imageRun, /loraGroupPrompt: composeGroupPrompt\(runGroups, ""\)/);
});

test("applying a new mounted list switches off the combinations it replaced", async () => {
  const style = createLoraGroup({
    id: "g-style", name: "画风组", enabled: true, presetPrompt: "kazutake style",
    members: [{ value: "Anima/Style/a.safetensors", weight: 0.75 }, { value: "Anima/Style/b.safetensors", weight: 0.3 }],
  });
  const off = createLoraGroup({ id: "g-off", name: "关闭的组", enabled: false, presetPrompt: "never", members: [{ value: "Anima/x.safetensors", weight: 1 }] });
  const groups = [style, off];
  const represented = [
    { value: "Anima/Style/a.safetensors", weight: 0.75, groupId: "g-style" },
    { value: "Anima/Style/b.safetensors", weight: 0.3, groupId: "g-style" },
  ];
  const applied = [{ value: "Anima/Style/c.safetensors", weight: 0.5 }];

  // While the group is still represented in the mounted list, nothing changes.
  const unchanged = disableUnmountedGroups(groups, represented);
  assert.equal(unchanged.changed, false);
  assert.equal(unchanged.groups[0].enabled, true);

  // An applied list replaces the mounted one outright, so the group is no
  // longer represented. Left on it would keep prefixing its preset prompt onto
  // every request while none of its LoRAs were loaded.
  const after = disableUnmountedGroups(groups, applied);
  assert.equal(after.changed, true);
  assert.equal(after.groups[0].enabled, false);
  assert.equal(composeGroupPrompt(groups, "a lantern"), "kazutake style, a lantern");
  assert.equal(composeGroupPrompt(after.groups, "a lantern"), "a lantern");
  // The member list is kept: switching the group back on is how it is restored.
  assert.equal(after.groups[0].members.length, 2);
  // Half a group is not a group — one surviving member still switches it off.
  assert.equal(disableUnmountedGroups(groups, [represented[0]]).groups[0].enabled, false);
  // A group that was already off is untouched, and an untagged mount of the
  // same file does not count as representation.
  assert.equal(after.groups[1].enabled, false);
  assert.equal(disableUnmountedGroups(groups, represented.map(({ groupId: _tag, ...item }) => item)).groups[0].enabled, false);
  assert.equal(disableUnmountedGroups([off], applied).changed, false);
  assert.deepEqual(disableUnmountedGroups(null, applied).groups, []);

  // The apply path has to run it, or the switch outlives what it describes.
  const app = await readSource("src/App.jsx");
  assert.match(app, /commitLoraGroups\(\(current\) => disableUnmountedGroups\(current, galleryTargetLoras\)\.groups, targetLoraScopeKey\)/);
  assert.match(app, /if \(selectedGroups\.has\("loras"\)\) \{[\s\S]{0,600}?disableUnmountedGroups/);
});
