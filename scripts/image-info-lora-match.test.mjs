// Image reader: which of several same-named LoRA files a picture's LoRA is shown (and applied) as.
import assert from "node:assert/strict";
import test from "node:test";
import {
  interpretImageMetadata,
  matchModelName,
  modelDisplayName,
  modelFolderAffinity,
  modelFolderKey,
  publicMatch,
  settleAmbiguousMatch,
  settleByFolder,
} from "../src/image-metadata.js";
import { buildImageInfoApplyPlan, retargetPromptLora } from "../src/image-info-apply.js";

function local(relative, engine = "iL") {
  const path = relative.toLowerCase();
  const base = path.split("/").pop();
  return { origin: "local", path, relative, display: modelDisplayName(relative), base, stem: base.replace(/\.[^.]+$/, ""), label: engine, value: relative, engine, engines: [] };
}

function shared(root, relative, engines = ["iL"]) {
  const entry = local(relative, "");
  return { ...entry, origin: "shared", value: `shared:${root}/${relative}`, engines };
}

const SET = ["shou_ad_xl_half_v1", "illustrious_flat_color_v2", "transparent-watercolor_IL_MIX", "matcha", "MomoSweetSpaceILL", "Amashiro Natsuki [style]Remake-Illus", "CUTE2712_Illustrious_v1.0"];

// This machine's layout: the whole set in Style_九V6.0, two of its files copied into Style_九V7.0,
// and one more copy in a shared ComfyUI folder.
const CATALOG = [
  ...SET.map((name) => local(`style/Style_九V6.0/${name}.safetensors`)),
  local("style/Style_九V7.0/shou_ad_xl_half_v1.safetensors"),
  local("style/Style_九V7.0/Amashiro Natsuki [style]Remake-Illus.safetensors"),
  shared("5738b73b9323", "Amashiro Natsuki [style]Remake-Illus.safetensors"),
];

// What the control plane does for a picture's recorded LoRA names (no hash in these fixtures).
function resolveSet(names, catalog) {
  const first = names.map((name) => matchModelName(name, catalog, { engine: "iL", exactBareNames: false }));
  const affinity = modelFolderAffinity(first);
  return first.map((match) => publicMatch(settleByFolder(match, affinity)));
}

test("same-named LoRAs settle into the folder the picture's set lives in", () => {
  const matches = resolveSet(SET.map((name) => `${name}.safetensors`), CATALOG);
  assert.deepEqual(matches.map((match) => match.status), SET.map(() => "local"));
  assert.ok(matches.every((match) => match.match.relative.startsWith("style/Style_九V6.0/")), matches.map((match) => match.match.relative));
  const shou = matches[0];
  assert.deepEqual([shou.chosenBy, shou.count, shou.precision, shou.display], ["folder", 2, "filename+folder", "Style_九V6.0\\shou_ad_xl_half_v1.safetensors"]);
  const amashiro = matches[5];
  assert.deepEqual([amashiro.chosenBy, amashiro.count], ["folder", 3]);
  // Unique names are untouched by the rule.
  assert.equal(matches[1].chosenBy, undefined);
  // The shared copy sits at its root, so its relative path equals the recorded name; a recorded name
  // is not an ID, so that copy still competes by folder instead of winning outright.
  assert.equal(matchModelName("Amashiro Natsuki [style]Remake-Illus.safetensors", CATALOG).status, "shared");
  // Nothing private reaches a browser.
  assert.ok(matches.every((match) => !("pending" in match)));
});

test("outside the home folder the nearest copy wins; a name with no file stays missing", () => {
  const catalog = [
    ...CATALOG,
    local("style/Style_九V7.0/extra.safetensors"),
    local("character/extra.safetensors"),
    shared("5738b73b9323", "extra.safetensors"),
  ];
  const [, extra, gone] = resolveSet(["matcha.safetensors", "extra.safetensors", "gone.safetensors"], catalog);
  // Style_九V6.0 has no `extra`; Style_九V7.0 shares `style/` with it, the others share only the root.
  assert.deepEqual([extra.status, extra.match.relative, extra.chosenBy, extra.count], ["local", "style/Style_九V7.0/extra.safetensors", "nearest", 3]);
  assert.equal(gone.status, "missing");
});

test("the folder keys separate engine roots and shared roots", () => {
  assert.equal(modelFolderKey(local("style/Style_九V6.0/matcha.safetensors")), "local/il/style/style_九v6.0");
  assert.equal(modelFolderKey(shared("abc", "anime/matcha.safetensors")), "shared/abc/anime");
  // Two copies in different shared roots, no local set: ties go to the shorter, then the first path.
  const [only] = resolveSet(["solo.safetensors"], [shared("b", "x/solo.safetensors"), shared("a", "solo.safetensors")]);
  assert.deepEqual([only.status, only.match.value, only.chosenBy], ["shared", "shared:a/solo.safetensors", "folder"]);
});

test("a hash that settles nothing leaves the folder rule to choose, and says the hash disagreed", () => {
  const ambiguous = matchModelName("shou_ad_xl_half_v1", CATALOG, { engine: "iL" });
  const unsettled = { ...settleAmbiguousMatch(ambiguous, []), pending: ambiguous.pending };
  const affinity = modelFolderAffinity([matchModelName("matcha", CATALOG), ambiguous]);
  const settled = settleByFolder(unsettled, affinity);
  assert.deepEqual([settled.match.relative, settled.hashMismatch], ["style/Style_九V6.0/shou_ad_xl_half_v1.safetensors", true]);
  // Without its candidates a match cannot be chosen for and is returned as it was.
  assert.equal(settleByFolder(settleAmbiguousMatch(ambiguous, null), affinity).status, "ambiguous");
});

test("an older picture of this program recovers each LoRA's ID from its inline-LoRA audit", () => {
  const info = interpretImageMetadata({
    prompt: "<lora:matcha:0.4>, 1girl",
    negative_prompt: "",
    parameters: JSON.stringify({
      job_id: "j", engine: "iL",
      loras: [{ name: "shou_ad_xl_half_v1.safetensors", weight: 1 }, { name: "matcha.safetensors", weight: 0.4 }, { name: "nowhere.safetensors", weight: 1 }],
      inline_loras: [
        { id: "style/Style_九V7.0/shou_ad_xl_half_v1.safetensors", weight: 1, source: "manual", match: "manual", order: 0 },
        { id: "style/Style_九V6.0/matcha.safetensors", weight: 0.4, source: "positive", match: "stem", order: 1 },
      ],
    }),
  });
  assert.deepEqual(info.loras.map((item) => item.path || ""), ["style/Style_九V7.0/shou_ad_xl_half_v1.safetensors", "style/Style_九V6.0/matcha.safetensors", ""]);
  // The recorded ID is matched exactly, even where the folder rule would have chosen another copy.
  const direct = matchModelName(info.loras[0].path, CATALOG, { engine: "iL" });
  assert.deepEqual([direct.status, direct.precision, direct.match.relative], ["local", "exact", "style/Style_九V7.0/shou_ad_xl_half_v1.safetensors"]);
});

test("applying a tagged prompt names the chosen file in its tag, so it is the one that runs", () => {
  assert.equal(
    retargetPromptLora("<lora:shou_ad_xl_half_v1:1>, <LORA:Shou_ad_xl_half_v1:0.5>, <lora:matcha:0.4>", "shou_ad_xl_half_v1", "style/Style_九V6.0/shou_ad_xl_half_v1.safetensors"),
    "<lora:style/Style_九V6.0/shou_ad_xl_half_v1:1>, <LORA:style/Style_九V6.0/shou_ad_xl_half_v1:0.5>, <lora:matcha:0.4>",
  );
  // A shared reference cannot be written into a tag.
  assert.equal(retargetPromptLora("<lora:a:1>", "a", "shared:root/a.safetensors"), "<lora:a:1>");
  const [shou, matcha] = resolveSet(["shou_ad_xl_half_v1", "matcha"], CATALOG);
  const info = {
    status: "ok", source: "a1111", positive: "<lora:shou_ad_xl_half_v1:1>,<lora:matcha:0.4>, 1girl", negative: "", parameters: [], flags: [],
    loras: [{ name: "shou_ad_xl_half_v1", weight: 1, match: shou }, { name: "matcha", weight: 0.4, match: matcha }],
  };
  const plan = buildImageInfoApplyPlan(info, ["prompts", "loras"], "t2i", { engine: "iL" });
  assert.equal(plan.overlay.positive, "<lora:style/Style_九V6.0/shou_ad_xl_half_v1:1>,<lora:matcha:0.4>, 1girl");
  assert.ok(plan.notes.some((line) => /1 个同名 LoRA 标签已改写/.test(line)));
  // Taken without the prompt, the chosen file is mounted instead.
  const mounted = buildImageInfoApplyPlan(info, ["loras"], "t2i", { engine: "iL" });
  assert.deepEqual(mounted.overlay.loras.map((item) => item.value), ["style/Style_九V6.0/shou_ad_xl_half_v1.safetensors", "style/Style_九V6.0/matcha.safetensors"]);
});
