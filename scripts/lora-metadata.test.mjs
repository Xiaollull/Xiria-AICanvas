import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { buildTriggerGroups, descriptionForLoraVersion, loraMetadataCacheValid, plainTextFromHtml, readLoraFileMetadata, summarizeSafetensorsMetadata } from "./lora-metadata.mjs";

test("LoRA detail cache is reused only for the current file identity and schema", () => {
  const fileStat = { size: 123456, mtimeMs: 987654.25 };
  const valid = { detailSchema: 1, triggerReviewSchema: 4, triggerReviewKind: "character", promptReview: { versionScope: "", versionScopeKind: "" }, previewSchema: 1, status: "found", fileSize: 123456, modifiedAt: 987654.25 };
  assert.equal(loraMetadataCacheValid(valid, fileStat, "character"), true);
  assert.equal(loraMetadataCacheValid({ ...valid, detailSchema: 0 }, fileStat), false);
  assert.equal(loraMetadataCacheValid({ ...valid, fileSize: 123457 }, fileStat), false);
  assert.equal(loraMetadataCacheValid({ ...valid, modifiedAt: 987655 }, fileStat), false);
  assert.equal(loraMetadataCacheValid({ ...valid, triggerReviewSchema: 0 }, fileStat), false);
  assert.equal(loraMetadataCacheValid({ ...valid, promptReview: {} }, fileStat), false);
  assert.equal(loraMetadataCacheValid({ ...valid, modelDescription: "当前 Prompt\n下为旧版tag\n旧 Prompt" }, fileStat), false);
  assert.equal(loraMetadataCacheValid({ ...valid, modelDescription: "当前 Prompt\n下为旧版tag\n旧 Prompt", versionIsLatest: true }, fileStat), true);
  assert.equal(loraMetadataCacheValid(valid, fileStat, "style"), false);
  assert.equal(loraMetadataCacheValid({ ...valid, previewSchema: 0 }, fileStat), false);
  assert.equal(loraMetadataCacheValid({ ...valid, previewSchema: 0, previewFile: "preview.webp" }, fileStat), true);
  assert.equal(loraMetadataCacheValid({ ...valid, previewSchema: 0, previewUrl: "https://image.civitai.com/example.webp" }, fileStat), true);
  assert.equal(loraMetadataCacheValid({ ...valid, previewSchema: 0, status: "not_found" }, fileStat), true);
});

test("Civitai HTML descriptions become bounded readable text", () => {
  assert.equal(plainTextFromHtml("<p>Hello &amp; world</p><ul><li>Outfit</li></ul><script>bad()</script>"), "Hello & world\n- Outfit");
});

test("Civitai character prompt explanations become labeled trigger groups", () => {
  const groups = buildTriggerGroups({
    trainedWords: ["deyui", "blue eyes"],
    description: "<p>人物基础词：deyui, 1girl, blue eyes</p><p>服装 1</p><p>school uniform, blue jacket</p><p>服装 2: white dress, ribbon</p>",
  });
  assert.deepEqual(groups.map((group) => group.label), ["人物基础词", "服装 1", "服装 2"]);
  assert.deepEqual(groups[1].words, ["school uniform", "blue jacket"]);
  assert.ok(groups[2].words.includes("white dress"));
});

test("feature prompts retain shared character words in every labeled group", () => {
  const groups = buildTriggerGroups({ description: "人物基础词: deyui, 1girl\n服装 1: deyui, school uniform\n服装 2: deyui, white dress" });
  assert.deepEqual(groups[1].words, ["deyui", "school uniform"]);
  assert.deepEqual(groups[2].words, ["deyui", "white dress"]);
});

test("prompt review removes announcements and keeps only labeled character and outfit prompts", () => {
  const groups = buildTriggerGroups({
    versionDescription: [
      "新建了一个QQ群，群号1072783590",
      "人物特征：shirobana, bangs, long hair, hair between eyes, red eyes, purple eyes, ahoge",
      "连衣裙",
      "hair ornament, shirobana daily, twintails, hair bow, dress, white dress, necklace, sandals",
      "浴衣: shirobana kimono, double bun, kimono, hair bun, wide sleeves, sash, obi",
      "具体发型可以调整，修改对应tag即可。",
      "推荐使用wainsfwillustrious进行生成，其他大模型效果可能不优秀。",
      "推荐使用插件adetailer中的面部修复，可以提高眼睛质量。",
      "5月22日更新，优化了炼图方式。",
    ].join("\n"),
  });
  assert.deepEqual(groups.map((group) => group.label), ["人物特征", "连衣裙", "浴衣"]);
  assert.ok(groups.every((group) => !group.words.join(" ").match(/QQ群|推荐|更新|adetailer/i)));
});

test("Detail and description are both reviewed and duplicate prompts collapse", () => {
  const groups = buildTriggerGroups({
    versionDescription: "Character: mieru, 1girl, purple eyes, purple hair, long hair\nSchool uniform: mieru school, cardigan, shirt, necktie, red skirt",
    modelDescription: "人物基础词：mieru, 1girl, purple eyes, purple hair, long hair\n睡衣：mieru nightwear, hood, shorts, hoodie, socks",
  });
  assert.deepEqual(groups.map((group) => group.label), ["Character", "School uniform", "睡衣"]);
});

test("unlabeled trainedWords prompts are classified and no prompt stays empty", () => {
  const groups = buildTriggerGroups({ trainedWords: [
    "maidena_angers, very long hair, grey hair, red eyes, bangs",
    "school uniform, black pantyhose, collared shirt, pleated skirt, red vest",
  ] });
  assert.deepEqual(groups.map((group) => group.label), ["人物基础词", "校服"]);
  assert.deepEqual(buildTriggerGroups({ description: "Join our Discord for updates and model recommendations." }), []);
});

test("multi-version Civitai descriptions use only the locally matched version and preserve author headings", () => {
  const modelDescription = [
    "V3.0 更新/update：",
    "Better outfits fidelity.",
    "V3 tags",
    "原作画风 (official art style)",
    "official style,",
    "人物特征标 (character tags)",
    "kuro,yellow hairclip,grey hair,pink hair,gradient hair,slit pupils,",
    "猫娘 (cat girl)",
    "cat girl,cat ears,cat tail,",
    "两种尾巴 (2 kinds of tails)",
    "two-tone tail,white tail,grey tail,",
    "two-tone tail,white tail,pink tail,",
    "私服 (regular outfit)",
    "official alternate costume,jingle bell,white shirt,white skirt,",
    "校服 (school uniform)",
    "school uniform,white shirt,black skirt,red bowtie,",
    "魔法少女 (magical girl)",
    "magical girl,purple bow,white shirt,black skirt,",
    "斗篷兜帽 (black cloak)",
    "hood up,black cloak,red scarf,hooded cloak,",
    "睡衣 (pajamas)",
    "hood up,grey hoodie,cat hood,paw print,",
    "V2.x tags",
    "角色特征标 (character)",
    "kuro,hairclip,ahoge,",
    "私服 (regular clothes)",
    "official outfit,crop top,white skirt,detached sleeves,",
    "V1.0 tags",
    "角色特征标",
    "Kuro,hair ornament,",
  ].join("\n");
  const groups = buildTriggerGroups({ versionName: "v3.0", modelDescription });
  assert.deepEqual(groups.map((group) => group.label), ["原作画风 (official art style)", "人物特征标 (character tags)", "猫娘 (cat girl)", "两种尾巴 (2 kinds of tails) 1", "两种尾巴 (2 kinds of tails) 2", "私服 (regular outfit)", "校服 (school uniform)", "魔法少女 (magical girl)", "斗篷兜帽 (black cloak)", "睡衣 (pajamas)"]);
  assert.deepEqual(groups[3].words, ["two-tone tail", "white tail", "grey tail"]);
  assert.ok(groups.every((group) => !group.words.includes("ahoge") && !group.words.includes("hair ornament")));
  const v21 = descriptionForLoraVersion(modelDescription, "v2.1");
  assert.equal(v21.versionScope, "v2.1");
  assert.match(v21.description, /kuro,hairclip,ahoge/);
  assert.doesNotMatch(v21.description, /yellow hairclip|hair ornament/);
  const versionGroups = buildTriggerGroups({ versionName: "v2.1", versionDescription: modelDescription });
  assert.ok(versionGroups.some((group) => group.words.includes("ahoge")));
  assert.ok(versionGroups.every((group) => !group.words.includes("yellow hairclip") && !group.words.includes("hair ornament")));
});

test("author-defined forms, ages and outfits keep their exact headings and latest-version boundary", () => {
  const modelDescription = [
    "人物特征：anzu, blonde hair, blue eyes, hair ornament, long hair, hairclip,",
    "校服：anzu school, red beret, yellow bowtie, black sleeves, school uniform, blue skirt, white thighhighs, loafers",
    "爱丽丝服：anzu alice, dress, white pantyhose, alice \\(alice in wonderland\\), apron, blue dress, mary janes",
    "运动服：anzu gym, buruma, gym uniform, ponytail, blue buruma, scrunchie, shirt",
    "*幼年：anzu child, thighhighs, short hair, mary janes, white thighhighs, child, overalls, green footwear",
    "幼年还可以把short hair替换为long hair,white headwear,child headwear来解锁另一种形态",
    "10岁形态：anzu age 10, petite, short hair, blue eyes",
    "成年/战斗形态：anzu adult, battle suit, armored boots, long hair",
    "觉醒形态：anzu awakened",
    "星祭礼装",
    "anzu star festival, layered robe, moon brooch, silver footwear",
    "Awakened phase",
    "anzu_awakened",
    "下为旧版tag，旧版应该也不错",
    "校服：anzu school, hat, very long hair, old uniform",
    "旧版幼年：anzu child, short hair, hat, old design",
  ].join("\n");
  const groups = buildTriggerGroups({ versionName: "v3.0", versionIsLatest: true, modelDescription });
  assert.deepEqual(groups.map((group) => group.label), ["人物特征", "校服", "爱丽丝服", "运动服", "幼年", "幼年（另一种形态）", "10岁形态", "成年/战斗形态", "觉醒形态", "星祭礼装", "Awakened phase"]);
  assert.ok(groups.find((group) => group.label === "爱丽丝服").words.includes("anzu alice"));
  assert.ok(groups.find((group) => group.label === "幼年（另一种形态）").words.includes("child headwear"));
  assert.ok(!groups.some((group) => group.words.includes("old uniform") || group.words.includes("old design")));
  const oldGroups = buildTriggerGroups({ versionName: "v2.0", versionIsLatest: false, modelDescription });
  assert.deepEqual(oldGroups.map((group) => group.label), ["校服", "旧版幼年"]);
  assert.ok(oldGroups.some((group) => group.words.includes("old uniform")));
  assert.ok(oldGroups.every((group) => !group.words.includes("anzu alice") && !group.words.includes("child headwear")));
});

test("style LoRA keeps one trigger group and never infers character or outfit groups from prose", () => {
  const groups = buildTriggerGroups({
    reviewKind: "style",
    trainedWords: ["mikzn", "mikozin_style"],
    versionDescription: "Reviewed and updated images to include halos, correct eye color prompts, etc",
    modelDescription: "Character: 1girl, long hair, blue eyes\nOutfit: school uniform, pleated skirt",
  });
  assert.deepEqual(groups, [{
    label: "触发词",
    words: ["mikzn", "mikozin_style"],
    source: "reviewed-style-triggers",
    sources: ["civitai-detail", "civitai-description", "civitai-trained-words"],
  }]);
});

test("style LoRA may collect multiple explicitly labeled triggers from Detail and description", () => {
  const groups = buildTriggerGroups({
    reviewKind: "style",
    versionDescription: "Trigger words: watercolor_style, soft_lineart",
    modelDescription: "触发词：pastel_style, ink_style\n服装：school uniform, skirt",
  });
  assert.deepEqual(groups[0].label, "触发词");
  assert.deepEqual(groups[0].words, ["watercolor_style", "soft_lineart", "pastel_style", "ink_style"]);
  assert.equal(groups.length, 1);
});

test("style LoRA accepts an unlabeled trigger-only Detail line but rejects section headings", () => {
  const groups = buildTriggerGroups({ reviewKind: "style", versionDescription: "Examples\nmikzn\nNotes\nReviewed images and corrected eye colors" });
  assert.deepEqual(groups[0].words, ["mikzn"]);
  assert.equal(groups.length, 1);
});

test("Safetensors metadata summary exposes training facts without raw tag-frequency payload", () => {
  const summary = summarizeSafetensorsMetadata({
    "modelspec.title": "Character One",
    "modelspec.author": "Artist",
    "modelspec.trigger_phrase": "char_one, red hair",
    ss_network_dim: "32",
    ss_network_alpha: "16",
    ss_num_train_images: "120",
    ss_tag_frequency: JSON.stringify({ set: { "red hair": 30, "blue eyes": 20 } }),
  });
  assert.equal(summary.title, "Character One");
  assert.equal(summary.network, "Rank 32 / Alpha 16");
  assert.deepEqual(summary.triggerWords, ["char_one", "red hair"]);
  assert.deepEqual(summary.topTags[0], { word: "red hair", count: 30 });
  assert.ok(!summary.fields.some((field) => field.key === "ss_tag_frequency"));
});

test("Safetensors metadata is parsed from the bounded JSON header only", async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), "xirai-lora-metadata-"));
  const modelPath = path.join(directory, "sample.safetensors");
  const header = Buffer.from(JSON.stringify({ __metadata__: { "modelspec.title": "Local title", ss_network_dim: "8" }, tensor: { dtype: "F16", shape: [1], data_offsets: [0, 2] } }));
  const length = Buffer.alloc(8);
  length.writeBigUInt64LE(BigInt(header.length));
  await writeFile(modelPath, Buffer.concat([length, header, Buffer.alloc(2)]));
  try {
    const metadata = await readLoraFileMetadata(modelPath);
    assert.equal(metadata.parsed, true);
    assert.equal(metadata.title, "Local title");
    assert.equal(metadata.rank, "8");
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
