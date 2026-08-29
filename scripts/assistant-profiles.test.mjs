import assert from "node:assert/strict";
import test from "node:test";

import {
  MAXIMUM_PROFILES,
  MAXIMUM_PROFILE_NAME,
  activateAssistantProfile,
  activeAssistantProfile,
  assistantProfileAt,
  canCreateAssistantProfile,
  canRemoveAssistantProfile,
  createAssistantProfile,
  defaultProfileName,
  duplicateAssistantProfile,
  normalizeAssistantProfileStore,
  redactAssistantProfileStore,
  removeAssistantProfile,
  uniqueProfileName,
  updateAssistantProfile,
  validProfileId,
} from "../src/assistant-profiles.js";

const SECRET = "sk-live-profile-secret-4417";

const v1File = Object.freeze({
  schemaVersion: 1,
  provider: "deepseek",
  baseUrl: "https://api.deepseek.com/v1",
  apiKey: SECRET,
  model: "deepseek-v4-pro",
  strength: 1.2,
  personaId: "prompt-architect",
  knowledgeIds: [],
});

const seeded = (count) => {
  let store = normalizeAssistantProfileStore(null);
  for (let index = 1; index < count; index += 1) {
    store = createAssistantProfile(store, { name: `配置 ${index}`, settings: { provider: "ollama", model: "qwen3:8b" } });
  }
  return store;
};

test("a v1 settings file reads as a single profile with nothing lost", () => {
  const store = normalizeAssistantProfileStore(v1File);
  assert.equal(store.schemaVersion, 2);
  assert.equal(store.profiles.length, 1);
  const [profile] = store.profiles;
  assert.equal(store.activeId, profile.id);
  assert.deepEqual(profile.settings, {
    schemaVersion: 1,
    provider: "deepseek",
    baseUrl: "https://api.deepseek.com/v1",
    apiKey: SECRET,
    model: "deepseek-v4-pro",
    strength: 1.2,
    personaId: "prompt-architect",
    knowledgeIds: [],
  });
  // Named from what it is, because a migrated profile has no name of its own to keep.
  assert.equal(profile.name, "DeepSeek · deepseek-v4-pro");
  assert.equal(activeAssistantProfile(store).settings.apiKey, SECRET);
});

test("ids are stable across reads, so a save addressed to a listed profile still lands", () => {
  // The settings page lists profiles and then edits one *by id*. An id minted freshly per read
  // would 404 every save made against a file that has not been rewritten yet — which is every
  // migrated file, every hand-edited one, and a fresh install with no file at all.
  for (const input of [null, {}, v1File, { profiles: [{ name: "无 id" }, { name: "也无 id" }] }]) {
    const first = normalizeAssistantProfileStore(input);
    const second = normalizeAssistantProfileStore(input);
    assert.deepEqual(first.profiles.map((profile) => profile.id), second.profiles.map((profile) => profile.id));
    assert.equal(first.activeId, second.activeId);
    for (const profile of first.profiles) assert.ok(validProfileId(profile.id), `${profile.id} must match the id shape`);
    assert.equal(new Set(first.profiles.map((profile) => profile.id)).size, first.profiles.length, "ids must be unique");
  }
});

test("a damaged store degrades to one usable profile rather than to an empty list", () => {
  for (const junk of [null, undefined, 5, "text", [], { profiles: [] }, { profiles: "no" }, { profiles: [null, 7] }]) {
    const store = normalizeAssistantProfileStore(junk);
    assert.ok(store.profiles.length >= 1, `${JSON.stringify(junk)} must still yield a profile`);
    assert.ok(store.profiles.some((profile) => profile.id === store.activeId), "the active id must point at a real profile");
    assert.equal(activeAssistantProfile(store).settings.schemaVersion, 1);
  }
});

test("an activeId pointing at nothing falls back to the first profile instead of leaving none live", () => {
  const store = normalizeAssistantProfileStore({
    activeId: "ffffffff-ffff-4fff-8fff-ffffffffffff",
    profiles: [{ name: "A", settings: v1File }, { name: "B", settings: v1File }],
  });
  assert.equal(store.activeId, store.profiles[0].id);
  assert.equal(activeAssistantProfile(store).name, "A");
});

test("a duplicated id is re-identified rather than dropped, because it still holds a configuration", () => {
  const shared = "3f4a2b1c-1111-2222-3333-444455556666";
  const store = normalizeAssistantProfileStore({
    profiles: [
      { id: shared, name: "第一套", settings: v1File },
      { id: shared, name: "第二套", settings: { ...v1File, model: "deepseek-v4-flash" } },
    ],
  });
  assert.deepEqual(store.profiles.map((profile) => profile.name), ["第一套", "第二套"]);
  assert.notEqual(store.profiles[0].id, store.profiles[1].id);
  assert.equal(store.profiles[0].id, shared, "the first claimant keeps the id it was stored under");
});

test("the stored set is capped and creating past the cap is a no-op rather than a silent eviction", () => {
  const overfull = normalizeAssistantProfileStore({
    profiles: Array.from({ length: MAXIMUM_PROFILES + 6 }, (_unused, index) => ({ name: `P${index}`, settings: v1File })),
  });
  assert.equal(overfull.profiles.length, MAXIMUM_PROFILES);

  const full = seeded(MAXIMUM_PROFILES);
  assert.equal(full.profiles.length, MAXIMUM_PROFILES);
  assert.equal(canCreateAssistantProfile(full), false);
  const refused = createAssistantProfile(full, { name: "一套之外" });
  assert.equal(refused.profiles.length, MAXIMUM_PROFILES);
  assert.equal(refused.activeId, full.activeId, "a refused create must not move the selection");
  assert.equal(duplicateAssistantProfile(full, full.profiles[0].id).profiles.length, MAXIMUM_PROFILES);
});

test("a new profile is created active, uniquely named, and leaves the others untouched", () => {
  const before = normalizeAssistantProfileStore(v1File);
  const after = createAssistantProfile(before, { name: "DeepSeek · deepseek-v4-pro" });
  assert.equal(after.profiles.length, 2);
  assert.equal(after.activeId, after.profiles[1].id, "the profile you just made is the one you are about to configure");
  assert.equal(after.profiles[1].name, "DeepSeek · deepseek-v4-pro 2", "a colliding name is suffixed, not refused");
  assert.deepEqual(after.profiles[0], before.profiles[0], "an existing profile must be byte-identical after a create");
  assert.equal(after.profiles[1].settings.apiKey, "", "a new profile starts without a credential");
});

test("update patches only what it is given", () => {
  const store = normalizeAssistantProfileStore(v1File);
  const id = store.profiles[0].id;

  const renamed = updateAssistantProfile(store, id, { name: "  主力  配置 " });
  assert.equal(renamed.profiles[0].name, "主力 配置");
  assert.deepEqual(renamed.profiles[0].settings, store.profiles[0].settings, "a rename must not touch the settings");

  const resettled = updateAssistantProfile(store, id, { settings: { ...v1File, model: "deepseek-v4-flash" } });
  assert.equal(resettled.profiles[0].name, store.profiles[0].name, "a settings save must not rename the profile");
  assert.equal(resettled.profiles[0].model, undefined);
  assert.equal(resettled.profiles[0].settings.model, "deepseek-v4-flash");

  // A blank name is not a name: it falls back to the derived one rather than leaving an empty row.
  assert.equal(updateAssistantProfile(store, id, { name: "   " }).profiles[0].name, defaultProfileName(v1File));
  assert.deepEqual(updateAssistantProfile(store, "not-a-profile", { name: "x" }).profiles, store.profiles);
});

test("removing the active profile promotes a neighbour, and the last one cannot be removed", () => {
  const store = seeded(3);
  const [first, second, third] = store.profiles.map((profile) => profile.id);
  assert.equal(store.activeId, third);

  const withoutActive = removeAssistantProfile(store, third);
  assert.deepEqual(withoutActive.profiles.map((profile) => profile.id), [first, second]);
  assert.equal(withoutActive.activeId, second, "the selection steps back rather than emptying");

  const withoutOther = removeAssistantProfile(store, first);
  assert.equal(withoutOther.activeId, third, "removing a profile you were not using must not switch services");

  const single = normalizeAssistantProfileStore(v1File);
  assert.equal(canRemoveAssistantProfile(single), false);
  assert.deepEqual(removeAssistantProfile(single, single.profiles[0].id), single);
  assert.deepEqual(removeAssistantProfile(store, "not-a-profile"), store);
});

test("activation only ever points at a profile that exists", () => {
  const store = seeded(2);
  const first = store.profiles[0].id;
  assert.equal(activateAssistantProfile(store, first).activeId, first);
  assert.equal(activateAssistantProfile(store, "ffffffff-ffff-4fff-8fff-ffffffffffff").activeId, store.activeId);
  assert.equal(assistantProfileAt(store, first).id, first);
  assert.equal(assistantProfileAt(store, "nope"), null);
});

test("duplicating copies the credential, because the point is the same service with a different model", () => {
  const store = normalizeAssistantProfileStore(v1File);
  const copied = duplicateAssistantProfile(store, store.profiles[0].id);
  assert.equal(copied.profiles.length, 2);
  assert.equal(copied.profiles[1].settings.apiKey, SECRET);
  assert.notEqual(copied.profiles[1].id, copied.profiles[0].id);
  assert.equal(copied.profiles[1].name, `${copied.profiles[0].name} 2`);
  assert.equal(copied.activeId, copied.profiles[1].id);
  assert.deepEqual(duplicateAssistantProfile(store, "not-a-profile"), store);
});

test("redaction removes every key in the store, not only the active profile's", () => {
  let store = normalizeAssistantProfileStore(v1File);
  store = createAssistantProfile(store, { name: "第二套", settings: { ...v1File, apiKey: "sk-second-key-9931" } });
  const redacted = redactAssistantProfileStore(store);
  assert.equal(JSON.stringify(redacted).includes(SECRET), false);
  assert.equal(JSON.stringify(redacted).includes("sk-second-key-9931"), false);
  for (const profile of redacted.profiles) {
    assert.equal(profile.settings.apiKey, undefined);
    assert.equal(profile.settings.hasApiKey, true);
    assert.match(profile.settings.apiKeyHint, /^••••/);
  }
  assert.deepEqual(redacted.profiles.map((profile) => profile.active), [false, true]);
  assert.equal(redacted.activeId, store.activeId);
});

test("names are trimmed, capped and suffixed without ever exceeding the cap", () => {
  const long = "长".repeat(MAXIMUM_PROFILE_NAME + 20);
  const store = createAssistantProfile(normalizeAssistantProfileStore(v1File), { name: long });
  assert.equal(store.profiles[1].name.length, MAXIMUM_PROFILE_NAME);

  assert.equal(uniqueProfileName("A", []), "A");
  assert.equal(uniqueProfileName("A", ["A"]), "A 2");
  assert.equal(uniqueProfileName("A", ["A", "A 2"]), "A 3");
  assert.ok(uniqueProfileName(long, [long.slice(0, MAXIMUM_PROFILE_NAME)]).length <= MAXIMUM_PROFILE_NAME);
  assert.equal(uniqueProfileName("   ", []), "未命名配置");
  // A vendor with no model yet is still nameable; the model half only appears once chosen.
  assert.equal(defaultProfileName({ provider: "ollama", model: "" }), "Ollama (Local)");
});
