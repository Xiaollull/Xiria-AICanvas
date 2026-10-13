import { emptyEngineSettingsMap, engineSettingsDefaults, SETTINGS_ENGINES } from "../src/engine-settings.js";
import { emptyMountedLoraMap } from "../src/lora-model-scope.js";
import { emptyLoraGroupMap } from "../src/lora-groups.js";
import { seededPromptPresetContainer } from "../src/prompt-presets.js";
import { createXYZDraft } from "../src/xyz-draft.js";
import { createLoraMixDraft } from "../src/lora-mix-draft.js";

export function uiStateFixture() {
  const settings = emptyEngineSettingsMap(), mounts = emptyMountedLoraMap(), groups = emptyLoraGroupMap();
  for (const [i, engine] of SETTINGS_ENGINES.entries()) {
    const ref = { name: `Fixture ${engine}`, value: `${engine}-portrait.safetensors`, category: "other", weight: 0.8137, precision: 4, enabled: true,
      identitySchema: 1, instanceId: `00000000-0000-4000-8000-${String(i + 1).padStart(12, "0")}`, contentHash: String(i + 1).repeat(64), contentRevision: 1 };
    settings.byEngine[engine] = { ...engineSettingsDefaults(engine), positive: `SENTINEL ${engine} positive`, negative: `SENTINEL ${engine} negative`,
      checkpoint: `${engine}-checkpoint.safetensors`, diffusionModel: `${engine}-diffusion.safetensors`, textEncoder: `${engine}-encoder.safetensors`, textEncoder2: `${engine}-encoder2.safetensors`, vae: `${engine}-vae.safetensors`,
      size: { width: 768 + i * 64, height: 1344 }, seed: "18446744073709551615", seedMode: "fixed",
      imageToImage: { ...engineSettingsDefaults(engine).imageToImage, positive: `IMAGE ${engine} positive`, negative: `IMAGE ${engine} negative`, width: 1216, height: 1216, sizeMode: "custom" } };
    mounts.byEngine[engine] = [ref]; groups.byEngine[engine] = [{ id: `fixture-${engine.toLowerCase()}`, name: `Fixture group ${engine}`, enabled: false, collecting: false, presetPrompt: "fixture group words", members: [structuredClone(ref)] }];
  }
  const baseline = { parameters: { engine: "Anima", diffusion_model: "Anima-diffusion.safetensors", text_encoder: "Anima-encoder.safetensors", vae: "Anima-vae.safetensors",
    prompt: "XYZ SENTINEL", negative_prompt: "XYZ NEGATIVE", width: 768, height: 1024, steps: 28, cfg: 6.5, seed: "18446744073709551615", sampler: "euler", scheduler: "simple", guidance: "none", pag: { scale: 0.3, applied_layers: "mid" }, postprocess_chain: [] },
    loras: structuredClone(mounts.byEngine.Anima), group_snapshots: [], prompt_without_groups: "XYZ SENTINEL", group_prompt: "", seed_mode: "fixed", library_revision: 5 };
  const mix = createLoraMixDraft({ ...structuredClone(baseline), parameters: { ...baseline.parameters, prompt: "MIX SENTINEL" }, prompt_without_groups: "MIX SENTINEL" });
  return { workspace: { ...structuredClone(settings.byEngine.Anima), model: "Anima", engineSettingsByEngine: settings, mountedLorasByEngine: mounts, loraGroupsByEngine: groups,
    promptPresets: seededPromptPresetContainer(), canvasMaxEdge: 2048, generationMode: "basic", xyzByEngine: { schema_version: 1, by_engine: { Anima: createXYZDraft(baseline) } }, loraMixByEngine: { schema_version: 1, by_engine: { Anima: mix } } } };
}
