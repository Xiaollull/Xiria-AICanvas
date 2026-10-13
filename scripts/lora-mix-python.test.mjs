import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { createLoraMixDraft, loraMixRecipe } from "../src/lora-mix-draft.js";
import { generateCandidates } from "../src/lora-mix-generate.js";
import { memberAnalysis, mixIssues } from "../src/lora-mix-model.js";
import { countSpace } from "../src/lora-mix-rules.js";

// QPlan2/P2: whatever the browser previews must be accepted by the authoritative Python validator, with the same
// vectors, bounds and exact space size. Skipped (not failed) where the project Python is not available.
const python = process.env.P10_PYTHON || "python", cwd = path.resolve(import.meta.dirname, "..");
const probe = spawnSync(python, ["-B", "-c", "import pydantic"], { cwd, encoding: "utf8", windowsHide: true });
const available = probe.status === 0;
const uuid = (n) => `00000000-0000-4000-8000-${String(n).padStart(12, "0")}`;
const hash = (n) => n.toString(16).padStart(2, "0").repeat(32);
const lora = (n, extra = {}) => ({ value: `style/l${n}.safetensors`, name: `L${n}`, category: "style", weight: 0.6, precision: 4, enabled: true, identitySchema: 1, instanceId: uuid(n), contentHash: hash(n), contentRevision: 2, ...extra });
const baseline = (loras = []) => ({ parameters: { engine: "Anima", diffusion_model: "d.safetensors", text_encoder: "t.safetensors", vae: "v.safetensors", prompt: "raw", negative_prompt: "", steps: 20, cfg: 4,
  width: 512, height: 512, seed: "7", sampler: "euler", scheduler: "normal", guidance: "none", pag: { scale: 0.3, applied_layers: "mid" }, postprocess_chain: [] },
  loras, group_snapshots: [], prompt_without_groups: "raw", group_prompt: "", seed_mode: "fixed", library_revision: 0 });
const rule = (base, min, max, step = "0.1", preference = "balanced") => ({ locked: false, base, min, max, step, preference });
function make(members, mix = {}, repeat = 1, globals = []) {
  const draft = createLoraMixDraft(baseline(globals));
  draft.mix.members = members.map(([ref, memberRule]) => ({ ref, rule: memberRule }));
  Object.assign(draft.mix, mix);
  draft.repeat_count = repeat;
  return draft;
}

const SCENARIOS = {
  spec: make([[lora(1), rule("0.6", "0.2", "1.0", "0.1", "max")], [lora(2), rule("-0.2", "-0.8", "0.6", "0.1", "min")], [lora(3), { locked: true, base: "0.5" }]], { budget: { preset: "16", custom: "" } }),
  offgrid: make([[lora(1), rule("0.65", "0.2", "1.0", "0.3")], [lora(2), rule("0", "-1", "1", "0.25", "max")]], { max_displacement: "1", budget: { preset: "32", custom: "" } }),
  large: make(Array.from({ length: 8 }, (_, i) => [lora(i + 1), rule("0", "-1", "1", "0.01", ["balanced", "min", "max"][i % 3])]), { max_displacement: "2", budget: { preset: "64", custom: "" }, sampling_seed: "18446744073709551615" }),
  multiseed: make([[lora(1), rule("0.5", "0", "1", "0.5")], [lora(2), rule("0.5", "0", "1", "0.5")]], { max_displacement: "1", budget: { preset: "custom", custom: "40" } }, 4, [lora(9, { weight: "0.5" })]),
  zeroish: make([[lora(1), rule("0", "-0.2", "0.2", "0.1")], [lora(2), rule("0", "-0.2", "0.2", "0.1")]], { max_displacement: "0.4", budget: { preset: "128", custom: "" } }),
};

test("every JS-generated preview is accepted by the Python validator with identical vectors, bounds and space", { skip: !available && "project Python (pydantic) not available" }, async () => {
  const drafts = {};
  for (const [name, source] of Object.entries(SCENARIOS)) {
    const result = generateCandidates(source);
    assert.equal(result.ok, true, `${name}: ${JSON.stringify(result.issues)}`);
    assert.equal(mixIssues(result.draft).issues.length, 0, name);
    drafts[name] = result.draft;
  }
  const dir = await mkdtemp(path.join(os.tmpdir(), "lora-mix-python-"));
  try {
    const input = path.join(dir, "drafts.json");
    await writeFile(input, JSON.stringify(Object.fromEntries(Object.entries(drafts).map(([name, draft]) => [name, loraMixRecipe(draft)]))));
    const code = "import json,sys\nfrom backend import lora_mix_search as m\nout={}\nfor name,d in json.load(open(sys.argv[1],encoding='utf-8')).items():\n"
      + "    r=m.validate_recipe(d)\n    out[name]={'ids':[c.id for c in r.candidates],'units':[list(c.units) for c in r.candidates],'disp':[c.displacement for c in r.candidates],'limit':r.limit,'slots':r.slots,'space':None if r.space is None else str(r.space),'dmax':r.analysis.dmax}\nprint(json.dumps(out))\n";
    const child = spawnSync(python, ["-B", "-c", code, input], { cwd, encoding: "utf8", windowsHide: true });
    assert.equal(child.status, 0, child.stderr);
    const server = JSON.parse(child.stdout);
    for (const [name, draft] of Object.entries(drafts)) {
      const report = mixIssues(draft), analysis = memberAnalysis(draft), count = countSpace(analysis.free, report.limit);
      assert.deepEqual(server[name].ids, draft.mix.candidates.map((candidate) => candidate.id), name);
      assert.deepEqual(server[name].units, report.candidates.map((entry) => entry.parsed.units), `${name}: the same integer vectors`);
      assert.deepEqual(server[name].disp, report.candidates.map((entry) => entry.displacement), `${name}: the same displacement`);
      assert.equal(server[name].limit, report.limit); assert.equal(server[name].slots, report.budget.slots); assert.equal(server[name].dmax, analysis.dmax);
      assert.equal(server[name].space, count.kind === "exact" ? count.count.toString() : null, `${name}: the same exact space`);
    }
  } finally { await rm(dir, { recursive: true, force: true }); }
});
