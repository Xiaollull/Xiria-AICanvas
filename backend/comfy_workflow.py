"""ComfyUI workflows that describe an XirAI run (P7).

Every picture a text-to-image run saves carries, beside its own parameter block, a ComfyUI
``workflow`` text chunk: the editable graph a user gets by dropping the PNG into ComfyUI. It is
built from what the job executed -- the runtime request after inline ``<lora:...>`` tags were
resolved, the text the encoders were given, the picture's own seed and each resample unit's recorded
seed, steps and sizes -- so the graph describes this picture rather than a generic template.

Clean-room: the JSON follows the UI workflow format (version 0.4) that the ComfyUI 0.33 frontend
saves, as observed in saved files, and node names and widget orders follow the node definitions it
registers. No ComfyUI source is copied or imported.

The graph holds only ComfyUI-native nodes. A stage those nodes cannot express -- ADetailer, USDU
tiled resampling, RTX VSR, background removal -- is left out and named in a Note, never replaced by
something that looks runnable. File names are the relative IDs XirAI mounted, in the backslash form
ComfyUI lists them by; no absolute path, user directory or hash is written.
"""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath

try:
    from .shared_model_paths import parse_shared_ref
except ImportError:
    from shared_model_paths import parse_shared_ref


WORKFLOW_FORMAT_VERSION = 0.4
WORKFLOW_ENGINES = ("SD", "iL", "Anima", "Flux", "Flux2", "Krea2", "MOSS")
# One forward per step, steered by a guidance scalar in the conditioning: the KSampler runs at CFG
# 1 and the request's CFG becomes FluxGuidance.
DISTILLED_ENGINES = frozenset({"Flux", "Flux2"})
# XirAI's native Krea 2 and Flux runtimes fuse a LoRA into the diffusion model only, so the graph
# loads them model-only; SD/iL also patch the text encoders, and Anima keeps the reference layout's
# LoraLoader (its LoRAs carry no text-encoder weights for the CLIP input to apply).
LORA_CLIP_ENGINES = frozenset({"SD", "iL", "Anima"})
NODE_PACK = {"cnr_id": "comfy-core", "ver": "0.33.0"}
# city96/ComfyUI-GGUF, the custom node ComfyUI needs to read a GGUF model or text encoder. Core
# ComfyUI's UNETLoader and CLIPLoader read safetensors only, so a GGUF component is exported through
# this pack's loaders and the graph says it needs them (P9).
GGUF_NODE_PACK = {"cnr_id": "comfyui-gguf"}
# Fixed so the same run always yields the same identifiers, and a workflow is recognisably XirAI's.
WORKFLOW_NAMESPACE = uuid.UUID("5f0c8a3e-7b21-4c6d-9e44-2a9d1b7c3e58")

ENGINE_LABELS = {"SD": "SD", "iL": "iL", "Anima": "Anima", "Flux": "FLUX.1", "Flux2": "FLUX.2", "Krea2": "Krea 2", "MOSS": "MOSS-Image"}
TEXT_ENCODER_TYPES = {"Anima": "stable_diffusion", "Krea2": "krea2", "Flux2": "flux2", "MOSS": "moss_image"}
TEXT_ENCODER_TITLES = {
    "Anima": "Anima Text Encoder · Qwen3",
    "Krea2": "Krea 2 Text Encoder · Qwen3-VL",
    "Flux2": "FLUX.2 Text Encoder",
    "MOSS": "MOSS-Image Text Encoder · Qwen3.5-9B",
}
# Engines whose latents are FLUX.2's 128-channel, stride-16 packing.
FLUX2_LATENT_ENGINES = frozenset({"Flux2", "MOSS"})
# The Comfy-Org MOSS files fuse each block's q/k/v; ComfyUI reads them only with core MOSS support
# (`UNETLoader` + `CLIPLoader(type=moss_image)`), which on 2026-10-16 exists as the unmerged
# Comfy-Org/ComfyUI PR #17095. The OpenMOSS plugin @9d799c1 loads HF directories and cannot read
# the fused files. The graph says so rather than claiming a core 0.33 node can run it.
MOSS_REQUIREMENT = (
    "ComfyUI with core MOSS-Image support (Comfy-Org/ComfyUI PR #17095, not in a release on 2026-10-16): "
    "UNETLoader and CLIPLoader type moss_image read the Comfy-Org single files. The OpenMOSS plugin's "
    "loaders read HF directories and cannot load these fused files."
)
# ComfyUI's ModelSamplingFlux anchors; FLUX.1's resolution shift is exactly this node's line.
FLUX_MAX_SHIFT = 1.15
FLUX_BASE_SHIFT = 0.5

# LatentUpscale offers nearest-exact / bilinear / area / bicubic / bislerp. The antialiased pair
# and torch's plain `nearest` have no native counterpart and fall back to the closest mode.
LATENT_UPSCALE_METHODS = {
    "latent": "bilinear",
    "latent-antialias": "bilinear",
    "latent-bicubic": "bicubic",
    "latent-bicubic-antialias": "bicubic",
    "latent-nearest": "nearest-exact",
    "latent-nearest-exact": "nearest-exact",
}
APPROXIMATE_LATENT_METHODS = frozenset({"latent-antialias", "latent-bicubic-antialias", "latent-nearest"})
# ImageScale offers nearest-exact / bilinear / area / bicubic / lanczos; Forge's PIL `nearest`
# becomes the torch nearest-exact it most resembles.
IMAGE_SCALE_METHODS = {
    "lanczos": "lanczos",
    "nearest": "nearest-exact",
    "nearest-exact": "nearest-exact",
    "bilinear": "bilinear",
    "bicubic": "bicubic",
    "area": "area",
}

# Layout, in canvas pixels. Sections sit side by side, every group is as tall as the tallest one,
# and a group widens (and everything after it moves right) instead of overlapping its neighbour.
GROUP_MIN_HEIGHT = 750
GROUP_BOTTOM_MARGIN = 30
NODE_GAP = 30
SECTION_WIDTHS = {"model": 360, "prompts": 430, "base": 400, "nsampler": 450, "final": 400}
SECTION_COLORS = {
    "model": "#665A96",
    "lora": "#9077A5",
    "prompts": "#587997",
    "base": "#8B7CFF",
    "nsampler": "#3f789e",
    "final": "#69A990",
}
CARD_COLORS = {"latent": "#BD8A68", "pixel": "#A77763"}
POSITIVE_COLORS = ("#232", "#353")
NEGATIVE_COLORS = ("#322", "#533")
NOTE_COLORS = ("#432", "#653")
LORA_COLUMNS = 4
LORA_SLOT_WIDTH = 330
LORA_SLOT_HEIGHT = 220
LORA_ORIGIN = (30, 80)
LORA_NODE_SIZE = (300, 190)
LORA_MODEL_ONLY_SIZE = (300, 130)
CARD_SIZE = (210, 140)
CARD_STEP = (210, 170)
CARD_ORIGIN = (10, 70)
VAE_TILE_WIDGETS = (("tile_size", 768), ("overlap", 64), ("temporal_size", 64), ("temporal_overlap", 8))
MAX_TITLE_NAME = 32

# Appendix A of the P7 plan: at most four cards a row, rows unlimited.
LORA_ROW_TABLE = {
    1: (1,), 2: (2,), 3: (2, 1), 4: (2, 2), 5: (2, 2, 1), 6: (2, 2, 2),
    7: (3, 2, 2), 8: (3, 3, 2), 9: (3, 3, 3), 10: (3, 3, 3, 1), 11: (3, 3, 3, 2), 12: (3, 3, 3, 3),
}
_BREAK_WORD = re.compile(r"\bBREAK\b")


@dataclass(frozen=True)
class ResampleStage:
    """One enabled resample unit as it executed for one picture."""

    number: int
    redraws: bool
    tiled: bool
    upscaler: str | None
    latent_algorithm: str | None
    resize_algorithm: str
    denoise: float
    steps: int | None
    cfg: float | None
    sampler: str | None
    scheduler: str | None
    seed: int | None
    source_size: tuple[int, int]
    target_size: tuple[int, int]
    redraw_size: tuple[int, int]
    output_size: tuple[int, int]


@dataclass(frozen=True)
class WorkflowFacts:
    """What one saved picture actually ran with, beyond the request itself.

    ``stages`` is the enabled post-processing chain in execution order: a :class:`ResampleStage`
    per resample unit, and the strings ``"adetailer"`` / ``"rtx"`` for the stages the graph omits.
    """

    identity: str
    seed: int
    positive: str
    negative: str
    loras: tuple[tuple[str, float], ...] = ()
    clip_skip: int | None = None
    stages: tuple = ()
    background_removal: bool = False


def lora_rows(count: int) -> tuple[int, ...]:
    """Cards per row for ``count`` LoRAs: the exact table to 12, then rows of four."""
    if count <= 0:
        return ()
    if count in LORA_ROW_TABLE:
        return LORA_ROW_TABLE[count]
    full, rest = divmod(count, LORA_COLUMNS)
    return (LORA_COLUMNS,) * full + ((rest,) if rest else ())


def lora_group_label(count: int) -> str:
    """``LoRA · 6 (2 × 3)``: the arrangement is spelled out while it fits in a glance."""
    rows = lora_rows(count)
    label = f"LoRA · {count}"
    if 2 <= len(rows) <= 3:
        if len(set(rows)) == 1:
            label += f" ({rows[0]} × {len(rows)})"
        else:
            label += " (" + " + ".join(str(row) for row in rows) + ")"
    return label


def comfy_asset_name(value) -> str:
    """A model reference as ComfyUI lists it: relative, backslash-separated, never absolute.

    A shared reference becomes its path inside the shared root, which is how a ComfyUI that uses
    the same folder names the file. Anything absolute or escaping keeps only its file name.
    """
    if not value:
        return ""
    text = str(value)
    shared = parse_shared_ref(text)
    if shared:
        text = shared[1]
    posix = PurePosixPath(text.replace("\\", "/"))
    windows = PureWindowsPath(text)
    if windows.is_absolute() or windows.drive or posix.is_absolute() or ".." in posix.parts:
        return posix.name
    return "\\".join(part for part in posix.parts if part not in ("", "."))


def asset_stem(value) -> str:
    name = comfy_asset_name(value).split("\\")[-1]
    stem = name.rsplit(".", 1)[0] if "." in name else name
    return stem if len(stem) <= MAX_TITLE_NAME else stem[: MAX_TITLE_NAME - 1] + "…"


def _number(value) -> str:
    return f"{float(value):g}"


class _Ids:
    """Node, link and group ids are unique across the root graph and every subgraph."""

    def __init__(self):
        self.node = 0
        self.link = 0
        self.group = 0

    def next_node(self) -> int:
        self.node += 1
        return self.node

    def next_link(self) -> int:
        self.link += 1
        return self.link

    def next_group(self) -> int:
        self.group += 1
        return self.group


class _Graph:
    def __init__(self, ids: _Ids):
        self.ids = ids
        self.nodes: list[dict] = []
        self.links: list[tuple] = []
        self.groups: list[dict] = []

    def add(self, node_type, title, pos, size, *, inputs=(), outputs=(), widgets=(), colors=None, properties=None):
        node = {
            "id": self.ids.next_node(),
            "type": node_type,
            "pos": [int(pos[0]), int(pos[1])],
            "size": [int(size[0]), int(size[1])],
            "flags": {},
            "order": 0,
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": name, "type": kind, "links": []} for name, kind in outputs],
            "title": title,
            "properties": dict(properties) if properties is not None else {**NODE_PACK, "Node name for S&R": node_type},
        }
        for entry in inputs:
            name, kind = entry[0], entry[1]
            slot = {"name": name, "type": kind, "link": None}
            if len(entry) > 2 and entry[2]:
                slot["widget"] = {"name": name}
            node["inputs"].append(slot)
        if widgets:
            node["widgets_values"] = [value for _, value in widgets]
            node["widgets_values_named"] = {name: value for name, value in widgets}
        if colors:
            node["color"], node["bgcolor"] = colors
        self.nodes.append(node)
        return node

    def connect(self, source, slot, target, input_name):
        index = next(position for position, entry in enumerate(target["inputs"]) if entry["name"] == input_name)
        if target["inputs"][index]["link"] is not None:
            raise ValueError(f"{target['type']}.{input_name} is already connected")
        link_id = self.ids.next_link()
        kind = source["outputs"][slot]["type"]
        source["outputs"][slot]["links"].append(link_id)
        target["inputs"][index]["link"] = link_id
        self.links.append((link_id, source["id"], slot, target["id"], index, kind))
        return link_id

    def group(self, title, bounding, color):
        entry = {"id": self.ids.next_group(), "title": title, "bounding": [int(value) for value in bounding], "color": color, "flags": {}}
        self.groups.append(entry)
        return entry

    def assign_order(self):
        """Execution order by dependency, ties by id. Links from or to a subgraph's IO nodes are not
        dependencies between its own nodes and are skipped."""
        by_id = {node["id"]: node for node in self.nodes}
        pending = {node["id"]: 0 for node in self.nodes}
        edges = {node["id"]: [] for node in self.nodes}
        for _link, origin, _slot, target, _index, _kind in self.links:
            if origin in by_id and target in by_id:
                pending[target] += 1
                edges[origin].append(target)
        ready = sorted(node_id for node_id, count in pending.items() if count == 0)
        order = 0
        while ready:
            node_id = ready.pop(0)
            by_id[node_id]["order"] = order
            order += 1
            for target in edges[node_id]:
                pending[target] -= 1
                if pending[target] == 0:
                    ready.append(target)
                    ready.sort()
        if order != len(self.nodes):
            raise ValueError("workflow graph has a cycle")

    def finish_outputs(self):
        for node in self.nodes:
            for output in node["outputs"]:
                if not output["links"]:
                    output["links"] = None


class _Subgraph(_Graph):
    """A card's definition: its own nodes, fed through a -10 input node and read at -20."""

    INPUT_NODE = -10
    OUTPUT_NODE = -20

    def __init__(self, ids: _Ids, identity: str, name: str):
        super().__init__(ids)
        self.identity = identity
        self.name = name
        self.inputs: list[dict] = []
        self.outputs: list[dict] = []

    def _io(self, name, kind, collection, role):
        entry = {
            "id": str(uuid.uuid5(WORKFLOW_NAMESPACE, f"{self.identity}/{role}/{name}")),
            "name": name,
            "type": kind,
            "linkIds": [],
            "pos": [0, 0],
        }
        collection.append(entry)
        return entry

    def add_input(self, name, kind):
        return self._io(name, kind, self.inputs, "input")

    def add_output(self, name, kind):
        return self._io(name, kind, self.outputs, "output")

    def feed(self, input_name, target, target_input):
        slot = next(index for index, entry in enumerate(self.inputs) if entry["name"] == input_name)
        index = next(position for position, entry in enumerate(target["inputs"]) if entry["name"] == target_input)
        link_id = self.ids.next_link()
        target["inputs"][index]["link"] = link_id
        self.inputs[slot]["linkIds"].append(link_id)
        self.links.append((link_id, self.INPUT_NODE, slot, target["id"], index, self.inputs[slot]["type"]))

    def emit(self, source, slot, output_name):
        index = next(position for position, entry in enumerate(self.outputs) if entry["name"] == output_name)
        link_id = self.ids.next_link()
        source["outputs"][slot]["links"].append(link_id)
        self.outputs[index]["linkIds"].append(link_id)
        self.links.append((link_id, source["id"], slot, self.OUTPUT_NODE, index, self.outputs[index]["type"]))

    def definition(self):
        xs = [node["pos"][0] for node in self.nodes]
        ys = [node["pos"][1] for node in self.nodes]
        right = max(node["pos"][0] + node["size"][0] for node in self.nodes)
        bottom = max(node["pos"][1] + node["size"][1] for node in self.nodes)
        middle = (min(ys) + bottom) // 2
        input_height = 48 + 20 * len(self.inputs)
        input_box = [min(xs) - 300, middle - input_height // 2, 128, input_height]
        output_height = 48 + 20 * len(self.outputs)
        output_box = [right + 170, middle - output_height // 2, 128, output_height]
        for index, entry in enumerate(self.inputs):
            entry["pos"] = [input_box[0] + 104, input_box[1] + 24 + 20 * index]
        for index, entry in enumerate(self.outputs):
            entry["pos"] = [output_box[0] + 24, output_box[1] + 24 + 20 * index]
        return {
            "id": str(uuid.uuid5(WORKFLOW_NAMESPACE, f"{self.identity}/subgraph")),
            "version": 1,
            "state": {"lastGroupId": 0, "lastNodeId": 0, "lastLinkId": 0, "lastRerouteId": 0},
            "revision": 0,
            "config": {},
            "name": self.name,
            "inputNode": {"id": self.INPUT_NODE, "bounding": input_box},
            "outputNode": {"id": self.OUTPUT_NODE, "bounding": output_box},
            "inputs": self.inputs,
            "outputs": self.outputs,
            "widgets": [],
            "nodes": self.nodes,
            "groups": self.groups,
            "links": [
                {"id": link_id, "origin_id": origin, "origin_slot": slot, "target_id": target, "target_slot": index, "type": kind}
                for link_id, origin, slot, target, index, kind in self.links
            ],
            "extra": {},
        }


def _column(graph, x, y, entries):
    """Stack nodes top to bottom from ``y``; ``entries`` are keyword dicts for ``graph.add``."""
    nodes = []
    for entry in entries:
        node = graph.add(entry.pop("type"), entry.pop("title"), (x, y), entry.pop("size"), **entry)
        nodes.append(node)
        y += node["size"][1] + NODE_GAP
    return nodes


def _prompt_text(text: str, steps: int, notes: list[str], label: str, *, allow_and: bool):
    """The text a CLIPTextEncode can hold for an SD/iL prompt, and a note for what it cannot."""
    try:
        from .dynamic_prompt import PromptPlanLimitError, PromptTimeline, static_prompt_plan
    except ImportError:
        from dynamic_prompt import PromptPlanLimitError, PromptTimeline, static_prompt_plan
    if _BREAK_WORD.search(text):
        notes.append(f"The {label} prompt uses BREAK, which ComfyUI's CLIPTextEncode reads as a word.")
    try:
        if allow_and and static_prompt_plan(text, allow_and=True).composable:
            notes.append(f"The {label} prompt uses AND composition, which ComfyUI's CLIPTextEncode reads as text.")
        plan = static_prompt_plan(text, allow_and=False)
    except (PromptPlanLimitError, ValueError):
        return text
    if not plan.dynamic:
        return text
    notes.append(
        f"The {label} prompt schedules or alternates text; this graph holds the state of its first step."
    )
    return plan.terms[0].timeline(PromptTimeline(max(1, int(steps)))).text_at(0)


class _WorkflowBuilder:
    def __init__(self, request, facts: WorkflowFacts):
        self.request = request
        self.facts = facts
        self.engine = request.engine
        self.distilled = self.engine in DISTILLED_ENGINES
        self.ids = _Ids()
        self.root = _Graph(self.ids)
        self.subgraphs: list[_Subgraph] = []
        self.notes: list[str] = []
        self.requirements: list[str] = []
        self.bottoms: dict[str, int] = {}

    # -- values ------------------------------------------------------------------------------

    def flux2_shift(self, width, height, steps):
        try:
            from .flux2_sampling import flux2_resolution_shift
        except ImportError:
            from flux2_sampling import flux2_resolution_shift
        return round(flux2_resolution_shift(int(width), int(height), int(steps)), 6)

    def model_sampling_widgets(self, width, height, steps):
        """ModelSamplingFlux arguments that reproduce this engine's shift at this canvas.

        FLUX.1's shift is the node's own base/max line. FLUX.2 fits its shift from the canvas and
        the step count; setting both anchors to that value makes the node return it exactly.
        """
        if self.engine == "Flux":
            maximum, base = FLUX_MAX_SHIFT, FLUX_BASE_SHIFT
        else:
            maximum = base = self.flux2_shift(width, height, steps)
        return (("max_shift", maximum), ("base_shift", base), ("width", int(width)), ("height", int(height)))

    def ksampler_widgets(self, seed, steps, cfg, sampler, scheduler, denoise):
        return (
            ("seed", int(seed)),
            ("control_after_generate", "fixed"),
            ("steps", int(steps)),
            ("cfg", 1.0 if self.distilled else float(cfg)),
            ("sampler_name", sampler),
            ("scheduler", scheduler),
            ("denoise", float(denoise)),
        )

    # -- sections ----------------------------------------------------------------------------

    def build(self):
        request, facts = self.request, self.facts
        cards = [stage for stage in facts.stages if isinstance(stage, ResampleStage) and not stage.tiled]
        rows = lora_rows(len(facts.loras))
        sections = ["model"] + (["lora"] if facts.loras else []) + ["prompts", "base"] + (["nsampler"] if cards else []) + ["final"]
        numbers = {name: index for index, name in enumerate(sections, start=1)}
        lora_width = 30 + max(rows, default=0) * LORA_SLOT_WIDTH
        widths = {**SECTION_WIDTHS, "lora": lora_width}
        lefts, x = {}, 0
        for name in sections:
            lefts[name] = x
            x += widths[name]
        total_width = x

        model, clip, vae = self.build_model(lefts["model"])
        if facts.loras:
            model, clip = self.build_loras(lefts["lora"], model, clip, rows)
        positive, negative = self.build_prompts(lefts["prompts"], clip)
        shared_model, latent, latent_size = self.build_base(lefts["base"], model, positive, negative)
        final_size = (int(request.width), int(request.height))
        last_resize = "lanczos"
        if cards:
            latent, latent_size, final_size, last_resize = self.build_cards(
                lefts["nsampler"], numbers["nsampler"], cards, latent, latent_size, shared_model, positive, negative, vae
            )
        self.note_omissions()
        self.build_final(lefts["final"], latent, vae, latent_size, final_size, last_resize)

        height = max([GROUP_MIN_HEIGHT] + [bottom + GROUP_BOTTOM_MARGIN for bottom in self.bottoms.values()])
        titles = {
            "model": "Model & VAE",
            "lora": lora_group_label(len(facts.loras)),
            "prompts": "Prompts & CLIP" if self.bottoms.get("clip_skip") else "Prompts",
            "base": "Base Sampler",
            "nsampler": "NSampler",
            "final": "Final Output",
        }
        for name in sections:
            self.root.group(f"{numbers[name]:02d} / {titles[name]}", (lefts[name], 0, widths[name], height), SECTION_COLORS[name])

        return self.serialize(total_width, height)

    def build_model(self, left):
        request, engine = self.request, self.engine
        x = left + 30
        if engine in ("SD", "iL"):
            loader = self.root.add(
                "CheckpointLoaderSimple", f"{engine} / {'SDXL ' if engine == 'iL' else ''}Checkpoint", (x, 110), (300, 110),
                outputs=(("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")),
                widgets=(("ckpt_name", comfy_asset_name(request.checkpoint)),),
            )
            self.bottoms["model"] = 220
            return (loader, 0), (loader, 1), (loader, 2)
        label = ENGINE_LABELS[engine]
        diffusion = comfy_asset_name(request.diffusion_model)
        if _is_gguf(diffusion):
            self.requirements.append("ComfyUI-GGUF (city96): UnetLoaderGGUF reads the GGUF diffusion model.")
            unet = self.root.add(
                "UnetLoaderGGUF", f"{label} Diffusion Model · GGUF", (x, 110), (300, 80),
                outputs=(("MODEL", "MODEL"),),
                widgets=(("unet_name", diffusion),),
                properties={**GGUF_NODE_PACK, "Node name for S&R": "UnetLoaderGGUF"},
            )
        else:
            unet = self.root.add(
                "UNETLoader", f"{label} Diffusion Model", (x, 110), (300, 105),
                outputs=(("MODEL", "MODEL"),),
                widgets=(("unet_name", diffusion), ("weight_dtype", "default")),
            )
        if engine == "Flux":
            names = (comfy_asset_name(request.text_encoder), comfy_asset_name(request.text_encoder_2))
            if any(_is_gguf(name) for name in names):
                self.requirements.append("ComfyUI-GGUF (city96): DualCLIPLoaderGGUF reads the GGUF text encoder.")
                encoder = self.root.add(
                    "DualCLIPLoaderGGUF", "FLUX.1 Text Encoders · CLIP-L + T5-XXL · GGUF", (x, 240), (300, 135),
                    outputs=(("CLIP", "CLIP"),),
                    widgets=(("clip_name1", names[0]), ("clip_name2", names[1]), ("type", "flux")),
                    properties={**GGUF_NODE_PACK, "Node name for S&R": "DualCLIPLoaderGGUF"},
                )
            else:
                encoder = self.root.add(
                    "DualCLIPLoader", "FLUX.1 Text Encoders · CLIP-L + T5-XXL", (x, 240), (300, 160),
                    outputs=(("CLIP", "CLIP"),),
                    widgets=(
                        ("clip_name1", names[0]),
                        ("clip_name2", names[1]),
                        ("type", "flux"),
                        ("device", "default"),
                    ),
                )
        else:
            name = comfy_asset_name(request.text_encoder)
            if _is_gguf(name):
                self.requirements.append("ComfyUI-GGUF (city96): CLIPLoaderGGUF reads the GGUF text encoder.")
                encoder = self.root.add(
                    "CLIPLoaderGGUF", f"{TEXT_ENCODER_TITLES[engine]} · GGUF", (x, 240), (300, 110),
                    outputs=(("CLIP", "CLIP"),),
                    widgets=(("clip_name", name), ("type", TEXT_ENCODER_TYPES[engine])),
                    properties={**GGUF_NODE_PACK, "Node name for S&R": "CLIPLoaderGGUF"},
                )
            else:
                encoder = self.root.add(
                    "CLIPLoader", TEXT_ENCODER_TITLES[engine], (x, 240), (300, 135),
                    outputs=(("CLIP", "CLIP"),),
                    widgets=(("clip_name", name), ("type", TEXT_ENCODER_TYPES[engine]), ("device", "default")),
                )
        vae_name = comfy_asset_name(request.vae)
        if _is_gguf(vae_name):
            self.notes.append(
                f"{vae_name.split(chr(92))[-1]} is a GGUF VAE. ComfyUI's VAELoader reads safetensors only and "
                "ComfyUI-GGUF has no VAE loader, so load this VAE's safetensors release there."
            )
        vae_y = encoder["pos"][1] + encoder["size"][1] + 25
        loader = self.root.add(
            "VAELoader", f"{label} VAE", (x, vae_y), (300, 100),
            outputs=(("VAE", "VAE"),),
            widgets=(("vae_name", vae_name),),
        )
        self.bottoms["model"] = vae_y + 100
        return (unet, 0), (encoder, 0), (loader, 0)

    def build_loras(self, left, model, clip, rows):
        with_clip = self.engine in LORA_CLIP_ENGINES
        index = 0
        for row, count in enumerate(rows):
            for column in range(count):
                path, weight = self.facts.loras[index]
                index += 1
                pos = (left + LORA_ORIGIN[0] + column * LORA_SLOT_WIDTH, LORA_ORIGIN[1] + row * LORA_SLOT_HEIGHT)
                name = comfy_asset_name(path)
                title = f"LoRA {index} · {asset_stem(path)}"
                if with_clip:
                    node = self.root.add(
                        "LoraLoader", title, pos, LORA_NODE_SIZE,
                        inputs=(("model", "MODEL"), ("clip", "CLIP")),
                        outputs=(("MODEL", "MODEL"), ("CLIP", "CLIP")),
                        widgets=(("lora_name", name), ("strength_model", float(weight)), ("strength_clip", float(weight))),
                    )
                    self.root.connect(clip[0], clip[1], node, "clip")
                    clip = (node, 1)
                else:
                    node = self.root.add(
                        "LoraLoaderModelOnly", title, pos, LORA_MODEL_ONLY_SIZE,
                        inputs=(("model", "MODEL"),),
                        outputs=(("MODEL", "MODEL"),),
                        widgets=(("lora_name", name), ("strength_model", float(weight))),
                    )
                self.root.connect(model[0], model[1], node, "model")
                model = (node, 0)
        self.bottoms["lora"] = LORA_ORIGIN[1] + (len(rows) - 1) * LORA_SLOT_HEIGHT + LORA_NODE_SIZE[1]
        return model, clip

    def build_prompts(self, left, clip):
        request, facts, engine = self.request, self.facts, self.engine
        x = left + 40
        if engine in ("SD", "iL"):
            positive_text = _prompt_text(facts.positive, request.steps, self.notes, "positive", allow_and=True)
            negative_text = _prompt_text(facts.negative, request.steps, self.notes, "negative", allow_and=False)
        else:
            positive_text, negative_text = facts.positive, facts.negative
        if engine in ("SD", "iL") and facts.clip_skip and facts.clip_skip > 1:
            skip = self.root.add(
                "CLIPSetLastLayer", f"{'SDXL' if engine == 'iL' else 'SD'} CLIP skip {facts.clip_skip}", (x, 70), (360, 110),
                inputs=(("clip", "CLIP"),),
                outputs=(("CLIP", "CLIP"),),
                widgets=(("stop_at_clip_layer", -int(facts.clip_skip)),),
            )
            self.root.connect(clip[0], clip[1], skip, "clip")
            clip = (skip, 0)
            self.bottoms["clip_skip"] = 180
        positive = self.root.add(
            "CLIPTextEncode", "Positive Prompt", (x, 210), (360, 240),
            inputs=(("clip", "CLIP"),), outputs=(("CONDITIONING", "CONDITIONING"),),
            widgets=(("text", positive_text),), colors=POSITIVE_COLORS,
        )
        self.root.connect(clip[0], clip[1], positive, "clip")
        if self.distilled:
            negative = self.root.add(
                "ConditioningZeroOut", "Negative · none (guidance distilled)", (x, 480), (360, 60),
                inputs=(("conditioning", "CONDITIONING"),), outputs=(("CONDITIONING", "CONDITIONING"),),
                colors=NEGATIVE_COLORS,
            )
            self.root.connect(positive, 0, negative, "conditioning")
            self.bottoms["prompts"] = 540
        else:
            negative = self.root.add(
                "CLIPTextEncode", "Negative Prompt", (x, 480), (360, 230),
                inputs=(("clip", "CLIP"),), outputs=(("CONDITIONING", "CONDITIONING"),),
                widgets=(("text", negative_text),), colors=NEGATIVE_COLORS,
            )
            self.root.connect(clip[0], clip[1], negative, "clip")
            self.bottoms["prompts"] = 710
        return positive, negative

    def build_base(self, left, model, positive, negative):
        request, engine = self.request, self.engine
        width, height = int(request.width), int(request.height)
        x = left + 40
        entries = [{
            "type": "EmptyFlux2LatentImage" if engine in FLUX2_LATENT_ENGINES else "EmptyLatentImage",
            "title": f"Base Canvas · {width} × {height}",
            "size": (330, 110),
            "outputs": (("LATENT", "LATENT"),),
            "widgets": (("width", width), ("height", height), ("batch_size", 1)),
        }]
        guidance = getattr(request, "guidance", "none")
        patch = None
        if guidance == "pag":
            pag = request.pag
            if engine in ("SD", "iL"):
                patch = {
                    "type": "PerturbedAttentionGuidance", "title": f"PAG · scale {_number(pag.scale)}", "size": (330, 80),
                    "inputs": (("model", "MODEL"),), "outputs": (("MODEL", "MODEL"),), "widgets": (("scale", float(pag.scale)),),
                }
                if getattr(pag, "applied_layers", "mid") != "mid":
                    self.notes.append("PAG ran on all UNet layers in XirAI; ComfyUI's PerturbedAttentionGuidance patches the middle block only.")
            else:
                self.notes.append(f"PAG ran in XirAI; ComfyUI's PerturbedAttentionGuidance targets SD UNets, so it is not in this {ENGINE_LABELS[engine]} graph.")
        elif guidance == "cfg_zero_star":
            patch = {
                "type": "CFGZeroStar", "title": "CFG-Zero*", "size": (330, 50),
                "inputs": (("model", "MODEL"),), "outputs": (("patched_model", "MODEL"),),
            }
        if patch:
            entries.append(patch)
        moss_nodes = []
        if engine == "MOSS":
            # The official workflow's order: ModelSamplingSD3 before SkipLayerGuidanceDiTSimple, so the
            # schedule and the skip window share one shift.
            moss = request.moss
            self.requirements.append(MOSS_REQUIREMENT)
            self.notes.append("XirAI decodes the MOSS VAE in float32; start ComfyUI with --fp32-vae for the matching decode.")
            if getattr(request, "moss_layout", None) is not None and any(
                getattr(unit, "enabled", False) for unit in getattr(request, "resample_units", ())
            ):
                # One positive node feeds every sampler here, but a layout's boxes are pixels of one canvas.
                self.notes.append(
                    "The positive prompt is a MOSS layout for the base canvas. XirAI re-encoded it for each Hires "
                    "canvas (boxes scaled); the resample samplers in this graph reuse the base-canvas layout."
                )
            moss_nodes.append({
                "type": "ModelSamplingSD3", "title": f"MOSS Shift · {_number(moss.shift)}", "size": (330, 80),
                "inputs": (("model", "MODEL"),), "outputs": (("MODEL", "MODEL"),),
                "widgets": (("shift", float(moss.shift)),),
            })
            if moss.uncond_skip_layers:
                moss_nodes.append({
                    "type": "SkipLayerGuidanceDiTSimple",
                    "title": "Negative-branch skip · " + ", ".join(str(layer) for layer in moss.uncond_skip_layers),
                    "size": (330, 170),
                    "inputs": (("model", "MODEL"),), "outputs": (("MODEL", "MODEL"),),
                    "widgets": (
                        ("double_layers", ""),
                        ("single_layers", ", ".join(str(layer) for layer in moss.uncond_skip_layers)),
                        ("start_percent", float(moss.uncond_skip_start)),
                        ("end_percent", float(moss.uncond_skip_end)),
                    ),
                })
            entries.extend(moss_nodes)
        sampling = None
        if self.distilled:
            sampling = {
                "type": "ModelSamplingFlux",
                "title": f"{ENGINE_LABELS[engine]} Shift · {width} × {height}",
                "size": (330, 170),
                "inputs": (("model", "MODEL"),), "outputs": (("MODEL", "MODEL"),),
                "widgets": self.model_sampling_widgets(width, height, request.steps),
            }
            entries.append(sampling)
            entries.append({
                "type": "FluxGuidance", "title": f"Guidance · {_number(request.cfg)}", "size": (330, 80),
                "inputs": (("conditioning", "CONDITIONING"),), "outputs": (("CONDITIONING", "CONDITIONING"),),
                "widgets": (("guidance", float(request.cfg)),),
            })
        entries.append({
            "type": "KSampler", "title": "Sampler 1 · Base Image", "size": (330, 500),
            "inputs": (("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("latent_image", "LATENT")),
            "outputs": (("LATENT", "LATENT"),),
            "widgets": self.ksampler_widgets(self.facts.seed, request.steps, request.cfg, request.sampler, request.scheduler, 1.0),
        })
        nodes = _column(self.root, x, 70, entries)
        latent, sampler = nodes[0], nodes[-1]
        position = 1
        if patch:
            self.root.connect(model[0], model[1], nodes[position], "model")
            model = (nodes[position], 0)
            position += 1
        for _node in moss_nodes:
            self.root.connect(model[0], model[1], nodes[position], "model")
            model = (nodes[position], 0)
            position += 1
        shared_model = model
        base_positive = positive
        if sampling:
            self.root.connect(model[0], model[1], nodes[position], "model")
            model = (nodes[position], 0)
            self.root.connect(positive, 0, nodes[position + 1], "conditioning")
            base_positive = nodes[position + 1]
        self.root.connect(model[0], model[1], sampler, "model")
        self.root.connect(base_positive, 0, sampler, "positive")
        self.root.connect(negative, 0, sampler, "negative")
        self.root.connect(latent, 0, sampler, "latent_image")
        self.bottoms["base"] = sampler["pos"][1] + sampler["size"][1]
        return shared_model, sampler, (int(request.width), int(request.height))

    def build_cards(self, left, section, cards, latent, latent_size, model, positive, negative, vae):
        sampler_number = 1
        grid = 16 if self.engine in FLUX2_LATENT_ENGINES else 8
        final_size = latent_size
        last_resize = "lanczos"
        for index, stage in enumerate(cards):
            if stage.redraws:
                sampler_number += 1
            subgraph = self.build_card(section, index + 1, sampler_number, stage)
            definition_id = str(uuid.uuid5(WORKFLOW_NAMESPACE, f"{subgraph.identity}/subgraph"))
            column, row = index % 2, index // 2
            pos = (left + CARD_ORIGIN[0] + column * CARD_STEP[0], CARD_ORIGIN[1] + row * CARD_STEP[1])
            widgets = (("seed", int(stage.seed)),) if stage.redraws else ()
            instance = self.root.add(
                definition_id, subgraph.name, pos, CARD_SIZE,
                inputs=[(entry["name"], entry["type"]) for entry in subgraph.inputs if entry["name"] != "seed"],
                outputs=(("LATENT", "LATENT"),),
                widgets=widgets,
                properties=dict(NODE_PACK),
            )
            self.root.connect(latent, 0, instance, "samples")
            if stage.redraws:
                self.root.connect(model[0], model[1], instance, "model")
                self.root.connect(positive, 0, instance, "positive")
                self.root.connect(negative, 0, instance, "negative")
            if not stage.latent_algorithm:
                self.root.connect(vae[0], vae[1], instance, "vae")
            latent = instance
            if stage.redraws:
                latent_size = tuple(stage.redraw_size)
            else:
                # A resize with no redraw lands on its exact target; re-encoding keeps the latent grid.
                latent_size = tuple((edge // grid) * grid for edge in stage.target_size)
            final_size = tuple(stage.output_size)
            last_resize = stage.resize_algorithm
            self.bottoms["nsampler"] = pos[1] + CARD_SIZE[1]
        return latent, latent_size, final_size, last_resize

    def build_card(self, section, index, sampler_number, stage: ResampleStage):
        name = f"{section}.{index} / {sampler_number}Sampler" if stage.redraws else f"{section}.{index} / Upscale"
        kind = "latent" if stage.latent_algorithm else "pixel"
        subgraph = _Subgraph(self.ids, f"{self.facts.identity}/card/{index}/{kind}", name)
        subgraph.add_input("samples", "LATENT")
        if stage.redraws:
            for input_name, input_type in (("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING")):
                subgraph.add_input(input_name, input_type)
        if kind == "pixel":
            subgraph.add_input("vae", "VAE")
        if stage.redraws:
            subgraph.add_input("seed", "INT")
        subgraph.add_output("LATENT", "LATENT")
        size = tuple(stage.redraw_size) if stage.redraws else tuple(stage.target_size)
        if kind == "latent":
            if stage.latent_algorithm in APPROXIMATE_LATENT_METHODS:
                self.notes.append(
                    f"Resample {stage.number} used {stage.latent_algorithm}; ComfyUI's LatentUpscale has no exact "
                    f"match, so its card uses {LATENT_UPSCALE_METHODS[stage.latent_algorithm]}."
                )
            upscale = subgraph.add(
                "LatentUpscale", f"Hires {index} / Latent Upscale", (0, 0), (370, 160),
                inputs=(("samples", "LATENT"),), outputs=(("LATENT", "LATENT"),),
                widgets=(
                    ("upscale_method", LATENT_UPSCALE_METHODS.get(stage.latent_algorithm, "bilinear")),
                    ("width", int(size[0])), ("height", int(size[1])), ("crop", "disabled"),
                ),
            )
            subgraph.feed("samples", upscale, "samples")
            result = upscale
            if stage.redraws:
                sampler = self.card_sampler(subgraph, (0, 190), (370, 310), sampler_number, "Hires Redraw", stage)
                subgraph.connect(upscale, 0, sampler, "latent_image")
                subgraph.feed("model", sampler, "model")
                subgraph.feed("positive", sampler, "positive")
                result = sampler
            subgraph.emit(result, 0, "LATENT")
            title = f"{section:02d} / Hires {index} · Latent"
        else:
            if stage.resize_algorithm == "nearest":
                self.notes.append(
                    f"Resample {stage.number} resized with Forge's PIL nearest; its card uses ComfyUI's nearest-exact."
                )
            decode = subgraph.add(
                "VAEDecodeTiled", f"Hires {index} / VAE Tiled Decode", (0, 0), (470, 180),
                inputs=(("samples", "LATENT"), ("vae", "VAE")), outputs=(("IMAGE", "IMAGE"),),
                widgets=VAE_TILE_WIDGETS,
            )
            subgraph.feed("samples", decode, "samples")
            subgraph.feed("vae", decode, "vae")
            image = decode
            column_y = 0
            if stage.upscaler:
                loader = subgraph.add(
                    "UpscaleModelLoader", f"UPS Model · {asset_stem(stage.upscaler)}", (0, 210), (470, 140),
                    outputs=(("UPSCALE_MODEL", "UPSCALE_MODEL"),),
                    widgets=(("model_name", comfy_asset_name(stage.upscaler)),),
                )
                upscaled = subgraph.add(
                    "ImageUpscaleWithModel", "Upscale with Model", (500, column_y), (470, 180),
                    inputs=(("upscale_model", "UPSCALE_MODEL"), ("image", "IMAGE")), outputs=(("IMAGE", "IMAGE"),),
                )
                subgraph.connect(loader, 0, upscaled, "upscale_model")
                subgraph.connect(decode, 0, upscaled, "image")
                image = upscaled
                column_y += 180 + NODE_GAP
            method = IMAGE_SCALE_METHODS.get(stage.resize_algorithm, "lanczos")
            scale = subgraph.add(
                "ImageScale", f"Target Scale · {method.title()} {size[0]} × {size[1]}", (500, column_y), (470, 140),
                inputs=(("image", "IMAGE"),), outputs=(("IMAGE", "IMAGE"),),
                widgets=(("upscale_method", method), ("width", int(size[0])), ("height", int(size[1])), ("crop", "disabled")),
            )
            subgraph.connect(image, 0, scale, "image")
            column_y += 140 + NODE_GAP
            encode = subgraph.add(
                "VAEEncodeTiled", "After Exact Scale · VAE Tiled Encode", (500, column_y), (470, 190),
                inputs=(("pixels", "IMAGE"), ("vae", "VAE")), outputs=(("LATENT", "LATENT"),),
                widgets=VAE_TILE_WIDGETS,
            )
            subgraph.connect(scale, 0, encode, "pixels")
            subgraph.feed("vae", encode, "vae")
            result = encode
            if stage.redraws:
                model_node, positive_node, y = None, None, 0
                if self.distilled:
                    model_node = subgraph.add(
                        "ModelSamplingFlux", f"{ENGINE_LABELS[self.engine]} Shift · {size[0]} × {size[1]}", (1000, y), (470, 170),
                        inputs=(("model", "MODEL"),), outputs=(("MODEL", "MODEL"),),
                        widgets=self.model_sampling_widgets(size[0], size[1], stage.steps),
                    )
                    y += 170 + NODE_GAP
                    positive_node = subgraph.add(
                        "FluxGuidance", f"Guidance · {_number(stage.cfg)}", (1000, y), (470, 80),
                        inputs=(("conditioning", "CONDITIONING"),), outputs=(("CONDITIONING", "CONDITIONING"),),
                        widgets=(("guidance", float(stage.cfg)),),
                    )
                    y += 80 + NODE_GAP
                label = "UPS Hires Redraw" if stage.upscaler else "Hires Redraw"
                sampler = self.card_sampler(subgraph, (1000, y), (470, 350), sampler_number, label, stage)
                subgraph.connect(encode, 0, sampler, "latent_image")
                if model_node:
                    subgraph.feed("model", model_node, "model")
                    subgraph.connect(model_node, 0, sampler, "model")
                    subgraph.feed("positive", positive_node, "conditioning")
                    subgraph.connect(positive_node, 0, sampler, "positive")
                else:
                    subgraph.feed("model", sampler, "model")
                    subgraph.feed("positive", sampler, "positive")
                result = sampler
            subgraph.emit(result, 0, "LATENT")
            title = f"{section:02d} / Hires {index} · {'Model Upscale' if stage.upscaler else 'Resize'}"
        right = max(node["pos"][0] + node["size"][0] for node in subgraph.nodes)
        bottom = max(node["pos"][1] + node["size"][1] for node in subgraph.nodes)
        subgraph.group(title, (-40, -80, right + 80, bottom + 120), CARD_COLORS[kind])
        subgraph.assign_order()
        self.subgraphs.append(subgraph)
        return subgraph

    def card_sampler(self, subgraph, pos, size, sampler_number, label, stage: ResampleStage):
        sampler = subgraph.add(
            "KSampler", f"Sampler {sampler_number} · {label}", pos, size,
            inputs=(
                ("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                ("latent_image", "LATENT"), ("seed", "INT", True),
            ),
            outputs=(("LATENT", "LATENT"),),
            widgets=self.ksampler_widgets(stage.seed, stage.steps, stage.cfg, stage.sampler, stage.scheduler, stage.denoise),
        )
        subgraph.feed("negative", sampler, "negative")
        subgraph.feed("seed", sampler, "seed")
        return sampler

    def note_omissions(self):
        for stage in self.facts.stages:
            if isinstance(stage, ResampleStage):
                if stage.tiled:
                    self.notes.append(
                        f"Resample {stage.number} ran as a USDU tiled redraw in XirAI; it has no native ComfyUI "
                        "equivalent and is not in this graph."
                    )
            elif stage == "adetailer":
                self.notes.append("ADetailer ran in XirAI; it has no native ComfyUI equivalent and is not in this graph.")
            elif stage == "rtx":
                self.notes.append("RTX VSR ran in XirAI; it has no native ComfyUI equivalent and is not in this graph.")
        if self.facts.background_removal:
            self.notes.append("XirAI removed the background after sampling; that step is not in this graph.")

    def build_final(self, left, latent, vae, latent_size, final_size, last_resize):
        x = left + 30
        entries = [{
            "type": "VAEDecodeTiled", "title": "Final VAE Decode", "size": (340, 220),
            "inputs": (("samples", "LATENT"), ("vae", "VAE")), "outputs": (("IMAGE", "IMAGE"),),
            "widgets": VAE_TILE_WIDGETS,
        }]
        exact = tuple(final_size) != tuple(latent_size)
        if exact:
            method = IMAGE_SCALE_METHODS.get(last_resize, "lanczos")
            entries.append({
                "type": "ImageScale", "title": f"Exact Output · {final_size[0]} × {final_size[1]}", "size": (340, 140),
                "inputs": (("image", "IMAGE"),), "outputs": (("IMAGE", "IMAGE"),),
                "widgets": (("upscale_method", method), ("width", int(final_size[0])), ("height", int(final_size[1])), ("crop", "disabled")),
            })
        entries.append({
            "type": "SaveImage", "title": "Save Image", "size": (340, 120),
            "inputs": (("images", "IMAGE"),), "outputs": (("images", "IMAGE"),),
            "widgets": (("filename_prefix", f"XirAI-{self.engine}"),),
        })
        sections = []
        if self.requirements:
            sections.append("Requires:\n" + "\n".join(f"- {line}" for line in dict.fromkeys(self.requirements)))
        if self.notes:
            sections.append("Not reproduced by this graph:\n" + "\n".join(f"- {line}" for line in dict.fromkeys(self.notes)))
        if sections:
            text = "\n\n".join(sections)
            lines = sum(1 + len(line) // 44 for line in text.split("\n"))
            entries.append({
                "type": "Note", "title": "XirAI Notes", "size": (340, 40 + 18 * lines),
                "widgets": (("text", text),), "colors": NOTE_COLORS, "properties": {},
            })
        nodes = _column(self.root, x, 70, entries)
        decode = nodes[0]
        self.root.connect(latent, 0, decode, "samples")
        self.root.connect(vae[0], vae[1], decode, "vae")
        image = decode
        if exact:
            self.root.connect(decode, 0, nodes[1], "image")
            image = nodes[1]
        save = nodes[2] if exact else nodes[1]
        self.root.connect(image, 0, save, "images")
        for node in nodes:
            if node["type"] == "Note":
                node.pop("widgets_values_named", None)
        self.bottoms["final"] = nodes[-1]["pos"][1] + nodes[-1]["size"][1]

    # -- output ------------------------------------------------------------------------------

    def serialize(self, total_width, height):
        self.root.assign_order()
        self.root.finish_outputs()
        definitions = []
        for subgraph in self.subgraphs:
            subgraph.finish_outputs()
            definition = subgraph.definition()
            definition["state"] = {
                "lastGroupId": self.ids.group,
                "lastNodeId": self.ids.node,
                "lastLinkId": self.ids.link,
                "lastRerouteId": 0,
            }
            definitions.append(definition)
        scale = max(0.2, min(0.65, 1800 / max(1, total_width), 1000 / max(1, height)))
        workflow = {
            "id": str(uuid.uuid5(WORKFLOW_NAMESPACE, f"{self.facts.identity}/workflow")),
            "revision": 0,
            "last_node_id": self.ids.node,
            "last_link_id": self.ids.link,
            "nodes": self.root.nodes,
            "links": [list(link) for link in self.root.links],
            "groups": self.root.groups,
        }
        if definitions:
            workflow["definitions"] = {"subgraphs": definitions}
        workflow.update({
            "config": {},
            "extra": {
                "ds": {"scale": round(scale, 3), "offset": [120, 90]},
                "xirai": {"format": 1, "engine": self.engine},
            },
            "version": WORKFLOW_FORMAT_VERSION,
        })
        return workflow


def workflow_supported(request) -> bool:
    """Text-to-image runs of the seven engines. Image-to-image and post-processing runs start from a
    picture this graph would have to fake, so they carry no workflow."""
    source = getattr(request, "source_image", None)
    return (
        getattr(request, "engine", None) in WORKFLOW_ENGINES
        and not getattr(request, "postprocess_only", False)
        and not getattr(source, "enabled", False)
    )


def build_workflow(request, facts: WorkflowFacts) -> dict | None:
    """The ComfyUI UI workflow for one saved picture, or ``None`` where no honest graph exists."""
    if not workflow_supported(request):
        return None
    workflow = _WorkflowBuilder(request, facts).build()
    validate_workflow(workflow)
    return workflow


def validate_workflow(workflow: dict) -> None:
    """Structural invariants: every link names real endpoints, ids are unique, nothing is absolute."""
    graphs = [("root", workflow["nodes"], workflow["links"], False)]
    for definition in workflow.get("definitions", {}).get("subgraphs", []):
        graphs.append((definition["name"], definition["nodes"], definition["links"], True))
    seen_nodes, seen_links = set(), set()
    for label, nodes, links, inner in graphs:
        by_id = {}
        for node in nodes:
            if node["id"] in seen_nodes:
                raise ValueError(f"{label}: duplicate node id {node['id']}")
            seen_nodes.add(node["id"])
            by_id[node["id"]] = node
        for link in links:
            if inner:
                link_id, origin, origin_slot, target, target_slot = (
                    link["id"], link["origin_id"], link["origin_slot"], link["target_id"], link["target_slot"]
                )
            else:
                link_id, origin, origin_slot, target, target_slot = link[:5]
            if link_id in seen_links:
                raise ValueError(f"{label}: duplicate link id {link_id}")
            seen_links.add(link_id)
            if origin in by_id:
                if link_id not in (by_id[origin]["outputs"][origin_slot]["links"] or []):
                    raise ValueError(f"{label}: link {link_id} is not listed on its origin")
            elif not (inner and origin == _Subgraph.INPUT_NODE):
                raise ValueError(f"{label}: link {link_id} starts at a missing node")
            if target in by_id:
                if by_id[target]["inputs"][target_slot]["link"] != link_id:
                    raise ValueError(f"{label}: link {link_id} is not listed on its target")
            elif not (inner and target == _Subgraph.OUTPUT_NODE):
                raise ValueError(f"{label}: link {link_id} ends at a missing node")
    if workflow["last_node_id"] < max(seen_nodes, default=0) or workflow["last_link_id"] < max(seen_links, default=0):
        raise ValueError("workflow id counters are behind its contents")
    for _label, nodes, _links, _inner in graphs:
        for node in nodes:
            named = node.get("widgets_values_named") or {}
            for field in ASSET_WIDGETS:
                value = named.get(field)
                if isinstance(value, str) and _looks_absolute(value):
                    raise ValueError(f"{node['type']}.{field} holds an absolute path")


# Widgets that name a file on disk; every one must stay a relative ID.
ASSET_WIDGETS = ("ckpt_name", "unet_name", "clip_name", "clip_name1", "clip_name2", "vae_name", "lora_name", "model_name")


def _is_gguf(name: str) -> bool:
    return name.lower().endswith(".gguf")


def _looks_absolute(value: str) -> bool:
    return bool(re.match(r"^(?:[A-Za-z]:|[\\/])", value)) or ".." in re.split(r"[\\/]", value)
