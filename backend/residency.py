"""Where the diffusion model lives, decided per stage instead of once per job.

ComfyUI recomputes this before every node (`comfy/model_management.py::load_models_gpu`): it sums
what the step needs, frees what is not in use, and derives from *current* free memory how much weight
may stay resident, streaming whatever does not fit. What Xiria had instead was one admission at load
time, sized against the most demanding stage of the whole chain, and a single fallback — Diffusers'
per-submodule offload — for everything that did not fit. That is why a job whose cheap 768x1280 base
pass had room to spare still ran it at the speed its 1152x1920 redraw could afford.

This module owns the mechanism and the choice. Five rungs, in order of how much of the UNet stays on
the card:

    T0  resident        pipeline.to("cuda")                      every component on the card
    T1  component       enable_model_cpu_offload()               UNet resident, others swap per call
    T2  block_stream    group offload, block_level, K blocks     a few blocks at a time
    T3  leaf_stream     group offload, leaf_level, CUDA stream   one leaf at a time, prefetched
    T4  submodule       enable_sequential_cpu_offload()          one submodule, synchronous

**The plan's tier table has T2 above T3 and reaches for T3 only when T2 will not fit. Measurement
says the opposite, and the order here follows the measurement.** On an RTX 4060 Laptop with
`waiIllustriousSDXL_v170` and six LoRAs (UNet 5.221 GiB), 8 steps with the warm-up step excluded:

                        768x1280                 1152x1920
    T1 component        1.924 s/it  5.69 GiB     4.899 s/it  6.15 GiB
    T2 block_level      2.037 s/it  5.16 GiB     5.147 s/it  5.43 GiB
    T3 leaf_level       2.091 s/it  0.38 GiB     4.589 s/it  0.83 GiB
    T4 sequential       6.349 s/it  0.37 GiB     9.096 s/it  0.82 GiB

T2 buys almost nothing: a UNet's `ModuleList` children are wildly uneven — SDXL's `up_blocks[0]` is
2.449 GiB on its own — so a block group plus its prefetch is nearly the whole model. T3 costs about
what residency costs and holds a seventh of the memory, because a leaf is small and the stream
prefetches the next one under the current one's compute. That is the same trade ComfyUI's
`DynamicVRAM` makes, and it is why T3 sits directly under T1 here and T2 is kept for the case that
justifies it: a device with no usable CUDA stream, where leaf-level would degenerate into T4's
synchronisation per layer and block grouping is the better of two bad options.

The four sampling tiers were checked for equivalence at both canvases: same seed, same conditioning,
**bit-identical latents**. Residency is weight placement, and §8.4 of the plan requires it to stay
that way — if a tier is ever found to move the fixed-seed hash, it is a bug in that tier and it
should be removed from the ladder rather than explained.
"""

import gc

import torch


GIB = 1024**3
MIB = 1024**2

T0_RESIDENT = "T0"
T1_COMPONENT = "T1"
T2_BLOCK_STREAM = "T2"
T3_LEAF_STREAM = "T3"
T4_SUBMODULE = "T4"

# Most resident first. `plan_for_stage` walks it in this order and takes the first that fits.
TIER_ORDER = (T0_RESIDENT, T1_COMPONENT, T3_LEAF_STREAM, T2_BLOCK_STREAM, T4_SUBMODULE)

TIER_LABELS = {
    T0_RESIDENT: "T0 常驻",
    T1_COMPONENT: "T1 组件调度",
    T2_BLOCK_STREAM: "T2 分块流式",
    T3_LEAF_STREAM: "T3 叶级流式",
    T4_SUBMODULE: "T4 逐层卸载",
}

# What each rung reports as the job's memory mode, so the existing health and job fields keep
# meaning what they meant. The streaming rungs are not `normal_vram` — the UNet is not resident —
# but they are emphatically not the per-submodule path either, which is what `low_vram` used to
# imply on its own; `offload_mode` is what tells those three apart.
TIER_MEMORY_MODES = {
    T0_RESIDENT: "high_vram",
    T1_COMPONENT: "normal_vram",
    T2_BLOCK_STREAM: "low_vram",
    T3_LEAF_STREAM: "low_vram",
    T4_SUBMODULE: "low_vram",
}

TIER_OFFLOAD_MODES = {
    T0_RESIDENT: "none",
    T1_COMPONENT: "model",
    T2_BLOCK_STREAM: "block_stream",
    T3_LEAF_STREAM: "leaf_stream",
    T4_SUBMODULE: "sequential",
}

# Diffusers registers these three under `_diffusers_hook`, and `pipeline.remove_all_hooks()` removes
# none of them — it only knows about accelerate's `_hf_hook`. Removing them in this order, after
# offloading every live group, is what `anima_pipeline._remove_transformer_group_offload` does.
GROUP_OFFLOAD_HOOKS = ("lazy_prefetch_group_offloading", "layer_execution_tracker", "group_offloading")

# Promote only with room to spare, so a tier change cannot chase a number that is drifting.
PROMOTION_MARGIN = 1.15
# What the allocator cap leaves unclaimed below the memory that was free when it was set.
ALLOCATOR_MARGIN_BYTES = 256 * 1024**2
# Demote when free memory falls to within this of the largest thing a single onload has to move.
DEMOTION_HEADROOM_BYTES = 400 * MIB


RESIDENCY_TIER_MEMORY_MODES = TIER_MEMORY_MODES
RESIDENCY_TIER_OFFLOAD_MODES = TIER_OFFLOAD_MODES


def tier_for_memory_mode(mode):
    """The rung `apply_memory_strategy` leaves installed for an admission mode.

    Admission still decides where a job starts — this is only how that starting point is named in
    the ladder, so the controller knows what is already on the pipeline and does not reinstall it.
    """
    return {
        "high_vram": T0_RESIDENT,
        "normal_vram": T1_COMPONENT,
        "low_vram": T4_SUBMODULE,
        "ultra_low_vram": T4_SUBMODULE,
    }.get(mode, T4_SUBMODULE)


def group_offload_active(module):
    registry = getattr(module, "_diffusers_hook", None)
    return registry is not None and registry.get_hook("group_offloading") is not None


def remove_group_offload(module):
    """Take diffusers' group-offload hooks off a module and bring its weights home.

    Every live group is offloaded first: the hooks are what would otherwise move those weights back,
    and removing them with a group still onloaded strands that group's tensors on the card. This is
    the first of the plan's two hook-lifecycle traps (P1-3), and it has to run before any accelerate
    mechanism is installed, because `apply_group_offloading` and `enable_model_cpu_offload` each
    refuse to be installed over the other.
    """
    if module is None:
        return False
    groups = {}
    for child in module.modules():
        registry = getattr(child, "_diffusers_hook", None)
        hook = registry.get_hook("group_offloading") if registry is not None else None
        if hook is not None:
            groups[id(hook.group)] = hook.group
    if not groups:
        return False
    for group in groups.values():
        group.offload_()
    registry = getattr(module, "_diffusers_hook", None)
    if registry is not None:
        for name in GROUP_OFFLOAD_HOOKS:
            registry.remove_hook(name, recurse=True)
    module.to("cpu")
    return True


def module_bytes(module):
    """Weight bytes of a module, counting shared storage once."""
    if module is None:
        return 0
    seen = set()
    total = 0
    for tensor in (*module.parameters(), *module.buffers()):
        pointer = tensor.data_ptr() or id(tensor)
        if pointer in seen:
            continue
        seen.add(pointer)
        total += tensor.numel() * tensor.element_size()
    return total


def largest_block_group_bytes(module, blocks_per_group=1):
    """The biggest set of consecutive `ModuleList`/`Sequential` children one onload has to move.

    Diffusers groups exactly those children (`_apply_group_offloading_block_level`); anything else —
    a UNet's `mid_block`, its embedders, its input and output convolutions — lands in one "unmatched"
    group that is onloaded with the top-level forward. Both are candidates for the largest single
    move, so both are measured.
    """
    if module is None:
        return 0
    largest = 0
    unmatched = 0
    for _name, child in module.named_children():
        if isinstance(child, (torch.nn.ModuleList, torch.nn.Sequential)):
            entries = list(child)
            for index in range(0, len(entries), max(1, blocks_per_group)):
                group = entries[index:index + max(1, blocks_per_group)]
                largest = max(largest, sum(module_bytes(entry) for entry in group))
        else:
            unmatched += module_bytes(child)
    return max(largest, unmatched)


def largest_leaf_bytes(module):
    """The biggest single leaf a leaf-level onload has to move."""
    if module is None:
        return 0
    largest = 0
    for child in module.modules():
        if any(True for _ in child.children()):
            continue
        largest = max(largest, module_bytes(child))
    return largest


def cuda_streams_available():
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


class ResidencyController:
    """Owns the diffusion model's placement for one loaded pipeline.

    Sole owner is the point: the plan's second hook-lifecycle trap (P1-3) is that the two mechanisms
    refuse to coexist, so exactly one thing may install either of them. Everything that used to
    switch offload behaviour behind the pipeline's back goes through `set_tier`, which removes what
    is there before it installs what is wanted.
    """

    def __init__(self, pipeline, *, reserved_bytes, free_memory, tier=None, stream_capable=None,
                 trace_limit=64, allocator_ceiling_bytes=None, on_transition=None):
        self.pipeline = pipeline
        # Told about every move that changed something, so the owner can put it in front of whoever
        # is watching. A failure to report is never allowed to become a failure to place.
        self.on_transition = on_transition
        self.reserved_bytes = int(reserved_bytes)
        # The user's VRAM wall when one is configured, or None for the whole device. The cap below
        # never goes above it, and lifting the cap puts it back rather than removing it.
        self.allocator_ceiling_bytes = allocator_ceiling_bytes
        self.allocator_cap_bytes = None
        # A zero-argument callable returning free bytes, so the controller can be tested without a
        # device and so it always reads the number at the moment it decides rather than at load.
        self._free_memory = free_memory
        self.stream_capable = cuda_streams_available() if stream_capable is None else bool(stream_capable)
        self.tier = tier
        self.blocks_per_group = 0
        self.stage = None
        self.parked = False
        self.trace = []
        self._trace_limit = trace_limit
        self._promoted_this_stage = False
        self.total_weight_bytes = 0
        self.unet_bytes = 0
        self.other_bytes = 0
        self.block_group_bytes = 0
        self.leaf_bytes = 0
        self.measure()

    # ---- measurement -------------------------------------------------------

    @property
    def unet(self):
        return getattr(self.pipeline, "unet", None)

    def measure(self):
        """Size the mount. Called again after adapters are mounted, because they change every figure.

        Order matters and it is the plan's (P1-6): load the LoRA, measure, then install the tier. A
        block-level group carries its adapter submodules with it, so what the adapters add is part of
        what a single onload has to move, not a separate budget.
        """
        components = getattr(self.pipeline, "components", {}) or {}
        seen = set()
        total = 0
        for component in components.values():
            if not isinstance(component, torch.nn.Module):
                continue
            for tensor in (*component.parameters(), *component.buffers()):
                pointer = tensor.data_ptr() or id(tensor)
                if pointer in seen:
                    continue
                seen.add(pointer)
                total += tensor.numel() * tensor.element_size()
        self.total_weight_bytes = total
        self.unet_bytes = module_bytes(self.unet)
        # Everything that is not the UNet: the two text towers and the autoencoder, 1.77 GiB
        # together for SDXL. A streaming rung keeps them on the card rather than on the host —
        # see `_install` for why that is not the saving it looks like.
        self.other_bytes = max(0, total - self.unet_bytes)
        self.block_group_bytes = largest_block_group_bytes(self.unet, 1)
        self.leaf_bytes = largest_leaf_bytes(self.unet)
        return self

    def observe(self):
        """Free bytes right now, as the caller's reader reports them."""
        return int(self._free_memory())

    # ---- what each rung needs ---------------------------------------------

    def requirement(self, tier, inference_bytes):
        """Bytes this rung needs free to run a stage whose activations are `inference_bytes`.

        The streaming rungs budget two of whatever a single onload moves, because the stream holds
        the next one under the current one — that is the memory the overlap costs, and it is the
        whole reason the overlap is free in time.
        """
        inference_bytes = int(inference_bytes)
        if tier == T0_RESIDENT:
            return self.total_weight_bytes + inference_bytes + self.reserved_bytes
        if tier == T1_COMPONENT:
            return self.unet_bytes + inference_bytes + self.reserved_bytes
        if tier == T2_BLOCK_STREAM:
            return self.other_bytes + 2 * self.block_group_bytes + inference_bytes + self.reserved_bytes
        if tier == T3_LEAF_STREAM:
            return self.other_bytes + 2 * self.leaf_bytes + inference_bytes + self.reserved_bytes
        return inference_bytes + self.reserved_bytes

    def candidate_tiers(self):
        """The ladder, minus the rungs this device cannot run.

        Without a usable CUDA stream, leaf-level offloading loses its prefetch and becomes T4 with
        extra bookkeeping, so it drops out and block grouping takes its place.
        """
        if self.stream_capable:
            return (T0_RESIDENT, T1_COMPONENT, T3_LEAF_STREAM, T4_SUBMODULE)
        return (T0_RESIDENT, T1_COMPONENT, T2_BLOCK_STREAM, T4_SUBMODULE)

    def next_lower_tier(self):
        """The rung below the current one on this device's ladder, or None at the bottom."""
        ladder = self.candidate_tiers()
        if self.tier not in ladder:
            return None
        index = ladder.index(self.tier)
        return ladder[index + 1] if index + 1 < len(ladder) else None

    def select(self, inference_bytes, free_bytes=None, margin=1.0):
        free_bytes = self.observe() if free_bytes is None else int(free_bytes)
        for tier in self.candidate_tiers():
            if self.requirement(tier, inference_bytes) * margin <= free_bytes:
                return tier
        return T4_SUBMODULE

    # ---- stage planning ----------------------------------------------------

    def reclaim(self):
        """Hand the allocator's cached blocks back to the driver.

        ComfyUI does this before every node (`free_memory`), and on a card this size it is not
        bookkeeping: leaf streaming allocates and frees a weight per layer, so the caching allocator
        grows to fill whatever is free and *keeps* it. The redraw that follows a decode then starts
        with 7.8 GiB of the card already reserved, and every new allocation has to evict a cached
        block first — measured at 6.6 s/it against 4.1 s/it for the same stage on a clean allocator.
        A stage boundary is the one place where nothing is in flight and this is free.
        """
        gc.collect()
        if cuda_streams_available():
            torch.cuda.empty_cache()

    def cap_allocator(self):
        """Hold the caching allocator to our share of the card plus what is free right now.

        A streaming rung frees a weight per layer, and `record_stream` defers each free until the
        side stream retires — so the allocator's cache grows into whatever the driver will give it.
        At the default fraction the driver will give it more than the card has: Windows serves the
        excess out of host memory rather than failing. That is how a redraw that ran at 4.1 s/it on
        a fresh process ran at 7.9 s/it on the next job, pegged at 7.84 GiB of an 8 GiB card with the
        desktop's share on top. Capped here, the allocator recycles its own cache instead, and an
        allocation that genuinely does not fit raises — which the OOM escalation can answer with
        the generator state restored — rather than silently running through PCIe.

        Measured against the free memory of the moment rather than a fixed reserve, because what
        other processes hold is exactly what a fixed reserve cannot know.
        """
        if not cuda_streams_available():
            return None
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        ceiling = self.allocator_ceiling_bytes or total_bytes
        cap = min(ceiling, torch.cuda.memory_reserved() + free_bytes - ALLOCATOR_MARGIN_BYTES)
        torch.cuda.set_per_process_memory_fraction(max(0.05, min(1.0, cap / total_bytes)), 0)
        self.allocator_cap_bytes = int(cap)
        return self.allocator_cap_bytes

    def uncap_allocator(self):
        """Give the allocator its configured ceiling back, for stages that want the whole card."""
        if self.allocator_cap_bytes is None or not cuda_streams_available():
            self.allocator_cap_bytes = None
            return
        total_bytes = torch.cuda.get_device_properties(0).total_memory
        ceiling = self.allocator_ceiling_bytes or total_bytes
        torch.cuda.set_per_process_memory_fraction(max(0.05, min(1.0, ceiling / total_bytes)), 0)
        self.allocator_cap_bytes = None

    def note_parked(self, stage=None, reason="park_for_decode"):
        """Record that everything has just been moved off the card for another stage.

        Parking the sampler so the autoencoder can have the card is a residency transition in every
        sense the plan means — it is the plan's own P1-5 ("park diffusion model for VAE") and
        ComfyUI's `free_memory` before `VAE.decode` — so it belongs in the trace beside the rung
        changes rather than happening invisibly.
        """
        if self.parked or self.tier is None:
            return False
        self.uncap_allocator()
        self.parked = True
        self._record(stage or self.stage, self.tier, "parked", self.observe(), reason)
        return True

    def note_restored(self, stage=None, reason="restore_after_stage"):
        if not self.parked:
            return False
        self.parked = False
        self._record(stage or self.stage, "parked", self.tier, self.observe(), reason)
        return True

    def begin_job(self):
        """Start a fresh trace: the record published on a job describes that job."""
        self.trace = []
        self.stage = None
        self._promoted_this_stage = False
        return self

    def plan_for_stage(self, stage, inference_bytes, canvas=None):
        """Choose and install the rung this stage should run on, and say so in the trace.

        Called at stage boundaries only. A promotion is weight placement between steps, never inside
        a forward, so the arithmetic of the stage it precedes is untouched.
        """
        self.stage = stage
        self._promoted_this_stage = False
        self.reclaim()
        free_bytes = self.observe()
        wanted = self.select(inference_bytes, free_bytes=free_bytes)
        if wanted == self.tier:
            self._record(stage, self.tier, self.tier, free_bytes, "stage_plan_unchanged",
                         inference_bytes=inference_bytes, canvas=canvas)
            self.cap_allocator()
            return self.tier
        reason = "stage_plan_promote" if self._is_promotion(wanted) else "stage_plan_demote"
        self.set_tier(wanted, reason=reason, stage=stage, inference_bytes=inference_bytes, canvas=canvas)
        self.cap_allocator()
        return self.tier

    def promote_if_fits(self, inference_bytes, stage=None):
        """Move up one rung when there is room to spare — at most once per stage."""
        if self._promoted_this_stage or self.tier is None:
            return False
        ladder = self.candidate_tiers()
        if self.tier not in ladder or ladder.index(self.tier) == 0:
            return False
        upper = ladder[ladder.index(self.tier) - 1]
        free_bytes = self.observe()
        if self.requirement(upper, inference_bytes) * PROMOTION_MARGIN > free_bytes:
            return False
        self._promoted_this_stage = True
        return self.set_tier(upper, reason="promote_fits", stage=stage or self.stage,
                             inference_bytes=inference_bytes)

    def demote_if_pressed(self, inference_bytes, stage=None):
        """Drop a rung when free memory is down to what a single onload has to move.

        A hard OOM is still the existing escalation's business — it restores the generator state and
        replays, which is what keeps the picture the same. This is the cheaper answer that happens
        at a safe boundary before the allocation fails.
        """
        if self.tier is None:
            return False
        ladder = self.candidate_tiers()
        if self.tier not in ladder or ladder.index(self.tier) == len(ladder) - 1:
            return False
        largest_move = max(self.leaf_bytes, self.block_group_bytes if self.tier == T2_BLOCK_STREAM else 0)
        free_bytes = self.observe()
        if free_bytes >= largest_move + DEMOTION_HEADROOM_BYTES:
            return False
        lower = ladder[ladder.index(self.tier) + 1]
        return self.set_tier(lower, reason="demote_pressed", stage=stage or self.stage,
                             inference_bytes=inference_bytes)

    # ---- mechanism ---------------------------------------------------------

    def set_tier(self, tier, *, reason, stage=None, inference_bytes=None, canvas=None,
                 blocks_per_group=None):
        """Install a rung. Idempotent, and ordered so the two mechanisms never overlap.

        Remove what is installed, empty the cache only when dropping to a smaller rung (going up,
        the memory is about to be used anyway and the call costs a synchronisation), then install.
        """
        if tier == self.tier:
            return False
        previous = self.tier
        downgrading = self._is_demotion(tier)
        self.release(empty_cache=downgrading)
        self._install(tier, blocks_per_group)
        self.tier = tier
        free_bytes = self.observe()
        self._record(stage if stage is not None else self.stage, previous, tier, free_bytes, reason,
                     inference_bytes=inference_bytes, canvas=canvas)
        return True

    def reinstall(self, tier=None):
        """Put a rung back on a pipeline whose hooks have just been torn down.

        `reconfigure_memory_strategy` and the parked-pipeline restore both strip every mechanism and
        then ask for one to be installed again. They must ask *here*: the mode-to-mechanism ladder
        they used to run would put a job the controller had placed on a streaming rung back onto the
        per-submodule path, because both streaming rungs report themselves as `low_vram`. That is
        the plan's "the controller is the sole owner" (P1-3), and it is worth exactly the difference
        between 5.6 s/it and 13 s/it on the reference job's redraw.
        """
        tier = tier or self.tier
        if tier is None:
            return False
        self._install(tier, self.blocks_per_group or None)
        self.tier = tier
        self.note_restored()
        return True

    def release(self, empty_cache=True):
        """Take off whatever mechanism is installed, leaving the pipeline on the host.

        Both halves are needed and neither removes the other: `remove_all_hooks` knows only about
        accelerate's, `remove_group_offload` only about diffusers'.
        """
        pipeline = self.pipeline
        self.uncap_allocator()
        remove_group_offload(self.unet)
        try:
            pipeline.remove_all_hooks()
        except AttributeError:
            pass
        pipeline.to("cpu", silence_dtype_warnings=True)
        if empty_cache:
            gc.collect()
            if cuda_streams_available():
                torch.cuda.empty_cache()

    def _install(self, tier, blocks_per_group=None):
        pipeline = self.pipeline
        if tier == T0_RESIDENT:
            pipeline.to("cuda")
            self.blocks_per_group = 0
            return
        if tier == T1_COMPONENT:
            pipeline.enable_model_cpu_offload(device="cuda")
            self.blocks_per_group = 0
            return
        if tier == T4_SUBMODULE:
            pipeline.enable_sequential_cpu_offload(device="cuda")
            self.blocks_per_group = 0
            return
        # Group offload is installed on the UNet alone, so every other component has to be placed
        # by hand. They go to the card rather than the host: a later stage may re-encode a prompt
        # or decode a picture, `_execution_device` resolves to the group's onload device either
        # way, and a component left on the host would meet CUDA inputs. Together they are 1.77 GiB
        # against the 5.22 GiB the UNet stops holding, so the trade is still overwhelmingly worth
        # making — and `requirement()` budgets them, so a card that cannot afford them falls to T4
        # instead, where accelerate places everything itself.
        for name, component in (getattr(pipeline, "components", {}) or {}).items():
            if name != "unet" and isinstance(component, torch.nn.Module):
                component.to("cuda")
        leaf = tier == T3_LEAF_STREAM
        # `_apply_group_offloading_block_level` forces K to 1 whenever a stream is in play, and says
        # so in a warning. Asking for 1 is honest about what will happen; without a stream the
        # requested K stands and is the dial the plan describes.
        requested = 1 if self.stream_capable else max(1, int(blocks_per_group or 2))
        self.unet.enable_group_offload(
            onload_device=torch.device("cuda"),
            offload_device=torch.device("cpu"),
            offload_type="leaf_level" if leaf else "block_level",
            use_stream=self.stream_capable,
            non_blocking=self.stream_capable,
            # `record_stream` is documented as "faster at the expense of slightly more memory", and
            # on this workload the speed is not marginal: turning it off put the base pass back from
            # 0.72 s/it to 1.19 s/it and the whole job from 121 s to 149 s. The memory it costs is
            # allocator cache rather than live tensors, and `reclaim()` hands that back at every
            # stage boundary, so the trade is taken.
            record_stream=self.stream_capable,
            **({} if leaf else {"num_blocks_per_group": requested}),
        )
        self.blocks_per_group = 0 if leaf else requested

    # ---- reporting ---------------------------------------------------------

    def _is_promotion(self, tier):
        ladder = TIER_ORDER
        if self.tier is None or tier not in ladder or self.tier not in ladder:
            return False
        return ladder.index(tier) < ladder.index(self.tier)

    def _is_demotion(self, tier):
        ladder = TIER_ORDER
        if self.tier is None or tier not in ladder or self.tier not in ladder:
            return False
        return ladder.index(tier) > ladder.index(self.tier)

    def _record(self, stage, from_tier, to_tier, free_bytes, reason, inference_bytes=None, canvas=None):
        entry = {
            # A move made before any stage has been planned is the job bringing a parked model back.
            "stage": stage or "job_start",
            "from": from_tier,
            "to": to_tier,
            "blocks_per_group": self.blocks_per_group,
            "free_bytes": int(free_bytes),
            "free_gb": round(free_bytes / GIB, 2),
            "required_bytes": 0 if to_tier == "parked" else int(self.requirement(to_tier, inference_bytes or 0)),
            "inference_bytes": int(inference_bytes) if inference_bytes is not None else None,
            "canvas": list(canvas) if canvas else None,
            "reason": reason,
            "changed": from_tier != to_tier,
        }
        self.trace.append(entry)
        # A long job would otherwise grow this without bound; the oldest records are the ones a
        # report needs least.
        if len(self.trace) > self._trace_limit:
            del self.trace[:-self._trace_limit]
        if entry["changed"] and self.on_transition is not None:
            try:
                self.on_transition(entry)
            except Exception:
                pass
        return entry

    def snapshot(self):
        return {
            "tier": self.tier,
            "tier_label": TIER_LABELS.get(self.tier),
            "offload_mode": TIER_OFFLOAD_MODES.get(self.tier),
            "memory_mode": TIER_MEMORY_MODES.get(self.tier),
            "blocks_per_group": self.blocks_per_group,
            "stage": self.stage,
            "stream_capable": self.stream_capable,
            "parked": self.parked,
            "allocator_cap_bytes": self.allocator_cap_bytes,
            "unet_bytes": self.unet_bytes,
            "total_weight_bytes": self.total_weight_bytes,
            "other_bytes": self.other_bytes,
            "block_group_bytes": self.block_group_bytes,
            "leaf_bytes": self.leaf_bytes,
            "transitions": sum(1 for entry in self.trace if entry["changed"]),
        }
