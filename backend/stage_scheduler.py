"""AUTO · 动态决策: one VRAM decision for every stage of every engine (QPlan P8).

`auto` used to mean "pick HIGH, NORMAL or LOW once, at load, against the most demanding stage of the
whole chain", and the stage controller underneath it then re-asked a narrower question with numbers
that did not agree with the ones admission used. The 2026-09-26 12 GB report is what that looks like
from outside: HIGH admission with 0.45 GiB to spare, a base pass demoted to streaming because the
controller could not see that the weights it was judging were already its own, a redraw promoted
back to residency with no margin at all, and an allocator cap installed 20 MiB above what was
already allocated — so the next CLIP forward was refused a 2 MiB block. Three scopes (the device
budget, the allocator cap and the stage peak) were each right about themselves and never reconciled
inside one decision.

This module is that one decision, and nothing else. It owns no mechanism: the residency controller
installs tiers and the native runtimes move their transformer, and both come here to be told what
to do. Everything it needs arrives as numbers, which is what lets every VRAM band from 6 to 24 GB be
exercised without a device.

Four rules carry it.

* **One budget.** A stage is judged against a single `StageBudget` reading: driver free memory,
  what PyTorch holds and caches, the platform reserve, the user's wall and — the part that was
  missing — the weights this pipeline already has on the card. Those are counted as available to
  the placement being judged, so a resident model is not told it cannot afford itself.
* **Hysteresis.** A stage stays where it is while it fits; it moves up only with 15% to spare
  (`PROMOTION_MARGIN`); it moves down only when it does not fit. Placement is fixed for the whole
  stage — nothing here ever runs inside a sampling iteration.
* **The cap invariant.** After a move the allocator cap must hold what is allocated, what this
  stage will add, and a safety margin sized for the card. A decision that would leave the cap
  hugging the allocation is not a decision this module returns; one that turns out that way after
  the move is demoted before the stage runs.
* **A floor, not a cliff.** AUTO's lowest rung is the streaming / per-submodule path. The serial-CFG
  survival mode (`ultra_low_vram`) stays manual; when even the floor cannot hold a stage, the answer
  is a readable stage error, never a smaller canvas, fewer steps or a silent CPU fallback.
"""

from dataclasses import dataclass, field


GIB = 1024**3
MIB = 1024**2

AUTO_MODE = "auto"
AUTO_LABEL = "AUTO · 动态决策"
AUTO_LABEL_EN = "AUTO · Dynamic Decisions"

# The recommended minimum is 6 GB. A card sold as 6 GB reports about 6.0 GiB, and the next size down
# is 4 GB, so half a gigabyte of tolerance — the same the preset ranges use — flags only the cards
# that really are below it. AUTO stays selectable there; the warning is all that changes.
RECOMMENDED_MINIMUM_VRAM_BYTES = 6 * GIB
BELOW_RECOMMENDED_THRESHOLD_BYTES = int(5.5 * GIB)
BELOW_RECOMMENDED_WARNING = "低于推荐配置，可能无法完成任务"

# Kept equal to `residency.PROMOTION_MARGIN` and `residency.DEMOTION_HEADROOM_BYTES`; they are
# restated rather than imported so this module stays free of torch and can be tested anywhere.
PROMOTION_MARGIN = 1.15
DEMOTION_HEADROOM_BYTES = 400 * MIB
# What `cap_allocator` leaves unclaimed below the free memory of the moment (residency).
ALLOCATOR_MARGIN_BYTES = 256 * MIB

# What one SD/iL text-encoding pass allocates beyond the tower weights. A CLIP forward over a few
# 77-token chunks is megabytes of activations; the headroom is for the cuBLAS workspace and the
# allocator block that has to be found for them, which is exactly what the 12 GB report ran out of.
TEXT_ENCODE_TRANSIENT_BYTES = 256 * MIB

TIER_PATHS = {"T0": "resident", "T1": "component", "T2": "block", "T3": "leaf", "T4": "submodule"}
PATH_LABELS = {
    "resident": "常驻",
    "component": "组件调度",
    "block": "分块流式",
    "leaf": "叶级流式",
    "submodule": "逐层卸载",
    # A decode boundary is not a rung: the model either left the card first or shared it.
    "evicted": "卸载后解码",
    "shared": "与模型同驻解码",
    # An upscaler boundary: whether the diffusion model was parked before the upscaler ran.
    "parked": "停放后放大",
}
# A native runtime's transformer is either resident for the stage or streamed a block at a time.
NATIVE_RESIDENT = "resident"
NATIVE_BLOCK = "block"


def allocator_safety_margin_bytes(total_bytes):
    """How much a stage keeps between its estimated peak and the allocator cap, by VRAM band.

    The estimates are generous but not measurements of every allocation a stage makes: the cuBLAS
    and cuDNN workspaces, a fragmented block that has to be split, a prompt longer than the one the
    figures were taken with. On a small card that difference is what separates finishing from a
    refused 2 MiB block, and on a large card it is noise, so the margin shrinks as the card grows.
    The 24 GB band keeps the controller's original 256 MiB, which is what "unchanged for residency
    decisions" means there. The 8 GB band keeps less than the 12 GB band on purpose: a 0.75 GiB
    margin already costs an 8 GB card a real fraction of its streaming headroom, and the margin is
    never allowed to become a blanket demotion.
    """
    total = max(0, int(total_bytes or 0))
    if total < 9 * GIB:
        return int(0.75 * GIB)
    if total < 13 * GIB:
        return GIB
    if total < 20 * GIB:
        return GIB // 2
    return ALLOCATOR_MARGIN_BYTES


def below_recommended_vram(total_bytes):
    total = int(total_bytes or 0)
    return 0 < total < BELOW_RECOMMENDED_THRESHOLD_BYTES


def path_for_tier(tier):
    return TIER_PATHS.get(tier)


def path_label(path):
    return PATH_LABELS.get(path, path)


def auto_label(path=None):
    """What a job running under AUTO calls itself: never a manual tier's name (QPlan P8 §2)."""
    return f"{AUTO_LABEL} · {path_label(path)}" if path else AUTO_LABEL


@dataclass(frozen=True)
class StageBudget:
    """Every number a stage decision reads, taken at one moment.

    `limit_bytes` is the budget scope: the device, or the user's VRAM wall when one is configured.
    `resident_bytes` is how much of *this* pipeline's weight is on the card right now; it is part of
    `allocated_bytes` and the decision may count it as available, because moving it is the decision.
    `allocator_ceiling_bytes` is the wall the allocator cap may never exceed (None: the device).
    """

    driver_free_bytes: int
    physical_total_bytes: int
    limit_bytes: int
    allocated_bytes: int
    reserved_bytes: int
    active_bytes: int
    resident_bytes: int = 0
    platform_reserve_bytes: int = 0
    allocator_ceiling_bytes: int | None = None

    @property
    def reclaimable_bytes(self):
        """Blocks PyTorch has cached but nothing is using: free to this process at a boundary."""
        return max(0, int(self.reserved_bytes) - int(self.active_bytes))

    @property
    def ambient_free_bytes(self):
        """What the pre-P8 controller called free: the pipeline's own weights count as spent."""
        return max(0, min(
            int(self.limit_bytes),
            int(self.driver_free_bytes) + self.reclaimable_bytes,
            int(self.limit_bytes) - int(self.active_bytes),
        ))

    @property
    def planning_free_bytes(self):
        """What a placement decision may spend: ambient free plus this pipeline's own resident weight.

        This is the reading that makes a decision independent of where the weights happen to be.
        Without it a resident model on a 12 GB card looks as if it needed its own 7 GB a second
        time and is demoted, and a streamed one looks rich and is promoted straight back.
        """
        resident = max(0, int(self.resident_bytes))
        return max(0, min(
            int(self.limit_bytes),
            int(self.driver_free_bytes) + self.reclaimable_bytes + resident,
            int(self.limit_bytes) - int(self.active_bytes) + resident,
        ))

    @property
    def allocator_room_bytes(self):
        """The cap `cap_allocator` would install now: what we hold plus what is free, less its margin.

        Held plus free barely moves when this process moves its own weights — the one grows as the
        other shrinks — so it is also the best prediction of the cap after a move.
        """
        room = int(self.reserved_bytes) + int(self.driver_free_bytes) - ALLOCATOR_MARGIN_BYTES
        if self.allocator_ceiling_bytes:
            room = min(room, int(self.allocator_ceiling_bytes))
        return max(0, room)

    @property
    def other_bytes(self):
        """What the card holds that this process's allocator does not: other programs, contexts."""
        return max(0, int(self.physical_total_bytes) - int(self.driver_free_bytes) - int(self.reserved_bytes))

    def as_dict(self):
        return {
            "driver_free_bytes": int(self.driver_free_bytes),
            "physical_total_bytes": int(self.physical_total_bytes),
            "limit_bytes": int(self.limit_bytes),
            "allocated_bytes": int(self.allocated_bytes),
            "reserved_bytes": int(self.reserved_bytes),
            "active_bytes": int(self.active_bytes),
            "resident_bytes": int(self.resident_bytes),
            "platform_reserve_bytes": int(self.platform_reserve_bytes),
            "allocator_ceiling_bytes": None if self.allocator_ceiling_bytes is None else int(self.allocator_ceiling_bytes),
            "ambient_free_bytes": self.ambient_free_bytes,
            "planning_free_bytes": self.planning_free_bytes,
            "allocator_room_bytes": self.allocator_room_bytes,
            "other_bytes": self.other_bytes,
        }


def cap_invariant(budget, workspace_bytes, margin_bytes, cap_bytes=None):
    """`cap >= allocated + this stage's workspace peak + safety margin` (QPlan P8 §3.4).

    `cap_bytes` is the cap actually installed when there is one; otherwise the cap this budget would
    produce. The record is what the diagnostics carry, so both sides of the comparison are in it.
    """
    cap = budget.allocator_room_bytes if cap_bytes is None else int(cap_bytes)
    required = int(budget.allocated_bytes) + max(0, int(workspace_bytes)) + max(0, int(margin_bytes))
    without_margin = int(budget.allocated_bytes) + max(0, int(workspace_bytes))
    return {
        "cap_bytes": int(cap),
        "required_cap_bytes": int(required),
        "holds": cap >= required,
        # At the floor there is nowhere lower to go. A stage whose own peak fits under the cap still
        # runs there — the margin is a buffer against estimate error, not part of the requirement —
        # and only one that cannot fit even without it is refused.
        "holds_without_margin": cap >= without_margin,
        "headroom_bytes": int(cap) - int(required),
    }


@dataclass
class TierChoice:
    tier: str
    reason: str
    fits: dict = field(default_factory=dict)


def choose_component_tier(*, ladder, current, requirement, resident_after, workspace, budget, margin_bytes,
                          promotion_margin=PROMOTION_MARGIN, allow_promotion=True):
    """The rung a component-engine stage runs on, under AUTO.

    `ladder` is most-resident first and ends at the floor. `requirement(tier)` is the bytes that
    rung needs free for this stage (weights it will hold, what the stage adds, the platform reserve);
    `resident_after(tier)` is the weight left on the card once the rung is installed and
    `workspace(tier)` what the stage adds on top of it. A rung fits when its requirement is inside
    the planning budget *and* the cap that would remain after the move still holds the stage.

    The current rung is kept while it fits — never demoted without cause — and a higher one is taken
    only with `promotion_margin` to spare, so a number drifting near a boundary cannot make the
    placement oscillate between stages. `allow_promotion=False` is for a stage too small to be worth
    a move of its own — a prompt encode fits almost anywhere, and promoting for it only buys a second
    move when the sampling stage right after it is placed.
    """
    ladder = tuple(ladder)
    free = budget.planning_free_bytes
    room = budget.allocator_room_bytes
    # What stays allocated whatever this decision does: everything that is not this pipeline's
    # resident weight (other pipelines' tensors, conditioning, the latents in flight).
    fixed = max(0, int(budget.allocated_bytes) - int(budget.resident_bytes))
    fits = {}

    def fits_tier(tier, factor=1.0):
        need = int(requirement(tier))
        under_budget = need * factor <= free
        under_cap = room >= fixed + int(resident_after(tier)) + int(workspace(tier)) + int(margin_bytes)
        fits[(tier, factor)] = under_budget and under_cap
        return under_budget and under_cap

    if current in ladder and fits_tier(current):
        for tier in ladder[:ladder.index(current)] if allow_promotion else ():
            if fits_tier(tier, promotion_margin):
                return TierChoice(tier, "stage_plan_promote", _fit_record(fits))
        return TierChoice(current, "stage_plan_unchanged", _fit_record(fits))
    candidates = ladder[ladder.index(current) + 1:] if current in ladder else ladder
    for tier in candidates:
        if fits_tier(tier):
            reason = "stage_plan_demote" if current in ladder else "stage_plan_select"
            return TierChoice(tier, reason, _fit_record(fits))
    # Nothing fits with its margin; the floor is where the stage goes, and the caller verifies it
    # against the cap it actually gets before anything runs.
    return TierChoice(ladder[-1], "stage_plan_floor", _fit_record(fits))


def _fit_record(fits):
    return {f"{tier}{'' if factor == 1.0 else '@' + format(factor, 'g')}": value for (tier, factor), value in fits.items()}


def legacy_component_tier(*, ladder, requirement, ambient_free):
    """The rung the pre-P8 controller would have chosen: first fit, no margin, own weights spent.

    AUTO no longer places anything by this rule. It is kept because one thing downstream still
    follows it: when the sampler is parked for a staged decode and restored for the next stage, the
    autoencoder's tiling is reconfigured from the mode that was published, and tiling decides
    pixels. Carrying this rung beside the real one is how a fixed seed keeps producing the same
    picture under a scheduler that places the model differently (QPlan P8 §2, pixel identity).
    """
    for tier in ladder:
        if int(requirement(tier)) <= int(ambient_free):
            return tier
    return ladder[-1]


def choose_native_placement(*, current, resident_required, block_required, budget, margin_bytes,
                            resident_workspace, block_workspace, transformer_bytes, block_bytes,
                            promotion_margin=PROMOTION_MARGIN, locked=False):
    """Resident or block-streamed transformer for one native-engine stage, under AUTO.

    The native runtimes are staged — encoders, transformer and autoencoder take the card in turn —
    so what a sampling stage has to hold is the transformer (whole, or one block group and its
    prefetch) beside that stage's activations. Same hysteresis and same cap rule as the component
    ladder; `locked` keeps the current placement when moving it would change the arithmetic (a
    compiled transformer cannot be handed to group-offload hooks and come out the same).
    """
    free = budget.planning_free_bytes
    room = budget.allocator_room_bytes
    fixed = max(0, int(budget.allocated_bytes) - int(budget.resident_bytes))

    def fits(required, weight_on_card, workspace, factor=1.0):
        return (int(required) * factor <= free
                and room >= fixed + int(weight_on_card) + int(workspace) + int(margin_bytes))

    resident_fits = fits(resident_required, transformer_bytes, resident_workspace)
    resident_fits_with_margin = fits(resident_required, transformer_bytes, resident_workspace, promotion_margin)
    block_fits = fits(block_required, block_bytes, block_workspace)
    record = {
        "resident_fits": resident_fits,
        "resident_fits_with_margin": resident_fits_with_margin,
        "block_fits": block_fits,
    }
    if locked:
        return TierChoice(current, "native_placement_locked", record)
    if current == NATIVE_RESIDENT:
        if resident_fits:
            return TierChoice(NATIVE_RESIDENT, "stage_plan_unchanged", record)
        return TierChoice(NATIVE_BLOCK, "stage_plan_demote" if block_fits else "stage_plan_floor", record)
    if resident_fits_with_margin:
        return TierChoice(NATIVE_RESIDENT, "stage_plan_promote", record)
    return TierChoice(NATIVE_BLOCK, "stage_plan_unchanged" if block_fits else "stage_plan_floor", record)


class StageMemoryError(RuntimeError):
    """A stage that cannot run on any rung AUTO may use, said in the terms a person can act on.

    QPlan P8 §3.6: canvas, stage, minimum requirement and what is available — and nothing done
    behind the user's back to make it fit.
    """

    def __init__(self, stage, canvas, required_bytes, available_bytes, detail=None):
        self.stage = stage
        self.canvas = tuple(canvas) if canvas else None
        self.required_bytes = int(required_bytes)
        self.available_bytes = int(max(0, available_bytes))
        self.detail = detail
        canvas_text = f"画布 {self.canvas[0]}×{self.canvas[1]}，" if self.canvas else ""
        message = (
            f"显存不足（{stage_display_name(stage)}）：{canvas_text}该阶段至少需要约 "
            f"{self.required_bytes / GIB:.2f} GB，当前可用约 {self.available_bytes / GIB:.2f} GB。"
            "AUTO 已降到最低驻留路径仍无法容纳；未缩小画布、未减少步数、未回退 CPU。"
            "请关闭其他占用显存的程序、减小画布或批量后重试"
        )
        if detail:
            message += f"（{detail}）"
        super().__init__(message)

    def diagnostics(self):
        return {
            "stage": self.stage,
            "canvas": list(self.canvas) if self.canvas else None,
            "required_bytes": self.required_bytes,
            "available_bytes": self.available_bytes,
            "detail": self.detail,
        }


STAGE_DISPLAY_NAMES = {
    "model_mount": "模型装载",
    "prompt_encode": "提示词编码",
    "base_sampling": "基础采样",
    "vae_decode": "VAE 解码",
    "hires_upscale": "重采样放大",
    "hires_latent_encode": "重采样潜空间编码",
    "hires_sampling": "重采样重绘",
    "adetailer_region": "ADetailer 区域重绘",
    "final_decode": "最终解码",
}


def stage_display_name(stage):
    stage = str(stage or "")
    for key, label in STAGE_DISPLAY_NAMES.items():
        if stage == key or stage.startswith(f"{key}_"):
            suffix = stage[len(key):].strip("_")
            return f"{label} {suffix}" if suffix else label
    return stage.replace("_", " ")


def decision_record(*, stage, engine, family, path, tier=None, canvas=None, reason, budget=None,
                    required_bytes=None, workspace_bytes=None, invariant=None, changed=False,
                    legacy_tier=None, detail=None):
    """One entry of a job's stage schedule, in the vocabulary shared by every engine.

    The same keys for a component rung and a native placement, so a job, the health report and the
    PNG note read one list whichever engine produced it.
    """
    record = {
        "stage": stage,
        "engine": engine,
        "family": family,
        "requested_mode": AUTO_MODE,
        "path": path,
        "path_label": path_label(path) if path else None,
        "tier": tier,
        "canvas": list(canvas) if canvas else None,
        "reason": reason,
        "changed": bool(changed),
    }
    if budget is not None:
        record.update(
            free_bytes=budget.planning_free_bytes,
            ambient_free_bytes=budget.ambient_free_bytes,
            allocated_bytes=int(budget.allocated_bytes),
            limit_bytes=int(budget.limit_bytes),
        )
    if required_bytes is not None:
        record["required_bytes"] = int(required_bytes)
    if workspace_bytes is not None:
        record["workspace_bytes"] = int(workspace_bytes)
    if invariant is not None:
        record.update(
            cap_bytes=invariant["cap_bytes"],
            required_cap_bytes=invariant["required_cap_bytes"],
            cap_holds=invariant["holds"],
        )
    if legacy_tier is not None:
        record["legacy_tier"] = legacy_tier
    if detail:
        record["detail"] = detail
    return record


def png_schedule_note(records, limit=48):
    """The compact stage path a PNG carries: which stage ran on which path, nothing more.

    Byte counts and free-memory readings describe one machine at one moment, which is job
    diagnostics; the picture only needs to say how it was produced.
    """
    stages = []
    for record in list(records or [])[-limit:]:
        if not isinstance(record, dict) or not record.get("path"):
            continue
        stages.append({"stage": record.get("stage"), "path": record.get("path"), "tier": record.get("tier")})
    return {"requested_mode": AUTO_MODE, "label": AUTO_LABEL, "stages": stages}
