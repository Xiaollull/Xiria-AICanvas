"""ComfyUI-parity sigma schedules and sampler vocabulary for the Krea 2 engine.

Krea 2 (K2) is registered in ``comfy/supported_models.py`` as

    class Krea2(supported_models_base.BASE):
        sampling_settings = {"multiplier": 1.0, "shift": 1.15}

and ``model_base.Krea2`` builds it with ``ModelType.FLUX``, which selects
``comfy/model_sampling.py::ModelSamplingFlux``.  That is the same 10 000-entry table FLUX.1 uses —

    sigma(t) = exp(mu) / (exp(mu) + (1 / t - 1))    for t = (i + 1) / 10000

— so every scheduler in :mod:`flux_sampling` applies to Krea 2 unchanged and this module reuses
them rather than re-deriving a second copy of the same nine selections.  ``multiplier`` is read by
the ``CONST``/``EPS`` sampling classes and is inert under ``ModelSamplingFlux``; it is recorded
here only so the value is not mistaken for something this runtime dropped.

**The shift does not move with the canvas.**  This is the one place Krea 2 deliberately parts
company with :mod:`flux_sampling`, and the reason is worth stating because the two look alike:

* FLUX.1 registers ``sampling_settings = {}``, so ComfyUI falls back to ``ModelSamplingFlux``'s own
  1.15 default, and BFL's reference implementation *and* Diffusers both compute a
  resolution-dependent ``mu`` for it.  ``flux_resolution_shift`` follows those, and its anchors
  resolve to exactly 1.15 at 1024x1024, so a default-resolution run still matches a stock graph.
* Krea 2 states 1.15 in its own model config.  Nothing in ComfyUI — no ``Krea2Scheduler`` node, no
  entry in ``comfy_extras`` — recomputes it from the canvas, and there is no reference
  implementation claiming otherwise.  Krea 2 shares FLUX.1's token geometry exactly (patch 2 over a
  stride-8 latent, so ``width * height / 256`` tokens), which makes FLUX.1's interpolation *look*
  transplantable; but transplanting it would triple the shift at 2048x2048 on nothing more than the
  resemblance.  A Krea 2 run here schedules with the shift Krea 2 declares, at every canvas.

``krea2_resolution_shift`` exists so a caller that wants FLUX.1's interpolation can ask for it
explicitly, and :func:`krea2_sampling_diagnostics` reports it beside the shift actually used, so
the difference is visible in a generation's metadata instead of being an undocumented default.
"""

import math

try:
    from .flux_sampling import (
        _ANCESTRAL_SAMPLERS,
        _HEUN_SAMPLERS,
        _MIDPOINT_SAMPLERS,
        _MULTISTEP_SAMPLERS,
        FLUX_MAX_REFINEMENT_SCHEDULE_STEPS,
        FLUX_SAMPLERS,
        FLUX_SCHEDULERS,
        flux_refinement_sigma_schedule,
        flux_resolution_shift,
        flux_sigma_schedule,
    )
except ImportError:
    from flux_sampling import (
        _ANCESTRAL_SAMPLERS,
        _HEUN_SAMPLERS,
        _MIDPOINT_SAMPLERS,
        _MULTISTEP_SAMPLERS,
        FLUX_MAX_REFINEMENT_SCHEDULE_STEPS,
        FLUX_SAMPLERS,
        FLUX_SCHEDULERS,
        flux_refinement_sigma_schedule,
        flux_resolution_shift,
        flux_sigma_schedule,
    )


# Every native engine is driven by the same ComfyUI KSampler node, so all of them offer the same
# names. `test_krea2_sampling` pins these to FLUX.1's so the four cannot drift apart.
KREA2_SAMPLERS = FLUX_SAMPLERS
KREA2_SCHEDULERS = FLUX_SCHEDULERS

KREA2_MAX_REFINEMENT_SCHEDULE_STEPS = FLUX_MAX_REFINEMENT_SCHEDULE_STEPS

# `comfy/supported_models.py::Krea2.sampling_settings`, verbatim.
KREA2_SHIFT = 1.15
# Inert under `ModelSamplingFlux`, which reads only `shift`. Recorded so its absence from the
# schedule reads as a fact about ComfyUI rather than as an omission here.
KREA2_MULTIPLIER = 1.0

# Patch size 2 over a stride-8 latent: the same packed token count FLUX.1 measures a canvas by.
KREA2_SEQUENCE_DIVISOR = 8 * 8 * 2 * 2


def krea2_sequence_length(width: int, height: int) -> int:
    """The packed token count the transformer runs over for this canvas."""
    if not isinstance(width, int) or isinstance(width, bool) or not isinstance(height, int) or isinstance(height, bool):
        raise ValueError("width and height must be integers")
    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")
    return round(width * height / KREA2_SEQUENCE_DIVISOR)


def krea2_resolution_shift(width: int, height: int) -> float:
    """FLUX.1's ``ModelSamplingFlux`` interpolation evaluated over Krea 2's token count.

    Not what a run uses — :data:`KREA2_SHIFT` is — and reported only as a comparison, because the
    two agree exactly at 1024x1024 and diverge sharply above it.
    """
    return flux_resolution_shift(width, height)


def _validate_shift(shift) -> float:
    if isinstance(shift, bool) or not isinstance(shift, (int, float)) or not math.isfinite(float(shift)):
        raise ValueError("shift must be finite")
    if not 0.0 < float(shift) <= 100.0:
        raise ValueError("shift must be greater than 0 and at most 100")
    return float(shift)


def krea2_sigma_schedule(steps: int, scheduler: str, shift: float = KREA2_SHIFT):
    if scheduler not in KREA2_SCHEDULERS:
        raise ValueError(f"Unsupported Krea2 scheduler: {scheduler}")
    return flux_sigma_schedule(steps, scheduler, shift)


def krea2_refinement_sigma_schedule(steps: int, denoise: float, scheduler: str, shift: float = KREA2_SHIFT):
    if scheduler not in KREA2_SCHEDULERS:
        raise ValueError(f"Unsupported Krea2 scheduler: {scheduler}")
    return flux_refinement_sigma_schedule(steps, denoise, scheduler, shift)


def resolve_krea2_sampler(sampler: str) -> tuple[str, str | None]:
    """Map a ComfyUI sampler name onto the rectified-flow update this runtime implements.

    Deliberately not :func:`flux_sampling.resolve_flux_sampler` with the wording swapped: two of
    its notes are statements about guidance distillation that are false here.  Krea 2 runs a real
    unconditional branch, so ``lcm`` is a plain distillation mismatch rather than an architectural
    impossibility, and ``cfg_pp`` is unavailable for an implementation reason rather than because
    there is nothing to post-condition against.
    """
    if sampler not in KREA2_SAMPLERS:
        raise ValueError(f"Unsupported Krea2 sampler: {sampler}")
    if sampler == "euler":
        return "euler", None
    if sampler == "euler_ancestral":
        return "euler_ancestral", None
    if sampler == "heun":
        return "heun", None
    if sampler == "dpm_2":
        return "midpoint", None
    if sampler == "lms":
        return "multistep", None
    if sampler == "lcm":
        return "flow_lcm", "Krea 2 is not an LCM-distilled model; lcm uses the rectified-flow LCM update and may reduce quality"
    if sampler == "ddim":
        return "euler", "ComfyUI's ddim sampler name maps to Euler; ddim_uniform remains a separate scheduler"
    if sampler in _ANCESTRAL_SAMPLERS:
        implementation = "euler_ancestral"
    elif sampler in _HEUN_SAMPLERS:
        implementation = "heun"
    elif sampler in _MIDPOINT_SAMPLERS:
        implementation = "midpoint"
    elif sampler in _MULTISTEP_SAMPLERS:
        implementation = "multistep"
    else:
        implementation = "euler"

    details = f"{sampler} uses the native rectified-flow {implementation} compatibility implementation"
    if sampler.endswith("_gpu"):
        details += "; deterministic CPU-seeded noise replaces the GPU-specific Brownian path"
    if "cfg_pp" in sampler:
        details += "; CFG++ is unavailable because guidance is combined after the branches rather than inside the sampler update"
    return implementation, details


def krea2_sampling_diagnostics(sampler: str, scheduler: str, width: int, height: int) -> dict:
    implementation, warning = resolve_krea2_sampler(sampler)
    if scheduler not in KREA2_SCHEDULERS:
        raise ValueError(f"Unsupported Krea2 scheduler: {scheduler}")
    shift = _validate_shift(KREA2_SHIFT)
    return {
        "requested_sampler": sampler,
        "sampler_implementation": implementation,
        "requested_scheduler": scheduler,
        "scheduler_implementation": f"rf_{scheduler}_shift_{shift:.4g}",
        "shift": round(shift, 4),
        "shift_source": "comfy_sampling_settings",
        "multiplier": KREA2_MULTIPLIER,
        "sequence_length": krea2_sequence_length(width, height),
        # What FLUX.1's canvas interpolation would have produced, for comparison only.
        "flux_resolution_shift": round(krea2_resolution_shift(width, height), 4),
        "warning": warning,
    }
