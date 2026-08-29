"""ComfyUI-parity sigma schedules for the FLUX.2 engine.

FLUX.2 reuses FLUX.1's rectified-flow model sampling — ``ModelSamplingFlux``'s

    sigma(t) = exp(mu) / (exp(mu) + (1 / t - 1))

table — so every scheduler in :mod:`flux_sampling` applies unchanged.  What differs is where
``mu`` comes from.  FLUX.1 reads it off the canvas alone through ``ModelSamplingFlux``'s
base/max interpolation; FLUX.2 ships its own node, ``comfy_extras/nodes_flux.py::Flux2Scheduler``,
which fits ``mu`` empirically from *both* the image sequence length and the step count:

    m_200 = a2 * seq + b2                     (the fit at 200 steps)
    m_10  = a1 * seq + b1                     (the fit at 10 steps)
    mu    = ((m_200 - m_10) / 190) * steps + (m_200 - 200 * that slope)

with a straight ``m_200`` past a sequence length of 4300.  That node then evaluates

    generalized_time_snr_shift(t, mu, 1.0) = exp(mu) / (exp(mu) + (1 / t - 1) ** 1)

over ``linspace(1, 0, steps + 1)`` — which is the same expression the sigma table holds, sampled
at the same uniform timesteps the ``simple`` scheduler selects.  Reproducing FLUX.2's reference
schedule therefore needs no new scheduler: it needs ``simple`` evaluated at the empirical ``mu``.

``Flux2``'s ``sampling_settings`` in ``comfy/supported_models.py`` also carry a static
``shift = 2.02``, which is what a bare ``KSampler`` graph uses.  That value is the empirical fit's
own answer near 1024x1024 at ~20 steps, so following the fit tracks the blueprint at the default
canvas and keeps tracking it everywhere else.
"""

import math

try:
    from .flux_sampling import (
        FLUX_MAX_REFINEMENT_SCHEDULE_STEPS,
        FLUX_SAMPLERS,
        FLUX_SCHEDULERS,
        flux_refinement_sigma_schedule,
        flux_sigma_schedule,
        resolve_flux_sampler,
    )
except ImportError:
    from flux_sampling import (
        FLUX_MAX_REFINEMENT_SCHEDULE_STEPS,
        FLUX_SAMPLERS,
        FLUX_SCHEDULERS,
        flux_refinement_sigma_schedule,
        flux_sigma_schedule,
        resolve_flux_sampler,
    )


# Both Flux engines are driven by the same ComfyUI KSampler, so they expose the same vocabulary.
# `test_flux2_sampling` pins these to FLUX.1's so the three native engines cannot drift apart.
FLUX2_SAMPLERS = FLUX_SAMPLERS
FLUX2_SCHEDULERS = FLUX_SCHEDULERS

FLUX2_MAX_REFINEMENT_SCHEDULE_STEPS = FLUX_MAX_REFINEMENT_SCHEDULE_STEPS

# `Flux2Scheduler.execute` divides the pixel count by 16 * 16: eight for the VAE stride and two
# for the autoencoder's own 2x2 latent packing, which together are the token count the schedule is
# stretched by. The transformer itself runs at patch size 1 on those already-packed tokens.
FLUX2_SEQUENCE_DIVISOR = 16 * 16

# `compute_empirical_mu`'s fitted coefficients, verbatim from ComfyUI.
FLUX2_MU_SLOPE_10 = 8.73809524e-05
FLUX2_MU_INTERCEPT_10 = 1.89833333
FLUX2_MU_SLOPE_200 = 0.00016927
FLUX2_MU_INTERCEPT_200 = 0.45666666
FLUX2_MU_SEQUENCE_PLATEAU = 4300

# The static shift `comfy/supported_models.py::Flux2` registers for a bare KSampler graph. Kept
# for reference and reported in diagnostics beside the fitted value.
FLUX2_STATIC_SHIFT = 2.02


def flux2_sequence_length(width: int, height: int) -> int:
    """The packed token count ``Flux2Scheduler`` measures a canvas by."""
    if not isinstance(width, int) or isinstance(width, bool) or not isinstance(height, int) or isinstance(height, bool):
        raise ValueError("width and height must be integers")
    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")
    return round(width * height / FLUX2_SEQUENCE_DIVISOR)


def flux2_empirical_mu(sequence_length: int, steps: int) -> float:
    """``comfy_extras.nodes_flux.compute_empirical_mu``.

    Past a sequence length of 4300 the step-count term drops out entirely: BFL's fit found the
    long-sequence behaviour step-independent, so a 2048-edge canvas uses one mu whatever the step
    count while a 1024-edge canvas still moves with it.
    """
    if not isinstance(sequence_length, int) or isinstance(sequence_length, bool) or sequence_length < 1:
        raise ValueError("sequence_length must be a positive integer")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer")

    m_200 = FLUX2_MU_SLOPE_200 * sequence_length + FLUX2_MU_INTERCEPT_200
    if sequence_length > FLUX2_MU_SEQUENCE_PLATEAU:
        return float(m_200)

    m_10 = FLUX2_MU_SLOPE_10 * sequence_length + FLUX2_MU_INTERCEPT_10
    slope = (m_200 - m_10) / 190.0
    intercept = m_200 - 200.0 * slope
    return float(slope * steps + intercept)


def flux2_resolution_shift(width: int, height: int, steps: int) -> float:
    """The ``mu`` a FLUX.2 run of ``steps`` steps on this canvas is scheduled with.

    ``steps`` is the *requested* step count, not the expanded image-to-image construction: the fit
    describes a trajectory of that many updates, and the Comfy suffix rule guarantees a refinement
    performs every requested step whatever the denoise.  Keeping the requested count here is also
    what keeps ``mu`` positive — the fit's step slope is negative, and feeding it a schedule
    expanded by ``1 / denoise`` would eventually drive it through zero.
    """
    mu = flux2_empirical_mu(flux2_sequence_length(width, height), steps)
    if not math.isfinite(mu) or mu <= 0.0:
        raise RuntimeError(
            f"FLUX.2 empirical schedule produced an unusable shift {mu!r} for {width}x{height} at {steps} steps"
        )
    return mu


def flux2_sigma_schedule(steps: int, scheduler: str, shift: float):
    if scheduler not in FLUX2_SCHEDULERS:
        raise ValueError(f"Unsupported Flux2 scheduler: {scheduler}")
    return flux_sigma_schedule(steps, scheduler, shift)


def flux2_refinement_sigma_schedule(steps: int, denoise: float, scheduler: str, shift: float):
    if scheduler not in FLUX2_SCHEDULERS:
        raise ValueError(f"Unsupported Flux2 scheduler: {scheduler}")
    return flux_refinement_sigma_schedule(steps, denoise, scheduler, shift)


def resolve_flux2_sampler(sampler: str) -> tuple[str, str | None]:
    """The rectified-flow update behind a ComfyUI sampler name, with FLUX.2 wording."""
    if sampler not in FLUX2_SAMPLERS:
        raise ValueError(f"Unsupported Flux2 sampler: {sampler}")
    implementation, warning = resolve_flux_sampler(sampler)
    if warning:
        warning = warning.replace("FLUX.1", "FLUX.2")
    return implementation, warning


def flux2_sampling_diagnostics(sampler: str, scheduler: str, shift: float) -> dict:
    implementation, warning = resolve_flux2_sampler(sampler)
    if scheduler not in FLUX2_SCHEDULERS:
        raise ValueError(f"Unsupported Flux2 scheduler: {scheduler}")
    return {
        "requested_sampler": sampler,
        "sampler_implementation": implementation,
        "requested_scheduler": scheduler,
        "scheduler_implementation": f"rf_{scheduler}_shift_{float(shift):.4g}",
        "shift": round(float(shift), 4),
        "static_shift": FLUX2_STATIC_SHIFT,
        "warning": warning,
    }
