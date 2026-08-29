"""ComfyUI-parity sigma schedules and sampler vocabulary for the Flux engine.

FLUX.1 is a rectified-flow model whose noise schedule ComfyUI expresses through
``comfy/model_sampling.py::ModelSamplingFlux``: a 10 000-entry table of

    sigma(t) = exp(mu) / (exp(mu) + (1 / t - 1))

for ``t = (i + 1) / 10000``, where ``mu`` is the *shift*.  Every scheduler in
``comfy/samplers.py`` then selects a trajectory out of that table, so matching
the table and the selection is what makes a run reproduce ComfyUI rather than
merely resemble it.

This deliberately mirrors :mod:`anima_sampling` in shape while keeping the two
engines independent: Anima is a Flow shift-3 model over a 1000-entry grid, Flux
is a shift-mu model over a 10 000-entry grid, and folding them together would
make either engine's schedule a change away from silently altering the other's
output.  ``test_flux_sampling`` pins the two sampler/scheduler vocabularies to
each other so the shared ComfyUI KSampler naming cannot drift apart unnoticed.
"""

import math

import torch
from scipy.stats import beta as beta_distribution


# ComfyUI's KSampler vocabulary. Both native engines expose the same names because both are
# driven by the same node; the implementations behind them are per-engine.
FLUX_SAMPLERS = (
    "euler",
    "euler_cfg_pp",
    "euler_ancestral",
    "euler_ancestral_cfg_pp",
    "heun",
    "heunpp2",
    "exp_heun_2_x0",
    "exp_heun_2_x0_sde",
    "dpm_2",
    "dpm_2_ancestral",
    "lms",
    "dpm_fast",
    "dpm_adaptive",
    "dpmpp_2s_ancestral",
    "dpmpp_2s_ancestral_cfg_pp",
    "dpmpp_sde",
    "dpmpp_sde_gpu",
    "dpmpp_2m",
    "dpmpp_2m_cfg_pp",
    "dpmpp_2m_sde",
    "dpmpp_2m_sde_gpu",
    "dpmpp_2m_sde_heun",
    "dpmpp_2m_sde_heun_gpu",
    "dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu",
    "ddpm",
    "lcm",
    "ipndm",
    "ipndm_v",
    "deis",
    "res_multistep",
    "res_multistep_cfg_pp",
    "res_multistep_ancestral",
    "res_multistep_ancestral_cfg_pp",
    "gradient_estimation",
    "gradient_estimation_cfg_pp",
    "er_sde",
    "seeds_2",
    "seeds_3",
    "sa_solver",
    "sa_solver_pece",
    "ddim",
    "uni_pc",
    "uni_pc_bh2",
)

FLUX_SCHEDULERS = (
    "simple",
    "sgm_uniform",
    "karras",
    "exponential",
    "ddim_uniform",
    "beta",
    "normal",
    "linear_quadratic",
    "kl_optimal",
)

# ``ModelSamplingFlux.set_parameters`` builds exactly this many discrete sigmas. Several
# schedulers index the table directly, so its length is part of the contract.
FLUX_SIGMA_TABLE_SIZE = 10000

# ComfyUI's ModelSamplingFlux node interpolates the shift between these anchors. The defaults
# resolve to 1.15 at 1024 x 1024, which is exactly the static shift a KSampler graph without the
# node uses, so a default-resolution run matches the stock blueprint while other canvases follow
# the resolution-aware value BFL's reference implementation and Diffusers both use.
FLUX_BASE_SHIFT = 0.5
FLUX_MAX_SHIFT = 1.15
FLUX_BASE_SEQUENCE_ANCHOR = 256
FLUX_MAX_SEQUENCE_ANCHOR = 4096

# ComfyUI's image-to-image schedule construction expands a requested trajectory by 1 / denoise.
# Keep that CPU-only construction bounded: 4096 covers the schema maximum of 100 requested steps
# down to denoise 0.025 while rejecting near-zero values before they build an unbounded schedule.
FLUX_MAX_REFINEMENT_SCHEDULE_STEPS = 4096

_ANCESTRAL_SAMPLERS = {
    "euler_ancestral",
    "euler_ancestral_cfg_pp",
    "exp_heun_2_x0_sde",
    "dpm_2_ancestral",
    "dpmpp_2s_ancestral",
    "dpmpp_2s_ancestral_cfg_pp",
    "dpmpp_sde",
    "dpmpp_sde_gpu",
    "dpmpp_2m_sde",
    "dpmpp_2m_sde_gpu",
    "dpmpp_2m_sde_heun",
    "dpmpp_2m_sde_heun_gpu",
    "dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu",
    "ddpm",
    "res_multistep_ancestral",
    "res_multistep_ancestral_cfg_pp",
    "er_sde",
    "seeds_2",
    "seeds_3",
    "sa_solver",
    "sa_solver_pece",
}

_HEUN_SAMPLERS = {"heun", "heunpp2", "exp_heun_2_x0"}
_MIDPOINT_SAMPLERS = {"dpm_2"}
_MULTISTEP_SAMPLERS = {
    "lms",
    "dpm_fast",
    "dpm_adaptive",
    "dpmpp_2m",
    "dpmpp_2m_cfg_pp",
    "ipndm",
    "ipndm_v",
    "deis",
    "res_multistep",
    "res_multistep_cfg_pp",
    "gradient_estimation",
    "gradient_estimation_cfg_pp",
    "uni_pc",
    "uni_pc_bh2",
}


def flux_time_shift(shift: float, sigma: float, t):
    """``comfy.model_sampling.flux_time_shift``: exp(mu) / (exp(mu) + (1/t - 1) ** sigma)."""
    return math.exp(shift) / (math.exp(shift) + (1 / t - 1) ** sigma)


def flux_resolution_shift(width: int, height: int) -> float:
    """ComfyUI's ModelSamplingFlux interpolation over the packed image sequence length.

    The node divides the pixel count by ``8 * 8 * 2 * 2``: eight for the VAE stride and two for
    the transformer's 2x2 latent patch, which is the token count the schedule is stretched by.
    """
    if not isinstance(width, int) or isinstance(width, bool) or not isinstance(height, int) or isinstance(height, bool):
        raise ValueError("width and height must be integers")
    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")
    slope = (FLUX_MAX_SHIFT - FLUX_BASE_SHIFT) / (FLUX_MAX_SEQUENCE_ANCHOR - FLUX_BASE_SEQUENCE_ANCHOR)
    intercept = FLUX_BASE_SHIFT - slope * FLUX_BASE_SEQUENCE_ANCHOR
    return (width * height / (8 * 8 * 2 * 2)) * slope + intercept


def _validate_shift(shift) -> float:
    if isinstance(shift, bool) or not isinstance(shift, (int, float)) or not math.isfinite(float(shift)):
        raise ValueError("shift must be finite")
    if not 0.0 < float(shift) <= 100.0:
        raise ValueError("shift must be greater than 0 and at most 100")
    return float(shift)


def flux_model_sigmas(shift: float) -> torch.Tensor:
    """The discrete sigma table ``ModelSamplingFlux`` registers, ascending, ending at 1.0.

    Built in float32 through the same expression ComfyUI evaluates, because several schedulers
    index this table directly and a higher-precision table would select different entries at the
    rounding boundaries.
    """
    shift = _validate_shift(shift)
    timesteps = torch.arange(1, FLUX_SIGMA_TABLE_SIZE + 1, 1) / FLUX_SIGMA_TABLE_SIZE
    return flux_time_shift(shift, 1.0, timesteps).to(dtype=torch.float32)


def _append_zero(values: torch.Tensor) -> torch.Tensor:
    return torch.cat((values.to(device="cpu", dtype=torch.float32), torch.zeros(1, dtype=torch.float32)))


def _simple(sigmas: torch.Tensor, steps: int) -> torch.Tensor:
    stride = len(sigmas) / steps
    selected = [sigmas[-(1 + int(index * stride))] for index in range(steps)]
    return _append_zero(torch.stack(selected))


def _normal(sigmas: torch.Tensor, steps: int, sgm: bool, shift: float) -> torch.Tensor:
    # ``ModelSamplingFlux.timestep`` is the identity, so ComfyUI's normal scheduler walks the
    # sigma range and applies the shift a second time through ``sigma()``. Reproducing that is
    # the point: a "cleaner" single application would not be the schedule ComfyUI ran.
    start = float(sigmas[-1])
    end = float(sigmas[0])
    append_zero = True
    if sgm:
        timesteps = torch.linspace(start, end, steps + 1, dtype=torch.float32)[:-1]
    else:
        if math.isclose(flux_time_shift(shift, 1.0, end), 0.0, abs_tol=0.00001):
            steps += 1
            append_zero = False
        timesteps = torch.linspace(start, end, steps, dtype=torch.float32)
    values = torch.tensor(
        [flux_time_shift(shift, 1.0, float(value)) for value in timesteps], dtype=torch.float32
    )
    return _append_zero(values) if append_zero else values


def _karras(sigmas: torch.Tensor, steps: int, rho: float = 7.0) -> torch.Tensor:
    sigma_min = float(sigmas[0])
    sigma_max = float(sigmas[-1])
    ramp = torch.linspace(0.0, 1.0, steps, dtype=torch.float32)
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    return _append_zero((max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho)


def _exponential(sigmas: torch.Tensor, steps: int) -> torch.Tensor:
    sigma_min = float(sigmas[0])
    sigma_max = float(sigmas[-1])
    return _append_zero(torch.linspace(math.log(sigma_max), math.log(sigma_min), steps, dtype=torch.float32).exp())


def _ddim_uniform(sigmas: torch.Tensor, steps: int) -> torch.Tensor:
    if math.isclose(float(sigmas[1]), 0.0, abs_tol=0.00001):
        steps += 1
        values = []
    else:
        values = [0.0]
    stride = max(len(sigmas) // steps, 1)
    index = 1
    while index < len(sigmas):
        values.append(float(sigmas[index]))
        index += stride
    values.reverse()
    return torch.tensor(values, dtype=torch.float32)


def _beta(sigmas: torch.Tensor, steps: int, alpha: float = 0.6, beta: float = 0.6) -> torch.Tensor:
    total_timesteps = len(sigmas) - 1
    quantiles = 1.0 - torch.arange(steps, dtype=torch.float64).div(steps).numpy()
    indices = beta_distribution.ppf(quantiles, alpha, beta)
    indices = torch.from_numpy(indices).nan_to_num(nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    indices = torch.round(indices * total_timesteps).to(torch.long)
    values = []
    last_index = -1
    for index in indices.tolist():
        if index != last_index:
            values.append(float(sigmas[index]))
        last_index = index
    values.append(0.0)
    return torch.tensor(values, dtype=torch.float32)


def _linear_quadratic(sigmas: torch.Tensor, steps: int, threshold_noise: float = 0.025) -> torch.Tensor:
    sigma_max = float(sigmas[-1])
    if steps == 1:
        schedule = [1.0, 0.0]
        return torch.tensor(schedule, dtype=torch.float32) * sigma_max
    linear_steps = steps // 2
    linear = [index * threshold_noise / linear_steps for index in range(linear_steps)]
    difference = linear_steps - threshold_noise * steps
    quadratic_steps = steps - linear_steps
    quadratic_coefficient = difference / (linear_steps * quadratic_steps**2)
    linear_coefficient = threshold_noise / linear_steps - 2 * difference / quadratic_steps**2
    constant = quadratic_coefficient * linear_steps**2
    quadratic = [
        quadratic_coefficient * index**2 + linear_coefficient * index + constant
        for index in range(linear_steps, steps)
    ]
    schedule = [1.0 - value for value in linear + quadratic + [1.0]]
    return torch.tensor(schedule, dtype=torch.float32) * sigma_max


def _kl_optimal(sigmas: torch.Tensor, steps: int) -> torch.Tensor:
    sigma_min = float(sigmas[0])
    sigma_max = float(sigmas[-1])
    if steps == 1:
        return torch.tensor([sigma_max, 0.0], dtype=torch.float32)
    ramp = torch.arange(steps, dtype=torch.float32).div(steps - 1)
    values = (ramp * math.atan(sigma_min) + (1.0 - ramp) * math.atan(sigma_max)).tan()
    return _append_zero(values)


def flux_sigma_schedule(steps: int, scheduler: str, shift: float) -> torch.Tensor:
    """Return the sigma trajectory ComfyUI would run for this scheduler, terminating at zero."""
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if scheduler not in FLUX_SCHEDULERS:
        raise ValueError(f"Unsupported Flux scheduler: {scheduler}")
    shift = _validate_shift(shift)
    table = flux_model_sigmas(shift)

    if scheduler == "simple":
        values = _simple(table, steps)
    elif scheduler == "normal":
        values = _normal(table, steps, sgm=False, shift=shift)
    elif scheduler == "sgm_uniform":
        values = _normal(table, steps, sgm=True, shift=shift)
    elif scheduler == "karras":
        values = _karras(table, steps)
    elif scheduler == "exponential":
        values = _exponential(table, steps)
    elif scheduler == "ddim_uniform":
        values = _ddim_uniform(table, steps)
    elif scheduler == "beta":
        values = _beta(table, steps)
    elif scheduler == "linear_quadratic":
        values = _linear_quadratic(table, steps)
    else:
        values = _kl_optimal(table, steps)

    values = values.to(device="cpu", dtype=torch.float32)
    # ``beta`` collapses duplicate table indices and ``ddim_uniform`` walks a fixed stride, so
    # neither guarantees the requested length. ComfyUI runs whatever length it produced; so does
    # this, and the caller is told the real count rather than being handed a padded schedule.
    if len(values) < 2 or not torch.isfinite(values).all() or values[-1].item() != 0.0:
        raise RuntimeError(f"Flux scheduler {scheduler} produced an invalid sigma sequence")
    if not torch.all(values[:-1] >= values[1:]):
        raise RuntimeError(f"Flux scheduler {scheduler} produced a non-monotonic sigma sequence")
    return values


def flux_refinement_sigma_schedule(steps: int, denoise: float, scheduler: str, shift: float):
    """Comfy KSampler image-to-image semantics: build ``int(steps / denoise)`` then keep the tail.

    A full-denoise request returns the base schedule unchanged.  Partial denoise builds the longer
    trajectory first and retains its final ``steps + 1`` sigmas, which is why a Flux refinement
    pass performs every requested step whatever the denoise — the denoise picks *where* on the
    trajectory it starts, not how many updates it runs.
    """
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if isinstance(denoise, bool) or not isinstance(denoise, (int, float)) or not math.isfinite(float(denoise)):
        raise ValueError("denoise must be finite")
    if float(denoise) <= 0.0 or float(denoise) > 1.0:
        raise ValueError("denoise must be greater than 0 and at most 1")
    if scheduler not in FLUX_SCHEDULERS:
        raise ValueError(f"Unsupported Flux scheduler: {scheduler}")

    if float(denoise) > 0.9999:
        sigmas = flux_sigma_schedule(steps, scheduler, shift)
        return sigmas, {
            "schedule_steps": steps,
            "schedule_mode": "full",
            "start_sigma": float(sigmas[0].item()),
        }

    schedule_steps = int(steps / float(denoise))
    if schedule_steps > FLUX_MAX_REFINEMENT_SCHEDULE_STEPS:
        raise ValueError(
            f"Flux refinement denoise requires {schedule_steps} schedule steps, exceeding the limit "
            f"of {FLUX_MAX_REFINEMENT_SCHEDULE_STEPS}; increase denoise or reduce steps"
        )
    full_schedule = flux_sigma_schedule(schedule_steps, scheduler, shift)
    # ComfyUI slices whatever the longer schedule produced. `beta` and `ddim_uniform` do not
    # guarantee an exact length, so the tail is taken as-is and the caller is told the real step
    # count through `len(sigmas) - 1` rather than being handed a padded schedule.
    sigmas = full_schedule[-(steps + 1):]
    if len(sigmas) < 2 or not torch.isfinite(sigmas).all() or sigmas[-1].item() != 0.0:
        raise RuntimeError("Flux refinement produced an invalid sigma suffix")
    if not torch.all(sigmas[:-1] >= sigmas[1:]):
        raise RuntimeError("Flux refinement produced a non-monotonic sigma suffix")
    return sigmas.clone(), {
        "schedule_steps": schedule_steps,
        "schedule_mode": "comfy_suffix",
        "start_sigma": float(sigmas[0].item()),
    }


def resolve_flux_sampler(sampler: str) -> tuple[str, str | None]:
    """Map a ComfyUI sampler name onto the rectified-flow update this runtime implements."""
    if sampler not in FLUX_SAMPLERS:
        raise ValueError(f"Unsupported Flux sampler: {sampler}")
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
        return "flow_lcm", "FLUX.1 is guidance-distilled rather than LCM-distilled; lcm uses the rectified-flow LCM update and may reduce quality"
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
        details += "; CFG++ is unavailable because FLUX.1 runs a single guidance-distilled forward pass"
    return implementation, details


def flux_sampling_diagnostics(sampler: str, scheduler: str, shift: float) -> dict:
    implementation, warning = resolve_flux_sampler(sampler)
    if scheduler not in FLUX_SCHEDULERS:
        raise ValueError(f"Unsupported Flux scheduler: {scheduler}")
    return {
        "requested_sampler": sampler,
        "sampler_implementation": implementation,
        "requested_scheduler": scheduler,
        "scheduler_implementation": f"rf_{scheduler}_shift_{_validate_shift(shift):.4g}",
        "shift": round(_validate_shift(shift), 4),
        "warning": warning,
    }
