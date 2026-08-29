import math

import torch
from scipy.stats import beta as beta_distribution


ANIMA_SAMPLERS = (
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

ANIMA_SCHEDULERS = (
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

# ComfyUI's image-to-image schedule construction can expand a requested
# trajectory by 1 / denoise.  Keep that CPU-only construction bounded: 4096
# covers the schema maximum of 100 requested steps down to denoise 0.025,
# while rejecting accidental near-zero values before they allocate/compute an
# unbounded schedule.
ANIMA_MAX_REFINEMENT_SCHEDULE_STEPS = 4096

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


def _flow_shift(values: torch.Tensor) -> torch.Tensor:
    return 3.0 * values / (1.0 + 2.0 * values)


def _flow_grid(timesteps=1000) -> torch.Tensor:
    return _flow_shift(torch.arange(1, timesteps + 1, dtype=torch.float32) / timesteps)


def _append_zero(values: torch.Tensor) -> torch.Tensor:
    return torch.cat((values.to(device="cpu", dtype=torch.float32), torch.zeros(1, dtype=torch.float32)))


def _linear_quadratic_sigmas(steps: int, threshold_noise=0.025) -> torch.Tensor:
    if steps == 1:
        return torch.tensor([1.0, 0.0], dtype=torch.float32)
    linear_steps = max(1, steps // 2)
    quadratic_steps = steps - linear_steps
    linear = [index * threshold_noise / linear_steps for index in range(linear_steps)]
    if quadratic_steps:
        difference = linear_steps - threshold_noise * steps
        quadratic_coefficient = difference / (linear_steps * quadratic_steps**2)
        linear_coefficient = threshold_noise / linear_steps - 2 * difference / quadratic_steps**2
        constant = quadratic_coefficient * linear_steps**2
        quadratic = [
            quadratic_coefficient * index**2 + linear_coefficient * index + constant
            for index in range(linear_steps, steps)
        ]
    else:
        quadratic = []
    return torch.tensor([1.0 - value for value in linear + quadratic + [1.0]], dtype=torch.float32)


def anima_sigma_schedule(steps: int, scheduler: str) -> torch.Tensor:
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if scheduler not in ANIMA_SCHEDULERS:
        raise ValueError(f"Unsupported Anima scheduler: {scheduler}")

    sigma_min = float(_flow_grid()[0])
    if scheduler == "simple":
        values = _flow_shift(torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32))
    elif scheduler in {"normal", "sgm_uniform"}:
        # ModelSamplingDiscreteFlow stores the 1000 discrete timesteps as
        # (i + 1) / 1000 before applying Flow shift-3.  Keep interpolation in
        # that timestep domain; ``sigma_min`` is already shifted and must not
        # be shifted a second time.
        count = steps + 1 if scheduler == "sgm_uniform" else steps
        timesteps = torch.linspace(1000.0, 1.0, count, dtype=torch.float32) / 1000.0
        if scheduler == "sgm_uniform":
            timesteps = timesteps[:-1]
        values = _append_zero(_flow_shift(timesteps))
    elif scheduler == "karras":
        ramp = torch.linspace(0.0, 1.0, steps, dtype=torch.float32)
        values = _append_zero((1.0 + ramp * (sigma_min ** (1.0 / 7.0) - 1.0)) ** 7)
    elif scheduler == "exponential":
        values = _append_zero(torch.exp(torch.linspace(0.0, math.log(sigma_min), steps, dtype=torch.float32)))
    elif scheduler == "ddim_uniform":
        grid = _flow_grid()
        stride = max(len(grid) // steps, 1)
        indices = torch.arange(1, 1 + stride * steps, stride).clamp(max=len(grid) - 1)
        values = _append_zero(grid[indices].flip(0))
    elif scheduler == "beta":
        quantiles = 1.0 - torch.arange(steps, dtype=torch.float64).numpy() / steps
        indices = beta_distribution.ppf(quantiles, 0.6, 0.6)
        indices = torch.from_numpy(indices).nan_to_num(nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        indices = torch.round(indices * 999).to(torch.long)
        values = _append_zero(_flow_grid()[indices])
    elif scheduler == "linear_quadratic":
        values = _linear_quadratic_sigmas(steps)
    else:
        if steps == 1:
            values = torch.tensor([1.0, 0.0], dtype=torch.float32)
        else:
            ramp = torch.arange(steps, dtype=torch.float32) / (steps - 1)
            values = _append_zero(torch.tan(ramp * math.atan(sigma_min) + (1.0 - ramp) * math.atan(1.0)))

    values = values.to(device="cpu", dtype=torch.float32)
    if len(values) != steps + 1 or not torch.isfinite(values).all() or values[-1].item() != 0.0:
        raise RuntimeError(f"Anima scheduler {scheduler} produced an invalid sigma sequence")
    if not torch.all(values[:-1] >= values[1:]):
        raise RuntimeError(f"Anima scheduler {scheduler} produced a non-monotonic sigma sequence")
    return values


def _comfy_simple_refinement_schedule(steps: int) -> torch.Tensor:
    """Comfy simple scheduler over its 1000-point Flow shift-3 index grid.

    This intentionally differs from the native base ``simple`` schedule.
    Comfy's img2img path first makes a longer simple schedule, then retains a
    suffix; matching its discrete index selection matters at low denoise.
    """
    # Express Comfy's simple_scheduler directly. ModelSamplingDiscreteFlow
    # first builds exactly 1000 entries for (i + 1) / 1000, then uses forward
    # x=0..steps-1 with ``sigmas[-(1 + int(x * 1000 / steps))]`` before
    # appending its literal terminal zero. Keeping the table and negative
    # index avoids changing results at integer-divisibility boundaries.
    model_sigmas = _flow_shift(torch.arange(1, 1001, dtype=torch.float32) / 1000.0)
    selected = [model_sigmas[-(1 + int(x * 1000 / steps))] for x in range(steps)]
    return _append_zero(torch.stack(selected))


def anima_refinement_sigma_schedule(steps: int, denoise: float, scheduler: str) -> tuple[torch.Tensor, dict]:
    """Return the requested-length native refinement trajectory and diagnostics.

    For partial denoise this follows Comfy KSampler semantics exactly at the
    trajectory level: ``new_steps=int(steps / denoise)`` followed by retaining
    the final ``steps + 1`` sigmas.  A full-denoise request preserves the
    existing native schedule byte-for-byte.
    """
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if not isinstance(denoise, (int, float)) or isinstance(denoise, bool) or not math.isfinite(float(denoise)):
        raise ValueError("denoise must be finite")
    if float(denoise) <= 0.0 or float(denoise) > 1.0:
        raise ValueError("denoise must be greater than 0 and at most 1")
    if scheduler not in ANIMA_SCHEDULERS:
        raise ValueError(f"Unsupported Anima scheduler: {scheduler}")

    if float(denoise) == 1.0:
        sigmas = anima_sigma_schedule(steps, scheduler)
        mode = "full"
        schedule_steps = steps
    else:
        schedule_steps = int(steps / float(denoise))
        if schedule_steps > ANIMA_MAX_REFINEMENT_SCHEDULE_STEPS:
            raise ValueError(
                f"Anima refinement denoise requires {schedule_steps} schedule steps, exceeding the limit "
                f"of {ANIMA_MAX_REFINEMENT_SCHEDULE_STEPS}; increase denoise or reduce steps"
            )
        full_schedule = (
            _comfy_simple_refinement_schedule(schedule_steps)
            if scheduler == "simple"
            else anima_sigma_schedule(schedule_steps, scheduler)
        )
        sigmas = full_schedule[-(steps + 1):]
        mode = "comfy_suffix"

    if len(sigmas) != steps + 1 or not torch.isfinite(sigmas).all() or sigmas[-1].item() != 0.0:
        raise RuntimeError("Anima refinement produced an invalid sigma suffix")
    if not torch.all(sigmas[:-1] >= sigmas[1:]):
        raise RuntimeError("Anima refinement produced a non-monotonic sigma suffix")
    return sigmas, {
        "schedule_steps": schedule_steps,
        "schedule_mode": mode,
        "start_sigma": float(sigmas[0].item()),
    }


def prepare_anima_refinement_sigmas(steps: int, denoise: float, scheduler: str) -> torch.Tensor:
    """Build a device-neutral refinement suffix suitable for repeated calls."""
    sigmas, _diagnostics = anima_refinement_sigma_schedule(steps, denoise, scheduler)
    return sigmas.detach().to(device="cpu", dtype=torch.float32).clone()


def validate_prepared_anima_refinement_sigmas(
    sigmas: torch.Tensor, steps: int, denoise: float, scheduler: str
) -> torch.Tensor:
    """Fail closed unless a reusable suffix exactly matches its request contract."""
    if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1:
        raise ValueError("prepared_sigmas must be a one-dimensional torch.Tensor")
    # Validate representation before any conversion: accepting a tensor after
    # downcasting or moving it would silently change a caller's schedule.
    if sigmas.device.type != "cpu":
        raise ValueError("prepared_sigmas must be a CPU tensor")
    if sigmas.dtype != torch.float32:
        raise ValueError("prepared_sigmas must have dtype torch.float32")
    if not sigmas.is_contiguous():
        raise ValueError("prepared_sigmas must be contiguous")
    candidate = sigmas
    if len(candidate) != steps + 1:
        raise ValueError("prepared_sigmas length does not match steps")
    if not torch.isfinite(candidate).all() or candidate[-1].item() != 0.0:
        raise ValueError("prepared_sigmas must be finite and terminate at zero")
    if not torch.all(candidate[:-1] >= candidate[1:]):
        raise ValueError("prepared_sigmas must be monotonic non-increasing")
    expected = prepare_anima_refinement_sigmas(steps, denoise, scheduler)
    if not torch.equal(candidate, expected):
        raise ValueError("prepared_sigmas do not match the requested steps, denoise, and scheduler")
    return candidate.clone()


def resolve_anima_sampler(sampler: str) -> tuple[str, str | None]:
    if sampler not in ANIMA_SAMPLERS:
        raise ValueError(f"Unsupported Anima sampler: {sampler}")
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
        return "flow_lcm", "Anima is not LCM-distilled; lcm uses the rectified-flow LCM update and may reduce quality"
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
        details += "; standard sequential CFG is used because native CFG++ is unavailable"
    return implementation, details


def anima_sampling_diagnostics(sampler: str, scheduler: str) -> dict:
    implementation, warning = resolve_anima_sampler(sampler)
    if scheduler not in ANIMA_SCHEDULERS:
        raise ValueError(f"Unsupported Anima scheduler: {scheduler}")
    return {
        "requested_sampler": sampler,
        "sampler_implementation": implementation,
        "requested_scheduler": scheduler,
        "scheduler_implementation": f"rf_{scheduler}_shift_3",
        "warning": warning,
    }
