"""ComfyUI-parity sigma schedules, samplers and guidance windows for the MOSS-Image engine.

MOSS-Image is a rectified-flow DiT that ComfyUI drives through ``ModelSamplingSD3(shift=5)``,
which installs ``comfy/model_sampling.py::ModelSamplingDiscreteFlow`` with ``multiplier=1000``
(``MOSSImageConfig.sampling_settings``) and the ``CONST`` prediction type. Everything in this
module reproduces that object and the KSampler code that reads it, from the pinned reference
``Comfy-Org/ComfyUI@bdc63e72f5ab29fde448c161db3057827148c0ef`` (0.41.0):

* the 1000-entry table ``sigma((i + 1) / 1000 * 1000)``, shifted by ``time_snr_shift``;
* ``timestep(sigma) = sigma * 1000`` -- note this is *not* the inverse of ``sigma()`` for a
  shifted flow, so ``normal`` / ``sgm_uniform`` interpolate between ``1000`` and ``1000 *
  shift(0.001)`` and then shift again, exactly as ComfyUI does;
* the CONST denoised estimate ``x - sigma * v`` in float32, classifier-free guidance combined on
  the denoised estimates, and the k-diffusion Euler / Heun / RF-ancestral updates.

The standalone Diffusers pipeline published with the weights snaps its sigmas to the nearest
training grid point; that profile is deliberately absent here, so no request can switch to it
silently. Shift is a model-sampling parameter applied exactly once -- the schedule is built from the
shifted table and nothing downstream shifts again.

Negative-branch skipping is ComfyUI's ``SkipLayerGuidanceDiTSimple``: only the unconditional
forward replaces the listed blocks with the identity, only while the current sigma lies inside the
``[percent_to_sigma(end), percent_to_sigma(start)]`` window, and only when an unconditional branch
runs at all (CFG != 1). It never adds a forward. Standard positive-branch SLG is a different
technique (a third forward) and is refused by this engine for now.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import torch

# The official workflow: `ModelSamplingSD3(shift=5)` -> `SkipLayerGuidanceDiTSimple(single_layers="22",
# 0, 1)` -> `KSampler(euler, sgm_uniform, 50 steps, cfg 3)`.
MOSS_DEFAULT_SHIFT = 5.0
MOSS_MULTIPLIER = 1000
MOSS_TIMESTEPS = 1000
MOSS_DEFAULT_STEPS = 50
MOSS_DEFAULT_CFG = 3.0
MOSS_DEFAULT_SAMPLER = "euler"
MOSS_DEFAULT_SCHEDULER = "sgm_uniform"
# Zero-based main DiT block indices. The DiT has 42 blocks.
MOSS_DEFAULT_UNCOND_SKIP_LAYERS = (22,)
MOSS_DEFAULT_SKIP_START = 0.0
MOSS_DEFAULT_SKIP_END = 1.0

# A sampler is offered only when its update is the k-diffusion one, term for term. Every other
# KSampler name is refused rather than approximated: an approximation would be a different image
# under the same name.
MOSS_SAMPLERS = ("euler", "euler_ancestral", "heun")
MOSS_SCHEDULERS = (
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

MOSS_MIN_SHIFT = 0.01
MOSS_MAX_SHIFT = 100.0
# `int(steps / denoise)` grows without bound as denoise approaches 0; past this the request is refused.
MOSS_MAX_REFINEMENT_SCHEDULE_STEPS = 4096


def time_snr_shift(alpha: float, t):
    """``comfy/model_sampling.py::time_snr_shift``, including its identity at alpha 1."""
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


def validate_shift(shift) -> float:
    if isinstance(shift, bool) or not isinstance(shift, (int, float)) or not math.isfinite(float(shift)):
        raise ValueError("MOSS shift must be a finite number")
    value = float(shift)
    if not MOSS_MIN_SHIFT <= value <= MOSS_MAX_SHIFT:
        raise ValueError(f"MOSS shift must be between {MOSS_MIN_SHIFT:g} and {MOSS_MAX_SHIFT:g}")
    return value


class MossModelSampling:
    """``ModelSamplingDiscreteFlow`` with ``set_parameters(shift, timesteps=1000, multiplier=1000)``."""

    def __init__(self, shift: float = MOSS_DEFAULT_SHIFT, multiplier: int = MOSS_MULTIPLIER, timesteps: int = MOSS_TIMESTEPS):
        self.shift = validate_shift(shift)
        self.multiplier = multiplier
        # `torch.arange(1, timesteps + 1, 1) / timesteps` is an int64 tensor divided by a Python int:
        # float32 under the default dtype, then multiplied back by the multiplier, exactly as ComfyUI.
        self.sigmas = self.sigma((torch.arange(1, timesteps + 1, 1) / timesteps) * multiplier)

    @property
    def sigma_min(self) -> torch.Tensor:
        return self.sigmas[0]

    @property
    def sigma_max(self) -> torch.Tensor:
        return self.sigmas[-1]

    def timestep(self, sigma):
        return sigma * self.multiplier

    def sigma(self, timestep):
        return time_snr_shift(self.shift, timestep / self.multiplier)

    def percent_to_sigma(self, percent: float) -> float:
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return time_snr_shift(self.shift, 1.0 - percent)


# -- schedulers (comfy/samplers.py and comfy/k_diffusion/sampling.py, verbatim semantics) ----------


def _append_zero(values: torch.Tensor) -> torch.Tensor:
    return torch.cat([values, values.new_zeros([1])])


def _simple(ms: MossModelSampling, steps: int) -> torch.Tensor:
    sigs = []
    ss = len(ms.sigmas) / steps
    for x in range(steps):
        sigs += [float(ms.sigmas[-(1 + int(x * ss))])]
    sigs += [0.0]
    return torch.FloatTensor(sigs)


def _ddim_uniform(ms: MossModelSampling, steps: int) -> torch.Tensor:
    x = 1
    if math.isclose(float(ms.sigmas[x]), 0, abs_tol=0.00001):
        steps += 1
        sigs = []
    else:
        sigs = [0.0]
    ss = max(len(ms.sigmas) // steps, 1)
    while x < len(ms.sigmas):
        sigs += [float(ms.sigmas[x])]
        x += ss
    sigs = sigs[::-1]
    return torch.FloatTensor(sigs)


def _normal(ms: MossModelSampling, steps: int, sgm: bool = False) -> torch.Tensor:
    start = ms.timestep(ms.sigma_max)
    end = ms.timestep(ms.sigma_min)
    append_zero = True
    if sgm:
        timesteps = torch.linspace(start, end, steps + 1)[:-1]
    else:
        if math.isclose(float(ms.sigma(end)), 0, abs_tol=0.00001):
            steps += 1
            append_zero = False
        timesteps = torch.linspace(start, end, steps)
    sigs = [float(ms.sigma(timesteps[x])) for x in range(len(timesteps))]
    if append_zero:
        sigs += [0.0]
    return torch.FloatTensor(sigs)


def _beta(ms: MossModelSampling, steps: int, alpha: float = 0.6, beta: float = 0.6) -> torch.Tensor:
    import numpy
    from scipy.stats import beta as beta_distribution

    total_timesteps = len(ms.sigmas) - 1
    ts = 1 - numpy.linspace(0, 1, steps, endpoint=False)
    ts = numpy.rint(beta_distribution.ppf(ts, alpha, beta) * total_timesteps)
    sigs = []
    last_t = -1
    for t in ts:
        if t != last_t:
            sigs += [float(ms.sigmas[int(t)])]
        last_t = t
    sigs += [0.0]
    return torch.FloatTensor(sigs)


def _linear_quadratic(ms: MossModelSampling, steps: int, threshold_noise: float = 0.025, linear_steps=None) -> torch.Tensor:
    if steps == 1:
        sigma_schedule = [1.0, 0.0]
    else:
        if linear_steps is None:
            linear_steps = steps // 2
        linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
        threshold_noise_step_diff = linear_steps - threshold_noise * steps
        quadratic_steps = steps - linear_steps
        quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps ** 2)
        linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps ** 2)
        const = quadratic_coef * (linear_steps ** 2)
        quadratic_sigma_schedule = [
            quadratic_coef * (i ** 2) + linear_coef * i + const for i in range(linear_steps, steps)
        ]
        sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
        sigma_schedule = [1.0 - x for x in sigma_schedule]
    return torch.FloatTensor(sigma_schedule) * ms.sigma_max.cpu()


def _karras(n: int, sigma_min: float, sigma_max: float, rho: float = 7.0) -> torch.Tensor:
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    return _append_zero((max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho)


def _exponential(n: int, sigma_min: float, sigma_max: float) -> torch.Tensor:
    return _append_zero(torch.linspace(math.log(sigma_max), math.log(sigma_min), n).exp())


def _kl_optimal(n: int, sigma_min: float, sigma_max: float) -> torch.Tensor:
    if n < 2:
        # ComfyUI divides by n - 1 here and would hand the sampler a NaN schedule.
        raise ValueError("The kl_optimal scheduler needs at least 2 steps")
    adj_idxs = torch.arange(n, dtype=torch.float).div_(n - 1)
    sigmas = adj_idxs.new_zeros(n + 1)
    sigmas[:-1] = (adj_idxs * math.atan(sigma_min) + (1 - adj_idxs) * math.atan(sigma_max)).tan_()
    return sigmas


def _calculate_sigmas(ms: MossModelSampling, scheduler: str, steps: int) -> torch.Tensor:
    if scheduler == "simple":
        return _simple(ms, steps)
    if scheduler == "sgm_uniform":
        return _normal(ms, steps, sgm=True)
    if scheduler == "normal":
        return _normal(ms, steps)
    if scheduler == "ddim_uniform":
        return _ddim_uniform(ms, steps)
    if scheduler == "beta":
        return _beta(ms, steps)
    if scheduler == "linear_quadratic":
        return _linear_quadratic(ms, steps)
    sigma_min, sigma_max = float(ms.sigma_min), float(ms.sigma_max)
    if scheduler == "karras":
        return _karras(steps, sigma_min, sigma_max)
    if scheduler == "exponential":
        return _exponential(steps, sigma_min, sigma_max)
    if scheduler == "kl_optimal":
        return _kl_optimal(steps, sigma_min, sigma_max)
    raise ValueError(f"Unsupported MOSS scheduler: {scheduler}")


def validate_steps(steps) -> int:
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer")
    return steps


def moss_sigma_schedule(steps: int, scheduler: str, shift: float = MOSS_DEFAULT_SHIFT) -> torch.Tensor:
    """The KSampler sigma schedule for a full-denoise text-to-image run."""
    validate_steps(steps)
    if scheduler not in MOSS_SCHEDULERS:
        raise ValueError(f"Unsupported MOSS scheduler: {scheduler}")
    sigmas = _calculate_sigmas(MossModelSampling(shift), scheduler, steps).to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(sigmas).all() or sigmas[-1].item() != 0.0:
        raise RuntimeError(f"MOSS scheduler {scheduler} produced an invalid sigma sequence")
    if not torch.all(sigmas[:-1] >= sigmas[1:]):
        raise RuntimeError(f"MOSS scheduler {scheduler} produced a non-monotonic sigma sequence")
    return sigmas.contiguous()


def validate_denoise(denoise) -> float:
    if isinstance(denoise, bool) or not isinstance(denoise, (int, float)) or not math.isfinite(float(denoise)):
        raise ValueError("denoise must be finite")
    if float(denoise) <= 0.0 or float(denoise) > 1.0:
        raise ValueError("denoise must be greater than 0 and at most 1")
    return float(denoise)


def moss_refinement_sigma_schedule(steps: int, denoise: float, scheduler: str, shift: float = MOSS_DEFAULT_SHIFT):
    """``KSampler.set_steps`` with a denoise: build ``int(steps / denoise)`` sigmas, keep the last ``steps + 1``.

    A denoise above 0.9999 is the full schedule. Below it the longer trajectory is built at the same
    shift and only its tail is kept, so every requested step runs and the denoise chooses where on the
    trajectory the redraw starts. Returns ``(sigmas, diagnostics)``.
    """
    validate_steps(steps)
    value = validate_denoise(denoise)
    if value > 0.9999:
        sigmas = moss_sigma_schedule(steps, scheduler, shift)
        return sigmas, {"schedule_steps": steps, "schedule_mode": "full", "start_sigma": float(sigmas[0].item())}
    schedule_steps = int(steps / value)
    if schedule_steps > MOSS_MAX_REFINEMENT_SCHEDULE_STEPS:
        raise ValueError(
            f"MOSS refinement denoise requires {schedule_steps} schedule steps, exceeding the limit of "
            f"{MOSS_MAX_REFINEMENT_SCHEDULE_STEPS}; increase denoise or reduce steps"
        )
    # `beta` and `ddim_uniform` do not promise an exact length; ComfyUI slices whatever came back, and
    # the caller reads the real step count from `len(sigmas) - 1`.
    sigmas = moss_sigma_schedule(schedule_steps, scheduler, shift)[-(steps + 1):].clone()
    if len(sigmas) < 2:
        raise RuntimeError("MOSS refinement produced an empty sigma suffix")
    return sigmas, {"schedule_steps": schedule_steps, "schedule_mode": "comfy_suffix", "start_sigma": float(sigmas[0].item())}


# -- guidance windows -------------------------------------------------------------------------


@dataclass(frozen=True)
class MossGuidancePlan:
    """What a MOSS run does beyond plain CFG, resolved once per job."""

    cfg: float
    uncond_skip_layers: tuple[int, ...]
    skip_start: float
    skip_end: float
    shift: float

    @property
    def runs_unconditional(self) -> bool:
        # ComfyUI drops the unconditional branch when cfg == 1 (`cfg_function`'s optimisation), so
        # negative-branch skipping has no branch to act on there.
        return float(self.cfg) != 1.0

    @property
    def skip_active(self) -> bool:
        return self.runs_unconditional and bool(self.uncond_skip_layers)

    def skip_window(self) -> tuple[float, float]:
        ms = MossModelSampling(self.shift)
        return float(ms.percent_to_sigma(self.skip_end)), float(ms.percent_to_sigma(self.skip_start))

    def skip_layers_at(self, sigma) -> frozenset:
        """The blocks the unconditional forward skips at this sigma (empty outside the window)."""
        if not self.skip_active:
            return frozenset()
        sigma_end, sigma_start = self.skip_window()
        value = float(torch.as_tensor(sigma).reshape(-1)[0].item())
        if sigma_end <= value <= sigma_start:
            return frozenset(self.uncond_skip_layers)
        return frozenset()


def validate_skip_layers(layers, depth: int) -> tuple[int, ...]:
    if layers is None:
        return ()
    if isinstance(layers, (str, bytes)) or not isinstance(layers, Sequence):
        raise ValueError("uncond_skip_layers must be a list of block indices")
    result = []
    for layer in layers:
        if isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < depth:
            raise ValueError(f"uncond_skip_layers must contain integer block indices in [0, {depth})")
        if layer in result:
            raise ValueError(f"uncond_skip_layers lists block {layer} twice")
        result.append(layer)
    return tuple(result)


def validate_percent_window(start, end) -> tuple[float, float]:
    for name, value in (("start", start), ("end", end)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"uncond skip {name} must be a finite number")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"uncond skip {name} must be between 0 and 1")
    if float(start) > float(end):
        raise ValueError("uncond skip start must not be later than its end")
    return float(start), float(end)


def build_guidance_plan(cfg, uncond_skip_layers=MOSS_DEFAULT_UNCOND_SKIP_LAYERS, skip_start=MOSS_DEFAULT_SKIP_START,
                        skip_end=MOSS_DEFAULT_SKIP_END, shift=MOSS_DEFAULT_SHIFT, depth: int = 42) -> MossGuidancePlan:
    if isinstance(cfg, bool) or not isinstance(cfg, (int, float)) or not math.isfinite(float(cfg)) or float(cfg) < 0:
        raise ValueError("cfg must be a finite, non-negative number")
    start, end = validate_percent_window(skip_start, skip_end)
    return MossGuidancePlan(
        cfg=float(cfg),
        uncond_skip_layers=validate_skip_layers(uncond_skip_layers, depth),
        skip_start=start,
        skip_end=end,
        shift=validate_shift(shift),
    )


# -- samplers (comfy/k_diffusion/sampling.py, CONST model) ------------------------------------


Denoiser = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]


def _to_d(x: torch.Tensor, sigma: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
    return (x - denoised) / sigma.reshape(-1, *([1] * (x.ndim - 1)))


def _cpu_noise(sample: torch.Tensor, generators: Sequence[torch.Generator]) -> torch.Tensor:
    if len(generators) != sample.shape[0]:
        raise ValueError(f"Expected {sample.shape[0]} CPU generators, got {len(generators)}")
    noise = []
    for index, generator in enumerate(generators):
        if not isinstance(generator, torch.Generator) or torch.device(generator.device).type != "cpu":
            raise ValueError(f"generators[{index}] must be a CPU torch.Generator")
        noise.append(torch.randn(sample[index:index + 1].shape, generator=generator, dtype=torch.float32, device="cpu"))
    return torch.cat(noise).to(sample.device)


def sample_moss(
    sampler: str,
    denoise: Denoiser,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    generators: Sequence[torch.Generator] = (),
    on_step: Callable[[int, int, torch.Tensor], None] | None = None,
    blend: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, int]:
    """Run one k-diffusion sampler over ``sigmas`` and return ``(latents, model evaluations)``.

    ``denoise(x, sigma_batch, step_index)`` returns the guided CONST denoised estimate in float32.
    The ancestral noise comes from the per-image CPU generators that already drew the initial noise,
    which is this project's seed contract for every native engine (ComfyUI would draw it on the GPU).
    ``blend(x, sigma_next)``, when given, runs after every update and before ``on_step`` -- a masked
    redraw puts the unmasked area back on the source's trajectory there.
    """
    if sampler not in MOSS_SAMPLERS:
        raise ValueError(f"Unsupported MOSS sampler: {sampler}")
    x = x.to(dtype=torch.float32)
    sigmas = sigmas.to(device=x.device, dtype=torch.float32)
    s_in = x.new_ones([x.shape[0]])
    total = len(sigmas) - 1
    evaluations = 0
    for i in range(total):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        denoised = denoise(x, sigma * s_in, i)
        evaluations += 1
        if sampler == "euler":
            d = _to_d(x, sigma * s_in, denoised)
            x = x + d * (sigma_next - sigma)
        elif sampler == "heun":
            d = _to_d(x, sigma * s_in, denoised)
            dt = sigma_next - sigma
            if sigma_next == 0:
                x = x + d * dt
            else:
                x_2 = x + d * dt
                denoised_2 = denoise(x_2, sigma_next * s_in, i)
                evaluations += 1
                d_2 = _to_d(x_2, sigma_next * s_in, denoised_2)
                x = x + (d + d_2) / 2 * dt
        else:  # euler_ancestral: sample_euler_ancestral_RF, eta 1, s_noise 1
            if sigma_next == 0:
                x = denoised
            else:
                downstep_ratio = 1 + (sigma_next / sigma - 1) * 1.0
                sigma_down = sigma_next * downstep_ratio
                alpha_ip1 = 1 - sigma_next
                alpha_down = 1 - sigma_down
                renoise_coeff = (sigma_next ** 2 - sigma_down ** 2 * alpha_ip1 ** 2 / alpha_down ** 2) ** 0.5
                ratio = sigma_down / sigma
                x = ratio * x + (1 - ratio) * denoised
                x = (alpha_ip1 / alpha_down) * x + _cpu_noise(x, generators) * renoise_coeff
        x = x.to(dtype=torch.float32)
        if blend is not None:
            x = blend(x, sigma_next)
        if on_step is not None:
            on_step(i + 1, total, x)
    return x, evaluations


def cfg_denoised(x: torch.Tensor, sigma: torch.Tensor, cond_v: torch.Tensor, uncond_v: torch.Tensor | None, cfg: float) -> torch.Tensor:
    """CONST denoised estimates combined the way ``comfy/samplers.py::cfg_function`` combines them."""
    sigma_b = sigma.reshape(-1, *([1] * (x.ndim - 1))).to(torch.float32)
    cond = x - cond_v.float() * sigma_b
    if uncond_v is None:
        return cond
    uncond = x - uncond_v.float() * sigma_b
    return uncond + (cond - uncond) * float(cfg)


def moss_sampling_diagnostics(sampler: str, scheduler: str, width: int, height: int, plan: MossGuidancePlan) -> dict:
    if sampler not in MOSS_SAMPLERS:
        raise ValueError(f"Unsupported MOSS sampler: {sampler}")
    if scheduler not in MOSS_SCHEDULERS:
        raise ValueError(f"Unsupported MOSS scheduler: {scheduler}")
    sigma_end, sigma_start = plan.skip_window()
    return {
        "requested_sampler": sampler,
        "sampler_implementation": f"k_diffusion_{sampler}" if sampler != "euler_ancestral" else "k_diffusion_euler_ancestral_RF",
        "requested_scheduler": scheduler,
        "scheduler_implementation": f"comfy_{scheduler}_discrete_flow",
        "model_sampling": "ModelSamplingDiscreteFlow",
        "prediction": "CONST",
        "shift": round(plan.shift, 6),
        "shift_source": "ModelSamplingSD3",
        "multiplier": MOSS_MULTIPLIER,
        "cfg_scale": round(plan.cfg, 4),
        "cfg_branches": 2 if plan.runs_unconditional else 1,
        "uncond_skip_layers": list(plan.uncond_skip_layers),
        "uncond_skip_active": plan.skip_active,
        "uncond_skip_percent": [plan.skip_start, plan.skip_end],
        "uncond_skip_sigma_window": [sigma_end, sigma_start],
        "standard_slg": "not_supported",
        "schedule_profile": "comfyui",
        "latent_grid": [int(height) // 16, int(width) // 16],
        "image_tokens": (int(height) // 16) * (int(width) // 16),
    }
