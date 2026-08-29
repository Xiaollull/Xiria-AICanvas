import math

import torch


GUIDANCE_MODES = ("none", "pag", "cfg_zero_star")
# Flow-matching engines whose sampler could carry the CFG-Zero* rescale. FLUX.1 is deliberately
# absent: it is guidance distilled and runs no unconditional branch, so there is nothing to
# rescale against — listing it here would advertise an enhancement it can never gain.
CFG_ZERO_STAR_ENGINES = frozenset({"Anima", "Krea2"})
PAG_SCALE = 0.3
PAG_APPLIED_LAYERS = "mid"
PAG_LAYER_PATTERNS = {"mid": "mid", "all": ".*"}


def cfg_zero_star_zero_steps(total_steps: int) -> int:
    """Use the paper's suggested first 4% of denoising steps."""
    return max(1, math.floor(total_steps * 0.04))


def guidance_prediction_copies(mode: str, guidance_scale: float) -> int:
    if mode == "pag":
        return 3 if guidance_scale > 1 else 2
    if mode == "cfg_zero_star":
        return 2
    return 2 if guidance_scale > 1 else 1


def cfg_zero_star_scale(conditioned: torch.Tensor, unconditioned: torch.Tensor, eps: float = 1e-8):
    conditioned_flat = conditioned.flatten(1).float()
    unconditioned_flat = unconditioned.flatten(1).float()
    dot_product = torch.sum(conditioned_flat * unconditioned_flat, dim=1, keepdim=True)
    squared_norm = torch.sum(unconditioned_flat**2, dim=1, keepdim=True) + eps
    scale = dot_product / squared_norm
    return scale.to(conditioned.dtype).view(-1, *(1,) * (conditioned.ndim - 1))


def apply_cfg_zero_star(
    conditioned: torch.Tensor,
    unconditioned: torch.Tensor,
    guidance_scale: float,
    step_index: int,
    total_steps: int,
):
    if step_index < cfg_zero_star_zero_steps(total_steps):
        return torch.zeros_like(conditioned)
    optimized_unconditioned = unconditioned * cfg_zero_star_scale(conditioned, unconditioned)
    return optimized_unconditioned + guidance_scale * (conditioned - optimized_unconditioned)


def apply_pag(
    conditioned: torch.Tensor,
    perturbed: torch.Tensor,
    pag_scale: float,
    *,
    unconditioned: torch.Tensor | None = None,
    guidance_scale: float = 1.0,
):
    """Combine sequential native PAG predictions without changing the sampler trajectory."""
    prediction = conditioned
    if unconditioned is not None:
        prediction = unconditioned + guidance_scale * (conditioned - unconditioned)
    return prediction + pag_scale * (conditioned - perturbed)


def pag_layer_pattern(scope: str) -> str:
    return PAG_LAYER_PATTERNS[scope]


def guidance_diagnostics(
    mode: str,
    total_steps: int,
    pag_scale: float = PAG_SCALE,
    pag_applied_layers: str = PAG_APPLIED_LAYERS,
    *,
    engine: str | None = None,
    guidance_scale: float = 1.0,
):
    if mode == "pag":
        if engine == "Anima":
            resolved_layers = []
            if pag_scale != 0:
                resolved_layers = (
                    [f"transformer_blocks.{index}.attn1" for index in range(28)]
                    if pag_applied_layers == "all"
                    else ["transformer_blocks.14.attn1"]
                )
            return {
                "type": "pag",
                "implementation": "native_cosmos_identity_self_attention",
                "prediction_type": "flow_velocity",
                "scale": pag_scale,
                "applied_layers": pag_applied_layers,
                "resolved_layers": resolved_layers,
                "resolved_layer_count": len(resolved_layers),
                "logical_prediction_branches": (
                    guidance_prediction_copies("none", guidance_scale)
                    if pag_scale == 0
                    else guidance_prediction_copies(mode, guidance_scale)
                ),
                "branch_execution": "sequential",
                "peak_forward_copies": 1,
            }
        return {
            "type": "pag",
            "scale": pag_scale,
            "applied_layers": pag_applied_layers,
            "attention_override": pag_applied_layers,
            "implementation": "diffusers_pag",
        }
    if mode == "cfg_zero_star":
        return {
            "type": "cfg_zero_star",
            "optimized_scale": True,
            "zero_steps": cfg_zero_star_zero_steps(total_steps),
            "implementation": "diffusers_native_formula",
        }
    return {"type": "none"}
