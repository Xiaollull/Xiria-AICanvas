import json
from pathlib import Path


DEFAULT_MODEL_PATHS = {
    "checkpoints": {"sd": "models/checkpoints/sd", "illustrious": "models/checkpoints/illustrious"},
    "loras": {
        "sd": "models/loras/sd",
        "illustrious": "models/loras/illustrious",
        "anima": "models/loras/anima",
        "flux": "models/loras/flux",
        "flux2": "models/loras/flux2",
        "krea2": "models/loras/krea2",
    },
    "vae": "models/vae",
    "diffusion_models": "models/diffusion_models",
    "text_encoders": "models/text_encoders",
    "embeddings": "models/embeddings",
    "upscalers": "models/upscalers",
    "configs": "models/configs",
    "yolo": "models/yolo",
    "background_removal": "models/background-removal",
}


def load_model_paths(project_root: str | Path):
    root = Path(project_root).resolve()
    manifest = root / "models" / "model-paths.json"
    try:
        configured = json.loads(manifest.read_text(encoding="utf-8"))
    except FileNotFoundError:
        configured = {}
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Model path configuration is unavailable or invalid") from error
    if not isinstance(configured, dict):
        raise ValueError("Model path configuration root must be an object")
    for key in ("checkpoints", "loras"):
        if key in configured and not isinstance(configured[key], dict):
            raise ValueError(f"Configured {key} model paths must be an object")
    merged = {**DEFAULT_MODEL_PATHS, **configured}
    merged["checkpoints"] = {**DEFAULT_MODEL_PATHS["checkpoints"], **(configured.get("checkpoints") or {})}
    merged["loras"] = {**DEFAULT_MODEL_PATHS["loras"], **(configured.get("loras") or {})}
    return merged


def resolve_model_directory(project_root: str | Path, config_key: str, engine_key: str | None = None):
    root = Path(project_root).resolve()
    models_root = (root / "models").resolve()
    config = load_model_paths(root)
    value = config.get(config_key)
    if engine_key is not None:
        value = value.get(engine_key) if isinstance(value, dict) else None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Configured {config_key} model path must be a non-empty string")
    relative = Path(value.strip())
    if relative.is_absolute():
        raise ValueError("Configured model paths must be project-relative")
    directory = (root / relative).resolve()
    if directory == models_root or models_root not in directory.parents:
        raise ValueError("Configured model paths must stay inside the project models directory")
    return directory
