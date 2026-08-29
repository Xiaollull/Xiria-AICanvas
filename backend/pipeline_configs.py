import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

try:
    from .model_paths import resolve_model_directory
except ImportError:
    from model_paths import resolve_model_directory


PIPELINE_CONFIG_REPOSITORIES = {
    "sd": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
}
PIPELINE_CONFIG_ENDPOINTS = ("https://huggingface.co", "https://hf-mirror.com")
PIPELINE_CONFIG_PATTERNS = ["**/*.json", "*.json", "*.txt", "**/*.txt", "**/*.model"]
COMMON_REQUIRED_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "tokenizer/merges.txt",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "unet/config.json",
    "vae/config.json",
)
FAMILY_REQUIRED_FILES = {
    "sd": COMMON_REQUIRED_FILES,
    "sdxl": COMMON_REQUIRED_FILES + (
        "text_encoder_2/config.json",
        "tokenizer_2/merges.txt",
        "tokenizer_2/tokenizer_config.json",
        "tokenizer_2/vocab.json",
    ),
}
_PIPELINE_CONFIG_CACHE = {}


def failure_reason(error: BaseException):
    chain = []
    seen = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    root = chain[-1]
    summary = " ".join(str(root).split())[:200] or type(root).__name__
    if root is error:
        return f"{type(error).__name__}: {summary}"
    return f"{type(error).__name__} <- {type(root).__name__}: {summary}"


def pipeline_config_complete(directory: str | Path, family: str):
    root = Path(directory)
    return family in FAMILY_REQUIRED_FILES and all((root / name).is_file() for name in FAMILY_REQUIRED_FILES[family])


def find_pipeline_config(family: str, cache_dir: str | Path | None = None):
    repository = PIPELINE_CONFIG_REPOSITORIES.get(family)
    if repository is None:
        raise ValueError(f"Unsupported pipeline family: {family}")
    cache_key = (family, str(Path(cache_dir).resolve()) if cache_dir else "")
    cached = _PIPELINE_CONFIG_CACHE.get(cache_key)
    if cached and pipeline_config_complete(cached, family):
        return cached
    _PIPELINE_CONFIG_CACHE.pop(cache_key, None)
    try:
        directory = snapshot_download(
            repository,
            cache_dir=str(cache_dir) if cache_dir else None,
            local_files_only=True,
            allow_patterns=PIPELINE_CONFIG_PATTERNS,
        )
    except Exception:
        return None
    if not pipeline_config_complete(directory, family):
        return None
    resolved = Path(directory)
    _PIPELINE_CONFIG_CACHE[cache_key] = resolved
    return resolved


def require_pipeline_config(family: str, cache_dir: str | Path | None = None):
    directory = find_pipeline_config(family, cache_dir)
    if directory is None:
        label = "SDXL / Illustrious" if family == "sdxl" else "Stable Diffusion 1.x"
        raise RuntimeError(f"缺少 {label} 本地运行配置，请联网重新运行环境配置器后再生成。")
    return directory


def installed_pipeline_families(project_root: str | Path, prepare_all: bool = True):
    checkpoint_directories = {
        "sd": resolve_model_directory(project_root, "checkpoints", "sd"),
        "sdxl": resolve_model_directory(project_root, "checkpoints", "illustrious"),
    }
    installed = [
        family
        for family, directory in checkpoint_directories.items()
        if directory.is_dir() and any(path.is_file() and path.suffix.lower() in {".safetensors", ".ckpt", ".pt", ".pth"} for path in directory.rglob("*"))
    ]
    if installed or not prepare_all:
        return installed
    return list(PIPELINE_CONFIG_REPOSITORIES)


def download_pipeline_configs(cache_dir: str | Path | None = None, families=None):
    directories = {}
    for family in families or PIPELINE_CONFIG_REPOSITORIES:
        repository = PIPELINE_CONFIG_REPOSITORIES[family]
        endpoints = dict.fromkeys(filter(None, (os.environ.get("HF_ENDPOINT"), *PIPELINE_CONFIG_ENDPOINTS)))
        reasons = {}
        last_error = None
        for endpoint in endpoints:
            print(f"Downloading local runtime config from {endpoint}: {repository}", flush=True)
            try:
                directory = snapshot_download(
                    repository,
                    cache_dir=str(cache_dir) if cache_dir else None,
                    endpoint=endpoint,
                    allow_patterns=PIPELINE_CONFIG_PATTERNS,
                )
                if not pipeline_config_complete(directory, family):
                    raise RuntimeError(f"Downloaded runtime config is incomplete: {repository}")
                directories[family] = Path(directory)
                break
            except Exception as error:
                last_error = error
                reasons[endpoint] = failure_reason(error)
                print(f"Runtime config route failed ({endpoint}): {reasons[endpoint]}", file=sys.stderr, flush=True)
        else:
            # All endpoints failed; check if a complete local cache already exists.
            existing = find_pipeline_config(family, cache_dir)
            if existing:
                print(f"Using existing local runtime config: {family} -> {existing}", flush=True)
                directories[family] = existing
            else:
                cache_hint = f" (cache: {Path(cache_dir).resolve()})" if cache_dir else ""
                routes = "; ".join(f"{endpoint} => {reason}" for endpoint, reason in reasons.items())
                raise RuntimeError(f"Unable to download runtime config: {repository}{cache_hint} [{routes}]") from last_error
    return directories


def main():
    cache_dir = os.environ.get("HF_HUB_CACHE")
    installed_index = sys.argv.index("--installed") if "--installed" in sys.argv else -1
    if installed_index >= 0 and installed_index + 1 >= len(sys.argv):
        raise RuntimeError("--installed requires the project root")
    # --required narrows a fresh install from "prepare both families" to "only what
    # installed checkpoints actually need", so a blocked Hugging Face route cannot
    # report a broken environment for a config nothing can use yet.
    prepare_all = "--required" not in sys.argv
    families = installed_pipeline_families(sys.argv[installed_index + 1], prepare_all) if installed_index >= 0 else list(PIPELINE_CONFIG_REPOSITORIES)
    if "--download" in sys.argv:
        download_pipeline_configs(cache_dir, families)
    elif "--check" in sys.argv:
        missing = [family for family in families if find_pipeline_config(family, cache_dir) is None]
        if missing:
            raise RuntimeError(f"Missing local runtime configs: {', '.join(missing)}")
    else:
        raise RuntimeError("Use --check or --download")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # A Hugging Face outage is an environment condition, not a defect: report the
        # root cause on one line instead of a traceback the operator has to read past.
        print(" ".join(str(error).split()) or failure_reason(error), file=sys.stderr, flush=True)
        raise SystemExit(1) from None
