"""Read-only access to model folders that live outside the project.

The control plane owns registration (`models/shared-paths.json`); this module
is the inference side of the same contract and deliberately only reads. It is
the Python twin of `src/shared-model-refs.js` + `scripts/shared-model-paths.mjs`,
so the reference format and the root-id derivation are mirrored here rather
than re-invented — `scripts/shared-model-paths.test.mjs` and
`backend/test_shared_model_paths.py` pin the same vectors on both sides.
"""

import hashlib
import json
import os
import re
from pathlib import Path

SHARED_REF_PREFIX = "shared:"
SHARED_ROOT_ID_LENGTH = 12
MAX_SHARED_ROOTS = 24
SHARED_ROOT_KINDS = ("auto", "loras", "checkpoints")
SHARED_LORA_ENGINES = ("SD", "iL", "Anima", "Flux", "Flux2", "Krea2")
# Full engine sets this field has previously shipped with. A root saved while one of these was
# the whole roster predates the engines added since, so it reads as "not configured" rather
# than as a deliberate exclusion. Mirrors `shared-model-refs.js`.
LEGACY_SHARED_LORA_ENGINE_SETS = (
    ("SD", "iL", "Anima"),
    ("SD", "iL", "Anima", "Flux"),
    ("SD", "iL", "Anima", "Flux", "Flux2"),
)
_ROOT_ID_PATTERN = re.compile(rf"^[0-9a-f]{{{SHARED_ROOT_ID_LENGTH}}}$")

# Mirrors SHARED_DIRECTORY_KINDS in src/shared-model-refs.js. Only the kinds the
# runtime can actually load are needed here; the rest are a control-plane
# reporting concern.
SHARED_DIRECTORY_KINDS = {
    "checkpoints": ("checkpoints", "checkpoint", "stable-diffusion", "stablediffusion", "sd", "illustrious", "ckpt"),
    "loras": ("loras", "lora", "locon", "lycoris", "lokr"),
}


def is_shared_ref(value) -> bool:
    return isinstance(value, str) and value.startswith(SHARED_REF_PREFIX)


def is_shared_root_id(value) -> bool:
    return isinstance(value, str) and bool(_ROOT_ID_PATTERN.match(value))


def normalize_shared_root_path(value) -> str:
    """Fold the spelling differences both platforms agree are the same folder."""
    text = str(value or "").strip().replace("\\", "/")
    collapsed = re.sub(r"/{2,}", "/", text)
    return collapsed.rstrip("/") if len(collapsed) > 1 else collapsed


# Case is the one difference the platforms disagree about. Windows treats
# `F:\AI` and `f:\ai` as one folder, so the id must survive a retype; Linux
# treats `/srv/Lora` and `/srv/lora` as two real directories, and folding them
# would hand both the same id. Mirrors SHARED_PATHS_CASE_INSENSITIVE in
# scripts/shared-model-paths.mjs — the two must agree on the same machine or a
# root registered by the control plane will not resolve here.
SHARED_PATHS_CASE_INSENSITIVE = os.name == "nt"


def shared_root_identity_path(value, case_insensitive: bool = False) -> str:
    normalized = normalize_shared_root_path(value)
    return normalized.lower() if case_insensitive else normalized


def shared_root_id(absolute_path, case_insensitive: bool | None = None) -> str:
    if case_insensitive is None:
        case_insensitive = SHARED_PATHS_CASE_INSENSITIVE
    identity = shared_root_identity_path(absolute_path, case_insensitive)
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:SHARED_ROOT_ID_LENGTH]


def parse_shared_ref(value):
    if not is_shared_ref(value):
        return None
    body = value[len(SHARED_REF_PREFIX):]
    separator = body.find("/")
    if separator <= 0:
        return None
    root_id = body[:separator]
    relative_path = body[separator + 1:]
    if not is_shared_root_id(root_id) or not relative_path:
        return None
    segments = relative_path.split("/")
    if any(segment in ("", ".", "..") for segment in segments):
        return None
    return root_id, relative_path


def normalize_shared_engines(value):
    if not isinstance(value, list):
        return list(SHARED_LORA_ENGINES)
    kept = [engine for engine in SHARED_LORA_ENGINES if engine in value]
    if tuple(kept) in LEGACY_SHARED_LORA_ENGINE_SETS:
        return list(SHARED_LORA_ENGINES)
    return kept or list(SHARED_LORA_ENGINES)


def normalize_shared_root(entry):
    if not isinstance(entry, dict):
        return None
    path_value = str(entry.get("path") or "").strip()
    root_id = entry.get("id")
    if not path_value or not is_shared_root_id(root_id):
        return None
    kind = entry.get("kind") if entry.get("kind") in SHARED_ROOT_KINDS else "auto"
    return {
        "id": root_id,
        "path": path_value,
        "kind": kind,
        "enabled": entry.get("enabled") is not False,
        "engines": normalize_shared_engines(entry.get("engines")),
    }


def normalize_shared_roots(config):
    source = config.get("roots") if isinstance(config, dict) else None
    roots = []
    seen = set()
    for entry in source if isinstance(source, list) else []:
        root = normalize_shared_root(entry)
        if root is None or root["id"] in seen:
            continue
        seen.add(root["id"])
        roots.append(root)
        if len(roots) >= MAX_SHARED_ROOTS:
            break
    return roots


def load_shared_roots(project_root: str | Path):
    manifest = Path(project_root).resolve() / "models" / "shared-paths.json"
    try:
        configured = json.loads(manifest.read_text(encoding="utf-8"))
    except FileNotFoundError:
        # Never having shared a folder is the default state, not a fault.
        return []
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Shared model path configuration is unavailable or invalid") from error
    if not isinstance(configured, dict):
        raise ValueError("Shared model path configuration root must be an object")
    return normalize_shared_roots(configured)


def find_shared_root(roots, root_id):
    for root in roots:
        if root["id"] == root_id:
            return root
    return None


def classify_directory_name(name) -> str:
    folded = str(name or "").strip().lower()
    for kind, aliases in SHARED_DIRECTORY_KINDS.items():
        if folded in aliases:
            return kind
    return ""


def shared_kind_directories(root, kind):
    """Directories inside one registered root that hold `kind`.

    Re-derived per call rather than cached in the config, so a folder added to
    the other tool later is picked up without re-registering anything here.
    """
    try:
        resolved = Path(root["path"]).resolve(strict=True)
    except OSError:
        return []
    if not resolved.is_dir():
        return []
    if root["kind"] == kind:
        return [resolved]
    if root["kind"] != "auto":
        return []
    matches = [child for child in sorted(resolved.iterdir()) if child.is_dir() and classify_directory_name(child.name) == kind]
    if not matches and classify_directory_name(resolved.name) == kind:
        return [resolved]
    return matches


def resolve_shared_model(value, project_root: str | Path, extensions=None, label: str = "Shared model") -> Path:
    """Turn a `shared:<id>/<relative>` reference into a real file path.

    Containment is checked against the *resolved* root, so a symlink planted
    inside somebody else's model folder cannot become a way out of it.
    """
    parsed = parse_shared_ref(value)
    if parsed is None:
        raise ValueError(f"{label} shared reference is malformed")
    root_id, relative_path = parsed
    root = find_shared_root(load_shared_roots(project_root), root_id)
    if root is None:
        raise ValueError(f"{label} shared directory is not registered")
    if not root["enabled"]:
        raise ValueError(f"{label} shared directory is disabled")
    if extensions is not None and Path(relative_path).suffix.lower() not in extensions:
        raise ValueError(f"{label} shared file type is not supported")
    try:
        root_real = Path(root["path"]).resolve(strict=True)
        file_real = (root_real / relative_path).resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} shared file does not exist") from error
    if not file_real.is_file() or root_real not in file_real.parents:
        raise ValueError(f"{label} is outside its registered shared directory")
    return file_real


def shared_model_reference(value):
    """A wire-safe label for a shared file: keep the ref, never the disk path."""
    parsed = parse_shared_ref(value)
    if parsed is None:
        return None
    return f"{SHARED_REF_PREFIX}{parsed[0]}/{parsed[1]}"
