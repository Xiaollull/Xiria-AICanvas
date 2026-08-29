"""Benchmark-only Hires stage artifact capture.

This is evidence tooling, not a product feature. It is default-off, only reachable from an
isolated benchmark child bound to a `hires_quality_artifacts` lease, writes nothing at all
when disabled, and never records Prompt plaintext, lease nonces or absolute paths.
"""
import hashlib
import json
import os
import uuid
from pathlib import Path, PurePosixPath, PureWindowsPath

try:
    from .benchmark_lease import HIRES_ARTIFACT_PURPOSE
except ImportError:
    from benchmark_lease import HIRES_ARTIFACT_PURPOSE

SCHEMA_VERSION = 1
PURPOSE = HIRES_ARTIFACT_PURPOSE
CAPTURE_ENV = "XIRAI_HIRES_ARTIFACT_CAPTURE"
COLOR_SPACE = "srgb_untagged_8bit"
BASE_DECODED = "base-decoded.png"
POST_SR = "hires-post-sr.png"
FINAL = "final-composited.png"
MANIFEST = "manifest.json"
MAIN_INFERENCE_PORT = 8718
# Keys whose values are free text or secrets; they must never reach the manifest.
FORBIDDEN_MANIFEST_KEYS = frozenset({
    "prompt", "negative_prompt", "nonce", "lease_nonce", "command_line", "cmdline",
    "image", "images", "blob", "data", "token", "secret", "password", "authorization",
})


def _sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def prompt_facts(prompt, negative_prompt):
    """Return only hash/length facts; the plaintext is never retained or returned."""
    prompt = "" if prompt is None else str(prompt)
    negative_prompt = "" if negative_prompt is None else str(negative_prompt)
    return {
        "prompt_sha256": _sha256_text(prompt), "prompt_length": len(prompt),
        "negative_prompt_sha256": _sha256_text(negative_prompt), "negative_prompt_length": len(negative_prompt),
    }


def canonical_parameter_digest(parameters):
    """Stable SHA-256 over a redaction-checked parameter mapping."""
    return _sha256_text(json.dumps(redacted(parameters), sort_keys=True, separators=(",", ":")))


def _looks_absolute(value):
    # `Path.is_absolute()` is platform dependent: on Windows a POSIX path such as "/home/x" is not
    # absolute, and on POSIX a "D:\..." path is not absolute. Both flavours must be refused on both
    # platforms, so never ask the host-flavoured Path.
    if not isinstance(value, str) or not value:
        return False
    if value[0] in "/\\":
        return True
    windows_drive = len(value) > 1 and value[0].isalpha() and value[1] == ":"
    return windows_drive or PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()


def redacted(value, _key=None):
    """Reject forbidden keys, absolute paths and non-JSON-safe values before they are recorded."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("artifact manifest keys must be strings")
            if key.lower() in FORBIDDEN_MANIFEST_KEYS:
                raise ValueError(f"artifact manifest refuses redacted field: {key}")
            result[key] = redacted(item, key)
        return result
    if isinstance(value, (list, tuple)):
        return [redacted(item, _key) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        if _looks_absolute(value):
            raise ValueError("artifact manifest refuses absolute paths")
        return value
    raise ValueError(f"artifact manifest refuses unsupported value type: {type(value).__name__}")


def capture_requested(environ=None):
    """Environment switch only; it is deliberately not a user setting."""
    environ = os.environ if environ is None else environ
    return environ.get(CAPTURE_ENV, "").strip() == "1"


def capture_gate(
    *, environ, inference_port, project_root, state_root, output_root, workspace_id,
    lease_validator, main_port=MAIN_INFERENCE_PORT,
):
    """Fail-closed isolation gate; returns diagnostics-safe fields only (never the nonce)."""
    environ = os.environ if environ is None else environ
    project_root, state_root, output_root = Path(project_root).resolve(), Path(state_root).resolve(), Path(output_root).resolve()
    checks = {
        "capture_requested": capture_requested(environ),
        "benchmark_child": environ.get("XIRAI_BENCHMARK_CHILD", "").strip() == "1",
        "purpose": environ.get("XIRAI_BENCHMARK_PURPOSE", "").strip() == PURPOSE,
        "isolated_port": inference_port != main_port,
        "isolated_state_root": state_root != (project_root / "state-cache").resolve(),
        "isolated_output_root": output_root != (project_root / "outputs").resolve(),
        "isolated_workspace": bool(str(workspace_id or "").strip()),
    }
    if not all(checks.values()):
        reason = next(key for key, value in checks.items() if not value)
        return {"enabled": False, "reason": reason, "purpose": PURPOSE, "lease_valid": False, "lease_expiry": None}
    lease = lease_validator()
    return {
        "enabled": bool(lease.get("valid")),
        "reason": "enabled" if lease.get("valid") else str(lease.get("reason")),
        "purpose": PURPOSE,
        "lease_valid": bool(lease.get("valid")),
        "lease_expiry": lease.get("expiry"),
    }


class NullStageCapture:
    """The production object: every method is a no-op and no directory is ever created."""

    enabled = False

    def stage(self, _name, _image):
        return None

    def tile(self, _index, _image):
        return None

    def composite(self, _index, _image):
        return None

    def finish(self, **_facts):
        return None

    def abort(self):
        return None


NULL_CAPTURE = NullStageCapture()


class StageArtifactCapture:
    """Contained, atomic, benchmark-only PNG + manifest writer."""

    enabled = True

    def __init__(self, root, *, schema_version=SCHEMA_VERSION, purpose=PURPOSE):
        self.root = Path(root).resolve()
        self.schema_version = schema_version
        self.purpose = purpose
        self.artifacts = []
        self._written = []
        self._created_root = False
        self._closed = False

    # -- internals -------------------------------------------------------
    def _target(self, relative):
        if not relative or relative.startswith("/") or relative.startswith("\\") or ".." in Path(relative).parts:
            raise ValueError("artifact path must be a contained relative path")
        target = (self.root / relative).resolve()
        try:
            target.relative_to(self.root)
        except ValueError as error:
            raise ValueError("artifact path escaped the capture root") from error
        return target

    def _ensure_root(self):
        if not self.root.exists():
            self.root.mkdir(parents=True, exist_ok=True)
            self._created_root = True

    def _write_atomic(self, target, write):
        self._ensure_root()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}-{uuid.uuid4().hex}.tmp")
        try:
            write(temporary)
            os.replace(temporary, target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        self._written.append(target)

    def _record(self, relative, image):
        if self._closed:
            raise RuntimeError("stage artifact capture is already closed")
        source_mode = image.mode
        # RGB output contract: no resize, no sharpening, no tone or colour adjustment.
        rgb = image if source_mode == "RGB" else image.convert("RGB")
        target = self._target(relative)
        self._write_atomic(target, lambda temporary: rgb.save(temporary, format="PNG", compress_level=6))
        payload = target.read_bytes()
        entry = {
            "path": Path(relative).as_posix(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "width": rgb.width,
            "height": rgb.height,
            "pixel_mode": "RGB",
            "source_pixel_mode": source_mode,
            "color_space": COLOR_SPACE,
        }
        self.artifacts.append(entry)
        return entry

    # -- public stages ---------------------------------------------------
    def stage(self, name, image):
        names = {"base": BASE_DECODED, "post_sr": POST_SR, "final": FINAL}
        if name not in names:
            raise ValueError(f"unknown capture stage: {name}")
        return self._record(names[name], image)

    def tile(self, index, image):
        return self._record(f"tiles/{int(index):02d}-restored.png", image)

    def composite(self, index, image):
        return self._record(f"composites/{int(index):02d}-after-composite.png", image)

    def finish(self, *, status="complete", **facts):
        """Write the manifest atomically and close the capture."""
        if self._closed:
            raise RuntimeError("stage artifact capture is already closed")
        manifest = redacted({
            "schema_version": self.schema_version,
            "purpose": self.purpose,
            "status": status,
            **facts,
        })
        manifest["artifacts"] = list(self.artifacts)
        target = self._target(MANIFEST)
        self._write_atomic(
            target,
            lambda temporary: temporary.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
            ),
        )
        self._closed = True
        return manifest

    def abort(self):
        """Remove every partial artifact this capture created; never touch anything else."""
        for target in reversed(self._written):
            try:
                target.unlink(missing_ok=True)
            except OSError:
                pass
        for directory in ("composites", "tiles"):
            candidate = self.root / directory
            if candidate.is_dir() and not any(candidate.iterdir()):
                candidate.rmdir()
        if self._created_root and self.root.is_dir() and not any(self.root.iterdir()):
            self.root.rmdir()
        self._written.clear()
        self.artifacts.clear()
        self._closed = True
