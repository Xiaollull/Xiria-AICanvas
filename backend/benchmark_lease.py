"""Local accidental-isolation lease for benchmark-only Anima probes; not a security boundary."""
import hashlib
import hmac
import json
import os
import secrets
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

LEASE_VERSION = 1
LEASE_PURPOSE = "resident_cfg_batch_probe"
HIRES_ARTIFACT_PURPOSE = "hires_quality_artifacts"
# Each purpose is a separate capability; a lease minted for one never satisfies another.
LEASE_PURPOSES = frozenset({LEASE_PURPOSE, HIRES_ARTIFACT_PURPOSE})
LEASE_FILENAME_PREFIXES = {LEASE_PURPOSE: "resident-cfg", HIRES_ARTIFACT_PURPOSE: "hires-artifacts"}
MAX_LEASE_BYTES = 8192
MAX_TTL_SECONDS = 15 * 60


def _resolved(value):
    return Path(value).expanduser().resolve()


def _inside(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _utc_now():
    return datetime.now(timezone.utc)


def create_lease(state_root, output_root, workspace_id, parent_script, ttl_seconds=600, parent_pid=None, parent_executable=None, purpose=LEASE_PURPOSE):
    """Atomically create a harness lease and return (path, raw_nonce)."""
    if not 1 <= int(ttl_seconds) <= MAX_TTL_SECONDS:
        raise ValueError("lease TTL must be between 1 second and 15 minutes")
    if purpose not in LEASE_PURPOSES:
        raise ValueError("lease purpose is not a known benchmark capability")
    state_root, output_root, parent_script = _resolved(state_root), _resolved(output_root), _resolved(parent_script)
    nonce = secrets.token_hex(32)
    lease_dir = state_root / "benchmark"
    lease_dir.mkdir(parents=True, exist_ok=True)
    actual_parent_pid = int(os.getpid() if parent_pid is None else parent_pid)
    if parent_executable is None:
        parent_executable = _default_inspector(actual_parent_pid)["executable"]
    payload = {
        "version": LEASE_VERSION, "purpose": purpose,
        "parent_pid": actual_parent_pid,
        "parent_executable": str(_resolved(parent_executable)),
        "parent_script": str(parent_script), "state_root": str(state_root), "output_root": str(output_root),
        "workspace_id": workspace_id, "nonce_sha256": hashlib.sha256(nonce.encode("utf-8")).hexdigest(),
        "expires_at_utc": (_utc_now() + timedelta(seconds=int(ttl_seconds))).isoformat().replace("+00:00", "Z"),
    }
    path = lease_dir / f"{LEASE_FILENAME_PREFIXES[purpose]}-{secrets.token_hex(8)}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, path)
    return path, nonce


def _default_inspector(pid):
    import psutil
    process = psutil.Process(pid)
    return {"executable": process.exe(), "command_line": process.cmdline()}


def validate_lease(lease_path, nonce, state_root, output_root, workspace_id, project_root, parent_pid=None, now=None, inspector=None, resolver=None, expected_purpose=LEASE_PURPOSE):
    """Return only diagnostics-safe validation fields; never return nonce or command line."""
    fail = lambda reason: {"valid": False, "reason": reason, "expiry": None, "purpose": None}
    try:
        if expected_purpose not in LEASE_PURPOSES:
            return fail("lease_purpose")
        resolve = _resolved if resolver is None else lambda value: Path(resolver(value))
        lease_path, state_root, output_root, project_root = resolve(lease_path), resolve(state_root), resolve(output_root), resolve(project_root)
        if not _inside(lease_path, state_root) or not lease_path.is_file() or lease_path.stat().st_size > MAX_LEASE_BYTES:
            return fail("lease_path")
        data = json.loads(lease_path.read_text(encoding="utf-8"))
        required = {"version", "purpose", "parent_pid", "parent_executable", "parent_script", "state_root", "output_root", "workspace_id", "nonce_sha256", "expires_at_utc"}
        if (
            not isinstance(data, dict) or set(data) != required or data["version"] != LEASE_VERSION
            or data["purpose"] != expected_purpose or not isinstance(data["parent_pid"], int) or isinstance(data["parent_pid"], bool)
            or not all(isinstance(data[key], str) and data[key] for key in required - {"version", "parent_pid"})
            or len(data["nonce_sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in data["nonce_sha256"].lower())
        ):
            return fail("lease_schema")
        try:
            expiry = datetime.fromisoformat(data["expires_at_utc"].replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return {"valid": False, "reason": "lease_expiry", "expiry": data["expires_at_utc"], "purpose": data["purpose"]}
        now = _utc_now() if now is None else now
        if not isinstance(now, datetime) or now.tzinfo is None or expiry.tzinfo is None or expiry <= now or expiry - now > timedelta(seconds=MAX_TTL_SECONDS):
            return {"valid": False, "reason": "lease_expiry", "expiry": data["expires_at_utc"], "purpose": data["purpose"]}
        if not isinstance(nonce, str) or not hmac.compare_digest(hashlib.sha256(nonce.encode("utf-8")).hexdigest(), data["nonce_sha256"]):
            return {"valid": False, "reason": "lease_nonce", "expiry": data["expires_at_utc"], "purpose": data["purpose"]}
        if resolve(data["state_root"]) != state_root or resolve(data["output_root"]) != output_root or data["workspace_id"] != workspace_id:
            return {"valid": False, "reason": "lease_identity", "expiry": data["expires_at_utc"], "purpose": data["purpose"]}
        actual_parent_pid = os.getppid() if parent_pid is None else parent_pid
        if not isinstance(actual_parent_pid, int) or isinstance(actual_parent_pid, bool) or actual_parent_pid != data["parent_pid"]:
            return {"valid": False, "reason": "lease_parent_pid", "expiry": data["expires_at_utc"], "purpose": data["purpose"]}
        script = resolve(data["parent_script"])
        benchmark_root = resolve(project_root / "state-cache" / "benchmark")
        if not script.is_file() or script.suffix.lower() != ".py" or not _inside(script, benchmark_root):
            return {"valid": False, "reason": "lease_parent_binding", "expiry": data["expires_at_utc"], "purpose": data["purpose"]}
        info = (inspector or _default_inspector)(int(data["parent_pid"]))
        command = info["command_line"]
        # Windows venv redirectors commonly expose the base interpreter in both
        # psutil.exe/cmdline; bind those two actual values to the lease rather
        # than falsely treating the redirector as a different parent.
        if not isinstance(command, list) or not command or resolve(info["executable"]) != resolve(data["parent_executable"]) or resolve(command[0]) != resolve(data["parent_executable"]):
            return {"valid": False, "reason": "lease_parent_process", "expiry": data["expires_at_utc"], "purpose": data["purpose"]}
        script_tokens = []
        parent_cwd = info.get("cwd")
        for token in command[1:]:
            candidate = Path(token)
            if not candidate.is_absolute() and parent_cwd:
                candidate = Path(parent_cwd) / candidate
            try:
                script_tokens.append(resolve(candidate))
            except Exception:
                continue
        if script not in script_tokens:
            return {"valid": False, "reason": "lease_parent_process", "expiry": data["expires_at_utc"], "purpose": data["purpose"]}
        return {"valid": True, "reason": "valid", "expiry": data["expires_at_utc"], "purpose": data["purpose"]}
    except Exception:
        return fail("lease_invalid")
