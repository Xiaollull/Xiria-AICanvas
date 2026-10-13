"""Same-host admission lease shared by actual jobs, Gallery and Node.

Ephemeral gate is outside the identity library; imports initialize nothing.
Tokens delegate ownership, not HTTP/user authentication. Never steal a live PID.
"""
import json
import os
import socket
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


class LibraryGateError(ValueError):
    pass


def verify_node_lease(state_directory, token):
    root = Path(state_directory)
    lease = root / ".lora-coordination.lock" / "lease.json"
    value = json.loads(lease.read_text(encoding="utf-8"))
    if not token or value.get("owner") != "node" or value.get("token") != token:
        raise LibraryGateError("Invalid Node coordination lease")
    return value


@contextmanager
def library_admission(state_directory, *, delegated=None, check_publication=True, wait=2.0):
    root = Path(state_directory)
    token = None
    lock = root / ".lora-coordination.lock"
    if delegated:
        verify_node_lease(root, delegated)
    else:
        root.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + wait
        while True:
            try:
                lock.mkdir()
                break
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise LibraryGateError("LoRA library busy or recovering")
                time.sleep(0.02)
        token = str(uuid.uuid4())
        (lock / "lease.json").write_text(json.dumps({"token": token, "owner": "python", "pid": os.getpid(), "host": socket.gethostname().lower()}), encoding="utf-8")
    try:
        from .lora_relink_store import publication
        # Bypass is only for unrelated jobs or the private Gallery owner, which
        # reads publication separately. Damaged repair state must not stop a
        # no-LoRA generation or prevent the coordinator from inspecting it.
        try:
            state = publication(root)
        except (ValueError, OSError, TypeError) as error:
            if check_publication:
                raise LibraryGateError("LoRA publication cannot be read safely") from error
            state = None
        if check_publication and state is not None and state.get("phase") != "published":
            raise LibraryGateError("LoRA repair recovery barrier is closed")
        yield state["revision"] if state is not None else 0
    finally:
        if token:
            value = json.loads((lock / "lease.json").read_text(encoding="utf-8"))
            if value.get("token") == token:
                (lock / "lease.json").unlink()
                lock.rmdir()
