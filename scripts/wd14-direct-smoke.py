"""Opt-in isolated-port API smoke of the *current* Python source (no Vite setup bypass).

The product's Vite configurator gate must remain active. Use this when setup is
not complete; it does not restart or contact an existing local service.
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def request(url, method="GET", body=None, origin=None):
    headers = {"Content-Type": "application/json"}
    if origin:
        headers["Origin"] = origin
    action = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                    headers=headers, method=method)
    try:
        with urllib.request.urlopen(action, timeout=10) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def main():
    parent = Path(os.environ["WD14_SMOKE_TEMP"])
    if not parent.is_dir():
        raise RuntimeError("WD14_SMOKE_TEMP must be an existing approved directory")
    with tempfile.TemporaryDirectory(prefix="wd14-direct-", dir=parent) as scratch:
        scratch = Path(scratch)
        (scratch / "outputs").mkdir()
        folder = scratch / "outside-outputs"
        folder.mkdir()
        (folder / "picture.png").write_bytes(b"not-an-image")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        origin = f"http://127.0.0.1:{port}"
        environment = {**os.environ, "INFERENCE_HOST": "127.0.0.1", "INFERENCE_PORT": str(port),
                       "XIRAI_STATE_DIR": str(scratch / "state"), "XIRAI_OUTPUT_DIR": str(scratch / "outputs"),
                       "XIRAI_CACHE_DIR": str(scratch / "cache")}
        process = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.inference_server:app",
                                    "--host", "127.0.0.1", "--port", str(port), "--no-access-log"],
                                   cwd=str(Path(__file__).resolve().parents[1]), env=environment,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            for _ in range(120):
                if process.poll() is not None:
                    raise RuntimeError(f"isolated inference exited {process.returncode}: {process.stderr.read()[-1000:]!r}")
                try:
                    status, health = request(origin + "/api/inference/health")
                    if status == 200 and health.get("status") == "ready":
                        break
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(.5)
            else:
                raise RuntimeError("isolated inference did not become ready")
            status, models = request(origin + "/api/inference/wd14/models")
            print("new-source models", status, len(models.get("models", [])))
            payload = {"folder": str(folder)}
            for label, route, allowed_origin, body, expected in [
                ("local outside-outputs batch", "/batch", origin, payload, 202),
                ("cross-origin batch", "/batch", "http://evil.example", payload, 403),
                ("cross-origin download", "/download/wd-swinv2-v3", "http://evil.example", None, 403),
                ("local invalid download", "/download/foreign", origin, None, 422),
            ]:
                code, response = request(origin + "/api/inference/wd14" + route, "POST", body, allowed_origin)
                print(label, code)
                if code != expected:
                    raise RuntimeError(f"{label}: expected {expected}, got {code}: {response}")
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    main()
