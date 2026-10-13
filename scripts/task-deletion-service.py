"""PRIVATE CPU-only HTTP fixture. Requires a pre-created owned evidence directory; no app lifespan/GPU."""
import argparse
import copy
import json
import os
import tempfile
import uuid
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True)
parser.add_argument("--port", type=int, required=True)
args = parser.parse_args()
root = Path(args.root).resolve()
approved = Path(os.environ["LOCALAPPDATA"]) / "Temp" / "opencode"
if approved.resolve() not in root.parents or not (root / "OWNED-FIXTURE.json").is_file():
    raise RuntimeError("Unowned fixture root refused")
tempfile.tempdir = str(root)
os.environ["XIRAI_STATE_DIR"] = str(root / "import-state")
os.environ["XIRAI_OUTPUT_DIR"] = str(root / "import-outputs")
os.environ["XIRAI_WD14_PROXY_PROOF"] = "task-deletion-private-proof"

from fastapi import FastAPI, Request
from backend import inference_server as server
from backend.tests import test_lora_mix_service as mix_fixture
from backend.tests.test_xyz_experiments import plan_for
from backend.tests.test_xyz_planner import baseline
from backend.xyz_experiments import ExperimentStore

info_path = root / "fixture-info.json"
if info_path.exists():
    info = json.loads(info_path.read_text(encoding="utf-8"))
    server.STATE_DIRECTORY, server.OUTPUT_DIRECTORY = Path(info["state"]), Path(info["outputs"])
    server._xyz_store = None
else:
    harness = mix_fixture.MixServiceTests(methodName="runTest")
    harness.server = server
    harness.setUp()  # Real tiny registry; executor.submit is mocked, never run generation.
    store = server.xyz_store()
    plain = []
    for engine in ("SD", "iL", "Anima", "Flux", "Flux2", "Krea2"):
        for n in range(10):
            draft = baseline(engine)
            key, _ = store.create(submission_id=str(uuid.uuid4()), draft=draft, on_error="pause", plan=plan_for(draft), library_revision=0, requested_concurrency=1)
            store.record(key)["state"]["status"] = "complete"
            store.save(key)
            plain.append({"id": key, "engine": engine})
    active_draft = baseline("SD")
    active, _ = store.create(submission_id=str(uuid.uuid4()), draft=active_draft, on_error="pause", plan=plan_for(active_draft), library_revision=0, requested_concurrency=1)
    store.record(active)["state"]["status"] = "paused"; store.save(active)
    sessions = []
    for n in range(3):
        draft = harness.draft()
        draft["baseline"]["parameters"]["prompt"] = f"private synthetic search {n}"
        draft["baseline"]["prompt_without_groups"] = f"private synthetic search {n}"
        response = server.confirm_mix_round(harness.submission(draft), harness.http())
        sid, first = response["session"]["id"], response["experiment"]["id"]
        state = store.record(first)["state"]; state["status"] = "paused" if n == 2 else "interrupted"; store.save(first)
        if n != 2:
            server.job_controls.pop(first, None); server._xyz_executor_refs.discard(first); server.jobs[first]["status"] = "interrupted"
        definition = store.record(first)["definition"]
        origin = copy.deepcopy(definition["origin"]); origin["round"] = 2
        second, _ = store.create(submission_id=str(uuid.uuid4()), draft=definition["draft"], on_error="pause", plan=definition["plan"], library_revision=1, requested_concurrency=1, origin=origin)
        store.record(second)["state"]["status"] = "interrupted"; store.save(second)
        sessions.append({"id": sid, "rounds": [first, second], "active": n == 2})
    (harness.outputs / "preserved-original.png").write_bytes(b"private original sentinel")
    (harness.outputs / "preserved-grid.png").write_bytes(b"private export sentinel")
    (harness.state / "preserved-gallery.json").write_bytes(b"private gallery sentinel")
    info = {"state": str(harness.state), "outputs": str(harness.outputs), "plain": plain, "active_xyz": active, "sessions": sessions}
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

app = FastAPI()
app.router.routes.extend(route for route in server.app.routes if getattr(route, "path", "").startswith(("/api/inference/xyz-plots", "/api/inference/lora-mix", "/api/inference/history/assets")))

@app.get("/fixture-info")
def fixture_info(): return info

@app.post("/fixture-failure")
async def fixture_failure(request: Request):
    if request.client.host != "127.0.0.1": raise RuntimeError("private fixture only")
    value = await request.json(); failed = value["id"]
    store = server.xyz_store(); original = store.tombstone
    def once(module, task_id, ids):
        if task_id == failed:
            store.tombstone = original
            raise OSError("private once-only publication refusal")
        return original(module, task_id, ids)
    store.tombstone = once
    return {"ok": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
