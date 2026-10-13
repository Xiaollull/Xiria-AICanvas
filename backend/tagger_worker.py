"""Bounded stdin/stdout worker, only launched by the tagger resource owner."""
import base64
import contextlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend import tagger_adapters, tagger_runtime_install
from backend.tagger_registry import validate_selection


def main():
    payload = json.loads(sys.stdin.read(29 * 1024**2))
    params = payload["params"]
    from backend import wd14
    validate_selection(params["engine"], params["device"], params["model"], wd14.REGISTRY)
    wd14.ROOT = Path(payload["root"])
    data = base64.b64decode(payload["data"], validate=True)
    if not wd14.installed(params["model"]):
        raise ValueError("模型 manifest 不完整或摘要不匹配")
    if params["device"] == "gpu":
        import torch
        if not torch.cuda.is_available():
            raise ValueError("CUDA GPU 不可用；不会 CPU fallback")
        if params["engine"] != "pixai":
            if str(torch.version.cuda).split(".")[0] != "13":
                raise ValueError("受管 ORT 1.28.0 固定 CUDA 13；当前 Torch CUDA major 不匹配")
            site = tagger_runtime_install.verify_layer()
            if "onnxruntime" in sys.modules:
                raise ValueError("worker 已加载其他 ORT distribution，拒绝混装")
            sys.path.insert(0, str(site))
            import onnxruntime as ort
            if not Path(ort.__file__).resolve().is_relative_to(site.resolve()):
                raise ValueError("实际 ORT 路径不属于验证后的固定依赖层")
            ort.preload_dlls()
    start = time.perf_counter()
    if params["engine"] == "wd":
        import onnxruntime as ort
        onnx, csv_file = wd14.model_files(params["model"])
        session = tagger_adapters.create_onnx(onnx, params["device"])
        rows = wd14._tags(csv_file)
        shape = session.get_inputs()[0].shape
        size = max(v for v in shape if isinstance(v, int) and v > 4)
        wd14._runtime = lambda model: (session, rows, size)
        result = wd14.tag_image(data, params["model"], params["general"], params["character"], params["limit"], params["alpha"])
        result.update(engine="wd", device=params["device"], params=params,
                      runtime=dict(name="ONNX Runtime", version=ort.__version__, provider=session.get_providers()[0], dtype="float32"))
    else:
        folder = wd14._safe_directory(wd14.ROOT / params["model"])
        result = tagger_adapters.infer(data, params, folder, params["device"])
    result["timing_seconds"] = round(time.perf_counter() - start, 4)
    return result


if __name__ == "__main__":
    # -I deliberately ignores PYTHONIOENCODING. Keep the IPC contract UTF-8
    # explicitly even on a cp936 host, including non-Latin labels and errors.
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        with contextlib.redirect_stdout(sys.stderr):
            result = main()
        print(json.dumps(result, ensure_ascii=False))
    except Exception as error:
        print(json.dumps({"error": str(error)[:800]}, ensure_ascii=False))
