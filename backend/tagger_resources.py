"""Tagger-owned admission and disposable worker lifecycle; never evicts generation."""
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

from .tagger_registry import resource_budget
from . import tagger_runtime_install as ort_layer

GPU_RUN = None  # Owning generation server injects its single-executor admission callback.
GPU_BUSY = None
_worker_lock = threading.Lock()


def snapshot():
    import torch
    reason = ""
    available = bool(torch.cuda.is_available())
    free = total = 0
    if available:
        free, total = torch.cuda.mem_get_info()
        if torch.version.cuda is None:
            available, reason = False, "当前 Torch 没有 CUDA runtime"
    else:
        reason = "当前没有可用 NVIDIA CUDA GPU"
    return dict(available=available, reason=reason, free=free, total=total, cuda_major=str(torch.version.cuda).split(".")[0],
                busy=bool(GPU_BUSY and GPU_BUSY()) or _worker_lock.locked())


def capability(model, engine, resource_snapshot=None, layer_status=None):
    # A catalog shares one fresh request snapshot; actual admission callers still
    # take their own fresh sample and uncached layer.status()/worker trust gate.
    sample = snapshot() if resource_snapshot is None else resource_snapshot
    budget = resource_budget(model)
    available, reason = sample["available"], sample["reason"]
    free, total = sample["free"], sample["total"]
    layer = (ort_layer.status() if layer_status is None else layer_status) if engine != "pixai" else None
    if engine != "pixai" and (not layer["installed"] or sample["cuda_major"] != "13"):
        available = False
        reason = layer["reason"] if not layer["installed"] else "受管 ORT 1.28.0 需要 CUDA 13，与当前 Torch CUDA major 不匹配"
    busy = sample["busy"]
    if busy:
        available, reason = False, "生图或模型 worker 正在运行，请稍后重试；不会卸载生图缓存"
    if available and free < budget["required_mib"] * 1024**2:
        available, reason = False, "当前可用显存低于模型单图预算；请使用 CPU 或自行释放其他任务，不会自动卸载生图模型"
    return dict(available=available, reason=reason, runtime="PyTorch" if engine == "pixai" else "ONNX Runtime",
                provider="CUDA" if engine == "pixai" else "CUDAExecutionProvider", free_vram_mib=free // 1024**2,
                total_vram_mib=total // 1024**2, budget=budget, ort_layer=layer, busy=busy)


def worker(data, params, root):
    # A separate process also contains CPU PixAI host memory; no resident GPU tensors.
    if not _worker_lock.acquire(blocking=False):
        raise ValueError("已有反推模型 worker 正在运行，请稍后重试")
    try:
        if params["device"] == "gpu":
            import torch
            free, _ = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
            if not torch.cuda.is_available() or free < resource_budget(params["model"])["required_mib"] * 1024**2:
                raise ValueError("GPU 不可用或可用显存不足；不会回退 CPU 或卸载生图缓存")
        env = dict(os.environ)
        for name in ["PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX", "PIP_CONFIG_FILE", "PIP_TARGET", "PIP_PREFIX", "PIP_USER",
                     "XIRAI_REMOTE_UPDATE_TOKEN", "XIRAI_UPDATE_TOKEN", "XIRAI_WD14_PROXY_PROOF", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"]:
            env.pop(name, None)
        env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
        command = [sys.executable, "-I", "-B", str(Path(__file__).with_name("tagger_worker.py"))]
        payload = dict(data=base64.b64encode(data).decode("ascii"), params=params, root=str(root))
        try:
            result = subprocess.run(command, input=json.dumps(payload), text=True, encoding="utf-8", errors="replace",
                                    capture_output=True, env=env, timeout=600, cwd=Path(__file__).resolve().parents[1])
        except subprocess.TimeoutExpired as error:
            raise ValueError("反推 worker 超时并已终止；GPU 资源由进程退出释放") from error
        if result.returncode or len(result.stdout) > 1024**2:
            raise ValueError("反推 worker 失败（可能显存不足或依赖不匹配）；未回退 CPU。" + result.stderr[-500:])
        body = json.loads(result.stdout)
        if body.get("error"):
            raise ValueError(body["error"])
        return body
    finally:
        _worker_lock.release()


def run_gpu(operation):
    if GPU_RUN is None:
        raise ValueError("GPU 资源准入未连接到生图服务；拒绝并发 GPU 反推")
    return GPU_RUN(operation)
