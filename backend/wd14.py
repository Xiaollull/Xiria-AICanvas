"""Isolated WD14 tagging service. No standalone server, credentials or GGUF runtime.

Both model files live under models/wd14/<registry-id>/ and are ignored at source.
Folder jobs require a loopback origin, with server-side local-path exclusions.
"""
import csv
import hashlib
import hmac
import ipaddress
import io
import json
import os
import re
import stat
import tempfile
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse, urlsplit
from urllib.parse import urljoin

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from PIL import Image, ImageOps, UnidentifiedImageError

try:
    from . import tagger_adapters, tagger_resources, tagger_runtime_install, tagger_store
    from .tagger_registry import ENGINES, MODELS, WD_MODELS, engine_for, validate_selection, resource_budget
except ImportError:
    from backend import tagger_adapters, tagger_resources, tagger_runtime_install, tagger_store
    from backend.tagger_registry import ENGINES, MODELS, WD_MODELS, engine_for, validate_selection, resource_budget

router = APIRouter(prefix="/api/inference/wd14")
ROOT = Path(__file__).resolve().parents[1] / "models" / "wd14"
REGISTRY = {
    "wd-eva02-large-tagger-v3": ("SmilingWolf/wd-eva02-large-tagger-v3", "main"),
    "wd-vit-large-tagger-v3": ("SmilingWolf/wd-vit-large-tagger-v3", "main"),
    "wd-vit-v3": ("SmilingWolf/wd-vit-tagger-v3", "main"),
    "wd-swinv2-v3": ("SmilingWolf/wd-swinv2-tagger-v3", "main"),
    "wd-convnext-v3": ("SmilingWolf/wd-convnext-tagger-v3", "main"),
    "wd14-convnextv2-v2": ("SmilingWolf/wd-v1-4-convnextv2-tagger-v2", "v2.0"),
    "wd14-moat-v2": ("SmilingWolf/wd-v1-4-moat-tagger-v2", "v2.0"),
    "wd14-vit-v2": ("SmilingWolf/wd-v1-4-vit-tagger-v2", "v2.0"),
    "wd14-convnext-v2": ("SmilingWolf/wd-v1-4-convnext-tagger-v2", "v2.0"),
    "wd14-swinv2-v2": ("SmilingWolf/wd-v1-4-swinv2-tagger-v2", "v2.0"),
    "wd14-convnextv2-v2-git": ("SmilingWolf/wd-v1-4-convnextv2-tagger-v2", "main"),
    "wd14-vit-v2-git": ("SmilingWolf/wd-v1-4-vit-tagger-v2", "main"),
    "wd14-convnext-v2-git": ("SmilingWolf/wd-v1-4-convnext-tagger-v2", "main"),
    "wd14-swinv2-v2-git": ("SmilingWolf/wd-v1-4-swinv2-tagger-v2", "main"),
    "wd14-vit": ("SmilingWolf/wd-v1-4-vit-tagger", "main"),
    "wd14-convnext": ("SmilingWolf/wd-v1-4-convnext-tagger", "main"),
}
DEFAULT = "wd-swinv2-v3"
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".avif"}
MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 32_000_000
MAX_MODEL_BYTES = 3 * 1024 ** 3
MAX_FILES = 500
_lock = threading.RLock()
_session_lock = threading.RLock()
_session = None
_session_id = None
_jobs = {}
OUTPUT_ROOT = None  # Set by the owning inference server at router registration.


def _loopback(address):
    try:
        value = ipaddress.ip_address(address)
        return value.is_loopback or bool(getattr(value, "ipv4_mapped", None) and value.ipv4_mapped.is_loopback)
    except ValueError:
        return False


def require_wd14_write(request: Request):
    """Direct Python callers must be local and same-Origin; Node supplies an
    unguessable per-process proof only after its own browser/loopback checks.
    An Origin header alone is not authentication; this guards browser CSRF and
    refuses a remote client if Python was accidentally exposed on a LAN port.
    """
    origin = request.headers.get("origin", "")
    try:
        parsed = urlsplit(origin)
        host = urlsplit("http://" + request.headers["host"])
        valid_origin = (parsed.scheme in ("http", "https") and parsed.hostname and parsed.port
                        and not parsed.username and not parsed.password and parsed.path in ("", "/")
                        and not parsed.query and not parsed.fragment)
    except (KeyError, ValueError):
        valid_origin = False
    if not valid_origin or not request.client or not _loopback(request.client.host):
        raise HTTPException(403, "WD14 写操作需要本机与有效同源页面")
    proof = os.environ.get("XIRAI_WD14_PROXY_PROOF", "")
    via_proxy = bool(proof and hmac.compare_digest(request.headers.get("x-xirai-wd14-proxy-proof", ""), proof))
    direct = (parsed.hostname in ("localhost", "127.0.0.1", "::1")
              and parsed.netloc.lower() == host.netloc.lower())
    if not (via_proxy or direct):
        raise HTTPException(403, "跨站来源不得修改 WD14 状态")
    if (request.url.path == "/api/inference/wd14/batch" or
        request.url.path == "/api/inference/wd14/models" or
        request.url.path == "/api/inference/wd14/runtime/install" or
        request.url.path.startswith("/api/inference/wd14/download/") or
        request.url.path.endswith("/cancel")) and not (via_proxy and parsed.hostname in ("localhost", "127.0.0.1", "::1") or direct):
        raise HTTPException(403, "仅本机允许批量、模型下载、取消任务和完整性强刷新")


def _regular(path):
    try:
        return stat.S_ISREG(path.lstat().st_mode) and path.stat().st_size > 0
    except OSError:
        return False


def _safe_directory(path, root=None):
    """Reject junctions/symlinks at *every* existing component, not only the leaf."""
    root = ROOT if root is None else root
    base = root.resolve()
    candidate = Path(path).absolute()
    if not candidate.is_relative_to(root.absolute()):
        raise ValueError("路径必须在允许的文件夹内")
    for part in [root.absolute(), *[root.absolute().joinpath(*candidate.relative_to(root.absolute()).parts[:n]) for n in range(1, len(candidate.relative_to(root.absolute()).parts) + 1)]]:
        if part.is_symlink() or (os.name == "nt" and part.exists() and bool(part.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)):
            raise ValueError("不允许符号链接或目录联接")
    if not candidate.resolve().is_relative_to(base):
        raise ValueError("路径越界")
    return candidate


def _local_path(value):
    """Local browser only. Reject roots, devices, protected directories and all
    existing reparse components. This is a preflight/recheck, NOT an OS handle
    capability against a hostile process racing directory replacement.
    """
    path = Path(value)
    if not path.is_absolute() or (os.name == "nt" and (str(path).startswith("\\\\") or not re.fullmatch(r"[A-Za-z]:\\", path.anchor))):
        raise ValueError("仅接受本机磁盘的绝对路径，不接受网络共享或设备")
    absolute = path.absolute()
    if absolute == Path(absolute.anchor):
        raise ValueError("不能将磁盘或文件系统根目录作为批量目录")
    sensitive = {".ssh", ".git", ".config", "appdata", "windows", "system32", "programdata", "program files",
                 "program files (x86)", ".venv", "node_modules", "state-cache", ".cache", "plugins"}
    # A dedicated subdirectory of the OS temp root is useful for safe local
    # scratch batches/tests; never admit the temp root itself or other AppData.
    temp_root = Path(tempfile.gettempdir()).resolve()
    scratch = absolute.is_relative_to(temp_root) and absolute != temp_root
    project = Path(__file__).resolve().parents[1]
    canonical = absolute.resolve()
    parts = [part.lower() for part in (*absolute.parts, *canonical.parts)]
    if any(part in sensitive for part in parts) and not (scratch and
        all(part not in sensitive - {"appdata"} for part in parts)):
        raise ValueError("拒绝系统或程序敏感目录")
    if canonical == Path(canonical.anchor) or canonical == Path.home().resolve():
        raise ValueError("不能直接使用磁盘根或用户主目录")
    for excluded in (project / "backend", project / "src", project / "scripts", project / "public", project / "docs",
                     project / "models", project / "logs", project / "dist"):
        if canonical.is_relative_to(excluded.resolve()):
            raise ValueError("拒绝在程序源码或模型目录中批量读写")
    if os.name == "posix" and any(canonical.is_relative_to(Path(segment)) for segment in
                                  ("/dev", "/proc", "/sys", "/etc", "/usr", "/bin", "/sbin", "/boot", "/run", "/var", "/lib", "/root")):
        raise ValueError("拒绝系统目录")
    current = Path(absolute.anchor)
    for segment in absolute.parts[1:]:
        current /= segment
        if current.is_symlink() or (os.name == "nt" and current.exists() and
                                    bool(current.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)):
            raise ValueError("不允许符号链接或目录联接")
    return absolute


def model_files(model):
    if model not in REGISTRY:
        raise ValueError("未知 WD14 模型")
    folder = _safe_directory(ROOT / model)
    return folder / "model.onnx", folder / "selected_tags.csv"


def _tags(csv_path):
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"name", "category"}.issubset(reader.fieldnames):
            raise ValueError("标签表缺少 name/category 列")
        rows = [(row["name"], row["category"]) for row in reader]
    if not 4 < len(rows) <= 200_000 or any(not name or len(name) > 256 for name, _ in rows):
        raise ValueError("标签表无效")
    return rows


def installed(model):
    if model in MODELS:
        return tagger_store.installed(ROOT, model, _safe_directory)
    if model in WD_MODELS:
        return tagger_store.installed(ROOT, model, _safe_directory, WD_MODELS[model])
    raise ValueError("未知反推模型")


def _providers():
    import onnxruntime as ort
    if "CPUExecutionProvider" not in ort.get_available_providers():
        raise ValueError("当前 ONNX Runtime 缺少 CPU 执行器")
    # Tagging must not race the single generation worker for GPU/VRAM resources.
    return ["CPUExecutionProvider"]


def _runtime(model):
    global _session, _session_id
    if not installed(model):
        raise ValueError("请先安装所选模型；ONNX 权重或标签缓存缺失")
    with _session_lock:
        if _session_id != model:
            import onnxruntime as ort
            _session = None
            _session_id = None
            onnx, csv_file = model_files(model)
            session = ort.InferenceSession(str(onnx), providers=_providers())
            rows = _tags(csv_file)
            shape = session.get_inputs()[0].shape
            size = max((v for v in shape if isinstance(v, int) and v > 4), default=448)
            if size > 1024 or len(session.get_outputs()) != 1 or len(rows) != session.get_outputs()[0].shape[-1]:
                raise ValueError("模型与标签表维度不匹配")
            _session = (session, rows, size)
            _session_id = model
        return _session


def tag_image(data, model, general, character, limit, alpha=False):
    import cv2
    import numpy as np
    if len(data) > MAX_BYTES:
        raise ValueError("图片超过 20 MiB")
    with _session_lock:  # Reuse the same ONNX session without competing with another tag job.
        session, rows, size = _runtime(model)
        try:
            with Image.open(io.BytesIO(data)) as source:
                if source.width * source.height > MAX_PIXELS:
                    raise ValueError("图片超过 3200 万像素")
                source = ImageOps.exif_transpose(source)
                rgba = source.convert("RGBA")
                canvas = Image.new("RGBA", rgba.size, "WHITE")
                canvas.paste(rgba, mask=rgba)
                rgb = np.asarray(canvas.convert("RGB"))[:, :, ::-1]
                h, w = rgb.shape[:2]
                side = max(h, w, size)
                rgb = cv2.copyMakeBorder(rgb, (side-h)//2, side-h-(side-h)//2,
                                         (side-w)//2, side-w-(side-w)//2, cv2.BORDER_CONSTANT, value=[255]*3)
                if side != size:
                    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
                scores = session.run([session.get_outputs()[0].name],
                                     {session.get_inputs()[0].name: np.expand_dims(rgb.astype(np.float32), 0)})[0][0]
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as error:
            raise ValueError("无法解码图片或尺寸不安全") from error
        if len(scores) != len(rows):
            raise ValueError("模型输出维度与标签表不一致")
        rating = {name: float(score) for (name, _), score in zip(rows[:4], scores[:4])}
        values = [(name, float(score), "character" if str(category) == "4" else "general") for (name, category), score in zip(rows[4:], scores[4:])
                  if float(score) >= (character if str(category) == "4" else general)]
        values.sort(key=lambda row: -row[1])
        values = values[:limit]
        if alpha:
            values.sort(key=lambda row: row[0])
        return {"rating": rating, "tags": [{"name": name.replace("_", " "), "confidence": score, "category": category}
                                              for name, score, category in values]}


class Options(BaseModel):
    engine: str = "wd"
    device: str = "cpu"
    model: str = DEFAULT
    general: float = Field(0.35, ge=0, le=1)
    character: float = Field(0.85, ge=0, le=1)
    limit: int = Field(50, ge=1, le=500)
    alpha: bool = False
    copyright: float | None = Field(None, ge=0, le=1)
    artist: float | None = Field(None, ge=0, le=1)
    style: float | None = Field(None, ge=0, le=1)
    meta: float | None = Field(None, ge=0, le=1)
    year: float | None = Field(None, ge=0, le=1)
    quality: float | None = Field(None, ge=0, le=1)
    tag_model: float | None = Field(None, ge=0, le=1)


class Batch(Options):
    folder: str
    output_folder: str = ""
    recursive: bool = False
    conflict: str = "ignore"


def _job_state(job):
    return {key: value for key, value in job.items()
            if not key.startswith("_") and key not in {"cancel", "thread", "publish_started"}}


def _identity(path):
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            return None
        return info.st_dev, info.st_ino
    except OSError:
        return None


def _record_stage_file(job, path, identity=None):
    identity = identity or _identity(path)
    if identity:
        job.setdefault("_wd14_stage_owned", {})[str(path)] = identity
    return identity


def _cleanup_stage(stage, identity, job):
    """Remove only unchanged files inside the exact private directory we made."""
    try:
        info = stage.lstat()
        if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != identity:
            raise OSError("WD14 暂存目录身份已变化")
        owned = job.get("_wd14_stage_owned", {})
        unsafe = False
        for path in stage.iterdir():
            expected = owned.get(str(path))
            if expected and _identity(path) == expected:
                path.unlink()
            else:
                unsafe = True
        if unsafe:
            raise OSError("暂存目录含有身份不匹配的文件")
        stage.rmdir()
    except OSError:
        job["warning"] = "WD14 私有暂存目录未能安全清理，请手工核查/修复模型目录"
        _log(job, job["warning"])


def _new_job(kind):
    with _lock:
        if any(j["state"] in ("queued", "running", "publishing") for j in _jobs.values()):
            raise HTTPException(409, "已有 WD14 任务在运行")
        if len(_jobs) >= 40:
            for key in list(_jobs):
                if _jobs[key]["state"] not in ("queued", "running"):
                    del _jobs[key]
                    break
        job = dict(id=uuid.uuid4().hex, kind=kind, state="queued", done=0, skipped=0,
                   failed=0, total=0, current="", logs=[], error="", cancel=threading.Event(), publish_started=False)
        _jobs[job["id"]] = job
        return job


def _log(job, line):
    job["logs"] = [*job["logs"][-79:], line[:240]]


def _run(job, operation):
    def worker():
        with _lock:
            job["state"] = "running"
        try:
            operation()
            with _lock:
                job["state"] = "cancelled" if not job["publish_started"] and job["cancel"].is_set() else "done"
        except Exception as error:
            with _lock:
                job["state"] = "cancelled" if not job["publish_started"] and job["cancel"].is_set() else "error"
                job["error"] = str(error)[:300] if job["state"] == "error" else ""
        finally:
            job["current"] = ""
    job["thread"] = threading.Thread(target=worker, daemon=True, name="wd14-"+job["kind"])
    job["thread"].start()
    return _job_state(job)


def _folder(value, root=None):
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("请输入本机图片文件夹的绝对路径")
    folder = _local_path(path)
    if not folder.is_dir():
        raise ValueError("图片目录不存在")
    return folder


def _images(folder, recursive, root=None):
    from os import scandir
    images, pending = [], [folder]
    visited = 0
    while pending:
        current = _local_path(pending.pop())
        if len(current.relative_to(folder).parts) > 16:
            raise ValueError("目录层级超过 16 层")
        with scandir(current) as iterator:
            for entry in iterator:
                visited += 1
                if visited > 5000:
                    raise ValueError("目录条目超过 5000 个")
                if entry.is_symlink() or (hasattr(entry, "is_junction") and entry.is_junction()):
                    continue
                if entry.is_dir(follow_symlinks=False) and recursive:
                    pending.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False) and Path(entry.name).suffix.lower() in IMAGE_EXT:
                    images.append(Path(entry.path))
                    if len(images) > MAX_FILES:
                        raise ValueError("一次最多处理 500 张图片")
    return sorted(images)


def _caption(path, text, conflict, root=None):
    path = _local_path(path)
    if path.is_symlink() or (path.exists() and not _regular(path)):
        raise ValueError("目标不是普通文本文件")
    if path.exists() and path.stat().st_size > 1024 * 1024:
        raise ValueError("已有文本超过 1 MiB")
    if conflict == "ignore" and path.exists():
        return False
    if path.exists() and conflict in ("append", "prepend"):
        old = path.read_text(encoding="utf-8")
        text = f"{old.strip()}, {text}" if conflict == "append" else f"{text}, {old.strip()}"
    # Never follow an existing link while publishing. Only own uniquely named temporary files.
    temporary = path.with_name(".wd14-" + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(text)
        _local_path(path)
        _local_path(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


@router.get("/models")
@router.post("/models")
def models(refresh: bool = False, request: Request = None):
    if request is not None and len(request.query_params.getlist("refresh")) > 1:
        raise HTTPException(422, "refresh 参数不得重复")
    if refresh or request is not None and request.method == "POST":
        if request is None:
            raise HTTPException(403, "强刷新需要本机同源请求")
        require_wd14_write(request)  # Before snapshot, cache scheduling or any validator.
    # The entire registry is fixed source data. Expensive display verification
    # runs as bounded single-flights; neither hints nor receipts authorize import.
    sample = tagger_resources.snapshot()
    layer = tagger_runtime_install.display_status(refresh=refresh)
    entries = []
    for name in [*REGISTRY, *MODELS]:
        engine = engine_for(name, REGISTRY)
        spec = MODELS.get(name)
        gpu = tagger_resources.capability(name, engine, resource_snapshot=sample, layer_status=layer)
        manifest = spec if spec else WD_MODELS[name]
        status = tagger_store.display_status(ROOT, name, _safe_directory, manifest, refresh=refresh)
        readiness = engine != "cl" or tagger_adapters.CL_PREPROCESSING_VERIFIED
        entries.append(dict(id=name, engine=engine, installed=status["installed"], verification=status["verification"],
            verification_pending=status["pending"], installation_reason=status.get("reason", ""), ready=readiness,
            readiness_reason="" if readiness else tagger_adapters.CL_PREPROCESSING_REASON,
            repo_url="https://huggingface.co/" + (spec["repo"] if spec else REGISTRY[name][0]),
            runtime=ENGINES[engine]["runtime"], devices={"cpu": {"available": readiness, "provider": "CPU" if engine == "pixai" else "CPUExecutionProvider"}, "gpu": gpu}))
    return {"models": entries, "engines": [{"id": name, **value} for name, value in ENGINES.items()],
            "providers": _providers(), "default": DEFAULT, "batch_root": str(OUTPUT_ROOT),
            "gpu_runtime": layer, "verification_pending": layer.get("pending", False) or any(row["verification_pending"] for row in entries)}


def _effective(options):
    validate_selection(options.engine, options.device, options.model, REGISTRY)
    values = options.model_dump()
    # Legacy defaults remain 0.35/0.85; omitted non-WD thresholds use engine defaults.
    general = values.pop("general") if "general" in options.model_fields_set or options.engine == "wd" else None
    character = values.pop("character") if "character" in options.model_fields_set or options.engine == "wd" else None
    return tagger_adapters.parameters(options.engine, options.model, options.device, general, character,
        options.limit, options.alpha, **{key: values[key] for key in ("copyright", "artist", "style", "meta", "year", "quality")},
        tag_model=options.tag_model)


def _dispatch(data, params):
    if not (params["engine"] == "wd" and params["device"] == "cpu") and not installed(params["model"]):
        raise ValueError("请先安装完整的所选模型；文件缺失或摘要验证失败")
    start = time.perf_counter()
    if params["device"] == "gpu":
        status = tagger_resources.capability(params["model"], params["engine"])
        if not status["available"]:
            raise ValueError(status["reason"])
        result = tagger_resources.run_gpu(lambda: tagger_resources.worker(data, params, ROOT))
    elif params["engine"] == "wd":
        result = tag_image(data, params["model"], params["general"], params["character"], params["limit"], params["alpha"])
        import onnxruntime as ort
        result.update(engine="wd", device="cpu", params=dict(params), runtime=dict(name="ONNX Runtime", version=ort.__version__, provider="CPUExecutionProvider", dtype="float32"))
    elif params["engine"] == "pixai":
        result = tagger_resources.worker(data, params, ROOT)
    else:
        result = tagger_adapters.infer(data, params, _safe_directory(ROOT / params["model"]), "cpu")
    result["timing_seconds"] = round(time.perf_counter() - start, 4)
    return result


@router.post("/tag", dependencies=[Depends(require_wd14_write)])
async def tag(request: Request, model: str = DEFAULT, general: float | None = None, character: float | None = None,
              limit: int = 50, alpha: bool = False, engine: str = "wd", device: str = "cpu",
              copyright: float | None = None, artist: float | None = None, style: float | None = None,
              meta: float | None = None, year: float | None = None, quality: float | None = None, tag_model: float | None = None):
    try:
        validate_selection(engine, device, model, REGISTRY)
        params = tagger_adapters.parameters(engine, model, device, general, character, limit, alpha,
            copyright=copyright, artist=artist, style=style, meta=meta, year=year, quality=quality)
        if engine == "cl" and tag_model is not None:
            if not 0 <= tag_model <= 1:
                raise ValueError("模型类别阈值须为 0..1")
            params["tag_model"] = tag_model
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "图片超过 20 MiB")
    try:
        from starlette.concurrency import run_in_threadpool
        return await run_in_threadpool(_dispatch, bytes(data), params)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@router.post("/batch", status_code=202, dependencies=[Depends(require_wd14_write)])
def batch(request: Batch):
    if request.conflict not in ("copy", "ignore", "append", "prepend"):
        raise HTTPException(422, "模型或冲突方式无效")
    try:
        params = _effective(request)
        if request.device == "gpu":
            status = tagger_resources.capability(request.model, request.engine)
            if not status["available"]:
                raise ValueError(status["reason"])
        folder = _folder(request.folder)
        destination = _folder(request.output_folder) if request.output_folder.strip() else None
        paths = _images(folder, request.recursive)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    if not paths:
        raise HTTPException(422, "目录中没有支持的图片")
    job = _new_job("batch")
    job.update(model=request.model, engine=request.engine, device=request.device, params=params)
    job["total"] = len(paths)
    def operation():
        if request.engine == "wd" and request.device == "cpu":
            _runtime(request.model)
        elif not installed(request.model):
            raise ValueError("请先安装完整所选模型")
        for source in paths:
            if job["cancel"].is_set():
                break
            job["current"] = source.name
            try:
                _local_path(source)
                if not _regular(source) or source.stat().st_size > MAX_BYTES:
                    raise ValueError("图片不是普通文件或超过 20 MiB")
                target = ((destination / source.relative_to(folder)) if destination else source).with_suffix(".txt")
                _local_path(target)
                if request.conflict == "ignore" and target.exists():
                    job["skipped"] += 1
                else:
                    with source.open("rb") as stream:
                        data = stream.read(MAX_BYTES + 1)
                    result = _dispatch(data, params)
                    if job["cancel"].is_set():
                        break
                    text = ", ".join(tag["name"] for tag in result["tags"])
                    if destination:
                        _local_path(target.parent).mkdir(parents=True, exist_ok=True)
                    if _caption(target, text, request.conflict):
                        job["done"] += 1
                    else:
                        job["skipped"] += 1
                _log(job, f"{source.name}: 完成")
            except (OSError, ValueError, UnicodeError) as error:
                job["failed"] += 1
                _log(job, f"{source.name}: {str(error)[:120]}")
    return _run(job, operation)


def _metadata(model):
    if model in MODELS or model in WD_MODELS:
        spec = MODELS[model] if model in MODELS else WD_MODELS[model]
        return spec["repo"], spec["revision"], {artifact.local: (artifact.size, artifact.digest) for artifact in spec["artifacts"]}
    raise ValueError("未知反推模型")


def _download_one(repo, commit, name, expected, target, job):
    spec = next((spec for spec in MODELS.values() if spec["repo"] == repo and spec["revision"] == commit), None)
    remote = next((artifact.remote for artifact in spec["artifacts"] if artifact.local == name), name) if spec else name
    url = f"https://huggingface.co/{repo}/resolve/{commit}/{remote}"
    return _download_url(url, expected, target, job)


def _download_url(url, expected, target, job):
    import requests
    size, digest = expected
    hasher = hashlib.sha256() if len(digest) == 64 else hashlib.sha1()
    if len(digest) == 40:
        hasher.update(f"blob {size}\0".encode())
    received = 0
    # Large official weights on a ~0.7 MiB/s link can exceed 30 minutes. Still
    # bounded, with per-read timeouts, byte/hash caps and cancellation checks.
    deadline = time.monotonic() + 7200
    for _ in range(6):
        parsed = urlparse(url)
        host = parsed.hostname or ""
        if parsed.scheme != "https" or parsed.port not in (None, 443) or parsed.username or parsed.password or not (host in ("huggingface.co", "files.pythonhosted.org") or host.endswith(".hf.co") or host.endswith(".xethub.hf.co")):
            raise ValueError("下载重定向到未经允许的站点")
        with requests.Session() as http:
            http.trust_env = False
            response_context = http.get(url, stream=True, timeout=(10, 30), allow_redirects=False)
        with response_context as response:
            if response.status_code in (301, 302, 303, 307, 308):
                url = urljoin(url, response.headers.get("Location", ""))
                continue
            response.raise_for_status()
            with target.open("xb") as stream:
                _record_stage_file(job, target, (os.fstat(stream.fileno()).st_dev,
                                                 os.fstat(stream.fileno()).st_ino))
                for chunk in response.iter_content(1024 * 1024):
                    if job["cancel"].is_set():
                        raise ValueError("已取消")
                    if time.monotonic() > deadline:
                        raise ValueError("制品下载总时限超限，可重试")
                    received += len(chunk)
                    if received > size:
                        raise ValueError("远程文件超过声明尺寸")
                    stream.write(chunk)
                    hasher.update(chunk)
                    job["done"] += len(chunk)
            if received != size or hasher.hexdigest().lower() != digest.lower():
                raise ValueError("下载文件尺寸或摘要校验失败")
            return
    raise ValueError("重定向次数过多")


class RuntimeInstall(BaseModel):
    model_config = {"extra": "forbid"}
    confirm: bool = False


@router.post("/runtime/install", status_code=202, dependencies=[Depends(require_wd14_write)])
def install_gpu_runtime(request: RuntimeInstall):
    if not request.confirm:
        raise HTTPException(422, "须明确确认安装固定 ORT-GPU 1.28.0 / CUDA 13 依赖层；主 .venv 不变")
    if not tagger_runtime_install.supported():
        raise HTTPException(422, "固定 GPU wheel 仅支持 Windows x64 / 项目 Python 3.12")
    job = _new_job("runtime")
    def publish(stage, folder):
        with _lock:
            if job["cancel"].is_set():
                return False
            job["publish_started"] = True
            job["state"] = "publishing"
        tagger_runtime_install.no_links(folder)
        if folder.exists():
            raise ValueError("GPU 依赖层目录已存在，拒绝覆盖；请手工核查")
        os.rename(stage, folder)
        return True
    def operation():
        job["_runtime_publish"] = publish
        tagger_runtime_install.install(job, _download_url, _log)
    return _run(job, operation)


@router.post("/download/{model}", status_code=202, dependencies=[Depends(require_wd14_write)])
def download(model: str):
    if model not in REGISTRY and model not in MODELS:
        raise HTTPException(422, "未知 WD14 模型")
    if installed(model):
        raise HTTPException(409, "模型已安装")
    job = _new_job("download")
    job.update(model=model, engine=engine_for(model, REGISTRY))
    def operation():
        try:
            repo, commit, files = _metadata(model)
        except ValueError:
            raise
        except Exception as error:
            raise ValueError("无法获取模型的固定版本与摘要，请检查网络") from error
        job["total"] = sum(size for size, _ in files.values())
        folder = _safe_directory(ROOT / model)
        folder.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".wd14-stage-", dir=folder))
        stage_stat = stage.lstat()
        stage_identity = (stage_stat.st_dev, stage_stat.st_ino)
        job["_wd14_stage_owned"] = {}
        temporary = {name: stage / (name + ".part")
                     for name in files}
        try:
            for name, target in temporary.items():
                if job["cancel"].is_set():
                    return
                try:
                    _download_one(repo, commit, name, files[name], target, job)
                    # Test doubles and alternate download adapters may not mark
                    # ownership from an open file descriptor; only adopt a file
                    # after a successful return when no descriptor identity exists.
                    if str(target) not in job["_wd14_stage_owned"]:
                        _record_stage_file(job, target)
                except ValueError:
                    raise
                except Exception as error:
                    raise ValueError("模型下载失败；可重试安装") from error
            if model in REGISTRY:
                _tags(temporary["selected_tags.csv"])
            if "model.onnx" in temporary:
                with temporary["model.onnx"].open("rb") as stream:
                    if stream.read(1) != b"\x08":
                        raise ValueError("ONNX 文件格式无效")
            targets = {name: folder / name for name in files}
            if model in REGISTRY:
                targets = {name: folder / name for name in ("selected_tags.csv", "model.onnx")}
            if _publish_artifacts(job, targets, temporary):
                _log(job, f"已校验并安装 {model} · {commit[:12]}")
        finally:
            if not job.get("_wd14_preserve_stage"):
                _cleanup_stage(stage, stage_identity, job)
    return _run(job, operation)


def _publish_model(job, onnx, csv_file, temporary):
    return _publish_artifacts(job, {"selected_tags.csv": csv_file, "model.onnx": onnx}, temporary)


def _publish_artifacts(job, old, temporary):
    """Cancellation linearizes *before* publishing. After that point cancel is
    refused, success is done, and failures restore only backups this job created.
    Any pre-existing marker or backup name fails closed for manual inspection.
    The marker prevents a partially published pair from reading as installed,
    including after a crash; it is not an OS multi-file transaction.
    """
    with _lock:
        if job["cancel"].is_set():
            return False
        job["publish_started"] = True
        job["state"] = "publishing"
    folder = next(iter(old.values())).parent
    _safe_directory(folder)
    marker = folder / ".wd14-installing"
    if marker.is_symlink() or (marker.exists() and not stat.S_ISREG(marker.lstat().st_mode)):
        raise ValueError("模型发布标记不是普通文件")
    if marker.exists():
        raise ValueError("发现未完成安装，须手工核查/修复模型目录后再重试")
    stage = next(iter(temporary.values())).parent
    backups = {name: stage / (name + ".old")
               for name in old}
    if any(backup.exists() or backup.is_symlink() for backup in backups.values()):
        raise ValueError("WD14 私有暂存备份名冲突，已停止安装")
    marker_created = False
    marker_identity = None
    try:
        stream = marker.open("x", encoding="ascii")
        marker_created = True
        marker_stat = os.fstat(stream.fileno())
        marker_identity = (marker_stat.st_dev, marker_stat.st_ino)
        with stream:
            stream.write(job["id"])
    except FileExistsError as error:
        raise ValueError("发现未完成安装，须手工核查/修复模型目录后再重试") from error
    except Exception:
        if marker_created:
            try:
                if _identity(marker) == marker_identity:
                    marker.unlink()
            except OSError:
                pass
        raise
    backups_created = {}
    published_identity = {}
    replaced = []
    rollback_ok = True
    published = False
    try:
        # Private same-volume stage holds the hardlinks; never use public job ids
        # as filesystem ownership. Verify each hardlink identity before tracking.
        for name, target in old.items():
            if target.is_symlink():
                raise ValueError("原模型路径是符号链接")
            if target.exists():
                if not stat.S_ISREG(target.lstat().st_mode):
                    raise ValueError("原模型路径不是普通文件")
                original_identity = _identity(target)
                os.link(target, backups[name])
                backup_identity = _identity(backups[name])
                if not original_identity or backup_identity != original_identity:
                    job["_wd14_preserve_stage"] = True
                    rollback_ok = False
                    raise ValueError("WD14 暂存备份身份校验失败；标记与暂存证据已保留，请手工核查/修复")
                backups_created[name] = backup_identity
                _record_stage_file(job, backups[name], backup_identity)
        for name in old:
            _safe_directory(folder)
            source_identity = job.get("_wd14_stage_owned", {}).get(str(temporary[name]))
            if not source_identity or _identity(temporary[name]) != source_identity:
                raise ValueError("WD14 下载暂存文件身份校验失败")
            os.replace(temporary[name], old[name])
            published_identity[name] = source_identity
            replaced.append(name)
        if _identity(marker) != marker_identity:
            job["_wd14_preserve_stage"] = True
            raise ValueError("WD14 发布标记身份已变化；保留安装证据，请手工核查/修复")
        marker.unlink()  # Installed becomes true only after both files are present.
        marker_created = False
        published = True
    except Exception:
        for name in reversed(replaced):
            try:
                if name in backups_created and _identity(backups[name]) == backups_created[name]:
                    if _identity(old[name]) != published_identity.get(name):
                        rollback_ok = False
                        job["_wd14_preserve_stage"] = True
                        continue
                    os.replace(backups[name], old[name])
                    backups_created.pop(name, None)
                    job.get("_wd14_stage_owned", {}).pop(str(backups[name]), None)
                elif name in backups_created:
                    rollback_ok = False
                    job["_wd14_preserve_stage"] = True
                else:
                    if _identity(old[name]) == published_identity.get(name):
                        old[name].unlink()
                    else:
                        rollback_ok = False
            except Exception:
                rollback_ok = False
        if rollback_ok and marker_created:
            try:
                if _identity(marker) != marker_identity:
                    raise OSError("安装标记身份已变化")
                marker.unlink()
                marker_created = False
            except Exception:
                rollback_ok = False
        # On incomplete rollback leave both marker and backups for recovery;
        # installed() must not claim an arbitrary mixed pair as ready.
        if not rollback_ok:
            job["_wd14_preserve_stage"] = True
            job["warning"] = "安装回滚未完成；marker 与私有暂存证据已保留，须手工核查/修复模型目录"
            _log(job, job["warning"])
        raise
    finally:
        if rollback_ok and published:
            for name, backup_identity in tuple(backups_created.items()):
                try:
                    if _identity(backups[name]) != backup_identity:
                        raise OSError("暂存备份身份已变化")
                    backups[name].unlink()
                    backups_created.pop(name, None)
                    job.get("_wd14_stage_owned", {}).pop(str(backups[name]), None)
                except Exception:
                    job["_wd14_preserve_stage"] = True
                    job["warning"] = "WD14 暂存备份身份异常，未删除替代文件；请手工核查/修复"
                    _log(job, job["warning"])
    if published:
        old_backups = [path.name for path in folder.iterdir()
                       if re.fullmatch(r"\.wd14-[0-9a-f]{32}-old-(?:model\.onnx|selected_tags\.csv)", path.name)]
        if old_backups:
            warning = ("检测到历史 WD14 备份候选（" + "、".join(old_backups[:4]) +
                       "），程序不会自动清理；请手工核查/修复模型目录")
            job["warning"] = warning
            _log(job, warning)
    return True


@router.get("/jobs/{job_id}")
def status(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "任务不存在")
        return _job_state(job)


@router.get("/active")
def active():
    with _lock:
        return {"jobs": [_job_state(job) for job in _jobs.values() if job["state"] in ("queued", "running", "publishing")]}


@router.post("/jobs/{job_id}/cancel", dependencies=[Depends(require_wd14_write)])
def cancel(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "任务不存在")
        if job["state"] == "publishing":
            raise HTTPException(409, "模型正在发布，不能取消；请等待安装结果")
        if job["state"] in ("done", "cancelled", "error"):
            return {"ok": False, "state": job["state"]}
        job["cancel"].set()
    return {"ok": True}
