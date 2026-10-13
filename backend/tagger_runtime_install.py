"""Optional closed ORT-GPU dependency layer. Never runs pip or touches .venv.

Only a fixed official wheel is admitted. Import authorization is derived from
the pinned wheel bytes, NOT from editable receipt file hashes. .pth files are
never processed; each worker checks all extracted files before adding sys.path.
"""
import hashlib
import json
import os
import platform
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import time
import zipfile
from .tagger_catalog_cache import DISPLAY_CACHE

VERSION = "1.28.0"
WHEEL_SHA = "3befb47dabd2843bb606a50abd6f14b1eb1324eb235d5e5f3b01dd8b8034af1f"
WHEEL_SIZE = 241459431
WHEEL_NAME = "onnxruntime_gpu-1.28.0-cp312-cp312-win_amd64.whl"
WHEEL_URL = "https://files.pythonhosted.org/packages/e5/9e/92554acd080db68f549fd0e653fcf51a9dea7cb31e70c497714a9f2310fc/" + WHEEL_NAME
ROOT = Path(__file__).resolve().parents[1] / ".cache" / "tagger-runtimes"
LAYER_ID = "cp312-win_amd64-cu13-" + WHEEL_SHA[:16]


def no_links(path):
    path = Path(path).absolute()
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink() or (os.name == "nt" and current.exists() and current.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise ValueError("GPU 依赖层不允许符号链接/联接")
    return path


def supported():
    return sys.platform == "win32" and sys.version_info[:2] == (3, 12) and sys.maxsize > 2**32 and platform.machine().lower() in ("amd64", "x86_64")


def paths():
    folder = no_links(ROOT / LAYER_ID)
    return folder, folder / "site-packages", folder / WHEEL_NAME, folder / "receipt.json"


def _hash_file(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _entries(wheel):
    if not wheel.is_file() or wheel.stat().st_size != WHEEL_SIZE or _hash_file(wheel) != WHEEL_SHA:
        raise ValueError("GPU wheel 不完整或摘要不匹配；须手工核查依赖层")
    with zipfile.ZipFile(wheel) as archive:
        entries = []
        total = 0
        for info in archive.infolist():
            name = PurePosixPath(info.filename)
            if info.is_dir():
                continue
            if name.is_absolute() or ".." in name.parts or "\\" in info.filename or not name.parts:
                raise ValueError("GPU wheel 含有不安全路径")
            if stat.S_ISLNK(info.external_attr >> 16) or any(part.endswith(".data") for part in name.parts) or name.suffix == ".pth":
                raise ValueError("GPU wheel 含有不支持的安装入口")
            total += info.file_size
            if total > 2 * 1024**3:
                raise ValueError("GPU wheel 解包尺寸超限")
            entries.append(info)
        if len({info.filename for info in entries}) != len(entries):
            raise ValueError("GPU wheel 含有重复路径")
        return entries


def verify_layer():
    if not supported():
        raise ValueError("受管 ORT-GPU 入口当前支持 Windows x64 / Python 3.12 / CUDA 13；其他平台须另行受管适配")
    folder, site, wheel, receipt = paths()
    no_links(wheel)
    no_links(receipt)
    if not receipt.is_file() or receipt.stat().st_size > 4096:
        raise ValueError("未安装受管 ONNX GPU 依赖，请在本机确认安装 GPU runtime")
    body = json.loads(receipt.read_text(encoding="utf-8"))
    required = dict(schema=1, version=VERSION, python_abi="cp312", platform="win_amd64", cuda_major=13,
                    wheel_sha256=WHEEL_SHA, python=str(Path(sys.executable).resolve()))
    if any(body.get(key) != value for key, value in required.items()):
        raise ValueError("GPU runtime 安装收据不匹配当前项目 Python/固定 wheel")
    expected = _entries(wheel)
    with zipfile.ZipFile(wheel) as archive:
        names = set()
        for info in expected:
            name = info.filename
            target = no_links(site / name)
            data = target.lstat()
            if not stat.S_ISREG(data.st_mode) or data.st_size != info.file_size:
                raise ValueError("GPU runtime 文件不完整")
            with archive.open(info) as stream:
                expected_hash = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    expected_hash.update(chunk)
            if _hash_file(target) != expected_hash.hexdigest():
                raise ValueError("GPU runtime 文件已被更改；拒绝加载")
            names.add(name)
        actual = set()
        for parent, dirs, files in os.walk(no_links(site), followlinks=False):
            for entry in dirs:
                no_links(Path(parent) / entry)
            for entry in files:
                path = no_links(Path(parent) / entry)
                actual.add(path.relative_to(site).as_posix())
        if actual != names:
            raise ValueError("GPU runtime 含有未知文件；拒绝加载用户 Python")
    return site


def status():
    try:
        site = verify_layer()
        return dict(installed=True, version=VERSION, cuda_major=13, site=str(site), reason="")
    except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        return dict(installed=False, version=VERSION, cuda_major=13, reason=str(error), install_supported=supported())


def _display_fingerprint():
    """Cheap bounded tree/stat + small receipt bytes, NOT code-byte authorization."""
    folder, site, wheel, receipt = paths()
    no_links(wheel)
    no_links(receipt)
    no_links(site)
    entries = []
    def record(path):
        value = path.lstat()
        attrs = getattr(value, "st_file_attributes", 0)
        if stat.S_ISLNK(value.st_mode) or attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
            raise ValueError("GPU 依赖层不允许符号链接/联接")
        if not (stat.S_ISREG(value.st_mode) or stat.S_ISDIR(value.st_mode)):
            raise ValueError("GPU 依赖层包含非普通文件")
        entries.append((str(path), value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_mode, attrs))
        if len(entries) > 4096:
            raise ValueError("GPU 依赖层文件数量超限")
        return value
    record(folder);record(site)
    if record(wheel).st_size != WHEEL_SIZE:
        raise ValueError("GPU wheel 不完整或尺寸不匹配")
    if record(receipt).st_size > 4096:
        raise ValueError("GPU runtime 安装收据尺寸超限")
    for parent, dirs, files in os.walk(site, followlinks=False):
        for name in sorted(dirs + files):
            record(Path(parent) / name)
    receipt_hash = hashlib.sha256(receipt.read_bytes()).hexdigest()
    return tuple(sorted(entries)), receipt_hash


def display_status(refresh=False):
    """Display-only cached hint; verify_layer() ALWAYS remains uncached."""
    base = dict(version=VERSION, cuda_major=13, install_supported=supported(), display_only=True)
    if not supported():
        return {**base, "installed": False, "verification": "unavailable", "pending": False, "reason": "固定 GPU wheel 不支持当前平台/ABI"}
    try:
        fingerprint = _display_fingerprint()
        folder, site, _, _ = paths()
        key = ("runtime-display", str(folder), str(Path(sys.executable).resolve()), VERSION, WHEEL_SHA, WHEEL_SIZE)
        def verify():
            result = status()  # Original whole-wheel AND every extracted file verification.
            if _display_fingerprint() != fingerprint:
                return dict(installed=False, reason="GPU 依赖层在校验期间发生变化，请刷新")
            return result
        return {**base, **DISPLAY_CACHE.get(key, fingerprint, verify, force=refresh)}
    except (OSError, ValueError, zipfile.BadZipFile):
        return {**base, "installed": False, "verification": "unavailable", "pending": False, "reason": "GPU 依赖层缺失、不完整或路径不安全；须本机核查"}


def install(job, download, log):
    if not supported():
        raise ValueError("此固定 GPU wheel 不支持当前平台或 Python ABI")
    folder, site, wheel, receipt = paths()
    if folder.exists():
        verify_layer()  # Never delete/replace an unknown or damaged dependency directory.
        return
    no_links(ROOT).mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".tagger-runtime-stage-", dir=ROOT))
    identity = (stage.stat().st_dev, stage.stat().st_ino)
    try:
        job["total"] = WHEEL_SIZE
        download(WHEEL_URL, (WHEEL_SIZE, WHEEL_SHA), stage / WHEEL_NAME, job)
        if job["cancel"].is_set():
            return
        entries = _entries(stage / WHEEL_NAME)
        extracted = stage / "site-packages"
        extracted.mkdir()
        with zipfile.ZipFile(stage / WHEEL_NAME) as archive:
            for info in entries:
                if job["cancel"].is_set():
                    return
                target = extracted / info.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream, archive.open(info) as source:
                    shutil.copyfileobj(source, stream, 1024 * 1024)
        body = dict(schema=1, version=VERSION, python_abi="cp312", platform="win_amd64", cuda_major=13,
                    wheel_sha256=WHEEL_SHA, python=str(Path(sys.executable).resolve()), installed_at=time.time())
        (stage / "receipt.json").write_text(json.dumps(body), encoding="utf-8")
        # Caller owns the same publication/cancel state lock used by model downloads.
        publish = job["_runtime_publish"]
        tagger_stat = no_links(stage).lstat()
        if not stat.S_ISDIR(tagger_stat.st_mode) or (tagger_stat.st_dev, tagger_stat.st_ino) != identity:
            raise ValueError("GPU runtime 私有暂存目录身份变化，拒绝发布")
        if not publish(stage, folder):
            return
        log(job, "受管 ORT-GPU 1.28.0 / CUDA 13 已安装；主 .venv 未更改")
    finally:
        # Cleanup only our private stage while it still has its original identity.
        # An unexpected identity/entry is retained for manual inspection, never adopted.
        if stage.exists() and (stage.stat().st_dev, stage.stat().st_ino) == identity:
            job["warning"] = "GPU runtime 暂存保留以供手工核查：未发布或安装中断；未修改现有依赖层"
        job.pop("_runtime_publish", None)
