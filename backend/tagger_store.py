"""Complete fixed-manifest verification for non-legacy taggers, no network on catalog."""
import hashlib
import stat
import threading
from pathlib import Path

from .tagger_registry import MODELS
from .tagger_catalog_cache import DISPLAY_CACHE

_verified = {}
_lock = threading.Lock()


def digest_file(path, size, digest):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size != size:
        return False
    hasher = hashlib.sha256() if len(digest) == 64 else hashlib.sha1()
    if len(digest) == 40:
        hasher.update(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest() == digest


def signature(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("制品不是普通文件")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def installed(root, model, safe_directory, spec=None, force=False):
    try:
        spec = spec if spec is not None else MODELS[model]
        folder = safe_directory(Path(root) / model)
        marker = folder / ".wd14-installing"
        if marker.exists() or marker.is_symlink():
            return False
        paths = [folder / artifact.local for artifact in spec["artifacts"]]
        key = (str(folder), spec["revision"], tuple((artifact.local, artifact.size, artifact.digest) for artifact in spec["artifacts"]))
        stamps = tuple(signature(path) for path in paths)
        with _lock:
            if not force and _verified.get(key) == stamps:
                return True
        if not all(digest_file(path, artifact.size, artifact.digest) for path, artifact in zip(paths, spec["artifacts"])):
            with _lock:
                _verified.pop(key, None)
            return False
        # Refuse a set swapped while hashing; Windows APIs are not a hostile-user sandbox.
        if tuple(signature(path) for path in paths) != stamps:
            return False
        with _lock:
            _verified[key] = stamps
        return True
    except (KeyError, OSError, ValueError):
        return False


def display_status(root, model, safe_directory, spec, refresh=False):
    """Nonblocking catalog status; inference keeps using installed() separately."""
    try:
        folder = safe_directory(Path(root) / model)
        marker = folder / ".wd14-installing"
        if marker.exists() or marker.is_symlink():
            return dict(installed=False, verification="unavailable", pending=False, reason="存在未完成安装 marker，须手工核查")
        paths = [folder / artifact.local for artifact in spec["artifacts"]]
        stamps = tuple(signature(path) for path in paths)
        if any(path.stat().st_size != artifact.size for path, artifact in zip(paths, spec["artifacts"])):
            return dict(installed=False, verification="unavailable", pending=False, reason="本地模型文件尺寸不匹配")
        manifest = tuple((artifact.local, artifact.size, artifact.digest) for artifact in spec["artifacts"])
        key = ("model-display", str(folder), spec["revision"], manifest)
        def verify():
            ok = installed(root, model, safe_directory, spec, force=True)
            if tuple(signature(path) for path in paths) != stamps or marker.exists() or marker.is_symlink():
                return dict(installed=False, reason="模型文件在校验期间发生变化，请刷新")
            return dict(installed=ok, reason="" if ok else "模型完整 manifest 摘要不匹配")
        return DISPLAY_CACHE.get(key, stamps, verify, force=refresh)
    except (OSError, ValueError, KeyError):
        return dict(installed=False, verification="unavailable", pending=False, reason="模型文件缺失或路径不安全")
