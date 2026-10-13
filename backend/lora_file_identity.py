"""Real OS file identity, optional evidence only (never a permanent UUID)."""
import os
from pathlib import Path


def file_identity(file):
    file = Path(file)
    stat = file.stat()
    if os.name != "nt":
        return {"kind": "posix", "key": f"{stat.st_dev}:{stat.st_ino}", "links": stat.st_nlink,
                "creation": str(getattr(stat, "st_birthtime_ns", 0))}
    import ctypes
    from ctypes import wintypes

    class FileId(ctypes.Structure):
        _fields_ = [("identifier", ctypes.c_ubyte * 16)]

    class FileIdInfo(ctypes.Structure):
        _fields_ = [("volume", ctypes.c_ulonglong), ("file_id", FileId)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                  wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.GetFileInformationByHandleEx.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.GetFileInformationByHandleEx.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.CreateFileW(str(file.resolve()), 0, 7, None, 3, 0, None)
    if handle == ctypes.c_void_p(-1).value:
        return None
    try:
        info = FileIdInfo()
        if not kernel.GetFileInformationByHandleEx(handle, 18, ctypes.byref(info), ctypes.sizeof(info)):
            return None
        return {"kind": "windows-file-id", "key": f"{info.volume}:{bytes(info.file_id.identifier).hex()}",
                "links": stat.st_nlink, "creation": str(stat.st_birthtime_ns)}
    finally:
        kernel.CloseHandle(handle)
