"""Bounded process-local DISPLAY hints, never model/code import authorization.

Cold/changed/expired entries fail closed as pending while a bounded single-flight
does the original byte verification. Runtime VRAM/busy data is NOT cached here.
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor


class DisplayCache:
    def __init__(self, ttl=30.0, max_entries=64, workers=2, clock=time.monotonic):
        self.ttl = ttl
        self.max_entries = max_entries
        self.clock = clock
        self.lock = threading.RLock()
        self.entries = {}
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tagger-display-check")

    def get(self, key, fingerprint, verify, force=False):
        with self.lock:
            entry = self.entries.get(key)
            if entry is None:
                if len(self.entries) >= self.max_entries:
                    idle = next((old for old, value in self.entries.items() if value["future"] is None), None)
                    if idle is None:
                        return dict(installed=False, verification="pending", pending=True, reason="校验队列繁忙，请稍后刷新")
                    self.entries.pop(idle)
                entry = dict(fingerprint=fingerprint, generation=0, result=None, checked=0.0, future=None)
                self.entries[key] = entry
            changed = entry["fingerprint"] != fingerprint
            expired = entry["result"] is not None and self.clock() - entry["checked"] >= self.ttl
            # Concurrent force refresh joins a current same-fingerprint verification,
            # rather than repeatedly invalidating the work it is waiting for.
            if changed or expired or force and entry["future"] is None:
                entry.update(fingerprint=fingerprint, generation=entry["generation"] + 1, result=None)
            if entry["result"] is not None:
                return {**entry["result"], "verification": "verified" if entry["result"]["installed"] else "unavailable", "pending": False}
            if entry["future"] is None:
                generation = entry["generation"]
                future = self.executor.submit(verify)
                entry["future"] = future
                def finished(done):
                    try:
                        result = done.result()
                    except Exception:
                        result = dict(installed=False, reason="制品校验失败，请检查文件或重新刷新")
                    with self.lock:
                        if self.entries.get(key) is entry and entry["future"] is done:
                            entry["future"] = None
                            if entry["generation"] == generation:
                                entry.update(result=dict(result), checked=self.clock())
                future.add_done_callback(finished)
            return dict(installed=False, verification="pending", pending=True, reason="本地文件完整性校验中；尚未标记已安装")


DISPLAY_CACHE = DisplayCache()
