"""QPlan2/P1 Phase B experiment owner: durable definitions, per-cell state and the
sequential orchestration loop.

Nothing here imports torch, FastAPI or the inference globals. The inference service
supplies admission, generation and job-record hooks, so every state transition is
testable without a GPU. Files live under one root (normally
`STATE_DIRECTORY/xyz-plots/<experiment_id>/`): `definition.json` is written once and
never changed; `state.json` is replaced atomically after every transition.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import socket
import threading
import time
import uuid
from pathlib import Path

SCHEMA_VERSION = 1
# Phase C: compatible cells (identical frozen request except the base seed) share one real tensor
# batch of at most this many images — the ordinary `images_per_batch` ceiling. A larger request is
# refused rather than silently reduced.
MAX_CONCURRENCY = 10
BATCH_LOG_LIMIT = 64
EXPORT_LIMIT = 20
EXPORT_ACTIVE = frozenset({"queued", "running"})
MODEL_KEYS = ("checkpoint", "diffusion_model", "text_encoder", "text_encoder_2", "vae")
FIELD_LABELS = {
    "checkpoint": "模型", "diffusion_model": "模型", "text_encoder": "模型", "text_encoder_2": "模型", "vae": "VAE",
    "loras": "LoRA 列表 / 权重", "prompt": "正向提示词", "negative_prompt": "负向提示词", "steps": "步数", "cfg": "CFG",
    "sampler": "采样器", "scheduler": "调度器", "width": "画布", "height": "画布", "guidance": "引导", "pag": "引导",
    "clip_skip": "CLIP skip", "postprocess_chain": "后处理链（含冻结重采样种子）", "denoise": "重绘幅度",
    "output_template": "输出命名", "background_removal_model": "透明背景模型",
}
EXPERIMENT_ACTIVE = frozenset({"queued", "running", "pausing", "paused", "cancelling"})
EXPERIMENT_STATUSES = EXPERIMENT_ACTIVE | {"complete", "cancelled", "interrupted", "error"}
CONTINUABLE = frozenset({"interrupted", "cancelled", "complete", "error"})
CELL_STATUSES = frozenset({"pending", "running", "complete", "error", "skipped", "cancelled", "missing"})
# A completed original is immutable: only these may be requeued, and a requeued
# cell always publishes a NEW file with its frozen request and seed.
RETRYABLE_CELLS = frozenset({"error", "skipped", "cancelled", "missing"})
LIST_LIMIT = 50
HISTORY_LIMIT = 20
ID_PATTERN = re.compile(r"[0-9a-f]{32}")
# Unique per process start: an `owner` written by an earlier run of the service is
# never mistaken for this one, even when the operating system reuses its PID.
PROCESS_TOKEN = uuid.uuid4().hex


class ExperimentError(ValueError):
    def __init__(self, code, message, status=409):
        super().__init__(message)
        self.code = code
        self.status = status


class CellRefused(Exception):
    """Execution-time admission refused one cell (identity, model, resource)."""


class ExperimentStopped(Exception):
    """Raised by the checkpoint hook when the parent is cancelled.

    `interrupted` marks a service shutdown: unfinished cells stay pending so the
    user can explicitly continue, instead of being recorded as user-cancelled.
    """

    def __init__(self, interrupted=False):
        super().__init__("interrupted" if interrupted else "stopped")
        self.interrupted = interrupted


def now():
    return time.time()


def process_owner():
    return {"pid": os.getpid(), "host": socket.gethostname().lower(), "token": PROCESS_TOKEN}


def write_json_atomic(path, value):
    """Publish a whole JSON document or nothing: temp file, fsync, replace."""
    path = Path(path)
    data = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(40):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                # Windows refuses to replace a file another thread is reading.
                if attempt == 39:
                    raise
                time.sleep(0.025)
    finally:
        temporary.unlink(missing_ok=True)


def submission_digest(draft, on_error):
    canonical = json.dumps({"draft": draft, "on_error": on_error}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def experiment_id_for(submission_id):
    """The submission UUID, as 32 hex digits, IS the experiment id: one click, one experiment."""
    if not isinstance(submission_id, str) or len(submission_id) != 36:
        raise ExperimentError("submission-id", "submission_id must be a canonical UUID string", 422)
    try:
        value = uuid.UUID(submission_id)
    except ValueError as error:
        raise ExperimentError("submission-id", "submission_id must be a canonical UUID string", 422) from error
    if str(value) != submission_id.lower():
        raise ExperimentError("submission-id", "submission_id must be a canonical UUID string", 422)
    return value.hex


def origin_summary(definition):
    """List-sized provenance. An origin this build does not know stays visible but is never interpreted."""
    origin = definition.get("origin")
    if not isinstance(origin, dict):
        return None
    if origin.get("kind") != "lora_mix_search" or origin.get("schema_version") != 1:
        return {"kind": str(origin.get("kind"))[:40], "known": False}
    return {"kind": "lora_mix_search", "known": True, "session_id": origin.get("session_id"), "round": origin.get("round"),
            "context_hash": origin.get("context_hash"), "candidates": len(origin.get("candidates") or [])}


def counts_of(state):
    counts = {status: 0 for status in sorted(CELL_STATUSES)}
    for cell in state["cells"].values():
        counts[cell["status"]] += 1
    return counts


def axis_summary(draft, plan):
    """Labels per axis index, read from the frozen plan (ranges are already expanded)."""
    labels = {"x": {}, "y": {}, "z": {}}
    for cell in plan["cells"]:
        coordinate = cell["coordinate"]
        for key in labels:
            labels[key].setdefault(coordinate[f"{key}_index"], cell["labels"].get(key, ""))
    axes = draft.get("axes") or {}
    return {key: {"field": (axes.get(key) or {}).get("field"), "labels": [labels[key][index] for index in sorted(labels[key])]}
            for key in labels}


def baseline_summary(draft):
    parameters = draft["baseline"]["parameters"]
    keys = ["engine", "checkpoint", "diffusion_model", "text_encoder", "text_encoder_2", "vae", "width", "height",
            "steps", "cfg", "sampler", "scheduler", "guidance"]
    summary = {key: parameters[key] for key in keys if parameters.get(key) is not None}
    summary["seed_mode"] = draft["baseline"].get("seed_mode", "fixed")
    summary["loras"] = len(draft["baseline"].get("loras") or [])
    return summary


PROMPT_BLOCK_SEPARATOR = "\n\n"


def cell_prompt_parts(draft, planned):
    """One cell's frozen positive prompt split into the LoRA-combination words that lead it and the
    rest (Phase E). Applying a cell to Basic writes only the rest: Basic composes the words of its own
    enabled combinations at generation time, so writing them into the prompt box would add them twice.

    The leading words are the baseline's enabled groups or — for a cell whose LoRA profile opted into
    its preset text — that profile's text, exactly as the planner composed them. A prompt that does not
    start with them (a positive-prompt axis replaced it, a replacement edited them) is returned whole
    with `removed: False`; nothing is guessed."""
    request = planned["request"]
    prompt = request.get("prompt") or ""
    lead, source = (draft.get("baseline") or {}).get("group_prompt") or "", "groups"
    for key in ("x", "y", "z"):
        axis = (draft.get("axes") or {}).get(key)
        if not axis or axis.get("field") not in {"lora_profile", "lora_multiplier"}:
            continue
        snapshot = axis.get("snapshot")
        if axis["field"] == "lora_profile":
            values, index = axis.get("values") or [], planned["coordinate"][f"{key}_index"]
            snapshot = values[index].get("value") if 0 <= index < len(values) and isinstance(values[index], dict) else None
        if isinstance(snapshot, dict) and snapshot.get("includePresetText") is True and isinstance(snapshot.get("preset_prompt", ""), str):
            lead, source = snapshot.get("preset_prompt", ""), "profile"
    rest, found = prompt, ""
    # The browser trims every block it joins; the planner joins a profile's text as given.
    for candidate in dict.fromkeys(text for text in (lead, lead.strip()) if text):
        if prompt == candidate or prompt.startswith(candidate + PROMPT_BLOCK_SEPARATOR):
            rest, found = prompt[len(candidate):].removeprefix(PROMPT_BLOCK_SEPARATOR), candidate
            break
    return {"prompt": prompt, "negative_prompt": request.get("negative_prompt") or "", "rest": rest,
            "removed": bool(found), "lead": found, "lead_source": source if found else None,
            "expected_lead": lead.strip(), "expected_source": source if lead.strip() else None}


def _validate_loaded(experiment_id, definition, state):
    if definition.get("schema_version") != SCHEMA_VERSION or definition.get("kind") != "xyz_experiment" or definition.get("experiment_id") != experiment_id:
        raise ValueError("definition")
    if state.get("schema_version") != SCHEMA_VERSION or state.get("kind") != "xyz_state" or state.get("experiment_id") != experiment_id:
        raise ValueError("state")
    order = definition["order"]
    if not isinstance(order, list) or set(order) != set(state["cells"]) or len(order) != len(set(order)):
        raise ValueError("cells")
    if {cell["cell_id"] for cell in definition["plan"]["cells"]} != set(order):
        raise ValueError("plan")
    if state["status"] not in EXPERIMENT_STATUSES or any(cell.get("status") not in CELL_STATUSES for cell in state["cells"].values()):
        raise ValueError("status")


class ExperimentStore:
    """Owns `<root>/<id>/{definition,state}.json`. Every mutation runs under `lock`.

    Lock order with the inference service: this lock may be held while taking its
    `jobs_lock`, never the reverse.
    """

    def __init__(self, root, *, owner=None, owner_alive=None, asset_exists=None, verify_asset=None):
        self.root = Path(root)
        self.owner = owner or process_owner()
        self.owner_alive = owner_alive or (lambda _owner: False)
        self.asset_exists = asset_exists or (lambda _relative: True)
        self.verify_asset = verify_asset
        self.lock = threading.RLock()
        self.records = {}
        self.deleted_records = {}
        self.deletions = {"schema_version": 1, "kind": "xyz_task_deletions", "entries": []}
        self.deleted_sessions = set()
        self.receipt_signature = None
        self.unreadable = {}
        self.loaded = False

    # Loading and restart recovery -------------------------------------------------------------

    def ensure_loaded(self):
        with self.lock:
            if self.loaded:
                if self._receipt_signature() != self.receipt_signature:
                    raise ExperimentError("deletion-recovery", "删除凭据已被另一写入者修改；停止任务访问，请重启隔离服务后核对恢复", 503)
                return
            self.safe_path(self.root)
            ledger = self.root / "task-deletions.json"
            deleted = set()
            if ledger.exists():
                self.safe_path(ledger)
                try:
                    value = json.loads(ledger.read_text(encoding="utf-8"))
                    if value.get("schema_version") != 1 or value.get("kind") != "xyz_task_deletions" or not isinstance(value.get("entries"), list):
                        raise ValueError("ledger")
                    for entry in value["entries"]:
                        if entry["module"] not in {"xyz", "lora_mix"} or not ID_PATTERN.fullmatch(entry["id"]):
                            raise ValueError("entry")
                        if not entry["experiment_ids"] or any(not ID_PATTERN.fullmatch(key) for key in entry["experiment_ids"]):
                            raise ValueError("ids")
                        deleted.update(entry["experiment_ids"])
                        if entry["module"] == "lora_mix":
                            self.deleted_sessions.add(entry["id"])
                    self.deletions = value
                except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
                    raise ExperimentError("deletion-recovery", "任务删除凭据无法安全读取；停止恢复，原文件保留未修改", 503) from error
            self.receipt_signature = self._receipt_signature()
            self.loaded = True
            if not self.root.is_dir():
                return
            for directory in sorted(self.root.iterdir()):
                if not directory.is_dir() or not ID_PATTERN.fullmatch(directory.name):
                    continue
                try:
                    self.safe_path(directory)
                    self.safe_path(directory / "definition.json")
                    self.safe_path(directory / "state.json")
                    definition = json.loads((directory / "definition.json").read_text(encoding="utf-8"))
                    state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
                    _validate_loaded(directory.name, definition, state)
                except (OSError, ValueError, KeyError, TypeError, AttributeError):
                    # Kept byte-for-byte; never repaired or rewritten by a guess.
                    self.unreadable[directory.name] = "实验记录无法安全读取；原始文件保留未修改。"
                    continue
                # Hide whole sessions too: a stray late round can never resurrect a deleted session.
                origin = definition.get("origin")
                if directory.name in deleted or (isinstance(origin, dict) and origin.get("session_id") in self.deleted_sessions):
                    self.deleted_records[directory.name] = self._index(definition, state)
                    continue
                self.records[directory.name] = self._index(definition, state)
                self._recover(directory.name)
            if deleted - self.deleted_records.keys():
                self.loaded = False
                raise ExperimentError("deletion-recovery", "已删除任务的历史档案缺失或无法读取；停止恢复与提交，避免丢失候选排重证据", 503)

    @staticmethod
    def safe_path(path):
        """No client paths, symlinks or Windows junctions. Local filesystem owners remain trusted."""
        path = Path(path).absolute()
        for part in (path, *path.parents):
            if part.is_symlink() or getattr(part, "is_junction", lambda: False)():
                raise ExperimentError("unsafe-store", "任务记录路径含链接或重解析目录；保留原文件，不执行删除", 409)
        if path.resolve() != path:
            raise ExperimentError("unsafe-store", "任务记录路径超出受信目录", 409)

    def _receipt_signature(self):
        path = self.root / "task-deletions.json"
        try:
            stat = path.stat()
            return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns
        except FileNotFoundError:
            return None

    def candidate_records(self):
        """Deletion hides tasks, not tested/reserved vector history. No budget/release algorithm changes."""
        with self.lock:
            self.ensure_loaded()
            return {**self.deleted_records, **self.records}

    def tombstone(self, module, task_id, experiment_ids):
        """One atomic receipt hides one XYZ task or an ENTIRE search session. Never removes image files.
        Definitions/state/thumbs stay as an archive (not a physical wipe), including candidate reservations.
        A failed receipt publication leaves every round visible; restart must read it before recovery."""
        with self.lock:
            self.ensure_loaded()
            self.safe_path(self.root)
            path = self.root / "task-deletions.json"
            self.safe_path(path)
            for key in experiment_ids:
                if not isinstance(key, str) or not ID_PATTERN.fullmatch(key):
                    raise ExperimentError("task-id", "任务编号无效", 422)
                self.record(key)
                for file in (self.root / key, self.root / key / "definition.json", self.root / key / "state.json"):
                    self.safe_path(file)
            receipt = copy.deepcopy(self.deletions)
            receipt["entries"].append({"module": module, "id": task_id, "experiment_ids": list(experiment_ids), "at": now()})
            # Cross-process publication is fail-closed, never last-writer-wins. A crashed lock is
            # retained for manual inspection, not stolen. This is not an OS sandbox against local owners.
            gate = self.root / ".task-deletion.lock"
            self.safe_path(gate)
            try:
                gate.mkdir()
            except FileExistsError as error:
                raise ExperimentError("deletion-busy", "删除记录正被另一写入者持有或需人工恢复；原任务保留", 409) from error
            try:
                if self._receipt_signature() != self.receipt_signature:
                    raise ExperimentError("deletion-recovery", "删除凭据发生并发变化；任务保留，请重启服务后核对", 503)
                try:
                    write_json_atomic(path, receipt)
                except OSError:
                    # A transport/filesystem error AFTER replace is not a failed deletion. Read the
                    # exact whole receipt before deciding; an ambiguous outcome blocks further access.
                    try:
                        published = json.loads(path.read_text(encoding="utf-8")) == receipt
                    except (OSError, ValueError):
                        published = False
                    if not published:
                        raise
                self.deletions = receipt
                if module == "lora_mix":
                    self.deleted_sessions.add(task_id)
                for key in experiment_ids:
                    self.deleted_records[key] = self.records.pop(key)
                self.receipt_signature = self._receipt_signature()
            finally:
                try:
                    gate.rmdir()
                except OSError:
                    pass  # Publication is committed; retained gate makes the next mutation fail closed.

    @staticmethod
    def _index(definition, state):
        cells = {cell["cell_id"]: cell for cell in definition["plan"]["cells"]}
        return {"definition": definition, "state": state, "cells": cells,
                "signatures": {key: cell_signature(cell) for key, cell in cells.items()},
                "resources": {key: resource_key(cell) for key, cell in cells.items()}}

    def _recover(self, experiment_id):
        """Restart marks interruption and re-checks completed originals. Nothing resumes by itself."""
        state = self.records[experiment_id]["state"]
        changed = False
        if state["status"] in EXPERIMENT_ACTIVE:
            owner = state.get("owner") or {}
            if owner.get("token") != self.owner["token"] and self.owner_alive(owner):
                # Another live service process owns it; never steal a live owner.
                state["foreign"] = True
                return
            state.update(status="interrupted", interrupted_at=now(), decision=None, current_cell=None, current_cells=[])
            for cell in state["cells"].values():
                if cell["status"] == "running":
                    cell.update(status="pending", note="interrupted")
            changed = True
        for record in state.get("exports") or []:
            if record.get("status") in EXPORT_ACTIVE:
                record.update(status="interrupted", error="服务重启时导出未完成；未发布任何文件", completed_at=now())
                changed = True
        changed = self._verify_completed(experiment_id) or changed
        if changed:
            self.save(experiment_id)

    def _verify_completed(self, experiment_id, cell_ids=None):
        """Recovery AND critical live reads. Never authenticates replacement pixels by backfilling SHA."""
        record = self.records[experiment_id]
        changed = False
        for cell_id in cell_ids if cell_ids is not None else record['definition']['order']:
            cell = record['state']['cells'][cell_id]
            if cell['status'] != 'complete':
                continue
            try:
                if not self.asset_exists((cell.get('output') or {}).get('relative')):
                    raise ValueError('原图文件已不在输出目录')
                if self.verify_asset is not None:
                    verification = self.verify_asset(record['definition'], record['cells'][cell_id], cell)
                    if cell.get('asset_verification') != verification:
                        cell['asset_verification'] = verification
                        changed = True
            except (OSError, ValueError, KeyError, TypeError) as error:
                cell.update(status='missing', asset_verification={'status': 'needs_repair'},
                            error=f'{error}；需明确重试生成新文件（不覆盖作品，不自动恢复完成状态）')
                changed = True
        return changed

    def verify_completed(self, experiment_id, cell_ids=None):
        with self.lock:
            self.record(experiment_id)
            if self._verify_completed(experiment_id, cell_ids):
                self.save(experiment_id)

    # Creation -------------------------------------------------------------------------------

    def create(self, *, submission_id, draft, on_error, plan, library_revision, requested_concurrency, created_at=None, origin=None):
        experiment_id = experiment_id_for(submission_id)
        digest = submission_digest(draft, on_error)
        with self.lock:
            self.ensure_loaded()
            if experiment_id in self.deleted_records or any(experiment_id in entry["experiment_ids"] for entry in self.deletions["entries"]):
                raise ExperimentError("deleted-submission", "该提交的任务已删除；不会重建或重新执行，请使用新的提交编号")
            if origin and origin.get("session_id") in self.deleted_sessions:
                raise ExperimentError("deleted-session", "该搜索任务已删除；旧草稿不能继续此会话，请明确新建搜索")
            existing = self.records.get(experiment_id)
            if existing is not None:
                if existing["definition"]["digest"] != digest:
                    raise ExperimentError("submission-conflict", "同一提交编号对应了不同的实验定义；请重新提交（不会合并或覆盖）")
                return experiment_id, False
            if experiment_id in self.unreadable:
                raise ExperimentError("submission-conflict", "该提交编号的实验记录不可读；请使用新的提交")
            directory = self.root / experiment_id
            self.root.mkdir(parents=True, exist_ok=True)
            try:
                directory.mkdir()
            except FileExistsError as error:
                raise ExperimentError("submission-conflict", "该提交编号已被占用；请使用新的提交") from error
            created = created_at or now()
            order = [cell["cell_id"] for cell in plan["cells"]]
            definition = {
                "schema_version": SCHEMA_VERSION, "kind": "xyz_experiment", "experiment_id": experiment_id,
                "submission_id": submission_id.lower(), "digest": digest, "created_at": created,
                "engine": plan["engine"], "draft": copy.deepcopy(draft), "on_error": on_error,
                "library_revision": library_revision, "requested_concurrency": requested_concurrency,
                "dimensions": plan["dimensions"], "total_cells": plan["total_cells"], "order": order,
                "axes": axis_summary(draft, plan), "baseline": baseline_summary(draft),
                # Policy at submission; an explicitly created continuation may set a new limit (state).
                "execution": {"policy": "sequential" if requested_concurrency == 1 else "compatible_batches",
                              "order": "grouped_by_resources" if draft.get("group_by_resources") else "z_sample_y_x",
                              "max_concurrency": MAX_CONCURRENCY},
                "plan": copy.deepcopy(plan),
            }
            # QPlan2/P2: a management-side provenance block (never part of the strict draft). It is part of the
            # write-once definition, so a search round can always be rebuilt from this record alone.
            if origin is not None:
                definition["origin"] = copy.deepcopy(origin)
            state = {
                "schema_version": SCHEMA_VERSION, "kind": "xyz_state", "experiment_id": experiment_id, "revision": 0,
                "status": "queued", "on_error": on_error, "decision": None, "current_cell": None,
                "owner": dict(self.owner), "requested_concurrency": requested_concurrency, "effective_concurrency": None,
                "concurrency_reason": "尚未开始；实际并发在每个批次执行时确定", "current_cells": [], "batch_log": [],
                "created_at": created, "updated_at": created, "started_at": None, "completed_at": None, "error": None,
                "cells": {cell_id: {"status": "pending", "attempts": 0, "history": []} for cell_id in order},
            }
            try:
                write_json_atomic(directory / "definition.json", definition)
                write_json_atomic(directory / "state.json", state)
            except BaseException:
                for name in ("state.json", "definition.json"):
                    (directory / name).unlink(missing_ok=True)
                directory.rmdir()
                raise
            self.records[experiment_id] = self._index(definition, state)
            return experiment_id, True

    # Access ---------------------------------------------------------------------------------

    def record(self, experiment_id):
        self.ensure_loaded()
        record = self.records.get(experiment_id) if isinstance(experiment_id, str) else None
        if record is None:
            if experiment_id in self.deleted_records:
                raise ExperimentError("deleted-task", "任务记录已删除；生成原图与已导出文件仍保留", 410)
            if experiment_id in self.unreadable:
                raise ExperimentError("unreadable", self.unreadable[experiment_id], 409)
            raise ExperimentError("not-found", "XYZ experiment not found", 404)
        return record

    def save(self, experiment_id):
        with self.lock:
            state = self.record(experiment_id)["state"]
            self.safe_path(self.root / experiment_id / "state.json")
            state["revision"] += 1
            state["updated_at"] = now()
            write_json_atomic(self.root / experiment_id / "state.json", state)

    def summary(self, experiment_id, *, verify_assets=True):
        with self.lock:
            record = self.record(experiment_id)
            if verify_assets:
                self.verify_completed(experiment_id)
            definition, state = record["definition"], record["state"]
            return {
                "id": experiment_id, "engine": definition["engine"], "created_at": definition["created_at"],
                "digest": definition["digest"], "status": state["status"], "revision": state["revision"], "total_cells": definition["total_cells"],
                "dimensions": definition["dimensions"], "repeat_count": definition["draft"].get("repeat_count", 1),
                "axes": {key: {"field": value["field"], "count": len(value["labels"])} for key, value in definition["axes"].items()},
                "counts": counts_of(state), "decision": copy.deepcopy(state["decision"]), "current_cell": state["current_cell"],
                "current_cells": list(state.get("current_cells") or []),
                "on_error": state["on_error"], "requested_concurrency": state["requested_concurrency"],
                "effective_concurrency": state["effective_concurrency"], "concurrency_reason": state["concurrency_reason"],
                "foreign": bool(state.get("foreign")), "error": state.get("error"),
                "group_by_resources": bool(definition["draft"].get("group_by_resources")),
                "last_batch": copy.deepcopy((state.get("batch_log") or [None])[-1]),
                "canvas": [definition["baseline"].get("width"), definition["baseline"].get("height")],
                "model": definition["baseline"].get("checkpoint") or definition["baseline"].get("diffusion_model"),
                "started_at": state["started_at"], "completed_at": state["completed_at"], "updated_at": state["updated_at"],
                "origin": origin_summary(definition),
            }

    def list(self, engine=None, limit=LIST_LIMIT, plain_only=False):
        """`plain_only` keeps experiments without an `origin` block (the XYZ workspace's own); it filters before the
        limit, so rounds owned by another workspace (e.g. LoRA mix search) can never push XYZ experiments off the page."""
        with self.lock:
            self.ensure_loaded()
            # This collection is polled; recovery and critical reads verify bytes without
            # making every list poll stream all completed originals again.
            entries = [self.summary(key, verify_assets=False) for key in self.records
                       if (engine is None or self.records[key]["definition"]["engine"] == engine)
                        and not (plain_only and self.records[key]["definition"].get("origin") is not None)]
            entries.sort(key=lambda item: item["created_at"], reverse=True)
            unreadable = [{"id": key, "error": reason} for key, reason in sorted(self.unreadable.items())]
            return {"experiments": entries[:limit], "total": len(entries), "unreadable": unreadable}

    def detail(self, experiment_id, asset_url):
        """Server snapshot for reconnection. Relative assets only; no local paths."""
        with self.lock:
            record = self.record(experiment_id)
            # Validate before taking the counts/cells projection so every field in this
            # response describes the same post-validation state; summary must not repeat it.
            self.verify_completed(experiment_id)
            definition, state = record["definition"], record["state"]
            cells = []
            for cell_id in definition["order"]:
                planned, current = record["cells"][cell_id], state["cells"][cell_id]
                output = current.get("output") or {}
                coordinate = planned["coordinate"]
                cells.append({
                    "id": cell_id,
                    "coordinate": [coordinate["x_index"], coordinate["y_index"], coordinate["z_index"], coordinate["sample_index"]],
                    "labels": planned["labels"], "status": current["status"], "attempts": current["attempts"],
                    "seed": str(planned["request"]["seed"]), "error": current.get("error"), "note": current.get("note"),
                    "asset_verification": copy.deepcopy(current.get("asset_verification")),
                    **({"asset_id": output["asset_id"], "asset_url": asset_url(output["asset_id"]), "output_name": output.get("name"), "width": output.get("width"),
                        "height": output.get("height"), "elapsed_seconds": output.get("elapsed_seconds"),
                        "batch": copy.deepcopy(output.get("batch"))} if current["status"] == "complete" and output.get("asset_id") else {}),
                })
            return {
                **self.summary(experiment_id, verify_assets=False),
                "axes": copy.deepcopy(definition["axes"]), "baseline": copy.deepcopy(definition["baseline"]),
                "library_revision": definition["library_revision"], "execution": copy.deepcopy(definition["execution"]),
                "seed_policy": definition["plan"].get("seed_policy"), "cells": cells,
                "batch_log": copy.deepcopy(state.get("batch_log") or []),
                "exports": copy.deepcopy(state.get("exports") or []),
                "origin": copy.deepcopy(definition.get("origin")), "mix_released": copy.deepcopy(state.get("mix_released")),
            }

    def cell_detail(self, experiment_id, cell_id, asset_url):
        """Actual frozen request, identities and attempt history of one cell."""
        with self.lock:
            record = self.record(experiment_id)
            planned = record["cells"].get(cell_id)
            if planned is None:
                raise ExperimentError("not-found", "XYZ cell not found", 404)
            self.verify_completed(experiment_id, [cell_id])
            current = copy.deepcopy(record["state"]["cells"][cell_id])
            output = current.get("output")
            if current['status'] == 'complete' and output and output.get("asset_id"):
                output["asset_url"] = asset_url(output["asset_id"])
            return {"experiment_id": experiment_id, "cell_id": cell_id, "coordinate": planned["coordinate"], "labels": planned["labels"],
                    "request": copy.deepcopy(planned["request"]), "lora_references": copy.deepcopy(planned["lora_references"]),
                    "replacement_matches": planned.get("replacement_matches", {}),
                    "prompt_parts": cell_prompt_parts(record["definition"]["draft"], planned), **current}

    # Exports (Phase D) ---------------------------------------------------------------------------

    def grid_snapshot(self, experiment_id, resolve_path):
        """Everything the grid renderer needs, copied under the lock (later cells don't move it).
        `resolve_path(relative)` returns a contained absolute path or None."""
        with self.lock:
            record = self.record(experiment_id)
            self.verify_completed(experiment_id)
            definition, state = record["definition"], record["state"]
            cells = []
            for cell_id in definition["order"]:
                planned, current = record["cells"][cell_id], state["cells"][cell_id]
                output = current.get("output") or {}
                coordinate = planned["coordinate"]
                cells.append({
                    "id": cell_id, "coordinate": [coordinate["x_index"], coordinate["y_index"], coordinate["z_index"], coordinate["sample_index"]],
                    "labels": copy.deepcopy(planned["labels"]), "status": current["status"], "seed": str(planned["request"]["seed"]),
                    "request": copy.deepcopy(planned["request"]), "lora_references": copy.deepcopy(planned["lora_references"]),
                    "batch": copy.deepcopy(output.get("batch")), "relative": output.get("relative") if current["status"] == "complete" else None,
                    "asset_id": output.get("asset_id"), "width": output.get("width"), "height": output.get("height"),
                    "asset_fingerprint": copy.deepcopy(output.get("asset_fingerprint")),
                    "asset_verification": copy.deepcopy(current.get("asset_verification")),
                    **({'asset_expectation': self._asset_expectation(definition, planned, current)} if self.verify_asset else {}),
                })
            snapshot = {"experiment_id": experiment_id, "digest": definition["digest"], "engine": definition["engine"],
                        "axes": copy.deepcopy(definition["axes"]), "dimensions": copy.deepcopy(definition["dimensions"]),
                        "repeat_count": definition["draft"].get("repeat_count", 1), "baseline": copy.deepcopy(definition["baseline"]),
                        "created_at": definition["created_at"], "cells": cells}
        for cell in cells:
            if cell["relative"]:
                path = resolve_path(cell["relative"])
                if path is None:
                    cell.update(status="missing", relative=None)
                else:
                    cell["path"] = path
        return snapshot

    @staticmethod
    def _asset_expectation(definition, planned, current):
        from .xyz_assets import asset_expectation
        output = current.get('output') or {}
        # Only completed originals need an expectation; failed cells keep their labels/records.
        return asset_expectation(definition, planned, output, current['attempts']) if current['status'] == 'complete' else None

    def add_export(self, experiment_id, export):
        with self.lock:
            state = self.record(experiment_id)["state"]
            state["exports"] = (state.get("exports") or [])[-(EXPORT_LIMIT - 1):] + [export]
            self.save(experiment_id)

    def update_export(self, experiment_id, export_id, **fields):
        with self.lock:
            state = self.record(experiment_id)["state"]
            for export in state.get("exports") or []:
                if export["id"] == export_id:
                    export.update(fields)
                    self.save(experiment_id)
                    return copy.deepcopy(export)
            raise ExperimentError("not-found", "XYZ export not found", 404)

    def export_record(self, experiment_id, export_id):
        with self.lock:
            for export in self.record(experiment_id)["state"].get("exports") or []:
                if export["id"] == export_id:
                    return copy.deepcopy(export)
            raise ExperimentError("not-found", "XYZ export not found", 404)

    def frozen_cell(self, experiment_id, cell_id):
        with self.lock:
            record = self.record(experiment_id)
            return copy.deepcopy(record["cells"][cell_id]), record["definition"]["engine"]

    # Explicit user transitions ------------------------------------------------------------------

    def decide(self, experiment_id, action):
        """Answer the pause a failed cell caused. Returns True when the experiment must be requeued."""
        with self.lock:
            state = self.record(experiment_id)["state"]
            decision = state.get("decision")
            if state["status"] != "paused" or not decision:
                raise ExperimentError("no-decision", "当前没有等待选择的失败格")
            members = decision.get("cell_ids") or [decision["cell_id"]]
            state["decision"] = None
            if action == "stop":
                self._mark_stopped(state)
                self.save(experiment_id)
                return False
            for cell_id in members:
                state["cells"][cell_id].update(status="pending" if action == "retry" else "skipped")
            state.update(status="queued", owner=dict(self.owner))
            self.save(experiment_id)
            return True

    def continue_experiment(self, experiment_id, retry="none", requested_concurrency=None):
        """Explicit continuation. Unexecuted/cancelled cells are requeued; failed ones only on request.
        A new concurrency limit applies to this continuation only; frozen requests never change."""
        with self.lock:
            state = self.record(experiment_id)["state"]
            if requested_concurrency is not None and (type(requested_concurrency) is not int or not 1 <= requested_concurrency <= MAX_CONCURRENCY):
                raise ExperimentError("concurrency", f"并发必须为 1–{MAX_CONCURRENCY} 的整数；不会自动调整", 422)
            if state.get("foreign"):
                raise ExperimentError("foreign-owner", "另一个仍在运行的服务进程拥有此实验；不会抢占")
            if state["status"] not in CONTINUABLE:
                raise ExperimentError("not-continuable", "实验仍在进行或等待选择；不能重复继续")
            cells = state["cells"]
            if isinstance(retry, list):
                for cell_id in retry:
                    if cell_id not in cells:
                        raise ExperimentError("not-found", f"XYZ cell {cell_id} not found", 404)
                    if cells[cell_id]["status"] not in RETRYABLE_CELLS:
                        raise ExperimentError("immutable-cell", f"格 {cell_id} 状态为 {cells[cell_id]['status']}；已完成原图不可重试覆盖")
                targets = set(retry)
            elif retry == "failed":
                targets = {cell_id for cell_id, cell in cells.items() if cell["status"] in RETRYABLE_CELLS}
            else:
                targets = set()
            for cell_id, cell in cells.items():
                if cell["status"] == "cancelled" or cell_id in targets:
                    cell["status"] = "pending"
            if not any(cell["status"] == "pending" for cell in cells.values()):
                raise ExperimentError("nothing-to-continue", "没有待完成的格；失败格需明确选择重试")
            state.update(status="queued", decision=None, error=None, completed_at=None, owner=dict(self.owner))
            state.pop("mix_released", None)
            if requested_concurrency is not None:
                state["requested_concurrency"] = requested_concurrency
            self.save(experiment_id)

    def release_reservation(self, experiment_id):
        """Explicitly free the unexecuted vectors of a round that is not running. Executed originals stay."""
        with self.lock:
            record = self.record(experiment_id)
            state = record["state"]
            if (record["definition"].get("origin") or {}).get("kind") != "lora_mix_search":
                raise ExperimentError("not-search", "该实验不是 LoRA 混合搜索回合", 422)
            if state["status"] not in {"interrupted", "cancelled", "error"} or state.get("foreign"):
                raise ExperimentError("not-releasable", "只能释放已中断 / 已停止 / 编排错误回合的未执行向量；运行中的回合请先停止")
            state["mix_released"] = {"at": now()}
            self.save(experiment_id)

    def stop_queued(self, experiment_id):
        """Stop an experiment whose orchestrator has not started (or has released the slot)."""
        with self.lock:
            state = self.record(experiment_id)["state"]
            state["decision"] = None
            self._mark_stopped(state)
            self.save(experiment_id)

    @staticmethod
    def _mark_stopped(state):
        for cell in state["cells"].values():
            if cell["status"] in {"pending", "running"}:
                cell["status"] = "cancelled"
        state.update(status="cancelled", current_cell=None, current_cells=[], completed_at=now())


def cell_signature(cell):
    """Compatibility key: the frozen request minus its base seed, plus the LoRA identities. Two
    cells share a tensor batch only when everything else — models, LoRA list/weights/identity,
    prompt, canvas, steps, CFG/guidance, sampler/scheduler and the post-processing chain including
    its frozen resample seeds — is byte-identical."""
    request = {key: value for key, value in cell["request"].items() if key != "seed"}
    identities = [[item.get("instanceId"), item.get("contentHash"), item.get("contentRevision")] for item in cell.get("lora_references") or []]
    return json.dumps({"request": request, "identities": identities}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def resource_key(cell):
    """What a switch costs: the mounted model components and LoRA list."""
    request = cell["request"]
    return json.dumps({"models": [request.get(key) for key in MODEL_KEYS], "loras": request.get("loras")}, sort_keys=True, ensure_ascii=False)


def difference_labels(first, second):
    a, b = first["request"], second["request"]
    labels = []
    for key in sorted(set(a) | set(b)):
        if key != "seed" and a.get(key) != b.get(key):
            label = FIELD_LABELS.get(key, key)
            if label not in labels:
                labels.append(label)
    return labels


def next_batch(order, cells, statuses, signatures, resources, limit, *, group_by_resources=False, last_resource=None):
    """The next batch: the first pending cell in logical Z/sample→Y→X order (or, when grouping by
    resources, the first pending cell that keeps the loaded model/LoRA set), plus further pending
    cells with the identical signature, up to `limit`. Coordinates never change; only execution
    order does. Returns (members, reason) or None."""
    pending = [key for key in order if statuses[key] == "pending"]
    if not pending:
        return None
    lead = pending[0]
    if group_by_resources and last_resource is not None:
        lead = next((key for key in pending if resources[key] == last_resource), lead)
    members = [lead] + [key for key in pending if key != lead and signatures[key] == signatures[lead]][:max(0, limit - 1)]
    return members, explain_batch(lead, members, pending, cells, limit)


def explain_batch(lead, members, pending, cells, limit):
    """Why this group has its size. Never claims an "actual" figure: that is only known after the
    run (runtime micro-batching or an OOM split can lower it) and is composed by the runner."""
    size = len(members)
    if limit <= 1:
        return "逐格顺序生成（一次一张）"
    if size >= limit:
        return "仅种子不同的兼容格合并为一个真实批次"
    others = [key for key in pending if key not in members]
    if not others:
        return f"只剩 {size} 个待生成格"
    labels = []
    for key in others[:32]:
        for label in difference_labels(cells[lead], cells[key]):
            if label not in labels:
                labels.append(label)
    return f"只有 {size} 个兼容格，其余待生成格的{'、'.join(labels[:4]) or '请求'}不同；不改参数凑批"


def project_batches(plan_cells, limit, *, group_by_resources=False):
    """Preview of how a plan would be grouped (runtime micro-batching may still lower actual size)."""
    order = [cell["cell_id"] for cell in plan_cells]
    cells = {cell["cell_id"]: cell for cell in plan_cells}
    signatures = {key: cell_signature(cell) for key, cell in cells.items()}
    resources = {key: resource_key(cell) for key, cell in cells.items()}
    statuses = dict.fromkeys(order, "pending")
    batches, last = [], None
    while True:
        step = next_batch(order, cells, statuses, signatures, resources, limit, group_by_resources=group_by_resources, last_resource=last)
        if step is None:
            return batches
        members, reason = step
        batches.append({"cells": members, "size": len(members), "reason": f"请求 {limit} / 分组 {len(members)}：{reason}"})
        for key in members:
            statuses[key] = "done"
        last = resources[members[0]]


class ExperimentRunner:
    """The loop that occupies ONE slot of the single inference executor.

    hooks must provide:
      checkpoint(experiment_id, phase)            blocks while paused; raises ExperimentStopped
      admit(experiment_id, cell_ids)              admitted batch request or raises CellRefused
      run(experiment_id, cell_ids, attempts, admitted) -> {status, outputs{cell: output}, error, oom,
                                                  job_id, sampling_batch_size, interrupted}
      publish(experiment_id)                      mirror the store into the parent job record
      finished(experiment_id)                     release the parent control/slot bookkeeping
    """

    def __init__(self, store, hooks):
        self.store = store
        self.hooks = hooks

    def run(self, experiment_id):
        store = self.store
        with store.lock:
            record = store.record(experiment_id)
            state, order = record["state"], record["definition"]["order"]
            status = state["status"]
            if status == "cancelling":
                # A stop raced the slot: nothing has run in this slot, so record it as stopped.
                ExperimentStore._mark_stopped(state)
                store.save(experiment_id)
            if status not in {"queued", "pausing"}:
                # Stopped (or otherwise settled) while still in the executor queue; nothing to run.
                self.hooks.finished(experiment_id)
                return
            # A pause requested before the slot came keeps `pausing`; the first checkpoint holds it.
            state.update(status="running" if status == "queued" else "pausing", started_at=state["started_at"] or now(), owner=dict(store.owner))
            store.save(experiment_id)
            grouping = bool(record["definition"]["draft"].get("group_by_resources"))
        self.hooks.publish(experiment_id)
        last_resource = None
        try:
            while True:
                self.hooks.checkpoint(experiment_id, "准备下一批")
                with store.lock:
                    statuses = {key: cell["status"] for key, cell in state["cells"].items()}
                    step = next_batch(order, record["cells"], statuses, record["signatures"], record["resources"],
                                      int(state.get("requested_concurrency") or 1), group_by_resources=grouping, last_resource=last_resource)
                    if step is None:
                        break
                    members, reason = step
                last_resource = record["resources"][members[0]]
                outcome = self._execute(experiment_id, state, members, reason, int(state.get("requested_concurrency") or 1))
                if outcome == "paused":
                    return
                self.hooks.publish(experiment_id)
            with store.lock:
                state.update(status="complete", current_cell=None, current_cells=[], completed_at=now())
                store.save(experiment_id)
        except ExperimentStopped as stop:
            with store.lock:
                if stop.interrupted:
                    for cell in state["cells"].values():
                        if cell["status"] == "running":
                            cell.update(status="pending", note="interrupted")
                    state.update(status="interrupted", current_cell=None, current_cells=[], interrupted_at=now())
                else:
                    ExperimentStore._mark_stopped(state)
                store.save(experiment_id)
        except Exception as error:  # noqa: BLE001 - an orchestration fault must keep every completed cell
            with store.lock:
                for cell in state["cells"].values():
                    if cell["status"] == "running":
                        cell.update(status="pending", note="orchestrator-error")
                state.update(status="error", current_cell=None, current_cells=[], error=f"{type(error).__name__}: {error}"[:2000])
                store.save(experiment_id)
        finally:
            self.hooks.finished(experiment_id)

    def _execute(self, experiment_id, state, members, reason, limit, oom_from=None):
        """Run one batch. An OOM batch of several images is split in halves and each half re-run
        with fresh generators from the same frozen seeds (bounded: at most log2(size) splits per
        cell); canvas, steps, seeds and models never change. Returns "paused" when a failure
        paused the experiment, otherwise None; raises ExperimentStopped."""
        store = self.store
        with store.lock:
            attempts = {}
            for cell_id in members:
                cell = state["cells"][cell_id]
                attempts[cell_id] = cell["attempts"] + 1
                # Persisted BEFORE execution: a crash leaves evidence that these cells were in flight.
                cell.update(status="running", attempts=attempts[cell_id], started_at=now(), error=None, note=None)
            # Planned only; the actual figure is written after the run.
            state.update(current_cell=members[0], current_cells=list(members), effective_concurrency=None,
                         concurrency_reason=f"请求 {limit} / 本批 {len(members)}：{reason}（执行后确认实际并发）")
            if state["status"] not in {"pausing", "paused", "cancelling"}:
                state["status"] = "running"
            store.save(experiment_id)
        self.hooks.publish(experiment_id)
        try:
            admitted = self.hooks.admit(experiment_id, list(members))
        except CellRefused as error:
            result = {"status": "error", "error": str(error), "stage": "admission", "sampling_started": False}
        else:
            result = self.hooks.run(experiment_id, list(members), attempts, admitted)
        actual = result.get("sampling_batch_size")
        # Group size is a plan, not evidence of sampling. Only the runtime marker can report N.
        actual = actual if type(actual) is int and actual > 0 else None
        sampling_status = "reported" if actual is not None else "not_started" if result.get("sampling_started") is False else "unknown"
        effective = min(len(members), actual) if actual is not None else 0 if sampling_status == "not_started" else None
        explanation = f"请求 {limit} / 实际 {effective if effective is not None else '未知'}：{reason}"
        if actual is None:
            explanation += "；采样尚未开始" if sampling_status == "not_started" else "；缺少可靠采样反馈，不将分组数计为实际并发"
        if actual and actual < len(members):
            # Runtime micro-batching is sequential sampling, never reported as concurrency.
            explanation += f"；{len(members)} 张同批分组，但显存策略按每批 {actual} 张采样（逐批执行，不计为并发）"
        if oom_from:
            explanation += f"；显存不足后由 {oom_from} 张拆分重试"
        batch = {"cells": list(members), "requested": limit, "size": len(members), "sampling_batch_size": actual,
                 "sampling_status": sampling_status,
                 "effective": effective, "status": result["status"], "reason": explanation, "grouping": reason,
                 "job_id": result.get("job_id"), "oom": bool(result.get("oom")), "at": now()}
        split = result["status"] == "error" and result.get("oom") and len(members) > 1
        with store.lock:
            state["batch_log"] = (state.get("batch_log") or [])[-(BATCH_LOG_LIMIT - 1):] + [batch]
            state.update(effective_concurrency=batch["effective"], concurrency_reason=explanation)
            if split:
                for cell_id in members:
                    self._history(state["cells"][cell_id], attempts[cell_id], "oom-split", result, len(members))
                    state["cells"][cell_id]["status"] = "pending"
                store.save(experiment_id)
            else:
                stopped = None
                for index, cell_id in enumerate(members):
                    stopped = self._record(state, cell_id, attempts[cell_id], result, batch, index) or stopped
                state.update(current_cell=None, current_cells=[])
                if stopped is None and result["status"] == "error" and state["on_error"] == "pause":
                    state.update(status="paused", decision={"cell_id": members[0], "cell_ids": list(members), "error": result.get("error"), "at": now()})
                    store.save(experiment_id)
                    return "paused"
                store.save(experiment_id)
                if stopped is not None:
                    raise stopped
                return None
        half = (len(members) + 1) // 2
        for part in (members[:half], members[half:]):
            self.hooks.checkpoint(experiment_id, "显存不足，拆分为较小批次重试")
            if self._execute(experiment_id, state, part, reason, limit, oom_from=len(members)) == "paused":
                return "paused"
        return None

    @staticmethod
    def _history(cell, attempt, status, result, size):
        entry = {"attempt": attempt, "status": status, "job_id": result.get("job_id"), "batch_size": size, "at": now()}
        if result.get("error"):
            entry["error"] = result["error"][:2000]
        cell["history"] = (cell.get("history") or [])[-(HISTORY_LIMIT - 1):] + [entry]

    @classmethod
    def _record(cls, state, cell_id, attempt, result, batch, index):
        """Fold one batch result into one cell. Returns ExperimentStopped when the run must end."""
        cell = state["cells"][cell_id]
        status = result["status"]
        cls._history(cell, attempt, status, result, batch["size"])
        cell["completed_at"] = now()
        if status == "complete":
            output = dict(result["outputs"][cell_id])
            output["batch"] = {"size": batch["size"], "index": index, "cells": batch["cells"], "requested": batch["requested"],
                               "sampling_batch_size": batch["sampling_batch_size"], "effective": batch["effective"]}
            cell.update(status="complete", error=None, output=output, job_id=result.get("job_id"))
            return None
        if status == "error":
            cell.update(status="error", error=(result.get("error") or "生成失败")[:2000], job_id=result.get("job_id"))
            return None
        interrupted = bool(result.get("interrupted"))
        cell.update(status="pending" if interrupted else "cancelled", note="interrupted" if interrupted else None)
        return ExperimentStopped(interrupted=interrupted)
