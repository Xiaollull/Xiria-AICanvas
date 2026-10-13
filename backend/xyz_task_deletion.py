"""Module-scoped, record-only task deletion. Caller holds store -> export -> jobs locks."""
import hashlib
import json

from .xyz_experiments import CONTINUABLE, EXPERIMENT_ACTIVE, EXPORT_ACTIVE, ID_PATTERN, ExperimentError


def options(payload, *, committing=False):
    allowed = {"scope", "ids", "fingerprint"} if committing else {"scope", "ids"}
    if not isinstance(payload, dict) or set(payload) - allowed or payload.get("scope") not in {"all", "selected"}:
        raise ExperimentError("delete-options", "删除范围必须为 selected 或 all；不接受客户端路径/引擎", 422)
    ids = payload.get("ids")
    if payload["scope"] == "all":
        if ids is not None:
            raise ExperimentError("delete-options", "删除全部由服务器枚举，不接受 ids", 422)
    elif not isinstance(ids, list) or not 1 <= len(ids) <= 200 or any(not isinstance(key, str) or not ID_PATTERN.fullmatch(key) for key in ids) or len(set(ids)) != len(ids):
        raise ExperimentError("task-id", "请选择 1–200 个不同的合法任务编号", 422)
    if committing and (not isinstance(payload.get("fingerprint"), str) or len(payload["fingerprint"]) != 64):
        raise ExperimentError("delete-confirmation", "请先读取并确认精确删除范围", 422)
    return payload["scope"], ids


def inactive_reason(store, experiment_id, refs, exports, jobs, controls):
    record = store.record(experiment_id)
    state = record["state"]
    if state.get("foreign") or store.owner_alive(state.get("owner") or {}):
        return "其他存活进程拥有此任务"
    if state["status"] in EXPERIMENT_ACTIVE:
        return f"任务状态为 {state['status']}（运行 / 排队 / 暂停中的任务保留）"
    if state["status"] not in CONTINUABLE or state.get("decision") or state.get("current_cell") or state.get("current_cells") or any(cell["status"] == "running" for cell in state["cells"].values()):
        return "任务仍持有活动单元或待处理决定"
    if experiment_id in refs or experiment_id in controls:
        return "执行器 / 队列仍持有任务"
    if any(entry.get("experiment_id") == experiment_id for entry in exports.values()) or any(entry.get("status") in EXPORT_ACTIVE for entry in state.get("exports") or []):
        return "任务正在导出或等待导出工作器释放"
    if any((key == experiment_id or job.get("xyz_parent") == experiment_id) and (job.get("status") in EXPERIMENT_ACTIVE or job.get("xyz_current_job")) for key, job in jobs.items()):
        return "任务仍有活动队列 / 子任务引用"
    try:
        for path in (store.root / experiment_id, store.root / experiment_id / "definition.json", store.root / experiment_id / "state.json"):
            store.safe_path(path)
    except (ExperimentError, OSError) as error:
        return str(error)
    return None


def task_map(store, module, refs, exports, jobs, controls):
    tasks = {}
    for experiment_id, record in store.records.items():
        definition = record["definition"]
        origin = definition.get("origin")
        if module == "xyz":
            if origin is not None:
                continue
            task_id = experiment_id
        else:
            if not isinstance(origin, dict) or origin.get("kind") != "lora_mix_search" or origin.get("schema_version") != 1 or not isinstance(origin.get("session_id"), str) or not ID_PATTERN.fullmatch(origin["session_id"]):
                continue
            task_id = origin["session_id"]
        reason = inactive_reason(store, experiment_id, refs, exports, jobs, controls)
        entry = tasks.setdefault(task_id, {"id": task_id, "engine": definition["engine"], "created_at": definition["created_at"], "updated_at": 0,
                                           "status": record["state"]["status"], "reason": None, "experiment_ids": [], "rounds": 0})
        entry["experiment_ids"].append(experiment_id)
        entry["rounds"] += 1
        if definition["created_at"] >= entry["updated_at"]:
            entry.update(updated_at=definition["created_at"], latest_experiment=experiment_id, status=record["state"]["status"])
        if reason:
            entry["reason"] = f"第 {origin.get('round')} 轮：{reason}" if module == "lora_mix" else reason
    for entry in tasks.values():
        if module == "lora_mix" and store.unreadable:
            entry["reason"] = "存在无法安全读取的实验记录，无法证明已涵盖此搜索的全部轮次；整任务保留"
        entry["deletable"] = entry["reason"] is None
        entry["experiment_ids"].sort()
    return tasks


def preview(store, module, payload, refs, exports, jobs, controls):
    scope, ids = options(payload)
    tasks = task_map(store, module, refs, exports, jobs, controls)
    entries = list(tasks.values()) if scope == "all" else [tasks.get(key) or {"id": key, "deletable": False, "reason": "任务不存在、已删除或不属于当前模块", "experiment_ids": []} for key in ids]
    entries.sort(key=lambda item: item["id"])
    # Revision includes membership, current cell state, and export transitions, not only a UI status.
    revisions = {key: store.records[key]["state"]["revision"] for entry in entries if entry["deletable"] for key in entry["experiment_ids"]}
    value = {"module": module, "scope": scope, "tasks": entries, "total": len(entries), "eligible": sum(item["deletable"] for item in entries)}
    value["skipped"] = value["total"] - value["eligible"]
    value["unreadable"] = [{"id": key, "error": reason} for key, reason in sorted(store.unreadable.items())]
    value["fingerprint"] = hashlib.sha256(json.dumps([value, revisions], sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    return value


def commit(store, module, payload, refs, exports, jobs, controls):
    options(payload, committing=True)
    value = preview(store, module, {key: val for key, val in payload.items() if key != "fingerprint"}, refs, exports, jobs, controls)
    if value["fingerprint"] != payload["fingerprint"]:
        raise ExperimentError("delete-changed", "任务范围或活动状态已变化；没有删除任何任务，请重新核对确认", 409)
    results = []
    for entry in value["tasks"]:
        outcome, reason = "skipped", entry["reason"]
        if entry["deletable"]:
            try:
                store.tombstone(module, entry["id"], entry["experiment_ids"])
                # Only inactive program-owned mirrors. Never modify ordinary generation or images.
                for key in list(jobs):
                    if key in entry["experiment_ids"] or jobs[key].get("xyz_parent") in entry["experiment_ids"]:
                        jobs.pop(key)
                outcome, reason = "deleted", None
            except (OSError, ExperimentError) as error:
                outcome, reason = "failed", f"删除未获确认，请刷新核对后重试（不会删除图像）：{error}"
        results.append({"id": entry["id"], "outcome": outcome, "reason": reason, "experiment_ids": entry["experiment_ids"]})
    return {"module": module, "scope": value["scope"], "results": results, **{key: sum(entry["outcome"] == key for entry in results) for key in ("deleted", "skipped", "failed")}}
