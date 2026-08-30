import { useEffect, useRef, useState } from "react";
import { ArrowLeft, Check, RefreshCw, Upload } from "lucide-react";
import { LoadingLogo } from "./BrandLogo";
import { formatFileSize } from "./format-size";
import { DEFAULT_THEME, applyThemeToDocument, loadThemeState } from "./theme";
import {
  clearUpdateRestart,
  markUpdateRestart,
  updateBusy,
  UPDATE_OCCUPIED_STATUSES,
  updateRestartPending,
  waitForUpdatedApplication,
} from "./update-navigation";

const updateArchivePattern = /\.(?:zip|7z|rar|tar|tar\.gz|tgz|tar\.xz|txz|tar\.bz2|tbz2|tar\.zst|tzst)$/i;

async function updatedApplicationReady() {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), 2500);
  try {
    const response = await fetch(`/api/inference/health?restart=${Date.now()}`, {
      cache: "no-store",
      signal: controller.signal,
    });
    if (!response.ok) return false;
    const health = await response.json();
    return health.status === "ready";
  } catch {
    return false;
  } finally {
    window.clearTimeout(timer);
  }
}

function returnToMainPage() {
  clearUpdateRestart(window.sessionStorage);
  window.location.replace(new URL("/", window.location.href).href);
}

export default function ManualUpdatePage() {
  const fileInput = useRef(null);
  const [updateState, setUpdateState] = useState({ status: "idle", phase: "idle", progress: 0, message: "请选择更新归档", maximum_bytes: 4 * 1024 ** 3 });
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");
  const [updateActionBusy, setUpdateActionBusy] = useState(false);
  const [restarting, setRestarting] = useState(() => updateRestartPending(window.sessionStorage));
  const restartReturnStarted = useRef(false);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/ui-state", { cache: "no-store" })
      .then(async (response) => response.ok ? response.json() : null)
      .then((payload) => { if (!cancelled) applyThemeToDocument(loadThemeState(payload?.state?.theme)); })
      .catch(() => { if (!cancelled) applyThemeToDocument({ ...DEFAULT_THEME }); });
    return () => { cancelled = true; };
  }, []);

  const finishRestartAndReturn = async () => {
    if (restartReturnStarted.current) return;
    restartReturnStarted.current = true;
    setRestarting(true);
    setUpdateState((current) => ({ ...current, message: "正在重启并加载更新后的程序", phase: "restart" }));
    try {
      await waitForUpdatedApplication({ checkHealth: updatedApplicationReady, returnHome: returnToMainPage });
    } catch (restartError) {
      restartReturnStarted.current = false;
      clearUpdateRestart(window.sessionStorage);
      setRestarting(false);
      setError(restartError.message);
    }
  };

  useEffect(() => {
    if (updateRestartPending(window.sessionStorage)) void finishRestartAndReturn();
  }, []);

  const refreshState = async () => {
    const response = await fetch("/api/system/update", { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "无法读取更新状态");
    setUpdateState(payload);
    setError(payload.status === "error" ? payload.message || "更新失败" : "");
  };

  useEffect(() => {
    let disposed = false;
    const poll = async () => {
      try {
        if (!disposed && !restarting) await refreshState();
      } catch (pollError) {
        if (!disposed && !restarting) setError(pollError.message);
      }
    };
    void poll();
    const timer = window.setInterval(poll, 600);
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }, [restarting]);

  const uploadArchive = (file) => {
    if (!file || uploading) return;
    if (!updateArchivePattern.test(file.name)) {
      setError("请选择 ZIP、7Z、RAR、TAR、TAR.GZ 或 TAR.XZ 项目更新包");
      return;
    }
    if (!file.size || file.size > updateState.maximum_bytes) {
      setError(`更新包必须小于 ${formatFileSize(updateState.maximum_bytes)}`);
      return;
    }
    setError("");
    setUploading(true);
    setUploadProgress(0);
    const request = new XMLHttpRequest();
    request.open("POST", "/api/system/update/archive");
    request.setRequestHeader("Content-Type", "application/octet-stream");
    request.setRequestHeader("X-Archive-Name", encodeURIComponent(file.name));
    request.upload.addEventListener("progress", (event) => {
      if (event.lengthComputable) setUploadProgress(Math.round(event.loaded / event.total * 100));
    });
    request.addEventListener("load", () => {
      let payload = {};
      try { payload = JSON.parse(request.responseText || "{}"); } catch {}
      setUploading(false);
      if (request.status < 200 || request.status >= 300) {
        setError(payload.error || "更新包上传失败");
        return;
      }
      setUploadProgress(100);
      setUpdateState((current) => ({ ...current, ...payload }));
    });
    request.addEventListener("error", () => {
      setUploading(false);
      setError("更新包上传中断，请重新选择文件");
    });
    request.send(file);
  };

  const startUpdate = async () => {
    if (updateActionBusy) return;
    setError("");
    setUpdateActionBusy(true);
    try {
      const response = await fetch("/api/system/update/apply", { method: "POST" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "无法开始更新");
      setUpdateState((current) => ({ ...current, ...payload }));
    } catch (updateError) {
      setError(updateError.message);
    } finally {
      setUpdateActionBusy(false);
    }
  };

  const repairEnvironment = async () => {
    if (updateActionBusy) return;
    setError("");
    setUpdateActionBusy(true);
    try {
      const response = await fetch("/api/system/update/repair", { method: "POST" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "无法开始环境修复");
      setUpdateState((current) => ({ ...current, ...payload }));
    } catch (repairError) {
      setError(repairError.message);
    } finally {
      setUpdateActionBusy(false);
    }
  };

  const returnHome = async () => {
    if (uploading || updateBusy(updateState.status) || updateState.restart_required) return;
    try { await fetch("/api/system/update", { method: "DELETE" }); } catch {}
    window.location.replace("/");
  };

  const restartAndReturn = async () => {
    setError("");
    markUpdateRestart(window.sessionStorage);
    setRestarting(true);
    try {
      const controller = new AbortController();
      const timer = window.setTimeout(() => controller.abort(), 5000);
      let response;
      try {
        response = await fetch("/api/system/update/restart", { method: "POST", signal: controller.signal });
      } catch {}
      window.clearTimeout(timer);
      if (!response) {
        void finishRestartAndReturn();
        return;
      }
      if (!response.ok) {
        let payload = {};
        try { payload = await response.json(); } catch {}
        throw new Error(payload.error || "无法重启应用");
      }
      void finishRestartAndReturn();
    } catch (restartError) {
      clearUpdateRestart(window.sessionStorage);
      setRestarting(false);
      setError(restartError.message);
    }
  };

  const busy = uploading || updateActionBusy || updateBusy(updateState.status) || restarting;
  // The two origins share every step after the archive reaches disk, so the server marks which one
  // started it and the wording follows that rather than the phase alone.
  const online = updateState.source === "online";
  const visualProgress = restarting ? 100
    : uploading ? uploadProgress * .2
      : updateState.status === "uploading" || updateState.status === "downloading" ? updateState.progress * .2
        : updateState.status === "preparing" ? 20 + updateState.progress * .35
          : updateState.status === "ready" ? 55
            : updateState.status === "applying" ? 55 + updateState.progress * .45
              : updateState.status === "repairing" ? Math.max(92, updateState.progress || 0)
                : updateState.status === "complete" ? 100 : 0;
  const phaseOrder = ["upload", "inspect", "extract", "dependencies", "backup", "apply", "verify", "environment", "complete"];
  const phaseMap = { idle: -1, download: 0, tools: 1, validate: 2, plan: 3, ready: 3, shutdown: 4, repair: 7, cleanup: 8, restart: 8, error: -1 };
  const phaseIndex = uploading || ["uploading", "downloading"].includes(updateState.status) ? 0 : phaseMap[updateState.phase] ?? phaseOrder.indexOf(updateState.phase);
  const phaseLabels = [online ? "下载更新包" : "上传更新包", "安全检查", "命令解压", "依赖分析", "创建回滚", "替换程序", "代码验证", "环境修复验证", "更新完成"];
  const replacementPlan = (updateState.prepared_plan || []).filter((item) => item.action !== "remove");
  const removalPlan = (updateState.prepared_plan || []).filter((item) => item.action === "remove");

  return <main className="update-shell">
    <div className="loader-grid" />
    <button type="button" className="update-back" disabled={busy || updateState.restart_required} onClick={returnHome}><ArrowLeft size={16} />返回主页</button>
    <section className="update-panel">
      <div className={`update-visual ${busy ? "busy" : ""} ${updateState.status === "complete" ? "complete" : ""}`}>
        <div className="update-emblem"><LoadingLogo /></div>
        <span>{online ? "ONLINE PACKAGE UPDATE" : "LOCAL PACKAGE UPDATE"}</span>
        <h1>{online ? "在线更新" : "手动更新"}</h1>
        <p>{online
          ? `正在获取并安装 ${updateState.online_release?.version || "新"} 版本的程序文件。已有环境、模型、输出、日志和创作状态不会被覆盖。`
          : "使用打包好的干净项目归档更新程序文件。已有环境、模型、输出、日志和创作状态不会被覆盖。"}</p>
        <div className="update-safety-list"><span><Check size={13} />新版镜像删除旧文件</span><span><Check size={13} />环境异常自动修复</span><span><Check size={13} />失败自动回滚</span></div>
      </div>
      <div className="update-workspace">
        <header><span>UPDATE WORKSPACE</span><strong>{updateState.status === "complete" ? "更新已完成" : updateState.status === "ready" ? "等待确认" : updateState.status === "downloading" ? "正在下载更新包" : busy ? "正在处理" : "选择本地更新包"}</strong></header>
        {!UPDATE_OCCUPIED_STATUSES.includes(updateState.status) && !updateState.restart_required && !updateState.repair_available && !restarting && <button
          type="button"
          className={`update-dropzone ${dragging ? "dragging" : ""}`}
          onClick={() => !uploading && fileInput.current?.click()}
          onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={(event) => { if (!event.currentTarget.contains(event.relatedTarget)) setDragging(false); }}
          onDrop={(event) => { event.preventDefault(); setDragging(false); uploadArchive(event.dataTransfer.files?.[0]); }}
          disabled={uploading}
        >
          <Upload size={30} />
          <strong>{uploading ? "正在上传更新包" : "点击选择，或将归档拖入此处"}</strong>
          <span>ZIP / 7Z / RAR / TAR / TAR.GZ / TAR.XZ · 最大 {formatFileSize(updateState.maximum_bytes)}</span>
          {uploading && <div className="update-upload-track"><i style={{ width: `${uploadProgress}%` }} /></div>}
        </button>}
        <input ref={fileInput} type="file" hidden accept=".zip,.7z,.rar,.tar,.gz,.tgz,.xz,.txz,.bz2,.tbz2,.zst,.tzst" onChange={(event) => { uploadArchive(event.target.files?.[0]); event.target.value = ""; }} />
        {(busy || updateState.restart_required || ["ready", "complete"].includes(updateState.status)) && <div className="update-progress-card">
          <div className="update-current"><span>{restarting ? "RESTART" : updateState.phase?.toUpperCase()}</span><strong>{updateState.message}</strong><small>{updateState.filename ? `${updateState.filename} · ${formatFileSize(updateState.bytes)}` : "环境、模型与用户数据保持原位"}</small></div>
          <div className="update-master-track"><i style={{ width: `${Math.max(0, Math.min(100, visualProgress))}%` }} /></div>
          <div className="update-master-meta"><span>整体进度</span><b>{Math.round(visualProgress)}%</b></div>
          <div className="update-phase-grid">{phaseLabels.map((label, index) => <div key={label} className={index < phaseIndex || updateState.status === "complete" ? "done" : index === phaseIndex ? "active" : ""}><i>{index < phaseIndex || updateState.status === "complete" ? <Check size={10} /> : String(index + 1).padStart(2, "0")}</i><span>{label}</span></div>)}</div>
        </div>}
        {updateState.status === "ready" && <div className="update-plan">
          <span>将替换以下程序项</span>
          <div>{replacementPlan.map((item) => <code key={item.relativePath}>{item.relativePath}</code>)}</div>
          {removalPlan.length > 0 && <><span className="update-remove-title">新版已删除，将从旧项目移除</span><div>{removalPlan.map((item) => <code className="remove" key={item.relativePath}>{item.relativePath}</code>)}</div></>}
          <p>{updateState.environment_repair_required ? "检测到依赖清单变化。替换完成后会自动修复 Python、后端和前端环境，并重新验证。" : "更新包已完成安全检查。替换完成后会重新检查 Python、后端、CUDA 与前端构建。"}验证成功前不会删除回滚备份，也不能重启返回主页面。</p>
        </div>}
        {(error || updateState.status === "error") && <div className="update-error"><strong>更新未完成</strong><p>{error || updateState.message}</p></div>}
        <footer className="update-actions">
          {!updateState.restart_required && <button type="button" onClick={returnHome} disabled={busy}>取消并返回主页</button>}
          {updateState.status === "ready" && <button type="button" className="primary" onClick={startUpdate}>开始更新</button>}
          {updateState.repair_available && !updateState.environment_ready && <button type="button" className="primary" disabled={busy} onClick={repairEnvironment}><RefreshCw size={15} />重新自动修复环境</button>}
          {updateState.restart_required && updateState.environment_ready && <button type="button" className="primary" disabled={restarting || busy} onClick={restartAndReturn}>{restarting ? <><RefreshCw className="spin" size={15} />正在重启</> : <><RefreshCw size={15} />重启并返回主页面</>}</button>}
        </footer>
      </div>
    </section>
    <span className="loader-corner loader-corner-tl" /><span className="loader-corner loader-corner-tr" /><span className="loader-corner loader-corner-bl" /><span className="loader-corner loader-corner-br" />
  </main>;
}
