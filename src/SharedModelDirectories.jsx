import { useEffect, useState } from "react";
import { Check, FolderPlus, RefreshCw, ShieldCheck, Trash2, X } from "lucide-react";
import { formatFileSize } from "./format-size";
import { hostPathExample } from "./host-platform";

const SHARED_ENTRY_POINTS = [
  {
    kind: "auto",
    title: "模型根目录",
    detail: "适合 ComfyUI、SD WebUI 等工具的完整 models 目录，自动识别其中的模型类型",
    placeholder: hostPathExample("modelsRoot"),
  },
  {
    kind: "loras",
    title: "单独 LoRA 目录",
    detail: "适合分散存放的 LoRA 文件夹，保留原有子目录结构并显示在「共享」分类中",
    placeholder: hostPathExample("loraDirectory"),
  },
];

export default function SharedModelDirectories({ onChanged }) {
  const [roots, setRoots] = useState([]);
  const [draftKind, setDraftKind] = useState("auto");
  const [draftPath, setDraftPath] = useState("");
  const [report, setReport] = useState(null);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const selectedEntry = SHARED_ENTRY_POINTS.find((entry) => entry.kind === draftKind) || SHARED_ENTRY_POINTS[0];

  const refreshRoots = async () => {
    setBusy((current) => current || "refresh");
    setError("");
    try {
      const response = await fetch("/api/shared-paths", { cache: "no-store" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "无法读取共享目录");
      setRoots(payload.roots || []);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy((current) => current === "refresh" ? "" : current);
    }
  };

  useEffect(() => { void refreshRoots(); }, []);

  const inspectSharedPath = async () => {
    const target = draftPath.trim();
    if (!target || busy) return;
    setBusy("inspect");
    setError("");
    try {
      const response = await fetch("/api/shared-paths/inspect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: target, kind: draftKind }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "目录校验失败");
      setReport({ ...payload.inspection, kind: draftKind, registered: payload.registered });
    } catch (requestError) {
      setReport(null);
      setError(requestError.message);
    } finally {
      setBusy("");
    }
  };

  const commitSharedRoot = async () => {
    if (!report || busy) return;
    setBusy("commit");
    setError("");
    try {
      const response = await fetch("/api/shared-paths", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: report.path, kind: report.kind }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "无法保存共享目录");
      setRoots(payload.roots || []);
      setDraftPath("");
      setReport(null);
      onChanged?.();
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy("");
    }
  };

  const updateSharedRoot = async (root, changes) => {
    if (busy) return;
    setBusy(root.id);
    setError("");
    try {
      const response = await fetch("/api/shared-paths", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: root.id, ...changes }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "无法更新共享目录");
      setRoots(payload.roots || []);
      onChanged?.();
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy("");
    }
  };

  const removeSharedRoot = async (root) => {
    if (busy) return;
    setBusy(root.id);
    setError("");
    try {
      const response = await fetch(`/api/shared-paths?id=${encodeURIComponent(root.id)}`, { method: "DELETE" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "无法移除共享目录");
      setRoots(payload.roots || []);
      onChanged?.();
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy("");
    }
  };

  return <div className="shared-model-directories">
    <section className="shared-entry">
      <header><div><span>ADD SOURCE</span><strong>添加一个共享来源</strong><small>确认后可以继续添加下一个目录；共享来源只读，不会成为模型下载位置。</small></div><b className="shared-source-count">来源数量（{roots.length}/24）</b></header>
      <div className="shared-kind-switch" role="radiogroup" aria-label="共享目录类型">
        {SHARED_ENTRY_POINTS.map((entry) => <button type="button" role="radio" aria-checked={draftKind === entry.kind} className={draftKind === entry.kind ? "active" : ""} key={entry.kind} onClick={() => { setDraftKind(entry.kind); setReport(null); setError(""); }}><strong>{entry.title}</strong><small>{entry.detail}</small></button>)}
      </div>
      <label>
        <input value={draftPath} spellCheck="false" placeholder={selectedEntry.placeholder} onChange={(event) => { setDraftPath(event.target.value); setReport(null); }} onKeyDown={(event) => { if (event.key === "Enter") { event.preventDefault(); void inspectSharedPath(); } }} />
        <button type="button" disabled={!draftPath.trim() || Boolean(busy)} onClick={() => void inspectSharedPath()}>{busy === "inspect" ? <RefreshCw className="spin" size={13} /> : <ShieldCheck size={13} />}{busy === "inspect" ? "正在校验" : "校验目录"}</button>
      </label>
    </section>

    {error && <p className="shared-paths-error"><X size={13} />{error}</p>}

    {report && <div className="shared-report">
      <header><span>SCAN RESULT</span><code title={report.path}>{report.path}</code></header>
      {report.entries.length ? <div className="shared-report-kinds">{report.entries.map((item) => <div key={`${item.kind}-${item.directory}`} className={item.supported ? "" : "unsupported"}><b>{item.label}</b><span>{item.directory === "." ? "当前目录" : item.directory}</span><small>{item.files} 个文件 · {formatFileSize(item.bytes)}</small>{!item.supported && <i>暂未接入</i>}</div>)}</div> : <p className="shared-report-empty">没有找到可识别的模型文件</p>}
      {report.warnings.map((warning) => <p className="shared-report-warning" key={warning}>{warning}</p>)}
      <footer><span>{report.registered ? "该目录已经注册；确认后会更新现有记录，不会新增重复路径" : "确认后仅读取该目录，不会写入、移动或删除任何文件"}</span><div><button type="button" className="secondary" onClick={() => setReport(null)}>取消</button><button type="button" disabled={!report.entries.length || Boolean(busy)} onClick={() => void commitSharedRoot()}>{busy === "commit" ? <RefreshCw className="spin" size={14} /> : <FolderPlus size={14} />}确认添加</button></div></footer>
    </div>}

    <section className="shared-root-section">
      <header><div><span>REGISTERED SOURCES</span><strong>已挂载目录</strong><small>每个路径都是独立来源，可分别停用或移除；移除记录不会删除磁盘文件。</small></div><button type="button" disabled={Boolean(busy)} onClick={() => void refreshRoots()}><RefreshCw className={busy === "refresh" ? "spin" : ""} size={13} />刷新</button></header>
      <div className="shared-root-list">
        {roots.length ? roots.map((root) => <div className={`shared-root-row ${root.enabled ? "" : "off"}`} key={root.id}><div><strong>{root.label}</strong><code title={root.path}>{root.path}</code></div><span>{root.kind === "loras" ? "LoRA 目录" : root.kind === "checkpoints" ? "底模目录" : "模型根目录"}</span><button type="button" disabled={Boolean(busy)} className={root.enabled ? "on" : ""} onClick={() => void updateSharedRoot(root, { enabled: !root.enabled })}>{root.enabled ? <><Check size={12} />已启用</> : "已停用"}</button><button type="button" className="shared-root-remove" disabled={Boolean(busy)} title="移除共享记录（不会删除磁盘文件）" onClick={() => void removeSharedRoot(root)}><Trash2 size={13} /></button></div>) : <p className="shared-root-empty">尚未共享任何目录。你可以分别添加 ComfyUI、SD WebUI 和其他模型库路径。</p>}
      </div>
    </section>
  </div>;
}
