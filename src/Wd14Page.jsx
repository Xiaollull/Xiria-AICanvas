import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";
import { Download, FolderOpen, ImagePlus, Play, Square, Tags, Copy, RotateCw, Cpu, Check, X } from "lucide-react";
import { Wd14Desk } from "./wd14-desk.js";
import TaggerInfoPopover from "./TaggerInfoPopover.jsx";
import { TAGGER_ENGINES, TAGGER_CATEGORY_LABELS, taggerDefaults, taggerParameters, resultSource } from "./tagger-settings.js";
import { createCatalogRequest, catalogReadIntent } from "./tagger-catalog-request.js";
import { initialInputGuidance, updateInputGuidance } from "./tagger-input-guidance.js";

const API = "/api/inference/wd14";
async function request(url, options) {
  const response = await fetch(`${API}${url}`, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const failure = new Error(typeof body.detail === "string" ? body.detail : `请求失败 (${response.status})`);
    failure.status = response.status;
    throw failure;
  }
  return body;
}
function textFile(name, text) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/plain;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `${name.replace(/\.[^.]+$/, "")}.txt`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function percent(job) { return job?.total ? Math.min(100, Math.round((job.done + (job.skipped || 0) + (job.failed || 0)) / job.total * 100)) : 0; }

function ImagePreview({ file }) {
  const [preview, setPreview] = useState(null);
  useEffect(() => {
    if (!file) return undefined;
    const url = URL.createObjectURL(file);
    setPreview({ file, url });
    return () => URL.revokeObjectURL(url);
  }, [file]);
  return preview?.file === file ? <img src={preview.url} alt={file.name} /> : null;
}

export default function Wd14Page() {
  const [catalog, setCatalog] = useState(null);
  const [model, setModel] = useState("wd-swinv2-v3");
  const [engine, setEngine] = useState("wd");
  const [device, setDevice] = useState("cpu");
  const [thresholds, setThresholds] = useState(() => taggerDefaults("wd"));
  const [limit, setLimit] = useState(50);
  const [alpha, setAlpha] = useState(false);
  const [desk] = useState(() => new Wd14Desk());
  const workspace = useSyncExternalStore(desk.subscribe, desk.getSnapshot, desk.getSnapshot);
  const [dragging, setDragging] = useState(false);
  const [filter, setFilter] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [folder, setFolder] = useState("");
  const [outputFolder, setOutputFolder] = useState("");
  const [recursive, setRecursive] = useState(false);
  const [conflict, setConflict] = useState("ignore");
  const localBrowser = ["localhost", "127.0.0.1", "[::1]"].includes(window.location.hostname);
  const [job, setJob] = useState(null);
  const [install, setInstall] = useState(null);
  const fileInput = useRef(null);
  const historyStrip = useRef(null);
  const alive = useRef(true);
  const catalogEpoch = useRef(0);
  const jobPollEpoch = useRef(0), installPollEpoch = useRef(0);
  const [catalogRequest] = useState(() => createCatalogRequest((force) => { const intent = catalogReadIntent(force, localBrowser); return request(intent.path, intent.options); }));
  const refresh = useCallback(async (force = false) => {
    const epoch = ++catalogEpoch.current;
    const value = await catalogRequest(force && localBrowser);
    if (alive.current && epoch === catalogEpoch.current) {
      setCatalog(value);
      setFolder((current) => current || value.batch_root);
    }
  }, [catalogRequest, localBrowser]);
  useEffect(() => {
    if (!catalog?.verification_pending) return undefined;
    const timer = setTimeout(() => { void refresh().catch((failure) => { if (alive.current) setError(failure.message); }); }, 750);
    return () => clearTimeout(timer);
  }, [catalog, refresh]);
  useEffect(() => {
    alive.current = true;
    desk.activate();
    refresh().catch((failure) => setError(failure.message));
    request("/active").then((value) => {
      if (!alive.current) return;
      for (const activeJob of value.jobs) {
        if (["download", "runtime"].includes(activeJob.kind)) setInstall(activeJob);
        if (activeJob.kind === "batch") setJob(activeJob);
      }
    }).catch(() => {});
    return () => { alive.current = false; catalogEpoch.current++; desk.dispose(); };
  }, [refresh, desk]);
  useEffect(() => {
    const epoch = ++jobPollEpoch.current;
    if (!job || !["queued", "running", "publishing"].includes(job.state)) return undefined;
    const timer = setInterval(() => request(`/jobs/${job.id}`).then((value) => {
      if (alive.current && epoch === jobPollEpoch.current) {
        setJob(value);
        if (value.state === "done") void refresh(true);
      }
    }).catch((failure) => { if (alive.current && epoch === jobPollEpoch.current) setError(failure.message); }), 700);
    return () => { jobPollEpoch.current++; clearInterval(timer); };
  }, [job?.id, job?.state, refresh]);
  useEffect(() => {
    const epoch = ++installPollEpoch.current;
    if (!install || !["queued", "running", "publishing"].includes(install.state)) return undefined;
    const timer = setInterval(() => request(`/jobs/${install.id}`).then((value) => {
      if (alive.current && epoch === installPollEpoch.current) {
        setInstall(value);
        if (value.state === "done") void refresh(true);
      }
    }).catch((failure) => { if (alive.current && epoch === installPollEpoch.current) setError(failure.message); }), 700);
    return () => { installPollEpoch.current++; clearInterval(timer); };
  }, [install?.id, install?.state, refresh]);
  function accept(newFiles) {
    const id = desk.add(newFiles);
    if (!id) return;
    setError(""); setNotice(""); setFilter("");
    if (installed && runnable && !busy) void analyze(id);
    else desk.update(id, initialInputGuidance({ engine, model, device }, catalog, busy));
  }
  function analyze(id = workspace.selectedId) {
    if (!installed || !runnable || busy) return;
    setError(""); setNotice("");
    return desk.run(id, taggerParameters(engine, model, device, thresholds, limit, alpha), (file, params, signal) =>
      request(`/tag?${new URLSearchParams(params)}`, { method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: file, signal }));
  }
  async function startInstall() {
    setError("");
    try { setInstall(await request(`/download/${encodeURIComponent(model)}`, { method: "POST" })); }
    catch (failure) { setError(failure.message); }
  }
  async function installGpuRuntime() {
    if (!window.confirm("安装固定官方 ORT-GPU 1.28.0 / CUDA 13 依赖层（约 230 MiB 下载）？仅写入项目 .cache/tagger-runtimes，主 .venv / Torch / transformers 不变。")) return;
    setError("");
    try { setInstall(await request("/runtime/install", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ confirm: true }) })); }
    catch (failure) { setError(failure.message); }
  }
  function changeEngine(next) {
    const selected = TAGGER_ENGINES.find(row => row.id === next);
    setEngine(next); setModel(selected.default); setThresholds(taggerDefaults(next)); setError(""); setNotice("");
    // No rerun effect: pending results keep their immutable originating snapshot.
  }
  async function startBatch(event) {
    event.preventDefault(); setError("");
    try {
      setJob(await request("/batch", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ folder, output_folder: outputFolder, ...taggerParameters(engine, model, device, thresholds, limit, alpha), recursive, conflict }) }));
    } catch (failure) { setError(failure.message); }
  }
  async function cancelJob(current) {
    const pollEpoch = ["download", "runtime"].includes(current.kind) ? installPollEpoch.current : jobPollEpoch.current;
    try {
      const response = await request(`/jobs/${current.id}/cancel`, { method: "POST" });
      if (!response.ok) setNotice(`任务已经${response.state}，无法再取消`);
    } catch (failure) {
      if (failure.status === 409 && ["download", "runtime"].includes(current.kind)) setNotice("模型或依赖层正在发布，不能取消；请等待安装结果");
      else setError(failure.message);
    } finally {
      request(`/jobs/${current.id}`).then((latest) => {
        const liveEpoch = ["download", "runtime"].includes(current.kind) ? installPollEpoch.current : jobPollEpoch.current;
        if (!alive.current || pollEpoch !== liveEpoch) return;
        if (["download", "runtime"].includes(current.kind)) setInstall(latest);
        else setJob(latest);
      }).catch(() => {});
    }
  }
  const item = workspace.entries.find((entry) => entry.id === workspace.selectedId);
  const history = workspace.entries;
  const running = workspace.entries.some((entry) => entry.status === "pending");
  const installed = catalog?.models.find((entry) => entry.id === model)?.installed;
  const selectedModel = catalog?.models.find((entry) => entry.id === model);
  const selectedEngine = (catalog?.engines || TAGGER_ENGINES).find(row => row.id === engine);
  const gpu = selectedModel?.devices?.gpu;
  const runnable = selectedModel?.ready !== false && (device === "cpu" || gpu?.available === true);
  const availabilityReason = selectedModel?.readiness_reason || (device === "gpu" && gpu?.available !== true ? gpu?.reason || "正在检查 GPU 能力；不会自动回退 CPU" : "");
  const busy = (job && ["queued", "running", "publishing"].includes(job.state)) || (install && ["queued", "running", "publishing"].includes(install.state));
  useEffect(() => {
    const patch = updateInputGuidance(item, { engine, model, device }, catalog, busy);
    if (patch) desk.update(item.id, patch);
  }, [item, engine, model, device, catalog, busy, desk]);
  const previewFile = item?.file;
  const matchedTags = item?.result?.tags.filter((tag) => tag.name.toLowerCase().includes(filter.toLowerCase())) || [];
  useEffect(() => {
    const strip = historyStrip.current;
    const selected = strip?.querySelector('button[aria-pressed="true"]');
    if (!selected) return;
    const keepVisible = () => {
      const shelf = strip.getBoundingClientRect(), thumb = selected.getBoundingClientRect();
      if (thumb.left < shelf.left) strip.scrollLeft += thumb.left - shelf.left;
      else if (thumb.right > shelf.right) strip.scrollLeft += thumb.right - shelf.right;
    };
    keepVisible();
    const observer = new ResizeObserver(keepVisible);
    observer.observe(strip);
    return () => observer.disconnect();
  }, [workspace.selectedId, history.length]);
  return <main className="wd14-page">
    <div className="wd14-grid">
      <section className="wd14-card wd14-work" aria-label="图片分析">
        <div className="wd14-card-head wd14-work-head"><h2><ImagePlus size={18} aria-hidden="true" />图片分析</h2><div className="wd14-actions"><small>单图自动反推 · 缓存最多 30 项</small><button className="wd14-primary" type="button" disabled={!item || item.status === "pending" || busy || !installed || !runnable} onClick={() => analyze()}><Play size={15} aria-hidden="true" />{item?.status === "pending" ? "正在反推…" : item?.result ? "重新反推" : "开始反推"}</button>{running && <button type="button" onClick={() => desk.stop()}><Square size={14} aria-hidden="true" />停止等待</button>}</div></div>
        <div className="wd14-analysis-deck">
          <div className="wd14-source">
            <div className="wd14-pane-head"><h3>图片输入</h3><small>每张 ≤20 MiB</small></div>
            <div className={`wd14-drop${dragging ? " dragging" : ""}${previewFile ? " has-preview" : ""}`} role="button" aria-label="选择图片 / 拖入 / 粘贴，新图替换并自动反推" tabIndex={0}
              onClick={() => fileInput.current?.click()} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); fileInput.current?.click(); } }}
              onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={(event) => { if (!event.currentTarget.contains(event.relatedTarget)) setDragging(false); }}
              onDrop={(event) => { event.preventDefault(); setDragging(false); accept(event.dataTransfer.files); }} onPaste={(event) => { if (event.clipboardData.files.length) { event.preventDefault(); accept(event.clipboardData.files); } }}>
              {previewFile ? <><ImagePreview file={previewFile} /><span className="wd14-preview-add"><ImagePlus size={16} aria-hidden="true" />{dragging ? "松开替换并反推" : "点击、拖入或粘贴以替换"}</span></> : <div className="wd14-drop-copy"><ImagePlus size={32} aria-hidden="true" /><strong>{dragging ? "松开替换并反推" : "选择图片 / 拖入 / 粘贴"}</strong><span>新图替换当前输入，模型就绪时自动反推</span><small>每次一张 · 聚焦后 Ctrl / ⌘ + V 粘贴</small></div>}
            </div>
            <input ref={fileInput} type="file" accept="image/*" hidden onChange={(event) => { accept(event.target.files); event.target.value = ""; }} />
            <div className="wd14-source-caption"><span title={previewFile?.name}>{previewFile?.name || "PNG、JPEG、WebP 等图片格式"}</span>{previewFile && <small>{(previewFile.size / 1024 / 1024).toFixed(2)} MiB</small>}</div>
          </div>
          <div className="wd14-results" aria-label="Prompt输出">
            <div className="wd14-pane-head"><h3><Tags size={16} aria-hidden="true" />Prompt输出</h3><small title={item?.params ? JSON.stringify(item.params) : ""}>{item?.status === "pending" ? "反推中（保留原结果）" : resultSource(item)}</small></div>
            <div className="wd14-ratings" aria-label="评级置信度">{["general", "sensitive", "questionable", "explicit"].map((name) => <span key={name}>{name}<b>{item?.result?.rating?.[name] == null ? "—" : `${(item.result.rating[name] * 100).toFixed(1)}%`}</b></span>)}</div>
            <label className="wd14-editor">可编辑 Prompt<textarea value={item?.text || ""} disabled={!item?.result} placeholder={item?.status === "pending" ? "正在分析当前图片…" : "添加图片后自动反推。结果可直接编辑、复制或导出。"} spellCheck={false} onChange={(event) => desk.edit(item.id, event.target.value)} /></label>
            <div className="wd14-actions"><button type="button" disabled={!item?.result} onClick={() => { const id = item.id; navigator.clipboard.writeText(item.text).then(() => { if (alive.current) desk.update(id, { notice: "已复制当前 Prompt" }); }).catch(() => { if (alive.current) desk.update(id, { error: "复制失败，请手动复制文本" }); }); }}><Copy size={15} aria-hidden="true" />复制</button><button type="button" disabled={!item?.result} onClick={() => textFile(item.file.name, item.text)}><Download size={15} aria-hidden="true" />导出 TXT</button><small>使用编辑后的文本</small></div>
            <details className="wd14-tag-details" open><summary>标签与置信度 · {item?.result?.tags.length || 0}</summary><div className="wd14-tag-body">
              <label className="wd14-search">搜索标签 <small>{matchedTags.length} / {item?.result?.tags.length || 0}</small><input value={filter} onChange={(event) => setFilter(event.target.value)} placeholder="输入关键词筛选置信度标签" /></label>
              <div className="wd14-chips">{matchedTags.map((tag, index) => <span key={`${tag.name}-${index}`} title={`${TAGGER_CATEGORY_LABELS[tag.category] || tag.category || "常规"} · 置信度 ${tag.confidence.toFixed(3)}`}>{tag.name}<small>{TAGGER_CATEGORY_LABELS[tag.category] || tag.category || "常规"} · {Math.round(tag.confidence * 100)}%</small></span>)}</div>
              {!matchedTags.length && <p className="wd14-note">{filter ? "没有匹配的标签，试试其他关键词。" : "当前阈值下没有标签，可调整参数后重新分析。"}</p>}
            </div></details>
          </div>
        </div>
        <section className="wd14-history" aria-label="反推结果"><div className="wd14-history-head"><h3>反推结果 <small>{history.length} / 30</small></h3><div><button type="button" disabled={!item} onClick={() => { desk.removeSelected(); setNotice(""); setError(""); }}>移除当前</button><button type="button" disabled={!workspace.entries.length} onClick={() => { desk.clear(); setNotice(""); setError(""); }}>清空结果</button></div></div><div ref={historyStrip} className="wd14-history-strip">{history.length ? history.map((entry) => <button type="button" key={entry.id} data-entry-id={entry.id} className={item?.id === entry.id ? "active" : ""} aria-pressed={item?.id === entry.id} aria-label={`查看反推结果 ${entry.file.name}`} title={entry.file.name} onClick={() => { desk.select(entry.id); setNotice(""); setFilter(""); }}><ImagePreview file={entry.file} /><span>{entry.file.name}<small>{entry.status === "pending" ? "反推中" : entry.status === "error" ? entry.result ? "更新失败，原结果保留" : "失败，可重试" : entry.status === "stopped" ? "已停止等待" : entry.result ? "已完成" : "待反推"}</small></span>{entry.result && <Check size={13} aria-hidden="true" />}</button>) : <p className="wd14-note">完成后的图片会暂存在这里，点击缩略图切换图片与 Prompt。</p>}</div></section>
        <div className="wd14-work-foot"><div className={error || workspace.message || item?.error ? "wd14-error" : "wd14-note"} role={error || workspace.message || item?.error ? "alert" : "status"}>{error || workspace.message || item?.error || notice || item?.notice || (!catalog ? "正在读取模型状态…" : "新图替换输入并自动反推；图片与结果仅在当前页面保留。")}</div></div>
      </section>
      <aside className="wd14-sidebar" aria-label="反推设置与文件夹批量">
        <section className="wd14-card wd14-controls" aria-label="反推参数">
          <div className="wd14-card-head"><h2>模型与参数</h2><button type="button" title="刷新模型状态" aria-label="刷新模型状态" onClick={() => refresh(true).catch((failure) => setError(failure.message))}><RotateCw size={16} aria-hidden="true" /></button></div>
          <div className="wd14-engine-field">
            <div className="wd14-engine-heading"><span>模型引擎</span><TaggerInfoPopover key={engine} engine={selectedEngine} /></div>
            <div className="model-segment wd14-engine-grid" role="group" aria-label="模型引擎">
              {(catalog?.engines || TAGGER_ENGINES).map((row) => {
                const presentation = {
                  wd: ["WD", "SmilingWolf"], cl: ["CL", "cella110n"],
                  camie: ["Camie", "Camais03"], pixai: ["PixAI", "PixAI Labs"],
                }[row.id] || [row.name, ""];
                return <button key={row.id} type="button" data-engine={row.id} className={`model-seg ${engine === row.id ? "active" : ""}`} aria-pressed={engine === row.id} onClick={() => changeEngine(row.id)}>
                  <span className="seg-name">{presentation[0]}</span><span className="seg-detail">{presentation[1]}</span>
                </button>;
              })}
            </div>
          </div>
          <label>模型<select value={model} onChange={(event) => setModel(event.target.value)}>{(catalog?.models?.filter(row => row.engine === engine || (!row.engine && engine === "wd")) || [{ id: model, installed: false }]).map((entry) => <option key={entry.id} value={entry.id}>{entry.id} · {entry.verification_pending ? "校验中" : entry.installed ? "已安装" : "未安装"}</option>)}</select></label>
          <div className="wd14-status"><span className={installed && runnable ? "ready" : ""}>{!catalog ? "读取状态中" : selectedModel?.verification_pending ? "固定列表已加载 · 本地摘要校验中" : !installed ? "未安装" : runnable ? "已就绪" : "模型已安装 · 运行不可用"}</span>{selectedModel?.repo_url && <a href={selectedModel.repo_url} target="_blank" rel="noopener noreferrer">官方模型仓库 ↗</a>}</div>
          {!localBrowser && <p className="wd14-note">远程只读刷新：读取当前状态与实时资源；完整性强校验仅允许在本机同源页面操作。</p>}
          <label>推理设备<select aria-label="推理设备" value={device} onChange={(event) => setDevice(event.target.value)}><option value="cpu">CPU</option><option value="gpu">GPU · NVIDIA CUDA</option></select></label>
          <div className="wd14-device"><Cpu size={16} aria-hidden="true" /><div><b>{engine === "pixai" ? "PyTorch" : "ONNX Runtime"} · {device === "gpu" ? "CUDA" : "CPU"}</b><span>{device === "cpu" ? engine === "pixai" ? "PyTorch CPU 推理不占用生图显存；不是 ONNX。" : "ONNX CPU 推理不占用生图显存。" : "生图忙碌时拒绝；独立 worker 退出释放自身显存，不卸载生图缓存。"}</span></div></div>
          {availabilityReason && <p className="wd14-error" role="status">{availabilityReason}</p>}
          {gpu?.budget?.recommend_gpu && <p className="wd14-note">大模型建议评估 GPU：{gpu.budget.parameters_million ? `${gpu.budget.parameters_million}M 参数 · ` : ""}{gpu.budget.input_size || "待验证"}px。FP32 权重约 {gpu.budget.weights_mib} MiB＋激活/工作区估计 {gpu.budget.activations_workspace_mib} MiB＋余量 {gpu.budget.margin_mib} MiB；当前可用 {gpu.free_vram_mib} MiB。估计并非保证；独立 worker 的完整校验与冷启动有额外开销，单图 CPU 可能更快，仍可手动选择。</p>}
          {engine !== "pixai" && catalog?.gpu_runtime?.installed === false && <button type="button" disabled={busy || !localBrowser || catalog.gpu_runtime.pending || !catalog.gpu_runtime.install_supported} onClick={installGpuRuntime}>{catalog.gpu_runtime.pending ? "GPU 依赖层校验中…" : "确认安装 ONNX GPU 依赖层"}</button>}
          {!installed && <button className="wd14-primary" type="button" disabled={busy || !localBrowser || !catalog || selectedModel?.verification_pending} onClick={startInstall}><Download size={16} aria-hidden="true" />{selectedModel?.verification_pending ? "本地模型校验中…" : "安装所选模型"}</button>}
          {install && <div className="wd14-progress" role="status"><div><span>{install.kind === "runtime" ? "GPU 依赖层" : `模型下载 · ${install.model || ""}`} · {install.state === "publishing" ? "正在发布 · 不可取消" : install.state}</span><b>{percent(install)}%</b></div><progress aria-label="模型下载进度" max={100} value={percent(install)} />{install.error && <p className="wd14-error">{install.error}</p>}{install.warning && <p className="wd14-note">{install.warning}</p>}{["queued", "running"].includes(install.state) && localBrowser && <button type="button" onClick={() => cancelJob(install)}>取消下载</button>}</div>}
          <div className="wd14-params">
            {Object.entries(thresholds).map(([category, value]) => <label key={category} className="wd14-slider"><span>{TAGGER_CATEGORY_LABELS[category]}阈值</span><output>{value.toFixed(2)}</output><input type="range" aria-label={`${TAGGER_CATEGORY_LABELS[category]}阈值`} min="0" max="1" step="0.01" value={value} style={{ "--wd14-progress": `${value * 100}%` }} onChange={(event) => setThresholds(current => ({ ...current, [category]: Number(event.target.value) }))} /></label>)}
            <div className="wd14-param-row"><label>最多标签<input type="number" min="1" max="500" value={limit} onChange={(event) => setLimit(Math.min(500, Math.max(1, Number(event.target.value) || 1)))} /></label><label className="wd14-check"><input type="checkbox" checked={alpha} onChange={(event) => setAlpha(event.target.checked)} />按字母排序</label></div>
            <p className="wd14-note">阈值越高，保留的标签越少。批量使用相同参数。</p>
          </div>
        </section>
        <section className="wd14-card wd14-batch" aria-label="本地文件夹批量">
          <div className="wd14-card-head"><h2><FolderOpen size={17} aria-hidden="true" />文件夹批量</h2><small>本机专用</small></div>
          <p className="wd14-note">最多 500 张 · 写入同名 TXT · 默认跳过已有文件</p>
          {!localBrowser && <p className="wd14-error">当前为远程页面；请在运行服务的电脑上通过回环地址打开此页面进行批量和模型安装。</p>}
          <form onSubmit={startBatch}>
            <label>服务端图片文件夹<input value={folder} onChange={(event) => setFolder(event.target.value)} placeholder={catalog?.batch_root || "本机绝对路径"} required disabled={!localBrowser} /></label>
            <label>TXT 输出文件夹（可留空，保持原目录）<input value={outputFolder} onChange={(event) => setOutputFolder(event.target.value)} placeholder="现有本机文件夹的绝对路径" disabled={!localBrowser} /></label>
            <div className="wd14-batch-options"><label>已有 TXT<select value={conflict} onChange={(event) => setConflict(event.target.value)} disabled={!localBrowser}><option value="ignore">跳过（默认）</option><option value="copy">覆盖</option><option value="append">追加</option><option value="prepend">前置</option></select></label><label className="wd14-check"><input type="checkbox" checked={recursive} onChange={(event) => setRecursive(event.target.checked)} disabled={!localBrowser} />包含子文件夹</label></div>
             <details className="wd14-safety"><summary>本机操作与目录安全说明</summary><p>仅限本机可信用户从 127.0.0.1 / localhost 操作。支持现有文件夹，默认指向程序输出目录；拒绝已识别的符号链接/联接、UNC/设备路径和部分敏感目录。无法识别映射网络盘，也不能阻止其他本机进程并发更换目录。最多 500 张；已有 TXT 默认跳过，覆盖须手动选择。四引擎模型仅从固定官方 manifest 下载，GPU 依赖仅为固定官方 wheel；不会执行下载的 Python。模型文件不会随源码发布。</p></details>
            <div className="wd14-actions"><button className="wd14-primary" disabled={busy || !installed || !runnable || !localBrowser} type="submit"><Play size={15} aria-hidden="true" />开始批量</button>{job && ["queued", "running"].includes(job.state) && localBrowser && <button type="button" onClick={() => cancelJob(job)}><Square size={14} aria-hidden="true" />取消任务</button>}</div>
          </form>
          {job && <div className="wd14-progress" role="status"><div><span>{job.state} · {job.current || "等待"}</span><b>{job.done} 成功 / {job.skipped} 跳过 / {job.failed} 失败 · {job.total} 总数</b></div><progress aria-label="批量任务进度" max={100} value={percent(job)} />{job.error && <p className="wd14-error">{job.error}</p>}<ul>{job.logs.map((line, index) => <li key={index}>{line}</li>)}</ul></div>}
        </section>
      </aside>
    </div>
  </main>;
}
