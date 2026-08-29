import { useEffect, useRef, useState } from "react";
import {
  ArrowLeft,
  Check,
  ChevronDown,
  Download,
  ExternalLink,
  Eye,
  EyeOff,
  FolderOpen,
  Key,
  RefreshCw,
  Sparkles,
  X,
} from "lucide-react";
import { formatFileSize } from "./format-size";

export default function ModelDownloader({ onDownloaded }) {
  const [civitaiKey, setCivitaiKey] = useState("");
  const [huggingfaceKey, setHuggingfaceKey] = useState("");
  const [modelscopeKey, setModelscopeKey] = useState("");
  const [keyManagerOpen, setKeyManagerOpen] = useState(false);
  const [visibleKeys, setVisibleKeys] = useState({ civitai: false, huggingface: false, modelscope: false });
  const keysLoaded = useRef(false);

  useEffect(() => {
    if (keysLoaded.current) return;
    keysLoaded.current = true;
    try {
      const saved = JSON.parse(window.localStorage.getItem("xirai_model_keys") || "{}");
      if (saved.c) setCivitaiKey(String(saved.c));
      if (saved.h) setHuggingfaceKey(String(saved.h));
      if (saved.m) setModelscopeKey(String(saved.m));
    } catch {}
  }, []);

  const saveKeys = () => {
    try {
      const payload = {};
      if (civitaiKey.trim()) payload.c = civitaiKey.trim();
      if (huggingfaceKey.trim()) payload.h = huggingfaceKey.trim();
      if (modelscopeKey.trim()) payload.m = modelscopeKey.trim();
      window.localStorage.setItem("xirai_model_keys", JSON.stringify(payload));
    } catch {}
    setKeyManagerOpen(false);
  };

  const toggleKeyVisibility = (site) => setVisibleKeys((current) => ({ ...current, [site]: !current[site] }));
  const [modelUrls, setModelUrls] = useState("");
  const [kind, setKind] = useState("checkpoint");
  const [engine, setEngine] = useState("SD");
  const [category, setCategory] = useState("character");
  const [modelPaths, setModelPaths] = useState(null);
  const [connections, setConnections] = useState(8);
  const [connectionDraft, setConnectionDraft] = useState(8);
  const [download, setDownload] = useState({ status: "checking", message: "正在恢复本地下载任务...", currentBytes: 0, totalBytes: 0, speedBps: 0, connections: 0, route: "" });
  const [submittingDownload, setSubmittingDownload] = useState(false);
  const [recommended, setRecommended] = useState({ loading: true, checkingInstalled: false, installedChecked: false, remoteRefreshed: false, families: [], error: "" });
  const recommendedRequestId = useRef(0);
  const completedDownloadJob = useRef("");
  const [recommendedZoneOpen, setRecommendedZoneOpen] = useState(false);
  const [recommendedCategoryId, setRecommendedCategoryId] = useState("");
  const [recommendedFamilyId, setRecommendedFamilyId] = useState("");
  const [recommendedSelection, setRecommendedSelection] = useState({ modelId: "", textEncoderId: "", vaeId: "" });

  const kindOptions = [
    { id: "checkpoint", label: "底模", detail: "Stable Diffusion / Illustrious" },
    { id: "diffusion_model", label: "扩散模型", detail: "Anima / Flux / Krea 2" },
    { id: "text_encoder", label: "文本编码器", detail: "独立 Text Encoder" },
    { id: "lora", label: "LoRA", detail: "按底模与分类保存" },
    { id: "vae", label: "VAE", detail: "独立 VAE 权重" },
    { id: "yolo", label: "YOLO", detail: "检测模型" },
    { id: "upscaler", label: "超分模型", detail: "Hires.fix / ESRGAN / SwinIR" },
    { id: "embedding", label: "Embedding", detail: "文本反演" },
    { id: "config", label: "配置", detail: "模型配置文件" },
  ];
  const selectedKind = kindOptions.find((item) => item.id === kind);
  const isDownloading = ["checking", "queued", "resolving", "downloading", "metadata"].includes(download.status);
  const modelLinks = modelUrls.split(/\r?\n/).map((value) => value.trim()).filter(Boolean);
  const requiresEngine = kind === "checkpoint" || kind === "lora";
  const progress = download.totalBytes > 0 ? Math.min(100, download.currentBytes / download.totalBytes * 100) : 0;
  const enginePathKey = engine === "Anima" ? "anima" : engine === "Flux" ? "flux" : engine === "Flux2" ? "flux2" : engine === "Krea2" ? "krea2" : engine === "iL" ? "illustrious" : "sd";
  const defaultTargetPaths = {
    checkpoint: `models/checkpoints/${enginePathKey}`,
    diffusion_model: "models/diffusion_models",
    text_encoder: "models/text_encoders",
    lora: `models/loras/${enginePathKey}`,
    yolo: "models/yolo",
    upscaler: "models/upscalers",
    vae: "models/vae",
    embedding: "models/embeddings",
    config: "models/configs",
  };
  const configuredTarget = kind === "checkpoint" ? modelPaths?.checkpoints?.[enginePathKey]
    : kind === "lora" ? modelPaths?.loras?.[enginePathKey]
      : modelPaths?.[{ diffusion_model: "diffusion_models", text_encoder: "text_encoders", embedding: "embeddings", config: "configs" }[kind] || kind];
  const targetLabel = `${configuredTarget || defaultTargetPaths[kind]}${kind === "lora" ? `/${category}` : ""}`;

  const refreshRecommendedInstallations = async (requestId, refreshRemote) => {
    setRecommended((current) => requestId === recommendedRequestId.current ? { ...current, checkingInstalled: true } : current);
    try {
      const response = await fetch(`/api/recommended-models?${refreshRemote ? "refresh=1&" : ""}installed=1`, { cache: "no-store" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "无法检查推荐模型安装状态");
      if (requestId === recommendedRequestId.current) {
        setRecommended((current) => ({ ...current, checkingInstalled: false, installedChecked: true, remoteRefreshed: current.remoteRefreshed || payload.remote_refreshed === true, families: payload.families || [] }));
      }
    } catch {
      if (requestId === recommendedRequestId.current) {
        setRecommended((current) => ({ ...current, checkingInstalled: false, installedChecked: true }));
      }
    }
  };

  const refreshRecommended = async (force = false) => {
    const requestId = recommendedRequestId.current + 1;
    recommendedRequestId.current = requestId;
    setRecommended((current) => ({ ...current, loading: true, error: "" }));
    try {
      const response = await fetch(`/api/recommended-models${force ? "?refresh=1" : ""}`, { cache: "no-store" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "无法读取推荐模型目录");
      if (requestId !== recommendedRequestId.current) return;
      setRecommended({ loading: false, checkingInstalled: false, installedChecked: false, remoteRefreshed: payload.remote_refreshed === true, families: payload.families || [], error: "" });
    } catch (error) {
      if (requestId === recommendedRequestId.current) {
        setRecommended((current) => ({ ...current, loading: false, checkingInstalled: false, installedChecked: true, error: error.message }));
      }
    }
  };

  useEffect(() => {
    void refreshRecommended();
    fetch("/api/model-paths", { cache: "no-store" })
      .then(async (response) => {
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "无法读取模型目录配置");
        setModelPaths(payload.paths || null);
      })
      .catch(() => setModelPaths(null));
  }, []);

  useEffect(() => {
    if (!recommendedZoneOpen || recommended.loading || recommended.checkingInstalled || recommended.installedChecked || !recommended.families.length) return;
    void refreshRecommendedInstallations(recommendedRequestId.current, !recommended.remoteRefreshed);
  }, [recommendedZoneOpen, recommended.loading, recommended.checkingInstalled, recommended.installedChecked, recommended.remoteRefreshed, recommended.families.length]);

  useEffect(() => {
    if (!recommendedZoneOpen && !recommendedFamilyId) return undefined;
    const close = (event) => {
      if (event.key !== "Escape") return;
      if (recommendedFamilyId) setRecommendedFamilyId("");
      else if (recommendedCategoryId) setRecommendedCategoryId("");
      else setRecommendedZoneOpen(false);
    };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [recommendedZoneOpen, recommendedCategoryId, recommendedFamilyId]);

  const recommendedCategories = [
    { id: "sd", badge: "SD", label: "Stable Diffusion", detail: "SD 1.x、SDXL 与通用 Stable Diffusion 底模", groups: ["Stable Diffusion", "SD"] },
    { id: "illustrious", badge: "iL", label: "Illustrious", detail: "Illustrious / SDXL 动漫底模", groups: ["Illustrious"] },
    { id: "anima", badge: "AN", label: "Anima", detail: "Anima 官方模型、衍生模型与共享组件", groups: ["Anima"] },
    { id: "flux", badge: "FX", label: "Flux", detail: "FLUX.2 Klein 系列与量化版本", groups: ["Flux"] },
    { id: "krea-2", badge: "K2", label: "Krea 2", detail: "Krea 2 Raw / Turbo 与文本编码器", groups: ["Krea 2"] },
    { id: "upscaler", badge: "4X", label: "超分模型", detail: "Hires.fix 像素预放大模型", groups: ["Upscaler"] },
  ].map((category) => ({ ...category, families: recommended.families.filter((family) => category.groups.includes(family.group)) }));
  const selectedRecommendedCategory = recommendedCategories.find((category) => category.id === recommendedCategoryId);
  const recommendedFamilyCount = recommended.families.length;
  const recommendedVersionCount = recommended.families.reduce((total, family) => total + family.models.length, 0);

  const selectedRecommendedFamily = recommended.families.find((family) => family.id === recommendedFamilyId);
  const selectedRecommendedModel = selectedRecommendedFamily?.models.find((item) => item.id === recommendedSelection.modelId);
  const selectedRecommendedArtifacts = selectedRecommendedFamily ? [
    selectedRecommendedModel,
    selectedRecommendedFamily.textEncoders.find((item) => item.id === recommendedSelection.textEncoderId),
    selectedRecommendedFamily.vaes.find((item) => item.id === recommendedSelection.vaeId),
  ].filter(Boolean) : [];
  const queuedRecommendedIds = new Set((download.items || []).filter((item) => item.status !== "complete").map((item) => item.artifactId).filter(Boolean));
  const queuedRecommendedDigests = new Set((download.items || []).filter((item) => item.status !== "complete").map((item) => item.sha256).filter(Boolean));
  const selectedArtifactIsQueued = (item) => queuedRecommendedIds.has(item.id) || queuedRecommendedDigests.has(item.sha256);
  const pendingRecommendedArtifacts = selectedRecommendedArtifacts.filter((item) => !item.installed && !selectedArtifactIsQueued(item));
  const installedSelectedArtifactCount = selectedRecommendedArtifacts.filter((item) => item.installed).length;
  const queuedSelectedArtifactCount = selectedRecommendedArtifacts.filter((item) => !item.installed && selectedArtifactIsQueued(item)).length;
  const installedRecommendedArtifacts = recommended.families.flatMap((family) => [
    ...family.models,
    ...family.textEncoders,
    ...family.vaes,
  ]).filter((item) => item.installed);
  const installedRecommendedLabels = new Set(installedRecommendedArtifacts.map((item) => item.label));
  const installedRecommendedDigests = new Set(installedRecommendedArtifacts.map((item) => item.sha256).filter(Boolean));
  const visibleDownloadItems = (download.items || []).filter((item) => (item.source || download.source) !== "recommended" || item.status !== "waiting" || !(installedRecommendedDigests.has(item.sha256) || installedRecommendedLabels.has(item.url)));
  const recommendedDirectoryForRole = (role) => {
    if (role === "checkpoint") return modelPaths?.checkpoints?.illustrious || "models/checkpoints/illustrious";
    const pathKey = { diffusion_model: "diffusion_models", text_encoder: "text_encoders", config: "configs" }[role] || role;
    return modelPaths?.[pathKey] || defaultTargetPaths[role] || `models/${pathKey}`;
  };
  const selectedRecommendedDirectories = [...new Set(selectedRecommendedArtifacts.map((artifact) => recommendedDirectoryForRole(artifact.role)))];
  const selectedRequiresCivitaiKey = Boolean(selectedRecommendedFamily?.requiresCivitaiKey && pendingRecommendedArtifacts.includes(selectedRecommendedModel));
  const selectedRequiresHuggingfaceKey = Boolean((selectedRecommendedFamily?.requiresHuggingfaceKey && pendingRecommendedArtifacts.includes(selectedRecommendedModel)) || pendingRecommendedArtifacts.some((item) => item.requiresHuggingfaceKey));

  const openRecommendedFamily = (family) => {
    const familyCategory = recommendedCategories.find((item) => item.groups.includes(family.group));
    if (familyCategory) setRecommendedCategoryId(familyCategory.id);
    setRecommendedSelection({
      modelId: family.models[0]?.id || "",
      textEncoderId: family.textEncoders[0]?.id || "",
      vaeId: family.vaes[0]?.id || "",
    });
    setRecommendedFamilyId(family.id);
  };

  const syncModelDownloadJob = async () => {
    const response = await fetch("/api/model-download/job", { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "无法读取模型下载状态");
    if (payload.job) setDownload(payload.job);
    else setDownload({ status: "idle", message: "粘贴模型链接后开始解析", currentBytes: 0, totalBytes: 0, speedBps: 0, connections: 0, route: "" });
    return payload.job;
  };

  useEffect(() => {
    let stopped = false;
    let timer;
    const poll = async () => {
      try {
        const job = await syncModelDownloadJob();
        if (stopped) return;
        timer = window.setTimeout(poll, job?.active ? 500 : 5000);
      } catch {
        if (!stopped) timer = window.setTimeout(poll, 5000);
      }
    };
    void poll();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, [download.active]);

  useEffect(() => {
    const completionKey = `${download.jobId}:${download.completedModels}:${download.failedModels}`;
    if (!download.jobId || download.active || !["complete", "partial"].includes(download.status) || !download.completedModels || completedDownloadJob.current === completionKey) return;
    completedDownloadJob.current = completionKey;
    onDownloaded({ kind: download.kind, engine: download.engine || undefined, targets: download.targets || [] });
    void refreshRecommended();
  }, [download.jobId, download.active, download.status, download.completedModels, download.failedModels, download.updatedAt, download.kind, download.engine, download.targets, onDownloaded]);

  const runDownload = async ({ endpoint, body, items, accepted }) => {
    if (submittingDownload || !items.length) return;
    setSubmittingDownload(true);
    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.error || "无法启动模型下载");
      }
      const payload = await response.json();
      if (!payload.job) throw new Error("本地下载服务没有返回任务状态");
      setDownload(payload.job);
      accepted?.(payload);
    } catch (error) {
      setDownload((current) => current.active
        ? { ...current, warning: `添加队列失败：${error.message}`, message: current.message }
        : { ...current, status: "error", message: error.message });
    } finally {
      setSubmittingDownload(false);
    }
  };

  const startDownload = (event) => {
    event.preventDefault();
    void runDownload({
      endpoint: "/api/model-download",
      body: { urls: modelLinks, civitai_key: civitaiKey, huggingface_key: huggingfaceKey, modelscope_key: modelscopeKey, kind, engine: requiresEngine ? engine : undefined, category: kind === "lora" ? category : undefined, connections },
      items: modelLinks,
      accepted: () => setModelUrls(""),
    });
  };

  const startRecommendedDownload = () => {
    if (!selectedRecommendedFamily || !recommendedSelection.modelId || recommended.checkingInstalled || !recommended.installedChecked || !pendingRecommendedArtifacts.length) return;
    void runDownload({
      endpoint: "/api/recommended-download",
      body: { family_id: selectedRecommendedFamily.id, model_id: recommendedSelection.modelId, text_encoder_id: recommendedSelection.textEncoderId, vae_id: recommendedSelection.vaeId, civitai_key: civitaiKey, huggingface_key: huggingfaceKey, connections },
      items: pendingRecommendedArtifacts.map((item) => item.label),
      accepted: () => { setRecommendedFamilyId(""); setRecommendedZoneOpen(false); void refreshRecommended(); },
    });
  };

  const retryFailedDownloads = () => {
    if (submittingDownload || !download.failedModels) return;
    void runDownload({
      endpoint: "/api/model-download/retry",
      body: { civitai_key: civitaiKey, huggingface_key: huggingfaceKey, modelscope_key: modelscopeKey, connections },
      items: (download.items || []).filter((item) => item.status === "error").map((item) => item.url),
    });
  };

  return <section className="model-downloader-page">
    <header className="downloader-hero">
      <div><span className="eyebrow">LOCAL MODEL ACQUISITION</span><p>从 Civitai、Hugging Face 或 ModelScope 解析模型文件，并根据线路测速自动选择可用来源进行多线程续传。</p></div>
      <div className="downloader-source-list"><span>Civitai</span><span>Hugging Face</span><span>ModelScope</span><span>GitHub Release</span></div>
    </header>
    <form className="downloader-form" onSubmit={startDownload}>
      <section className="downloader-card credentials-card">
        <div className="downloader-section-head"><span>01</span><div><strong>API Key 管理</strong><small>可选，为 Civitai、Hugging Face 或 ModelScope 分别配置密钥</small></div></div>
        <button type="button" className="key-manager-trigger" onClick={() => setKeyManagerOpen(true)}>
          <Key size={15} />
          <span>管理 API 密钥<small>{[civitaiKey, huggingfaceKey, modelscopeKey].filter(Boolean).length ? `${[civitaiKey, huggingfaceKey, modelscopeKey].filter(Boolean).length} 个已配置` : "未配置"}</small></span>
          <ChevronDown size={14} />
        </button>
        <p className="downloader-security-note"><i />密钥保存在当前浏览器本地存储，下载时仅发送给对应站点，不会写入模型目录或运行日志。</p>
      </section>
      <section className="downloader-card link-card">
        <div className="downloader-section-head"><span>02</span><div><strong>模型链接</strong><small>优先粘贴文件详情页或直接下载链接；仓库存在多个文件时需提供具体文件链接</small></div></div>
        <label className="model-link-field"><span>MODEL URLS · 每行一个链接</span><textarea value={modelUrls} onChange={(event) => setModelUrls(event.target.value)} placeholder={"https://civitai.com/models/...\nhttps://huggingface.co/owner/repository/blob/main/model.safetensors\nhttps://modelscope.cn/models/..."} required spellCheck="false" /></label>
      </section>
      <section className="downloader-card recommended-card">
        <div className="downloader-section-head"><span>04</span><div><strong>推荐模型专区</strong><small>按类别浏览精选模型、版本和配套组件</small></div></div>
        <button className="recommended-zone-trigger" type="button" onClick={() => { setRecommendedCategoryId(""); setRecommendedFamilyId(""); setRecommendedZoneOpen(true); }}>
          <span><Sparkles size={18} /><i /></span>
          <div><strong>打开推荐模型专区</strong><small>SD · iL · Anima · Flux · Krea 2 · 超分</small></div>
          <b>{recommended.loading ? "正在检索" : `${recommendedFamilyCount} 个系列 · ${recommendedVersionCount} 个版本`}<ChevronDown size={13} /></b>
        </button>
        {recommended.error && <p className="recommended-model-error">{recommended.error}</p>}
      </section>
      <section className="downloader-card type-card">
        <div className="downloader-section-head"><span>03</span><div><strong>保存类型</strong><small>类型决定可用格式和本地模型目录，不能由链接覆盖</small></div></div>
        <div className="download-kind-grid" role="radiogroup" aria-label="模型类型">{kindOptions.map((item) => <button key={item.id} type="button" role="radio" aria-checked={kind === item.id} className={kind === item.id ? "active" : ""} onClick={() => { setKind(item.id); if (item.id !== "lora" && ["Anima", "Flux", "Flux2"].includes(engine)) setEngine("SD"); }}><strong>{item.label}</strong><small>{item.detail}</small></button>)}</div>
        <div className="download-options">
          {requiresEngine && <label>兼容底模<select value={engine} onChange={(event) => setEngine(event.target.value)}><option value="SD">Stable Diffusion</option><option value="iL">Illustrious / SDXL</option>{kind === "lora" && <option value="Anima">Anima</option>}{kind === "lora" && <option value="Flux">Flux</option>}{kind === "lora" && <option value="Flux2">Flux 2</option>}{kind === "lora" && <option value="Krea2">Krea 2</option>}</select></label>}
          {kind === "lora" && <label>LoRA 分类<select value={category} onChange={(event) => setCategory(event.target.value)}><option value="character">角色 Character</option><option value="style">风格 Style</option><option value="concept">概念 Concept</option><option value="other">其他 Other</option></select></label>}
          <div className="download-connection-setting">
            <span>下载线程</span>
            <select value={connectionDraft} onChange={(event) => setConnectionDraft(Number(event.target.value))}>{[1, 2, 4, 8, 12, 16, 24, 32].map((value) => <option value={value} key={value}>{value} 路</option>)}</select>
            <button type="button" disabled={connectionDraft === connections} onClick={() => setConnections(connectionDraft)}>{connectionDraft === connections ? <><Check size={12} />已应用</> : "确认应用"}</button>
            <small>当前生效 {connections} 路 · 最大 32 路</small>
          </div>
        </div>
        <div className="download-destination"><FolderOpen size={15} /><span>下载位置</span><code>{targetLabel}</code>{selectedKind && <small>{selectedKind.label}</small>}</div>
      </section>
      <section className={`download-progress-card ${download.status}`} aria-live="polite">
        <div className="download-progress-head"><div><span>{download.status === "complete" ? "READY" : download.status === "error" ? "DOWNLOAD ERROR" : download.status === "idle" ? "WAITING" : download.provider?.toUpperCase() || "PREPARING"}</span><strong>{download.filename || "等待模型链接"}</strong><small>{download.destination || "选择类型后自动确定本地保存位置"}</small></div>{isDownloading && <RefreshCw className="spin" size={19} />}{download.status === "complete" && <Check size={19} />}{download.status === "error" && <X size={19} />}</div>
        <div className="download-progress-track"><i style={{ width: `${progress}%` }} /></div>
        <div className="download-progress-meta"><span>{download.message}</span><b>{download.totalBytes ? `${formatFileSize(download.currentBytes)} / ${formatFileSize(download.totalBytes)}` : download.verified ? "SHA-256 将校验" : "等待远程文件信息"}</b></div>
        {(download.speedBps > 0 || download.connections > 0 || download.route) && <div className="download-transfer-meta"><span>{download.route || "正在测速"}</span>{download.speedBps > 0 && <span>{formatFileSize(download.speedBps)}/s</span>}{download.connections > 1 && <span>{download.connections} 路分片</span>}</div>}
        {download.warning && <p className="download-warning">{download.warning}</p>}
      </section>
      {visibleDownloadItems.length > 1 && <div className="download-batch-list">{visibleDownloadItems.map((item) => <div key={item.index} className={item.status}><b>{String(item.index).padStart(2, "0")}</b><span>{item.filename || item.url}</span><small>{item.status === "complete" ? item.destination : item.error || (item.status === "waiting" ? "等待队列" : item.status === "metadata" ? "正在同步 LoRA 信息" : "正在处理")}</small></div>)}</div>}
      <footer className="downloader-actions"><span>{isDownloading ? "当前下载继续运行，可追加链接或推荐模型" : modelLinks.length > 1 ? `${modelLinks.length} 个链接将按顺序下载并自动归类` : "自动测速 · 镜像回退 · 断点续传"} · 单文件最大 128 GB · 最多 32 路</span><div className="downloader-action-buttons">{download.failedModels > 0 && download.retryAvailable && <button type="button" className="secondary" disabled={submittingDownload} onClick={retryFailedDownloads}><RefreshCw className={submittingDownload ? "spin" : ""} size={16} />重试失败项（{download.failedModels}）</button>}<button type="submit" disabled={modelLinks.length === 0 || submittingDownload}>{submittingDownload ? <><RefreshCw className="spin" size={16} />正在提交</> : <><Download size={16} />{isDownloading ? "添加到队列" : "开始下载"}{modelLinks.length > 1 ? `（${modelLinks.length}）` : ""}</>}</button></div></footer>
    </form>
    {recommendedZoneOpen && <div className="recommended-zone-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) { setRecommendedZoneOpen(false); setRecommendedFamilyId(""); setRecommendedCategoryId(""); } }}>
      <section className="recommended-zone-dialog" role="dialog" aria-modal="true" aria-label="推荐模型专区">
        <header className="recommended-zone-head">
          <div>
            <span className="eyebrow">RECOMMENDED MODEL ZONE</span>
            <h2>推荐模型专区</h2>
            <p>按类别浏览推荐模型，查看模型介绍、可下载版本、来源和配套组件。</p>
          </div>
          <div className="recommended-zone-head-actions"><button type="button" title="刷新推荐模型版本" disabled={recommended.loading} onClick={() => void refreshRecommended(true)}><RefreshCw className={recommended.loading ? "spin" : ""} size={17} /></button><button type="button" className="modal-close" onClick={() => { setRecommendedZoneOpen(false); setRecommendedFamilyId(""); setRecommendedCategoryId(""); }}><X size={21} /></button></div>
        </header>
        <div className="recommended-zone-layout">
          <nav className="recommended-zone-nav" aria-label="推荐模型类别">
            <button type="button" className={!selectedRecommendedCategory ? "active" : ""} onClick={() => { setRecommendedCategoryId(""); setRecommendedFamilyId(""); }}><span>全部类别</span><b>{recommendedCategories.length}</b></button>
            <span>模型类别</span>
            {recommendedCategories.map((category) => <button type="button" className={selectedRecommendedCategory?.id === category.id ? "active" : ""} key={category.id} onClick={() => { setRecommendedCategoryId(category.id); setRecommendedFamilyId(""); }}><span>{category.label}</span><b>{category.families.length}</b></button>)}
            <div className="recommended-zone-nav-summary"><span>模型目录</span><p>{recommended.loading ? "正在读取本地目录..." : recommended.checkingInstalled ? `${recommendedFamilyCount} 个系列\n后台核对版本与安装状态...` : `${recommendedFamilyCount} 个系列\n${recommendedVersionCount} 个可选版本`}</p></div>
          </nav>
          <div className="recommended-zone-browser">
            <header className="recommended-zone-browser-head">
              <div>
                {selectedRecommendedFamily && <button type="button" onClick={() => setRecommendedFamilyId("")}><ArrowLeft size={15} />返回 {selectedRecommendedCategory?.label || "模型列表"}</button>}
                <h3>{selectedRecommendedFamily ? `${selectedRecommendedCategory?.label || selectedRecommendedFamily.group} · 模型详情` : selectedRecommendedCategory?.label || "全部模型类别"}</h3>
                <p>{selectedRecommendedFamily ? "查看模型信息，选择需要下载的版本和配套组件。" : selectedRecommendedCategory?.detail || "选择左侧类别，查看对应推荐模型；点击模型卡片进入介绍和下载页面。"}</p>
              </div>
              {selectedRecommendedFamily && <a href={selectedRecommendedFamily.sourceUrl} target="_blank" rel="noreferrer"><ExternalLink size={15} />模型主页</a>}
            </header>
            <div className="recommended-zone-content">
              {recommended.error && <p className="recommended-zone-error">{recommended.error}</p>}
              {selectedRecommendedFamily ? <>
                <section className="recommended-family-intro">
                  <div className="recommended-family-mark"><span>{selectedRecommendedCategory?.badge || "AI"}</span><small>{selectedRecommendedFamily.group}</small></div>
                  <div className="recommended-family-copy"><span>{selectedRecommendedFamily.provider}</span><h4>{selectedRecommendedFamily.name}</h4><p>{selectedRecommendedFamily.description}</p><div><span><b>{selectedRecommendedFamily.models.length}</b><small>可下载版本</small></span><span><b>{selectedRecommendedFamily.textEncoders.length}</b><small>文本编码器</small></span><span><b>{selectedRecommendedFamily.vaes.length}</b><small>VAE 选项</small></span></div></div>
                  <aside><span>保存目录</span><code>{selectedRecommendedDirectories.join(" · ")}</code><small>Anima tokenizer 已随项目内置，不进入模型下载队列。</small></aside>
                </section>
                <div className="recommended-picker-notices">
                  {selectedRequiresCivitaiKey && <div className="recommended-civitai-warning"><Key size={16} /><span><strong>Civitai 下载提示</strong>需要开启可访问 Civitai 的 VPN 代理，并填写对应 API Key。</span><button type="button" onClick={() => setKeyManagerOpen(true)}>{civitaiKey ? "API Key 已配置" : "填写 API Key"}</button></div>}
                  {selectedRequiresHuggingfaceKey && <div className="recommended-civitai-warning"><Key size={16} /><span><strong>Hugging Face 授权仓库</strong>请先在模型主页接受许可协议，再填写 Hugging Face Token。</span><button type="button" onClick={() => setKeyManagerOpen(true)}>{huggingfaceKey ? "Token 已配置" : "填写 Token"}</button></div>}
                </div>
                <section className="recommended-artifact-section"><header><div><strong>可下载模型</strong><small>选择一个模型版本或精度</small></div><b>{selectedRecommendedFamily.models.length} OPTIONS</b></header><div className="recommended-artifact-grid">
                  {selectedRecommendedFamily.models.map((item) => <label className={`recommended-artifact-card ${recommendedSelection.modelId === item.id ? "selected" : ""} ${item.installed ? "installed" : ""}`} key={item.id}><input type="radio" name="recommended-model" value={item.id} checked={recommendedSelection.modelId === item.id} onChange={() => setRecommendedSelection((current) => ({ ...current, modelId: item.id }))} /><span className="recommended-artifact-radio"><i /></span><div><span>{item.provider || selectedRecommendedFamily.provider}</span><strong>{item.label}</strong><small>{item.filename || `Civitai Version ${item.versionId}`}</small>{item.detail && <p>{item.detail}</p>}</div><b>{item.installed ? <><Check size={13} />已下载</> : formatFileSize(item.size)}</b></label>)}
                </div></section>
                {selectedRecommendedFamily.textEncoders.length > 0 && <section className="recommended-artifact-section"><header><div><strong>Text Encoders</strong><small>选择一个兼容文本编码器</small></div><b>{selectedRecommendedFamily.textEncoders.length} OPTIONS</b></header><div className="recommended-artifact-grid dependencies">
                  {selectedRecommendedFamily.textEncoders.map((item) => <label className={`recommended-artifact-card ${recommendedSelection.textEncoderId === item.id ? "selected" : ""} ${item.installed ? "installed" : ""}`} key={item.id}><input type="radio" name="recommended-encoder" value={item.id} checked={recommendedSelection.textEncoderId === item.id} onChange={() => setRecommendedSelection((current) => ({ ...current, textEncoderId: item.id }))} /><span className="recommended-artifact-radio"><i /></span><div><span>TEXT ENCODER</span><strong>{item.label}</strong><small>{item.filename}</small>{item.detail && <p>{item.detail}</p>}</div><b>{item.installed ? <><Check size={13} />已下载</> : formatFileSize(item.size)}</b></label>)}
                </div></section>}
                {selectedRecommendedFamily.vaes.length > 0 && <section className="recommended-artifact-section"><header><div><strong>VAE</strong><small>共享 VAE 会按 SHA-256 检测和复用</small></div><b>{selectedRecommendedFamily.vaes.length} OPTIONS</b></header><div className="recommended-artifact-grid dependencies">
                  {selectedRecommendedFamily.vaes.map((item) => <label className={`recommended-artifact-card ${recommendedSelection.vaeId === item.id ? "selected" : ""} ${item.installed ? "installed" : ""}`} key={item.id}><input type="radio" name="recommended-vae" value={item.id} checked={recommendedSelection.vaeId === item.id} onChange={() => setRecommendedSelection((current) => ({ ...current, vaeId: item.id }))} /><span className="recommended-artifact-radio"><i /></span><div><span>VAE</span><strong>{item.label}</strong><small>{item.filename}</small>{item.detail && <p>{item.detail}</p>}</div><b>{item.installed ? <><Check size={13} />已下载</> : formatFileSize(item.size)}</b></label>)}
                </div></section>}
              </> : !selectedRecommendedCategory ? <div className="recommended-category-grid">
                {recommendedCategories.map((category) => {
                  const versions = category.families.reduce((total, family) => total + family.models.length, 0);
                  const installed = category.families.reduce((total, family) => total + [...family.models, ...family.textEncoders, ...family.vaes].filter((item) => item.installed).length, 0);
                  return <button type="button" key={category.id} onClick={() => setRecommendedCategoryId(category.id)}><span>{category.badge}</span><div><strong>{category.label}</strong><p>{category.detail}</p><small>{category.families.length ? `${category.families.length} 个系列 · ${versions} 个版本` : "暂未收录推荐模型"}</small></div><b>{installed ? `已下载 ${installed}` : "查看模型"}<ChevronDown size={14} /></b></button>;
                })}
              </div> : selectedRecommendedCategory.families.length ? <div className="recommended-zone-family-list">
                {selectedRecommendedCategory.families.map((family) => {
                  const installedModels = family.models.filter((item) => item.installed).length;
                  const dependencyCount = family.textEncoders.length + family.vaes.length;
                  const installedDependencies = [...family.textEncoders, ...family.vaes].filter((item) => item.installed).length;
                  return <button type="button" key={family.id} onClick={() => openRecommendedFamily(family)}><div className="recommended-model-card-visual"><span>{selectedRecommendedCategory.badge}</span><small>{family.provider}</small><i /></div><div className="recommended-model-card-copy"><span>{family.group}</span><strong>{family.name}</strong><p>{family.description}</p><div><small>{family.models.length} 个版本{dependencyCount ? ` · ${dependencyCount} 类组件` : ""}</small><b>{installedModels || installedDependencies ? `已下载 ${installedModels + installedDependencies}` : "查看介绍与下载"}<ChevronDown size={13} /></b></div></div></button>;
                })}
              </div> : <div className="recommended-zone-empty"><span>{selectedRecommendedCategory.badge}</span><strong>{selectedRecommendedCategory.label} 推荐模型待补充</strong><p>当前类别尚未收录推荐模型，后续添加后会直接显示在这里。</p></div>}
            </div>
          </div>
        </div>
        <footer className="recommended-zone-foot">
          {selectedRecommendedFamily ? <><span><FolderOpen size={14} />{recommended.checkingInstalled || !recommended.installedChecked ? "正在校验所选文件" : pendingRecommendedArtifacts.length ? `待下载 ${pendingRecommendedArtifacts.map((item) => item.label).join(" + ")} · ${formatFileSize(pendingRecommendedArtifacts.reduce((total, item) => total + (item.size || 0), 0))}${installedSelectedArtifactCount ? ` · 本地已有 ${installedSelectedArtifactCount} 项` : ""}${queuedSelectedArtifactCount ? ` · 队列已有 ${queuedSelectedArtifactCount} 项` : ""}` : queuedSelectedArtifactCount ? `所选缺失资源已有 ${queuedSelectedArtifactCount} 项在队列中` : `所选 ${installedSelectedArtifactCount} 项均已通过本地校验`}</span><button type="button" disabled={submittingDownload || recommended.checkingInstalled || !recommended.installedChecked || !recommendedSelection.modelId || !pendingRecommendedArtifacts.length || (selectedRequiresCivitaiKey && !civitaiKey) || (selectedRequiresHuggingfaceKey && !huggingfaceKey)} onClick={startRecommendedDownload}><Download size={16} />{submittingDownload ? "正在添加到队列" : recommended.checkingInstalled || !recommended.installedChecked ? "正在校验本地文件" : !pendingRecommendedArtifacts.length ? queuedSelectedArtifactCount ? "已在下载队列" : "已全部安装" : selectedRequiresCivitaiKey && !civitaiKey ? "请先填写 API Key" : selectedRequiresHuggingfaceKey && !huggingfaceKey ? "请先填写 HF Token" : isDownloading ? `添加缺失项（${pendingRecommendedArtifacts.length}）` : `下载缺失项（${pendingRecommendedArtifacts.length}）`}</button></> : <><span>{recommended.loading ? "正在读取本地推荐模型目录" : recommended.checkingInstalled ? `${recommendedFamilyCount} 个模型系列已就绪 · 正在后台核对版本与安装状态` : `${recommendedFamilyCount} 个模型系列 · ${recommendedVersionCount} 个可选版本 · 已下载状态按 SHA-256 检测`}</span><button type="button" className="secondary" onClick={() => { setRecommendedZoneOpen(false); setRecommendedFamilyId(""); setRecommendedCategoryId(""); }}>关闭专区</button></>}
        </footer>
      </section>
    </div>}
    {keyManagerOpen && <div className="key-manager-backdrop" onMouseDown={(event) => event.target === event.currentTarget && setKeyManagerOpen(false)}>
      <section className="key-manager-dialog" role="dialog" aria-modal="true" aria-label="API 密钥管理">
        <header className="key-manager-head">
          <div><span className="eyebrow">API KEY MANAGER</span><h2>API 密钥管理</h2><p>分别为三个站点填写密钥后可保存；留空表示不发送认证头。</p></div>
          <button className="modal-close" onClick={() => setKeyManagerOpen(false)}><X size={20} /></button>
        </header>
        <div className="key-manager-body">
          <div className="key-site-fields">
            <label className="key-site-field"><span>Civitai API Key</span><div className="key-site-input"><input type={visibleKeys.civitai ? "text" : "password"} value={civitaiKey} onChange={(event) => setCivitaiKey(event.target.value)} placeholder="适用于 civitai.com · 授权后可下载私密模型" autoComplete="off" spellCheck="false" /><button type="button" onClick={() => toggleKeyVisibility("civitai")}>{visibleKeys.civitai ? <EyeOff size={15} /> : <Eye size={15} />}</button></div></label>
            <label className="key-site-field"><span>Hugging Face Token</span><div className="key-site-input"><input type={visibleKeys.huggingface ? "text" : "password"} value={huggingfaceKey} onChange={(event) => setHuggingfaceKey(event.target.value)} placeholder="格式 hf_... · 用于私有仓库或限速模型下载" autoComplete="off" spellCheck="false" /><button type="button" onClick={() => toggleKeyVisibility("huggingface")}>{visibleKeys.huggingface ? <EyeOff size={15} /> : <Eye size={15} />}</button></div></label>
            <label className="key-site-field"><span>ModelScope Token</span><div className="key-site-input"><input type={visibleKeys.modelscope ? "text" : "password"} value={modelscopeKey} onChange={(event) => setModelscopeKey(event.target.value)} placeholder="适用于 modelscope.cn · 授权后可访问私有模型" autoComplete="off" spellCheck="false" /><button type="button" onClick={() => toggleKeyVisibility("modelscope")}>{visibleKeys.modelscope ? <EyeOff size={15} /> : <Eye size={15} />}</button></div></label>
          </div>
        </div>
        <footer className="key-manager-foot">
          <span>密钥仅通过 local storage 持久化，与每次下载请求一同发往对应站点后即丢弃，绝不写入工作区或日志文件。</span>
          <button type="button" className="key-manager-save" onClick={saveKeys}><Check size={15} />保存</button>
        </footer>
      </section>
    </div>}
  </section>;
}
