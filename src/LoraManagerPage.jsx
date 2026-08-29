import { useEffect, useRef, useState } from "react";
import { gsap } from "gsap";
import {
  BarChart3,
  Check,
  ChevronDown,
  Database,
  FolderOpen,
  FileText,
  HardDrive,
  Layers3,
  Plus,
  RefreshCw,
  Search,
  Share2,
  X,
} from "lucide-react";
import { LORA_SYNC_CHANNEL, normalizeMountedLoras, sameMountedLoras } from "./lora-state";
import {
  applyMountedLoraSync,
  emptyMountedLoraMap,
   engineScopeKey,
  mountedLorasForScope,
  normalizeMountedLoraMap,
  withMountedLorasForScope,
} from "./lora-model-scope";
import LoraDetailsDialog from "./LoraDetailsDialog";
import LoraMountPanel from "./LoraMountPanel.jsx";
import LoraGroupPanel from "./LoraGroupPanel.jsx";
import {
  emptyLoraGroupMap,
  loraGroupsForScope,
  mountedEntryForGroups,
  syncMountedIntoGroups,
  normalizeLoraGroupMap,
  withLoraGroupsForScope,
  updateLoraGroupsForScope,
} from "./lora-groups";
import { pageLoraDragLocked } from "./lora-drag-handle";
import { createLoraPersistenceEpochGuard, runLoraPersistenceEpoch } from "./lora-persistence-epoch";
import { applyThemeToDocument, loadThemeState } from "./theme.js";

const ENGINES = ["SD", "iL", "Anima"];
const CHART_COLORS = ["var(--lime)", "#62d0ff", "#ffb55e", "#ff7548", "#8f9389"];
const DONUT_CIRCUMFERENCE = 2 * Math.PI * 48;

function formatBytes(value) {
  if (!Number.isFinite(value) || value <= 0) return "0 B";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  const index = Math.min(units.length - 1, Math.floor(Math.log(value) / Math.log(1024)));
  return `${(value / 1024 ** index).toFixed(index < 2 ? 0 : 2)} ${units[index]}`;
}

function applySavedTheme(theme) {
  applyThemeToDocument(loadThemeState(theme));
}

function analyticsFor(libraries) {
  const files = ENGINES.flatMap((engine) => (libraries[engine]?.categories || []).flatMap((category) => category.models.map((model) => ({ ...model, engine, categoryId: category.id, categoryLabel: category.label, directory: category.directory }))));
  const totalBytes = files.reduce((total, file) => total + (file.size || 0), 0);
  const categoryMap = new Map();
  for (const file of files) {
    const current = categoryMap.get(file.categoryId) || { id: file.categoryId, label: file.categoryLabel, count: 0, bytes: 0 };
    current.count += 1;
    current.bytes += file.size || 0;
    categoryMap.set(file.categoryId, current);
  }
  const categories = [...categoryMap.values()].sort((first, second) => second.bytes - first.bytes);
  const engines = ENGINES.map((engine) => {
    const engineFiles = files.filter((file) => file.engine === engine);
    return { engine, count: engineFiles.length, bytes: engineFiles.reduce((total, file) => total + (file.size || 0), 0) };
  });
  const folders = ENGINES.flatMap((engine) => (libraries[engine]?.categories || []).map((category) => ({ engine, id: category.id, label: category.label, directory: category.directory, count: category.models.length, bytes: category.models.reduce((total, file) => total + (file.size || 0), 0) })));
  return {
    files,
    totalBytes,
    categories,
    engines,
    folders,
    largest: [...files].sort((first, second) => (second.size || 0) - (first.size || 0)).slice(0, 8),
  };
}

export default function LoraManagerPage() {
  const pageRef = useRef(null);
  const channelRef = useRef(null);
  const localChangeRef = useRef(false);
  const initializedRef = useRef(false);
  const modelRef = useRef("SD");
  const lorasRef = useRef([]);
  const scopeKeyRef = useRef(null);
  const mountedLorasMapRef = useRef(emptyMountedLoraMap());
  const loraGroupsMapRef = useRef(emptyLoraGroupMap());
  const activeLoraGroupsRef = useRef([]);
  const canPersistRef = useRef(true);
  const syncReadyRef = useRef(false);
  const pendingScopedSyncRef = useRef(new Map());
  const workspaceLockedRef = useRef(true);
  const libraryScanTokenRef = useRef(0);
  const persistenceEpochRef = useRef(createLoraPersistenceEpochGuard());
  const [model, setModel] = useState("SD");
  const [loras, setLoras] = useState([]);
  const [mountedLorasByEngine, setMountedLorasByEngine] = useState(emptyMountedLoraMap);
  const [loraGroupsByEngine, setLoraGroupsByEngine] = useState(emptyLoraGroupMap);
  const [libraries, setLibraries] = useState({ SD: null, iL: null, Anima: null });
  const [activeView, setActiveView] = useState("overview");
  const [category, setCategory] = useState("mounted");
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [stateReady, setStateReady] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [workspaceLocked, setWorkspaceLocked] = useState(true);
  const [syncReady, setSyncReady] = useState(false);
  const [lookups, setLookups] = useState({});
  const [collapsedSharedFolders, setCollapsedSharedFolders] = useState({});
  const [detail, setDetail] = useState(null);
  modelRef.current = model;
  lorasRef.current = loras;
  workspaceLockedRef.current = workspaceLocked;
  const loraDragLocked = pageLoraDragLocked({ workspaceLocked, syncReady, canPersist: canPersistRef.current, scopeKey: scopeKeyRef.current });
  const currentLoraDragLocked = () => pageLoraDragLocked({
    workspaceLocked: workspaceLockedRef.current,
    syncReady: syncReadyRef.current,
    canPersist: canPersistRef.current,
    scopeKey: scopeKeyRef.current,
  });

  const applyWorkspaceLock = (locked) => {
    workspaceLockedRef.current = locked;
    // The mount panel ends its own gesture when the lock reaches it as a prop,
    // so only the scan token has to be invalidated here.
    if (locked) libraryScanTokenRef.current += 1;
    setWorkspaceLocked(locked);
  };

  const loadLibraries = async (refresh = false) => {
    if (workspaceLockedRef.current) return false;
    const requestToken = ++libraryScanTokenRef.current;
    refresh ? setRefreshing(true) : setLoading(true);
    setError("");
    try {
      const results = await Promise.all(ENGINES.map(async (engine) => {
        const response = await fetch(`/api/loras?engine=${encodeURIComponent(engine)}`, { cache: "no-store" });
        const payload = await response.json();
        if (!response.ok) throw new Error(`${engine}: ${payload.error || "无法读取 LoRA 目录"}`);
        return [engine, payload];
      }));
      if (workspaceLockedRef.current || requestToken !== libraryScanTokenRef.current) return false;
      setLibraries(Object.fromEntries(results));
    } catch (loadError) {
      if (!workspaceLockedRef.current && requestToken === libraryScanTokenRef.current) setError(loadError.message);
    } finally {
      if (requestToken === libraryScanTokenRef.current) { setLoading(false); setRefreshing(false); }
    }
  };

  useEffect(() => {
    document.title = "LoRA 资产管理 - XiriaCanvas AI";
    applySavedTheme();
    let cancelled = false;
    fetch("/api/ui-state", { cache: "no-store" })
      .then(async (response) => {
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "无法读取工作区状态");
        return payload.state || {};
      })
      .then((state) => {
        if (cancelled) return;
        applySavedTheme(state.theme);
        const workspace = state.workspace && typeof state.workspace === "object" ? state.workspace : {};
        const engine = ENGINES.includes(workspace.model) ? workspace.model : "SD";
        const scopeKey = engineScopeKey(engine);
        const normalized = normalizeMountedLoraMap(workspace.mountedLorasByEngine, {
          fieldMissing: !Object.prototype.hasOwnProperty.call(workspace, "mountedLorasByEngine"),
          legacyLoras: workspace.loras,
          activeEngine: scopeKey,
        });
        let map = normalized.container;
        // The parent can answer the BroadcastChannel handshake before this
        // disk read resolves. Overlay only those named scopes so a stale file
        // cannot erase the parent’s current in-memory list or unrelated scopes.
        for (const [pendingScopeKey, pendingLoras] of pendingScopedSyncRef.current) {
          map = withMountedLorasForScope(map, pendingScopeKey, pendingLoras);
        }
        const initialLoras = mountedLorasForScope(map, scopeKey);
        const groupMap = normalizeLoraGroupMap(workspace.loraGroupsByEngine, {
          fieldMissing: !Object.prototype.hasOwnProperty.call(workspace, "loraGroupsByEngine"),
        });
        loraGroupsMapRef.current = groupMap.container;
        setLoraGroupsByEngine(groupMap.container);
        mountedLorasMapRef.current = map;
        scopeKeyRef.current = scopeKey;
        canPersistRef.current = !normalized.fatal;
        setMountedLorasByEngine(map);
        setModel(engine);
        setLoras(initialLoras);
        if (normalized.warning) setNotice(normalized.warning);
      })
      .catch((loadError) => !cancelled && setError(loadError.message))
      .finally(() => {
        if (cancelled) return;
        initializedRef.current = true;
        setStateReady(true);
      });
    return () => { cancelled = true; };
  }, []);

  // The parent handshake starts locked. Scan once only after workspace state is
  // ready and the latest lock says it is safe; re-locking invalidates responses.
  useEffect(() => {
    if (!stateReady || workspaceLocked) return;
    void loadLibraries();
  }, [stateReady, workspaceLocked]);

  useEffect(() => {
    if (typeof BroadcastChannel === "undefined") return undefined;
    persistenceEpochRef.current.mount();
    const channel = new BroadcastChannel(LORA_SYNC_CHANNEL);
    channelRef.current = channel;
    channel.onmessage = (event) => {
      const payload = event.data;
      if (payload?.type === "workspace-lora-lock") {
        persistenceEpochRef.current.invalidate();
        applyWorkspaceLock(payload.locked === true);
        syncReadyRef.current = true;
        setSyncReady(true);
        return;
      }
      if (payload?.type !== "workspace-loras") return;
      persistenceEpochRef.current.invalidate();
      applyWorkspaceLock(payload.locked === true);
      syncReadyRef.current = true;
      setSyncReady(true);
      const synced = applyMountedLoraSync(mountedLorasMapRef.current, { engine: payload.engine || payload.scopeKey, loras: payload.loras }, { activeEngine: scopeKeyRef.current, locked: payload.locked === true });
      if (!synced.applied) return;
      pendingScopedSyncRef.current.set(payload.engine || payload.scopeKey, normalizeMountedLoras(payload.loras));
      mountedLorasMapRef.current = synced.container;
      setMountedLorasByEngine(synced.container);
      // A message for another complete model identity updates only the map.
      // It must not silently switch this page's active engine/card grid.
      if ((payload.engine || payload.scopeKey) === scopeKeyRef.current && !sameMountedLoras(synced.activeLoras, lorasRef.current)) {
        localChangeRef.current = false;
        lorasRef.current = synced.activeLoras;
        setLoras(synced.activeLoras);
        setNotice("已同步当前模型的 LoRA 挂载库");
      }
    };
    channel.postMessage({ type: "request-workspace-loras" });
    // A dedicated page also works when the parent tab is closed. The existing
    // active-job route supplies a conservative lock after the handshake window;
    // a live parent always wins through its immediate lock message.
    const fallbackTimer = window.setTimeout(async () => {
      if (syncReadyRef.current) return;
      try {
        const response = await fetch("/api/inference/jobs/active", { cache: "no-store" });
        const payload = await response.json();
        const active = ["queued", "running", "pausing", "paused", "cancelling"].includes(payload.job?.status);
        if (!syncReadyRef.current) applyWorkspaceLock(active);
      } catch {
        if (!syncReadyRef.current) applyWorkspaceLock(false);
      } finally {
        if (!syncReadyRef.current) {
          syncReadyRef.current = true;
          setSyncReady(true);
        }
      }
    }, 450);
    return () => {
      window.clearTimeout(fallbackTimer);
      persistenceEpochRef.current.unmount();
      channel.close();
    };
  }, []);

  useEffect(() => {
    if (!initializedRef.current || !localChangeRef.current || workspaceLocked || !syncReady || !canPersistRef.current) return undefined;
    const scopeKey = scopeKeyRef.current;
    if (!scopeKey) return undefined;
    channelRef.current?.postMessage({ type: "workspace-loras", engine: scopeKey, loras, locked: false });
    const epoch = persistenceEpochRef.current.nextEpoch();
    const admission = () => ({ locked: workspaceLockedRef.current, syncReady: syncReadyRef.current });
    const timer = window.setTimeout(async () => {
      const outcome = await runLoraPersistenceEpoch({
        guard: persistenceEpochRef.current,
        epoch,
        admission,
        get: async (signal) => {
          const response = await fetch("/api/ui-state", { cache: "no-store", signal });
          const payload = await response.json();
          if (!response.ok) throw new Error(payload.error || "无法同步工作区状态");
          return payload;
        },
        prepare: (payload) => {
          const workspace = payload.state?.workspace && typeof payload.state.workspace === "object" ? payload.state.workspace : {};
          const serverScopeKey = engineScopeKey(workspace.model);
          const normalized = normalizeMountedLoraMap(workspace.mountedLorasByEngine, {
            fieldMissing: !Object.prototype.hasOwnProperty.call(workspace, "mountedLorasByEngine"), legacyLoras: workspace.loras, activeEngine: serverScopeKey,
          });
          if (normalized.fatal) throw new Error("LoRA 挂载库格式不可安全保存");
          const map = withMountedLorasForScope(normalized.container, scopeKey, lorasRef.current);
          // Groups ride the same write. They are never fatal, so a damaged
          // stored value is replaced rather than blocking the mounted map.
          const groupMap = normalizeLoraGroupMap(workspace.loraGroupsByEngine, {
            fieldMissing: !Object.prototype.hasOwnProperty.call(workspace, "loraGroupsByEngine"),
          }).container;
          const groups = withLoraGroupsForScope(groupMap, scopeKey, loraGroupsForScope(loraGroupsMapRef.current, scopeKey));
          return { ...payload.state, workspace: { ...workspace, mountedLorasByEngine: map, loraGroupsByEngine: groups, loras: mountedLorasForScope(map, serverScopeKey) } };
        },
        put: async (body, signal) => {
          const response = await fetch("/api/ui-state", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal });
          if (!response.ok) throw new Error("无法保存 LoRA 工作区状态");
          return response;
        },
      });
      if (outcome.written || outcome.error) localChangeRef.current = false;
      if (outcome.error) setNotice(outcome.error.message);
    }, 350);
    return () => { window.clearTimeout(timer); persistenceEpochRef.current.invalidate(); };
  }, [loras, mountedLorasByEngine, loraGroupsByEngine, workspaceLocked, syncReady]);

  const updateMountedLoras = (updater) => {
    if (currentLoraDragLocked()) {
      setNotice(workspaceLockedRef.current ? "生成或模型切换期间不能修改 LoRA" : "正在确认工作区锁定状态");
      return;
    }
    const next = normalizeMountedLoras(updater(lorasRef.current));
    if (sameMountedLoras(next, lorasRef.current)) return;
    const map = withMountedLorasForScope(mountedLorasMapRef.current, scopeKeyRef.current, next);
    persistenceEpochRef.current.invalidate();
    localChangeRef.current = true;
    mountedLorasMapRef.current = map;
    lorasRef.current = next;
    setMountedLorasByEngine(map);
    setLoras(next);
  };

  const updateLoraGroups = (updater) => {
    if (currentLoraDragLocked()) {
      setNotice(workspaceLockedRef.current ? "生成或模型切换期间不能修改 LoRA 组合" : "正在确认工作区锁定状态");
      return;
    }
    const result = updateLoraGroupsForScope(loraGroupsMapRef.current, scopeKeyRef.current, updater);
    if (!result.changed) return;
    persistenceEpochRef.current.invalidate();
    localChangeRef.current = true;
    loraGroupsMapRef.current = result.container;
    setLoraGroupsByEngine(result.container);
  };

  const analytics = analyticsFor(libraries);
  const pageLoading = loading || !stateReady;
  const currentLibrary = libraries[model];
  const activeCategory = currentLibrary?.categories.find((item) => item.id === category);
  const matchesSearch = (item) => item.name.toLowerCase().includes(search.trim().toLowerCase());
  const visibleFiles = (activeCategory?.models || []).filter(matchesSearch);
  const mountedValues = new Set(loras.map((item) => item.value));
  const activeLoraGroups = loraGroupsForScope(loraGroupsByEngine, scopeKeyRef.current || model);
  activeLoraGroupsRef.current = activeLoraGroups;
  // Mounted rows carry no metadata; the preview belongs to the library listing,
  // so an asset whose source was never looked up simply has no image.
  const loraPreviewUrlFor = (item) => {
    const metadata = (currentLibrary?.categories || []).flatMap((entry) => entry.models).find((entry) => entry.value === item.value)?.metadata;
    return metadata?.status === "found"
      ? `/api/lora-preview?engine=${encodeURIComponent(model)}&path=${encodeURIComponent(item.value)}&v=${encodeURIComponent(metadata.queriedAt)}`
      : "";
  };

  useEffect(() => {
    const reducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    if (pageLoading || !pageRef.current || document.documentElement.dataset.motion === "off" || reducedMotion) return undefined;
    const context = gsap.context(() => {
      const timeline = gsap.timeline({ defaults: { ease: "power3.out" } });
      timeline.from(".lora-page-command > *", { y: 14, opacity: 0, duration: .42, stagger: .06 });
      if (activeView === "overview") {
        timeline.from(".lora-metric-card", { y: 14, opacity: 0, duration: .42, stagger: .05 }, "-=.2")
          .from(".lora-analysis-panel", { y: 16, opacity: 0, duration: .46, stagger: .06 }, "-=.25");
      } else {
        timeline.from(".lora-manager-workspace", { y: 16, opacity: 0, duration: .48 }, "-=.2");
      }
      gsap.fromTo(".lora-analysis-bar i", { scaleX: 0 }, { scaleX: 1, duration: .85, stagger: .07, ease: "power3.out", transformOrigin: "left" });
      gsap.from(".lora-donut-segment", { strokeDasharray: `0 ${DONUT_CIRCUMFERENCE}`, duration: 1.1, stagger: .08, ease: "power2.out" });
      for (const node of pageRef.current.querySelectorAll("[data-count]")) {
        const target = Number(node.dataset.count);
        const counter = { value: 0 };
        gsap.to(counter, { value: target, duration: .9, ease: "power2.out", onUpdate: () => { node.textContent = Math.round(counter.value).toLocaleString(); } });
      }
    }, pageRef);
    return () => context.revert();
  }, [activeView, pageLoading, libraries, model]);

  useEffect(() => {
    if (!notice) return undefined;
    const timer = window.setTimeout(() => setNotice(""), 3200);
    return () => window.clearTimeout(timer);
  }, [notice]);

  const toggleLora = (item, categoryData) => {
    let next = null;
    updateMountedLoras((current) => {
      if (current.some((entry) => entry.value === item.value)) {
        next = current.filter((entry) => entry.value !== item.value);
        return next;
      }
      if (current.length >= 16) {
        setNotice("单次生成最多挂载 16 个 LoRA");
        return current;
      }
      // A collecting group claims new mounts, so picking LoRAs from a category
      // builds that combination directly instead of landing them unsorted.
      next = [...current, mountedEntryForGroups({ ...item, category: categoryData.label, weight: 1, precision: 1, enabled: true }, activeLoraGroupsRef.current)];
      return next;
    });
    // Mounting from the library bypasses the mount panel, so the write-back has
    // to happen here too or a collected LoRA would never reach its definition.
    if (next) updateLoraGroups((current) => syncMountedIntoGroups(current, next));
  };
  const lookupLora = async (item, refresh = false, categoryId = "") => {
    if (lookups[item.value]?.loading) return;
    setLookups((current) => ({ ...current, [item.value]: { loading: true, error: "" } }));
    try {
      const response = await fetch("/api/lora-lookup", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ engine: model, path: item.value, refresh, category: categoryId }) });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "查询失败");
      setLibraries((current) => ({ ...current, [model]: { ...current[model], categories: current[model].categories.map((categoryData) => ({ ...categoryData, models: categoryData.models.map((modelItem) => modelItem.value === item.value ? { ...modelItem, metadata: payload.metadata } : modelItem) })) } }));
      setLookups((current) => ({ ...current, [item.value]: { loading: false, error: payload.metadata.status === "not_found" ? "未找到来源" : "" } }));
      return payload.metadata;
    } catch (lookupError) {
      setLookups((current) => ({ ...current, [item.value]: { loading: false, error: lookupError.message } }));
    }
    return null;
  };

  const openDetails = (item, categoryId) => {
    setDetail({ value: item.value, name: item.name, categoryId });
    if (item.metadata?.detailSchema !== 1 || item.metadata?.triggerReviewSchema !== 4) void lookupLora(item, false, categoryId);
  };

  // One card definition for both the flat categories and the shared tree, so a
  // shared LoRA mounts, previews and opens details exactly like a local one.
  const renderLibraryCard = (item, categoryLabel) => {
    const mounted = mountedValues.has(item.value);
    const metadata = item.metadata;
    const lookup = lookups[item.value];
    const previewUrl = metadata?.status === "found" ? `/api/lora-preview?engine=${encodeURIComponent(model)}&path=${encodeURIComponent(item.value)}&v=${encodeURIComponent(metadata.queriedAt)}` : "";
    return <article aria-disabled={loraDragLocked} className={`lora-library-card lora-page-library-card ${mounted ? "mounted" : ""} ${loraDragLocked ? "locked" : ""}`} key={item.value} role="button" tabIndex="0" onClick={() => toggleLora(item, activeCategory)} onKeyDown={(event) => { if (event.target === event.currentTarget && (event.key === "Enter" || event.key === " ")) { event.preventDefault(); toggleLora(item, activeCategory); } }}>
      <div className="lora-card-preview"><span>NO PREVIEW</span>{previewUrl && <img src={previewUrl} alt="" onError={(event) => event.currentTarget.remove()} />}<button type="button" className="lora-card-query" disabled={lookup?.loading} onClick={(event) => { event.stopPropagation(); void lookupLora(item, false, activeCategory?.id); }}>{lookup?.loading ? <RefreshCw className="spin" size={13} /> : <Search size={13} />}{lookup?.loading ? "查询中" : "查询"}</button><span className="lora-card-state">{mounted ? <><Check size={13} />已挂载</> : <><Plus size={13} />挂载</>}</span></div>
      <div className="lora-card-copy"><strong title={item.name}>{item.name}</strong><small>{formatBytes(item.size)} · {categoryLabel}</small>{metadata?.status === "found" && <div className="lora-card-metadata"><span>{metadata.modelName || metadata.versionName || "已识别模型"}</span><small>{metadata.baseModel || "来源元数据已载入"}</small></div>}{lookup?.error && <span className="lora-card-error">{lookup.error}</span>}<button type="button" className="lora-card-detail" onClick={(event) => { event.stopPropagation(); openDetails(item, activeCategory?.id); }}><FileText size={12} />查看详细信息</button></div>
    </article>;
  };

  let donutOffset = 0;
  return <main ref={pageRef} className="lora-page-shell">
    <header className="lora-page-topbar">
      <div><span><Layers3 size={18} /></span><strong>XiriaCanvas</strong><b>LoRA ASSET INTELLIGENCE</b></div>
      <nav className="lora-page-view-switch" aria-label="LoRA 页面视图">
        <button type="button" className={activeView === "overview" ? "active" : ""} aria-pressed={activeView === "overview"} onClick={() => setActiveView("overview")}><BarChart3 size={14} />资产概览</button>
        <button type="button" className={activeView === "manager" ? "active" : ""} aria-pressed={activeView === "manager"} onClick={() => setActiveView("manager")}><Layers3 size={14} />挂载管理</button>
      </nav>
      <div className="lora-page-topbar-actions"><span><i />{model} 同步在线</span><button type="button" onClick={() => { window.close(); window.setTimeout(() => window.location.assign("/"), 80); }}><X size={16} />关闭此页</button></div>
    </header>

    <section className="lora-page-content">
      <header className="lora-page-command">
        <div><span>{activeView === "manager" ? "LIVE WORKSPACE CONTROL" : "LOCAL ASSET OBSERVATORY"}</span><h1>{activeView === "manager" ? `${model} LoRA 挂载管理` : "LoRA 资产概览"}</h1><p>{activeView === "manager" ? "浏览当前引擎目录，挂载与排序会实时同步到主工作区。" : "汇总 SD、iL 与 Anima 的容量、分类、引擎和文件夹分布。"}</p></div>
        <div><span>{activeView === "manager" ? `${loras.length} / 16 MOUNTED` : `${analytics.files.length} FILES · ${formatBytes(analytics.totalBytes)}`}</span><button type="button" disabled={refreshing || workspaceLocked} onClick={() => void loadLibraries(true)}><RefreshCw className={refreshing ? "spin" : ""} size={15} />刷新目录</button></div>
      </header>

      <div className="lora-page-stage">
        {error && <div className="lora-page-error"><X size={18} />{error}</div>}
        {pageLoading ? <div className="lora-page-loading"><RefreshCw className="spin" size={28} /><span>正在扫描全部 LoRA 目录</span></div> : activeView === "overview" ? <section className="lora-page-view lora-overview-view">
          <section className="lora-metric-grid">
            <article className="lora-metric-card"><span><Database size={15} />总文件</span><strong data-count={analytics.files.length}>{analytics.files.length}</strong><small>SD + iL + Anima 完整目录</small></article>
            <article className="lora-metric-card"><span><HardDrive size={15} />磁盘占用</span><strong>{formatBytes(analytics.totalBytes)}</strong><small>{analytics.folders.length} 个分类目录</small></article>
            <article className="lora-metric-card"><span><Layers3 size={15} />当前挂载</span><strong data-count={loras.length}>{loras.length}</strong><small>{loras.filter((item) => item.enabled !== false).length} 个已启用</small></article>
            <article className="lora-metric-card"><span><BarChart3 size={15} />平均体积</span><strong>{formatBytes(analytics.files.length ? analytics.totalBytes / analytics.files.length : 0)}</strong><small>按全部文件计算</small></article>
          </section>
          <section className="lora-overview-board">
            <article className="lora-analysis-panel lora-storage-chart">
              <header><span>STORAGE DISTRIBUTION</span><strong>分类占用分布</strong></header>
              <div className="lora-donut-wrap"><svg viewBox="0 0 120 120" aria-label="LoRA 分类磁盘占用分布"><circle cx="60" cy="60" r="48" fill="none" stroke="#252623" strokeWidth="12" />{analytics.categories.map((item, index) => { const span = analytics.totalBytes ? item.bytes / analytics.totalBytes * DONUT_CIRCUMFERENCE : 0; const circle = <circle className="lora-donut-segment" key={item.id} cx="60" cy="60" r="48" fill="none" stroke={CHART_COLORS[index % CHART_COLORS.length]} strokeWidth="12" strokeDasharray={`${span} ${DONUT_CIRCUMFERENCE - span}`} strokeDashoffset={-donutOffset} transform="rotate(-90 60 60)" />; donutOffset += span; return circle; })}</svg><div><strong>{formatBytes(analytics.totalBytes)}</strong><span>TOTAL</span></div></div>
              <div className="lora-chart-legend">{analytics.categories.map((item, index) => <div key={item.id}><i style={{ background: CHART_COLORS[index % CHART_COLORS.length] }} /><span>{item.label}</span><b>{formatBytes(item.bytes)}</b><small>{item.count} files</small></div>)}</div>
            </article>
            <article className="lora-analysis-panel lora-engine-panel"><header><span>ENGINE FOOTPRINT</span><strong>引擎目录占用</strong></header><div className="lora-engine-bars">{analytics.engines.map((item) => <div key={item.engine}><span><strong>{item.engine}</strong><small>{item.count} FILES</small><b>{formatBytes(item.bytes)}</b></span><div className="lora-analysis-bar"><i style={{ width: `${analytics.totalBytes ? item.bytes / analytics.totalBytes * 100 : 0}%` }} /></div><p>{libraries[item.engine]?.directory || "目录不可用"}</p></div>)}</div></article>
            <article className="lora-analysis-panel lora-folder-panel"><header><span>FOLDER TOPOLOGY</span><strong>文件夹占用</strong></header><div>{analytics.folders.map((folder) => <div key={`${folder.engine}-${folder.id}`}><span><b>{folder.engine}</b><strong>{folder.label}</strong></span><small>{folder.count} files</small><em>{formatBytes(folder.bytes)}</em></div>)}</div></article>
            <article className="lora-analysis-panel lora-largest-panel"><header><span>LARGEST ASSETS</span><strong>大文件排行</strong></header><div>{analytics.largest.map((item, index) => <div key={`${item.engine}-${item.value}`}><i>{String(index + 1).padStart(2, "0")}</i><span><strong title={item.name}>{item.name}</strong><small>{item.engine} · {item.categoryLabel}</small></span><b>{formatBytes(item.size)}</b></div>)}</div></article>
          </section>
        </section> : <section className="lora-page-view lora-manager-view">
          <section className="lora-manager-workspace">
          <div className="lora-page-manager-body">
            <nav className="lora-page-categories">
              <button type="button" className={category === "mounted" ? "active" : ""} onClick={() => setCategory("mounted")}><span>已挂载</span><b>{loras.length}</b></button>
              <button type="button" className={category === "groups" ? "active" : ""} onClick={() => setCategory("groups")}><span>组合</span><b>{activeLoraGroups.length}</b></button>
              {(currentLibrary?.categories || []).map((item) => <button type="button" className={category === item.id ? "active" : ""} key={item.id} onClick={() => setCategory(item.id)}><span>{item.label}</span><b>{item.models.length}</b></button>)}
            </nav>
            <div className="lora-page-browser">
              {category === "mounted" ? <>
                <header><div><strong>已挂载队列</strong><span>拖动排序 · 权重 -5 到 5 · 即时同步</span></div></header>
                <LoraMountPanel
                  variant="page"
                  loras={loras}
                  groups={activeLoraGroups}
                  locked={loraDragLocked}
                  previewUrlFor={loraPreviewUrlFor}
                  onUpdateLoras={updateMountedLoras}
                  onUpdateGroups={updateLoraGroups}
                />
              </> : category === "groups" ? <>
                <header><div><strong>LoRA 组合</strong><span>保存常用搭配 · 启用后挂载到队列</span></div></header>
                <LoraGroupPanel
                  groups={activeLoraGroups}
                  loras={loras}
                  locked={loraDragLocked}
                  onUpdateGroups={updateLoraGroups}
                  onUpdateLoras={updateMountedLoras}
                  onNotice={setNotice}
                />
              </> : <>
                <header><div><strong>{activeCategory?.label || "LoRA 分类"}</strong><span>{activeCategory?.directory || "选择分类浏览"}</span></div><label><Search size={15} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="搜索文件名" /></label></header>
                {activeCategory?.shared ? <div className="lora-shared-view">
                  {(activeCategory.roots || []).map((root) => {
                    const folders = (root.folders || []).map((folder) => ({ ...folder, models: folder.models.filter(matchesSearch) })).filter((folder) => folder.models.length);
                    return <section className="lora-shared-root" key={root.id}>
                      <header>
                        <div><Share2 size={14} /><strong>{root.label}</strong><code title={root.path}>{root.path}</code></div>
                        <span>{root.files} 个文件 · {formatBytes(root.bytes)}</span>
                      </header>
                      {folders.length ? folders.map((folder) => {
                        const folderKey = `${root.id}/${folder.name}`;
                        // Keep the hierarchy compact until the user opens a folder.
                        const collapsed = collapsedSharedFolders[folderKey] !== false;
                        return <div className={`lora-shared-folder ${collapsed ? "collapsed" : ""}`} key={folderKey}>
                          <button type="button" aria-expanded={!collapsed} onClick={() => setCollapsedSharedFolders((current) => ({ ...current, [folderKey]: !collapsed }))}>
                            <ChevronDown size={13} /><FolderOpen size={13} /><strong>{folder.label}</strong><b>{folder.models.length}</b>
                          </button>
                          {!collapsed && <div className="lora-page-library-grid">{folder.models.map((item) => renderLibraryCard(item, folder.label))}</div>}
                        </div>;
                      }) : <p className="lora-shared-empty">{search ? "该共享目录中没有匹配的文件" : "该共享目录为空"}</p>}
                    </section>;
                  })}
                </div> : <div className="lora-page-library-grid">
                  {visibleFiles.map((item) => renderLibraryCard(item, activeCategory?.label))}
                  {!visibleFiles.length && <div className="lora-page-empty"><FolderOpen size={30} /><strong>没有匹配的 LoRA</strong><span>{search ? "尝试其他关键词" : "当前分类目录为空"}</span></div>}
                </div>}
              </>}
            </div>
          </div>
          </section>
        </section>}
      </div>
    </section>
    {notice && <button type="button" className="lora-page-notice" onClick={() => setNotice("")}>{notice}</button>}
    {detail && (() => {
      const item = (libraries[model]?.categories || []).flatMap((categoryData) => categoryData.models).find((modelItem) => modelItem.value === detail.value) || detail;
      const lookup = lookups[detail.value];
      return <LoraDetailsDialog item={item} metadata={item.metadata} loading={lookup?.loading} error={lookup?.error} onRefresh={() => void lookupLora(item, true, detail.categoryId)} onClose={() => setDetail(null)} />;
    })()}
  </main>;
}
