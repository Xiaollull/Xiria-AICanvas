import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import LoraManagerModal from "./LoraManagerModal.jsx";
import LoraLibraryCard from "./LoraLibraryCard.jsx";
import LoraGroupPanel from "./LoraGroupPanel.jsx";
import XYZGroupCopyDialog from "./XYZGroupCopyDialog.jsx";
import { loraNotice } from "./lora-notice.js";
import LoraDetailsDialog from "./LoraDetailsDialog.jsx";
import LoraCardEditor from "./LoraCardEditor.jsx";
import { LoraRepairEntry } from "./LoraRepairPanel.jsx";
import { loraFolderView } from "./lora-folders.js";
import { groupCardFor, loraCardFor, loraCardPresentation } from "./lora-cards.js";
import { loraFetch as fetch } from "./lora-library-client.js";
import { mountXYZCatalogItem, replaceXYZCatalogSelection, updateXYZManagerGroups, updateXYZManagerMounts } from "./xyz-manager-context.js";
import { analyzeXYZGroupCopy } from "./xyz-group-resolution.js";
import { useXYZGroupCopy } from "./use-xyz-group-copy.js";
import { patchXYZPrompt, xyzPromptSource } from "./xyz-prompt-source.js";
import { xyzStrongLora } from "./xyz-sidebar-context.js";
import { registerEngineLoraIdentities } from "./xyz-lora-identity.js";

export default function XYZLoraManager({ draft, onChange, catalog, libraryGroups, engines, onEngine, onClose, cardsStore, libraryLocked, transaction = false, single = false, axisMarks = {}, title = '全局 LoRA', onConfirm }) {
  const [category, setCategory] = useState("mounted"), [search, setSearch] = useState(""), [folderPath, setFolderPath] = useState("");
  const [path, setPath] = useState(""), [pathBusy, setPathBusy] = useState(false), [collapsedFolders, setCollapsedFolders] = useState({});
  const [notice, setNotice] = useState(null), [detail, setDetail] = useState(null), [cardEdit, setCardEdit] = useState(null);
  const [metadata, setMetadata] = useState({}), [lookups, setLookups] = useState({});
  const [navigation, setNavigation] = useState(null);
  const current = useRef(draft), controllers = useRef(new Set()), engineRef = useRef(draft.engine), mounted = useRef(true);
  current.current = draft; engineRef.current = draft.engine;
  // Severity comes from the operation that produced the message; a bare string is information.
  const notify = (message, kind = "info") => setNotice(loraNotice(message, kind));
  // Cards without a registered identity are locked here; say why, and offer the registration that
  // unlocks them instead of leaving the user to find the four-step repair panel on their own.
  const [registering, setRegistering] = useState("");
  const unregistered = (catalog.loras || []).filter((item) => !xyzStrongLora(item)).length;
  const register = async () => {
    if (registering) return;
    const engine = draft.engine;
    setRegistering("正在准备…");
    try {
      const result = await registerEngineLoraIdentities(engine, { onProgress: (text) => { if (mounted.current) setRegistering(text); } });
      if (!mounted.current) return;
      await catalog.refresh();
      if (mounted.current) notify(`已登记 ${engine} 的 ${result.files} 个 LoRA 身份，现在可以挂载`, "success");
    } catch (error) {
      if (mounted.current) notify(error.message, "error");
    } finally {
      if (mounted.current) setRegistering("");
    }
  };
  const identityEntry = unregistered > 0 && !catalog.loading && <section className="lora-repair-entry xyz-identity-entry" aria-label="LoRA 身份登记">
    <div><strong>{unregistered} 个 LoRA 尚未登记身份</strong><span>XYZ 与混合搜索只能使用已登记身份的 LoRA，未登记的卡片显示为灰色。登记会完整读取 {draft.engine} 的 LoRA 文件并记录身份，不移动文件、不改变任何挂载。</span></div>
    <button type="button" className="lora-repair-primary" disabled={Boolean(registering)} onClick={() => void register()}>{registering || `登记 ${draft.engine} 的 LoRA 身份`}</button>
  </section>;
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; for (const controller of controllers.current) controller.abort(); controllers.current.clear(); };
  }, []);
  useEffect(() => {
    for (const controller of controllers.current) controller.abort(); controllers.current.clear();
    setCategory("mounted"); setSearch(""); setFolderPath(""); setMetadata({}); setLookups({}); setDetail(null); setCardEdit(null); setNotice(null); setPathBusy(false);
  }, [draft.engine]);
  useEffect(() => {
    const target = navigation;
    if (!target || target.engine !== draft.engine || catalog.loading || catalog.engine !== draft.engine) return;
    setNavigation(null);
    if (target.origin === "shared") {
      setCategory("shared"); setFolderPath(""); setCollapsedFolders((old) => ({ ...old, [`${target.rootId}/${target.folder}`]: false }));
    } else if ((catalog.library || []).some((entry) => entry.id === target.categoryId)) {
      setCategory(target.categoryId); setFolderPath(target.folder || "");
    } else { notify(`${target.display || "该文件夹"} 中没有可挂载的 LoRA 文件`, "warning"); return; }
    setPath(target.display || ""); setSearch("");
  }, [navigation, draft.engine, catalog.engine, catalog.loading, catalog.library]);
  const attempt = (edit) => {
    try { const next = edit(current.current); current.current = next; onChange(next); return { ok: true, draft: next }; }
    catch (error) { return { ok: false, error: error.message }; }
  };
  const commit = (edit) => {
    const result = attempt(edit);
    if (result.ok) setNotice(null); else notify(result.error, "error");
    return result.ok ? result.draft : null;
  };
  const copy = useXYZGroupCopy({ engine: draft.engine, draftRef: current, attempt, onDone: ({ group, mode }) => notify({ kind: "success",
    detail: `已复制「${group.name}」${group.members.length} 项为独立快照${mode === "copy-only" ? "（未启用）" : "并启用"}；真实组合库与资产库保持原样。` }) });
  const copyAnalysis = useMemo(() => copy.state?.projection?.ok ? analyzeXYZGroupCopy(draft, copy.state.projection.group) : null, [draft, copy.state]);
  const library = (catalog.library || []).map((entry) => ({ ...entry, models: entry.models.map((item) => ({ ...item, metadata: metadata[item.value] || item.metadata })) }));
  const selected = library.find((entry) => entry.id === category);
  const lookup = async (item, refresh = false) => {
    if (lookups[item.value]?.loading) return;
    const engine = draft.engine, controller = new AbortController(); controllers.current.add(controller);
    setLookups((old) => ({ ...old, [item.value]: { loading: true, error: "" } }));
    try {
      const response = await fetch("/api/lora-lookup", { method: "POST", headers: { "Content-Type": "application/json" }, signal: controller.signal,
        body: JSON.stringify({ engine, path: item.value, refresh, category: selected?.id || item.categoryId || "" }) });
      const payload = await response.json(); if (!response.ok) throw new Error(payload.error || "查询失败");
      if (mounted.current && !controller.signal.aborted && engineRef.current === engine) {
        setMetadata((old) => ({ ...old, [item.value]: payload.metadata }));
        setLookups((old) => ({ ...old, [item.value]: { loading: false, error: "" } }));
      }
    } catch (error) {
      if (mounted.current && !controller.signal.aborted && engineRef.current === engine) setLookups((old) => ({ ...old, [item.value]: { loading: false, error: error.message } }));
    } finally { controllers.current.delete(controller); }
  };
  const locate = async () => {
    if (pathBusy || !path.trim()) return;
    const engine = draft.engine, controller = new AbortController(); controllers.current.add(controller); setPathBusy(true);
    try {
      const response = await fetch("/api/lora-folder", { method: "POST", headers: { "Content-Type": "application/json" }, signal: controller.signal, body: JSON.stringify({ path: path.trim(), engine }) });
      const payload = await response.json(); if (!response.ok || !payload.folder || !engines.includes(payload.folder.engine)) throw new Error(payload.error || "无法定位文件夹");
      if (!mounted.current || controller.signal.aborted || engineRef.current !== engine) return;
       const target = payload.folder;
       if (transaction && target.engine !== engine) throw new Error("轴选择仅属于当前引擎；取消后可在 XYZ 切换引擎");
       setNavigation(target);
      if (target.engine !== engine) onEngine(target.engine);
      setNotice(null);
    } catch (error) { if (mounted.current && !controller.signal.aborted && engineRef.current === engine) notify(error.message, "error"); }
    finally { controllers.current.delete(controller); if (mounted.current && engineRef.current === engine) setPathBusy(false); }
  };
  const presentationFor = (item) => loraCardPresentation({ engine: draft.engine, item, card: loraCardFor(cardsStore.store, draft.engine, item), metadata: metadata[item.value] || library.flatMap((entry) => entry.models).find((entry) => entry.value === item.value)?.metadata });
  const customize = (item) => libraryLocked ? notify("共享资产管理当前锁定；XYZ 草稿编辑仍可用。", "warning") : setCardEdit(item);
  const uploadImage = async (...args) => { if (libraryLocked) throw new Error("共享资产管理当前锁定"); return cardsStore.uploadCardImage(...args); };
  const cards = {
    presentationFor, groupCardFor: (id) => groupCardFor(cardsStore.store, draft.engine, id),
    onEditLora: customize,
    onPatchGroup: (id, patch) => { if (!libraryLocked) cardsStore.patchGroupCard(draft.engine, id, patch); else notify("共享组合卡片当前锁定；快照开关与权重不受影响。", "warning"); },
    // Deleting a snapshot never deletes global artwork.
    onUploadImage: uploadImage,
  };
  const snapshotIds = new Set(draft.baseline.group_snapshots.map((group) => group.id));
  const availableGroups = libraryGroups.filter((group) => !snapshotIds.has(group.id)).map((group) => ({ ...group, enabled: false, collecting: false }));
  return <>
    <LoraManagerModal title={title} className="xyz-owned-lora-manager" repairEntry={<>{!transaction && <LoraRepairEntry engine={draft.engine} locked={libraryLocked} />}{identityEntry}</>} context={{
      independent: true, engine: draft.engine, engines, onEngine: transaction ? undefined : onEngine, library, category, onCategory: (value) => { setCategory(value); setFolderPath(""); },
      engineBlockedReason: transaction ? '轴选择仅属于当前引擎；取消后可在 XYZ 切换引擎' : '',
      loras: draft.baseline.loras, groups: draft.baseline.group_snapshots, directory: catalog.loraDirectory || "", loading: catalog.loading, refreshing: catalog.refreshing,
      error: catalog.error, notice, onDismissNotice: () => setNotice(null), onRefresh: catalog.refresh, locked: false, cards,
      onLoras: (updater) => commit((d) => updateXYZManagerMounts(d, updater)),
      onGroupLoras: (updater) => commit((d) => updateXYZManagerMounts(d, updater, { syncGroups: false })),
      onGroups: (updater) => commit((d) => updateXYZManagerGroups(d, updater)), onNotice: notify,
      onToggleGroup: (group) => commit((d) => updateXYZManagerGroups(d, (groups) => groups.map((item) => item.id === group.id ? { ...item, enabled: !item.enabled } : item))),
      onCollectGroup: (group, id) => commit((d) => updateXYZManagerGroups(d, (groups) => groups.map((item) => ({ ...item, collecting: item.id === id, ...(item.id === id ? { enabled: true } : {}) })))),
      onCreateGroup: ({ group, members, fromMounted }) => commit((d) => updateXYZManagerGroups(d, (groups) => [...groups, { ...group, ...(fromMounted ? { members } : {}) }])),
      groupLibrary: availableGroups.length > 0 && <LoraGroupPanel groups={availableGroups} cards={{ groupCardFor: cards.groupCardFor }} libraryCopy={(group) => void copy.start(group)} />,
      search, onSearch: setSearch, folder: selected && !selected.shared ? loraFolderView(selected, folderPath, search) : null, onFolder: setFolderPath,
      path, onPath: setPath, pathBusy, onLocate: locate, collapsedFolders, onCollapsedFolders: setCollapsedFolders,
      formatBytes: (bytes) => `${(Number(bytes || 0) / 1024 / 1024).toFixed(1)} MB`,
      renderCard: (item, label, displayName) => <LoraLibraryCard key={item.instanceId || item.value} item={item} categoryLabel={label} displayName={displayName} presentation={presentationFor(item)} axisMarks={axisMarks[item.instanceId] || axisMarks[item.value] || []}
        lookup={lookups[item.value]} mounted={draft.baseline.loras.some((entry) => entry.instanceId ? entry.instanceId === item.instanceId : entry.value === item.value)}
         locked={!xyzStrongLora(item)} onToggle={() => commit((d) => single ? replaceXYZCatalogSelection(d, { ...item, category: label }) : mountXYZCatalogItem(d, { ...item, category: label }))} onLookup={() => void lookup(item)} onCustomize={transaction ? undefined : () => customize(item)}
        onDetails={() => { setDetail(item); if (item.metadata?.detailSchema !== 1 || item.metadata?.triggerReviewSchema !== 4) void lookup(item); }} />,
      onExternal: transaction ? undefined : () => window.open("/lora", "_blank", "noopener,noreferrer"),
      externalBlockedReason: transaction ? '轴选择是独立临时事务；不能打开可修改全局 LoRA 的资产页' : '',
      onClose, onConfirm: transaction ? () => onConfirm(current.current) : undefined, confirmLabel:transaction?'确认轴选择':undefined,
      footer: transaction ? (single ? '单选：点击其他 LoRA 会替换当前选择。仅编辑本轴 / 本项的临时集合；关闭或取消不保存，确认后冻结强身份与权重。' : '仅编辑本轴 / 本项的临时集合。关闭或取消不保存；确认后冻结强身份、权重与预设词，不改左侧全局 LoRA。') : "挂载、组合开关与权重只写 XYZ 草稿；来源查询、卡片自定义与修复是显式共享资产管理。",
    }} />
    {copy.state && <XYZGroupCopyDialog state={copy.state} analysis={copyAnalysis} onClose={copy.cancel} onConfirm={(mode, shown) => void copy.confirm(mode, shown)} onRetry={() => void copy.start(copy.state.source)} />}
    {detail && <LoraDetailsDialog item={detail} metadata={metadata[detail.value] || detail.metadata} loading={lookups[detail.value]?.loading} error={lookups[detail.value]?.error} onRefresh={() => void lookup(detail, true)} onClose={() => setDetail(null)} />}
    {cardEdit && createPortal(<LoraCardEditor engine={draft.engine} item={cardEdit} card={loraCardFor(cardsStore.store, draft.engine, cardEdit)} metadata={metadata[cardEdit.value] || cardEdit.metadata}
      loading={lookups[cardEdit.value]?.loading} onPatch={(patch) => { if (!libraryLocked) cardsStore.patchLoraCard(draft.engine, cardEdit.value, { ...patch, ...(cardEdit.instanceId ? { identitySchema: 1, instanceId: cardEdit.instanceId, contentHash: cardEdit.contentHash, contentRevision: cardEdit.contentRevision } : {}) }); }}
      onUploadImage={uploadImage} onInsertPrompt={(text) => commit((d) => { const raw = xyzPromptSource(d).positive; return patchXYZPrompt(d, "positive", raw.trim() ? `${raw.replace(/[,\s]+$/, "")}, ${text}` : text); })} onClose={() => setCardEdit(null)} />, document.body)}
  </>;
}
