import { useEffect, useState } from "react";
import { Check, ImageIcon, Images, Plus, RefreshCw, Sparkles, X } from "lucide-react";
import { galleryOutputSeedSettings } from "./hires-settings";
import { displayTitle, galleryRequest, normalizedSettings, useDialogLifecycle } from "./gallery-core";

export function AddToGalleryDialog({ outputs, selectedOutputIndex, settings, onClose, onSaved }) {
  const dialogRef = useDialogLifecycle(true, onClose);
  const [data, setData] = useState({ collections: [], cards: [] });
  const [collectionId, setCollectionId] = useState("");
  const [target, setTarget] = useState("new");
  const [replaceCardId, setReplaceCardId] = useState("");
  const [groupMode, setGroupMode] = useState(settings?.imagesPerBatch > 1 && settings?.batchCount === 1 ? "combined" : "separate");
  const [selected, setSelected] = useState(() => new Set([outputs[selectedOutputIndex]?.asset_id].filter(Boolean)));
  const [title, setTitle] = useState("");
  const [newCollectionId, setNewCollectionId] = useState("");
  const [newCollectionDescription, setNewCollectionDescription] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    galleryRequest().then((payload) => {
      setData(payload);
      setCollectionId(payload.collections[0]?.id || "");
    }).catch((requestError) => setError(requestError.message));
  }, []);
  useEffect(() => {
    const cards = data.cards.filter((card) => card.collection_id === collectionId);
    setReplaceCardId((current) => cards.some((card) => card.id === current) ? current : cards[0]?.id || "");
  }, [collectionId, data.cards]);
  const selectedOutputs = outputs.filter((output) => selected.has(output.asset_id));
  const current = outputs[selectedOutputIndex];
  const currentBatch = outputs.filter((output) => output.batch_index === current?.batch_index);
  const selectOnly = (items) => setSelected(new Set(items.map((item) => item.asset_id).filter(Boolean)));
  const toggle = (assetId) => setSelected((currentSelection) => { const next = new Set(currentSelection); if (next.has(assetId)) next.delete(assetId); else next.add(assetId); return next; });
  const ensureCollection = async () => {
    if (data.collections.length) return collectionId;
    if (!newCollectionId.trim()) throw new Error("请先填写收藏夹 ID");
    const created = await galleryRequest("/collections", { method: "POST", body: JSON.stringify({ id: newCollectionId.trim(), description: newCollectionDescription.trim() || null }) });
    setData((currentData) => ({ ...currentData, collections: [...currentData.collections, created] }));
    setCollectionId(created.id);
    return created.id;
  };
  const settingsForOutput = (output, selectedItems, combined = false) => {
    const normalized = normalizedSettings(settings);
    return galleryOutputSeedSettings(normalized, output, selectedItems, combined);
  };
  const save = async () => {
    if (!selectedOutputs.length || busy) return;
    setBusy(true);
    setError("");
    try {
      const destination = await ensureCollection();
      let savedCards = [];
      if (target === "replace") {
        if (!replaceCardId) throw new Error("所选收藏夹中没有可覆盖的卡片");
        const updated = await galleryRequest(`/cards/${replaceCardId}`, { method: "PATCH", body: JSON.stringify({ collection_id: destination, title: title.trim() || null, settings: settingsForOutput(selectedOutputs[0], selectedOutputs, true), images: selectedOutputs.map((output) => ({ asset_id: output.asset_id })) }) });
        savedCards = [updated];
      } else if (groupMode === "combined") {
        const created = await galleryRequest("/cards", { method: "POST", body: JSON.stringify({ collection_id: destination, title: title.trim() || null, settings: settingsForOutput(selectedOutputs[0], selectedOutputs, true), images: selectedOutputs.map((output) => ({ asset_id: output.asset_id })) }) });
        savedCards = [created];
      } else {
        const payload = await galleryRequest("/cards/bulk", { method: "POST", body: JSON.stringify({ collection_id: destination, cards: selectedOutputs.map((output) => ({ title: title.trim() || null, settings: settingsForOutput(output, [output]), images: [{ asset_id: output.asset_id }] })) }) });
        savedCards = payload.cards || [];
      }
      onSaved({ collectionId: destination, cardId: savedCards[0]?.id, count: savedCards.length, imageCount: selectedOutputs.length });
    } catch (requestError) {
      setError(requestError.message);
      setBusy(false);
    }
  };
  const collectionCards = data.cards.filter((card) => card.collection_id === collectionId);
  return <div className="gallery-add-backdrop" onMouseDown={(event) => event.target === event.currentTarget && !busy && onClose()}>
    <section ref={dialogRef} className="gallery-add-dialog" role="dialog" aria-modal="true" aria-label="加入画廊" tabIndex="-1">
      <header><div><span>ADD TO GALLERY</span><h2>加入画廊</h2><p>选择一张或多张生成结果，创建新卡片或覆盖已有卡片。</p></div><button type="button" aria-label="关闭加入画廊窗口" disabled={busy} onClick={onClose}><X size={20} /></button></header>
      <div className="gallery-add-body">
        <section className="gallery-output-picker"><header><div><strong>选择图片</strong><span>{selectedOutputs.length} / {outputs.length}</span></div><div><button type="button" onClick={() => selectOnly(current ? [current] : [])}>当前</button><button type="button" onClick={() => selectOnly(currentBatch)}>当前批次</button><button type="button" onClick={() => selectOnly(outputs)}>全部</button></div></header><div>{outputs.map((output, index) => <button type="button" className={selected.has(output.asset_id) ? "selected" : ""} disabled={!output.asset_id} key={output.asset_id || index} onClick={() => toggle(output.asset_id)}><img src={output.url} alt={`结果 ${index + 1}`} /><i>{selected.has(output.asset_id) && <Check size={13} />}</i><span>批 {output.batch_index} / 图 {output.image_index}</span></button>)}</div></section>
        <section className="gallery-add-config">
          {data.collections.length ? <label className="gallery-field"><span>目标收藏夹</span><select value={collectionId} onChange={(event) => setCollectionId(event.target.value)}>{data.collections.map((item) => <option value={item.id} key={item.id}>{item.id} · {item.card_count} 卡片</option>)}</select></label> : <div className="gallery-inline-collection"><strong>先创建收藏夹</strong><label className="gallery-field"><span>收藏夹 ID</span><input value={newCollectionId} maxLength={64} onChange={(event) => setNewCollectionId(event.target.value)} /></label><label className="gallery-field"><span>简介（可选）</span><textarea value={newCollectionDescription} maxLength={1000} onChange={(event) => setNewCollectionDescription(event.target.value)} /></label></div>}
          <div className="gallery-target-switch"><button type="button" className={target === "new" ? "active" : ""} onClick={() => setTarget("new")}><Plus size={14} />新建卡片</button><button type="button" className={target === "replace" ? "active" : ""} disabled={!collectionCards.length} onClick={() => setTarget("replace")}><RefreshCw size={14} />替换覆盖</button></div>
          {target === "replace" ? <label className="gallery-field"><span>覆盖收藏夹内卡片</span><select value={replaceCardId} onChange={(event) => setReplaceCardId(event.target.value)}>{collectionCards.map((card) => <option value={card.id} key={card.id}>{displayTitle(card)} · {card.image_count} 张</option>)}</select></label> : <div className="gallery-group-choice"><button type="button" className={groupMode === "separate" ? "active" : ""} onClick={() => setGroupMode("separate")}><ImageIcon size={16} /><span><strong>每图一张卡片</strong><small>适合单图多批次</small></span></button><button type="button" className={groupMode === "combined" ? "active" : ""} onClick={() => setGroupMode("combined")}><Images size={16} /><span><strong>合并为多图卡片</strong><small>详情内可切换图片</small></span></button></div>}
          <label className="gallery-field"><span>卡片标题（可选）</span><input value={title} maxLength={160} onChange={(event) => setTitle(event.target.value)} placeholder="留空时显示 Prompt 概括" /></label>
          <div className="gallery-add-summary"><span><b>{selectedOutputs.length}</b> 图片</span><span><b>{target === "replace" || groupMode === "combined" ? selectedOutputs.length ? 1 : 0 : selectedOutputs.length}</b> 卡片</span><span><b>{settings?.loras?.filter((item) => item.enabled !== false).length || 0}</b> LoRA</span></div>
          {error && <p className="gallery-form-error">{error}</p>}
        </section>
      </div>
      <footer><span>保存为独立精选副本，完整参数与原 PNG 元数据都会保留。</span><button type="button" disabled={busy || !selectedOutputs.length || (!data.collections.length && !newCollectionId.trim()) || (target === "replace" && !replaceCardId)} onClick={() => void save()}>{busy ? <RefreshCw className="spin" size={15} /> : <Sparkles size={15} />}{target === "replace" ? "确认覆盖卡片" : "加入精选集"}</button></footer>
    </section>
  </div>;
}
