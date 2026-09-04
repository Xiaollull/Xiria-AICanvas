import { useState } from "react";
import { ImagePlus, RefreshCw, Star, Trash2 } from "lucide-react";

import LoraImagePicker from "./LoraImagePicker.jsx";
import { MAXIMUM_SHOWCASE, cardAssetUrl } from "./lora-cards.js";

// Effect images for a saved combination.
//
// A group is the one thing in the LoRA library with no picture of its own: its
// members each have a preview from their source page, but what the combination
// actually produces only exists once the user has run it. So the pictures come
// from their own outputs or from a file they choose, and the first one is the
// cover — one ordered list rather than a list plus a cover pointer, so the cover
// can never name an image the gallery no longer holds.

export default function LoraGroupArtwork({ group, card, disabled = false, onPatch, onUploadImage, onNotice }) {
  const [picking, setPicking] = useState(false);
  const [saving, setSaving] = useState(false);
  const showcase = card.showcase;
  const full = showcase.length >= MAXIMUM_SHOWCASE;

  const addImage = async (source) => {
    setSaving(true);
    try {
      const id = await onUploadImage(source);
      // Re-adding a picture the group already holds would otherwise silently do
      // nothing: the id is a content hash, so the same image keeps the same id.
      if (showcase.includes(id)) onNotice?.("这张图片已经在该组合里了");
      else onPatch({ showcase: [...showcase, id] });
      setPicking(false);
    } catch (uploadError) {
      onNotice?.(uploadError.message);
    } finally {
      setSaving(false);
    }
  };

  const makeCover = (id) => onPatch({ showcase: [id, ...showcase.filter((entry) => entry !== id)] });
  const removeImage = (id) => onPatch({ showcase: showcase.filter((entry) => entry !== id) });

  return <div className="lora-group-artwork">
    <span className="lora-group-artwork-head">
      效果图
      <small>{showcase.length ? `${showcase.length} / ${MAXIMUM_SHOWCASE} · 第一张作为封面` : "还没有效果图"}</small>
    </span>
    <div className="lora-group-artwork-strip">
      {showcase.map((id, index) => (
        <figure key={id} className={index === 0 ? "cover" : ""}>
          <img src={cardAssetUrl(id)} alt="" loading="lazy" />
          {index === 0 && <figcaption>封面</figcaption>}
          <div>
            {index > 0 && <button type="button" disabled={disabled} title="设为封面" aria-label={`把这张图片设为 ${group.name} 的封面`} onClick={() => makeCover(id)}><Star size={12} /></button>}
            <button type="button" className="remove" disabled={disabled} title="移除这张图片" aria-label={`从 ${group.name} 移除这张效果图`} onClick={() => removeImage(id)}><Trash2 size={12} /></button>
          </div>
        </figure>
      ))}
      <button
        type="button"
        className="lora-group-artwork-add"
        disabled={disabled || saving || full}
        title={full ? `最多保存 ${MAXIMUM_SHOWCASE} 张效果图` : "从生成结果中选择，或上传一张图片"}
        onClick={() => setPicking(true)}
      >{saving ? <RefreshCw className="spin" size={16} /> : <ImagePlus size={16} />}<span>{full ? "已达上限" : "添加效果图"}</span></button>
    </div>
    {picking && <LoraImagePicker title={`为「${group.name}」添加效果图`} busy={saving} onPick={addImage} onClose={() => setPicking(false)} />}
  </div>;
}
