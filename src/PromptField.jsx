import { Sparkles } from "lucide-react";

/** The i2i mother field, including its real textarea, count and watermark. */
export default function PromptField({ type, value, onChange, disabled = false, supported = true, transparent = false, placeholder, textareaProps = {} }) {
  const positive = type === "positive";
  return <label className={`prompt-field ${positive ? "positive-field" : "negative-field"}${transparent ? " transparent-enabled" : ""}`}>
    <div><span>{positive ? "正向提示词" : "反向提示词"}{transparent && <b className="special-tag-mark">TRANSPARENT PNG</b>}</span><small>{supported ? `${value.length} 字符` : "当前引擎不使用"}</small></div>
    <textarea {...textareaProps} value={value} disabled={disabled || !supported} placeholder={placeholder} spellCheck={false} onChange={onChange} />
    {positive && <Sparkles className="field-watermark" size={46} />}
  </label>;
}
