import PromptField from "./PromptField.jsx";
import { PromptPresetStrip } from "./PromptPresetWindow.jsx";
import { XYZ_CAPABILITIES } from "./xyz-draft.js";
import { patchXYZPrompt, xyzPromptSource } from "./xyz-prompt-source.js";

export default function XYZPromptDeck({ draft, container, open, module, libraryError, onOpen, onToggle, onChange, kicker = "XYZ 独立草稿" }) {
  const source = xyzPromptSource(draft), supported = XYZ_CAPABILITIES.engines[draft.engine].negative_prompt;
  return <div className="i2i-prompt-deck xyz-prompt-deck" aria-label="XYZ 提示词">
    <div className="section-heading"><span>01</span><h2>提示词</h2><small className="i2i-section-kicker">{kicker}</small></div>
    {["positive", "negative"].map((type) => <div className="i2i-prompt-column" key={type}>
      <PromptField type={type} value={source[type]} supported={type === "positive" || supported}
        textareaProps={{ "aria-label": `XYZ 基线${type === "positive" ? "正向" : "负向"}提示词` }}
        placeholder={type === "positive" ? "描述画面主体、环境、光线与风格..." : supported ? "不希望出现的内容" : "当前引擎不使用负向提示词；已保存文本保留"}
        onChange={(event) => onChange(patchXYZPrompt(draft, type, event.target.value))} />
      <PromptPresetStrip type={type} container={container} activeIds={source.active[type]} open={open && module === type} libraryError={libraryError}
        disabled={type === "negative" && !supported} onOpen={onOpen} onToggle={onToggle} />
    </div>)}
    {source.mode === "legacy_actual" && <p className="xyz-legacy-prompt-note" role="status">旧草稿按已组合的实际文本保留，不猜测删除历史组合词。选择预设会新增拼接；可先编辑文本，或明确同步基础提示词来源。</p>}
  </div>;
}
