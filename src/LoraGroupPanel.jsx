import { useEffect, useRef, useState } from "react";
import { Check, CircleStop, FolderPlus, Layers3, MessageSquareText, Pencil, Plus, Target, Trash2, X } from "lucide-react";

import { normalizeMountedLoras } from "./lora-state.js";
import {
  MAXIMUM_GROUPS,
  MAXIMUM_GROUP_NAME,
  applyGroupEnabled,
  collectingGroupId,
  createLoraGroup,
  groupIdOfMountedLora,
  setCollectingGroup,
} from "./lora-groups.js";

// The group editor. A group is a saved combination: a name, a set of LoRAs with
// their weights, a preset positive prompt, and whether it is currently on.
//
// Enabling one mounts its members into the mounted queue; the mounted queue is
// still what generation reads, so nothing here is on the inference path except
// the preset prompt.

export default function LoraGroupPanel({ groups = [], loras = [], locked = false, onUpdateGroups, onUpdateLoras, onNotice }) {
  const [renaming, setRenaming] = useState("");
  const [nameDraft, setNameDraft] = useState("");
  const [expanded, setExpanded] = useState("");
  const [creating, setCreating] = useState(false);
  const createRef = useRef(null);

  const mounted = normalizeMountedLoras(loras);
  const standalone = mounted.filter((item) => !groupIdOfMountedLora(item, groups));
  const collecting = collectingGroupId(groups);

  // The choice menu is a transient surface; anything else the user does closes it.
  useEffect(() => {
    if (!creating) return undefined;
    const dismiss = (event) => { if (!createRef.current?.contains(event.target)) setCreating(false); };
    const escape = (event) => { if (event.key === "Escape") setCreating(false); };
    window.addEventListener("pointerdown", dismiss);
    window.addEventListener("keydown", escape);
    return () => {
      window.removeEventListener("pointerdown", dismiss);
      window.removeEventListener("keydown", escape);
    };
  }, [creating]);

  const patchGroup = (id, changes) => onUpdateGroups?.((current) => current.map((group) => group.id === id ? { ...group, ...changes } : group));

  /**
   * Toggling a group is one user action but two state changes — the flag and the
   * mounted queue — and they must not half-apply. The mount is computed first so
   * a group that would overflow the 16-entry cap is refused with the flag
   * untouched, rather than switching on and silently mounting nothing.
   */
  const toggleGroup = (group) => {
    const next = applyGroupEnabled(mounted, group, !group.enabled);
    if (next.overflow) {
      onNotice?.(`启用「${group.name}」会超过 16 个挂载上限，请先停用其它组或卸载部分 LoRA`);
      return;
    }
    onUpdateLoras?.(() => next.loras);
    patchGroup(group.id, { enabled: !group.enabled });
  };

  /**
   * Two intents, asked rather than assumed. The old single button silently took
   * whatever happened to be mounted, which is right about half the time and
   * surprising the rest.
   */
  const createGroup = ({ fromMounted }) => {
    setCreating(false);
    if (groups.length >= MAXIMUM_GROUPS) {
      onNotice?.(`最多只能保存 ${MAXIMUM_GROUPS} 个 LoRA 组合`);
      return;
    }
    const members = fromMounted ? standalone.map(({ groupId: _groupId, ...member }) => member) : [];
    const group = createLoraGroup({
      name: `LoRA 组 ${groups.length + 1}`,
      members,
      // Capturing the current mounts produces a finished group; an empty one is
      // a workbench, so it starts enabled and collecting whatever is mounted next.
      enabled: fromMounted,
      collecting: !fromMounted,
    }, groups.length);
    onUpdateGroups?.((current) => [...current, group]);
    if (fromMounted) {
      // The captured entries are already mounted; tagging them makes the group a
      // live view of them rather than a detached copy.
      const values = new Set(members.map((member) => member.value));
      onUpdateLoras?.((current) => current.map((item) => values.has(item.value) ? { ...item, groupId: group.id } : item));
    }
    setRenaming(group.id);
    setNameDraft(group.name);
    setExpanded(group.id);
    onNotice?.(fromMounted
      ? `已用 ${members.length} 个未分组 LoRA 新建组合，可继续重命名`
      : `已新建「${group.name}」，现在挂载的 LoRA 都会进入这个组合`);
  };

  const toggleCollecting = (group) => {
    const next = collecting === group.id ? "" : group.id;
    if (next) {
      // Collecting implies mounted: the group needs a block for new entries to
      // appear in, so switching it on also enables it.
      const mount = applyGroupEnabled(mounted, group, true);
      if (mount.overflow) {
        onNotice?.(`启用「${group.name}」会超过 16 个挂载上限，请先停用其它组或卸载部分 LoRA`);
        return;
      }
      if (mount.changed) onUpdateLoras?.(() => mount.loras);
    }
    onUpdateGroups?.((current) => setCollectingGroup(current, next));
    onNotice?.(next ? `接下来挂载的 LoRA 会加入「${group.name}」` : "已停止收集，新挂载的 LoRA 不再进入组合");
  };

  const removeGroup = (group) => {
    // Deleting an enabled group has to take its mounted entries with it,
    // otherwise its LoRAs stay mounted carrying a tag that no longer resolves.
    if (group.enabled) onUpdateLoras?.(() => applyGroupEnabled(mounted, group, false).loras);
    onUpdateGroups?.((current) => current.filter((entry) => entry.id !== group.id));
    onNotice?.(`已删除组合「${group.name}」`);
  };

  const addStandaloneTo = (group) => {
    if (!standalone.length) return;
    const members = [...group.members, ...standalone.map(({ groupId: _groupId, ...member }) => member)];
    patchGroup(group.id, { members });
    // An enabled group is a live view of the mounted queue, so its new members
    // have to acquire the tag immediately or the next write-back drops them.
    if (group.enabled) {
      const values = new Set(standalone.map((item) => item.value));
      onUpdateLoras?.((current) => current.map((item) => values.has(item.value) ? { ...item, groupId: group.id } : item));
    }
    onNotice?.(`已将 ${standalone.length} 个 LoRA 加入「${group.name}」`);
  };

  const removeMember = (group, value) => {
    patchGroup(group.id, { members: group.members.filter((member) => member.value !== value) });
    if (group.enabled) onUpdateLoras?.((current) => current.filter((item) => !(item.value === value && item.groupId === group.id)));
  };

  const commitName = (group) => {
    const name = nameDraft.trim().slice(0, MAXIMUM_GROUP_NAME);
    setRenaming("");
    if (name && name !== group.name) patchGroup(group.id, { name });
  };

  return (
    <div className="lora-group-panel">
      <div className="lora-group-head">
        <div>
          <h3>LoRA 组合</h3>
          <p>保存常用的 LoRA 搭配与预设提示词，启用后会挂载到挂载区，可同时启用多个组合。</p>
        </div>
        <div className="lora-group-create-wrap" ref={createRef}>
          <button type="button" className="lora-group-create" disabled={locked} aria-expanded={creating} aria-haspopup="menu" onClick={() => setCreating((current) => !current)}>
            <FolderPlus size={14} />新建组合
          </button>
          {creating && (
            <div className="lora-group-create-menu" role="menu">
              <button type="button" role="menuitem" disabled={!standalone.length} onClick={() => createGroup({ fromMounted: true })}>
                <Layers3 size={15} />
                <span>
                  <strong>收录当前挂载的 LoRA</strong>
                  <small>{standalone.length ? `把挂载区中 ${standalone.length} 个未分组的 LoRA 存成一个组合` : "挂载区没有未分组的 LoRA"}</small>
                </span>
              </button>
              <button type="button" role="menuitem" onClick={() => createGroup({ fromMounted: false })}>
                <Target size={15} />
                <span>
                  <strong>创建空组合并开始收集</strong>
                  <small>先建一个空组合，之后挂载的 LoRA 会自动加入，边试边调</small>
                </span>
              </button>
            </div>
          )}
        </div>
      </div>

      {collecting && (
        <p className="lora-group-collecting-banner">
          <Target size={13} />
          正在向「{groups.find((group) => group.id === collecting)?.name}」收集：现在挂载的 LoRA 都会加入该组合。
          <button type="button" disabled={locked} onClick={() => toggleCollecting(groups.find((group) => group.id === collecting))}><CircleStop size={12} />停止收集</button>
        </p>
      )}

      {!groups.length && (
        <div className="lora-library-empty">
          <Layers3 size={30} />
          <strong>还没有 LoRA 组合</strong>
          <p>点「新建组合」：可以收录当前挂载的 LoRA，也可以先建一个空组合，边挂载边试</p>
        </div>
      )}

      <div className="lora-group-list">
        {groups.map((group) => (
          <article className={`lora-group-card ${group.enabled ? "enabled" : ""} ${collecting === group.id ? "collecting" : ""}`} key={group.id}>
            <header>
              <button
                type="button"
                className={`mounted-toggle ${group.enabled ? "active" : ""}`}
                role="switch"
                aria-checked={group.enabled}
                disabled={locked}
                title={group.enabled ? "停用该组合" : "启用该组合"}
                onClick={() => toggleGroup(group)}
              ><i /></button>
              {renaming === group.id ? (
                <input
                  className="lora-group-name-input"
                  autoFocus
                  maxLength={MAXIMUM_GROUP_NAME}
                  value={nameDraft}
                  aria-label="组合名称"
                  onChange={(event) => setNameDraft(event.target.value)}
                  onBlur={() => commitName(group)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") { event.preventDefault(); commitName(group); }
                    if (event.key === "Escape") { event.preventDefault(); setRenaming(""); }
                  }}
                />
              ) : (
                <strong title={group.name}>{group.name}</strong>
              )}
              <small>{group.members.length} 个 LoRA{group.presetPrompt ? " · 含预设提示词" : ""}</small>
              <span className="lora-mount-spacer" />
              <button
                type="button"
                className={`lora-group-collect ${collecting === group.id ? "active" : ""}`}
                disabled={locked}
                aria-pressed={collecting === group.id}
                title={collecting === group.id ? "停止收集新挂载的 LoRA" : "把之后挂载的 LoRA 收集到该组合"}
                aria-label={collecting === group.id ? `停止向 ${group.name} 收集` : `向 ${group.name} 收集新挂载的 LoRA`}
                onClick={() => toggleCollecting(group)}
              >{collecting === group.id ? <CircleStop size={13} /> : <Target size={13} />}</button>
              <button type="button" disabled={locked} title="重命名" aria-label={`重命名 ${group.name}`} onClick={() => { setRenaming(group.id); setNameDraft(group.name); }}><Pencil size={13} /></button>
              <button type="button" disabled={locked} title="编辑内容" aria-expanded={expanded === group.id} aria-label={`编辑 ${group.name}`} onClick={() => setExpanded((current) => current === group.id ? "" : group.id)}>{expanded === group.id ? <Check size={13} /> : <Layers3 size={13} />}</button>
              <button type="button" className="remove" disabled={locked} title="删除组合" aria-label={`删除 ${group.name}`} onClick={() => removeGroup(group)}><Trash2 size={13} /></button>
            </header>

            {expanded === group.id && (
              <div className="lora-group-body">
                <label className="lora-group-prompt">
                  <span><MessageSquareText size={12} />预设正向提示词</span>
                  <textarea
                    rows={3}
                    spellCheck={false}
                    disabled={locked}
                    value={group.presetPrompt}
                    placeholder="启用该组合时，这段提示词会加在正向提示词最前面"
                    onChange={(event) => patchGroup(group.id, { presetPrompt: event.target.value })}
                  />
                  <small>生成页的提示词输入框不会被改写；多个组合按此列表顺序依次拼接。</small>
                </label>

                <div className="lora-group-members">
                  {!group.members.length && <p className="lora-group-empty-members">{collecting === group.id ? "该组合还是空的 — 去分类里挂载 LoRA，它们会自动加进来。" : "该组合还没有 LoRA。"}</p>}
                  {group.members.map((member) => (
                    <div className="lora-group-member" key={member.value}>
                      <strong title={member.name}>{member.name}</strong>
                      <small>{member.category}</small>
                      <b>{member.weight}</b>
                      <button type="button" disabled={locked} title="从组合中移除" aria-label={`从 ${group.name} 中移除 ${member.name}`} onClick={() => removeMember(group, member.value)}><X size={12} /></button>
                    </div>
                  ))}
                </div>

                <button type="button" className="lora-group-add" disabled={locked || !standalone.length} onClick={() => addStandaloneTo(group)}>
                  <Plus size={13} />加入当前未分组的 {standalone.length} 个 LoRA
                </button>
              </div>
            )}
          </article>
        ))}
      </div>
    </div>
  );
}
