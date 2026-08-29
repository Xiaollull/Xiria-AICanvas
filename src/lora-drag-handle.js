// Native HTML drag-and-drop has no trustworthy global "currently dragging"
// signal. Keep both source and transfer proof explicit so mounted-row controls
// remain ordinary inputs and buttons rather than accidental drag sources.
export const LORA_SORT_TRANSFER_TYPE = "application/x-xiria-lora-sort";

export function appLoraDragLocked({ status, modelSwitching, loraWorkspaceLocked, shouldPersistMountedLoras }) {
  return status === "running" || Boolean(modelSwitching) || Boolean(loraWorkspaceLocked) || !shouldPersistMountedLoras;
}

export function pageLoraDragLocked({ workspaceLocked, syncReady, canPersist, scopeKey }) {
  return Boolean(workspaceLocked) || !syncReady || !canPersist || !scopeKey;
}

// `event.target` may be the GripVertical SVG or its inner path. It is legal
// whenever it is inside the explicit marked handle; requiring target === button
// would make the visible six-dot icon impossible to drag.
export function isLoraDragHandleSource(event) {
  const handle = event?.currentTarget;
  const target = event?.target;
  if (handle?.dataset?.loraDragHandle !== "true" || !target) return false;
  try {
    const markedHandle = target.closest?.("[data-lora-drag-handle]");
    if (markedHandle) return markedHandle === handle;
  } catch {}
  return target === handle || Boolean(handle.contains?.(target));
}

// The semantic button provides focus and an accessible name. Native HTML DnD
// has no keyboard reorder protocol, so activation keys deliberately do nothing.
export function suppressLoraDragHandleKeyboard(event) {
  if (event?.key !== "Enter" && event?.key !== " ") return false;
  event.preventDefault?.();
  return true;
}

function createLoraDragNonce() {
  const crypto = globalThis.crypto;
  if (!crypto || typeof crypto.getRandomValues !== "function") return null;
  const bytes = new Uint8Array(18);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (value) => value.toString(16).padStart(2, "0")).join("");
}

function transferValueFor({ index, sourceValue, nonce }) {
  return JSON.stringify({ index, sourceValue, nonce });
}

function transferTypesInclude(transfer, type) {
  const types = transfer?.types;
  if (!types) return false;
  try {
    if (typeof types.contains === "function" && types.contains(type)) return true;
    if (typeof types.includes === "function" && types.includes(type)) return true;
    return Array.from(types).includes(type);
  } catch {
    return false;
  }
}

function readLoraTransferValue(transfer) {
  if (!transfer || typeof transfer.getData !== "function") return { readable: false, value: "" };
  try {
    const value = transfer.getData(LORA_SORT_TRANSFER_TYPE);
    // Browser privacy rules may deliberately hide drag data before drop. An
    // empty/non-string read is therefore unknown at dragover, never a proof.
    return typeof value === "string" && value ? { readable: true, value } : { readable: false, value: "" };
  } catch {
    return { readable: false, value: "" };
  }
}

export function establishLoraDragSession({ event, index, sourceValue, items, locked }) {
  if (locked || !Array.isArray(items) || !Number.isInteger(index) || index < 0 || index >= items.length || typeof sourceValue !== "string" || !sourceValue || items[index]?.value !== sourceValue || !isLoraDragHandleSource(event)) return null;
  const transfer = event?.dataTransfer;
  if (!transfer || typeof transfer.setData !== "function") return null;
  const nonce = createLoraDragNonce();
  if (!nonce) return null;
  const session = { index, sourceValue, nonce };
  const transferValue = transferValueFor(session);
  try {
    transfer.effectAllowed = "move";
    transfer.setData(LORA_SORT_TRANSFER_TYPE, transferValue);
    // Supplemental only: admission never trusts text/plain.
    transfer.setData("text/plain", sourceValue);
  } catch {
    return null;
  }
  return { ...session, transferValue };
}

export function isValidLoraDragSession({ session, items, locked }) {
  return !locked
    && Array.isArray(items)
    && Number.isInteger(session?.index)
    && session.index >= 0
    && session.index < items.length
    && typeof session.sourceValue === "string"
    && items[session.index]?.value === session.sourceValue
    && typeof session.nonce === "string"
    && /^[0-9a-f]{36}$/.test(session.nonce)
    && session.transferValue === transferValueFor(session);
}

function hasValidLoraDropPosition({ session, items, targetIndex, locked }) {
  return isValidLoraDragSession({ session, items, locked })
    && Number.isInteger(targetIndex)
    && targetIndex >= 0
    && targetIndex < items.length;
}

// Dragover data reads are hidden by some native browsers. When the private MIME
// type is visible but its value is unreadable, current-session feedback is safe
// to show; the later drop still requires the exact value and cannot reorder.
export function acceptsLoraDragOver({ session, items, targetIndex, locked, dataTransfer }) {
  if (!hasValidLoraDropPosition({ session, items, targetIndex, locked }) || !transferTypesInclude(dataTransfer, LORA_SORT_TRANSFER_TYPE)) return false;
  const transfer = readLoraTransferValue(dataTransfer);
  return !transfer.readable || transfer.value === session.transferValue;
}

// Drop is the commit boundary: a current unlocked session and the exact private
// MIME payload (index, source value and nonce) are all mandatory.
export function isValidLoraDropTarget({ session, items, targetIndex, locked, dataTransfer }) {
  if (!hasValidLoraDropPosition({ session, items, targetIndex, locked }) || !transferTypesInclude(dataTransfer, LORA_SORT_TRANSFER_TYPE)) return false;
  const transfer = readLoraTransferValue(dataTransfer);
  return transfer.readable && transfer.value === session.transferValue;
}

export function shouldCommitLoraDrop(options) {
  return isValidLoraDropTarget(options) && options.session.index !== options.targetIndex;
}

export function reorderLoraItems(items, fromIndex, toIndex) {
  if (!Array.isArray(items) || !Number.isInteger(fromIndex) || !Number.isInteger(toIndex) || fromIndex < 0 || toIndex < 0 || fromIndex >= items.length || toIndex >= items.length || fromIndex === toIndex) return items;
  const next = [...items];
  const [moved] = next.splice(fromIndex, 1);
  next.splice(toIndex, 0, moved);
  return next;
}
