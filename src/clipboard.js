// Copy text to the clipboard, including from a page served over plain HTTP.
//
// `navigator.clipboard` only exists in a secure context -- HTTPS, or localhost. XiriaCanvas is
// routinely opened from another machine: over the LAN, or on a cloud GPU instance, over plain HTTP,
// where the API is simply undefined. The image reader's copy buttons called it through `?.`, so on
// such a page the whole chain short-circuited: nothing was copied and nothing said so. The Gallery
// and the LoRA details dialog each carried a private copy of the fallback below and worked; this is
// that fallback, in one place, so the next copy button does not have to remember it.
//
// The fallback is the selection-based `execCommand("copy")`. It is deprecated, but it predates the
// clipboard permission model and is still what every browser honours outside a secure context.

export async function copyText(value) {
  const text = String(value ?? "");
  if (!text) return false;
  if (globalThis.isSecureContext !== false && globalThis.navigator?.clipboard?.writeText) {
    try {
      await globalThis.navigator.clipboard.writeText(text);
      return true;
    } catch {
      // A denied permission or an expired user gesture: the older path can still succeed.
    }
  }
  return copyWithSelection(text);
}

function copyWithSelection(text) {
  const doc = globalThis.document;
  if (!doc?.body || typeof doc.execCommand !== "function") return false;
  const textarea = doc.createElement("textarea");
  textarea.value = text;
  // Read-only so a phone does not raise its keyboard for an element nobody can see.
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.top = "0";
  textarea.style.left = "0";
  textarea.style.opacity = "0";
  const previous = doc.activeElement;
  doc.body.appendChild(textarea);
  textarea.focus();
  textarea.select();
  // iOS ignores `select()` on a read-only field; an explicit range is what it honours.
  textarea.setSelectionRange?.(0, text.length);
  let copied = false;
  try {
    copied = doc.execCommand("copy") === true;
  } catch {
    copied = false;
  }
  textarea.remove();
  // Hand focus back, so copying from inside a dialog does not strand the dialog's focus on <body>.
  if (typeof previous?.focus === "function" && previous.isConnected !== false) previous.focus();
  return copied;
}
