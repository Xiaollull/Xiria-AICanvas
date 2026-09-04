import { useState } from "react";

// A text field that holds a draft while it is focused and commits when it loses
// focus.
//
// Every store behind these boxes trims and collapses what it is given, which a
// controlled input bound straight to the stored value cannot survive: type a
// space at the end and normalization removes it before the next render, so the
// character never appears and the next word can never be started. The same bug
// is why the weight box grew a draft of its own.
//
// Escape reverts to the stored value. Enter commits a single-line field; in a
// textarea it stays a newline.

/** No draft means the box shows what is stored; a draft means it shows itself. */
export function draftText(value, draft) {
  return draft === null || draft === undefined ? value : draft;
}

/**
 * What losing focus should do. The comparison is against the stored value, not
 * against the draft's own history, so opening a dialog and closing it again
 * cannot mark the store dirty.
 */
export function draftCommit(value, draft) {
  const text = draftText(value, draft);
  return { text, changed: text !== value };
}

export function useDraftField(value, onCommit, { multiline = false } = {}) {
  const [draft, setDraft] = useState(null);
  const text = draftText(value, draft);

  const commit = () => {
    const outcome = draftCommit(value, draft);
    setDraft(null);
    if (outcome.changed) onCommit(outcome.text);
  };

  return {
    value: text,
    dirty: draft !== null && draft !== value,
    onChange: (event) => setDraft(event.target.value),
    onBlur: commit,
    onKeyDown: (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        setDraft(null);
        return;
      }
      if (!multiline && event.key === "Enter") {
        event.preventDefault();
        event.currentTarget.blur();
      }
    },
  };
}

/**
 * Whether a key event came from a text box, so a dialog listening for Escape on
 * the window can let the field revert itself instead of closing over the top of
 * it. The dialog's listener is on the capture phase and would otherwise always
 * win.
 */
export function fromTextField(event) {
  const target = event?.target;
  // Duck typed rather than `instanceof HTMLElement`: this is called from a
  // capture listener that sees every key event, including ones whose target is
  // the document or a node from another realm.
  return typeof target?.closest === "function"
    && Boolean(target.closest("input[type='text'], input:not([type]), textarea"));
}
