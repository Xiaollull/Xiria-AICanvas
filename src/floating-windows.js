// Which floating window is in front.
//
// The task queue and the prompt preset window can be open over the same page at once. Like windows
// on a desktop, the one opened, clicked, focused or dragged most recently is drawn on top; the
// others keep their order beneath it. The stack is one module-level list shared by every window —
// they are portalled into the document from different components, so no common parent owns them —
// and each window asks for its layer by id. Layers start at the page's popover layer and stay below
// dialogs, menus and hover previews, which must cover every floating window.

import { useCallback, useEffect, useSyncExternalStore } from "react";

/** `--z-popover`, the layer the floating windows have always been drawn on. */
export const FLOATING_WINDOW_BASE_Z = 45;

let stack = [];
const listeners = new Set();

function publish(next) {
  stack = next;
  for (const listener of listeners) listener();
}

/** The stack with `id` moved to the top. Unchanged (the same array) when it already is on top. */
export function raiseInStack(current, id) {
  return current.at(-1) === id ? current : [...current.filter((item) => item !== id), id];
}

/** The stack without `id`. */
export function removeFromStack(current, id) {
  return current.includes(id) ? current.filter((item) => item !== id) : current;
}

/** The z-index a window is drawn at: its position in the stack above the base layer. */
export function floatingWindowZIndex(current, id) {
  const index = current.indexOf(id);
  return FLOATING_WINDOW_BASE_Z + (index < 0 ? current.length : index);
}

export function raiseFloatingWindow(id) {
  const next = raiseInStack(stack, id);
  if (next !== stack) publish(next);
}

export function releaseFloatingWindow(id) {
  const next = removeFromStack(stack, id);
  if (next !== stack) publish(next);
}

function subscribe(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

const snapshot = () => stack;

/**
 * A floating window's place in the stack. It comes to the top when it opens; spread `bind` onto
 * its root so a press or focus anywhere inside it — the header it is dragged by included — raises
 * it again.
 */
export function useFloatingWindowLayer(id) {
  const current = useSyncExternalStore(subscribe, snapshot, snapshot);
  useEffect(() => {
    raiseFloatingWindow(id);
    return () => releaseFloatingWindow(id);
  }, [id]);
  const raise = useCallback(() => raiseFloatingWindow(id), [id]);
  return {
    zIndex: floatingWindowZIndex(current, id),
    raise,
    bind: { onPointerDownCapture: raise, onFocusCapture: raise },
  };
}
