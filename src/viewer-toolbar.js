export const VIEWER_TOOLBAR_POPOVER_NONE = "none";
export const VIEWER_TOOLBAR_POPOVER_LAYOUT = "layout";
export const VIEWER_TOOLBAR_POPOVER_TEMPLATES = "templates";

export function viewerToolbarPopoverTransition(current, target, reason = "toggle") {
  const requested = target === VIEWER_TOOLBAR_POPOVER_LAYOUT || target === VIEWER_TOOLBAR_POPOVER_TEMPLATES ? target : VIEWER_TOOLBAR_POPOVER_NONE;
  if (requested === current) return { next: requested, focusTarget: null, focusReturn: null };
  if (requested !== VIEWER_TOOLBAR_POPOVER_NONE) return { next: requested, focusTarget: requested, focusReturn: null };
  return { next: VIEWER_TOOLBAR_POPOVER_NONE, focusTarget: null, focusReturn: reason === "context-menu" || reason === "viewer-close" || reason === "unmount" ? null : current };
}

export function viewerToolbarPopoverState(current, action) {
  return viewerToolbarPopoverTransition(current, current === action ? VIEWER_TOOLBAR_POPOVER_NONE : action).next;
}

export function viewerEscapeAction({ historyDelete, contextMenu, popover, historyBatch }) {
  if (historyDelete) return "historyDelete";
  if (contextMenu) return "contextMenu";
  if (popover === VIEWER_TOOLBAR_POPOVER_LAYOUT) return "layout";
  if (popover === VIEWER_TOOLBAR_POPOVER_TEMPLATES) return "templates";
  if (historyBatch) return "historyBatch";
  return "viewer";
}
