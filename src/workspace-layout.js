// Layout of the generate page: how wide the left configuration panel is, and whether it is
// collapsed at all.
//
// Deliberately kept out of `workspace` in `ui-state.json`, which holds creation parameters. The
// reset control has to restore the layout *without* touching a single parameter, and the strongest
// guarantee of that is not a careful implementation but a store the parameters do not share: this
// lives in one browser-local key of its own, so resetting it cannot reach them. It is also the
// right home on its own merits — a comfortable panel width depends on the monitor in front of you,
// not on the project.

export const WORKSPACE_LAYOUT_STORAGE_KEY = "xirai-workspace-layout-v1";
export const WORKSPACE_LAYOUT_SCHEMA_VERSION = 1;

// Below the minimum the panel's own controls stop fitting; above the maximum it starts crowding the
// composer it exists to support. Enforced here rather than in the drag handler, so a hand-edited
// storage value is bounded by the same rule as a gesture.
export const LEFT_PANEL_MINIMUM_WIDTH = 248;
export const LEFT_PANEL_MAXIMUM_WIDTH = 620;
// Dragging the panel shut is the same intent as pressing the collapse toggle, so a gesture that
// crosses this point collapses instead of clamping to the minimum and refusing to go further.
export const LEFT_PANEL_COLLAPSE_WIDTH = 176;
export const LEFT_PANEL_WIDTH_STEP = 16;

// `leftWidth: 0` means "whatever the stylesheet lays out". That is what keeps the default width
// exactly the current one: no number in this file has to be kept in sync with the grid template,
// and a build that changes the template changes the default with it.
export const DEFAULT_WORKSPACE_LAYOUT = Object.freeze({ leftWidth: 0, leftCollapsed: false });

export function clampLeftPanelWidth(value) {
  const numeric = typeof value === "number" ? value : Number.parseFloat(String(value ?? ""));
  if (!Number.isFinite(numeric) || numeric <= 0) return 0;
  return Math.round(Math.min(LEFT_PANEL_MAXIMUM_WIDTH, Math.max(LEFT_PANEL_MINIMUM_WIDTH, numeric)));
}

// Never throws: a damaged value falls back to the default layout rather than leaving the workspace
// with no columns. Layout is a convenience over a page that works fine without it.
export function normalizeWorkspaceLayout(value) {
  const source = value && typeof value === "object" && !Array.isArray(value) ? value : {};
  return {
    leftWidth: clampLeftPanelWidth(source.leftWidth),
    leftCollapsed: source.leftCollapsed === true,
  };
}

export function isDefaultWorkspaceLayout(layout) {
  const normalized = normalizeWorkspaceLayout(layout);
  return normalized.leftWidth === DEFAULT_WORKSPACE_LAYOUT.leftWidth
    && normalized.leftCollapsed === DEFAULT_WORKSPACE_LAYOUT.leftCollapsed;
}

// `startWidth` is measured at pointer-down rather than read from the layout, because until the
// panel has been resized once its width is the stylesheet's and this module does not know it.
// Every position is computed from that starting rect, so a clamped drag cannot accumulate drift
// between the pointer and the edge it is holding.
export function resizeLeftPanel(layout, startWidth, deltaX) {
  const current = normalizeWorkspaceLayout(layout);
  const target = (Number.isFinite(startWidth) ? startWidth : 0) + (Number.isFinite(deltaX) ? deltaX : 0);
  if (target < LEFT_PANEL_COLLAPSE_WIDTH) {
    // The width it had is remembered, so expanding returns there rather than to the default.
    return { leftWidth: clampLeftPanelWidth(startWidth) || current.leftWidth, leftCollapsed: true };
  }
  return { leftWidth: clampLeftPanelWidth(target), leftCollapsed: false };
}

// Keyboard resizing from the handle. `measuredWidth` plays the same role as `startWidth` above:
// the first press has to start from the width the stylesheet chose.
export function steppedLeftPanel(layout, direction, measuredWidth) {
  const current = normalizeWorkspaceLayout(layout);
  if (current.leftCollapsed) {
    // The first keystroke out of a collapsed panel reopens it rather than nudging a hidden edge.
    return direction > 0 ? { ...current, leftCollapsed: false } : current;
  }
  const base = current.leftWidth || clampLeftPanelWidth(measuredWidth) || LEFT_PANEL_MINIMUM_WIDTH;
  return { leftWidth: clampLeftPanelWidth(base + direction * LEFT_PANEL_WIDTH_STEP), leftCollapsed: false };
}

export function toggleLeftPanel(layout) {
  const current = normalizeWorkspaceLayout(layout);
  return { ...current, leftCollapsed: !current.leftCollapsed };
}

// The class marks *which* rule applies, and the custom property carries the width. A layout that
// has never been resized emits neither, so the untouched grid template is what lays the page out
// and the default cannot drift away from what shipped.
export function workspaceLayoutClassName(layout) {
  const normalized = normalizeWorkspaceLayout(layout);
  if (normalized.leftCollapsed) return "workspace left-collapsed";
  return normalized.leftWidth ? "workspace left-sized" : "workspace";
}

export function workspaceLayoutStyle(layout) {
  const normalized = normalizeWorkspaceLayout(layout);
  if (normalized.leftCollapsed || !normalized.leftWidth) return undefined;
  return { "--left-panel-width": `${normalized.leftWidth}px` };
}

export function readWorkspaceLayout(storage) {
  try {
    const saved = storage?.getItem(WORKSPACE_LAYOUT_STORAGE_KEY);
    if (!saved) return { ...DEFAULT_WORKSPACE_LAYOUT };
    return normalizeWorkspaceLayout(JSON.parse(saved));
  } catch {
    // A blocked or corrupt storage costs the remembered layout, never the page.
    return { ...DEFAULT_WORKSPACE_LAYOUT };
  }
}

// The default is stored as *absence*, so resetting leaves nothing behind to be read back.
export function writeWorkspaceLayout(storage, layout) {
  const normalized = normalizeWorkspaceLayout(layout);
  try {
    if (isDefaultWorkspaceLayout(normalized)) storage?.removeItem(WORKSPACE_LAYOUT_STORAGE_KEY);
    else storage?.setItem(WORKSPACE_LAYOUT_STORAGE_KEY, JSON.stringify({ schemaVersion: WORKSPACE_LAYOUT_SCHEMA_VERSION, ...normalized }));
  } catch {
    // Same reasoning as the read: a layout that cannot be remembered is not a failed session.
  }
  return normalized;
}

// The image workspace owns a separate right-hand rail. Keeping its state in a different key means
// a comfortable image-to-image layout never changes the generate page's left configuration panel.
export const IMAGE_WORKSPACE_LAYOUT_STORAGE_KEY = "xirai-image-workspace-layout-v1";
export const IMAGE_WORKSPACE_LAYOUT_SCHEMA_VERSION = 1;
export const IMAGE_CONTROLS_MINIMUM_WIDTH = 300;
export const IMAGE_CONTROLS_MAXIMUM_WIDTH = 540;
export const IMAGE_CONTROLS_COLLAPSE_WIDTH = 208;
export const IMAGE_CONTROLS_WIDTH_STEP = 16;
export const DEFAULT_IMAGE_WORKSPACE_LAYOUT = Object.freeze({ controlsWidth: 0, controlsCollapsed: false });

export function clampImageControlsWidth(value) {
  const numeric = typeof value === "number" ? value : Number.parseFloat(String(value ?? ""));
  if (!Number.isFinite(numeric) || numeric <= 0) return 0;
  return Math.round(Math.min(IMAGE_CONTROLS_MAXIMUM_WIDTH, Math.max(IMAGE_CONTROLS_MINIMUM_WIDTH, numeric)));
}

export function normalizeImageWorkspaceLayout(value) {
  const source = value && typeof value === "object" && !Array.isArray(value) ? value : {};
  return {
    controlsWidth: clampImageControlsWidth(source.controlsWidth),
    controlsCollapsed: source.controlsCollapsed === true,
  };
}

export function isDefaultImageWorkspaceLayout(layout) {
  const normalized = normalizeImageWorkspaceLayout(layout);
  return normalized.controlsWidth === DEFAULT_IMAGE_WORKSPACE_LAYOUT.controlsWidth
    && normalized.controlsCollapsed === DEFAULT_IMAGE_WORKSPACE_LAYOUT.controlsCollapsed;
}

// The rail is on the right, so moving its left seam left increases its width. As with the generate
// page, every movement starts from the pointer-down width so a clamped edge cannot accumulate drift.
export function resizeImageControlsPanel(layout, startWidth, deltaX) {
  const current = normalizeImageWorkspaceLayout(layout);
  const target = (Number.isFinite(startWidth) ? startWidth : 0) - (Number.isFinite(deltaX) ? deltaX : 0);
  if (target < IMAGE_CONTROLS_COLLAPSE_WIDTH) {
    return { controlsWidth: clampImageControlsWidth(startWidth) || current.controlsWidth, controlsCollapsed: true };
  }
  return { controlsWidth: clampImageControlsWidth(target), controlsCollapsed: false };
}

export function steppedImageControlsPanel(layout, direction, measuredWidth) {
  const current = normalizeImageWorkspaceLayout(layout);
  if (current.controlsCollapsed) return direction > 0 ? { ...current, controlsCollapsed: false } : current;
  const base = current.controlsWidth || clampImageControlsWidth(measuredWidth) || IMAGE_CONTROLS_MINIMUM_WIDTH;
  return { controlsWidth: clampImageControlsWidth(base + direction * IMAGE_CONTROLS_WIDTH_STEP), controlsCollapsed: false };
}

export function toggleImageControlsPanel(layout) {
  const current = normalizeImageWorkspaceLayout(layout);
  return { ...current, controlsCollapsed: !current.controlsCollapsed };
}

export function imageWorkspaceLayoutClassName(layout) {
  const normalized = normalizeImageWorkspaceLayout(layout);
  if (normalized.controlsCollapsed) return "image-workspace controls-collapsed";
  return normalized.controlsWidth ? "image-workspace controls-sized" : "image-workspace";
}

export function imageWorkspaceLayoutStyle(layout) {
  const normalized = normalizeImageWorkspaceLayout(layout);
  if (normalized.controlsCollapsed || !normalized.controlsWidth) return undefined;
  return { "--i2i-controls-width": `${normalized.controlsWidth}px` };
}

export function readImageWorkspaceLayout(storage) {
  try {
    const saved = storage?.getItem(IMAGE_WORKSPACE_LAYOUT_STORAGE_KEY);
    if (!saved) return { ...DEFAULT_IMAGE_WORKSPACE_LAYOUT };
    return normalizeImageWorkspaceLayout(JSON.parse(saved));
  } catch {
    return { ...DEFAULT_IMAGE_WORKSPACE_LAYOUT };
  }
}

export function writeImageWorkspaceLayout(storage, layout) {
  const normalized = normalizeImageWorkspaceLayout(layout);
  try {
    if (isDefaultImageWorkspaceLayout(normalized)) storage?.removeItem(IMAGE_WORKSPACE_LAYOUT_STORAGE_KEY);
    else storage?.setItem(IMAGE_WORKSPACE_LAYOUT_STORAGE_KEY, JSON.stringify({ schemaVersion: IMAGE_WORKSPACE_LAYOUT_SCHEMA_VERSION, ...normalized }));
  } catch {
    // A browser that blocks storage still has a functional, session-only layout.
  }
  return normalized;
}
