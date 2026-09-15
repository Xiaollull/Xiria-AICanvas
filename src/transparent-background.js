export const TRANSPARENT_BACKGROUND_TAG = "({Transparent background})";
export const TRANSPARENT_BACKGROUND_PATTERN = /\(\{\s*transparent\s+background\s*\}\)/i;
export const TRANSPARENT_BACKGROUND_PATTERN_ALL = /\(\{\s*transparent\s+background\s*\}\)/gi;

export function transparentBackgroundEnabled(prompt) {
  return TRANSPARENT_BACKGROUND_PATTERN.test(String(prompt || ""));
}

export function transparentBackgroundSubject(prompt) {
  return String(prompt || "").replace(TRANSPARENT_BACKGROUND_PATTERN_ALL, "").replace(/^\s*,\s*|\s*,\s*$/g, "").trim();
}

export function toggleTransparentBackground(prompt) {
  const current = String(prompt || "");
  return transparentBackgroundEnabled(current)
    ? current.replace(TRANSPARENT_BACKGROUND_PATTERN_ALL, "").replace(/\s*,\s*,+/g, ", ").replace(/^\s*,\s*|\s*,\s*$/g, "").trim()
    : `${current.trim()}${current.trim() ? ", " : ""}${TRANSPARENT_BACKGROUND_TAG}`;
}
