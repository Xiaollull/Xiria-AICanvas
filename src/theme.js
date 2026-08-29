// Theme palette derivation and document application. Shared by the workspace and the manual update
// page so a lazily loaded route can paint the saved palette without pulling in the whole workspace.
export const LEGACY_DEFAULT_ACCENT = "#D6FF3F";
// The default the withdrawn 2026-08-15 visual pass shipped. It is kept only as a migration source:
// a stored record matching it in all five roles was written by that default rather than chosen by
// anyone, so it follows the default back to violet. A vermilion a user picked does not match in
// every role and is left alone.
export const WITHDRAWN_DEFAULT_THEME = {
  overall: "#D96846",
  accent: "#D96846",
  ink: "#111112",
  disc: "#FFF9F2",
  guide: "#E3C7BD",
};
export const DEFAULT_THEME = {
  overall: "#8B7CFF",
  accent: "#8B7CFF",
  ink: "#111112",
  disc: "#FFFFFF",
  guide: "#C9C5E8",
  light: false,
};

export function normalizeHex(value, fallback = null) {
  const match = typeof value === "string" && value.trim().match(/^#?([0-9a-f]{3}|[0-9a-f]{6})$/i);
  if (!match) return fallback;
  const digits = match[1].length === 3 ? [...match[1]].map((digit) => digit + digit).join("") : match[1];
  return `#${digits.toUpperCase()}`;
}

export function hexRgb(hex) {
  return [1, 3, 5].map((index) => parseInt(hex.slice(index, index + 2), 16));
}

export function rgbToHsl(hex) {
  const channels = hexRgb(hex).map((value) => value / 255);
  const maximum = Math.max(...channels);
  const minimum = Math.min(...channels);
  const delta = maximum - minimum;
  let hue = 0;
  if (delta) {
    if (maximum === channels[0]) hue = ((channels[1] - channels[2]) / delta) % 6;
    else if (maximum === channels[1]) hue = (channels[2] - channels[0]) / delta + 2;
    else hue = (channels[0] - channels[1]) / delta + 4;
    hue = (hue * 60 + 360) % 360;
  }
  const lightness = (maximum + minimum) / 2;
  const saturation = delta ? delta / (1 - Math.abs(2 * lightness - 1)) : 0;
  return [hue, saturation, lightness];
}

export function hslToHex(hue, saturation, lightness) {
  const chroma = (1 - Math.abs(2 * lightness - 1)) * saturation;
  const section = hue / 60;
  const secondary = chroma * (1 - Math.abs(section % 2 - 1));
  const values = section < 1 ? [chroma, secondary, 0]
    : section < 2 ? [secondary, chroma, 0]
      : section < 3 ? [0, chroma, secondary]
        : section < 4 ? [0, secondary, chroma]
          : section < 5 ? [secondary, 0, chroma]
            : [chroma, 0, secondary];
  const offset = lightness - chroma / 2;
  return `#${values.map((value) => Math.round((value + offset) * 255).toString(16).padStart(2, "0")).join("").toUpperCase()}`;
}

export function mixHex(first, second, amount) {
  const mixed = hexRgb(first).map((value, index) => Math.round(value + (hexRgb(second)[index] - value) * amount));
  return `#${mixed.map((value) => value.toString(16).padStart(2, "0")).join("").toUpperCase()}`;
}

export function colorLuminance(hex) {
  const channels = hexRgb(hex).map((value) => {
    const channel = value / 255;
    return channel <= .04045 ? channel / 12.92 : ((channel + .055) / 1.055) ** 2.4;
  });
  return .2126 * channels[0] + .7152 * channels[1] + .0722 * channels[2];
}

export function colorContrast(first, second) {
  const [lighter, darker] = [colorLuminance(first), colorLuminance(second)].sort((a, b) => b - a);
  return (lighter + .05) / (darker + .05);
}

export function accentContrastColor(color) {
  return colorContrast(color, "#111112") >= colorContrast(color, "#FFFFFF") ? "#111112" : "#FFFFFF";
}

export function readableAccentText(color) {
  let candidate = color;
  for (let amount = .08; amount <= .72 && colorContrast(candidate, "#FFFFFF") < 4.5; amount += .08) {
    candidate = mixHex(color, "#111112", amount);
  }
  return candidate;
}

export function deriveLogoTheme(seed) {
  const normalized = normalizeHex(seed, DEFAULT_THEME.overall);
  if (normalized === DEFAULT_THEME.overall) return {
    accent: DEFAULT_THEME.accent,
    ink: DEFAULT_THEME.ink,
    disc: DEFAULT_THEME.disc,
    guide: DEFAULT_THEME.guide,
  };
  const [hue, saturation, lightness] = rgbToHsl(normalized);
  const chromaticSaturation = saturation < .08 ? 0 : Math.max(.55, saturation);
  const accent = hslToHex(hue, chromaticSaturation, Math.min(.68, Math.max(.56, lightness)));
  const ink = hslToHex((hue + 180) % 360, saturation < .08 ? 0 : Math.min(.14, saturation * .14), .065);
  const disc = hslToHex(hue, saturation < .08 ? 0 : Math.min(.12, saturation * .12), .985);
  const guideTarget = hslToHex(hue, saturation < .08 ? 0 : Math.min(.22, saturation * .22), .22);
  return { accent, ink, disc, guide: mixHex(disc, guideTarget, .26) };
}

export function loadThemeState(saved) {
  try {
    if (!saved || typeof saved !== "object") return { ...DEFAULT_THEME };
    const overall = normalizeHex(saved.overall, DEFAULT_THEME.overall);
    const legacyDefault = overall === LEGACY_DEFAULT_ACCENT
      && normalizeHex(saved.accent, LEGACY_DEFAULT_ACCENT) === LEGACY_DEFAULT_ACCENT
      && normalizeHex(saved.ink, "#111112") === "#111112"
      && normalizeHex(saved.disc, "#FFFFFF") === "#FFFFFF"
      && normalizeHex(saved.guide, "#C9CCBE") === "#C9CCBE";
    const withdrawnDefault = overall === WITHDRAWN_DEFAULT_THEME.overall
      && normalizeHex(saved.accent, DEFAULT_THEME.accent) === WITHDRAWN_DEFAULT_THEME.accent
      && normalizeHex(saved.ink, DEFAULT_THEME.ink) === WITHDRAWN_DEFAULT_THEME.ink
      && normalizeHex(saved.disc, DEFAULT_THEME.disc) === WITHDRAWN_DEFAULT_THEME.disc
      && normalizeHex(saved.guide, DEFAULT_THEME.guide) === WITHDRAWN_DEFAULT_THEME.guide;
    if (legacyDefault || withdrawnDefault) return { ...DEFAULT_THEME, light: saved.light === true || saved.monochrome === true };
    return {
      overall,
      accent: normalizeHex(saved.accent, deriveLogoTheme(overall).accent),
      ink: normalizeHex(saved.ink, deriveLogoTheme(overall).ink),
      disc: normalizeHex(saved.disc, deriveLogoTheme(overall).disc),
      guide: normalizeHex(saved.guide, deriveLogoTheme(overall).guide),
      light: saved.light === true || saved.monochrome === true,
    };
  } catch {
    return { ...DEFAULT_THEME };
  }
}

export function applyThemeToDocument(theme) {
  const root = document.documentElement;
  const logoPalette = [theme.accent, theme.ink, theme.disc, theme.guide]
    .map((color) => color.replace("#", "").toLowerCase()).join("-");
  root.style.setProperty("--lime", theme.overall);
  root.style.setProperty("--accent-rgb", hexRgb(theme.overall).join(" "));
  root.style.setProperty("--accent-contrast", accentContrastColor(theme.overall));
  root.style.setProperty("--accent-text", readableAccentText(theme.overall));
  root.style.setProperty("--logo-accent", theme.accent);
  root.style.setProperty("--logo-ink", theme.ink);
  root.style.setProperty("--logo-disc", theme.disc);
  root.style.setProperty("--logo-guide", theme.guide);
  root.style.setProperty("--orange", "#FF7548");
  root.dataset.themeMode = theme.light ? "light" : "dark";
  document.querySelector('meta[name="theme-color"]')?.setAttribute("content", theme.light ? "#F4F4F1" : "#0B0B0C");
  document.querySelector('link[rel~="icon"]')?.setAttribute("href", `/xiriacanvas-favicon.svg?palette=${logoPalette}`);
}
