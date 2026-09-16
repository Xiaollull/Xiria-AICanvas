/**
 * How tall the two picture panes are on the image-to-image stage.
 *
 * The panes used to be square, which decided their height from the width of the column they happen
 * to sit in. On a wide stage that is a very tall square: scrolled to the bottom the frame still ran
 * past the edge of the window, so there was no scroll position at which a whole picture and its
 * caption were on screen at once.
 *
 * So the height is fitted instead. The run bar is sticky, and everything above the pictures --
 * the prompt boxes -- scrolls away under it, which means the last scroll position shows exactly the
 * run bar plus whatever is left. Sizing the pair to that remainder makes the bottom of the stage a
 * definite thing: the SOURCE and RESULT headings land against the bar's lower edge, the frames are
 * whole, and the captions finish at the bottom of the scrollport.
 *
 * The result is a landscape frame on a typical laptop, which suits the pictures anyway; whether it
 * is landscape or not is an outcome of the window, not a ratio anyone chose.
 */

// The fixed parts of a pane, from the stylesheet: the heading row, the caption row, and the margin
// above and below the frame. What is left over is the picture.
export const I2I_COMPARE_HEAD_HEIGHT = 34;
export const I2I_COMPARE_FOOT_HEIGHT = 34;
export const I2I_COMPARE_FIGURE_MARGIN = 7;
// Below this a picture stops being one, so the fit gives up rather than shrinking further. Past
// that point the stage simply scrolls, as it did before.
export const I2I_COMPARE_FIGURE_MIN_HEIGHT = 180;
export const I2I_COMPARE_MIN_HEIGHT = I2I_COMPARE_HEAD_HEIGHT + I2I_COMPARE_FOOT_HEIGHT
  + (I2I_COMPARE_FIGURE_MARGIN * 2) + I2I_COMPARE_FIGURE_MIN_HEIGHT;

/**
 * The height of the compare block, given the stage's scrollport and the sticky bar over it.
 *
 * `stageHeight` is the scrollport (`clientHeight`, so it already excludes a horizontal scrollbar
 * and includes the padding), `barHeight` the bar that covers its top, and `bottomPadding` the
 * stage's own padding below the last block. Returns 0 when there is nothing measurable yet, which
 * the caller reads as "leave the stylesheet's own height alone".
 */
export function i2iCompareHeight({ stageHeight, barHeight, bottomPadding = 0 }) {
  const stage = Number(stageHeight);
  const bar = Number(barHeight);
  const padding = Number(bottomPadding);
  if (!Number.isFinite(stage) || stage <= 0) return 0;
  const available = stage - (Number.isFinite(bar) ? bar : 0) - (Number.isFinite(padding) ? padding : 0);
  return Math.max(I2I_COMPARE_MIN_HEIGHT, Math.round(available));
}
