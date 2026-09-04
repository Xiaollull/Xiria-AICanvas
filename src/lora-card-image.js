// Turning a picture the user picked into something the card store will accept.
//
// The store caps an image at 4 MB, and a card is never drawn wider than a few
// hundred CSS pixels, so a 12 MP generation is downscaled here rather than
// refused there. Doing it in the page also means the oversized original never
// crosses the request boundary at all.
//
// The DOM pieces are injected so the arithmetic and the format fallback can be
// tested without a browser.

export const CARD_IMAGE_MAX_EDGE = 1024;
export const CARD_IMAGE_QUALITY = 0.86;
/** Canvas encoders that every supported browser can produce, best first. */
export const CARD_IMAGE_FORMATS = ["image/webp", "image/jpeg"];

export function scaledCardSize(width, height, maxEdge = CARD_IMAGE_MAX_EDGE) {
  const sourceWidth = Math.max(1, Math.round(Number(width) || 0));
  const sourceHeight = Math.max(1, Math.round(Number(height) || 0));
  const longest = Math.max(sourceWidth, sourceHeight);
  if (!Number.isFinite(maxEdge) || maxEdge <= 0 || longest <= maxEdge) return { width: sourceWidth, height: sourceHeight, scaled: false };
  const ratio = maxEdge / longest;
  return {
    width: Math.max(1, Math.round(sourceWidth * ratio)),
    height: Math.max(1, Math.round(sourceHeight * ratio)),
    scaled: true,
  };
}

function browserImageLoader(source) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = async () => {
      try { if (image.decode) await image.decode(); } catch {}
      resolve(image);
    };
    image.onerror = () => reject(new Error("图片读取失败"));
    image.src = source;
  });
}

function browserCanvas(width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

export function readImageFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(new Error("图片读取失败"));
    reader.readAsDataURL(file);
  });
}

/**
 * Encodes the canvas as the best format it will actually produce. `toDataURL`
 * answers with PNG when it does not know the type asked for rather than
 * failing, so the returned prefix — not the request — decides whether a format
 * worked.
 */
export function encodeCardCanvas(canvas, quality = CARD_IMAGE_QUALITY, formats = CARD_IMAGE_FORMATS) {
  for (const format of formats) {
    const encoded = canvas.toDataURL(format, quality);
    if (typeof encoded === "string" && encoded.startsWith(`data:${format};base64,`)) return { dataUrl: encoded, contentType: format };
  }
  throw new Error("当前浏览器无法编码这张图片");
}

export async function prepareCardImage(source, {
  maxEdge = CARD_IMAGE_MAX_EDGE,
  quality = CARD_IMAGE_QUALITY,
  formats = CARD_IMAGE_FORMATS,
  loadImage = browserImageLoader,
  createCanvas = browserCanvas,
} = {}) {
  const image = await loadImage(source);
  const naturalWidth = image.naturalWidth || image.width || 0;
  const naturalHeight = image.naturalHeight || image.height || 0;
  if (!naturalWidth || !naturalHeight) throw new Error("图片尺寸无法识别");
  const size = scaledCardSize(naturalWidth, naturalHeight, maxEdge);
  const canvas = createCanvas(size.width, size.height);
  const context = canvas.getContext("2d");
  if (!context) throw new Error("当前浏览器无法处理这张图片");
  // A transparent source encoded as JPEG would otherwise composite onto black.
  // WebP keeps the alpha channel and paints over this untouched.
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, size.width, size.height);
  context.drawImage(image, 0, 0, size.width, size.height);
  const encoded = encodeCardCanvas(canvas, quality, formats);
  return { ...encoded, width: size.width, height: size.height, scaled: size.scaled };
}
