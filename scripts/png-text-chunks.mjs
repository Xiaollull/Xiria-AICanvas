import { inflateSync } from "node:zlib";

// PNG text chunks, decoded. Everything that interprets them lives in
// `src/image-metadata.js`; this file only knows the container format.

export const PNG_SIGNATURE = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
// Some producers write a whole ComfyUI workflow into one chunk. The cap keeps a
// hostile or merely enormous file from turning into an unbounded response.
export const MAX_CHUNK_TEXT_BYTES = 4 * 1024 * 1024;

export function isPng(buffer) {
  return Buffer.isBuffer(buffer) && buffer.length >= 8 && buffer.subarray(0, 8).equals(PNG_SIGNATURE);
}

function inflate(data) {
  try {
    return inflateSync(data);
  } catch {
    // A corrupt compressed chunk costs that one chunk, never the whole read.
    return null;
  }
}

function decodeLatin1(buffer) {
  return buffer.toString("latin1");
}

function readNullTerminated(buffer, start) {
  const end = buffer.indexOf(0, start);
  return end === -1 ? null : { value: buffer.subarray(start, end), next: end + 1 };
}

function readTextChunk(type, data) {
  if (type === "tEXt") {
    const keyword = readNullTerminated(data, 0);
    if (!keyword) return null;
    return { keyword: decodeLatin1(keyword.value), text: decodeLatin1(data.subarray(keyword.next)) };
  }
  if (type === "zTXt") {
    const keyword = readNullTerminated(data, 0);
    if (!keyword || data.length <= keyword.next) return null;
    const decompressed = inflate(data.subarray(keyword.next + 1));
    return decompressed ? { keyword: decodeLatin1(keyword.value), text: decodeLatin1(decompressed) } : null;
  }
  if (type === "iTXt") {
    const keyword = readNullTerminated(data, 0);
    if (!keyword || data.length < keyword.next + 2) return null;
    const compressed = data[keyword.next] === 1;
    const language = readNullTerminated(data, keyword.next + 2);
    if (!language) return null;
    const translated = readNullTerminated(data, language.next);
    if (!translated) return null;
    const payload = data.subarray(translated.next);
    const body = compressed ? inflate(payload) : payload;
    // iTXt is the one text chunk defined as UTF-8, which is why CJK prompts
    // survive here but arrive mojibake if a producer used tEXt instead.
    return body ? { keyword: decodeLatin1(keyword.value), text: body.toString("utf8") } : null;
  }
  return null;
}

export function readPngTextChunks(buffer) {
  if (!isPng(buffer)) return null;
  const chunks = {};
  let offset = 8;
  while (offset + 8 <= buffer.length) {
    const length = buffer.readUInt32BE(offset);
    const type = buffer.toString("latin1", offset + 4, offset + 8);
    const dataStart = offset + 8;
    if (length > buffer.length - dataStart) break;
    if (type === "IEND") break;
    if (type === "tEXt" || type === "zTXt" || type === "iTXt") {
      const entry = readTextChunk(type, buffer.subarray(dataStart, dataStart + length));
      // First writer wins: a producer that repeats a keyword is describing the
      // same thing twice, and the first copy is the one it wrote deliberately.
      if (entry?.keyword && chunks[entry.keyword] === undefined) {
        chunks[entry.keyword] = entry.text.length > MAX_CHUNK_TEXT_BYTES ? entry.text.slice(0, MAX_CHUNK_TEXT_BYTES) : entry.text;
      }
    }
    offset = dataStart + length + 4;
  }
  return chunks;
}

// A tEXt chunk is Latin-1 by spec, but A1111 and ComfyUI both write raw UTF-8
// bytes into it anyway. Latin-1 decoding of UTF-8 is lossless byte-wise, so a
// re-encode recovers the original when it round-trips cleanly.
export function repairLatin1Utf8(text) {
  if (typeof text !== "string" || !/[-ÿ]/.test(text)) return text;
  const bytes = Buffer.from(text, "latin1");
  const decoded = bytes.toString("utf8");
  return decoded.includes("�") ? text : decoded;
}

export function readPngMetadataChunks(buffer) {
  const chunks = readPngTextChunks(buffer);
  if (!chunks) return null;
  return Object.fromEntries(Object.entries(chunks).map(([keyword, text]) => [keyword, repairLatin1Utf8(text)]));
}
