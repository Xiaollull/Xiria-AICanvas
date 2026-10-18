# Bundled MOSS-Image Tokenizer Resource

This file is bundled with XiriaCanvas AI so native MOSS-Image inference works offline without a separate
user download. The Comfy-Org single-file repackage of MOSS-Image ships no tokenizer.

## Qwen3.5 Tokenizer

- File: `moss-qwen35-tokenizer.json.gz` (deterministic `gzip -9 -n` of the original file; 3 760 595 bytes,
  SHA-256 `70765620bd5db1fd13e40a868c10419c7c8de7dbb2b5abd3a43d69aaa6c6f5b0`)
- Source: `OpenMOSS-Team/MOSS-Image-Base`
- Revision: `5f40f07a304541840b6019d2d34d30f966756787`
- Original path: `tokenizer/tokenizer.json` (identical LFS object in `OpenMOSS-Team/MOSS-Image-SFT@daf706d`)
- Decompressed: 19 989 325 bytes, SHA-256 `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523`
  (verified at load; a mismatch makes the engine unavailable)
- License: Apache License 2.0
- Modification: compression only; the decompressed bytes are the upstream file unchanged.

The Apache License 2.0 text is included as `LICENSE-APACHE-2.0.txt` in this directory (copied from the
same pinned repository).
