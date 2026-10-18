# MOSS-Image reference pipeline (test oracle only)

`pipeline_moss_image.py` is the official standalone MOSS-Image pipeline, copied unmodified so the native
MOSS DiT port can be compared against it in `backend/tests/test_moss_engine.py`. It is never imported by
the application at runtime.

- Source: `OpenMOSS-Team/MOSS-Image-Base`, revision `5f40f07a304541840b6019d2d34d30f966756787`,
  path `pipeline_moss_image.py` (byte-identical copy in `OpenMOSS-Team/MOSS-Image-SFT@daf706d`)
- SHA-256: `1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31` (checked by the test before import)
- License: Apache License 2.0 (`backend/resources/moss-tokenizers/LICENSE-APACHE-2.0.txt`, same upstream LICENSE)
- Imports `diffusers` and `transformers` symbols that exist in the pinned development environment
  (diffusers 0.38.0, transformers 4.57.6); only `MOSSImageTransformer` is exercised.
