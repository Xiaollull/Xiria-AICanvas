<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="public/xiriacanvas-wordmark-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="public/xiriacanvas-wordmark-light.svg">
    <img src="public/xiriacanvas-wordmark-light.svg" alt="XiriaCanvas AI" width="680">
  </picture>
</h1>

<p align="center">
  <img src="public/xiriacanvas-logo.svg" alt="XiriaCanvas AI logo" width="180">
</p>

<p align="center">
  <strong>English</strong> · <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <a href="https://xiaollull.github.io/Xiria-AICanvas/"><img alt="Documentation" src="https://img.shields.io/badge/docs-xiaollull.github.io-6a55e6"></a>
  <a href="https://github.com/Xiaollull/Xiria-AICanvas/wiki"><img alt="Wiki" src="https://img.shields.io/badge/wiki-troubleshooting-6a55e6"></a>
  <a href="https://github.com/Xiaollull/Xiria-AICanvas/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/Xiaollull/Xiria-AICanvas"></a>
  <a href="LICENSE"><img alt="License: AGPL-3.0" src="https://img.shields.io/badge/license-AGPL--3.0-6a55e6"></a>
</p>

XiriaCanvas AI is a local image-generation UI for Stable Diffusion, Illustrious,
and the native Anima, FLUX.1, FLUX.2, and Krea 2 engines. The same source tree
supports Windows and Linux. The project does not depend on ComfyUI or Stable
Diffusion WebUI at runtime.

## Engines

| Engine | Model | Components |
| --- | --- | --- |
| **SD** | Stable Diffusion | one checkpoint |
| **iL** | Illustrious / SDXL | one checkpoint |
| **Anima** | native Flow Matching | diffusion model + text encoder + VAE |
| **Flux** | FLUX.1, distilled guidance | diffusion model + CLIP-L + T5-XXL + VAE |
| **Flux2** | FLUX.2, large-model guidance | diffusion model + text encoder + VAE |
| **Krea2** | Krea 2 single-stream DiT | diffusion model + text encoder + VAE |

## Quick start

Node.js 22.21 or newer, Python 3.10–3.13 (or `uv`), and an NVIDIA GPU with a
current driver. The setup command opens a local configuration window that
detects your GPU and selects a matching PyTorch build before installing
anything.

```text
npm run setup   # first time only
npm start       # afterwards
```

Windows users can double-click `Setup-XirAI.bat` and then `Start-XirAI.bat`;
Linux users can run `sh Setup-XirAI.sh` and `sh Start-XirAI.sh`. The
configuration window is served at `http://localhost:7709/config`.

## Documentation

The full manual lives at **[xiaollull.github.io/Xiria-AICanvas](https://xiaollull.github.io/Xiria-AICanvas/)**.

- [Installation](https://xiaollull.github.io/Xiria-AICanvas/setup.html) — requirements, the setup wizard, GPU and PyTorch selection, network policy, CLI reference.
- [Models](https://xiaollull.github.io/Xiria-AICanvas/models.html) — where files go, the recommended-model browser, file formats and quantization.
- [Generating images](https://xiaollull.github.io/Xiria-AICanvas/usage.html) — prompt weights, guidance, samplers, batches, the image workspace.
- [Runtime & updates](https://xiaollull.github.io/Xiria-AICanvas/runtime.html) — configuration, VRAM memory modes, downloads, program updates.

Troubleshooting, the quantization support matrix, the release process, and the
development and CI notes are in the
[wiki](https://github.com/Xiaollull/Xiria-AICanvas/wiki).

## License

Copyright © 2026 Xiaollull. Released under the
[GNU Affero General Public License v3.0](LICENSE).

You may use, study, modify and redistribute this program, provided derivative
works carry the same licence. The Affero clause adds one condition beyond the
GPL: if you run a modified version and let other people use it over a network,
those users must be offered its source.

The licence is not only a preference. XiriaCanvas AI performs ADetailer face and
hand detection through [Ultralytics](https://github.com/ultralytics/ultralytics)
YOLO, which is itself AGPL-3.0, so a combined work carrying any other licence
would need a commercial licence from Ultralytics. The remaining dependencies —
PyTorch, Diffusers, Transformers, FastAPI, React and the rest — are Apache-2.0,
MIT, BSD or ISC, all of which this licence can absorb.

There are no additional permissions and no carve-outs: every dependency the build
ships can be conveyed under this licence, so what you receive is AGPL-3.0 all the
way through. The one exception that would have been needed — GSAP, whose Standard
License is free of charge but is not free software — was removed instead; its
entrance animation now runs on the browser's own Web Animations API.

Model weights are not part of this program and are not covered by this licence.
Each checkpoint you download keeps the terms of its own publisher, and some of
the recommended models are gated or non-commercial. Check them before you build
a product on one.

## Star History

<a href="https://star-history.com/#Xiaollull/Xiria-AICanvas&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Xiaollull/Xiria-AICanvas&type=Date&theme=dark">
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Xiaollull/Xiria-AICanvas&type=Date">
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Xiaollull/Xiria-AICanvas&type=Date" width="620">
  </picture>
</a>
