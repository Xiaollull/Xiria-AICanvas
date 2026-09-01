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
  <a href="README.md">English</a> · <strong>简体中文</strong>
</p>

<p align="center">
  <a href="https://xiaollull.github.io/Xiria-AICanvas/zh/"><img alt="文档" src="https://img.shields.io/badge/%E6%96%87%E6%A1%A3-xiaollull.github.io-6a55e6"></a>
  <a href="https://github.com/Xiaollull/Xiria-AICanvas/wiki"><img alt="Wiki" src="https://img.shields.io/badge/wiki-%E6%95%85%E9%9A%9C%E6%8E%92%E6%9F%A5-6a55e6"></a>
  <a href="https://github.com/Xiaollull/Xiria-AICanvas/releases/latest"><img alt="最新版本" src="https://img.shields.io/github/v/release/Xiaollull/Xiria-AICanvas"></a>
</p>

XiriaCanvas AI 是一款本地运行的图像生成界面，支持 Stable Diffusion、Illustrious，以及原生 Anima、FLUX.1、FLUX.2 与 Krea 2 引擎。同一套源代码同时支持 Windows 与 Linux。项目在运行时不依赖 ComfyUI 或 Stable Diffusion WebUI。

## 引擎

| 引擎 | 模型 | 组件 |
| --- | --- | --- |
| **SD** | Stable Diffusion | 单个大模型 |
| **iL** | Illustrious / SDXL | 单个大模型 |
| **Anima** | 原生 Flow Matching | 扩散模型 + 文本编码器 + VAE |
| **Flux** | FLUX.1，蒸馏引导 | 扩散模型 + CLIP-L + T5-XXL + VAE |
| **Flux2** | FLUX.2，大模型引导 | 扩散模型 + 文本编码器 + VAE |
| **Krea2** | Krea 2 单流 DiT | 扩散模型 + 文本编码器 + VAE |

## 快速开始

需要 Node.js 22.21 或更新版本、Python 3.10-3.13（或使用 `uv`），以及一块搭配当前驱动的 NVIDIA GPU。安装命令会打开一个本地配置窗口，在安装任何内容之前先检测 GPU 并选择匹配的 PyTorch 构建。

```text
npm run setup   # 仅首次
npm start       # 之后
```

Windows 用户可双击 `Setup-XirAI.bat`，随后双击 `Start-XirAI.bat`；Linux 用户可运行 `sh Setup-XirAI.sh` 与 `sh Start-XirAI.sh`。配置窗口位于 `http://localhost:7709/config`。

## 文档

完整手册位于 **[xiaollull.github.io/Xiria-AICanvas](https://xiaollull.github.io/Xiria-AICanvas/zh/)**。

- [安装](https://xiaollull.github.io/Xiria-AICanvas/zh/setup.html) —— 环境要求、安装向导、GPU 与 PyTorch 选择、网络策略、命令行参考。
- [模型](https://xiaollull.github.io/Xiria-AICanvas/zh/models.html) —— 文件放在哪里、推荐模型浏览器、文件格式与量化。
- [生成图像](https://xiaollull.github.io/Xiria-AICanvas/zh/usage.html) —— 提示词权重、引导、采样器、批次、图像工作区。
- [运行与更新](https://xiaollull.github.io/Xiria-AICanvas/zh/runtime.html) —— 配置、显存模式、下载、程序更新。

故障排查、量化支持矩阵、发布流程以及开发与 CI 说明放在 [Wiki](https://github.com/Xiaollull/Xiria-AICanvas/wiki)。

## Star History

<a href="https://star-history.com/#Xiaollull/Xiria-AICanvas&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Xiaollull/Xiria-AICanvas&type=Date&theme=dark">
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Xiaollull/Xiria-AICanvas&type=Date">
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Xiaollull/Xiria-AICanvas&type=Date" width="620">
  </picture>
</a>
