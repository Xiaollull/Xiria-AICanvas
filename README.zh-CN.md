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
  <a href="LICENSE"><img alt="许可证：AGPL-3.0" src="https://img.shields.io/badge/license-AGPL--3.0-6a55e6"></a>
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

## 许可证

版权所有 © 2026 Xiaollull。本项目以
[GNU Affero 通用公共许可证第 3 版](LICENSE)发布。

你可以使用、研究、修改并再分发本程序，但衍生作品必须沿用同一许可证。相比 GPL，Affero
条款多出一条：如果你把修改后的版本部署起来、让别人通过网络使用，就必须向这些使用者提供
其源代码。

选择这个许可证并不只是偏好。XiriaCanvas AI 的 ADetailer 面部与手部检测经由
[Ultralytics](https://github.com/ultralytics/ultralytics) YOLO 完成，而它本身即为
AGPL-3.0；因此组合后的作品若采用其他许可证，就需要向 Ultralytics 另行购买商业授权。其余
依赖——PyTorch、Diffusers、Transformers、FastAPI、React 等——均为 Apache-2.0、MIT、BSD
或 ISC，都可以被本许可证吸收。

本项目没有任何附加许可或例外条款：构建产物中的每一个依赖都可以在本许可证下分发，因此你拿到
的是彻底的 AGPL-3.0。原本唯一需要例外的 GSAP——其 Standard License 虽然免费，却不是自由
软件许可证——已被移除，相关入场动画改用浏览器自带的 Web Animations API 实现。

模型权重不属于本程序，也不受本许可证约束。你下载的每个模型各自遵循其发布方的条款，其中
部分推荐模型受限或禁止商用。在以其构建产品之前请先自行确认。

## Star History

<a href="https://star-history.com/#Xiaollull/Xiria-AICanvas&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Xiaollull/Xiria-AICanvas&type=Date&theme=dark">
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Xiaollull/Xiria-AICanvas&type=Date">
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Xiaollull/Xiria-AICanvas&type=Date" width="620">
  </picture>
</a>
