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

XiriaCanvas AI 是一款本地运行的 Stable Diffusion、Illustrious 与原生 Anima 图像生成界面。同一套源代码同时支持 Windows 与 Linux。项目在运行时不依赖 ComfyUI 或 Stable Diffusion WebUI。

## 环境要求

- Windows 10/11，或现代的 x86-64 Linux 发行版。
- Node.js 22.21 或更新版本。
- Python 3.10 至 3.13，或使用 `uv` 自动安装 Python 3.12。
- 用于图像生成的 NVIDIA GPU 与当前版本的 NVIDIA 驱动程序。

PyTorch wheel 自带各自的 CUDA 运行时。系统级 CUDA Toolkit 并非必需，即使存在，其版本也会被刻意忽略。已有的全局或项目级 PyTorch 安装同样不会影响自动选择。安装脚本会检测 NVIDIA GPU 以及其驱动程序支持的最大 CUDA ABI，然后查询当前官方 wheel 索引，选择最新稳定版 PyTorch，并搭配仍处于该驱动上限之内的最新捆绑 CUDA 运行时。它只下载这一个组合，然后运行一次真实的 CUDA 张量运算；它不会下载多个大型 wheel 来试装不同版本。它不会安装或修改系统 GPU 驱动。

## 安装

最简单的安装方式会打开一个本地可视化配置窗口。向导首先检测你的 GPU 并查询官方 PyTorch wheel 目录，此时不会安装任何内容。然后你可以选择：

- **自动安装（Auto setup）** —— 一步选择与驱动兼容的最新稳定版 PyTorch 及最高的捆绑 CUDA 运行时。
- **手动安装（Manual setup）** —— 从 wheel 目录中挑选一个 CUDA 运行时和具体的 PyTorch 版本。比当前最新稳定版更旧的版本，或使用非首选 CUDA 运行时的版本，会以 **Non-recommended（不推荐）** 标记并附注原因。安装只下载所选组合。

两种模式都包含 xformers 开关（默认开启）。手动模式还可以启用 **auto-repair（自动修复）**：如果已安装的 PyTorch/CUDA 组合未通过安装后的 CUDA 张量运算，安装程序会自动尝试下一个最合适的兼容组合，而无需重新打开向导。

**Windows：** 双击 `Setup-XirAI.bat`。

**Linux：** 运行 `sh Setup-XirAI.sh`，或将 `XirAI-Setup.desktop` 标记为受信任后双击它。两个启动器都会根据自身文件所在位置定位项目，因此安装路径中包含空格和非英文字符也受支持。

以下单行命令在 Windows 与 Linux 上均适用：

```text
npm run setup
```

配置窗口始终在 `/config` 提供服务：

```text
http://localhost:7709/config
http://YOUR-LAN-IP:7709/config
```

在安装与构建校验完成之前，`/` 及所有主应用 API 均处于锁定状态。打开 `http://localhost:7709/` 或 `http://YOUR-LAN-IP:7709/` 会重定向到 `/config`。锁定期间推理后端不会启动。

窗口中会显示 XirAI 标志、机器检测结果、所选 Python/PyTorch/CUDA 版本、完整的下载计划、每个文件的 MB 数/总下载进度、已完成下载数/总下载数、实时日志以及最终的构建校验。当它报告配置与构建已通过时，点击 **Restart and enter the main page（重启并进入主页面）**。配置服务器会释放 7709 端口，在同一端口启动主服务，等待其就绪，然后自动打开主应用。

安装完成后，常规启动命令是：

```text
npm start
```

Windows 用户可双击 `Start-XirAI.bat`。Linux 用户可运行 `sh Start-XirAI.sh`，或将 `XirAI-Start.desktop` 标记为受信任后双击它。启动器绝不会安装或更改环境。如果缺少安装标记、Python 环境、Python 依赖、Node 依赖或构建工具，它会显示 **Please configure the environment first（请先配置环境）** 并退出，同时指向 `Setup-XirAI`。

## 模型下载器

模型下载器为粘贴的链接和推荐的模型选择维护一条串行队列。当一个文件正在下载时，你可以继续添加链接、打开推荐模型浏览器并追加更多选择，而不会中断当前传输。已经通过目录大小与 SHA-256 校验、或已存在于队列中的推荐资源，会从等待列表中省略。

原生 Anima 的 Qwen3 tokenizer、tokenizer config 与 T5 tokenizer JSON 文件随 `backend/resources/anima-tokenizers` 一并打包。用户只需下载所选 Anima DiT、文本编码器与 VAE；安装、启动和程序更新会自动校验打包的 tokenizer 字节。

失败的行可以从下载器页面重试。重试会保留原始分段数并复用已保留的 `.part`/`.part.N` 文件，因此可用字节会从已保存的偏移量处继续下载。Provider 令牌仅保存在浏览器本地，并在重试时重新提交；它们绝不会写入持久化的队列状态。

**Settings → About → Manual program update（设置 → 关于 → 手动程序更新）** 页面接受受信任的干净 XiriaCanvas AI 项目归档，例如 ZIP、7Z、RAR、TAR、TAR.GZ 与 TAR.XZ。归档会使用命令行工具检查并解压到项目外部的临时目录。当系统命令无法读取归档时，更新程序会为当前 Windows 或 Linux 架构下载带校验和固定的官方 7-Zip 命令行文件到 `.cache/tools/7zip/`；它绝不运行安装程序，也不使用桌面文件关联。

手动更新需要与已安装环境相同的 Node 和 Python 依赖清单。只替换程序文件。`.venv`、`node_modules`、`models`、`outputs`、`logs`、`state-cache`、`.cache` 与 `.env` 保持原样。替换前，受管理的文件会备份到项目外部。复制失败、Python 校验失败和生产构建失败都会自动回滚。

安装脚本会执行以下步骤：

1. 检测 Windows 或 Linux，并使用各平台的原生约定查找 Python。
2. 使用 Python 3.10-3.13 创建 `.venv`。
3. 当缺少 Python 时，在项目缓存下使用强制性的各平台 SHA-256 哈希安装经过测试且固定版本的 `uv` 发布版，然后在同一缓存下安装受管理的 Python 3.12。
4. 使用 `nvidia-smi`（包括更新的 `CUDA UMD Version` 输出）以及 CUDA Driver API 检测 NVIDIA GPU 与驱动兼容上限。这不是对已安装 CUDA Toolkit 的检测。
5. 在下载每个项目之前，对官方、阿里云与清华线路上实际的目标包进行基准测试，若该线路不可用则自动切换。
6. 解析适用于当前操作系统、Python 版本、架构、GPU 与驱动的最新稳定版 PyTorch 版本。PyTorch 版本优先；在该版本的各 wheel 中，选择最新的驱动兼容捆绑 CUDA 运行时。
7. 使用最多 8 个经过验证的 HTTP Range 分段下载大型 PyTorch wheel。其他 Python wheel 使用 `uv` 包级并发，随后所有内容都安装在本仓库隔离的 `.venv` 中。
8. 当存在兼容 wheel 时默认安装 `xformers`。不兼容的 wheel 会被安全移除，而不会替换已选的 Torch。
9. 验证确切的 PyTorch/CUDA 版本，运行一次真实的 CUDA 张量运算，检查依赖一致性，并运行生产前端构建。

如果安装被中断，**Continue setup from checkpoint（从检查点继续安装）** 会复用缓存的驱动/PyTorch/CUDA 运行时选择与 wheel 元数据，跳过已完成且已验证的安装步骤，并避免再次查询版本索引。`.cache/downloads` 中的大文件会从每个未完成 Range 分段的已保存字节偏移量处续传；本地 `uv` 缓存中的 Python 包以及 npm 的包缓存也会自动复用。重新创建 `.venv` 或更改依赖清单只会使受影响的检查点失效。仅在需要重新选择 CUDA/PyTorch 时使用 `--refresh-selection`。

包源覆盖项：

```text
npm run setup:cli -- --source=official
npm run setup:cli -- --source=aliyun
npm run setup:cli -- --source=tsinghua
npm run setup:cli -- --torch=cu126
npm run setup:cli -- --torch=cu126 --torch-version=2.13.0+cu126
npm run setup:cli -- --torch=cpu
npm run setup:cli -- --diagnose
npm run setup:cli -- --without-xformers
npm run setup:cli -- --connections=12
npm run setup:cli -- --refresh-selection
```

CUDA 运行时变体会从当前稳定版 PyTorch wheel 索引动态发现；脚本没有固定的最大 CUDA 版本。如果检测或兼容性选择失败，NVIDIA 机器绝不会静默回退到 CPU wheel。仅凭 GPU 硬件代数并不会被当作超越当前已安装驱动所报告兼容上限的许可。例如，一块能够运行 CUDA 12.8 构建的 GPU，在当前驱动报告 12.6 上限时仍会收到 `cu126`。`--torch=cpu` 仍然是一个明确的诊断性覆盖项。`xformers` 默认在无依赖的情况下安装，因此它不可能替换已选的确切 PyTorch 版本。如果导入兼容性检查失败，它会被自动移除。高级用户可以使用 `--without-xformers` 禁用它。

同样的选项可以通过 `XIRAI_PACKAGE_SOURCE`、`XIRAI_TORCH_VARIANT`、`XIRAI_TORCH_INDEX`、`XIRAI_DOWNLOAD_CONNECTIONS`（1-16）和 `XIRAI_PACKAGE_CONCURRENCY`（1-32）配置。自动路由是默认行为。显式的 `--source` 会保持该源优先，并仅将官方源用作失败时的后备。

项目本地 uv 引导会在不执行下载脚本的情况下复现官方 Bash/PowerShell 安装程序的安全相关行为。它选择 Windows `.zip` 或 Linux `.tar.*` 构件，检查官方最低 glibc 版本并在需要时回退到静态 musl 构建，从 Astral 发布线路下载并提供 GitHub 后备，验证固定的 SHA-256，在暂存目录中测试确切的 uv 版本，然后原子地提升到 `.cache/tools/uv/current`。解压出的 `uvx` 与 Windows `uvw.exe` 二进制会被保留。

默认的 uv 目标文件基准测试还会在 Astral 与 GitHub 后备之前包含 `ghfast.top` 和 `ghproxy.net` 加速线路。它们仅被视为不可信的字节传输通道：每个完成的归档都必须匹配内嵌的官方 SHA-256，否则即被删除。由于 Windows uv 归档约为 25 MB，uv 使用专门的 8 MB 阈值，因此会运行最多 8 个 HTTP Range 连接，而不是回退到单个连接。

高级镜像可以使用官方安装程序变量 `UV_DOWNLOAD_URL`、`INSTALLER_DOWNLOAD_URL`、`UV_INSTALLER_GHE_BASE_URL` 或 `UV_INSTALLER_GITHUB_BASE_URL`；每个都是基础 URL，归档文件名会自动追加。多个直接基础地址可以用空白字符分隔。`UV_GITHUB_TOKEN` 只会转发到 GitHub 或明确配置的 GitHub Enterprise 主机。每个镜像都必须通过构件哈希校验。设置这些源变量中的任何一个都会替换内置的加速线路列表，因此仅官方或私有镜像策略仍然可用。引导请求通过 Node 内置的代理支持遵循 `HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY` 和 `NO_PROXY`（包括小写形式）。

与用户级 uv 安装不同，XirAI 会刻意忽略安装目录和 PATH 修改设置：它绝不修改 shell 配置文件、Windows 注册表、`PATH`、Cargo/XDG 目录或用户级 uv receipt，也不启用 `uv self update`。应用始终直接调用其固定的项目本地可执行文件。

安装绝不会安装到系统 Python、修改全局 pip 配置、安装 CUDA Toolkit 或更改 NVIDIA 驱动。运行时只使用规范的项目 `.venv`；Python 与 pip 重定向变量会被移除。旧的未隔离 `.venv` 会在创建新环境前保留为 `.venv.backup-<timestamp>`。ComfyUI、SD WebUI 及其环境不会被读取、导入、升级或修改。

## 配置

仅当需要更改默认值时，才将 `.env.example` 复制为 `.env`。相对路径在两种操作系统上都从仓库根目录解析。

默认服务如下：

- 配置与 Web UI：`0.0.0.0:7709`
- 推理后端：`127.0.0.1:8718`

Web UI 可通过 `http://localhost:7709/` 与 `http://YOUR-LAN-IP:7709/` 访问。只有 Web UI 对局域网暴露；Python 推理服务仅限回环访问，并通过 Web 代理访问。

仅在受信任的网络中使用局域网地址。它会把开发 UI 和经代理的本地推理 API 暴露给能够访问 7709 端口的设备。

## 提示词权重与引导

SD、iL/SDXL 与原生 Anima 共享核心的 ComfyUI 括号语法：

- `(text)` 应用默认的 `1.1` 强调。
- `((text))` 对嵌套强调进行连乘。
- `(text:1.25)` 应用显式的有限权重。
- `\(text\)` 保留字面括号。

方括号减弱、`BREAK`、文本反转（textual inversion）标签、Prompt LoRA 标签与 A1111 调度语法不由此语法解释。Anima 独立解析 Qwen 与 T5 的 token 边界，并将权重应用于 T5 对齐的 LLM 适配器输出。

PAG 可用于 SD、iL 与原生 Anima，scale 范围 `0..5`，scope 为 `mid|all`。Anima 使用原生的 Cosmos 恒等自注意力分支，而不是 Diffusers UNet 适配器。它按顺序运行分支，以始终保持一个物理 transformer 前向过程存活，并将相同的设置传播到原生 Hires 与 ADetailer 细化中。CFG-Zero* 仍然是 Anima 独有。

原生 Anima 适配器融合接受标准 LoRA、静态全秩 T-LoRA、线性 LoHa、受支持的线性 LoKr 形式以及使用 `dora_scale` 的输出轴 LyCORIS 线性 DoRA。受支持的线性 LoKr 也可以携带输出轴 `dora_scale`。权重为零即为精确禁用。Tucker/卷积形式、分解的 LoKr 第一因子、PEFT `lora_magnitude_vector` 与 `lora_mid.weight` 会以失败关闭（fail closed）。

Anima 在 Generate 与 Gallery 中使用相同的 44 个采样器名称和 9 个调度器名称。全部 9 个调度器都是原生的 Flow-shift-3 轨迹。具名的兼容性求解器映射会在任务警告和 PNG 采样元数据中报告，而不是静默地伪装成另一种实现。

## 图像批次

采样参数包含两个独立的输出控制：

- **每批图像数（Images per batch）** 接受 1-10。一次 Diffusers 管道调用会使用 `num_images_per_prompt` 和等长的生成器列表，因此该批次中的所有图像会一起采样，而不是作为顺序执行的单图任务。
- **批次数（Batch count）** 接受 1-20。批次在一个可控制的任务内排队并按顺序运行。例如，3 张图像 × 4 批会执行四次管道调用并保存 12 张图像。最大任务规模为 10 张图像 × 20 批，即最多保存 200 张图像。

种子会按输出位置在整个任务中递增，并在无符号 64 位范围内安全回绕。结果浏览器按批次对输出分组，并允许用户选择当前批次中的任意图像。缩放与下载操作始终使用所选图像。取消或失败的多批次任务会移除部分输出，而完成的 PNG 文件会在元数据中记录其批次、图像、种子和总批次设置。

## 图像查看器工作区

本地图像工作区可以从预览头部打开，或在任何图像生成之前从其空状态打开。右上角的 **全屏** 按钮无需使用 F11 即可将工作区扩展到浏览器窗口。可选的左侧图像轨道默认显示自当前 Web UI 会话开始以来创建的 PNG/GIF 文件。其资源管理器条带可以打开配置的输出根目录、进入任意嵌套文件夹，并一次返回一层。每一层都会同时显示直接子文件夹和直接图像，因此一个文件夹可以同时包含两者。单图批次显示为单独的卡片。多图批次分组为一个带数量徽标的卡片；打开该卡片会显示其中的图像，返回按钮会恢复之前卡片的滚动位置。

历史卡片可以左键点击以切换活动预览，拖入工作区，或右键点击以添加新图层并管理删除。删除提供仅预览的隐藏或永久移除相应 PNG 源文件两种选择。图层可以移动、从角落手柄调整大小，并通过工具栏或 Ctrl + 鼠标滚轮独立缩放。网格吸附是可选的，网格单元大小可以按像素编辑。

工作区还为 2 到 9 的每种图像数量包含三种推荐布局：水平对比、规则网格和特色图像布局。模板单元格接受从历史轨道拖入的图像或已有的工作区图层。确认已填充的模板会在浏览器中创建无损 PNG 合成图；结果可以再次编辑、以不携带 PNG 生成元数据的方式复制、不保存直接删除，或保存到 `outputs` 下按当前日期命名的目录中。

工作区默认使用 16 px 网格。网格吸附与图像边缘吸附相互独立：边缘吸附会检测附近自由形式图像图层的可见边界，显示对齐参考线，并在释放时使它们的边缘对齐。可选的边缘线面板可以在已吸附图像和模板单元格周围绘制实线、虚线、点线、双线或发光线条。它支持标准调色板、浏览器可用的屏幕取色器以及可编辑的线宽。模板布局会响应图像宽高比，每个选中的单元格都支持图像缩放以及左/中/右和上/中/下对齐。对于手动排列的组，**一键拼图** 会根据当前可见布局创建一张无损合成图。在未保存的已完成拼图上更改边缘线设置会恢复其可编辑形式，并需要再次确认。

运行时路径可以移到只读检出目录之外：

```text
XIRAI_CACHE_DIR=.cache
XIRAI_OUTPUT_DIR=outputs/my-project
```

运行时 VRAM 管理默认采用 `XIRAI_MEMORY_MODE=auto`，遵循 ComfyUI 的内存预算行为，但不导入或依赖 ComfyUI。在第一次加载检查点之前，XirAI 会测量当前可用/总 VRAM，根据其 safetensors 张量元数据估算检查点的 FP16 运行时权重大小，考虑请求的画布大小，为推理至少预留 0.8 GB，并保持与 ComfyUI 相同的操作系统余量（Windows 上 600 MB，超过 15 GB 的 GPU 上再加 100 MB；其他情况 400 MB）。

- `HIGH_VRAM` 仅在完整管道和推理工作区都能容纳时被选中。它将所有模型组件保留在 CUDA 上，并启用最快的 CUDA/TF32/cuDNN 路径。
- `NORMAL_VRAM` 使用组件级动态 CPU/GPU 调度。去噪器在其整个采样循环期间保持于 GPU 上，而非活动组件按需移动。在满足条件的 8 GB 预算上，原生 Anima 会在采样期间保持完整的 Cosmos Transformer 驻留，以实现高利用率。
- `LOW_VRAM` 对无法安全容纳最大组件和推理工作区的 GPU 使用顺序层卸载并辅以 VAE 切片与分块。原生 Anima 在需要时使用同步单块 Cosmos 组卸载和串行图像微批次。

如果 Anima 的驻留 Transformer 路径遇到真实的 CUDA 内存不足（OOM），XirAI 会恢复 CPU 生成器状态，并使用单块组卸载自动重放该采样块。这使自动模式保持速度优先，同时不会让 VRAM 紧张的机器变成失败的任务，也不会改变其 Seed 轨迹。

所有三种模式都会在成功、已取消或非 OOM 失败的生成之后缓存已加载的管道对象，因此使用相同引擎与模型身份的另一次生成无需再次读取并重建模型。对于原生 Anima，该身份包括扩散模型、文本编码器、VAE、固定的 tokenizer 修订版以及按顺序排列的非零适配器修订版/权重。当引擎/模型选择改变、CUDA 报告内存不足错误、中断的卸载链无法安全恢复，或推理服务关闭时，缓存会被释放。SD/iL LoRA 更改不会重新加载基础模型；原生 Anima 适配器会被密集融合到其组合运行时身份中。可用时自动使用 PyTorch SDPA，安装兼容原生扩展时优先使用 xformers。

高级用户可以用 `XIRAI_MEMORY_MODE` 强制 `high_vram`、`normal_vram` 或 `low_vram`；不安全的强制模式会自动降级。设置 `XIRAI_RESERVE_VRAM_GB` 以覆盖默认的操作系统/应用预留量。

输出值可以使用自定义文件夹名称或任意嵌套的相对路径。`.env` 中也接受绝对的本地值；仓库中不会提交任何绝对的用户或系统路径。空白输出路径会被拒绝。

## 模型

参见 `models/README.md`。`models/model-paths.json` 中的模型配置使用 `models` 下的项目相对路径，在 Windows 与 Linux 上均无需改动即可工作。每个根目录都可以使用自定义名称和任意嵌套。检查点、LoRA、放大模型、YOLO 检测器与兼容的背景移除模型会被递归扫描；程序更新会保留用户的路径清单。

## 开发

```text
npm ci
npm run dev
npm run test:backend
npm run build
npm run inference
```

不要在操作系统之间复制 `node_modules` 或 `.venv`。在目标机器上使用 `npm run setup` 重新创建两者。

GitHub Actions 会在每次 push 和 pull request 时，对 `windows-latest` 与 `ubuntu-latest` 运行语法、环境检测和前端构建检查。
