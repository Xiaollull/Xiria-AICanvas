"""Closed tagger registry. No request can supply a repository, URL or Python module."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Artifact:
    remote: str
    local: str
    size: int
    digest: str  # SHA-256 for LFS, git-blob SHA-1 for ordinary repository files.


ENGINES = {
    "wd": dict(name="WD · SmilingWolf", author="SmilingWolf", default="wd-swinv2-v3",
               runtime="ONNX Runtime", repo_url="https://huggingface.co/SmilingWolf",
               description="动漫与插画标签、角色和内容评级；保留 16 个官方 WD 模型。默认模型延续原有像素处理与阈值。",
               license="各模型以官方模型卡为准；默认 swinv2-v3 标注 Apache-2.0。",
               defaults={"general": .35, "character": .85}, categories=["general", "character"]),
    "cl": dict(name="CL Tagger v1.02", author="cella110n", default="cl-tagger-v1.02",
               runtime="ONNX Runtime", repo_url="https://huggingface.co/cella110n/cl_tagger",
               description="51,213 个标签，覆盖角色、作品、作者、质量与模型等类别；448px 输入、约 1.43 GB 权重。采用作者链接的 v1 推理契约：白底正方形、BICUBIC、BGR/CHW [-1,1]，logits 取 sigmoid；不是 v2 NaFlex。",
               license="官方仓库声明 Apache-2.0；该 revision 未提供独立 LICENSE 文件。",
               defaults={"general": .9, "character": .9, "copyright": .9, "artist": .9, "meta": .9, "tag_model": .9, "quality": .9},
               categories=["general", "character", "copyright", "artist", "meta", "model", "quality"]),
    "camie": dict(name="Camie Tagger v2", author="Camais03", default="camie-tagger-v2",
                  runtime="ONNX Runtime", repo_url="https://huggingface.co/Camais03/camie-tagger-v2",
                  description="143M 参数、512px 输入；70,527 个标签。使用细化输出，包含角色、作者、作品及年份分类。",
                  license="官方权重及代码标注 GPL-3.0。请阅读官方使用和分发条款；本程序不复制其整应用或训练代码。",
                  defaults={"general": .5, "character": .5, "copyright": .5, "artist": .5, "meta": .5, "year": .5},
                  categories=["general", "character", "copyright", "artist", "meta", "year"]),
    "pixai": dict(name="PixAI Tagger v1.0", author="PixAI Labs", default="pixai-tagger-v1.0",
                  runtime="PyTorch", repo_url="https://huggingface.co/pixai-labs/pixai-tagger-v1.0",
                  description="官方 PyTorch 权重；SAM3 ViTDet 骨干、1008px 输入、30,877 个标签，包含独立风格分类。无官方 ONNX，不使用第三方转换。",
                  license="PixAI 模型卡标注 Apache-2.0；静态骨干适配还包含 Meta SAM License (2025-11-19) 归属与条款，见随附 notices。",
                  defaults={"general": .17, "character": .27, "copyright": .24, "style": .15, "meta": .17},
                  categories=["general", "character", "copyright", "style", "meta"]),
}

MODELS = {
    "cl-tagger-v1.02": dict(engine="cl", repo="cella110n/cl_tagger", revision="0b6e9b4e145b1423bfd1715119074a24a301b471",
        input_size=448, parameters_million=None, weight_memory_mib=1360, activation_memory_mib=1024,
        artifacts=(Artifact("cl_tagger_1_02/tag_mapping.json", "tag_mapping.json", 4088861, "c62454002b724ce91224cbcd45e89f997a437b03"),
                   Artifact("cl_tagger_1_02/model.onnx", "model.onnx", 1425860192, "1459e946ffc083159015e0eb26239016349ddd36c3daecfeb70bfa4ccf38b944"))),
    "camie-tagger-v2": dict(engine="camie", repo="Camais03/camie-tagger-v2", revision="7d40c1b85b86ab4f607b2caf26b1b50c99db743e",
        input_size=512, parameters_million=143, weight_memory_mib=753, activation_memory_mib=1024,
        artifacts=(Artifact("camie-tagger-v2-metadata.json", "metadata.json", 7771946, "c7f1db1bb632e02c5ab4f9702a4b66c66e2292e2"),
                   Artifact("camie-tagger-v2.onnx", "model.onnx", 788983561, "ab0aaf253e3d546090001bec9bebc776c354ab6800f442ab9167af87b4a953ac"))),
    "pixai-tagger-v1.0": dict(engine="pixai", repo="pixai-labs/pixai-tagger-v1.0", revision="9fe10addf9326e292da8a85a98ea74cd91b41771",
        input_size=1008, parameters_million=486, weight_memory_mib=1856, activation_memory_mib=2048,
        artifacts=(Artifact("config.json", "config.json", 693918, "28f158ec13f4d160d6dbd5132bc68e2405621649"),
                   Artifact("preprocessor_config.json", "preprocessor_config.json", 600, "470614548d6add68d314983761d489fe595ba347"),
                   Artifact("model.safetensors", "model.safetensors", 1945425796, "f29e475205cbcbc25b52a075840c7809d6215be15f45375113ba09c49bf90292"))),
}

# Official anonymous HF metadata, pinned 2026-09-30. Keep every legacy id/path,
# including branch aliases. Size/hash verification also admits unchanged legacy
# official caches without writing a marker/receipt into their directories.
def _wd(repo, revision, size, digest, csv_size=253906, csv_digest="71796801c13109547bc017d40fc6f5b89bfd9cc0"):
    return dict(repo="SmilingWolf/" + repo, revision=revision,
                artifacts=(Artifact("selected_tags.csv", "selected_tags.csv", csv_size, csv_digest),
                           Artifact("model.onnx", "model.onnx", size, digest)))


WD_MODELS = {
    "wd-eva02-large-tagger-v3": _wd("wd-eva02-large-tagger-v3", "b25b82a03f7282e41aa2f257a52c7583b710bd1c", 1260435999, "9e768793060c7939b277ccb382783e8670e8a042d29d77aa736be0c8cc898bfc", 308468, "a2a653a5c89c8e387765eb1de9cfd2aedc92ee8d"),
    "wd-vit-large-tagger-v3": _wd("wd-vit-large-tagger-v3", "ae469aa2e4706a3af08d3673cf73a11d1add314c", 1260645673, "e4c8001b000a6c98f2db10794f7c406daa79873d071d6ca924330fa053fa1845", 308468, "a2a653a5c89c8e387765eb1de9cfd2aedc92ee8d"),
    "wd-vit-v3": _wd("wd-vit-tagger-v3", "7f6b584d0bd3f55c4531f14ba3d4761b2bccdc0f", 378536310, "35f23693620b668f4d53fd3c62bf65e40af739bc52c7eb0fbc49258b58d065b6", 308468, "a2a653a5c89c8e387765eb1de9cfd2aedc92ee8d"),
    "wd-swinv2-v3": _wd("wd-swinv2-tagger-v3", "627aef95638667ddcaa3ac8ae625e88ea5b02f51", 467460978, "e6774bff34d43bd49f75a47db4ef217dce701c9847b546523eb85ff6dbba1db1", 308468, "a2a653a5c89c8e387765eb1de9cfd2aedc92ee8d"),
    "wd-convnext-v3": _wd("wd-convnext-tagger-v3", "d39e46de298d27340111b64965e20b8185c407e6", 394990732, "1b8a7abf13d9b8368267df47501d523789c4aeae66b2296ad98483239dfa32eb", 308468, "a2a653a5c89c8e387765eb1de9cfd2aedc92ee8d"),
    "wd14-convnextv2-v2": _wd("wd-v1-4-convnextv2-tagger-v2", "bf364499ea843a403cc2770072f31f5cfb2ffa58", 388479878, "e91daa19cd9e8725125b7d70702d1560855fb687f8d8c4218eddaa821f41834a"),
    "wd14-moat-v2": _wd("wd-v1-4-moat-tagger-v2", "8452cddf280b952281b6e102411c50e981cb2908", 326197340, "b8cef913be4c9e8d93f9f903e74271416502ce0b4b04df0ff1e2f00df488aa03"),
    "wd14-vit-v2": _wd("wd-v1-4-vit-tagger-v2", "6ce0bccf89af92bdd57bc7266161e5a8eae42e70", 373287959, "8a21cadd1f88a095094cafbffe3028c3cc3d97f4d58c54344c5994bcf48e24ac"),
    "wd14-convnext-v2": _wd("wd-v1-4-convnext-tagger-v2", "fcc81c2413593a45375b0de1606cde275cba504d", 387820405, "71f06ecb7b9df81d8f271da4d43997ea2ed363cdac29aa64fcb256c9631e656a"),
    "wd14-swinv2-v2": _wd("wd-v1-4-swinv2-tagger-v2", "8350dd529d920aab46156928654cba67c71dc348", 455481275, "67740df7ede9a53e50d6e29c6a5c0d6c862f1876c22545d810515bad3ae17bb1"),
    "wd14-convnextv2-v2-git": _wd("wd-v1-4-convnextv2-tagger-v2", "dbd4dbe553ee51feb3bc745b614fb762080e3912", 388479878, "e91daa19cd9e8725125b7d70702d1560855fb687f8d8c4218eddaa821f41834a"),
    "wd14-vit-v2-git": _wd("wd-v1-4-vit-tagger-v2", "1f3f3e8ae769634e31e1ef696df11ec37493e4f2", 373287959, "8a21cadd1f88a095094cafbffe3028c3cc3d97f4d58c54344c5994bcf48e24ac"),
    "wd14-convnext-v2-git": _wd("wd-v1-4-convnext-tagger-v2", "4b34d1b07bdd8e95494072648960b8a6adcbc0ff", 387820405, "71f06ecb7b9df81d8f271da4d43997ea2ed363cdac29aa64fcb256c9631e656a"),
    "wd14-swinv2-v2-git": _wd("wd-v1-4-swinv2-tagger-v2", "cdb0c7fdc70646f0af29c6f80f8df564344a69b6", 455409152, "04ec04fdf7db74b4fed7f4b52f52e04dec4dbad9e4d88d2d178f334079a29fde"),
    "wd14-vit": _wd("wd-v1-4-vit-tagger", "213a7bd66d93407911b8217e806a95edc3593eed", 365487723, "22e88a3226e427998fdf669bdbd035ee7040f3229796dd66ec35b8dd90e852b5", 173829, "33ab5e6f93fe352338b6837f32849e9815c3bd08"),
    "wd14-convnext": _wd("wd-v1-4-convnext-tagger", "4036ca51f1c082b0e7c4496890bbf9eadad5764a", 377430976, "b7d7c9923e0056a2def0f4418df01a1274467b3da8480f146b851289257734de", 173829, "33ab5e6f93fe352338b6837f32849e9815c3bd08"),
}


def engine_for(model, wd_registry):
    if model in wd_registry:
        return "wd"
    if model in MODELS:
        return MODELS[model]["engine"]
    raise ValueError("未知反推模型")


def validate_selection(engine, device, model, wd_registry):
    if engine not in ENGINES or device not in ("cpu", "gpu"):
        raise ValueError("未知反推引擎或设备；仅支持 CPU/GPU")
    if engine_for(model, wd_registry) != engine:
        raise ValueError("模型与所选引擎不匹配")


def resource_budget(model):
    # FP32 resident weights PLUS activation/workspace estimate PLUS safety margin;
    # not a download-size-as-VRAM claim. Estimates are disclosed, not a guarantee.
    spec = MODELS.get(model)
    if spec:
        weights, activation = spec["weight_memory_mib"], spec["activation_memory_mib"]
        size, parameters = spec["input_size"], spec["parameters_million"]
    else:
        large = "large" in model
        weights, activation = (1300, 1024) if large else (640, 768)
        size, parameters = 448, None
    margin = 512
    return dict(weights_mib=weights, activations_workspace_mib=activation, margin_mib=margin,
                required_mib=weights + activation + margin, input_size=size, parameters_million=parameters,
                estimated=True, recommend_gpu=(weights >= 750 or size >= 512),
                reason="FP32 常驻权重 + 单图激活/工作区估计 + 512 MiB 余量；实际峰值依模型与 provider 而异，不保证无 OOM。")
