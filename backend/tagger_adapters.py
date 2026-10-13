"""Tagger mathematical adapters. The WD CPU preprocessing remains owned by wd14.

Non-WD preprocessing is deliberately separate: no BGR/CSV assumptions leak into
RGB/CHW/logit engines. CL admission stays fail-closed until its contract is proven.
"""
import io
import json
import threading
import time
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from .tagger_registry import ENGINES, MODELS

CL_PREPROCESSING_VERIFIED = True
CL_PREPROCESSING_REASON = ""
CL_CONTRACT_SOURCE = "作者在官方 discussion #1 指向 celll1/tagutl；固定 commit 1c2aeaa0d575d150b5cf6c738aa94e70f33d84d3 的 onnx_predict.py/export，已与固定 v1.02 ONNX 图及预处理 oracle 对照。"
_lock = threading.RLock()
_onnx_cache = None


def image_from_bytes(data):
    if len(data) > 20 * 1024**2:
        raise ValueError("图片超过 20 MiB")
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.width * source.height > 32_000_000:
                raise ValueError("图片超过 3200 万像素")
            return ImageOps.exif_transpose(source).copy()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as error:
        raise ValueError("无法解码图片或尺寸不安全") from error


def parameters(engine, model, device, general=None, character=None, limit=50, alpha=False, **categories):
    defaults = ENGINES[engine]["defaults"]
    values = {"general": general, "character": character, **categories}
    effective = {key: defaults[key] if values.get(key) is None else float(values[key]) for key in defaults}
    if any(not 0 <= value <= 1 for value in effective.values()) or not 1 <= limit <= 500:
        raise ValueError("阈值须为 0..1，标签上限须为 1..500")
    return dict(engine=engine, model=model, device=device, limit=int(limit), alpha=bool(alpha), **effective)


def read_labels(folder, engine):
    if engine == "cl":
        mapping = json.loads((folder / "tag_mapping.json").read_text(encoding="utf-8"))
        rows = [(mapping[str(index)]["tag"], mapping[str(index)]["category"].lower()) for index in range(51213)]
        if len(mapping) != len(rows):
            raise ValueError("CL 标签表编号不连续或数量错误")
    elif engine == "camie":
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        mapping = metadata["dataset_info"]["tag_mapping"]
        rows = [(mapping["idx_to_tag"][str(index)], mapping["tag_to_category"][mapping["idx_to_tag"][str(index)]]) for index in range(70527)]
    else:
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        categories = [category for category, count in config["tags_split"] for _ in range(count)]
        rows = list(zip(config["tags"], categories))
        if len(rows) != 30877 or len(categories) != len(config["tags"]):
            raise ValueError("PixAI 分类/标签数量不匹配")
    if any(not isinstance(name, str) or not name or len(name) > 256 or category not in
           {"general", "character", "copyright", "artist", "meta", "rating", "style", "year", "model", "quality"} for name, category in rows):
        raise ValueError("标签名称或分类无效")
    return rows


def camie_input(image):
    import numpy as np
    image = image.convert("RGB")
    w, h = image.size
    ratio = 512 / max(w, h)
    new_w, new_h = max(1, int(w * ratio)), max(1, int(h * ratio))
    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (512, 512), (124, 116, 104))
    canvas.paste(image, ((512 - new_w) // 2, (512 - new_h) // 2))
    value = np.asarray(canvas).astype(np.float32) / 255.0
    value = (value - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    return value.transpose(2, 0, 1)[None]


def cl_input(image):
    """Author-linked tagutl 1c2aeaa...: white square, PIL bicubic, BGR/CHW [-1,1].

    Not WD's cv2 AREA/BGR 0..255 and not v2's SigLIP2 NaFlex processor.
    Model graph admission and real output/oracle verification are still mandatory.
    """
    import numpy as np
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    w, h = image.size
    if w != h:
        side = max(w, h)
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        canvas.paste(image, ((side - w) // 2, (side - h) // 2))
        image = canvas
    value = np.asarray(image.resize((448, 448), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    value = value.transpose(2, 0, 1)[::-1]
    return ((value - np.full((3, 1, 1), .5, np.float32)) / np.full((3, 1, 1), .5, np.float32))[None]


def pixai_input(image):
    import torch
    from torchvision.transforms import functional as VF
    if image.mode != "RGB":
        rgba = image.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (255, 255, 255))
        canvas.alpha_composite(rgba)
        image = canvas.convert("RGB")
    tensor = VF.to_tensor(image)
    h, w = tensor.shape[-2:]
    if h != 1008 or w != 1008:
        ratio = min(1008 / h, 1008 / w)
        nh, nw = max(1, int(h * ratio)), max(1, int(w * ratio))
        tensor = VF.resize(tensor, [nh, nw], antialias=True)
        ph, pw = 1008 - nh, 1008 - nw
        tensor = VF.pad(tensor, [pw // 2, ph // 2, pw - pw // 2, ph - ph // 2], 0)
    return VF.normalize(tensor, [.5]*3, [.5]*3)[None]


def decode(scores, rows, params):
    import numpy as np
    scores = np.asarray(scores)
    if scores.shape != (len(rows),) or not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("模型概率输出形状/数值无效")
    ratings, tags = {}, []
    ratings_map = {"rating:g": "general", "rating:s": "sensitive", "rating:q": "questionable", "rating:e": "explicit"}
    for (name, category), score in zip(rows, scores):
        if category == "rating":
            ratings[ratings_map.get(name, name)] = float(score)
        elif score >= params.get("tag_model" if category == "model" else category, params["general"]):
            tags.append(dict(name=name.replace("_", " "), confidence=float(score), category=category))
    tags.sort(key=lambda tag: -tag["confidence"])
    tags = tags[:params["limit"]]
    if params["alpha"]:
        tags.sort(key=lambda tag: tag["name"])
    return dict(rating=ratings, tags=tags)


def create_onnx(path, device):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    providers = ["CPUExecutionProvider"]
    if device == "gpu":
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise ValueError("受管 ONNX runtime 缺少 CUDA provider；不会回退 CPU")
        providers = [("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0,
            "arena_extend_strategy": "kSameAsRequested", "cudnn_conv_use_max_workspace": 0})]
    session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
    session.disable_fallback()
    expected = "CUDAExecutionProvider" if device == "gpu" else "CPUExecutionProvider"
    if session.get_providers()[0] != expected:
        raise ValueError("实际 ONNX provider 不匹配请求；拒绝静默 CPU fallback")
    return session


def infer(data, params, folder, device):
    global _onnx_cache
    import numpy as np
    engine = params["engine"]
    if engine == "cl" and not CL_PREPROCESSING_VERIFIED:
        raise ValueError(CL_PREPROCESSING_REASON)
    image = image_from_bytes(data)
    rows = read_labels(folder, engine)
    if engine == "pixai":
        import torch
        from .pixai_tagger import load_model
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        target = "cuda" if device == "gpu" else "cpu"
        model = load_model(folder / "model.safetensors", config, target)
        with torch.inference_mode():
            values = model(pixai_input(image).to(target)).sigmoid()[0].float().cpu().numpy()
        result = decode(values, rows, params)
        import_version = torch.__version__
        runtime = dict(name="PyTorch", version=import_version, provider="CUDA" if device == "gpu" else "CPU", dtype="float32")
    else:
        with _lock:
            path = folder / "model.onnx"
            info = path.stat()
            key = (str(path), info.st_ino, info.st_size, info.st_mtime_ns, device)
            if not _onnx_cache or _onnx_cache[0] != key:
                _onnx_cache = None
                _onnx_cache = (key, create_onnx(path, device))
            session = _onnx_cache[1]
            if engine == "camie":
                inputs, outputs = session.get_inputs(), session.get_outputs()
                if len(inputs) != 1 or inputs[0].shape[1:] != [3, 512, 512] or len(outputs) != 3 or outputs[1].shape[-1] != len(rows):
                    raise ValueError("Camie 固定 ONNX 输入/三输出契约不匹配")
                logits = session.run([outputs[1].name], {inputs[0].name: camie_input(image)})[0][0]
                # Numerically stable sigmoid of logits, exactly once.
                positive = logits >= 0
                values = np.empty_like(logits)
                values[positive] = 1 / (1 + np.exp(-logits[positive]))
                exp = np.exp(logits[~positive])
                values[~positive] = exp / (1 + exp)
            else:
                inputs, outputs = session.get_inputs(), session.get_outputs()
                if (len(inputs) != 1 or len(inputs[0].shape) != 4 or inputs[0].shape[1] != 3
                    or any(isinstance(dim, int) and dim != 448 for dim in inputs[0].shape[2:])
                    or len(outputs) != 1 or outputs[0].shape[-1] != len(rows)):
                    raise ValueError("CL v1.02 ONNX 输入/标签输出契约不匹配")
                logits = session.run([outputs[0].name], {inputs[0].name: cl_input(image)})[0][0]
                positive = logits >= 0
                values = np.empty_like(logits)
                values[positive] = 1 / (1 + np.exp(-logits[positive]))
                exp = np.exp(logits[~positive])
                values[~positive] = exp / (1 + exp)
            result = decode(values, rows, params)
            import onnxruntime as ort
            runtime = dict(name="ONNX Runtime", version=ort.__version__, provider=session.get_providers()[0], dtype="float32")
    return {**result, "engine": engine, "device": device, "runtime": runtime, "params": dict(params)}
