"""Strict, streaming LoRA/LyCORIS patches for Diffusers SD/SDXL components.

The module intentionally knows only tensor headers, live parameter names and the
small set of adapter dialects implemented below.  It never deserializes a whole
Safetensors file and never guesses when more than one live weight can match an
alias.
"""

from __future__ import annotations

import math
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
from safetensors import safe_open


MAX_LORA_KEYS = 100_000
MAX_LORA_GROUPS = 50_000
MAX_LORA_KEY_LENGTH = 512
MAX_LORA_ERROR_LENGTH = 960
MAX_GROUP_WORKSPACE_BYTES = 2 * 1024**3

_FLOAT_DTYPES = {
    "F8_E4M3",
    "F8_E5M2",
    "F16",
    "BF16",
    "F32",
    "F64",
}
_ROLE_SUFFIXES = (
    (".lora_down.weight", "down"),
    (".lora_up.weight", "up"),
    (".lora_A.weight", "down"),
    (".lora_B.weight", "up"),
    (".hada_w1_a", "hada_w1_a"),
    (".hada_w1_b", "hada_w1_b"),
    (".hada_w2_a", "hada_w2_a"),
    (".hada_w2_b", "hada_w2_b"),
    (".hada_t1", "hada_t1"),
    (".hada_t2", "hada_t2"),
    (".lokr_w1_a", "lokr_w1_a"),
    (".lokr_w1_b", "lokr_w1_b"),
    (".lokr_w2_a", "lokr_w2_a"),
    (".lokr_w2_b", "lokr_w2_b"),
    (".lokr_w1", "lokr_w1"),
    (".lokr_w2", "lokr_w2"),
    (".lokr_t2", "lokr_t2"),
    (".dora_scale.weight", "dora_scale"),
    (".dora_scale", "dora_scale"),
    (".alpha", "alpha"),
)
_TARGET_PREFIXES = (
    "lora_unet_",
    "lora_te_",
    "lora_te1_",
    "lora_te2_",
    "lycoris_",
    "unet.",
    "text_encoder.",
    "text_encoder_2.",
    "down_blocks.",
    "up_blocks.",
    "mid_block.",
    "input_blocks.",
    "output_blocks.",
    "middle_block.",
)
_DIRECT_PATCH_MARKERS = (".hada_", ".lokr_", ".dora_scale")
_EXPLICIT_FORBIDDEN_MARKERS = ("lora_mid", "magnitude_vector")
_UPSTREAM_KEY_LIST_OMITTED = "上游标准 LoRA 加载失败（键列表已省略）"
_DIFFUSERS_KEY_DUMP_PHRASES = (
    "the following keys have not been correctly renamed",
    "loading adapter weights from state_dict led to unexpected keys found in the model",
    "loading adapter weights from state_dict led to missing keys in the model",
    "unsupported keys found in state dict",
    "the following keys will be ignored",
    "as unexpected keys while trying to load",
)
_LORA_KEY_MARKERS = (
    "lora_unet_",
    "lora_te_",
    "lora_te1_",
    "lora_te2_",
    ".lora_down",
    ".lora_up",
    ".lora_a.",
    ".lora_b.",
    ".lokr_",
    ".hada_",
    ".dora_scale",
)
_KEY_LIKE_PATTERN = re.compile(
    r"(?i)(?:[a-z0-9_-]+\.){2,}[a-z0-9_-]*(?:weight|bias|alpha)"
)


def _looks_like_upstream_key_dump(text: str) -> bool:
    lower = text.lower()
    if any(phrase in lower for phrase in _DIFFUSERS_KEY_DUMP_PHRASES):
        return True
    # A dependency exception this large is not useful in a user-facing LoRA
    # record. Treat it as an opaque dump even if an upstream release changes
    # key spelling or list punctuation.
    if len(text) > 16_384:
        return True
    separators = text.count(",") + text.count("\n") + text.count("\r")
    marker_hits = sum(lower.count(marker) for marker in _LORA_KEY_MARKERS)
    key_like_hits = 0
    for key_like_hits, _match in enumerate(_KEY_LIKE_PATTERN.finditer(text), start=1):
        if key_like_hits >= 8:
            break
    if marker_hits >= 8 or (key_like_hits >= 8 and len(text) > 512):
        return True
    if separators >= 32 and separators * 128 >= len(text):
        return True
    if separators < 8:
        return False
    return (separators >= 16 and marker_hits >= 2) or (
        separators >= 16 and key_like_hits >= 4
    ) or (len(text) > 2_048 and key_like_hits >= 6)


def _sanitize_lora_error_text(value, source=None) -> str:
    text = str(value)
    if _looks_like_upstream_key_dump(text):
        return _UPSTREAM_KEY_LIST_OMITTED
    text = " ".join(text.split())
    if source is not None:
        source = str(source)
        basename = _display_name(source)
        text = text.replace(source, basename)
        text = text.replace(source.replace("\\", "/"), basename)
    text = re.sub(r"(?i)\b[a-z]:[\\/][^\s]+", "<local-path>", text)
    text = re.sub(r"(?<![\w.])/(?:[^\s]+)", "<local-path>", text)
    return text


class LoraPatchError(ValueError):
    """A bounded exception that is safe to put in a job record or diagnostic log."""

    def __init__(self, message: str):
        super().__init__(_sanitize_lora_error_text(message)[:MAX_LORA_ERROR_LENGTH])


def _display_name(value) -> str:
    name = Path(str(value)).name or "LoRA"
    return name[:160]


def _fail(file_name: str, format_name: str, reason: str):
    raise LoraPatchError(
        f"LoRA 格式不受支持（{format_name}）：{_display_name(file_name)}：{reason}"
    )


def readable_lora_error(file_name: str, format_name: str, reason) -> LoraPatchError:
    """Wrap a dependency/runtime error without leaking a path or an unbounded key dump."""

    text = _sanitize_lora_error_text(reason, source=file_name)
    return LoraPatchError(
        f"LoRA 应用失败（{format_name}）：{_display_name(file_name)}：{text}"
    )


@dataclass(frozen=True)
class TensorHeader:
    key: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class LoraHeader:
    file_name: str
    tensors: tuple[TensorHeader, ...]

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(item.key for item in self.tensors)


@dataclass(frozen=True)
class LoraClassification:
    families: tuple[str, ...]
    family: str
    advanced: bool
    dora: bool
    key_count: int
    group_count: int


@dataclass(frozen=True)
class LoraGroup:
    name: str
    family: str
    roles: tuple[tuple[str, TensorHeader], ...]
    expected_shape: tuple[int, int]
    rank: int | None
    dora: bool
    workspace_bytes: int
    file_name: str = "LoRA"

    def role_map(self) -> dict[str, TensorHeader]:
        return dict(self.roles)


@dataclass(frozen=True)
class TargetRef:
    component: str
    parameter: str
    shape: tuple[int, ...]

    @property
    def qualified_name(self) -> str:
        return f"{self.component}.{self.parameter}"


@dataclass(frozen=True)
class PlannedGroup:
    group: LoraGroup
    target: TargetRef


@dataclass(frozen=True)
class LoraPatchPlan:
    file_name: str
    classification: LoraClassification
    groups: tuple[PlannedGroup, ...]

    @property
    def family(self) -> str:
        return self.classification.family

    @property
    def target_count(self) -> int:
        return len(self.groups)

    @property
    def dora(self) -> bool:
        return self.classification.dora

    def report(self) -> dict:
        return {
            "name": self.file_name,
            "family": self.family,
            "target_count": self.target_count,
            "dora": self.dora,
        }


@dataclass(frozen=True)
class LoraApplyReport:
    family: str
    target_count: int
    dora: bool

    def as_dict(self) -> dict:
        return {
            "family": self.family,
            "target_count": self.target_count,
            "dora": self.dora,
        }


def _shape_numel(shape: Sequence[int], file_name: str, role: str) -> int:
    if len(shape) > 8 or any(not isinstance(value, int) or value <= 0 for value in shape):
        _fail(file_name, role, f"非法 shape {tuple(shape)}")
    value = math.prod(shape)
    if value > 2**40:
        _fail(file_name, role, "shape 超出安全上限")
    return value


def read_lora_header(path) -> LoraHeader:
    """Read keys, shapes and dtypes without calling ``get_tensor``."""

    source = Path(path)
    file_name = _display_name(source)
    try:
        with safe_open(str(source), framework="pt", device="cpu") as handle:
            keys = tuple(handle.keys())
            if not keys:
                _fail(file_name, "未知", "文件不含张量")
            if len(keys) > MAX_LORA_KEYS:
                _fail(file_name, "未知", f"键数量超过上限 {MAX_LORA_KEYS}")
            tensors = []
            for key in keys:
                if not isinstance(key, str) or not key or len(key) > MAX_LORA_KEY_LENGTH:
                    _fail(file_name, "未知键", "键名为空或过长")
                view = handle.get_slice(key)
                shape = tuple(int(value) for value in view.get_shape())
                dtype = str(view.get_dtype())
                _shape_numel(shape or (1,), file_name, key[:80])
                tensors.append(TensorHeader(key=key, shape=shape, dtype=dtype))
    except LoraPatchError:
        raise
    except Exception as error:
        raise readable_lora_error(source, "Safetensors", error) from None
    return LoraHeader(file_name=file_name, tensors=tuple(tensors))


def lora_header_requires_direct_patch(keys: Iterable[str] | LoraHeader) -> bool:
    """Select the strict patch path without parsing native Diffusers/PEFT keys.

    Standard Safetensors naming is owned by Diffusers and is intentionally much
    wider than Kohya. Only an advanced marker (or an explicitly forbidden
    dialect that needs our bounded error) opts a whole file into strict parsing.
    """

    raw_keys = keys.keys if isinstance(keys, LoraHeader) else tuple(keys)
    for key in raw_keys:
        lower = str(key).lower()
        if any(marker in lower for marker in _DIRECT_PATCH_MARKERS):
            return True
        if any(marker in lower for marker in _EXPLICIT_FORBIDDEN_MARKERS):
            return True
    return False


def _parse_key(key: str, file_name: str) -> tuple[str, str]:
    lower = key.lower()
    if "magnitude_vector" in lower:
        _fail(file_name, "DoRA", "不支持 magnitude_vector 别名")
    if "lora_mid" in lower:
        _fail(file_name, "LoCon", "不支持 lora_mid")
    for suffix, role in _ROLE_SUFFIXES:
        if lower.endswith(suffix.lower()):
            name = key[: -len(suffix)]
            if not name.startswith(_TARGET_PREFIXES):
                _fail(file_name, "未知键", f"目标前缀不受支持：{name[:120]}")
            if role in {"lokr_t2", "lokr_w1_a", "lokr_w1_b"}:
                _fail(file_name, "LoKr", f"不支持 {role}")
            if role in {"hada_t1", "hada_t2"}:
                _fail(file_name, "LoHa", f"不支持 {role}")
            return name, role
    _fail(file_name, "未知键", f"无法识别：{key[:160]}")


def classify_lora_keys(keys: Iterable[str] | LoraHeader, file_name="LoRA") -> LoraClassification:
    if isinstance(keys, LoraHeader):
        file_name = keys.file_name
        raw_keys = keys.keys
    else:
        raw_keys = tuple(keys)
    if not raw_keys:
        _fail(file_name, "未知", "文件不含张量")
    if len(raw_keys) > MAX_LORA_KEYS:
        _fail(file_name, "未知", f"键数量超过上限 {MAX_LORA_KEYS}")
    families = set()
    groups = set()
    dora = False
    for key in raw_keys:
        if not isinstance(key, str) or not key or len(key) > MAX_LORA_KEY_LENGTH:
            _fail(file_name, "未知键", "键名为空或过长")
        group, role = _parse_key(key, file_name)
        groups.add(group)
        if role in {"down", "up"}:
            families.add("lora")
        elif role.startswith("hada_"):
            families.add("loha")
        elif role.startswith("lokr_"):
            families.add("lokr")
        elif role == "dora_scale":
            dora = True
    if len(groups) > MAX_LORA_GROUPS:
        _fail(file_name, "未知", f"目标组数量超过上限 {MAX_LORA_GROUPS}")
    ordered = tuple(name for name in ("lora", "loha", "lokr") if name in families)
    if not ordered:
        _fail(file_name, "未知", "没有可支持的 LoRA、LoHa 或 LoKr 组")
    family = "+".join(ordered)
    return LoraClassification(
        families=ordered,
        family=family,
        advanced=bool(set(ordered) & {"loha", "lokr"}) or dora,
        dora=dora,
        key_count=len(raw_keys),
        group_count=len(groups),
    )


def _require_float(entry: TensorHeader, file_name: str, family: str):
    if entry.dtype not in _FLOAT_DTYPES:
        _fail(file_name, family, f"{entry.key[:120]} 不是浮点张量")


def _left_factor_shape(entry: TensorHeader, file_name: str, family: str) -> tuple[int, int]:
    shape = entry.shape
    if len(shape) < 2 or math.prod(shape[2:]) != 1:
        _fail(file_name, family, f"左因子 shape 非法：{shape}")
    return shape[0], shape[1]


def _right_factor_shape(entry: TensorHeader, file_name: str, family: str) -> tuple[int, int]:
    shape = entry.shape
    if len(shape) < 2:
        _fail(file_name, family, f"右因子 shape 非法：{shape}")
    return shape[0], math.prod(shape[1:])


def _workspace_bytes(
    entries: Iterable[TensorHeader], output_elements: int, *, dora: bool
) -> int:
    # All factors converted to FP32 plus a deliberately conservative dense peak.
    # Normal fusion budgets delta/base/result/cast and LoHa's two products with
    # spare room. DoRA additionally holds direction, dense direction, row-scaled
    # target, result and cast while the base and delta are still live. Counting
    # twelve FP32-sized outputs also covers a float64 cast buffer and row norms.
    factor_elements = sum(math.prod(entry.shape or (1,)) for entry in entries)
    dense_copies = 12 if dora else 8
    return 4 * (factor_elements + output_elements * dense_copies)


def build_lora_groups(header: LoraHeader) -> tuple[LoraGroup, ...]:
    classification = classify_lora_keys(header)
    grouped: dict[str, dict[str, TensorHeader]] = {}
    for entry in header.tensors:
        name, role = _parse_key(entry.key, header.file_name)
        roles = grouped.setdefault(name, {})
        if role in roles:
            _fail(header.file_name, classification.family, f"{name[:120]} 含重复角色 {role}")
        roles[role] = entry

    results = []
    for name, role_map in grouped.items():
        for entry in role_map.values():
            _require_float(entry, header.file_name, classification.family)
        roles = set(role_map) - {"alpha", "dora_scale"}
        has_lora = bool(roles & {"down", "up"})
        has_loha = any(role.startswith("hada_") for role in roles)
        has_lokr = any(role.startswith("lokr_") for role in roles)
        if sum((has_lora, has_loha, has_lokr)) != 1:
            _fail(header.file_name, "混合家族", f"{name[:120]} 同时包含多个或没有适配器家族")
        has_dora = "dora_scale" in role_map
        if has_loha and has_dora:
            _fail(header.file_name, "LoHa+DoRA", f"{name[:120]} 的 DoRA 仅支持标准 LoRA/LoKr")
        if "alpha" in role_map and math.prod(role_map["alpha"].shape or (1,)) != 1:
            _fail(header.file_name, "alpha", f"{name[:120]} 的 alpha 必须是标量")

        if has_lora:
            required = {"down", "up"}
            missing = required - roles
            extras = roles - required
            if missing or extras:
                _fail(header.file_name, "LoRA", f"{name[:120]} 缺键或多键 missing={sorted(missing)} extras={sorted(extras)}")
            down = role_map["down"]
            up = role_map["up"]
            rank, columns = _right_factor_shape(down, header.file_name, "LoRA")
            rows, up_rank = _left_factor_shape(up, header.file_name, "LoRA")
            if rank < 1 or rank != up_rank:
                _fail(header.file_name, "LoRA", f"{name[:120]} 的 rank shape 不兼容")
            expected = (rows, columns)
            family = "lora"
        elif has_loha:
            unsupported = roles & {"hada_t1", "hada_t2"}
            if unsupported:
                _fail(header.file_name, "LoHa", f"{name[:120]} 不支持 {sorted(unsupported)}")
            required = {"hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"}
            missing = required - roles
            extras = roles - required
            if missing or extras:
                _fail(header.file_name, "LoHa", f"{name[:120]} 缺键或多键 missing={sorted(missing)} extras={sorted(extras)}")
            rows1, rank1 = _left_factor_shape(role_map["hada_w1_a"], header.file_name, "LoHa")
            rank1b, cols1 = _right_factor_shape(role_map["hada_w1_b"], header.file_name, "LoHa")
            rows2, rank2 = _left_factor_shape(role_map["hada_w2_a"], header.file_name, "LoHa")
            rank2b, cols2 = _right_factor_shape(role_map["hada_w2_b"], header.file_name, "LoHa")
            if rank1 < 1 or not (rank1 == rank1b == rank2 == rank2b):
                _fail(header.file_name, "LoHa", f"{name[:120]} 的四因子 rank 不一致")
            if (rows1, cols1) != (rows2, cols2):
                _fail(header.file_name, "LoHa", f"{name[:120]} 的两个乘积 shape 不一致")
            rank = rank1
            expected = (rows1, cols1)
            family = "loha"
        else:
            unsupported = roles & {"lokr_t2", "lokr_w1_a", "lokr_w1_b"}
            if unsupported:
                _fail(header.file_name, "LoKr", f"{name[:120]} 不支持 {sorted(unsupported)}")
            if "lokr_w1" not in roles:
                _fail(header.file_name, "LoKr", f"{name[:120]} 缺少 lokr_w1")
            w1 = role_map["lokr_w1"]
            if len(w1.shape) != 2:
                _fail(header.file_name, "LoKr", f"{name[:120]} 的 lokr_w1 必须是二维")
            full = "lokr_w2" in roles
            lowrank = bool(roles & {"lokr_w2_a", "lokr_w2_b"})
            if full and lowrank:
                _fail(header.file_name, "LoKr", f"{name[:120]} 混用了 full 与 low-rank w2")
            if not full and not lowrank:
                _fail(header.file_name, "LoKr", f"{name[:120]} 缺少第二因子")
            if full:
                extras = roles - {"lokr_w1", "lokr_w2"}
                w2 = role_map["lokr_w2"]
                if extras or len(w2.shape) < 2 or len(w2.shape) > 5:
                    _fail(header.file_name, "LoKr", f"{name[:120]} 的 full w2 shape/角色非法")
                expected = (w1.shape[0] * w2.shape[0], w1.shape[1] * math.prod(w2.shape[1:]))
                rank = None
                family = "lokr_full"
            else:
                required = {"lokr_w1", "lokr_w2_a", "lokr_w2_b"}
                missing = required - roles
                extras = roles - required
                w2a = role_map.get("lokr_w2_a")
                w2b = role_map.get("lokr_w2_b")
                if missing or extras or len(w2a.shape) != 2 or len(w2b.shape) != 2:
                    _fail(header.file_name, "LoKr", f"{name[:120]} 的 low-rank w2 缺键或 shape 非法")
                rank = w2b.shape[0]
                if rank < 1 or w2a.shape[1] != rank:
                    _fail(header.file_name, "LoKr", f"{name[:120]} 的 low-rank rank shape 不兼容")
                expected = (w1.shape[0] * w2a.shape[0], w1.shape[1] * w2b.shape[1])
                family = "lokr_lowrank"

        if has_dora:
            magnitude = role_map["dora_scale"]
            if not magnitude.shape or magnitude.shape[0] != expected[0] or math.prod(magnitude.shape[1:] or (1,)) != 1:
                _fail(header.file_name, "DoRA", f"{name[:120]} 的 magnitude 必须沿输出行，期望 {expected[0]} 行")
        output_elements = expected[0] * expected[1]
        workspace = _workspace_bytes(role_map.values(), output_elements, dora=has_dora)
        if workspace > MAX_GROUP_WORKSPACE_BYTES:
            _fail(header.file_name, family, f"{name[:120]} 的 FP32 workspace 超出安全上限")
        results.append(
            LoraGroup(
                name=name,
                family=family,
                roles=tuple(sorted(role_map.items())),
                expected_shape=expected,
                rank=rank,
                dora=has_dora,
                workspace_bytes=workspace,
                file_name=header.file_name,
            )
        )
    return tuple(results)


def _component_records(named_params) -> tuple[dict[str, torch.nn.Module], dict[str, torch.Tensor]]:
    modules: dict[str, torch.nn.Module] = {}
    records: dict[str, torch.Tensor] = {}
    if isinstance(named_params, torch.nn.Module):
        modules["unet"] = named_params
    elif isinstance(named_params, Mapping) and named_params and all(
        isinstance(value, torch.nn.Module) or value is None for value in named_params.values()
    ):
        modules = {
            str(name): value for name, value in named_params.items() if isinstance(value, torch.nn.Module)
        }
    elif isinstance(named_params, Mapping):
        for name, value in named_params.items():
            if isinstance(value, torch.Tensor):
                records[str(name)] = value
        return modules, records
    else:
        _fail("LoRA", "目标映射", "named_params 必须是模块或参数映射")
    for component, module in modules.items():
        for name, parameter in module.named_parameters():
            qualified = name if name.startswith(f"{component}.") else f"{component}.{name}"
            records[qualified] = parameter
    return modules, records


def _flatten_alias(prefix: str, path: str) -> str:
    return prefix + path.replace(".", "_")


def _resnet_to_ldm(path: str) -> str:
    replacements = {
        "norm1": "in_layers.0",
        "conv1": "in_layers.2",
        "time_emb_proj": "emb_layers.1",
        "norm2": "out_layers.0",
        "conv2": "out_layers.3",
        "conv_shortcut": "skip_connection",
    }
    first, separator, rest = path.partition(".")
    return replacements.get(first, first) + (separator + rest if separator else "")


def _unet_aliases(records: Mapping[str, torch.Tensor]) -> dict[str, set[str]]:
    paths = [name[len("unet.") :] for name in records if name.startswith("unet.") and name.endswith(".weight")]
    aliases: dict[str, set[str]] = {}

    def add(alias: str, path: str):
        qualified = f"unet.{path}"
        aliases.setdefault(alias, set()).add(qualified)
        if alias.startswith("lora_unet_"):
            aliases.setdefault("lycoris_" + alias[len("lora_unet_") :], set()).add(qualified)

    down_counts = {}
    up_counts = {}
    down_blocks = set()
    up_blocks = set()
    attention_blocks = set()
    for path in paths:
        match = re.match(r"down_blocks\.(\d+)\.resnets\.(\d+)\.", path)
        if match:
            block, index = map(int, match.groups())
            down_blocks.add(block)
            down_counts[block] = max(down_counts.get(block, 0), index + 1)
        match = re.match(r"up_blocks\.(\d+)\.resnets\.(\d+)\.", path)
        if match:
            block, index = map(int, match.groups())
            up_blocks.add(block)
            up_counts[block] = max(up_counts.get(block, 0), index + 1)
        match = re.match(r"up_blocks\.(\d+)\.attentions\.", path)
        if match:
            attention_blocks.add(int(match.group(1)))
    max_down = max(down_blocks, default=-1)

    for path in paths:
        base = path[: -len(".weight")]
        add(_flatten_alias("lora_unet_", base), path)
        add(_flatten_alias("lycoris_", base), path)
        # Comfy-style files may already carry a Diffusers path rather than a
        # Kohya SGM name. These are exact aliases; collisions still fail closed.
        add(base, path)
        add("unet." + base, path)
        processor_base = base.replace(".to_", ".processor.to_")
        if processor_base.endswith(".to_out.0"):
            processor_base = processor_base[:-2]
        add(processor_base, path)
        add("unet." + processor_base, path)
        folded = base.replace("down_blocks", "input_blocks").replace("up_blocks", "output_blocks").replace("mid_block", "middle_block")
        folded_variants = {folded, folded.replace("conv_shortcut", "skip_connection")}
        folded_variants |= {
            value.replace("downsamplers.0.conv", "downsamplers.0.op") for value in tuple(folded_variants)
        }
        for value in folded_variants:
            add(_flatten_alias("lora_unet_", value), path)
            add(_flatten_alias("lycoris_", value), path)
            add(value, path)
            add("unet." + value, path)

        match = re.match(r"down_blocks\.(\d+)\.resnets\.(\d+)\.(.+)", base)
        if match:
            block, index, rest = int(match.group(1)), int(match.group(2)), match.group(3)
            start = 1 + sum(down_counts.get(prior, 0) + (1 if prior < max_down else 0) for prior in range(block))
            add(_flatten_alias("lora_unet_", f"input_blocks.{start + index}.0.{_resnet_to_ldm(rest)}"), path)
            continue
        match = re.match(r"down_blocks\.(\d+)\.attentions\.(\d+)\.(.+)", base)
        if match:
            block, index, rest = int(match.group(1)), int(match.group(2)), match.group(3)
            start = 1 + sum(down_counts.get(prior, 0) + (1 if prior < max_down else 0) for prior in range(block))
            add(_flatten_alias("lora_unet_", f"input_blocks.{start + index}.1.{rest}"), path)
            continue
        match = re.match(r"down_blocks\.(\d+)\.downsamplers\.0\.conv", base)
        if match:
            block = int(match.group(1))
            start = 1 + sum(down_counts.get(prior, 0) + (1 if prior < max_down else 0) for prior in range(block))
            add(_flatten_alias("lora_unet_", f"input_blocks.{start + down_counts.get(block, 0)}.0.op"), path)
            continue
        match = re.match(r"mid_block\.resnets\.(\d+)\.(.+)", base)
        if match:
            index, rest = int(match.group(1)), match.group(2)
            add(_flatten_alias("lora_unet_", f"middle_block.{index * 2}.{_resnet_to_ldm(rest)}"), path)
            continue
        match = re.match(r"mid_block\.attentions\.(\d+)\.(.+)", base)
        if match:
            add(_flatten_alias("lora_unet_", f"middle_block.{int(match.group(1)) * 2 + 1}.{match.group(2)}"), path)
            continue
        match = re.match(r"up_blocks\.(\d+)\.resnets\.(\d+)\.(.+)", base)
        if match:
            block, index, rest = int(match.group(1)), int(match.group(2)), match.group(3)
            output_index = sum(up_counts.get(prior, 0) for prior in range(block)) + index
            add(_flatten_alias("lora_unet_", f"output_blocks.{output_index}.0.{_resnet_to_ldm(rest)}"), path)
            continue
        match = re.match(r"up_blocks\.(\d+)\.attentions\.(\d+)\.(.+)", base)
        if match:
            block, index, rest = int(match.group(1)), int(match.group(2)), match.group(3)
            output_index = sum(up_counts.get(prior, 0) for prior in range(block)) + index
            add(_flatten_alias("lora_unet_", f"output_blocks.{output_index}.1.{rest}"), path)
            continue
        match = re.match(r"up_blocks\.(\d+)\.upsamplers\.0\.conv", base)
        if match:
            block = int(match.group(1))
            output_index = sum(up_counts.get(prior, 0) for prior in range(block + 1)) - 1
            component_index = 2 if block in attention_blocks else 1
            add(_flatten_alias("lora_unet_", f"output_blocks.{output_index}.{component_index}.conv"), path)
    return aliases


def _text_aliases(records: Mapping[str, torch.Tensor]) -> dict[str, set[str]]:
    aliases: dict[str, set[str]] = {}
    for component, prefixes in (
        ("text_encoder", ("lora_te_", "lora_te1_")),
        ("text_encoder_2", ("lora_te2_",)),
    ):
        marker = component + "."
        for qualified in records:
            if not qualified.startswith(marker) or not qualified.endswith(".weight"):
                continue
            path = qualified[len(marker) : -len(".weight")]
            aliases.setdefault(qualified[: -len(".weight")], set()).add(qualified)
            aliases.setdefault(path, set()).add(qualified)
            for prefix in prefixes:
                aliases.setdefault(_flatten_alias(prefix, path), set()).add(qualified)
    return aliases


def _private_converter_candidate(name: str) -> tuple[str, str] | None:
    """Use Diffusers 0.38.x's private converter only as a candidate generator.

    Private symbols are explicitly version-gated. A missing module, changed
    signature, or converter exception deterministically leaves resolution to the
    local exact alias table instead of making availability affect the answer.
    """

    try:
        diffusers_module = import_module("diffusers")
        version = str(getattr(diffusers_module, "__version__", ""))
        match = re.match(r"^(\d+)\.(\d+)\.", version)
        if match is None or tuple(map(int, match.groups())) != (0, 38):
            return None
        converter_module = import_module("diffusers.loaders.lora_conversion_utils")
        convert_text = getattr(converter_module, "_convert_text_encoder_lora_key")
        convert_unet = getattr(converter_module, "_convert_unet_lora_key")
    except Exception:
        return None
    try:
        if name.startswith("lora_unet_"):
            converted = convert_unet(name + ".lora_down.weight")
            component = "unet"
        elif name.startswith(("lora_te_", "lora_te1_", "lora_te2_")):
            converted = convert_text(name + ".lora_down.weight", name)
            component = "text_encoder_2" if name.startswith("lora_te2_") else "text_encoder"
        else:
            return None
    except Exception:
        return None
    converted = converted.replace(".processor.", ".")
    suffixes = (
        ".lora_linear_layer.down.weight",
        ".lora.down.weight",
        "_lora.down.weight",
        ".lora_A.weight",
        ".lora_down.weight",
    )
    for suffix in suffixes:
        if converted.endswith(suffix):
            converted = converted[: -len(suffix)]
            break
    return component, converted + ".weight"


class LoraTargetIndex:
    """The live parameter names and every alias for them, built once per LoRA transaction.

    Resolving a group needs the whole table — every UNet/TE parameter name and every Kohya, SGM,
    Comfy, LyCORIS and Diffusers spelling of it — but only ever looks one entry up. Building that
    table per group made an advanced file cost one full scan of the model per target (788 scans for
    the reference LoKr); one index per `configure_loras` transaction costs one.

    It is deliberately short-lived and holds no tensor: component modules are kept so a lookup
    still reads the *live* parameter (shape, dtype, device) at resolution time, and a mapping of
    names is all the aliases need. An index built for one set of components refuses to answer for
    another, so a rebuilt pipeline or a replaced component can never be resolved against a stale
    table. Nothing here is a cache across transactions.
    """

    def __init__(self, named_params):
        modules, records = _component_records(named_params)
        self._source = named_params
        self._modules = dict(modules)
        # Flat test mappings have no module to read a live parameter from; they keep their tensors.
        self._records = None if modules else dict(records)
        self._names = frozenset(records)
        aliases = _unet_aliases(records)
        for alias, targets in _text_aliases(records).items():
            aliases.setdefault(alias, set()).update(targets)
        self._aliases = {alias: frozenset(targets) for alias, targets in aliases.items()}
        self._converted = {}
        self.builds = 1
        self.lookups = 0

    def covers(self, named_params) -> bool:
        if named_params is self._source:
            return True
        if not self._modules or not isinstance(named_params, Mapping):
            return False
        live = {
            str(name): value for name, value in named_params.items() if isinstance(value, torch.nn.Module)
        }
        return live.keys() == self._modules.keys() and all(
            live[name] is module for name, module in self._modules.items()
        )

    def aliases_for(self, name: str) -> frozenset[str]:
        return self._aliases.get(name, frozenset())

    def has(self, qualified: str) -> bool:
        return qualified in self._names

    def converter_candidate(self, name: str):
        # Looked up through the module-level function every time it is first asked, so the version
        # gate (and a test's patch of it) still decides; the memo only spans this transaction.
        if name not in self._converted:
            self._converted[name] = _private_converter_candidate(name)
        return self._converted[name]

    def parameter(self, qualified: str):
        component, parameter_name = qualified.split(".", 1)
        if self._records is not None:
            return self._records[qualified]
        return self._modules[component].get_parameter(parameter_name)

    def counts(self) -> dict:
        return {"target_index_builds": self.builds, "target_lookups": self.lookups}


def build_target_index(named_params) -> LoraTargetIndex:
    return LoraTargetIndex(named_params)


def resolve_target(
    named_params,
    kohya_name: str,
    family: str,
    expected_shape: Sequence[int] | None = None,
    *,
    file_name="LoRA",
    require_materialized=False,
    index: LoraTargetIndex | None = None,
) -> TargetRef:
    """Resolve one adapter group against real live parameter names.

    ``named_params`` may be ``{"unet": module, "text_encoder": module, ...}``
    (the production path) or a flat qualified-name-to-tensor mapping (tests).
    ``index`` is the transaction's `LoraTargetIndex` for those same components;
    without one, a private index is built for this single lookup.
    """

    if index is None:
        index = LoraTargetIndex(named_params)
    elif not index.covers(named_params):
        _fail(file_name, "目标映射", "目标索引不属于当前 pipeline 组件，必须重建")
    index.lookups += 1
    if family == "sd" and kohya_name.startswith(("lora_te2_", "text_encoder_2.")):
        _fail(file_name, "TE2", "SD 模型没有 text_encoder_2，te2 目标只能用于 SDXL/iL")
    fallback_matches = set(index.aliases_for(kohya_name))
    matches = set(fallback_matches)
    candidate = index.converter_candidate(kohya_name)
    if candidate is not None:
        qualified = f"{candidate[0]}.{candidate[1]}"
        if index.has(qualified):
            if fallback_matches and fallback_matches != {qualified}:
                _fail(
                    file_name,
                    "目标映射",
                    f"Diffusers映射版本不兼容：{kohya_name[:120]} 的私有转换结果与受限 fallback 冲突",
                )
            matches.add(qualified)
    if not matches:
        _fail(file_name, "目标映射", f"找不到目标 {kohya_name[:180]}")
    if len(matches) != 1:
        _fail(file_name, "目标映射", f"目标 {kohya_name[:140]} 存在歧义（{len(matches)} 个候选）")
    qualified = next(iter(matches))
    component, parameter_name = qualified.split(".", 1)
    try:
        parameter = index.parameter(qualified)
    except (AttributeError, KeyError) as error:
        _fail(file_name, "目标映射", f"实时参数 {qualified[:180]} 不存在：{error}")
    shape = tuple(parameter.shape)
    if not parameter.is_floating_point():
        _fail(file_name, "目标映射", f"目标 {qualified[:180]} 不是浮点权重")
    if len(shape) < 2:
        _fail(file_name, "目标映射", f"目标 {qualified[:180]} shape 非法：{shape}")
    flat_shape = (shape[0], math.prod(shape[1:]))
    if expected_shape is not None and tuple(expected_shape) != flat_shape:
        _fail(
            file_name,
            "shape",
            f"{kohya_name[:120]} 重建 {tuple(expected_shape)}，目标 {qualified[:120]} 为 {shape}",
        )
    if require_materialized and (parameter.device.type == "meta" or (parameter.numel() and parameter.data_ptr() == 0)):
        _fail(file_name, "驻留状态", f"目标 {qualified[:160]} 尚未 materialize，不能直接补丁")
    return TargetRef(component=component, parameter=parameter_name, shape=shape)


def preflight_lora_patch(
    path,
    named_params,
    family: str,
    *,
    header: LoraHeader | None = None,
    groups: Sequence[LoraGroup] | None = None,
    require_materialized=True,
    index: LoraTargetIndex | None = None,
) -> LoraPatchPlan:
    """Map every group of one file before anything is written.

    ``index`` is the transaction's `LoraTargetIndex`; every advanced file of one
    transaction shares it. Without one, the file builds its own, once.
    """

    header = header or read_lora_header(path)
    classification = classify_lora_keys(header)
    groups = tuple(groups) if groups is not None else build_lora_groups(header)
    if index is None:
        index = LoraTargetIndex(named_params)
    planned = []
    claimed = {}
    for group in groups:
        target = resolve_target(
            named_params,
            group.name,
            family,
            group.expected_shape,
            file_name=header.file_name,
            require_materialized=require_materialized,
            index=index,
        )
        previous = claimed.setdefault(target.qualified_name, group.name)
        if previous != group.name:
            _fail(
                header.file_name,
                "重复目标",
                f"{previous[:100]} 与 {group.name[:100]} 指向同一参数 {target.qualified_name[:120]}",
            )
        target_elements = math.prod(target.shape)
        if target_elements != group.expected_shape[0] * group.expected_shape[1]:
            _fail(header.file_name, "shape", f"{group.name[:120]} 的重建元素数与目标不一致")
        planned.append(PlannedGroup(group=group, target=target))
    return LoraPatchPlan(
        file_name=header.file_name,
        classification=classification,
        groups=tuple(planned),
    )


def validate_lora_plan_materialized(plan: LoraPatchPlan, named_params) -> None:
    """Recheck every planned live target after residency preparation.

    Structural mapping can run against meta/offloaded parameters, but no tensor
    is read and no group is written until every target is materialized again.
    """

    for planned in plan.groups:
        try:
            weight = _get_target_weight(named_params, planned.target)
        except Exception as error:
            raise readable_lora_error(plan.file_name, "目标映射", error) from None
        if tuple(weight.shape) != planned.target.shape:
            _fail(plan.file_name, "shape", f"{planned.target.qualified_name[:160]} 在 preflight 后发生变化")
        if not weight.is_floating_point():
            _fail(plan.file_name, "目标映射", f"目标 {planned.target.qualified_name[:160]} 不是浮点权重")
        if weight.device.type == "meta" or (weight.numel() and weight.data_ptr() == 0):
            _fail(plan.file_name, "驻留状态", f"目标 {planned.target.qualified_name[:160]} 尚未 materialize，不能直接补丁")


def _normalized_device(device) -> torch.device:
    """`cuda` and `cuda:0` are one card; compare them as one."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _require(flag, fail, checks) -> None:
    """Run one numeric check now, or queue it to be read back with the rest of its group.

    ``checks`` is None on the established path: every check is read as it is made. A device pass
    queues its checks instead and `_settle` reads them in one transfer; the queue keeps their order,
    so the first failure reported is the one the immediate path would have reported.
    """
    if checks is None:
        if not (flag.item() if isinstance(flag, torch.Tensor) else flag):
            fail()
    else:
        checks.append((flag, fail))


def _settle(checks) -> None:
    """Read every queued check back at once and raise the first that failed, in queue order."""
    if not checks:
        return
    remote = [flag for flag, _fail in checks if isinstance(flag, torch.Tensor) and flag.device.type != "cpu"]
    values = iter(torch.stack([flag.reshape(()) for flag in remote]).tolist()) if remote else iter(())
    pending = list(checks)
    checks.clear()
    for flag, fail in pending:
        if isinstance(flag, torch.Tensor):
            value = next(values) if flag.device.type != "cpu" else flag.item()
        else:
            value = flag
        if not value:
            fail()


def _finite_tensor(
    tensor: torch.Tensor, role: str, group: LoraGroup, device, checks=None, upload=None
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
        _settle(checks)
        _fail(group.file_name, group.family, f"{group.name[:120]} 的 {role} 不是浮点张量")
    if upload is not None and _normalized_device(device) != tensor.device:
        # Crosses at its stored width, widened where it lands: the same FP32 values.
        value = upload(tensor, f"factor:{role}").to(dtype=torch.float32)
    else:
        value = tensor.detach().to(device=device, dtype=torch.float32)
    _require(
        torch.isfinite(value).all(),
        lambda: _fail(group.file_name, group.family, f"{group.name[:120]} 的 {role} 含 NaN/Inf"),
        checks,
    )
    return value


def _alpha_value(group: LoraGroup, tensors: Mapping[str, torch.Tensor], rank: int, device, checks=None) -> float:
    if "alpha" not in tensors:
        return float(rank)
    # A scalar is read where it already is: reading it off a device would cost a synchronisation.
    source = tensors["alpha"]
    alpha = _finite_tensor(source, "alpha", group, source.device if isinstance(source, torch.Tensor) else device, checks)
    if alpha.numel() != 1:
        _settle(checks)
        _fail(group.file_name, group.family, f"{group.name[:120]} 的 alpha 必须是标量")
    value = float(alpha.item())
    _require(
        math.isfinite(value),
        lambda: _fail(group.file_name, group.family, f"{group.name[:120]} 的 alpha 非有限"),
        checks,
    )
    return value


def _left_matrix(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0], tensor.shape[1])


def _right_matrix(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0], -1)


def lora_delta(
    group: LoraGroup,
    tensors: Mapping[str, torch.Tensor],
    *,
    device=None,
    product_device=None,
    checks=None,
    upload=None,
) -> torch.Tensor:
    """Reconstruct one validated group in FP32 on ``device``.

    ``product_device`` is where the factors of a matrix product are checked and multiplied; it
    defaults to ``device``. Only the element-wise assembly (Kronecker product, scaling) happens on
    ``device`` after that, and an element-wise FP32 operation rounds the same on every device — so
    a host product with a device assembly is bit-for-bit the all-host reconstruction, while a
    device product would not be (a different summation order, and TF32 wherever it is enabled).
    Full LoKr forms no product, so its factors go straight to ``device``. The scalar alpha is always
    read where it was loaded. ``checks`` queues the numeric checks (see `_require`); ``upload(tensor,
    name)`` moves a host tensor to ``device`` (a `_WriteBackPipe` stages it through page-locked
    memory, so a transfer never waits for the device work queued before it).
    """

    device = _normalized_device(device or next(iter(tensors.values())).device)
    product_device = _normalized_device(product_device) if product_device is not None else device
    if group.family == "lokr_full":
        product_device = device

    def moved(tensor, name):
        if tensor.device == device:
            return tensor
        return upload(tensor, name) if upload is not None else tensor.to(device)

    values = {
        role: _finite_tensor(
            tensor, role, group, tensor.device if role == "alpha" and isinstance(tensor, torch.Tensor) else product_device,
            checks, upload,
        )
        for role, tensor in tensors.items()
        if role != "dora_scale"
    }
    if group.family == "lora":
        down = _right_matrix(values["down"])
        up = _left_matrix(values["up"])
        alpha = _alpha_value(group, values, down.shape[0], product_device, checks)
        delta = moved(up @ down, "product") * (alpha / down.shape[0])
    elif group.family == "loha":
        w1a = _left_matrix(values["hada_w1_a"])
        w1b = _right_matrix(values["hada_w1_b"])
        w2a = _left_matrix(values["hada_w2_a"])
        w2b = _right_matrix(values["hada_w2_b"])
        rank = w1b.shape[0]
        alpha = _alpha_value(group, values, rank, product_device, checks)
        delta = moved(w1a @ w1b, "product") * moved(w2a @ w2b, "product_2") * (alpha / rank)
    elif group.family == "lokr_full":
        # Alpha is accepted and validated for compatibility, but full LoKr has
        # no rank normalisation and therefore no alpha multiplier.
        _alpha_value(group, values, 1, product_device, checks)
        w1 = moved(values["lokr_w1"], "w1")
        w2 = moved(values["lokr_w2"], "w2")
        for _ in range(w2.ndim - w1.ndim):
            w1 = w1.unsqueeze(-1)
        delta = torch.kron(w1, w2).reshape(group.expected_shape)
    elif group.family == "lokr_lowrank":
        w2b = values["lokr_w2_b"]
        rank = w2b.shape[0]
        alpha = _alpha_value(group, values, rank, product_device, checks)
        product = moved(values["lokr_w2_a"] @ w2b, "product")
        delta = torch.kron(moved(values["lokr_w1"], "w1"), product) * (alpha / rank)
    else:  # pragma: no cover - LoraGroup is constructed only by the strict builder
        _fail(group.file_name, group.family, "未知重建家族")

    def invalid():
        _fail(group.file_name, group.family, f"{group.name[:120]} 的重建 shape/数值非法")

    if tuple(delta.shape) != group.expected_shape:
        _settle(checks)
        invalid()
    _require(torch.isfinite(delta).all(), invalid, checks)
    return delta


def apply_lora_patch(
    module_weight: torch.Tensor,
    delta: torch.Tensor,
    multiplier: float,
    dora_magnitude: torch.Tensor | None = None,
    *,
    compute_device=None,
    checks=None,
) -> None:
    """Apply one dense direction in FP32 and copy back only after all checks pass.

    ``compute_device`` runs the dense arithmetic somewhere other than where the weight lives; every
    step is element-wise (the FP32 cast, scale, sum, finite checks and the round to the weight's
    dtype), so the bytes written back are the ones the weight's own device would have produced.
    DoRA's row norms are a reduction and are therefore never moved. ``checks`` carries the group's
    queued checks (see `_require`); all of them are read back before the only write.
    """

    cast = _patched_weight(
        module_weight, delta, multiplier, dora_magnitude, compute_device=compute_device, checks=checks,
    )
    if cast is None:
        return
    _settle(checks)
    with torch.no_grad():
        module_weight.copy_(cast)


def _patched_weight(
    module_weight: torch.Tensor,
    delta: torch.Tensor,
    multiplier: float,
    dora_magnitude: torch.Tensor | None = None,
    *,
    compute_device=None,
    checks=None,
    upload=None,
):
    """Everything `apply_lora_patch` does except the write: the patched weight in its own dtype.

    None for an exact-zero multiplier (nothing is ever written then). With ``checks`` the numeric
    checks are queued, not read, and the caller reads them before writing. ``upload`` moves the base
    to ``compute_device`` (a `_WriteBackPipe` stages it through page-locked memory).
    """

    def refuse(message):
        # Anything queued earlier in the group was checked first on the immediate path.
        _settle(checks)
        raise LoraPatchError(message)

    if isinstance(multiplier, bool) or not isinstance(multiplier, (int, float)) or not math.isfinite(float(multiplier)):
        refuse("LoRA 应用失败（权重）：multiplier 必须是有限数")
    if float(multiplier) == 0.0:
        _settle(checks)
        return None
    if not isinstance(module_weight, torch.Tensor) or not module_weight.is_floating_point():
        refuse("LoRA 应用失败（目标）：目标必须是浮点权重")
    if module_weight.device.type == "meta" or (module_weight.numel() and module_weight.data_ptr() == 0):
        refuse("LoRA 应用失败（驻留状态）：目标权重尚未 materialize")
    device = module_weight.device if compute_device is None or dora_magnitude is not None else _normalized_device(compute_device)
    # Move at the weight's own width, widen where the arithmetic runs: half the transfer, same bits.
    moved = upload(module_weight) if upload is not None and device != module_weight.device else (
        module_weight.detach().to(device=device)
    )
    base = moved.to(dtype=torch.float32).reshape(module_weight.shape[0], -1)
    direction = delta.detach().to(device=device, dtype=torch.float32)
    if tuple(direction.shape) != tuple(base.shape):
        refuse(f"LoRA 应用失败（shape）：重建 {tuple(direction.shape)}，目标 {tuple(module_weight.shape)}")

    def numeric(message):
        return lambda: refuse(message)

    _require(
        torch.isfinite(base).all() & torch.isfinite(direction).all(),
        numeric("LoRA 应用失败（数值）：基础权重或增量含 NaN/Inf"),
        checks,
    )
    if dora_magnitude is None:
        result = base + float(multiplier) * direction
    else:
        _settle(checks)
        magnitude = dora_magnitude.detach().to(device=device, dtype=torch.float32).reshape(-1)
        if magnitude.numel() != base.shape[0] or not torch.isfinite(magnitude).all().item():
            raise LoraPatchError("LoRA 应用失败（DoRA）：magnitude 行数或数值非法")
        dense_direction = base + direction
        norms = torch.linalg.vector_norm(dense_direction, dim=1)
        if not torch.isfinite(norms).all().item() or torch.any(norms == 0).item():
            raise LoraPatchError("LoRA 应用失败（DoRA）：direction 输出行范数必须有限且非零")
        target = dense_direction * (magnitude / norms).unsqueeze(1)
        result = base + float(multiplier) * (target - base)
    _require(torch.isfinite(result).all(), numeric("LoRA 应用失败（数值）：FP32 重建结果溢出或非有限"), checks)
    cast = result.reshape(module_weight.shape).to(dtype=module_weight.dtype)
    _require(torch.isfinite(cast).all(), numeric("LoRA 应用失败（数值）：回写 dtype 溢出或非有限"), checks)
    return cast


class _WriteBackPipe:
    """Two-deep pipeline for device units whose results are written back into host tensors.

    A unit is staged through page-locked memory and uploaded without blocking, computed, downloaded
    into a page-locked slot beside its queued check flags, and an event is recorded (`queue`). Only
    then is the unit before it finished (`advance`): its event waited on, its checks read back in
    queue order — the first failure raising exactly what the immediate path would have raised — and
    its bytes copied into its host tensor. Writes therefore land in unit order, each only after its
    own checks, and the host work of one unit overlaps the device work of the one before. A unit
    whose own queueing fails is never written; `drain` finishes the one before it first, so an error
    in that earlier unit still takes precedence, as it would serially. Copies only: the bytes written
    are the bytes computed.
    """

    def __init__(self, device):
        self.device = _normalized_device(device)
        if self.device.type != "cuda":
            raise ValueError("LoRA compute_device must be a CUDA device")
        self._slots = {}
        self._turn = 0
        self._pending = None
        self._next = None

    def _slot(self, name, count, dtype):
        key = (name, self._turn)
        buffer = self._slots.get(key)
        if buffer is None or buffer.dtype != dtype or buffer.numel() < count:
            # The unit that last used this slot was finished two units ago, so replacing it is safe.
            buffer = torch.empty(max(count, 64), dtype=dtype, pin_memory=True)
            self._slots[key] = buffer
        return buffer[:count]

    def upload(self, tensor, name="weight"):
        staged = self._slot(name, tensor.numel(), tensor.dtype).view(tensor.shape)
        staged.copy_(tensor.detach())
        return staged.to(self.device, non_blocking=True)

    def upload_parts(self, parts, name="parts"):
        """Pack several host tensors of one dtype into one staged transfer; returns device views."""
        total = sum(part.numel() for part in parts)
        staged = self._slot(name, total, parts[0].dtype)
        torch.cat([part.detach().reshape(-1) for part in parts], out=staged)
        packed = staged.to(self.device, non_blocking=True)
        views, offset = [], 0
        for part in parts:
            views.append(packed[offset:offset + part.numel()].view(part.shape))
            offset += part.numel()
        return views

    def queue(self, target, result, checks, after=None):
        out = self._slot("result", result.numel(), result.dtype).view(result.shape)
        out.copy_(result, non_blocking=True)
        remote = [flag for flag, _fail in checks if isinstance(flag, torch.Tensor) and flag.device.type != "cpu"]
        flags = None
        if remote:
            flags = self._slot("flags", len(remote), torch.bool)
            flags.copy_(torch.stack([flag.reshape(()) for flag in remote]), non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self._next = (event, out, target, list(checks), flags, after)
        checks.clear()
        self._turn ^= 1

    def advance(self):
        """Finish the unit before the one just queued; its error, if any, is raised here."""
        previous, self._pending, self._next = self._pending, self._next, None
        if previous is not None:
            self._finish(previous)

    def drain(self):
        """Finish whatever is outstanding, in order; a unit whose queueing failed is not among it."""
        pending, self._pending = self._pending, None
        queued, self._next = self._next, None
        for unit in (pending, queued):
            if unit is not None:
                self._finish(unit)

    @staticmethod
    def _finish(unit):
        event, out, target, checks, flags, after = unit
        event.synchronize()
        values = iter(flags.tolist()) if flags is not None else iter(())
        for flag, fail in checks:
            if isinstance(flag, torch.Tensor):
                value = next(values) if flag.device.type != "cpu" else flag.item()
            else:
                value = flag
            if not value:
                fail()
        with torch.no_grad():
            target.copy_(out)
        if after is not None:
            after()


HOST_PREFAULT_CHUNK_BYTES = 256 * 1024**2


class HostWeightPrefault:
    """Make a mount's host target weights privately resident before anything is written to them.

    A checkpoint loaded from Safetensors keeps its weights as copy-on-write views of the mapped file.
    The first write to each page then faults, and when that write is a device-to-host copy the driver
    takes every fault itself: measured on the reference job, the six standard folds spent about 15 s
    of their 21 s there, while the same bytes written by a host copy cost about 4 s (QPlan P5 Phase C).

    This takes those faults early, on a thread, while Diffusers is still mapping and injecting the
    standard adapters — Python work that never reads or writes base weight data. Each queued range is
    rewritten with ``x | 0`` on its raw bytes, which changes no bit (NaN payloads included). The
    owner must call `finish` before it writes any queued weight: a read-modify-write racing the patch
    could otherwise undo it. Ranges are merged per storage and processed in a few large chunks, so
    the thread holds the interpreter lock only between them. A failure here is never a mount
    failure; the patch then simply takes the faults itself.
    """

    def __init__(self):
        self._queue = []
        self._seen = set()
        self._condition = threading.Condition()
        self._closed = False
        self.bytes = 0
        self.busy_seconds = 0.0
        self._thread = threading.Thread(target=self._run, name="lora-host-prefault", daemon=True)
        self._thread.start()

    def add(self, tensors) -> None:
        batch = []
        for tensor in tensors:
            if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or not tensor.numel():
                continue
            if not tensor.is_contiguous():
                continue
            key = (tensor.untyped_storage().data_ptr(), tensor.storage_offset() * tensor.element_size())
            if key in self._seen:
                continue
            self._seen.add(key)
            batch.append(tensor)
        ranges = _merged_storage_ranges(batch)
        if not ranges:
            return
        with self._condition:
            self._queue.extend(ranges)
            self._condition.notify()

    # Page faults, not arithmetic, bound this work, and every extra thread taking faults contends
    # with the Diffusers load it is meant to hide behind. The count is this thread's own: OpenMP keeps
    # it per thread, so the caller's intra-op pool is untouched.
    threads = 2

    def _run(self):
        torch.set_num_threads(max(1, int(self.threads)))
        while True:
            with self._condition:
                while not self._queue and not self._closed:
                    self._condition.wait()
                if not self._queue:
                    return
                storage, start, length = self._queue.pop(0)
            begin = time.perf_counter()
            try:
                with torch.no_grad():
                    offset = 0
                    while offset < length:
                        size = min(HOST_PREFAULT_CHUNK_BYTES, length - offset)
                        view = torch.empty(0, dtype=torch.uint8).set_(storage, start + offset, (size,))
                        view.bitwise_or_(0)
                        offset += size
                self.bytes += length
            except Exception:
                pass
            self.busy_seconds += time.perf_counter() - begin

    def finish(self) -> float:
        """Drain the queue and stop the thread; afterwards the queued weights may be written."""
        with self._condition:
            self._closed = True
            self._condition.notify()
        self._thread.join()
        return round(self.busy_seconds, 3)


def _merged_storage_ranges(tensors):
    spans = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        start = tensor.storage_offset() * tensor.element_size()
        spans.setdefault(storage.data_ptr(), (storage, []))[1].append(
            (start, start + tensor.numel() * tensor.element_size())
        )
    merged = []
    for storage, intervals in spans.values():
        intervals.sort()
        current_start, current_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                merged.append((storage, current_start, current_end - current_start))
                current_start, current_end = start, end
        merged.append((storage, current_start, current_end - current_start))
    return merged


def _device_out_of_memory(error: BaseException) -> bool:
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()


def _release_device_workspace(device) -> None:
    if device is not None and torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


@contextmanager
def _ieee_fp32_matmul():
    """Full-precision FP32 products for the span of a fold, whatever the process's CUDA math mode.

    PEFT's host fold forms its products in IEEE FP32; the balanced CUDA math setting would
    otherwise let an FP32 product on the card round through TF32. Restored on every exit.
    """

    precision = torch.get_float32_matmul_precision()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = tf32


def _host_vanilla_lora_linear(module, names, linear_type) -> bool:
    """Only the case whose host arithmetic is replicated below; everything else stays with PEFT."""

    if type(module) is not linear_type or getattr(module, "fan_in_fan_out", False):
        return False
    weight = getattr(module.get_base_layer(), "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.device.type != "cpu":
        return False
    if weight.dtype not in (torch.float16, torch.bfloat16):
        return False
    for name in names:
        if name in getattr(module, "lora_variant", {}) or getattr(module, "lora_bias", {}).get(name, False):
            return False
        scaling = module.scaling.get(name)
        if isinstance(scaling, bool) or not isinstance(scaling, (int, float)):
            return False
        for factor in (module.lora_A[name].weight, module.lora_B[name].weight):
            if factor.device.type != "cpu" or factor.dtype != weight.dtype:
                return False
    return True


def _folded_on_device(module, names, device, pipe=None) -> torch.Tensor:
    """One layer's base with every named adapter folded in, on ``device``, in PEFT's host order."""

    weight = module.get_base_layer().weight
    # Every factor of the layer crosses in one transfer rather than two per adapter; they share the
    # weight's dtype (checked by `_host_vanilla_lora_linear`), so packing them is a plain byte copy.
    factors = [
        tensor.detach()
        for name in names
        for tensor in (module.lora_A[name].weight, module.lora_B[name].weight)
    ]
    if pipe is not None:
        work = pipe.upload(weight)
        unpacked = pipe.upload_parts(factors, name="factors")
    else:
        work = weight.detach().to(device)
        packed = torch.cat([tensor.reshape(-1) for tensor in factors]).to(device)
        offset = 0
        unpacked = []
        for tensor in factors:
            unpacked.append(packed[offset:offset + tensor.numel()].view(tensor.shape))
            offset += tensor.numel()
    for index, name in enumerate(names):
        factor_a = unpacked[2 * index].float()
        factor_b = unpacked[2 * index + 1].float()
        work += ((factor_b @ factor_a) * module.scaling[name]).to(weight.dtype)
    return work


def fold_peft_adapters(components, adapter_names: Sequence[str], compute_device) -> dict:
    """Fold mounted standard adapters into their base weights, as Diffusers' `fuse_lora` does.

    Diffusers has already done everything that decides *what* is folded — key conversion, the
    TE1/TE2 mapping, PEFT injection and the per-adapter scaling `set_adapters` wrote. This is only
    the fold. PEFT folds a host-resident FP16/BF16 layer by widening both factors to FP32, forming
    ``B @ A``, scaling it, rounding it to the weight's dtype and adding it in place, adapter by
    adapter in request order. For a plain LoRA Linear on the host that is replicated here with the
    weight moved to ``compute_device`` once for all adapters: every step but the product is
    element-wise and rounds identically there; the product runs in IEEE FP32 (never TF32), so it
    can differ from the host's only by the summation order inside one FP32 dot product.

    Every other layer — convolutions, DoRA and other variants, LoRA biases, fan-in/fan-out,
    anything already on an accelerator — is folded by PEFT's own ``merge``. PEFT's bookkeeping
    (``merged_adapters``) is kept for both, so a failure part-way leaves each layer either wholly
    folded and marked or untouched, exactly as PEFT's own partial failure does.
    """

    from peft.tuners.lora.layer import Linear as PeftLoraLinear
    from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

    device = torch.device(compute_device)
    requested = list(adapter_names)
    folded = delegated = fallbacks = 0
    merged = set()
    pipe = _WriteBackPipe(device)

    def mark(module, names):
        # PEFT's own bookkeeping, recorded only once the layer's bytes are in place.
        def done():
            module.merged_adapters.extend(names)
            merged.update(module.merged_adapters)
        return done

    with _ieee_fp32_matmul():
        for component in components.values():
            if not isinstance(component, torch.nn.Module):
                continue
            for module in component.modules():
                if not isinstance(module, BaseTunerLayer):
                    continue
                names = [name for name in check_adapters_to_merge(module, requested) if name in module.lora_A]
                if not names:
                    continue
                if device is not None and _host_vanilla_lora_linear(module, names, PeftLoraLinear):
                    weight = module.get_base_layer().weight
                    try:
                        work = _folded_on_device(module, names, device, pipe)
                        pipe.queue(weight, work, [], after=mark(module, names))
                    except Exception as error:
                        # Layers already queued are complete computations: land them first.
                        pipe.drain()
                        if not _device_out_of_memory(error):
                            raise
                        # Nothing of this layer has been written: the base is only replaced once
                        # queued. PEFT folds this layer and every one after it on the host instead.
                        _release_device_workspace(device)
                        device = None
                        fallbacks += 1
                    else:
                        pipe.advance()
                        folded += 1
                        continue
                module.merge(safe_merge=False, adapter_names=requested)
                delegated += 1
                merged.update(module.merged_adapters)
        pipe.drain()
    return {
        "folded_layers": folded,
        "delegated_layers": delegated,
        "device_fallbacks": fallbacks,
        "merged_adapters": merged,
    }


def _get_target_weight(named_params, target: TargetRef) -> torch.Tensor:
    if isinstance(named_params, Mapping) and target.component in named_params and isinstance(
        named_params[target.component], torch.nn.Module
    ):
        return named_params[target.component].get_parameter(target.parameter)
    _modules, records = _component_records(named_params)
    return records[target.qualified_name]


def apply_lora_plan(
    path, plan: LoraPatchPlan, named_params, multiplier: float, *, compute_device=None, stats=None
) -> LoraApplyReport:
    """Stream one planned file into its targets, one group at a time.

    ``compute_device`` (an accelerator) takes the dense, element-wise part of each group whose
    target lives on the host: the factors are still read, checked and multiplied on the host in
    FP32 exactly as before, so what is written back is bit-for-bit the all-host result. Targets
    already on an accelerator, and DoRA groups (whose row norms are a reduction), keep the
    established path on the weight's own device. An accelerator that runs out of memory hands the
    group it failed on, and every later one, to the host; ``stats`` counts those hand-overs.
    """

    if isinstance(multiplier, bool) or not isinstance(multiplier, (int, float)) or not math.isfinite(float(multiplier)):
        raise readable_lora_error(plan.file_name, plan.family, "multiplier 必须是有限数")
    # Exact zero is still fully structurally preflighted, but it performs no file
    # tensor reads and no write whatsoever.
    if float(multiplier) == 0.0:
        return LoraApplyReport(plan.family, plan.target_count, plan.dora)
    pipe = _WriteBackPipe(compute_device) if compute_device is not None else None
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for planned in plan.groups:
                group = planned.group
                try:
                    weight = _get_target_weight(named_params, planned.target)
                    if tuple(weight.shape) != planned.target.shape:
                        _fail(plan.file_name, "shape", f"{planned.target.qualified_name[:160]} 在 preflight 后发生变化")
                    tensors = {}
                    for role, entry in group.roles:
                        tensors[role] = handle.get_tensor(entry.key)
                except Exception:
                    # Serially, every group before this one was written before it was even read.
                    if pipe is not None:
                        pipe.drain()
                    raise
                magnitude = tensors.get("dora_scale") if group.dora else None
                if pipe is not None and weight.device.type == "cpu" and not group.dora:
                    try:
                        # One read-back per group instead of one per check; same checks, same order.
                        checks = []
                        delta = lora_delta(
                            group, tensors, device=compute_device, product_device=weight.device, checks=checks,
                            upload=pipe.upload,
                        )
                        cast = _patched_weight(
                            weight, delta, multiplier, magnitude,
                            compute_device=compute_device, checks=checks, upload=pipe.upload,
                        )
                        if cast is not None:
                            pipe.queue(weight, cast, checks)
                    except Exception as error:
                        # The groups before this one are finished first: serially they were written
                        # before this one was read, and an error of theirs takes precedence.
                        pipe.drain()
                        if not _device_out_of_memory(error):
                            raise
                        # The write back is the last step, so this group is untouched; it and every
                        # group after it take the host path, which produces the same bytes.
                        delta = cast = None
                        _release_device_workspace(compute_device)
                        compute_device = pipe = None
                        if stats is not None:
                            stats["device_fallbacks"] = stats.get("device_fallbacks", 0) + 1
                    else:
                        pipe.advance()
                        del delta, cast, tensors, magnitude
                        continue
                if pipe is not None:
                    # A group kept on its own device writes after every group queued before it.
                    pipe.drain()
                delta = lora_delta(group, tensors, device=weight.device)
                apply_lora_patch(weight, delta, multiplier, magnitude)
                del delta, tensors, magnitude
            if pipe is not None:
                pipe.drain()
    except LoraPatchError as error:
        # Ensure every public failure names the actual file even when the low-level
        # numeric helper produced a generic bounded message.
        if plan.file_name in str(error):
            raise
        raise readable_lora_error(path, plan.family, error) from None
    except Exception as error:
        raise readable_lora_error(path, plan.family, error) from None
    return LoraApplyReport(plan.family, plan.target_count, plan.dora)


__all__ = [
    "HostWeightPrefault",
    "LoraApplyReport",
    "LoraClassification",
    "LoraGroup",
    "LoraHeader",
    "LoraPatchError",
    "LoraPatchPlan",
    "LoraTargetIndex",
    "PlannedGroup",
    "TargetRef",
    "TensorHeader",
    "apply_lora_patch",
    "apply_lora_plan",
    "build_lora_groups",
    "build_target_index",
    "classify_lora_keys",
    "fold_peft_adapters",
    "lora_header_requires_direct_patch",
    "lora_delta",
    "preflight_lora_patch",
    "read_lora_header",
    "readable_lora_error",
    "resolve_target",
    "validate_lora_plan_materialized",
]
