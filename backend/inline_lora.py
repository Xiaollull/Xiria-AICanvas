"""Bounded, trusted-root resolution for Forge-style inline LoRA tags."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath


MAX_INLINE_LORAS = 16
MAX_CANDIDATES = 50_000
MAX_HEADER_BYTES = 8 * 1024 * 1024
MAX_NAME_LENGTH = 500
# Forge removes every `<type:arguments>` extra-network tag and nothing else:
# the separators around a tag stay in the prompt that reaches CLIP.
_TAG = re.compile(r"<(\w+):([^>]+)>")
_HEADER_CACHE: dict[tuple, str | None] = {}
_DIGEST_CACHE: dict[tuple, str] = {}


class InlineLoraError(ValueError):
    pass


@dataclass(frozen=True)
class LoraCandidate:
    path: Path
    reference: str
    relative: str
    stem: str
    output_name: str | None


@dataclass(frozen=True)
class InlineLoraMatch:
    path: Path
    reference: str
    weight: float
    source: str
    match: str
    order: int
    identical_copies: int = 0

    def audit(self):
        audit = {
            "id": self.reference,
            "weight": self.weight,
            "source": self.source,
            "match": self.match,
            "order": self.order,
        }
        if self.identical_copies:
            audit["identical_copies"] = self.identical_copies
        return audit


@dataclass(frozen=True)
class InlineLoraResult:
    prompt: str
    negative_prompt: str
    matches: tuple[InlineLoraMatch, ...]


def _fold(value: str) -> str:
    return unicodedata.normalize("NFC", value).replace("\\", "/").casefold()


def _safe_identifier(value: str) -> str:
    value = unicodedata.normalize("NFC", value.strip())
    if not value or len(value) > MAX_NAME_LENGTH or "\x00" in value:
        raise InlineLoraError("inline LoRA name is empty or too long")
    windows = PureWindowsPath(value)
    posix = PurePosixPath(value.replace("\\", "/"))
    if windows.is_absolute() or posix.is_absolute() or any(part in {"", ".", ".."} for part in posix.parts):
        raise InlineLoraError("inline LoRA name must be a safe relative identifier")
    return value


def has_extra_network_tag(text: str) -> bool:
    return isinstance(text, str) and (_TAG.search(text) is not None or re.search(r"<\s*lora\s*:", text, re.IGNORECASE) is not None)


def _tag_weight(value: str, label: str) -> float:
    try:
        weight = float(value)
    except ValueError as error:
        raise InlineLoraError(f"inline LoRA {label} is not a number") from error
    if not math.isfinite(weight) or not -5 <= weight <= 5:
        raise InlineLoraError("inline LoRA weight must be finite and between -5 and 5")
    return weight


def parse_inline_tags(text: str, source: str):
    """Strip ``<lora:name[:te[:unet]]>`` tags the way Forge does.

    Arguments split on ``:``; ``te=``/``unet=`` may also be named. XirAI mounts
    one weight per adapter, so a tag whose text-encoder and UNet weights differ
    is refused, as are ``dyn`` ranks and extra-network types other than LoRA.
    """

    if not isinstance(text, str):
        raise InlineLoraError("prompt must be a string")
    tags = []

    def replace(match):
        kind, arguments = match.group(1), match.group(2)
        if kind != "lora":
            raise InlineLoraError(f"unsupported extra network tag <{kind[:40]}:…>; only <lora:…> is supported")
        items = arguments.split(":")
        positional = [item for item in items if "=" not in item]
        named = dict(item.split("=", 1) for item in items if "=" in item)
        if not positional:
            raise InlineLoraError("inline LoRA name is empty or too long")
        name = _safe_identifier(positional[0])
        if len(positional) > 3 or "dyn" in named or set(named) - {"te", "unet", "dyn"}:
            raise InlineLoraError("inline LoRA tags support a name and one weight; dyn ranks and extra arguments are unavailable")
        te = _tag_weight(positional[1], "weight") if len(positional) > 1 else 1.0
        te = _tag_weight(named["te"], "te weight") if "te" in named else te
        unet = _tag_weight(positional[2], "unet weight") if len(positional) > 2 else te
        unet = _tag_weight(named["unet"], "unet weight") if "unet" in named else unet
        if te != unet:
            raise InlineLoraError("separate text-encoder and UNet LoRA weights are not supported")
        tags.append((name, float(te), source))
        return ""

    stripped = _TAG.sub(replace, text)
    if re.search(r"<\s*lora\s*:", stripped, re.IGNORECASE):
        raise InlineLoraError("inline LoRA tag is malformed or unclosed")
    return stripped, tags


def _revision(path: Path):
    stat = path.stat()
    return (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def safetensors_output_name(path: Path) -> str | None:
    if path.suffix.casefold() != ".safetensors":
        return None
    revision = _revision(path)
    if revision in _HEADER_CACHE:
        return _HEADER_CACHE[revision]
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise InlineLoraError(f"LoRA header is truncated: {path.name}")
            header_length = struct.unpack("<Q", prefix)[0]
            if header_length <= 1 or header_length > MAX_HEADER_BYTES or header_length > path.stat().st_size - 8:
                raise InlineLoraError(f"LoRA header exceeds the bounded metadata limit: {path.name}")
            header = json.loads(handle.read(header_length))
    except InlineLoraError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, struct.error) as error:
        raise InlineLoraError(f"LoRA metadata is unavailable: {path.name}") from error
    metadata = header.get("__metadata__", {}) if isinstance(header, dict) else {}
    value = metadata.get("ss_output_name") if isinstance(metadata, dict) else None
    output = unicodedata.normalize("NFC", value.strip())[:MAX_NAME_LENGTH] if isinstance(value, str) and value.strip() else None
    _HEADER_CACHE[revision] = output
    # Remove stale revisions of this path without allowing an unbounded cache.
    for key in list(_HEADER_CACHE):
        if key[0] == str(path) and key != revision:
            _HEADER_CACHE.pop(key, None)
    if len(_HEADER_CACHE) > MAX_CANDIDATES:
        _HEADER_CACHE.clear()
        _HEADER_CACHE[revision] = output
    return output


def _contained_files(root: Path, extensions):
    try:
        real_root = root.resolve(strict=True)
    except OSError:
        return
    if not real_root.is_dir():
        return
    for path in sorted(real_root.rglob("*"), key=lambda item: item.as_posix().casefold()):
        if path.suffix.casefold() not in extensions:
            continue
        try:
            real = path.resolve(strict=True)
            relative = real.relative_to(real_root)
        except (OSError, ValueError):
            continue
        if real.is_file():
            yield real_root, real, relative.as_posix()


def discover_candidates(local_root, shared_roots, engine, extensions):
    candidates = []
    seen = set()
    roots = [(Path(local_root), None, Path(local_root))]
    for root in shared_roots or ():
        if not root.get("enabled", True) or root.get("kind") not in {"auto", "loras"} or engine not in root.get("engines", (engine,)):
            continue
        root_path = Path(root["path"])
        search_roots = [root_path]
        if root.get("kind") == "auto":
            search_roots = [child for child in root_path.iterdir() if child.is_dir() and child.name.casefold() in {"lora", "loras", "locon", "lycoris", "lokr"}]
            if not search_roots and root_path.name.casefold() in {"lora", "loras", "locon", "lycoris", "lokr"}:
                search_roots = [root_path]
        for search in search_roots:
            roots.append((search, root["id"], root_path))
    for search_root, shared_id, shared_root in roots:
        for _real_root, path, local_relative in _contained_files(search_root, extensions):
            identity = os.path.normcase(str(path))
            if identity in seen:
                continue
            seen.add(identity)
            if shared_id:
                relative_to_shared = path.relative_to(shared_root.resolve(strict=True)).as_posix()
                reference = f"shared:{shared_id}/{relative_to_shared}"
                relative = relative_to_shared
            else:
                reference = local_relative
                relative = local_relative
            candidates.append(LoraCandidate(path, reference, relative, Path(relative).stem, None))
            if len(candidates) > MAX_CANDIDATES:
                raise InlineLoraError("trusted LoRA roots contain too many files for inline resolution")
    return tuple(candidates)


def file_digest(path: Path) -> str:
    """SHA-256 of a candidate file, cached per revision like the header reader."""

    revision = _revision(path)
    cached = _DIGEST_CACHE.get(revision)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise InlineLoraError(f"LoRA file is unreadable: {path.name}") from error
    value = digest.hexdigest()
    for key in [key for key in _DIGEST_CACHE if key[0] == str(path)]:
        _DIGEST_CACHE.pop(key, None)
    if len(_DIGEST_CACHE) > MAX_CANDIDATES:
        _DIGEST_CACHE.clear()
    _DIGEST_CACHE[revision] = value
    return value


def _one_file(name: str, matches, label: str):
    """Collapse byte-identical copies to the first one; any real difference stays ambiguous.

    Style packs are often copied into several folders. Identical bytes mount
    identically, so they are one LoRA; different bytes under one name are a
    choice only the user can make.
    """

    if len(matches) == 1:
        return matches[0], matches
    try:
        sizes = {candidate.path.stat().st_size for candidate in matches}
    except OSError as error:
        raise InlineLoraError(f"inline LoRA {label} is ambiguous: {name[:120]}") from error
    if len(sizes) != 1 or len({file_digest(candidate.path) for candidate in matches}) != 1:
        raise InlineLoraError(f"inline LoRA {label} is ambiguous: {name[:120]}")
    return matches[0], matches


def _resolve(name: str, candidates):
    folded = _fold(name)
    suffix = Path(name).suffix.casefold()
    exact = []
    for candidate in candidates:
        choices = [candidate.reference, candidate.relative]
        if not suffix:
            choices.extend(str(Path(choice).with_suffix("")) for choice in tuple(choices))
        if folded in {_fold(choice) for choice in choices}:
            exact.append(candidate)
    if exact:
        return (*_one_file(name, exact, "identifier"), "exact")
    stems = [candidate for candidate in candidates if _fold(candidate.stem) == folded]
    if stems:
        return (*_one_file(name, stems, "stem"), "stem")
    aliases = []
    for candidate in candidates:
        output_name = candidate.output_name
        if output_name is None and candidate.path.suffix.casefold() == ".safetensors":
            output_name = safetensors_output_name(candidate.path)
        if output_name and _fold(output_name) == folded:
            aliases.append(candidate)
    if aliases:
        return (*_one_file(name, aliases, "metadata alias"), "ss_output_name")
    raise InlineLoraError(f"inline LoRA was not found: {name[:120]}")


def resolve_inline_loras(prompt, negative_prompt, manual_paths, candidates):
    stripped_prompt, positive = parse_inline_tags(prompt, "positive")
    stripped_negative, negative = parse_inline_tags(negative_prompt, "negative")
    tags = positive + negative
    if len(manual_paths) + len(tags) > MAX_INLINE_LORAS:
        raise InlineLoraError(f"at most {MAX_INLINE_LORAS} manual and inline LoRAs may be used")
    manual_identities = {os.path.normcase(str(Path(path).resolve())) for path in manual_paths}
    matches = []
    for order, (name, weight, source) in enumerate(tags, start=len(manual_paths)):
        candidate, copies, match = _resolve(name, candidates)
        # A manual mount of any identical copy is the same adapter mounted twice.
        if any(os.path.normcase(str(copy.path.resolve())) in manual_identities for copy in copies):
            raise InlineLoraError(f"inline LoRA conflicts with an explicitly selected LoRA: {candidate.reference[:160]}")
        matches.append(InlineLoraMatch(
            candidate.path, candidate.reference, weight, source, match, order, len(copies) - 1
        ))
    return InlineLoraResult(stripped_prompt, stripped_negative, tuple(matches))


__all__ = [
    "InlineLoraError", "InlineLoraMatch", "InlineLoraResult", "LoraCandidate", "discover_candidates",
    "has_extra_network_tag", "parse_inline_tags", "resolve_inline_loras", "safetensors_output_name",
]
