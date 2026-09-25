"""Where a finished picture is written, and how it gets there without overwriting anything.

Two naming schemes share this module. The historical one is a date folder holding
`XirAI-<date>-<hour>-<minute>.png`, with `-02`, `-03`, ... appended when a name is taken; every
record that says nothing about naming keeps it byte-for-byte. The other is a user template — a
relative path such as `Test/image` whose last segment is a file stem — which writes
`Test/image_0001.png`, `Test/image_0002.png`, ... beneath the configured output root.

A template is data from a browser, so it is treated as untrusted twice. When a job is submitted it
is parsed and every segment is checked against a fixed rule set (placeholder allowlist, Unicode
normalisation, no empty/`.`/`..` segments, no drive letters or absolute paths, no Windows reserved
names, no trailing dots or spaces, bounded lengths). When a picture is saved the expanded path is
walked one directory at a time: an existing link or junction anywhere on the way is refused, and
the real path of every directory has to stay inside the real output root.

Publishing is exclusive. The picture is encoded into a hidden temporary file beside its final name
and then hard-linked into place, which fails rather than replaces when the name exists — so two
workers, even in two processes, can race for `image_0003.png` and the loser simply takes
`image_0004.png`. A volume that cannot hard-link falls back to reserving the name with an exclusive
create and replacing that reservation. Either way an existing picture is never overwritten, and a
failed or cancelled save leaves neither the temporary file nor a reservation behind.
"""

from __future__ import annotations

import errno
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DEFAULT_OUTPUT_TEMPLATE = "{date}/XirAI-{date}-{hour}-{minute}"
OUTPUT_TEMPLATE_MAX_LENGTH = 200
OUTPUT_TEMPLATE_MAX_SEGMENTS = 8
OUTPUT_SEGMENT_MAX_CHARACTERS = 96
OUTPUT_SEGMENT_MAX_BYTES = 240
OUTPUT_RELATIVE_MAX_CHARACTERS = 220
OUTPUT_COUNTER_DIGITS = 4
# Each placeholder names one unambiguous value. Month and minute are spelled out on purpose: a
# strftime-like `MM`/`mm` pair is exactly the confusion a naming field should not invite.
OUTPUT_TEMPLATE_PLACEHOLDERS = {
    "date": "日期 YYYY-MM-DD",
    "year": "年 YYYY",
    "month": "月 MM",
    "day": "日 DD",
    "hour": "时 HH（24 小时）",
    "minute": "分 mm",
    "second": "秒 ss",
    "engine": "引擎（SD、iL、Anima、Flux、Flux2、Krea2）",
}
_PLACEHOLDER = re.compile(r"\{([^{}]*)\}")
_FORBIDDEN_CHARACTERS = frozenset('<>:"/\\|?*')
_WINDOWS_RESERVED = frozenset(
    ["CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"]
    + [f"COM{index}" for index in range(1, 10)]
    + [f"LPT{index}" for index in range(1, 10)]
    + ["COM¹", "COM²", "COM³", "LPT¹", "LPT²", "LPT³"]
)
_MAX_PUBLISH_ATTEMPTS = 100_000


class OutputPathError(ValueError):
    """A template, or the directory it resolves to, cannot be written safely."""


@dataclass(frozen=True)
class OutputTarget:
    directory: Path
    relative_directory: str
    stem: str
    legacy: bool


def normalize_output_template(value) -> str | None:
    """The template a job carries: `None` for the historical scheme, otherwise the checked text.

    The default template is the historical scheme spelled out, so it normalises to `None` as well;
    a record carrying either one names its files exactly as before, collision suffixes included.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise OutputPathError("输出路径模板必须是文本")
    text = unicodedata.normalize("NFC", value).strip()
    if not text or text == DEFAULT_OUTPUT_TEMPLATE:
        return None
    if len(text) > OUTPUT_TEMPLATE_MAX_LENGTH:
        raise OutputPathError(f"输出路径模板不能超过 {OUTPUT_TEMPLATE_MAX_LENGTH} 个字符")
    if text.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", text):
        raise OutputPathError("输出路径模板必须是输出目录下的相对路径，不能是绝对路径或盘符")
    text = text.replace("\\", "/")
    if text.lower().endswith(".png"):
        text = text[:-4]
    segments = text.split("/")
    if len(segments) > OUTPUT_TEMPLATE_MAX_SEGMENTS:
        raise OutputPathError(f"输出路径模板最多 {OUTPUT_TEMPLATE_MAX_SEGMENTS} 级")
    for segment in segments:
        _check_placeholders(segment)
    # A sample expansion catches every rule a placeholder cannot change — an empty segment, a
    # reserved name spelled literally, a trailing dot — at submission rather than at save time.
    expand_output_template(text, datetime(2026, 1, 2, 3, 4, 5), "iL")
    return text


def _check_placeholders(segment: str):
    remainder = _PLACEHOLDER.sub("", segment)
    if "{" in remainder or "}" in remainder:
        raise OutputPathError("输出路径模板的花括号不成对；占位符写法为 {date}")
    for name in _PLACEHOLDER.findall(segment):
        if name not in OUTPUT_TEMPLATE_PLACEHOLDERS:
            allowed = "、".join(f"{{{key}}}" for key in OUTPUT_TEMPLATE_PLACEHOLDERS)
            raise OutputPathError(f"未知占位符 {{{name}}}；可用：{allowed}")


def _placeholder_values(moment: datetime, engine: str) -> dict[str, str]:
    return {
        "date": moment.strftime("%Y-%m-%d"),
        "year": moment.strftime("%Y"),
        "month": moment.strftime("%m"),
        "day": moment.strftime("%d"),
        "hour": moment.strftime("%H"),
        "minute": moment.strftime("%M"),
        "second": moment.strftime("%S"),
        "engine": engine or "",
    }


def check_output_segment(segment: str) -> str:
    """One expanded directory or stem segment, or an `OutputPathError` saying what is wrong."""
    value = unicodedata.normalize("NFC", segment)
    if not value:
        raise OutputPathError("输出路径不能包含空的目录名")
    if value in {".", ".."}:
        raise OutputPathError("输出路径不能包含 . 或 ..")
    if value.startswith("."):
        raise OutputPathError(f"目录或文件名不能以点开头：{value}")
    if value != value.strip():
        raise OutputPathError(f"目录或文件名首尾不能有空格：{value}")
    if value.endswith("."):
        raise OutputPathError(f"目录或文件名不能以点结尾：{value}")
    if any(character in _FORBIDDEN_CHARACTERS or ord(character) < 32 or ord(character) == 127 for character in value):
        raise OutputPathError(f"目录或文件名包含不允许的字符（< > : \" / \\ | ? * 或控制字符）：{value}")
    if value.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
        raise OutputPathError(f"{value} 是 Windows 保留名称，不能用作目录或文件名")
    if len(value) > OUTPUT_SEGMENT_MAX_CHARACTERS or len(value.encode("utf-8")) > OUTPUT_SEGMENT_MAX_BYTES:
        raise OutputPathError(f"目录或文件名过长（最多 {OUTPUT_SEGMENT_MAX_CHARACTERS} 个字符）：{value[:24]}…")
    return value


def expand_output_template(template: str, moment: datetime, engine: str) -> list[str]:
    values = _placeholder_values(moment, engine)
    segments = [check_output_segment(_PLACEHOLDER.sub(lambda match: values[match.group(1)], raw)) for raw in template.split("/")]
    if len("/".join(segments)) > OUTPUT_RELATIVE_MAX_CHARACTERS:
        raise OutputPathError(f"展开后的输出路径不能超过 {OUTPUT_RELATIVE_MAX_CHARACTERS} 个字符")
    return segments


def _is_link(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(os.path, "isjunction", None)
        return bool(is_junction and is_junction(path))
    except OSError:
        return True


def _inside(root: Path, candidate: Path) -> bool:
    return candidate == root or root in candidate.parents


def output_target(root: Path, template: str | None, *, moment: datetime, engine: str, create: bool) -> OutputTarget:
    """The directory and stem one picture is written under.

    With `create` false nothing is made: every directory that already exists is still checked,
    so a template whose folder is a link, a file or a path out of the root fails at submission.
    """
    real_root = Path(os.path.realpath(root))
    if template is None:
        segments = [moment.strftime("%Y-%m-%d")]
        stem = f"XirAI-{moment:%Y-%m-%d-%H-%M}"
    else:
        expanded = expand_output_template(template, moment, engine)
        segments, stem = expanded[:-1], expanded[-1]
    current = real_root
    for segment in segments:
        candidate = current / segment
        if candidate.exists() or candidate.is_symlink():
            if _is_link(candidate):
                raise OutputPathError(f"输出目录 {segment} 是符号链接或目录联接，已拒绝写入")
            if not candidate.is_dir():
                raise OutputPathError(f"输出路径 {segment} 已存在且不是文件夹")
        elif create:
            try:
                candidate.mkdir()
            except FileExistsError:
                if _is_link(candidate) or not candidate.is_dir():
                    raise OutputPathError(f"输出目录 {segment} 在创建时被替换，已拒绝写入") from None
        resolved = Path(os.path.realpath(candidate)) if candidate.exists() else candidate
        if not _inside(real_root, resolved):
            raise OutputPathError("输出路径解析到了输出目录之外，已拒绝写入")
        current = candidate
    return OutputTarget(current, "/".join(segments), stem, template is None)


def _legacy_names(stem: str):
    yield f"{stem}.png"
    for index in range(2, _MAX_PUBLISH_ATTEMPTS):
        yield f"{stem}-{index:02d}.png"


def next_counter(directory: Path, stem: str) -> int:
    """One past the highest `<stem>_NNNN.png` already present, so a counter never goes backwards."""
    flags = re.IGNORECASE if os.name == "nt" else 0
    pattern = re.compile(rf"^{re.escape(stem)}_(\d{{{OUTPUT_COUNTER_DIGITS},}})\.png$", flags)
    highest = 0
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                match = pattern.match(entry.name)
                if match:
                    highest = max(highest, int(match.group(1)))
    except FileNotFoundError:
        return 1
    return highest + 1


def _counter_names(directory: Path, stem: str):
    start = next_counter(directory, stem)
    for index in range(start, start + _MAX_PUBLISH_ATTEMPTS):
        yield f"{stem}_{index:0{OUTPUT_COUNTER_DIGITS}d}.png"


def candidate_names(target: OutputTarget):
    return _legacy_names(target.stem) if target.legacy else _counter_names(target.directory, target.stem)


def preview_output_name(target: OutputTarget) -> str:
    """The name the next picture would take, without reserving it."""
    for name in candidate_names(target):
        if not (target.directory / name).exists():
            return name
    raise OutputPathError("输出目录中的同名文件过多")


def temporary_output_path(target: OutputTarget) -> Path:
    return target.directory / f".{target.stem}-{os.urandom(8).hex()}.tmp"


def _link_unsupported(error: OSError) -> bool:
    if isinstance(error, FileExistsError):
        return False
    winerror = getattr(error, "winerror", None)
    return error.errno in {errno.EPERM, errno.EXDEV, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EMLINK, errno.EACCES} or winerror in {1, 50, 1142}


def publish_exclusive(temporary: Path, target: OutputTarget) -> Path:
    """Move a finished temporary file to the first free name, never replacing an existing file."""
    use_links = True
    for name in candidate_names(target):
        destination = target.directory / name
        if use_links:
            try:
                os.link(temporary, destination)
            except FileExistsError:
                continue
            except OSError as error:
                if not _link_unsupported(error):
                    raise
                use_links = False
            else:
                temporary.unlink(missing_ok=True)
                return destination
        # No hard links on this volume: reserve the name exclusively, then replace the reservation.
        try:
            descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        os.close(descriptor)
        try:
            os.replace(temporary, destination)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
        return destination
    raise OutputPathError("输出目录中的同名文件过多")


def output_root_display(root: Path, project_root: Path) -> str:
    """How the settings page names the active output root: project-relative when it can be."""
    real_root = Path(os.path.realpath(root))
    real_project = Path(os.path.realpath(project_root))
    if _inside(real_project, real_root):
        relative = real_root.relative_to(real_project).as_posix()
        return relative or "."
    return str(real_root)


__all__ = [
    "DEFAULT_OUTPUT_TEMPLATE", "OUTPUT_TEMPLATE_MAX_LENGTH", "OUTPUT_TEMPLATE_PLACEHOLDERS", "OutputPathError",
    "OutputTarget", "check_output_segment", "expand_output_template", "next_counter", "normalize_output_template",
    "output_root_display", "output_target", "preview_output_name", "publish_exclusive", "temporary_output_path",
]
