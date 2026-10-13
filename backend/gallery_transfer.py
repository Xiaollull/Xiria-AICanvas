"""Gallery collections as a portable package.

Export copies the chosen collections into a new folder under the output root, beside a
`gallery-info.json` that records collections, cards (title, settings, order) and which file each
image is. That folder is the package; a ZIP of it is built on demand for a browser download.

Import reads such a ZIP (`gallery-info.json` at its root, or inside its single top-level folder),
writes the images into a folder under the output root, and creates the collections from those
files. Nothing a package names is trusted as a path: every file written gets a name made here, and
every image is decoded before it is kept.
"""
import json
import os
import re
import shutil
import uuid
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from PIL import Image

from .gallery import MAX_UPLOAD_BYTES, GalleryNotFoundError, _collection_id, _description, _safe_name, _settings_json, _title
from .output_paths import OutputPathError, _inside, _is_link, check_output_segment

PACKAGE_FORMAT = "xiria-gallery"
PACKAGE_VERSION = 1
INFO_NAME = "gallery-info.json"
EXPORT_PREFIX = "XiriusA-Gallery-"
EXPORT_NAME = re.compile(r"XiriusA-Gallery-\d{8}-\d{6}(?:-\d{1,4})?")
MAX_PACKAGE_BYTES = 4 * 1024 ** 3
MAX_INFO_BYTES = 16 * 1024 * 1024
MAX_PACKAGE_IMAGES = 20_000
MAX_PACKAGE_IMAGE_BYTES = 16 * 1024 ** 3
MAX_FOLDER_SEGMENTS = 8
MIME_SUFFIX = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif"}
PIL_MIME = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp", "GIF": "image/gif"}
_UNSAFE_CHARACTERS = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')


class GalleryTransferError(ValueError):
    """The request or the package cannot be used (HTTP 422)."""


class GalleryTransferConflict(GalleryTransferError):
    """The destination already holds something (HTTP 409)."""


class GalleryTransferNotFound(GalleryTransferError):
    """A named collection, export or file does not exist (HTTP 404)."""


def _now():
    return datetime.now().astimezone()


def _iso(moment):
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def folder_segment(value, fallback):
    """A readable directory name for a collection id: unsafe characters become `_`."""
    name = _UNSAFE_CHARACTERS.sub("_", str(value)).strip().strip(".").strip() or fallback
    name = name[:80].rstrip(" .") or fallback
    try:
        return check_output_segment(name)
    except OutputPathError:
        return check_output_segment(f"_{name}"[:80].rstrip(" ."))


def _unique_segment(name, used):
    candidate, index = name, 2
    while candidate.casefold() in used:
        candidate = f"{name} ({index})"
        index += 1
    used.add(candidate.casefold())
    return candidate


def _real_root(output_root):
    root = Path(os.path.realpath(output_root))
    if not root.is_dir():
        raise GalleryTransferError("输出目录不存在")
    return root


def _remove_tree(path):
    shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------------------------
# Export


def export_collections(store, output_root, collection_ids, *, moment=None):
    if not isinstance(collection_ids, list) or not collection_ids:
        raise GalleryTransferError("请至少选择一个收藏夹")
    try:
        wanted = [_collection_id(value) for value in collection_ids]
    except ValueError as error:
        raise GalleryTransferError(str(error)) from error
    if len(set(wanted)) != len(wanted):
        raise GalleryTransferError("收藏夹重复选择")
    snapshot = store.list_gallery()
    known = {collection["id"]: collection for collection in snapshot["collections"]}
    missing = [value for value in wanted if value not in known]
    if missing:
        raise GalleryTransferNotFound(f"收藏夹不存在：{'、'.join(missing)}")
    # Gallery order, not click order: the package lists collections the way the page does.
    chosen = [collection for collection in snapshot["collections"] if collection["id"] in set(wanted)]
    cards_by_collection = {collection["id"]: [] for collection in chosen}
    for card in snapshot["cards"]:
        if card["collection_id"] in cards_by_collection:
            cards_by_collection[card["collection_id"]].append(card)
    files = store.image_files([image["id"] for cards in cards_by_collection.values() for card in cards for image in card["images"]])

    root = _real_root(output_root)
    moment = moment or _now()
    staging = root / f".{EXPORT_PREFIX}{uuid.uuid4().hex}.tmp"
    staging.mkdir()
    try:
        used = {INFO_NAME.casefold()}
        skipped = []
        collections = []
        receipt = []
        for collection in chosen:
            folder = _unique_segment(folder_segment(collection["id"], "collection"), used)
            (staging / folder).mkdir()
            cards = []
            image_total = 0
            for card_number, card in enumerate(cards_by_collection[collection["id"]], 1):
                images = []
                for image_number, image in enumerate(card["images"], 1):
                    source = files.get(image["id"])
                    if source is None or not source["path"].is_file() or source["mime_type"] not in MIME_SUFFIX:
                        skipped.append({"collection": collection["id"], "card": card.get("title") or "", "image": image.get("name") or image["id"]})
                        continue
                    relative = f"{folder}/{card_number:03d}-{image_number:02d}{MIME_SUFFIX[source['mime_type']]}"
                    shutil.copyfile(source["path"], staging / relative)
                    images.append({"file": relative, "name": image.get("name"), "mime_type": source["mime_type"]})
                settings = dict(card.get("settings") or {})
                # The revision belongs to this machine's LoRA library; the importer stamps its own.
                settings.pop("loraLibraryRevision", None)
                cards.append({"title": card.get("title"), "settings": settings, "created_at": card.get("created_at"), "images": images})
                image_total += len(images)
            collections.append({"id": collection["id"], "description": collection.get("description"),
                                "created_at": collection.get("created_at"), "folder": folder, "cards": cards})
            receipt.append({"id": collection["id"], "card_count": len(cards), "image_count": image_total})
        info = {"format": PACKAGE_FORMAT, "version": PACKAGE_VERSION, "exported_at": _iso(moment), "collections": collections}
        (staging / INFO_NAME).write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        final = _publish_directory(staging, root, f"{EXPORT_PREFIX}{moment:%Y%m%d-%H%M%S}")
    except BaseException:
        _remove_tree(staging)
        raise
    return {
        "folder": final.name,
        "collections": receipt,
        "card_count": sum(item["card_count"] for item in receipt),
        "image_count": sum(item["image_count"] for item in receipt),
        "skipped": skipped,
    }


def _publish_directory(staging, parent, name):
    for index in range(1, 10_000):
        target = parent / (name if index == 1 else f"{name}-{index}")
        if target.exists() or target.is_symlink():
            continue
        try:
            os.rename(staging, target)
        except FileExistsError:
            continue
        except OSError:
            if target.exists():
                continue
            raise
        return target
    raise GalleryTransferConflict("输出目录中的同名导出文件夹过多")


def export_folder(output_root, name):
    if not isinstance(name, str) or not EXPORT_NAME.fullmatch(name):
        raise GalleryTransferNotFound("导出文件夹不存在")
    root = _real_root(output_root)
    folder = root / name
    if _is_link(folder) or not folder.is_dir():
        raise GalleryTransferNotFound("导出文件夹不存在")
    return folder


def write_export_zip(output_root, name, target):
    """Write the named export folder as a ZIP to `target` (a path or a binary file object)."""
    folder = export_folder(output_root, name)
    info_path = folder / INFO_NAME
    if _is_link(info_path) or not info_path.is_file() or info_path.stat().st_size > MAX_INFO_BYTES:
        raise GalleryTransferNotFound("导出文件夹缺少 gallery-info.json")
    info = _parse_info(info_path.read_bytes())
    real_folder = Path(os.path.realpath(folder))
    with zipfile.ZipFile(target, "w", allowZip64=True) as archive:
        archive.write(info_path, INFO_NAME, compress_type=zipfile.ZIP_DEFLATED)
        for entry in _info_files(info):
            source = folder / entry
            if _is_link(source) or not source.is_file() or not _inside(real_folder, Path(os.path.realpath(source))):
                raise GalleryTransferNotFound(f"导出文件夹缺少图片：{entry}")
            # Images are already compressed; storing them keeps the ZIP fast and the same size.
            archive.write(source, entry, compress_type=zipfile.ZIP_STORED)


# ---------------------------------------------------------------------------------------------
# Package parsing


def _member_name(value):
    if not isinstance(value, str) or not value or len(value) > 512 or "\\" in value or "\x00" in value:
        raise GalleryTransferError("gallery-info.json 中的图片路径无效")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts) or str(path) != value:
        raise GalleryTransferError(f"gallery-info.json 中的图片路径无效：{value[:80]}")
    return value


def _parse_info(raw):
    try:
        info = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GalleryTransferError("gallery-info.json 不是有效的 JSON") from error
    if not isinstance(info, dict) or info.get("format") != PACKAGE_FORMAT:
        raise GalleryTransferError("这不是画廊导出包（gallery-info.json 格式不符）")
    if info.get("version") != PACKAGE_VERSION:
        raise GalleryTransferError(f"不支持的画廊导出包版本：{info.get('version')!r}")
    collections = info.get("collections")
    if not isinstance(collections, list) or not collections:
        raise GalleryTransferError("导出包中没有收藏夹")
    images = 0
    for collection in collections:
        if not isinstance(collection, dict) or not isinstance(collection.get("cards"), list):
            raise GalleryTransferError("gallery-info.json 中的收藏夹格式无效")
        try:
            _collection_id(collection.get("id"))
            _description(collection.get("description"))
            for card in collection["cards"]:
                if not isinstance(card, dict) or not isinstance(card.get("images", []), list):
                    raise GalleryTransferError("gallery-info.json 中的卡片格式无效")
                _title(card.get("title"))
                _settings_json(card.get("settings", {}))
                for image in card.get("images", []):
                    if not isinstance(image, dict):
                        raise GalleryTransferError("gallery-info.json 中的图片格式无效")
                    _member_name(image.get("file"))
                    _safe_name(image.get("name"), "image")
                    images += 1
        except GalleryTransferError:
            raise
        except ValueError as error:
            raise GalleryTransferError(f"gallery-info.json 无效：{error}") from error
    if images > MAX_PACKAGE_IMAGES:
        raise GalleryTransferError(f"导出包图片过多（上限 {MAX_PACKAGE_IMAGES} 张）")
    return info


def _info_files(info):
    return [image["file"] for collection in info["collections"] for card in collection["cards"] for image in card.get("images", [])]


def _member_index(archive):
    """File members by name. A member without the UTF-8 flag was decoded as CP437, so a folder
    re-zipped by a tool writing the local code page also answers to its UTF-8 or GBK spelling.
    macOS resource forks (`__MACOSX/`) are not part of any package."""
    index = {}
    for item in archive.infolist():
        if item.is_dir() or item.filename.startswith("__MACOSX/"):
            continue
        index.setdefault(item.filename, item)
        if not item.flag_bits & 0x800:
            try:
                raw = item.filename.encode("cp437")
            except UnicodeEncodeError:
                continue
            for encoding in ("utf-8", "gbk"):
                try:
                    index.setdefault(raw.decode(encoding), item)
                except UnicodeDecodeError:
                    pass
    return index


def _locate_info(index):
    """The prefixes the package may live under: [""], or every spelling of the one top-level
    folder that holds gallery-info.json. Two such folders are ambiguous and refused."""
    if INFO_NAME in index:
        return [""]
    nested = {name[: -len(INFO_NAME)]: index[name] for name in index if name.endswith(f"/{INFO_NAME}") and name.count("/") == 1}
    if len({id(item) for item in nested.values()}) == 1:
        return list(nested)
    raise GalleryTransferError("ZIP 中找不到 gallery-info.json（应位于根目录或唯一的顶层文件夹内）")


def _read_member(archive, item, limit):
    if item.file_size > limit:
        raise GalleryTransferError(f"导出包中的文件过大：{item.filename[:80]}")
    with archive.open(item) as source:
        data = source.read(limit + 1)
    if len(data) > limit:
        raise GalleryTransferError(f"导出包中的文件过大：{item.filename[:80]}")
    return data


def _copy_member(archive, item, target, limit):
    if item.file_size > limit:
        raise GalleryTransferError(f"图片超过 {limit // (1024 * 1024)} MiB：{item.filename[:80]}")
    written = 0
    with archive.open(item) as source, target.open("xb") as output:
        while chunk := source.read(1024 * 1024):
            written += len(chunk)
            if written > limit:
                raise GalleryTransferError(f"图片超过 {limit // (1024 * 1024)} MiB：{item.filename[:80]}")
            output.write(chunk)
    return written


def detect_image(path):
    """The MIME type of a decodable PNG, JPEG, WebP or GIF file, or a transfer error."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                mime_type = PIL_MIME.get(image.format)
                width, height = image.size
                image.verify()
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
        raise GalleryTransferError(f"导出包中有无法解码的图片：{path.name}") from error
    if mime_type is None or width < 1 or height < 1:
        raise GalleryTransferError(f"导出包中有不支持的图片格式：{path.name}")
    return mime_type


# ---------------------------------------------------------------------------------------------
# Import


def default_import_folder(moment=None):
    return f"{(moment or _now()):%Y%m%d-%H%M%S}"


def import_segments(folder, moment=None):
    """The checked directory segments under the output root an import writes into."""
    if folder is None or not str(folder).strip():
        return [default_import_folder(moment)]
    if not isinstance(folder, str):
        raise GalleryTransferError("目标文件夹必须是文本")
    text = folder.strip().replace("\\", "/").strip("/")
    if re.match(r"^[A-Za-z]:", folder.strip()) or folder.strip().startswith(("/", "\\")):
        raise GalleryTransferError("目标文件夹必须是输出目录下的相对路径，不能是绝对路径或盘符")
    segments = text.split("/")
    if len(segments) > MAX_FOLDER_SEGMENTS:
        raise GalleryTransferError(f"目标文件夹最多 {MAX_FOLDER_SEGMENTS} 级")
    try:
        return [check_output_segment(segment) for segment in segments]
    except OutputPathError as error:
        raise GalleryTransferError(str(error)) from error


def _prepare_parent(root, segments):
    current = root
    for segment in segments:
        candidate = current / segment
        if candidate.exists() or candidate.is_symlink():
            if _is_link(candidate):
                raise GalleryTransferError(f"目标文件夹 {segment} 是符号链接或目录联接，已拒绝写入")
            if not candidate.is_dir():
                raise GalleryTransferConflict(f"目标路径 {segment} 已存在且不是文件夹")
        else:
            candidate.mkdir()
        if not _inside(root, Path(os.path.realpath(candidate))):
            raise GalleryTransferError("目标文件夹解析到了输出目录之外，已拒绝写入")
        current = candidate
    return current


def zip_source_path(value):
    """A local ZIP path as typed or pasted; surrounding quotes from "Copy as path" are dropped."""
    if not isinstance(value, str):
        raise GalleryTransferError("请填写 ZIP 文件路径")
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    if not text:
        raise GalleryTransferError("请填写 ZIP 文件路径")
    if "\x00" in text:
        raise GalleryTransferError("ZIP 文件路径无效")
    path = Path(text)
    if not path.is_absolute():
        raise GalleryTransferError("ZIP 文件路径必须是完整的绝对路径")
    if path.suffix.lower() != ".zip":
        raise GalleryTransferError("路径必须指向 .zip 文件本身")
    if not path.is_file():
        raise GalleryTransferNotFound("找不到这个 ZIP 文件")
    return path


def import_package(store, output_root, zip_path, folder=None, *, moment=None, source_name=None):
    zip_path = Path(zip_path)
    try:
        size = zip_path.stat().st_size
    except OSError as error:
        raise GalleryTransferNotFound("找不到这个 ZIP 文件") from error
    if size > MAX_PACKAGE_BYTES:
        raise GalleryTransferError(f"ZIP 超过 {MAX_PACKAGE_BYTES // 1024 ** 3} GiB 上限")
    segments = import_segments(folder, moment)
    root = _real_root(output_root)
    try:
        archive = zipfile.ZipFile(zip_path)
    except (zipfile.BadZipFile, OSError) as error:
        raise GalleryTransferError("文件不是有效的 ZIP") from error
    with archive:
        index = _member_index(archive)
        prefixes = _locate_info(index)
        info = _parse_info(_read_member(archive, index[f"{prefixes[0]}{INFO_NAME}"], MAX_INFO_BYTES))
        entries = _info_files(info)
        # Image paths in the JSON are UTF-8; use the folder spelling under which they resolve.
        prefix = max(prefixes, key=lambda candidate: sum(f"{candidate}{entry}" in index for entry in entries))
        members = {}
        total = 0
        for entry in entries:
            item = index.get(f"{prefix}{entry}")
            if item is None:
                raise GalleryTransferError(f"ZIP 中缺少图片：{entry[:80]}")
            if item.flag_bits & 0x1:
                raise GalleryTransferError(f"ZIP 中的图片不可读取（目录或已加密）：{entry[:80]}")
            total += item.file_size
            members[entry] = item
        if total > MAX_PACKAGE_IMAGE_BYTES:
            raise GalleryTransferError(f"导出包解压后超过 {MAX_PACKAGE_IMAGE_BYTES // 1024 ** 3} GiB 上限")

        parent = _prepare_parent(root, segments[:-1])
        final = parent / segments[-1]
        if folder is None or not str(folder).strip():
            # A time-named folder is the program's choice, so a second import in the same second takes `-2`.
            base = segments[-1]
            for index in range(2, 10_000):
                if not (final.exists() or final.is_symlink()):
                    break
                segments[-1] = f"{base}-{index}"
                final = parent / segments[-1]
        if final.exists() or final.is_symlink():
            if _is_link(final) or not final.is_dir():
                raise GalleryTransferConflict(f"目标路径 {'/'.join(segments)} 已存在且不是文件夹")
            if any(final.iterdir()):
                raise GalleryTransferConflict(f"目标文件夹 {'/'.join(segments)} 已存在且不为空，请换一个名称")
        staging = parent / f".gallery-import-{uuid.uuid4().hex}.tmp"
        staging.mkdir()
        published = None
        try:
            used = {INFO_NAME.casefold()}
            plan = []
            rewritten = []
            for collection in info["collections"]:
                directory = _unique_segment(folder_segment(collection["id"], "collection"), used)
                (staging / directory).mkdir()
                cards = []
                rewritten_cards = []
                for card_number, card in enumerate(collection["cards"], 1):
                    images = []
                    rewritten_images = []
                    for image_number, image in enumerate(card.get("images", []), 1):
                        temporary = staging / directory / f".{uuid.uuid4().hex}.part"
                        _copy_member(archive, members[image["file"]], temporary, MAX_UPLOAD_BYTES)
                        try:
                            mime_type = detect_image(temporary)
                        except BaseException:
                            temporary.unlink(missing_ok=True)
                            raise
                        relative = f"{directory}/{card_number:03d}-{image_number:02d}{MIME_SUFFIX[mime_type]}"
                        os.replace(temporary, staging / relative)
                        name = _safe_name(image.get("name"), PurePosixPath(image["file"]).name)
                        images.append({"relative": relative, "name": name, "mime_type": mime_type})
                        rewritten_images.append({"file": relative, "name": name, "mime_type": mime_type})
                    settings = dict(card.get("settings") or {})
                    settings.pop("loraLibraryRevision", None)
                    cards.append({"title": card.get("title"), "settings": settings, "images": images})
                    rewritten_cards.append({"title": card.get("title"), "settings": settings,
                                            "created_at": card.get("created_at"), "images": rewritten_images})
                plan.append({"id": collection["id"], "description": collection.get("description"), "cards": cards})
                rewritten.append({"id": collection["id"], "description": collection.get("description"),
                                  "created_at": collection.get("created_at"), "folder": directory, "cards": rewritten_cards})
            record = {"format": PACKAGE_FORMAT, "version": PACKAGE_VERSION, "exported_at": info.get("exported_at"),
                      "imported_at": _iso(moment or _now()), "source": source_name or zip_path.name, "collections": rewritten}
            (staging / INFO_NAME).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            if final.exists():
                final.rmdir()  # Checked empty above; a directory rename cannot replace it on Windows.
            os.rename(staging, final)
            published = final
            for collection in plan:
                for card in collection["cards"]:
                    for image in card["images"]:
                        image["path"] = final / image.pop("relative")
            created = store.import_collections(plan)
        except BaseException:
            _remove_tree(published or staging)
            raise
    for entry, collection in zip(created, rewritten):
        entry["folder"] = collection["folder"]
    return {
        "folder": "/".join(segments),
        "collections": created,
        "card_count": sum(item["card_count"] for item in created),
        "image_count": sum(item["image_count"] for item in created),
    }


__all__ = [
    "EXPORT_NAME", "GalleryTransferConflict", "GalleryTransferError", "GalleryTransferNotFound", "INFO_NAME",
    "MAX_PACKAGE_BYTES", "PACKAGE_FORMAT", "PACKAGE_VERSION", "default_import_folder", "export_collections",
    "export_folder", "import_package", "import_segments", "write_export_zip", "zip_source_path",
]
