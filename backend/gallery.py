import base64
import binascii
import io
import json
import os
import shutil
import sqlite3
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


MAX_UPLOAD_BYTES = 50 * 1024 * 1024
GALLERY_IMAGE_URL = "/api/inference/gallery/images/{image_id}"
GALLERY_THUMBNAIL_URL = "/api/inference/gallery/images/{image_id}?variant=thumb"
UNSET = object()

# Curated images are independent copies at generation size: a 2048x2944 PNG is
# six megapixels and five megabytes. Grid tiles render them at roughly 300 px and
# the focus strip at 60 px, so every browse decoded tens of megapixels to paint
# thumbnails. The derivative below is generated once, on first request, and cached
# beside the original; it is never part of the stored record, so no schema change
# and no migration is involved and deleting it only costs one regeneration.
THUMBNAIL_MAX_EDGE = 640
THUMBNAIL_SUFFIX = ".thumb.webp"
THUMBNAIL_MIME = "image/webp"
THUMBNAIL_QUALITY = 82
# Below this an original is already cheap to send and decode, so deriving from it
# would spend CPU and disk to save nothing — and for small synthetic images a
# lossy derivative can genuinely come out larger than the source.
THUMBNAIL_MIN_ORIGINAL_BYTES = 256 * 1024


class GalleryError(Exception):
    pass


class GalleryValidationError(GalleryError, ValueError):
    pass


class GalleryConflictError(GalleryError, ValueError):
    pass


class GalleryNotFoundError(GalleryError, LookupError):
    pass


class GalleryStorageError(GalleryError):
    pass


@dataclass
class _StagedImage:
    image_id: str
    temporary_path: Path
    final_path: Path
    storage_name: str
    name: str
    mime_type: str


def _timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _collection_id(value):
    if not isinstance(value, str):
        raise GalleryValidationError("Collection id must be a string")
    cleaned = value.strip()
    if not 1 <= len(cleaned) <= 64:
        raise GalleryValidationError("Collection id must contain 1 to 64 characters")
    if cleaned in {".", ".."} or "/" in cleaned or "\\" in cleaned:
        raise GalleryValidationError("Collection id must not contain path separators or dot-path values")
    return cleaned


def _description(value):
    if value is not None and not isinstance(value, str):
        raise GalleryValidationError("Collection description must be a string or null")
    if value is not None and len(value) > 1000:
        raise GalleryValidationError("Collection description must not exceed 1000 characters")
    return value


def _title(value):
    if value is not None and not isinstance(value, str):
        raise GalleryValidationError("Card title must be a string or null")
    if value is not None and len(value) > 160:
        raise GalleryValidationError("Card title must not exceed 160 characters")
    return value


def _prompt_title(value):
    if not isinstance(value, str):
        raise GalleryValidationError("Prompt title must be a string")
    cleaned = value.strip()
    if not 1 <= len(cleaned) <= 160:
        raise GalleryValidationError("Prompt title must contain 1 to 160 characters")
    return cleaned


def _prompt_text(value, field):
    if not isinstance(value, str):
        raise GalleryValidationError(f"{field} must be a string")
    if len(value) > 8000:
        raise GalleryValidationError(f"{field} must not exceed 8000 characters")
    return value


def _prompt_notes(value):
    if value is not None and not isinstance(value, str):
        raise GalleryValidationError("Prompt notes must be a string or null")
    if value is not None and len(value) > 2000:
        raise GalleryValidationError("Prompt notes must not exceed 2000 characters")
    return value


def _settings_json(value):
    if not isinstance(value, dict):
        raise GalleryValidationError("Card settings must be a JSON object")
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise GalleryValidationError("Card settings must contain valid JSON values") from error


def _safe_name(value, fallback):
    if value is None:
        return fallback
    if not isinstance(value, str):
        raise GalleryValidationError("Image name must be a string")
    name = value.replace("\\", "/").rsplit("/", 1)[-1].strip()
    name = "".join(character for character in name if ord(character) >= 32 and ord(character) != 127)
    if not name:
        return fallback
    if len(name) > 255:
        raise GalleryValidationError("Image name must not exceed 255 characters")
    return name


class GalleryStore:
    def __init__(self, state_directory, asset_resolver=None):
        self.state_directory = Path(state_directory).resolve()
        self.database_path = self.state_directory / "gallery.sqlite3"
        self.assets_directory = self.state_directory / "gallery-assets"
        self.asset_resolver = asset_resolver
        self.state_directory.mkdir(parents=True, exist_ok=True)
        self.assets_directory.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize_schema(self):
        connection = self._connect()
        try:
            self._begin(connection)
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version > 3:
                raise RuntimeError(f"Gallery database version {version} is newer than supported version 3")
            if version == 0:
                statements = (
                    """CREATE TABLE collections (
                        id TEXT PRIMARY KEY,
                        description TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        CHECK(length(id) BETWEEN 1 AND 64),
                        CHECK(description IS NULL OR length(description) <= 1000)
                    )""",
                    """CREATE TABLE cards (
                        id TEXT PRIMARY KEY,
                        collection_id TEXT NOT NULL,
                        title TEXT,
                        settings_json TEXT NOT NULL,
                        sort_index INTEGER NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        FOREIGN KEY(collection_id) REFERENCES collections(id)
                            ON UPDATE CASCADE ON DELETE CASCADE,
                        CHECK(title IS NULL OR length(title) <= 160),
                        CHECK(sort_index >= 0)
                    )""",
                    """CREATE TABLE card_images (
                        id TEXT PRIMARY KEY,
                        card_id TEXT NOT NULL,
                        sort_index INTEGER NOT NULL,
                        original_name TEXT NOT NULL,
                        mime_type TEXT NOT NULL,
                        storage_name TEXT NOT NULL UNIQUE,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(card_id) REFERENCES cards(id) ON DELETE CASCADE,
                        CHECK(sort_index >= 0),
                        CHECK(mime_type IN ('image/png', 'image/jpeg', 'image/webp', 'image/gif'))
                    )""",
                    "CREATE INDEX cards_collection_created_idx ON cards(collection_id, created_at DESC)",
                    "CREATE INDEX cards_collection_sort_idx ON cards(collection_id, sort_index)",
                    "CREATE INDEX card_images_card_order_idx ON card_images(card_id, sort_index)",
                    """CREATE TABLE prompt_entries (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        positive_prompt TEXT NOT NULL,
                        negative_prompt TEXT NOT NULL,
                        notes TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        CHECK(length(title) BETWEEN 1 AND 160),
                        CHECK(length(positive_prompt) <= 8000),
                        CHECK(length(negative_prompt) <= 8000),
                        CHECK(notes IS NULL OR length(notes) <= 2000)
                    )""",
                    "CREATE INDEX prompt_entries_updated_idx ON prompt_entries(updated_at DESC)",
                )
                for statement in statements:
                    connection.execute(statement)
                connection.execute("PRAGMA user_version = 3")
            else:
                if version == 1:
                    connection.execute(
                        "ALTER TABLE cards ADD COLUMN sort_index INTEGER NOT NULL DEFAULT 0 CHECK(sort_index >= 0)"
                    )
                    collection_rows = connection.execute("SELECT id FROM collections").fetchall()
                    for collection in collection_rows:
                        card_rows = connection.execute(
                            "SELECT id FROM cards WHERE collection_id = ? ORDER BY created_at DESC, id DESC",
                            (collection["id"],),
                        ).fetchall()
                        connection.executemany(
                            "UPDATE cards SET sort_index = ? WHERE id = ?",
                            [(index, row["id"]) for index, row in enumerate(card_rows)],
                        )
                    connection.execute(
                        "CREATE INDEX cards_collection_sort_idx ON cards(collection_id, sort_index)"
                    )
                    version = 2
                if version == 2:
                    connection.execute(
                        """CREATE TABLE prompt_entries (
                            id TEXT PRIMARY KEY,
                            title TEXT NOT NULL,
                            positive_prompt TEXT NOT NULL,
                            negative_prompt TEXT NOT NULL,
                            notes TEXT,
                            created_at TEXT NOT NULL,
                            updated_at TEXT NOT NULL,
                            CHECK(length(title) BETWEEN 1 AND 160),
                            CHECK(length(positive_prompt) <= 8000),
                            CHECK(length(negative_prompt) <= 8000),
                            CHECK(notes IS NULL OR length(notes) <= 2000)
                        )"""
                    )
                    connection.execute("CREATE INDEX prompt_entries_updated_idx ON prompt_entries(updated_at DESC)")
                    connection.execute("PRAGMA user_version = 3")
            connection.commit()
        except Exception:
            self._rollback(connection)
            raise
        finally:
            connection.close()

    @staticmethod
    def _begin(connection):
        connection.execute("BEGIN IMMEDIATE")

    @staticmethod
    def _rollback(connection):
        if connection.in_transaction:
            connection.rollback()

    def _asset_path(self, storage_name):
        if not isinstance(storage_name, str) or not storage_name or Path(storage_name).name != storage_name:
            raise GalleryStorageError("Gallery image has an unsafe storage path")
        root = self.assets_directory.resolve()
        path = (root / storage_name).resolve()
        if path.parent != root:
            raise GalleryStorageError("Gallery image is outside the gallery asset directory")
        return path

    @staticmethod
    def _validate_image_file(path, expected_mime):
        formats = {
            "PNG": "image/png",
            "JPEG": "image/jpeg",
            "WEBP": "image/webp",
            "GIF": "image/gif",
        }
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(path) as image:
                    mime_type = formats.get(image.format)
                    width, height = image.size
                    image.verify()
        except (OSError, SyntaxError, ValueError, Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
            raise GalleryValidationError("Image data is not a valid supported image") from error
        if width < 1 or height < 1 or mime_type != expected_mime:
            raise GalleryValidationError(f"Image data does not match declared type {expected_mime}")

    def _write_temporary(self, content):
        temporary_path = self.assets_directory / f".{uuid.uuid4()}.tmp"
        try:
            with temporary_path.open("xb") as target:
                if callable(content):
                    content(target)
                else:
                    target.write(content)
                target.flush()
                os.fsync(target.fileno())
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        return temporary_path

    def _stage_generated_asset(self, asset_id):
        if not isinstance(asset_id, str) or not asset_id or len(asset_id) > 4096:
            raise GalleryValidationError("asset_id must be a non-empty string")
        if self.asset_resolver is None:
            raise GalleryValidationError("Generated asset imports are not configured")
        try:
            source = Path(self.asset_resolver(asset_id))
        except Exception as error:
            raise GalleryValidationError("asset_id is invalid") from error
        if not source.is_file():
            raise GalleryNotFoundError("Generated image asset was not found")
        suffix = source.suffix.lower()
        generated_types = {".png": "image/png", ".gif": "image/gif"}
        if suffix not in generated_types:
            raise GalleryValidationError("Generated gallery assets must be PNG or GIF files")
        original_name = _safe_name(source.name, f"image{suffix}")

        def copy_source(target):
            with source.open("rb") as source_file:
                shutil.copyfileobj(source_file, target, length=1024 * 1024)

        temporary_path = self._write_temporary(copy_source)
        try:
            self._validate_image_file(temporary_path, generated_types[suffix])
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        image_id = str(uuid.uuid4())
        storage_name = f"{image_id}{suffix}"
        return _StagedImage(
            image_id=image_id,
            temporary_path=temporary_path,
            final_path=self._asset_path(storage_name),
            storage_name=storage_name,
            name=original_name,
            mime_type=generated_types[suffix],
        )

    def _stage_data_url(self, data_url, name=None):
        if not isinstance(data_url, str):
            raise GalleryValidationError("data_url must be a string")
        try:
            header, encoded = data_url.split(",", 1)
        except ValueError as error:
            raise GalleryValidationError("data_url must contain base64 image data") from error
        mime_extensions = {
            "data:image/png;base64": ("image/png", ".png"),
            "data:image/jpeg;base64": ("image/jpeg", ".jpg"),
            "data:image/webp;base64": ("image/webp", ".webp"),
            "data:image/gif;base64": ("image/gif", ".gif"),
        }
        if header not in mime_extensions:
            raise GalleryValidationError("data_url must be a base64 PNG, JPEG, WebP, or GIF")
        if len(encoded) > ((MAX_UPLOAD_BYTES + 2) // 3) * 4:
            raise GalleryValidationError("Uploaded image exceeds the 50 MiB limit")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as error:
            raise GalleryValidationError("data_url contains invalid base64 data") from error
        if len(raw) > MAX_UPLOAD_BYTES:
            raise GalleryValidationError("Uploaded image exceeds the 50 MiB limit")
        mime_type, suffix = mime_extensions[header]
        original_name = _safe_name(name, f"upload{suffix}")
        temporary_path = self._write_temporary(raw)
        try:
            self._validate_image_file(temporary_path, mime_type)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        image_id = str(uuid.uuid4())
        storage_name = f"{image_id}{suffix}"
        return _StagedImage(
            image_id=image_id,
            temporary_path=temporary_path,
            final_path=self._asset_path(storage_name),
            storage_name=storage_name,
            name=original_name,
            mime_type=mime_type,
        )

    @staticmethod
    def _normalize_image_specs(images):
        if not isinstance(images, list):
            raise GalleryValidationError("images must be a list")
        normalized = []
        allowed = {"gallery_image_id", "asset_id", "data_url", "name"}
        for image in images:
            if not isinstance(image, dict):
                raise GalleryValidationError("Each image must be an object")
            if set(image) - allowed:
                raise GalleryValidationError("Image contains unsupported fields")
            sources = [key for key in ("gallery_image_id", "asset_id", "data_url") if image.get(key) is not None]
            if len(sources) != 1:
                raise GalleryValidationError(
                    "Each image must contain exactly one of gallery_image_id, asset_id, or data_url"
                )
            normalized.append(image)
        return normalized

    def _stage_new_image(self, image):
        if image.get("asset_id") is not None:
            return self._stage_generated_asset(image["asset_id"])
        return self._stage_data_url(image["data_url"], image.get("name"))

    @staticmethod
    def _install_staged(staged):
        for image in staged:
            os.replace(image.temporary_path, image.final_path)

    @staticmethod
    def _cleanup_staged(staged):
        for image in staged:
            image.temporary_path.unlink(missing_ok=True)
            image.final_path.unlink(missing_ok=True)
            image.final_path.with_name(f"{image.final_path.stem}{THUMBNAIL_SUFFIX}").unlink(missing_ok=True)

    @staticmethod
    def _remove_paths(paths):
        # Each original takes its cached derivative with it; an orphaned
        # thumbnail would otherwise linger with no record pointing at it.
        expanded = []
        for path in paths:
            expanded.append(path)
            expanded.append(path.with_name(f"{path.stem}{THUMBNAIL_SUFFIX}"))
        for path in expanded:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError as error:
                # The database transaction is already committed. Leave an orphaned
                # private asset for later cleanup instead of reporting a false failure.
                warnings.warn(f"Unable to remove orphaned gallery image {path.name}: {error}", RuntimeWarning)

    def _image_record(self, row):
        return {
            "id": row["id"],
            "name": row["original_name"],
            "mime_type": row["mime_type"],
            "url": GALLERY_IMAGE_URL.format(image_id=row["id"]),
            "thumb_url": GALLERY_THUMBNAIL_URL.format(image_id=row["id"]),
            "sort_index": row["sort_index"],
        }

    def _card_record(self, connection, row):
        image_rows = connection.execute(
            """
            SELECT id, original_name, mime_type, sort_index
            FROM card_images WHERE card_id = ? ORDER BY sort_index, id
            """,
            (row["id"],),
        ).fetchall()
        images = [self._image_record(image) for image in image_rows]
        return {
            "id": row["id"],
            "collection_id": row["collection_id"],
            "title": row["title"],
            "settings": json.loads(row["settings_json"]),
            "sort_index": row["sort_index"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "image_count": len(images),
            "images": images,
        }

    def _fetch_card(self, connection, card_id):
        row = connection.execute("SELECT * FROM cards WHERE id = ?", (card_id,)).fetchone()
        if row is None:
            raise GalleryNotFoundError("Gallery card was not found")
        return self._card_record(connection, row)

    @staticmethod
    def _normalize_card_order(connection, collection_id):
        rows = connection.execute(
            "SELECT id FROM cards WHERE collection_id = ? ORDER BY sort_index, created_at DESC, id DESC",
            (collection_id,),
        ).fetchall()
        connection.executemany(
            "UPDATE cards SET sort_index = ? WHERE id = ?",
            [(index, row["id"]) for index, row in enumerate(rows)],
        )

    @staticmethod
    def _collection_record(row):
        return {
            "id": row["id"],
            "description": row["description"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "card_count": row["card_count"],
            "image_count": row["image_count"],
        }

    def _fetch_collection(self, connection, collection_id):
        row = connection.execute(
            """
            SELECT c.id, c.description, c.created_at, c.updated_at,
                   COUNT(DISTINCT ca.id) AS card_count,
                   COUNT(ci.id) AS image_count
            FROM collections c
            LEFT JOIN cards ca ON ca.collection_id = c.id
            LEFT JOIN card_images ci ON ci.card_id = ca.id
            WHERE c.id = ?
            GROUP BY c.id
            """,
            (collection_id,),
        ).fetchone()
        if row is None:
            raise GalleryNotFoundError("Gallery collection was not found")
        return self._collection_record(row)

    @staticmethod
    def _prompt_record(row):
        return {
            "id": row["id"],
            "title": row["title"],
            "positive_prompt": row["positive_prompt"],
            "negative_prompt": row["negative_prompt"],
            "notes": row["notes"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _fetch_prompt_entry(self, connection, prompt_id):
        row = connection.execute("SELECT * FROM prompt_entries WHERE id = ?", (prompt_id,)).fetchone()
        if row is None:
            raise GalleryNotFoundError("Prompt entry was not found")
        return self._prompt_record(row)

    def list_prompt_entries(self):
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT * FROM prompt_entries ORDER BY updated_at DESC, rowid DESC"
            ).fetchall()
            return [self._prompt_record(row) for row in rows]
        finally:
            connection.close()

    def create_prompt_entry(self, title, positive_prompt="", negative_prompt="", notes=None):
        title = _prompt_title(title)
        positive_prompt = _prompt_text(positive_prompt, "Positive prompt")
        negative_prompt = _prompt_text(negative_prompt, "Negative prompt")
        notes = _prompt_notes(notes)
        if not positive_prompt.strip() and not negative_prompt.strip():
            raise GalleryValidationError("At least one positive or negative prompt is required")
        prompt_id = str(uuid.uuid4())
        now = _timestamp()
        connection = self._connect()
        try:
            self._begin(connection)
            connection.execute(
                """INSERT INTO prompt_entries(
                    id, title, positive_prompt, negative_prompt, notes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (prompt_id, title, positive_prompt, negative_prompt, notes, now, now),
            )
            connection.commit()
            return self._fetch_prompt_entry(connection, prompt_id)
        except Exception:
            self._rollback(connection)
            raise
        finally:
            connection.close()

    def update_prompt_entry(
        self,
        prompt_id,
        *,
        title=UNSET,
        positive_prompt=UNSET,
        negative_prompt=UNSET,
        notes=UNSET,
    ):
        if not isinstance(prompt_id, str) or not prompt_id:
            raise GalleryValidationError("Prompt id must be a non-empty string")
        connection = self._connect()
        try:
            self._begin(connection)
            current = connection.execute("SELECT * FROM prompt_entries WHERE id = ?", (prompt_id,)).fetchone()
            if current is None:
                raise GalleryNotFoundError("Prompt entry was not found")
            next_title = _prompt_title(title) if title is not UNSET else current["title"]
            next_positive = _prompt_text(positive_prompt, "Positive prompt") if positive_prompt is not UNSET else current["positive_prompt"]
            next_negative = _prompt_text(negative_prompt, "Negative prompt") if negative_prompt is not UNSET else current["negative_prompt"]
            next_notes = _prompt_notes(notes) if notes is not UNSET else current["notes"]
            if not next_positive.strip() and not next_negative.strip():
                raise GalleryValidationError("At least one positive or negative prompt is required")
            connection.execute(
                """UPDATE prompt_entries
                   SET title = ?, positive_prompt = ?, negative_prompt = ?, notes = ?, updated_at = ?
                   WHERE id = ?""",
                (next_title, next_positive, next_negative, next_notes, _timestamp(), prompt_id),
            )
            connection.commit()
            return self._fetch_prompt_entry(connection, prompt_id)
        except Exception:
            self._rollback(connection)
            raise
        finally:
            connection.close()

    def delete_prompt_entry(self, prompt_id):
        if not isinstance(prompt_id, str) or not prompt_id:
            raise GalleryValidationError("Prompt id must be a non-empty string")
        connection = self._connect()
        try:
            self._begin(connection)
            if connection.execute("SELECT 1 FROM prompt_entries WHERE id = ?", (prompt_id,)).fetchone() is None:
                raise GalleryNotFoundError("Prompt entry was not found")
            connection.execute("DELETE FROM prompt_entries WHERE id = ?", (prompt_id,))
            connection.commit()
        except Exception:
            self._rollback(connection)
            raise
        finally:
            connection.close()

    def list_gallery(self, collection_id=None):
        if collection_id is not None:
            collection_id = _collection_id(collection_id)
        connection = self._connect()
        try:
            collection_rows = connection.execute(
                """
                SELECT c.id, c.description, c.created_at, c.updated_at,
                       COUNT(DISTINCT ca.id) AS card_count,
                       COUNT(ci.id) AS image_count
                FROM collections c
                LEFT JOIN cards ca ON ca.collection_id = c.id
                LEFT JOIN card_images ci ON ci.card_id = ca.id
                GROUP BY c.id
                ORDER BY c.created_at, c.id
                """
            ).fetchall()
            if collection_id is None:
                card_rows = connection.execute(
                    "SELECT * FROM cards ORDER BY sort_index, created_at DESC, rowid DESC"
                ).fetchall()
            else:
                card_rows = connection.execute(
                    "SELECT * FROM cards WHERE collection_id = ? ORDER BY sort_index, created_at DESC, id DESC",
                    (collection_id,),
                ).fetchall()
            return {
                "collections": [self._collection_record(row) for row in collection_rows],
                "cards": [self._card_record(connection, row) for row in card_rows],
            }
        finally:
            connection.close()

    def create_collection(self, collection_id, description=None):
        collection_id = _collection_id(collection_id)
        description = _description(description)
        now = _timestamp()
        connection = self._connect()
        try:
            self._begin(connection)
            connection.execute(
                "INSERT INTO collections(id, description, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (collection_id, description, now, now),
            )
            connection.commit()
            return self._fetch_collection(connection, collection_id)
        except sqlite3.IntegrityError as error:
            self._rollback(connection)
            raise GalleryConflictError(f"Gallery collection '{collection_id}' already exists") from error
        finally:
            connection.close()

    def update_collection(self, collection_id, *, new_id=UNSET, description=UNSET):
        collection_id = _collection_id(collection_id)
        target_id = _collection_id(new_id) if new_id is not UNSET else collection_id
        target_description = _description(description) if description is not UNSET else UNSET
        connection = self._connect()
        try:
            self._begin(connection)
            existing = connection.execute(
                "SELECT description FROM collections WHERE id = ?", (collection_id,)
            ).fetchone()
            if existing is None:
                raise GalleryNotFoundError("Gallery collection was not found")
            if target_description is UNSET:
                target_description = existing["description"]
            connection.execute(
                "UPDATE collections SET id = ?, description = ?, updated_at = ? WHERE id = ?",
                (target_id, target_description, _timestamp(), collection_id),
            )
            connection.commit()
            return self._fetch_collection(connection, target_id)
        except sqlite3.IntegrityError as error:
            self._rollback(connection)
            raise GalleryConflictError(f"Gallery collection '{target_id}' already exists") from error
        except Exception:
            self._rollback(connection)
            raise
        finally:
            connection.close()

    def delete_collection(self, collection_id):
        collection_id = _collection_id(collection_id)
        connection = self._connect()
        paths = []
        try:
            self._begin(connection)
            exists = connection.execute("SELECT 1 FROM collections WHERE id = ?", (collection_id,)).fetchone()
            if exists is None:
                raise GalleryNotFoundError("Gallery collection was not found")
            storage_rows = connection.execute(
                """
                SELECT ci.storage_name FROM card_images ci
                JOIN cards ca ON ca.id = ci.card_id WHERE ca.collection_id = ?
                """,
                (collection_id,),
            ).fetchall()
            paths = [self._asset_path(row["storage_name"]) for row in storage_rows]
            connection.execute("DELETE FROM collections WHERE id = ?", (collection_id,))
            connection.commit()
        except Exception:
            self._rollback(connection)
            raise
        finally:
            connection.close()
        self._remove_paths(paths)

    def create_card(self, collection_id, title=None, settings=None, images=None):
        collection_id = _collection_id(collection_id)
        title = _title(title)
        settings_json = _settings_json({} if settings is None else settings)
        image_specs = self._normalize_image_specs([] if images is None else images)
        if any(image.get("gallery_image_id") is not None for image in image_specs):
            raise GalleryValidationError("gallery_image_id can only preserve images while updating a card")
        card_id = str(uuid.uuid4())
        now = _timestamp()
        staged = []
        committed = False
        connection = self._connect()
        try:
            self._begin(connection)
            if connection.execute("SELECT 1 FROM collections WHERE id = ?", (collection_id,)).fetchone() is None:
                raise GalleryNotFoundError("Gallery collection was not found")
            connection.execute(
                "UPDATE cards SET sort_index = sort_index + 1 WHERE collection_id = ?", (collection_id,)
            )
            for image in image_specs:
                staged.append(self._stage_new_image(image))
            connection.execute(
                """
                INSERT INTO cards(id, collection_id, title, settings_json, sort_index, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (card_id, collection_id, title, settings_json, 0, now, now),
            )
            for sort_index, image in enumerate(staged):
                connection.execute(
                    """
                    INSERT INTO card_images(
                        id, card_id, sort_index, original_name, mime_type, storage_name, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        image.image_id,
                        card_id,
                        sort_index,
                        image.name,
                        image.mime_type,
                        image.storage_name,
                        now,
                    ),
                )
            self._install_staged(staged)
            connection.commit()
            committed = True
            return self._fetch_card(connection, card_id)
        except Exception:
            self._rollback(connection)
            raise
        finally:
            connection.close()
            if not committed:
                self._cleanup_staged(staged)

    def create_cards(self, collection_id, cards):
        collection_id = _collection_id(collection_id)
        if not isinstance(cards, list) or not cards:
            raise GalleryValidationError("cards must contain at least one card object")
        prepared = []
        for card in cards:
            if not isinstance(card, dict) or set(card) - {"title", "settings", "images"}:
                raise GalleryValidationError("Each card must contain only title, settings, and images")
            title = _title(card.get("title"))
            settings_json = _settings_json(card.get("settings", {}))
            image_specs = self._normalize_image_specs(card.get("images", []))
            if any(image.get("gallery_image_id") is not None for image in image_specs):
                raise GalleryValidationError("gallery_image_id cannot be used while creating cards")
            prepared.append((str(uuid.uuid4()), title, settings_json, image_specs))

        staged = []
        card_ids = []
        committed = False
        connection = self._connect()
        try:
            self._begin(connection)
            if connection.execute("SELECT 1 FROM collections WHERE id = ?", (collection_id,)).fetchone() is None:
                raise GalleryNotFoundError("Gallery collection was not found")
            now = _timestamp()
            connection.execute(
                "UPDATE cards SET sort_index = sort_index + ? WHERE collection_id = ?",
                (len(prepared), collection_id),
            )
            for card_sort_index, (card_id, title, settings_json, image_specs) in enumerate(prepared):
                card_images = []
                for image in image_specs:
                    staged_image = self._stage_new_image(image)
                    staged.append(staged_image)
                    card_images.append(staged_image)
                connection.execute(
                    """
                    INSERT INTO cards(id, collection_id, title, settings_json, sort_index, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (card_id, collection_id, title, settings_json, card_sort_index, now, now),
                )
                for sort_index, image in enumerate(card_images):
                    connection.execute(
                        """
                        INSERT INTO card_images(
                            id, card_id, sort_index, original_name, mime_type, storage_name, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (image.image_id, card_id, sort_index, image.name, image.mime_type, image.storage_name, now),
                    )
                card_ids.append(card_id)
            self._install_staged(staged)
            connection.commit()
            committed = True
            return [self._fetch_card(connection, card_id) for card_id in card_ids]
        except Exception:
            self._rollback(connection)
            raise
        finally:
            connection.close()
            if not committed:
                self._cleanup_staged(staged)

    def update_card(
        self,
        card_id,
        *,
        collection_id=UNSET,
        title=UNSET,
        settings=UNSET,
        images=UNSET,
    ):
        if not isinstance(card_id, str) or not card_id:
            raise GalleryValidationError("Card id must be a non-empty string")
        target_collection = _collection_id(collection_id) if collection_id is not UNSET else UNSET
        target_title = _title(title) if title is not UNSET else UNSET
        target_settings = _settings_json(settings) if settings is not UNSET else UNSET
        image_specs = self._normalize_image_specs(images) if images is not UNSET else UNSET
        staged = []
        deleted_paths = []
        committed = False
        connection = self._connect()
        try:
            self._begin(connection)
            card = connection.execute("SELECT * FROM cards WHERE id = ?", (card_id,)).fetchone()
            if card is None:
                raise GalleryNotFoundError("Gallery card was not found")
            if target_collection is not UNSET and connection.execute(
                "SELECT 1 FROM collections WHERE id = ?", (target_collection,)
            ).fetchone() is None:
                raise GalleryNotFoundError("Gallery collection was not found")
            source_collection = card["collection_id"]
            moving_collection = target_collection is not UNSET and target_collection != source_collection
            if moving_collection:
                connection.execute(
                    "UPDATE cards SET sort_index = sort_index + 1 WHERE collection_id = ?",
                    (target_collection,),
                )

            if image_specs is not UNSET:
                current_rows = connection.execute(
                    "SELECT * FROM card_images WHERE card_id = ? ORDER BY sort_index, id", (card_id,)
                ).fetchall()
                current = {row["id"]: row for row in current_rows}
                preserved = set()
                ordered = []
                for image_spec in image_specs:
                    gallery_image_id = image_spec.get("gallery_image_id")
                    if gallery_image_id is not None:
                        if not isinstance(gallery_image_id, str) or gallery_image_id not in current:
                            raise GalleryValidationError("gallery_image_id does not belong to this card")
                        if gallery_image_id in preserved:
                            raise GalleryValidationError("gallery_image_id cannot appear more than once")
                        preserved.add(gallery_image_id)
                        ordered.append(("existing", current[gallery_image_id]))
                    else:
                        staged_image = self._stage_new_image(image_spec)
                        staged.append(staged_image)
                        ordered.append(("new", staged_image))

                omitted = [row for row in current_rows if row["id"] not in preserved]
                deleted_paths = [self._asset_path(row["storage_name"]) for row in omitted]
                if omitted:
                    connection.executemany("DELETE FROM card_images WHERE id = ?", [(row["id"],) for row in omitted])
                for sort_index, (kind, image) in enumerate(ordered):
                    if kind == "existing":
                        connection.execute(
                            "UPDATE card_images SET sort_index = ? WHERE id = ?",
                            (sort_index, image["id"]),
                        )
                    else:
                        connection.execute(
                            """
                            INSERT INTO card_images(
                                id, card_id, sort_index, original_name, mime_type, storage_name, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                image.image_id,
                                card_id,
                                sort_index,
                                image.name,
                                image.mime_type,
                                image.storage_name,
                                _timestamp(),
                            ),
                        )

            updates = ["updated_at = ?"]
            values = [_timestamp()]
            if target_collection is not UNSET:
                updates.append("collection_id = ?")
                values.append(target_collection)
                if moving_collection:
                    updates.append("sort_index = ?")
                    values.append(0)
            if target_title is not UNSET:
                updates.append("title = ?")
                values.append(target_title)
            if target_settings is not UNSET:
                updates.append("settings_json = ?")
                values.append(target_settings)
            values.append(card_id)
            connection.execute(f"UPDATE cards SET {', '.join(updates)} WHERE id = ?", values)
            if moving_collection:
                self._normalize_card_order(connection, source_collection)
            self._install_staged(staged)
            connection.commit()
            committed = True
            result = self._fetch_card(connection, card_id)
        except Exception:
            self._rollback(connection)
            raise
        finally:
            connection.close()
            if not committed:
                self._cleanup_staged(staged)
        self._remove_paths(deleted_paths)
        return result

    def delete_card(self, card_id):
        if not isinstance(card_id, str) or not card_id:
            raise GalleryValidationError("Card id must be a non-empty string")
        connection = self._connect()
        paths = []
        try:
            self._begin(connection)
            card = connection.execute("SELECT collection_id FROM cards WHERE id = ?", (card_id,)).fetchone()
            if card is None:
                raise GalleryNotFoundError("Gallery card was not found")
            storage_rows = connection.execute(
                "SELECT storage_name FROM card_images WHERE card_id = ?", (card_id,)
            ).fetchall()
            paths = [self._asset_path(row["storage_name"]) for row in storage_rows]
            connection.execute("DELETE FROM cards WHERE id = ?", (card_id,))
            self._normalize_card_order(connection, card["collection_id"])
            connection.commit()
        except Exception:
            self._rollback(connection)
            raise
        finally:
            connection.close()
        self._remove_paths(paths)

    def reorder_cards(self, collection_id, card_ids):
        collection_id = _collection_id(collection_id)
        if not isinstance(card_ids, list) or not 1 <= len(card_ids) <= 10000:
            raise GalleryValidationError("card_ids must contain 1 to 10000 card ids")
        if any(not isinstance(card_id, str) or not card_id for card_id in card_ids):
            raise GalleryValidationError("Every card id must be a non-empty string")
        if len(set(card_ids)) != len(card_ids):
            raise GalleryValidationError("card_ids must not contain duplicates")
        connection = self._connect()
        try:
            self._begin(connection)
            if connection.execute("SELECT 1 FROM collections WHERE id = ?", (collection_id,)).fetchone() is None:
                raise GalleryNotFoundError("Gallery collection was not found")
            existing = connection.execute(
                "SELECT id FROM cards WHERE collection_id = ?", (collection_id,)
            ).fetchall()
            existing_ids = {row["id"] for row in existing}
            if len(existing_ids) != len(card_ids) or existing_ids != set(card_ids):
                raise GalleryValidationError(
                    "card_ids must contain every card in the collection exactly once"
                )
            connection.executemany(
                "UPDATE cards SET sort_index = ? WHERE id = ?",
                [(index, card_id) for index, card_id in enumerate(card_ids)],
            )
            connection.execute(
                "UPDATE collections SET updated_at = ? WHERE id = ?", (_timestamp(), collection_id)
            )
            connection.commit()
            return [self._fetch_card(connection, card_id) for card_id in card_ids]
        except Exception:
            self._rollback(connection)
            raise
        finally:
            connection.close()

    def _thumbnail_path(self, storage_name):
        # Derived from the validated original path, so the containment check that
        # guards the original guards the derivative too.
        original = self._asset_path(storage_name)
        return original.with_name(f"{original.stem}{THUMBNAIL_SUFFIX}")

    def _ensure_thumbnail(self, storage_name):
        """Returns a cached thumbnail path, or None when one cannot be produced.

        A missing or unreadable derivative is never fatal: the caller falls back
        to the original, so a corrupt cache costs bandwidth rather than a broken
        image.
        """
        original = self._asset_path(storage_name)
        thumbnail = self._thumbnail_path(storage_name)
        try:
            original_stat = original.stat()
            if thumbnail.is_file():
                thumbnail_stat = thumbnail.stat()
                # A re-imported original reuses the id, so staleness is checked
                # rather than assumed.
                if thumbnail_stat.st_mtime_ns >= original_stat.st_mtime_ns and thumbnail_stat.st_size > 0:
                    return thumbnail
        except OSError:
            return None

        # An original that is already small costs less to send than to derive from.
        if original_stat.st_size <= THUMBNAIL_MIN_ORIGINAL_BYTES:
            return None

        temporary_path = thumbnail.with_name(f"{thumbnail.name}.{uuid.uuid4().hex}.tmp")
        try:
            with Image.open(original) as handle:
                if max(handle.size) <= THUMBNAIL_MAX_EDGE:
                    return None
                handle.draft("RGB", (THUMBNAIL_MAX_EDGE, THUMBNAIL_MAX_EDGE))
                frame = handle.convert("RGBA" if "A" in handle.getbands() else "RGB")
                frame.thumbnail((THUMBNAIL_MAX_EDGE, THUMBNAIL_MAX_EDGE), Image.LANCZOS)
                encoded = io.BytesIO()
                frame.save(encoded, format="WEBP", quality=THUMBNAIL_QUALITY, method=4)
            payload = encoded.getvalue()
            # A derivative that is not actually smaller is not worth serving, and
            # publishing one would make browsing slower rather than faster.
            if len(payload) >= original_stat.st_size:
                return None
            temporary_path.write_bytes(payload)
            # Atomic publish: a reader never observes a half-written derivative.
            os.replace(temporary_path, thumbnail)
            return thumbnail
        except (OSError, ValueError, Image.DecompressionBombError, Image.UnidentifiedImageError):
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
            return None

    def get_image_file(self, image_id, variant="original"):
        if not isinstance(image_id, str) or not image_id:
            raise GalleryValidationError("Image id must be a non-empty string")
        if variant not in ("original", "thumb"):
            raise GalleryValidationError("Image variant must be original or thumb")
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT id, original_name, mime_type, storage_name FROM card_images WHERE id = ?", (image_id,)
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise GalleryNotFoundError("Gallery image was not found")
        path = self._asset_path(row["storage_name"])
        if not path.is_file():
            raise GalleryNotFoundError("Gallery image file is missing")
        if variant == "thumb":
            thumbnail = self._ensure_thumbnail(row["storage_name"])
            if thumbnail is not None:
                return {
                    "id": row["id"],
                    "name": f"{path.stem}{THUMBNAIL_SUFFIX}",
                    "mime_type": THUMBNAIL_MIME,
                    "path": thumbnail,
                    "variant": "thumb",
                }
            # Fall through to the original rather than failing the request.
        return {
            "id": row["id"],
            "name": row["original_name"],
            "mime_type": row["mime_type"],
            "path": path,
            "variant": "original",
        }
