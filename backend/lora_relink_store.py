"""Gallery-owner participation in the Node LoRA repair publication barrier.

No import-time database or state writes. The private delegation token is a
cross-process lock capability, NOT user authentication or an HTTP credential.
"""
import json
import os
import sqlite3
import socket
import uuid
import re
from pathlib import Path


def publication(state_directory):
    file = Path(state_directory) / "lora-library" / "publication.json"
    value = json.loads(file.read_text(encoding="utf-8")) if file.exists() else None
    operations = file.parent / "operations"
    if operations.exists():
        for journal_file in operations.glob("*.json"):
            if not re.fullmatch(r"[0-9a-f-]{36}\.json", journal_file.name):
                continue
            record = json.loads(journal_file.read_text(encoding="utf-8"))
            if (not isinstance(record, dict) or record.get("schemaVersion") != 1
                    or not re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", str(record.get("operationId", "")))
                    or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("requestDigest", "")))
                    or type(record.get("baseRevision")) is not int or not 0 <= record["baseRevision"] < 2**53 - 1
                    or type(record.get("revision")) is not int or record["revision"] != record["baseRevision"] + 1
                    or record.get("phase") not in {"intent", "prepared", "applied", "published"}
                    or not isinstance(record.get("files"), dict) or set(record["files"]) - {"registry", "uiState", "cards"}
                    or not isinstance(record.get("galleryPatches"), list)):
                raise ValueError("Invalid LoRA operation journal; evidence retained")
            if record.get("phase") != "published":
                return {"schemaVersion": 1, "revision": record.get("baseRevision", 0), "phase": "recovering"}
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("schemaVersion") != 1 or type(value.get("revision")) is not int or value["revision"] < 0 or value.get("phase") not in {"published", "preparing"}:
        raise ValueError("Invalid LoRA publication manifest")
    return value


class LibraryConnection(sqlite3.Connection):
    lease_path = None
    lease_token = None
    library_revision = None
    admission = None

    def close(self):
        try:
            super().close()
        finally:
            if self.admission is not None:
                self.admission.__exit__(None, None, None)
                self.admission = None
            if self.lease_path is not None:
                lease = self.lease_path / "lease.json"
                value = json.loads(lease.read_text(encoding="utf-8"))
                if value.get("token") == self.lease_token:
                    lease.unlink()
                    self.lease_path.rmdir()
                self.lease_path = None


def connect_gallery(store):
    from .gallery import GalleryConflictError
    from .lora_library_gate import library_admission, LibraryGateError
    delegated = getattr(store, "_relink_token", None)
    admission = library_admission(store.state_directory, delegated=delegated, check_publication=not bool(delegated))
    entered = False
    try:
        revision = admission.__enter__()
        entered = True
        state = publication(store.state_directory)
        connection = sqlite3.connect(store.database_path, timeout=30, factory=LibraryConnection)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.admission = admission
        connection.library_revision = revision if state is not None else None
        return connection
    except LibraryGateError as error:
        if entered:
            admission.__exit__(None, None, None)
        raise GalleryConflictError(str(error)) from error
    except Exception:
        if entered:
            admission.__exit__(None, None, None)
        raise


def _assert_settings_identity(settings):
    from .gallery import GalleryConflictError
    loras = settings.get("loras", [])
    for item in loras if isinstance(loras, list) else []:
        if not isinstance(item, dict):
            continue  # Legacy validation belongs to the existing settings owner.
        if (("identitySchema" in item and (type(item["identitySchema"]) is not int or item["identitySchema"] != 1))
                or ("instanceId" in item and not re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", str(item["instanceId"])))
                or ("contentHash" in item and not re.fullmatch(r"[0-9a-f]{64}", str(item["contentHash"])))
                or ("contentRevision" in item and (type(item["contentRevision"]) is not int or item["contentRevision"] < 1))):
            raise GalleryConflictError("Unrecognized Gallery LoRA identity; original settings retained")
        if "originalReference" in item:
            ref = item["originalReference"]
            if (not isinstance(ref, dict) or not isinstance(ref.get("value"), str) or not 0 < len(ref["value"]) <= 500 or "\0" in ref["value"]
                    or ("instanceId" in ref and not re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", str(ref["instanceId"])))
                    or ("contentHash" in ref and not re.fullmatch(r"[0-9a-f]{64}", str(ref["contentHash"])))):
                raise GalleryConflictError("Unrecognized Gallery LoRA identity; original settings retained")


def require_settings_revision(store, settings, previous=None):
    from .gallery import GalleryConflictError
    _assert_settings_identity(settings)
    if previous is not None:
        _assert_settings_identity(previous)
    state = publication(store.state_directory)
    if state and (type(settings.get("loraLibraryRevision", 0)) is not int or settings.get("loraLibraryRevision", 0) != state["revision"]):
        raise GalleryConflictError("Stale Gallery LoRA revision; refetch before saving")


def _validate_patches(connection, patches):
    from .gallery import GalleryConflictError, GalleryValidationError, _settings_json
    seen = set()
    for patch in patches:
        card_id = patch.get("cardId")
        if not isinstance(card_id, str) or card_id in seen:
            raise GalleryValidationError("Duplicate or invalid Gallery participant")
        seen.add(card_id)
        row = connection.execute("SELECT settings_json FROM cards WHERE id = ?", (card_id,)).fetchone()
        expected = patch.get("before")
        actual = json.loads(row["settings_json"]) if row is not None else None
        if isinstance(expected, dict):
            expected = {key: value for key, value in expected.items() if key != "loraLibraryRevision"}
        if isinstance(actual, dict):
            actual = {key: value for key, value in actual.items() if key != "loraLibraryRevision"}
        if row is None or actual != expected:
            raise GalleryConflictError("Gallery consumer edited before repair")
        _settings_json(patch.get("after"))
        _assert_settings_identity(actual)
        _assert_settings_identity(patch["after"])


def gallery_relink(store, action, operation_id, patches):
    """Called ONLY through GalleryStore's private transactional owner entry."""
    from .gallery import GalleryConflictError, GalleryValidationError, _settings_json
    if not getattr(store, "_relink_token", None):
        raise GalleryConflictError("Gallery repair requires the coordinator lease")
    if action not in {"check", "prepare", "publish"}:
        raise GalleryValidationError("Unsupported Gallery repair action")
    connection = store._connect()
    try:
        store._begin(connection)
        if action == "check":
            _validate_patches(connection, patches)
            connection.rollback()
            return {"status": "checked"}
        # Additive, lazy participant table, only created on explicit prepare.
        # It does not change gallery user_version or any image/collection schema.
        connection.execute("CREATE TABLE IF NOT EXISTS lora_relink_operations (id TEXT PRIMARY KEY, phase TEXT NOT NULL, patches_json TEXT NOT NULL)")
        row = connection.execute("SELECT phase, patches_json FROM lora_relink_operations WHERE id = ?", (operation_id,)).fetchone()
        if action == "prepare":
            if row:
                if json.loads(row["patches_json"]) != patches:
                    raise GalleryConflictError("Gallery operation id reused with different patches")
            else:
                _validate_patches(connection, patches)
                connection.execute("INSERT INTO lora_relink_operations VALUES (?, 'prepared', ?)", (operation_id, json.dumps(patches, ensure_ascii=False)))
        else:
            if not row:
                raise GalleryConflictError("Gallery publication has no prepared intent")
            if row["phase"] != "published":
                owned = json.loads(row["patches_json"])
                _validate_patches(connection, owned)
                for patch in owned:
                    connection.execute("UPDATE cards SET settings_json = ? WHERE id = ?", (_settings_json(patch["after"]), patch["cardId"]))
                connection.execute("UPDATE lora_relink_operations SET phase = 'published' WHERE id = ?", (operation_id,))
        connection.commit()
        return {"status": "published" if action == "publish" else "prepared"}
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
