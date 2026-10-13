import base64
import io
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from PIL import Image
from backend.gallery import GalleryStore, GalleryConflictError


class GalleryRelinkTests(unittest.TestCase):
    def setUp(self):
        base = os.environ.get("P10_EVIDENCE_DIR")
        if base:
            Path(base).mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="p10-gallery-", dir=base))
        self.store = GalleryStore(self.root)
        self.store.create_collection("test")
        self.before = {"model": "Anima", "positive": "用户文字 <lora:IDA:0.8137>", "loras": [{"value": "style/A.safetensors", "weight": 0.8137}], "loraGroups": [{"id": "historic", "presetPrompt": "keep"}], "imageSeeds": ["42"]}
        self.card = self.store.create_card("test", title="user title", settings=self.before)

    def delegate(self):
        token = str(uuid.uuid4())
        lock = self.root / ".lora-coordination.lock"
        lock.mkdir(parents=True)
        (lock / "lease.json").write_text(json.dumps({"owner": "node", "token": token, "pid": os.getpid()}), encoding="utf-8")
        return GalleryStore(self.root, relink_token=token)

    def manifest(self, phase, revision):
        file = self.root / "lora-library" / "publication.json"
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps({"schemaVersion": 1, "phase": phase, "revision": revision}), encoding="utf-8")

    def test_after_a_publication_a_new_card_must_carry_the_current_revision(self):
        # The contract the browser honours with `withGalleryLoraRevision` (found by QPlan2/P1 Phase E):
        # new gallery content without the revision is refused once a LoRA publication exists.
        self.manifest("published", 3)
        with self.assertRaises(GalleryConflictError):
            self.store.create_card("test", settings=self.before)
        with self.assertRaises(GalleryConflictError):
            self.store.create_card("test", settings={**self.before, "loraLibraryRevision": 2})
        created = self.store.create_card("test", settings={**self.before, "loraLibraryRevision": 3})
        self.assertEqual(created["settings"]["loraLibraryRevision"], 3)
        bulk = self.store.create_cards("test", [{"settings": {**self.before, "loraLibraryRevision": 3}}])
        self.assertEqual(len(bulk), 1)

    def test_legacy_owner_crud_does_not_initialize_library_or_participant_table(self):
        self.store.update_card(self.card["id"], settings=self.before, title="changed")
        self.assertFalse((self.root / "lora-library").exists())
        connection = self.store._connect()
        try:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 4)
            self.assertIsNone(connection.execute("SELECT name FROM sqlite_master WHERE name = 'lora_relink_operations'").fetchone())
        finally:
            connection.close()

    def test_prepare_publish_idempotency_does_not_mutate_before_publish(self):
        owner = self.delegate()
        operation = str(uuid.uuid4())
        after = {**self.before, "loras": [{"value": "concept/A.safetensors", "weight": 0.8137}]}
        patches = [{"cardId": self.card["id"], "before": self.before, "after": after}]
        owner.relink_transaction("check", operation, patches)
        owner.relink_transaction("prepare", operation, patches)
        owner.relink_transaction("prepare", operation, patches)
        connection = owner._connect()
        try:
            record = owner._fetch_card(connection, self.card["id"])
            self.assertEqual(record["settings"], self.before)
        finally:
            connection.close()
        owner.relink_transaction("publish", operation, [])
        owner.relink_transaction("publish", operation, [])
        connection = owner._connect()
        try:
            self.assertEqual(owner._fetch_card(connection, self.card["id"])["settings"], after)
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 4)
        finally:
            connection.close()

    def test_settings_cas_and_duplicate_operation_conflicts(self):
        owner = self.delegate()
        operation = str(uuid.uuid4())
        patch = {"cardId": self.card["id"], "before": {"wrong": "baseline"}, "after": self.before}
        with self.assertRaises(GalleryConflictError):
            owner.relink_transaction("prepare", operation, [patch])
        patch["before"] = self.before
        owner.relink_transaction("prepare", operation, [patch])
        with self.assertRaises(GalleryConflictError):
            owner.relink_transaction("prepare", operation, [{**patch, "after": {"different": True}}])

    def test_normal_gallery_blocks_prepare_window_then_rejects_late_old_settings(self):
        owner = self.delegate()
        self.manifest("preparing", 0)
        with self.assertRaises(GalleryConflictError):
            GalleryStore(self.root)
        lock = self.root / ".lora-coordination.lock"
        (lock / "lease.json").unlink()
        lock.rmdir()
        self.manifest("published", 1)
        current = GalleryStore(self.root)
        with self.assertRaises(GalleryConflictError):
            current.update_card(self.card["id"], settings=self.before)
        settings = {**self.before, "loraLibraryRevision": 1}
        changed = current.update_card(self.card["id"], settings=settings, title="later edit")
        self.assertEqual(changed["settings"]["positive"], self.before["positive"])
        self.assertEqual(changed["title"], "later edit")

    def test_repair_preserves_original_image_bytes_title_timestamp_and_run_snapshot(self):
        image = io.BytesIO()
        Image.new("RGB", (8, 8), "red").save(image, format="PNG")
        card = self.store.create_card("test", title="image card", settings=self.before,
                                      images=[{"data_url": "data:image/png;base64," + base64.b64encode(image.getvalue()).decode("ascii"), "name": "original.png"}])
        assets = {file.name: file.read_bytes() for file in (self.root / "gallery-assets").iterdir()}
        owner = self.delegate()
        after = {**self.before, "loras": [{"value": "concept/A.safetensors", "weight": 0.8137}]}
        operation = str(uuid.uuid4())
        owner.relink_transaction("prepare", operation, [{"cardId": card["id"], "before": self.before, "after": after}])
        owner.relink_transaction("publish", operation, [])
        connection = owner._connect()
        try:
            patched = owner._fetch_card(connection, card["id"])
            self.assertEqual(patched["title"], card["title"])
            self.assertEqual(patched["created_at"], card["created_at"])
            self.assertEqual(patched["updated_at"], card["updated_at"])
            self.assertEqual(patched["settings"]["loraGroups"], card["settings"]["loraGroups"])
        finally:
            connection.close()
        self.assertEqual({file.name: file.read_bytes() for file in (self.root / "gallery-assets").iterdir()}, assets)

    def test_future_identity_survives_late_settings_save_but_unrelated_title_edit_is_allowed(self):
        future = {**self.before, "loras": [{"value": "style/A.safetensors", "identitySchema": 999, "weight": .8137}]}
        connection = self.store._connect()
        try:
            connection.execute("UPDATE cards SET settings_json = ? WHERE id = ?", (json.dumps(future), self.card["id"]))
            connection.commit()
        finally:
            connection.close()
        with self.assertRaises(GalleryConflictError):
            self.store.update_card(self.card["id"], settings=self.before)
        with self.assertRaises(GalleryConflictError):
            self.store.create_card("test", settings=future)
        changed = self.store.update_card(self.card["id"], title="unrelated title")
        self.assertEqual(changed["settings"], future)
        self.assertEqual(changed["title"], "unrelated title")


if __name__ == "__main__":
    unittest.main()
