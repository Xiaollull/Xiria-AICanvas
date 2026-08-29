import base64
import io
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from PIL.PngImagePlugin import PngInfo
from pydantic import ValidationError

try:
    from . import gallery
    from .gallery import (
        GalleryConflictError,
        GalleryNotFoundError,
        GalleryStorageError,
        GalleryStore,
        GalleryValidationError,
    )
except ImportError:
    import gallery
    from gallery import (
        GalleryConflictError,
        GalleryNotFoundError,
        GalleryStorageError,
        GalleryStore,
        GalleryValidationError,
    )


class GalleryStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.outputs = self.root / "outputs"
        self.outputs.mkdir()

        def resolve_asset(asset_id):
            path = (self.outputs / asset_id).resolve()
            if self.outputs.resolve() not in path.parents:
                raise ValueError("unsafe asset")
            return path

        self.store = GalleryStore(self.root / "state-cache", asset_resolver=resolve_asset)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def image_bytes(image_format="PNG", color="red"):
        output = io.BytesIO()
        image = Image.new("RGBA" if image_format in {"PNG", "WEBP"} else "RGB", (8, 6), color)
        save_options = {}
        if image_format == "PNG":
            metadata = PngInfo()
            metadata.add_text("parameters", '{"prompt":"preserve me"}')
            save_options["pnginfo"] = metadata
        image.save(output, format=image_format, **save_options)
        return output.getvalue()

    @staticmethod
    def data_url(raw, mime_type):
        return f"data:{mime_type};base64," + base64.b64encode(raw).decode("ascii")

    def add_output(self, name="generated.png", image_format="PNG", color="red"):
        path = self.outputs / name
        raw = self.image_bytes(image_format, color)
        path.write_bytes(raw)
        return name, path, raw

    def create_collection(self, collection_id="Favorites"):
        return self.store.create_collection(collection_id, "Curated images")

    def test_create_list_filter_and_duplicate_collection(self):
        first = self.store.create_collection("  梦境 Favorites  ", "Night studies")
        second = self.store.create_collection("Portraits")
        old_card = self.store.create_card(first["id"], title="Old", settings={"seed": 1})
        new_card = self.store.create_card(second["id"], title="New", settings={"seed": 2})

        gallery_result = self.store.list_gallery()
        self.assertEqual(first["id"], "梦境 Favorites")
        self.assertEqual([card["id"] for card in gallery_result["cards"]], [new_card["id"], old_card["id"]])
        self.assertEqual(
            {collection["id"]: collection["card_count"] for collection in gallery_result["collections"]},
            {"梦境 Favorites": 1, "Portraits": 1},
        )
        filtered = self.store.list_gallery(" 梦境 Favorites ")
        self.assertEqual([card["id"] for card in filtered["cards"]], [old_card["id"]])
        self.assertEqual(len(filtered["collections"]), 2)
        with self.assertRaisesRegex(GalleryConflictError, "already exists"):
            self.store.create_collection("梦境 Favorites")
        for unsafe in (".", "..", "folder/name", "folder\\name"):
            with self.subTest(unsafe=unsafe), self.assertRaises(GalleryValidationError):
                self.store.create_collection(unsafe)

    def test_empty_card_has_required_response_shape(self):
        self.create_collection()
        card = self.store.create_card("Favorites")

        self.assertEqual(card["collection_id"], "Favorites")
        self.assertIsNone(card["title"])
        self.assertEqual(card["settings"], {})
        self.assertEqual(card["images"], [])
        self.assertEqual(card["image_count"], 0)
        self.assertTrue(card["created_at"].endswith("Z"))
        self.assertTrue(card["updated_at"].endswith("Z"))

    def test_all_collections_use_insertion_order_when_card_timestamps_tie(self):
        self.create_collection("First")
        self.create_collection("Second")
        with patch.object(gallery, "_timestamp", return_value="2026-07-30T10:00:00.000000Z"):
            older = self.store.create_card("First", title="Older")
            newer = self.store.create_card("Second", title="Newer")

        cards = self.store.list_gallery()["cards"]
        self.assertEqual([card["id"] for card in cards], [newer["id"], older["id"]])

    def test_generated_asset_copy_preserves_exact_png_bytes_and_metadata(self):
        self.create_collection()
        asset_id, source, raw = self.add_output()
        card = self.store.create_card("Favorites", images=[{"asset_id": asset_id}])
        image = card["images"][0]
        copied = self.store.get_image_file(image["id"])

        self.assertEqual(copied["path"].read_bytes(), raw)
        self.assertNotEqual(copied["path"], source)
        self.assertEqual(image["mime_type"], "image/png")
        self.assertEqual(image["url"], f"/api/inference/gallery/images/{image['id']}")
        source.unlink()
        with Image.open(copied["path"]) as saved:
            self.assertEqual(saved.info["parameters"], '{"prompt":"preserve me"}')

    def test_multiple_images_keep_request_order(self):
        self.create_collection()
        png_id, _, _ = self.add_output("first.png", "PNG", "red")
        gif_id, _, gif_raw = self.add_output("second.gif", "GIF", "blue")
        upload = self.data_url(self.image_bytes("JPEG", "green"), "image/jpeg")
        card = self.store.create_card(
            "Favorites",
            images=[
                {"asset_id": gif_id},
                {"data_url": upload, "name": "manual.jpg"},
                {"asset_id": png_id},
            ],
        )

        self.assertEqual([image["sort_index"] for image in card["images"]], [0, 1, 2])
        self.assertEqual([image["name"] for image in card["images"]], ["second.gif", "manual.jpg", "first.png"])
        self.assertEqual(
            [image["mime_type"] for image in card["images"]],
            ["image/gif", "image/jpeg", "image/png"],
        )
        self.assertEqual(self.store.get_image_file(card["images"][0]["id"])["path"].read_bytes(), gif_raw)
        collection = self.store.list_gallery()["collections"][0]
        self.assertEqual(collection["card_count"], 1)
        self.assertEqual(collection["image_count"], 3)

    def test_bulk_card_creation_is_atomic(self):
        self.create_collection()
        first_id, _, _ = self.add_output("first.png")
        second_id, _, _ = self.add_output("second.png", color="blue")
        cards = self.store.create_cards("Favorites", [
            {"title": "First", "settings": {"seed": 1}, "images": [{"asset_id": first_id}]},
            {"title": "Second", "settings": {"seed": 2}, "images": [{"asset_id": second_id}]},
        ])
        self.assertEqual([card["title"] for card in cards], ["First", "Second"])
        before = {path.name for path in self.store.assets_directory.iterdir()}
        with self.assertRaises(GalleryValidationError):
            self.store.create_cards("Favorites", [
                {"title": "Valid", "images": [{"asset_id": first_id}]},
                {"title": "Invalid", "images": [{"data_url": "data:image/png;base64,not-base64"}]},
            ])
        self.assertEqual(len(self.store.list_gallery()["cards"]), 2)
        self.assertEqual({path.name for path in self.store.assets_directory.iterdir()}, before)

    def test_bulk_card_creation_has_no_image_count_limit(self):
        self.create_collection()
        asset_id, _, _ = self.add_output("unlimited.png")
        cards = self.store.create_cards("Favorites", [
            {"title": "Large card", "images": [{"asset_id": asset_id} for _ in range(201)]},
        ])
        self.assertEqual(cards[0]["image_count"], 201)
        collection = self.store.list_gallery()["collections"][0]
        self.assertEqual(collection["image_count"], 201)

    def test_gallery_api_models_have_no_card_or_image_count_limit(self):
        try:
            from . import inference_server
        except ImportError:
            import inference_server

        images = [{"asset_id": f"asset-{index}.png"} for index in range(201)]
        card = inference_server.GalleryCardCreateInput(collection_id="Favorites", images=images)
        update = inference_server.GalleryCardUpdateInput(images=images)
        bulk = inference_server.GalleryCardsCreateInput(
            collection_id="Favorites",
            cards=[{"images": [{"asset_id": f"asset-{index}.png"}]} for index in range(201)],
        )
        order = inference_server.GalleryCardOrderInput(card_ids=[f"card-{index}" for index in range(10001)])

        self.assertEqual(len(card.images), 201)
        self.assertEqual(len(update.images), 201)
        self.assertEqual(len(bulk.cards), 201)
        self.assertEqual(len(order.card_ids), 10001)

    def test_gallery_prompt_api_models_are_strict(self):
        try:
            from . import inference_server
        except ImportError:
            import inference_server

        created = inference_server.GalleryPromptCreateInput(
            title=" Library ", positive_prompt="positive", negative_prompt="", notes=None
        )
        self.assertEqual(created.title, "Library")
        updated = inference_server.GalleryPromptUpdateInput(negative_prompt="negative", notes=None)
        self.assertEqual(updated.negative_prompt, "negative")
        with self.assertRaises(ValidationError):
            inference_server.GalleryPromptCreateInput(title="Empty", positive_prompt=" ", negative_prompt="")
        with self.assertRaises(ValidationError):
            inference_server.GalleryPromptCreateInput(title="Prompt", positive_prompt="value", extra="refused")
        with self.assertRaises(ValidationError):
            inference_server.GalleryPromptUpdateInput()
        with self.assertRaises(ValidationError):
            inference_server.GalleryPromptUpdateInput(positive_prompt=None)

    def test_card_order_is_persistent_and_requires_complete_collection(self):
        self.create_collection()
        first = self.store.create_card("Favorites", title="First")
        second = self.store.create_card("Favorites", title="Second")
        third = self.store.create_card("Favorites", title="Third")
        self.assertEqual(
            [card["id"] for card in self.store.list_gallery("Favorites")["cards"]],
            [third["id"], second["id"], first["id"]],
        )

        requested = [first["id"], third["id"], second["id"]]
        reordered = self.store.reorder_cards("Favorites", requested)
        self.assertEqual([card["id"] for card in reordered], requested)
        listed = self.store.list_gallery("Favorites")["cards"]
        self.assertEqual([card["id"] for card in listed], requested)
        self.assertEqual([card["sort_index"] for card in listed], [0, 1, 2])

        with self.assertRaisesRegex(GalleryValidationError, "every card"):
            self.store.reorder_cards("Favorites", requested[:-1])
        with self.assertRaisesRegex(GalleryValidationError, "duplicates"):
            self.store.reorder_cards("Favorites", [first["id"], first["id"], second["id"]])
        self.assertEqual(
            [card["id"] for card in self.store.list_gallery("Favorites")["cards"]], requested
        )

    def test_new_and_moved_cards_enter_collection_at_the_top(self):
        self.create_collection("First")
        self.create_collection("Second")
        older = self.store.create_card("First", title="Older")
        newer = self.store.create_card("First", title="Newer")
        destination = self.store.create_card("Second", title="Destination")
        self.assertEqual(
            [card["id"] for card in self.store.list_gallery("First")["cards"]],
            [newer["id"], older["id"]],
        )

        self.store.update_card(older["id"], collection_id="Second")
        self.assertEqual(
            [card["id"] for card in self.store.list_gallery("Second")["cards"]],
            [older["id"], destination["id"]],
        )
        self.assertEqual(
            [card["sort_index"] for card in self.store.list_gallery("First")["cards"]], [0]
        )

    def test_version_one_database_migrates_without_changing_visible_order(self):
        state_directory = self.root / "legacy-state"
        state_directory.mkdir()
        database_path = state_directory / "gallery.sqlite3"
        connection = sqlite3.connect(database_path)
        try:
            connection.executescript(
                """
                CREATE TABLE collections (
                    id TEXT PRIMARY KEY, description TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE cards (
                    id TEXT PRIMARY KEY, collection_id TEXT NOT NULL, title TEXT,
                    settings_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    FOREIGN KEY(collection_id) REFERENCES collections(id) ON UPDATE CASCADE ON DELETE CASCADE
                );
                CREATE TABLE card_images (
                    id TEXT PRIMARY KEY, card_id TEXT NOT NULL, sort_index INTEGER NOT NULL,
                    original_name TEXT NOT NULL, mime_type TEXT NOT NULL, storage_name TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL, FOREIGN KEY(card_id) REFERENCES cards(id) ON DELETE CASCADE
                );
                PRAGMA user_version = 1;
                """
            )
            connection.execute(
                "INSERT INTO collections VALUES (?, ?, ?, ?)",
                ("Legacy", None, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
            )
            connection.executemany(
                "INSERT INTO cards VALUES (?, ?, ?, ?, ?, ?)",
                [
                    ("older", "Legacy", "Older", "{}", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
                    ("newer", "Legacy", "Newer", "{}", "2026-01-02T00:00:00Z", "2026-01-02T00:00:00Z"),
                ],
            )
            connection.commit()
        finally:
            connection.close()

        migrated = GalleryStore(state_directory)
        cards = migrated.list_gallery("Legacy")["cards"]
        self.assertEqual([card["id"] for card in cards], ["newer", "older"])
        self.assertEqual([card["sort_index"] for card in cards], [0, 1])
        connection = sqlite3.connect(database_path)
        try:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 3)
        finally:
            connection.close()

    def test_version_two_database_adds_prompt_library_without_changing_cards(self):
        state_directory = self.root / "version-two-state"
        state_directory.mkdir()
        database_path = state_directory / "gallery.sqlite3"
        connection = sqlite3.connect(database_path)
        try:
            connection.executescript(
                """
                CREATE TABLE collections (
                    id TEXT PRIMARY KEY, description TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE cards (
                    id TEXT PRIMARY KEY, collection_id TEXT NOT NULL, title TEXT, settings_json TEXT NOT NULL,
                    sort_index INTEGER NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    FOREIGN KEY(collection_id) REFERENCES collections(id) ON UPDATE CASCADE ON DELETE CASCADE
                );
                CREATE TABLE card_images (
                    id TEXT PRIMARY KEY, card_id TEXT NOT NULL, sort_index INTEGER NOT NULL,
                    original_name TEXT NOT NULL, mime_type TEXT NOT NULL, storage_name TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL, FOREIGN KEY(card_id) REFERENCES cards(id) ON DELETE CASCADE
                );
                CREATE INDEX cards_collection_created_idx ON cards(collection_id, created_at DESC);
                CREATE INDEX cards_collection_sort_idx ON cards(collection_id, sort_index);
                CREATE INDEX card_images_card_order_idx ON card_images(card_id, sort_index);
                PRAGMA user_version = 2;
                """
            )
            connection.execute(
                "INSERT INTO collections VALUES (?, ?, ?, ?)",
                ("Existing", None, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
            )
            connection.execute(
                "INSERT INTO cards VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("card", "Existing", "Kept", '{"seed":7}', 0, "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
            )
            connection.commit()
        finally:
            connection.close()

        migrated = GalleryStore(state_directory)
        self.assertEqual(migrated.list_gallery()["cards"][0]["settings"], {"seed": 7})
        self.assertEqual(migrated.list_prompt_entries(), [])
        prompt = migrated.create_prompt_entry("New library", "prompt")
        self.assertEqual(migrated.list_prompt_entries()[0]["id"], prompt["id"])
        connection = sqlite3.connect(database_path)
        try:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 3)
        finally:
            connection.close()

    def test_post_commit_file_cleanup_failure_does_not_report_database_failure(self):
        self.create_collection()
        asset_id, _, _ = self.add_output()
        card = self.store.create_card("Favorites", images=[{"asset_id": asset_id}])
        with patch.object(Path, "unlink", side_effect=PermissionError("locked")):
            with self.assertWarnsRegex(RuntimeWarning, "orphaned gallery image"):
                self.store.delete_card(card["id"])
        self.assertEqual(self.store.list_gallery()["cards"], [])

    def test_update_without_images_preserves_then_replacement_reorders_and_deletes_omitted(self):
        self.create_collection()
        first_id, _, first_raw = self.add_output("first.png", "PNG", "red")
        second_id, _, _ = self.add_output("second.png", "PNG", "blue")
        card = self.store.create_card(
            "Favorites", title="Before", images=[{"asset_id": first_id}, {"asset_id": second_id}]
        )
        first, second = card["images"]
        first_path = self.store.get_image_file(first["id"])["path"]
        second_path = self.store.get_image_file(second["id"])["path"]

        metadata_only = self.store.update_card(card["id"], title="After", settings={"cfg": 7})
        self.assertEqual([image["id"] for image in metadata_only["images"]], [first["id"], second["id"]])
        reordered = self.store.update_card(
            card["id"],
            images=[{"gallery_image_id": second["id"]}, {"gallery_image_id": first["id"]}],
        )
        self.assertEqual([image["id"] for image in reordered["images"]], [second["id"], first["id"]])
        replacement = self.data_url(self.image_bytes("WEBP", "green"), "image/webp")
        updated = self.store.update_card(
            card["id"],
            images=[
                {"gallery_image_id": first["id"]},
                {"data_url": replacement, "name": "replacement.webp"},
            ],
        )

        self.assertEqual(updated["title"], "After")
        self.assertEqual(updated["settings"], {"cfg": 7})
        self.assertEqual(updated["images"][0]["id"], first["id"])
        self.assertEqual(updated["images"][0]["sort_index"], 0)
        self.assertEqual(updated["images"][1]["name"], "replacement.webp")
        self.assertEqual(first_path.read_bytes(), first_raw)
        self.assertFalse(second_path.exists())

    def test_failed_replacement_preserves_database_and_existing_files(self):
        self.create_collection()
        asset_id, _, _ = self.add_output()
        card = self.store.create_card("Favorites", images=[{"asset_id": asset_id}])
        existing = card["images"][0]
        existing_path = self.store.get_image_file(existing["id"])["path"]
        valid = self.data_url(self.image_bytes("JPEG", "blue"), "image/jpeg")

        with self.assertRaises(GalleryValidationError):
            self.store.update_card(
                card["id"],
                images=[
                    {"data_url": valid, "name": "staged.jpg"},
                    {"gallery_image_id": existing["id"]},
                    {"data_url": "data:image/png;base64,not-base64"},
                ],
            )

        current = self.store.list_gallery()["cards"][0]
        self.assertEqual([image["id"] for image in current["images"]], [existing["id"]])
        self.assertTrue(existing_path.is_file())
        self.assertEqual(list(self.store.assets_directory.glob(".*.tmp")), [])

    def test_manual_data_urls_accept_supported_formats(self):
        self.create_collection()
        formats = [
            ("PNG", "image/png", "a.png"),
            ("JPEG", "image/jpeg", "b.jpg"),
            ("WEBP", "image/webp", "c.webp"),
            ("GIF", "image/gif", "d.gif"),
        ]
        images = [
            {"data_url": self.data_url(self.image_bytes(image_format), mime_type), "name": name}
            for image_format, mime_type, name in formats
        ]
        card = self.store.create_card("Favorites", images=images)

        self.assertEqual([image["mime_type"] for image in card["images"]], [item[1] for item in formats])
        for image in card["images"]:
            with Image.open(self.store.get_image_file(image["id"])["path"]) as saved:
                saved.verify()

    def test_manual_data_url_rejects_invalid_base64_mime_content_and_size(self):
        self.create_collection()
        png = self.image_bytes("PNG")
        invalid = [
            {"data_url": "data:image/png;base64,%%%"},
            {"data_url": self.data_url(png, "image/svg+xml")},
            {"data_url": self.data_url(png, "image/jpeg")},
            {"data_url": self.data_url(b"not an image", "image/png")},
            {"data_url": self.data_url(png, "image/png"), "name": "x" * 256},
        ]
        for image in invalid:
            with self.subTest(image=image["data_url"][:30]):
                with self.assertRaises(GalleryValidationError):
                    self.store.create_card("Favorites", images=[image])
        with patch.object(gallery, "MAX_UPLOAD_BYTES", 4):
            with self.assertRaisesRegex(GalleryValidationError, "50 MiB"):
                self.store.create_card("Favorites", images=[{"data_url": self.data_url(png, "image/png")}])
        self.assertEqual(self.store.list_gallery()["cards"], [])
        self.assertEqual(list(self.store.assets_directory.iterdir()), [])

    def test_delete_card_and_collection_cascade_remove_files(self):
        self.create_collection()
        first_id, _, _ = self.add_output("first.png")
        second_id, _, _ = self.add_output("second.png", color="blue")
        first_card = self.store.create_card("Favorites", images=[{"asset_id": first_id}])
        second_card = self.store.create_card("Favorites", images=[{"asset_id": second_id}])
        first_path = self.store.get_image_file(first_card["images"][0]["id"])["path"]
        second_path = self.store.get_image_file(second_card["images"][0]["id"])["path"]

        self.store.delete_card(first_card["id"])
        self.assertFalse(first_path.exists())
        self.assertTrue(second_path.exists())
        with self.assertRaises(GalleryNotFoundError):
            self.store.delete_card(first_card["id"])

        self.store.delete_collection("Favorites")
        self.assertFalse(second_path.exists())
        self.assertEqual(self.store.list_gallery(), {"collections": [], "cards": []})
        connection = sqlite3.connect(self.store.database_path)
        try:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM card_images").fetchone()[0], 0)
        finally:
            connection.close()

    def test_path_containment_blocks_tampered_storage_name(self):
        self.create_collection()
        asset_id, _, _ = self.add_output()
        card = self.store.create_card("Favorites", images=[{"asset_id": asset_id}])
        image_id = card["images"][0]["id"]
        outside = self.root / "outside.png"
        outside.write_bytes(self.image_bytes())
        connection = sqlite3.connect(self.store.database_path)
        try:
            connection.execute(
                "UPDATE card_images SET storage_name = ? WHERE id = ?", ("../outside.png", image_id)
            )
            connection.commit()
        finally:
            connection.close()

        with self.assertRaises(GalleryStorageError):
            self.store.get_image_file(image_id)
        with self.assertRaises(GalleryStorageError):
            self.store.delete_card(card["id"])
        self.assertTrue(outside.is_file())
        self.assertEqual(self.store.list_gallery()["cards"][0]["id"], card["id"])
        with self.assertRaises(GalleryValidationError):
            self.store.create_card("Favorites", images=[{"asset_id": "../outside.png"}])

    def test_collection_rename_cascades_to_cards(self):
        self.create_collection("Old Name")
        card = self.store.create_card("Old Name")
        updated = self.store.update_collection("Old Name", new_id="新 Name", description=None)

        self.assertEqual(updated["id"], "新 Name")
        self.assertEqual(self.store.list_gallery()["cards"][0]["collection_id"], "新 Name")
        with self.assertRaises(GalleryNotFoundError):
            self.store.update_card(card["id"], collection_id="Old Name")

    def test_prompt_library_crud_persists_both_prompt_directions_and_notes(self):
        created = self.store.create_prompt_entry(
            "  雨夜街景  ",
            "neon street, rain",
            "daylight, low quality",
            "适合赛博朋克场景",
        )
        self.assertEqual(created["title"], "雨夜街景")
        self.assertEqual(created["positive_prompt"], "neon street, rain")
        self.assertEqual(created["negative_prompt"], "daylight, low quality")
        self.assertTrue(created["created_at"].endswith("Z"))

        updated = self.store.update_prompt_entry(created["id"], positive_prompt="wet asphalt", notes=None)
        self.assertEqual(updated["positive_prompt"], "wet asphalt")
        self.assertEqual(updated["negative_prompt"], "daylight, low quality")
        self.assertIsNone(updated["notes"])
        self.assertEqual(self.store.list_prompt_entries(), [updated])

        reopened = GalleryStore(self.root / "state-cache", asset_resolver=self.store.asset_resolver)
        self.assertEqual(reopened.list_prompt_entries()[0]["id"], created["id"])
        reopened.delete_prompt_entry(created["id"])
        self.assertEqual(reopened.list_prompt_entries(), [])
        with self.assertRaises(GalleryNotFoundError):
            reopened.delete_prompt_entry(created["id"])

    def test_prompt_library_rejects_empty_or_oversized_records(self):
        with self.assertRaises(GalleryValidationError):
            self.store.create_prompt_entry("Empty", "  ", "")
        with self.assertRaises(GalleryValidationError):
            self.store.create_prompt_entry("", "prompt")
        with self.assertRaises(GalleryValidationError):
            self.store.create_prompt_entry("Prompt", "x" * 8001)
        created = self.store.create_prompt_entry("Valid", "prompt")
        with self.assertRaises(GalleryValidationError):
            self.store.update_prompt_entry(created["id"], positive_prompt="", negative_prompt="")
        self.assertEqual(self.store.list_prompt_entries()[0]["positive_prompt"], "prompt")


if __name__ == "__main__":
    unittest.main()
