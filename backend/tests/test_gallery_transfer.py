"""Gallery import/export: collections to a folder under the output root plus a ZIP, and back."""
import base64
import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from backend.gallery import GalleryStore
from backend.gallery_transfer import (
    INFO_NAME,
    GalleryTransferConflict,
    GalleryTransferError,
    GalleryTransferNotFound,
    export_collections,
    import_package,
    import_segments,
    write_export_zip,
    zip_source_path,
)


def image_bytes(image_format, color, size=(24, 16)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format=image_format)
    return buffer.getvalue()


def data_url(image_format, color):
    mime = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}[image_format]
    return f"data:{mime};base64,{base64.b64encode(image_bytes(image_format, color)).decode('ascii')}"


class LegacyNameInfo(zipfile.ZipInfo):
    """Writes its name in GBK without the UTF-8 flag, the way older Windows archivers do."""

    def _encodeFilenameFlags(self):
        return self.filename.encode("gbk"), self.flag_bits & ~0x800


class GalleryTransferTests(unittest.TestCase):
    def setUp(self):
        base = os.environ.get("XYZ_EVIDENCE_ROOT")
        if base:
            Path(base).mkdir(parents=True, exist_ok=True)
        self.work = Path(tempfile.mkdtemp(prefix="gallery-transfer-", dir=base))
        self.output = self.work / "outputs"
        self.output.mkdir()
        self.store = GalleryStore(self.work / "state")
        self.store.create_collection("人像 portraits", "旧收藏夹")
        self.store.create_collection("landscapes")
        settings = {"model": "iL", "positive": "a cat <lora:A:0.7>", "loras": [{"value": "style/A.safetensors", "weight": 0.7}], "seed": "42"}
        self.store.create_cards("人像 portraits", [
            {"title": "first", "settings": settings, "images": [{"data_url": data_url("PNG", "red"), "name": "red.png"}, {"data_url": data_url("JPEG", "blue"), "name": "blue.jpg"}]},
            {"title": "text only", "settings": {"positive": "no image"}, "images": []},
            {"title": "third", "settings": {"positive": "webp"}, "images": [{"data_url": data_url("WEBP", "green"), "name": "green.webp"}]},
        ])
        self.store.create_cards("landscapes", [{"title": "hill", "settings": {}, "images": [{"data_url": data_url("PNG", "white"), "name": "hill.png"}]}])

    def cards(self, collection_id):
        return self.store.list_gallery(collection_id)["cards"]

    def original_bytes(self, image):
        return Path(self.store.get_image_file(image["id"])["path"]).read_bytes()

    def export(self, ids=("人像 portraits",)):
        return export_collections(self.store, self.output, list(ids))

    def zip_export(self, name):
        target = self.work / f"{name}.zip"
        write_export_zip(self.output, name, target)
        return target

    def test_an_export_folder_holds_every_image_and_the_info_that_rebuilds_it(self):
        result = self.export(["landscapes", "人像 portraits"])
        folder = self.output / result["folder"]
        self.assertRegex(result["folder"], r"^XiriusA-Gallery-\d{8}-\d{6}$")
        self.assertEqual((result["card_count"], result["image_count"], result["skipped"]), (4, 4, []))
        info = json.loads((folder / INFO_NAME).read_text(encoding="utf-8"))
        self.assertEqual((info["format"], info["version"]), ("xiria-gallery", 1))
        # Gallery order (creation), not the order the ids were sent in.
        self.assertEqual([item["id"] for item in info["collections"]], ["人像 portraits", "landscapes"])
        portraits = info["collections"][0]
        self.assertEqual([card["title"] for card in portraits["cards"]], [card["title"] for card in self.cards("人像 portraits")])
        first = portraits["cards"][0]
        self.assertEqual([image["file"] for image in first["images"]], ["人像 portraits/001-01.png", "人像 portraits/001-02.jpg"])
        self.assertEqual(portraits["cards"][1]["images"], [])
        self.assertNotIn("loraLibraryRevision", first["settings"])
        stored = self.cards("人像 portraits")[0]["images"]
        self.assertEqual((folder / first["images"][0]["file"]).read_bytes(), self.original_bytes(stored[0]))
        self.assertEqual([path.name for path in self.output.iterdir()], [result["folder"]], "no staging folder is left behind")

    def test_a_round_trip_creates_new_collections_and_writes_the_images_under_the_chosen_folder(self):
        result = self.export(["人像 portraits", "landscapes"])
        package = self.zip_export(result["folder"])
        with zipfile.ZipFile(package) as archive:
            self.assertIn(INFO_NAME, archive.namelist())
        imported = import_package(self.store, self.output, package, "imports/batch one")
        self.assertEqual(imported["folder"], "imports/batch one")
        self.assertEqual([(item["id"], item["requested_id"], item["renamed"]) for item in imported["collections"]],
                         [("人像 portraits (2)", "人像 portraits", True), ("landscapes (2)", "landscapes", True)])
        self.assertEqual((imported["card_count"], imported["image_count"]), (4, 4))
        source, copy = self.cards("人像 portraits"), self.cards("人像 portraits (2)")
        self.assertEqual([card["title"] for card in copy], [card["title"] for card in source])
        self.assertEqual(copy[0]["settings"], source[0]["settings"])
        self.assertEqual([image["mime_type"] for image in copy[0]["images"]], ["image/png", "image/jpeg"])
        self.assertEqual([image["name"] for image in copy[0]["images"]], ["red.png", "blue.jpg"])
        for before, after in zip(source[0]["images"], copy[0]["images"]):
            self.assertEqual(self.original_bytes(after), self.original_bytes(before))
        description = {item["id"]: item["description"] for item in self.store.list_gallery()["collections"]}
        self.assertEqual(description["人像 portraits (2)"], "旧收藏夹")
        folder = self.output / "imports" / "batch one"
        self.assertEqual((folder / "人像 portraits" / "001-02.jpg").read_bytes(), self.original_bytes(copy[0]["images"][1]))
        record = json.loads((folder / INFO_NAME).read_text(encoding="utf-8"))
        self.assertEqual(record["source"], package.name)
        self.assertEqual(record["collections"][1]["cards"][0]["images"][0]["file"], "landscapes/001-01.png")
        self.assertEqual(sorted(path.name for path in (self.output / "imports").iterdir()), ["batch one"])

    def test_a_blank_folder_is_named_after_the_current_time(self):
        package = self.zip_export(self.export(["landscapes"])["folder"])
        imported = import_package(self.store, self.output, package, "  ")
        self.assertRegex(imported["folder"], r"^\d{8}-\d{6}$")
        self.assertTrue((self.output / imported["folder"] / "landscapes" / "001-01.png").is_file())
        from datetime import datetime
        moment = datetime(2026, 10, 12, 22, 15, 30)
        first = import_package(self.store, self.output, package, None, moment=moment)
        second = import_package(self.store, self.output, package, "", moment=moment)
        self.assertEqual((first["folder"], second["folder"]), ("20261012-221530", "20261012-221530-2"))

    def test_a_folder_zipped_by_hand_imports_from_its_single_top_level_folder_in_a_legacy_encoding(self):
        result = self.export(["人像 portraits"])
        folder = self.output / result["folder"]
        package = self.work / "by-hand.zip"
        with zipfile.ZipFile(package, "w") as archive:
            for path in sorted(folder.rglob("*")):
                if path.is_file():
                    name = f"{result['folder']}/{path.relative_to(folder).as_posix()}"
                    archive.writestr(LegacyNameInfo(name), path.read_bytes())
            archive.writestr("__MACOSX/._gallery-info.json", b"resource fork")
        with zipfile.ZipFile(package) as archive:
            legacy = [item.filename for item in archive.infolist() if "portraits" in item.filename]
            self.assertTrue(legacy and all("\u4eba\u50cf" not in name for name in legacy), "the fixture must really carry GBK names read back as CP437")
        imported = import_package(self.store, self.output, package, "by-hand")
        self.assertEqual(imported["collections"][0]["id"], "人像 portraits (2)")
        self.assertEqual(imported["image_count"], 3)

    def test_a_refused_package_leaves_no_folder_and_no_collection(self):
        before = self.store.list_gallery()
        info = {"format": "xiria-gallery", "version": 1, "collections": [{"id": "bad", "cards": [{"title": "x", "settings": {}, "images": [{"file": "bad/001-01.png"}]}]}]}
        cases = {
            "traversal": ({**info, "collections": [{"id": "bad", "cards": [{"settings": {}, "images": [{"file": "../escape.png"}]}]}]}, {"../escape.png": image_bytes("PNG", "red")}),
            "missing image": (info, {}),
            "not an image": (info, {"bad/001-01.png": b"plain text, not a picture"}),
            "wrong format": ({**info, "format": "other"}, {"bad/001-01.png": image_bytes("PNG", "red")}),
            "bad collection id": ({**info, "collections": [{"id": "a/b", "cards": []}]}, {}),
        }
        for label, (document, files) in cases.items():
            with self.subTest(label):
                package = self.work / f"{label}.zip"
                with zipfile.ZipFile(package, "w") as archive:
                    archive.writestr(INFO_NAME, json.dumps(document))
                    for name, content in files.items():
                        archive.writestr(name, content)
                with self.assertRaises(GalleryTransferError):
                    import_package(self.store, self.output, package, f"case-{len(label)}")
                self.assertEqual([path.name for path in self.output.iterdir()], [], "nothing is left under the output root")
        self.assertEqual(self.store.list_gallery(), before)
        self.assertFalse((self.output.parent / "escape.png").exists())

    def test_a_gallery_failure_after_extraction_removes_the_written_folder(self):
        package = self.zip_export(self.export(["landscapes"])["folder"])
        original = self.store.import_collections

        def fail(plan):
            self.assertTrue((self.output / "late" / "landscapes" / "001-01.png").is_file())
            raise RuntimeError("database refused")

        self.store.import_collections = fail
        try:
            with self.assertRaises(RuntimeError):
                import_package(self.store, self.output, package, "late")
        finally:
            self.store.import_collections = original
        self.assertFalse((self.output / "late").exists())

    def test_destinations_must_stay_inside_the_output_root_and_be_new_or_empty(self):
        package = self.zip_export(self.export(["landscapes"])["folder"])
        for folder in ("C:/elsewhere", "/abs", "../up", "a/../b", "CON", "trail.", "a" * 97):
            with self.subTest(folder), self.assertRaises(GalleryTransferError):
                import_segments(folder)
        (self.output / "taken").mkdir()
        (self.output / "taken" / "note.txt").write_text("user file", encoding="utf-8")
        with self.assertRaises(GalleryTransferConflict):
            import_package(self.store, self.output, package, "taken")
        self.assertEqual([path.name for path in (self.output / "taken").iterdir()], ["note.txt"])
        (self.output / "empty").mkdir()
        self.assertEqual(import_package(self.store, self.output, package, "empty")["image_count"], 1)

    def test_a_typed_path_must_name_an_existing_zip_file(self):
        package = self.zip_export(self.export(["landscapes"])["folder"])
        self.assertEqual(zip_source_path(f'  "{package}"  '), package)
        with self.assertRaises(GalleryTransferError):
            zip_source_path("relative/package.zip")
        with self.assertRaises(GalleryTransferError):
            zip_source_path(str(package.parent))
        with self.assertRaises(GalleryTransferNotFound):
            zip_source_path(str(package.with_name("absent.zip")))
        renamed = package.with_suffix(".bin")
        renamed.write_bytes(package.read_bytes())
        with self.assertRaises(GalleryTransferError):
            zip_source_path(str(renamed))

    def test_imported_settings_carry_the_current_lora_library_revision(self):
        package = self.zip_export(self.export(["人像 portraits"])["folder"])
        manifest = self.work / "state" / "lora-library" / "publication.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({"schemaVersion": 1, "phase": "published", "revision": 7}), encoding="utf-8")
        imported = import_package(self.store, self.output, package, "rev")
        card = self.cards(imported["collections"][0]["id"])[0]
        self.assertEqual(card["settings"]["loraLibraryRevision"], 7)
        connection = self.store._connect()
        try:
            stored = json.loads(connection.execute("SELECT settings_json FROM cards WHERE id = ?", (card["id"],)).fetchone()[0])
        finally:
            connection.close()
        self.assertEqual(stored["loraLibraryRevision"], 7)

    def test_only_export_folders_are_served_as_zip(self):
        (self.output / "XiriusA-Gallery-20260101-000000").mkdir()
        for name in ("../state", "imports", "XiriusA-Gallery-20260101-000000", "XiriusA-Gallery-x"):
            with self.subTest(name), self.assertRaises(GalleryTransferNotFound):
                write_export_zip(self.output, name, io.BytesIO())
        with self.assertRaises(GalleryTransferNotFound):
            export_collections(self.store, self.output, ["absent"])


class GalleryTransferHTTPTests(unittest.TestCase):
    """The actual service routes on a private app, private state and output roots."""

    @classmethod
    def setUpClass(cls):
        from backend import inference_server
        cls.server = inference_server

    def setUp(self):
        from unittest import mock
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        self.fixture = GalleryTransferTests(methodName="cards")
        self.fixture.setUp()
        self.patches = [mock.patch.object(self.server, "OUTPUT_DIRECTORY", self.fixture.output),
                        mock.patch.object(self.server, "STATE_DIRECTORY", self.fixture.work / "state")]
        for patch in self.patches:
            patch.start()
        app = FastAPI()
        app.router.routes.extend(route for route in self.server.app.routes if getattr(route, "path", "").startswith("/api/inference/gallery"))
        self.app = app
        self.client = TestClient(app, client=("127.0.0.1", 50000), base_url="http://127.0.0.1:9017")
        self.local = {"origin": "http://127.0.0.1:9017"}

    def tearDown(self):
        self.client.close()
        for patch in reversed(self.patches):
            patch.stop()

    def test_export_download_and_both_import_routes(self):
        from fastapi.testclient import TestClient
        from unittest import mock
        route = "/api/inference/gallery/transfer"
        for headers in ({}, {"origin": "http://evil.test:9017"}):
            self.assertEqual(self.client.post(f"{route}/export", headers=headers, json={"collection_ids": ["landscapes"]}).status_code, 403)
        with TestClient(self.app, client=("192.168.2.4", 50000), base_url="http://127.0.0.1:9017") as remote:
            self.assertEqual(remote.post(f"{route}/export", headers=self.local, json={"collection_ids": ["landscapes"]}).status_code, 403)
        exported = self.client.post(f"{route}/export", headers=self.local, json={"collection_ids": ["landscapes"]})
        self.assertEqual(exported.status_code, 201, exported.text)
        receipt = exported.json()
        self.assertEqual(receipt["zip_url"], f"{route}/exports/{receipt['folder']}.zip")
        download = self.client.get(receipt["zip_url"])
        self.assertEqual(download.status_code, 200, download.text)
        self.assertEqual(download.headers["content-type"], "application/zip")
        self.assertIn(f'{receipt["folder"]}.zip', download.headers["content-disposition"])
        with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
            self.assertEqual(sorted(archive.namelist()), [INFO_NAME, "landscapes/001-01.png"])
        self.assertEqual(list((self.fixture.work / "state" / "gallery-transfer").iterdir()), [], "the ZIP scratch file is removed after sending")
        self.assertEqual(self.client.get(f"{route}/exports/..%2Fstate.zip").status_code, 404)

        package = self.fixture.work / "download.zip"
        package.write_bytes(download.content)
        by_path = self.client.post(f"{route}/import", headers=self.local, json={"zip_path": f'"{package}"', "folder": "by-path"})
        self.assertEqual(by_path.status_code, 201, by_path.text)
        self.assertEqual(by_path.json()["collections"][0]["id"], "landscapes (2)")
        with mock.patch.dict(os.environ, {"XIRAI_WD14_PROXY_PROOF": "private-proof"}):
            # A remote page through Node: uploads are allowed, reading a path on this machine is not.
            remote = {"origin": "http://192.168.2.9:7709", "x-xirai-wd14-proxy-proof": "private-proof"}
            self.assertEqual(self.client.post(f"{route}/import", headers=remote, json={"zip_path": str(package)}).status_code, 403)
            uploaded = self.client.post(f"{route}/import-upload?folder=uploaded&name=download.zip", headers={**remote, "content-type": "application/zip"}, content=download.content)
            self.assertEqual(uploaded.status_code, 201, uploaded.text)
        self.assertEqual(uploaded.json()["collections"][0]["id"], "landscapes (3)")
        self.assertEqual(json.loads((self.fixture.output / "uploaded" / INFO_NAME).read_text(encoding="utf-8"))["source"], "download.zip")
        self.assertEqual(self.client.post(f"{route}/import-upload", headers=self.local, content=b"").status_code, 422)
        self.assertEqual(self.client.post(f"{route}/import-upload?folder=uploaded", headers=self.local, content=download.content).status_code, 409)
        self.assertEqual(self.client.post(f"{route}/import", headers=self.local, json={"zip_path": str(package.with_suffix(".png"))}).status_code, 422)
        self.assertEqual(list((self.fixture.work / "state" / "gallery-transfer").iterdir()), [], "uploads are removed after import")


if __name__ == "__main__":
    unittest.main()
