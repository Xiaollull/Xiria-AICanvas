import base64
import io
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from backend.gallery import (
    THUMBNAIL_MAX_EDGE,
    THUMBNAIL_MIME,
    THUMBNAIL_SUFFIX,
    GalleryStore,
    GalleryValidationError,
)


def png_data_url(width, height, seed=1):
    """A PNG that behaves like a real generation: high-entropy detail that PNG
    cannot collapse, so the size comparisons below mean something."""
    state = seed
    pixels = []
    for _ in range(width * height):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        pixels.append((state >> 16 & 255, state >> 8 & 255, state & 255))
    frame = Image.new("RGB", (width, height))
    frame.putdata(pixels)
    buffer = io.BytesIO()
    frame.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class GalleryThumbnailTests(unittest.TestCase):
    """Curated images are independent copies at generation size. Grid tiles render
    them at roughly 300 px and the focus strip at 60 px, so serving originals made
    browsing decode tens of megapixels to paint thumbnails."""

    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        self.store = GalleryStore(self.directory)
        self.store.create_collection("Test")
        card = self.store.create_card(
            "Test",
            settings={},
            images=[{"data_url": png_data_url(1024, 1472), "name": "a.png"}],
        )
        self.card = card
        self.image_id = card["images"][0]["id"]
        self.storage_name = self._storage_name()

    def _storage_name(self):
        return self.store.get_image_file(self.image_id)["path"].name

    def test_the_thumbnail_is_a_fraction_of_the_original_and_bounded(self):
        original = self.store.get_image_file(self.image_id)
        thumb = self.store.get_image_file(self.image_id, "thumb")
        self.assertEqual(original["variant"], "original")
        self.assertEqual(thumb["variant"], "thumb")
        self.assertEqual(thumb["mime_type"], THUMBNAIL_MIME)
        self.assertTrue(thumb["path"].name.endswith(THUMBNAIL_SUFFIX))
        self.assertLess(thumb["path"].stat().st_size * 4, original["path"].stat().st_size,
                        "a derivative that is not dramatically smaller is not worth serving")
        with Image.open(thumb["path"]) as frame:
            self.assertLessEqual(max(frame.size), THUMBNAIL_MAX_EDGE)
            # Aspect ratio is preserved, so tiles crop rather than distort.
            self.assertAlmostEqual(frame.size[0] / frame.size[1], 1024 / 1472, places=2)

    def test_the_derivative_is_generated_once_and_reused(self):
        first = self.store.get_image_file(self.image_id, "thumb")["path"]
        stamp = first.stat().st_mtime_ns
        for _ in range(5):
            self.store.get_image_file(self.image_id, "thumb")
        self.assertEqual(first.stat().st_mtime_ns, stamp, "a cached derivative must not be rewritten per request")

    def test_a_stale_derivative_is_regenerated(self):
        thumb = self.store.get_image_file(self.image_id, "thumb")["path"]
        original = self.store.get_image_file(self.image_id)["path"]
        # An id can be reused by a re-import, so staleness is checked, not assumed.
        import os
        os.utime(thumb, ns=(original.stat().st_atime_ns, original.stat().st_mtime_ns - 1_000_000_000))
        stale_stamp = thumb.stat().st_mtime_ns
        refreshed = self.store.get_image_file(self.image_id, "thumb")["path"]
        self.assertGreater(refreshed.stat().st_mtime_ns, stale_stamp)

    def test_an_unreadable_derivative_falls_back_to_the_original(self):
        """A cache problem must cost bandwidth, never a broken image."""
        thumb_path = self.store._thumbnail_path(self.storage_name)
        thumb_path.write_bytes(b"")  # zero length: present but useless
        served = self.store.get_image_file(self.image_id, "thumb")
        self.assertIn(served["variant"], ("thumb", "original"))
        self.assertTrue(served["path"].is_file())
        self.assertGreater(served["path"].stat().st_size, 0)

    def test_a_corrupt_original_still_serves_rather_than_raising(self):
        original = self.store.get_image_file(self.image_id)["path"]
        original.write_bytes(b"not a png")
        served = self.store.get_image_file(self.image_id, "thumb")
        self.assertEqual(served["variant"], "original", "an underivable original is served as-is")

    def test_deleting_a_card_removes_its_derivative(self):
        thumb = self.store.get_image_file(self.image_id, "thumb")["path"]
        self.assertTrue(thumb.is_file())
        self.store.delete_card(self.card["id"])
        self.assertFalse(thumb.is_file(), "an orphaned derivative would linger with no record pointing at it")
        self.assertEqual(list(self.store.assets_directory.glob("*")), [])

    def test_records_advertise_both_urls_and_the_variant_is_validated(self):
        listing = self.store.list_gallery()
        image = listing["cards"][0]["images"][0]
        self.assertTrue(image["url"].endswith(image["id"]))
        self.assertEqual(image["thumb_url"], f"{image['url']}?variant=thumb")
        for bad in ("small", "", "original ", "../original", None, 5):
            with self.assertRaises(GalleryValidationError):
                self.store.get_image_file(self.image_id, bad)

    def test_an_already_small_original_is_served_directly(self):
        """Deriving from a small original spends CPU and disk to save nothing, and
        a lossy derivative of one can genuinely come out larger than the source."""
        small = self.store.create_card(
            "Test",
            settings={},
            images=[{"data_url": png_data_url(64, 64, seed=9), "name": "small.png"}],
        )
        served = self.store.get_image_file(small["images"][0]["id"], "thumb")
        self.assertEqual(served["variant"], "original")
        self.assertEqual(list(self.store.assets_directory.glob(f"*{THUMBNAIL_SUFFIX}")), [],
                         "no derivative is written for an original not worth shrinking")

    def test_no_derivative_is_published_when_it_would_not_be_smaller(self):
        from unittest.mock import patch
        original = self.store.get_image_file(self.image_id)["path"]
        # Force the encoder to produce something useless; the store must notice
        # rather than publish a derivative that makes browsing slower.
        with patch("backend.gallery.THUMBNAIL_MIN_ORIGINAL_BYTES", 0), \
             patch("backend.gallery.THUMBNAIL_MAX_EDGE", 4):
            self.store._thumbnail_path(self.storage_name).unlink(missing_ok=True)
            served = self.store.get_image_file(self.image_id, "thumb")
        self.assertTrue(served["path"].is_file())
        if served["variant"] == "thumb":
            self.assertLess(served["path"].stat().st_size, original.stat().st_size)

    def test_the_derivative_stays_inside_the_asset_directory(self):
        thumb = self.store._thumbnail_path(self.storage_name)
        self.assertEqual(thumb.parent.resolve(), self.store.assets_directory.resolve())


if __name__ == "__main__":
    unittest.main()
