import unittest

from PIL import Image

from backend import usdu_tiles


class UsduTileTests(unittest.TestCase):
    def test_frozen_large_grid_geometry(self):
        plan = usdu_tiles.plan_tiles((2176, 2944), (1088, 1472), 32)
        self.assertEqual((plan.rows, plan.cols, plan.processing_size), (2, 2, (1120, 1504)))
        self.assertEqual([region.mask_rect for region in plan.regions], [
            (0, 0, 1088, 1472), (1088, 0, 2175, 1472),
            (0, 1472, 1088, 2943), (1088, 1472, 2175, 2943),
        ])
        self.assertEqual([region.crop for region in plan.regions], [
            (0, 0, 1120, 1504), (1056, 0, 2176, 1504),
            (0, 1440, 1120, 2944), (1056, 1440, 2176, 2944),
        ])
        self.assertTrue(all(region.mask_semantic == "inclusive" for region in plan.regions))
        self.assertEqual([region.bbox for region in plan.regions], [
            (0, 0, 1088, 1472), (1088, 0, 2176, 1472),
            (0, 1472, 1088, 2944), (1088, 1472, 2176, 2944),
        ])
        self.assertEqual([region.model_size for region in plan.regions], [(1120, 1504)] * 4)
        self.assertEqual([region.paste_origin for region in plan.regions], [(0, 0), (1056, 0), (0, 1440), (1056, 1440)])

    def test_frozen_small_blurred_composite_pixels(self):
        plan = usdu_tiles.plan_tiles((16, 12), (8, 6), 2)
        self.assertEqual(plan.processing_size, (8, 8))
        self.assertEqual([region.crop for region in plan.regions], [
            (0, 0, 10, 10), (6, 0, 16, 10), (0, 2, 10, 12), (6, 2, 16, 12),
        ])
        self.assertEqual([region.mask_rect for region in plan.regions], [
            (0, 0, 8, 6), (8, 0, 15, 6), (0, 6, 8, 11), (8, 6, 15, 11),
        ])
        self.assertEqual([region.bbox for region in plan.regions], [
            (0, 0, 8, 6), (8, 0, 16, 6), (0, 6, 8, 12), (8, 6, 16, 12),
        ])
        self.assertEqual([region.model_size for region in plan.regions], [(8, 8)] * 4)
        self.assertEqual([region.paste_origin for region in plan.regions], [(0, 0), (6, 0), (0, 2), (6, 2)])
        result = usdu_tiles.composite_tiles(Image.new("RGB", (16, 12), "black"), plan,
            [Image.new("RGB", region.model_size, "white") for region in plan.regions], 2)
        self.assertEqual(result.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(result.getpixel((10, 5)), (203, 203, 203))
        self.assertEqual(result.getpixel((10, 6)), (199, 199, 199))
        self.assertEqual(result.getpixel((12, 2)), (251, 251, 251))
        self.assertEqual({point: result.getpixel(point)[0] for point in [
            (0, 0), (15, 0), (0, 11), (15, 11),  # canvas edges
            (8, 3), (3, 6), (8, 6),              # vertical, horizontal, crossing
        ]}, {
            (0, 0): 255, (15, 0): 255, (0, 11): 255, (15, 11): 255,
            (8, 3): 214, (3, 6): 213, (8, 6): 210,
        })

    def test_non_divisible_single_padding_zero_and_resize(self):
        plan = usdu_tiles.plan_tiles((17, 13), (8, 6), 0)
        self.assertEqual((plan.rows, plan.cols), (3, 3))
        self.assertEqual(plan.regions[-1].mask_rect, (16, 12, 16, 12))
        single = usdu_tiles.plan_tiles((5, 5), (8, 8), 0)
        self.assertEqual((single.rows, single.cols, single.regions[0].crop), (1, 1, (0, 0, 5, 5)))
        prepared = usdu_tiles.prepare_tile(Image.new("RGBA", (5, 5), "red"), single.regions[0])
        self.assertEqual((prepared.mode, prepared.size), ("RGB", (8, 8)))
        self.assertEqual(usdu_tiles.restore_tile(prepared, single.regions[0]).size, (5, 5))

    def test_expand_crop_uses_banker_rounding_and_plugin_remainder_order(self):
        plan = usdu_tiles.plan_tiles((16, 21), (8, 7), 0)
        self.assertEqual(plan.processing_size, (8, 8))
        self.assertEqual([region.crop for region in plan.regions], [
            (0, 0, 8, 8), (8, 0, 16, 8),
            (0, 6, 8, 14), (8, 6, 16, 14),
            (0, 13, 8, 21), (8, 13, 16, 21),
        ])
        # 20 / 8 is 2.5: Python's ties-to-even round yields two blocks.
        tie = usdu_tiles.plan_tiles((20, 20), (20, 20), 0)
        self.assertEqual(tie.processing_size, (16, 16))
        edge = usdu_tiles.plan_tiles((17, 15), (8, 7), 0)
        self.assertEqual([region.mask_rect for region in edge.regions], [
            (0, 0, 8, 7), (8, 0, 16, 7), (16, 0, 16, 7),
            (0, 7, 8, 14), (8, 7, 16, 14), (16, 7, 16, 14),
            (0, 14, 8, 14), (8, 14, 16, 14), (16, 14, 16, 14),
        ])

    def test_blur_zero_covers_every_pixel_and_order_is_significant(self):
        plan = usdu_tiles.plan_tiles((16, 12), (8, 6), 2)
        white = [Image.new("RGB", region.model_size, "white") for region in plan.regions]
        covered = usdu_tiles.composite_tiles(Image.new("RGB", plan.canvas, "black"), plan, white, 0)
        self.assertEqual(set(covered.get_flattened_data()), {(255, 255, 255)})
        colors = ["red", "green", "blue", "yellow"]
        tiles = [Image.new("RGB", region.model_size, color) for region, color in zip(plan.regions, colors)]
        forward = usdu_tiles.composite_tiles(Image.new("RGB", plan.canvas, "black"), plan, tiles, 2)
        reverse = usdu_tiles.composite_tiles(Image.new("RGB", plan.canvas, "black"), plan, list(reversed(tiles)), 2)
        self.assertNotEqual(forward.tobytes(), reverse.tobytes())

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError): usdu_tiles.plan_tiles((0, 1), (1, 1))
        with self.assertRaises(ValueError): usdu_tiles.plan_tiles((1, 1), (1, 1), -1)
        with self.assertRaises(ValueError): usdu_tiles.plan_tiles((1, 1), (1, 1), seam_mode="fake")
        plan = usdu_tiles.plan_tiles((8, 8), (8, 8), 0)
        with self.assertRaises(ValueError): usdu_tiles.composite_tiles(Image.new("RGB", (8, 8)), plan, [])
        with self.assertRaises(ValueError): usdu_tiles.TileCompositor(Image.new("RGB", (7, 8)), plan)
        with self.assertRaises(TypeError): usdu_tiles.restore_tile("not an image", plan.regions[0])
        with self.assertRaises(ValueError): usdu_tiles.restore_tile(Image.new("RGBA", (8, 8)), plan.regions[0])
        with self.assertRaises(TypeError): usdu_tiles.TileCompositor("not an image", plan)
        with self.assertRaises(TypeError): usdu_tiles.composite_tiles(Image.new("RGB", (8, 8)), plan, None)
        with self.assertRaises(ValueError): usdu_tiles.prepare_tile(Image.new("RGB", (7, 8)), plan.regions[0])

    def test_compositor_rejects_out_of_order_and_duplicate_regions(self):
        plan = usdu_tiles.plan_tiles((16, 12), (8, 6), 2)
        compositor = usdu_tiles.TileCompositor(Image.new("RGB", plan.canvas), plan, 0)
        with self.assertRaises(ValueError):
            compositor.composite(Image.new("RGB", plan.regions[1].model_size), plan.regions[1])
        compositor.composite(Image.new("RGB", plan.regions[0].model_size), plan.regions[0])
        with self.assertRaises(ValueError):
            compositor.composite(Image.new("RGB", plan.regions[0].model_size), plan.regions[0])


if __name__ == "__main__":
    unittest.main()
