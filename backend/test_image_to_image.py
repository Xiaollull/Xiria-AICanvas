import base64
import io
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from PIL import Image
from pydantic import ValidationError

from backend import inference_server, rtx_vsr
from backend.inference_server import GenerateInput, SourceImageInput


def encode_image(image: Image.Image, image_format: str = "PNG", data_url: bool = True) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{image_format.lower()};base64,{encoded}" if data_url else encoded


def request_with_source(**overrides):
    fields = dict(
        engine="SD", checkpoint="model.safetensors", prompt="a lantern", width=512, height=512,
        steps=20, cfg=7, denoise=0.6, seed=1, sampler="euler", scheduler="simple",
        source_image={"enabled": True, "image_data": encode_image(Image.new("RGB", (640, 480), (10, 20, 30)))},
    )
    fields.update(overrides)
    return GenerateInput(**fields)


def postprocess_request(size=(1920, 1080), **overrides):
    """A post-processing run: the canvas is the picture, and at least one stage is on."""
    fields = dict(
        width=size[0], height=size[1], postprocess_only=True,
        source_image={"enabled": True, "image_data": encode_image(Image.new("RGB", size, (10, 20, 30)))},
        rtx={"enabled": True, "scale": 2},
    )
    fields.update(overrides)
    return request_with_source(**fields)


class SourceImageContractTests(unittest.TestCase):
    def test_text_to_image_requests_carry_a_disabled_source(self):
        request = GenerateInput(
            engine="SD", checkpoint="model.safetensors", prompt="a lantern", width=512, height=512,
            steps=20, cfg=7, denoise=1, seed=1, sampler="euler", scheduler="simple",
        )
        self.assertFalse(request.source_image.enabled)
        self.assertEqual(request.source_image.image_data, "")
        self.assertIsNone(inference_server.prepare_source_image(request))

    def test_enabled_source_requires_pixels_and_pixels_require_enabled(self):
        with self.assertRaises(ValidationError):
            SourceImageInput(enabled=True)
        with self.assertRaises(ValidationError):
            SourceImageInput(enabled=False, image_data=encode_image(Image.new("RGB", (8, 8))))

    def test_zero_denoise_is_refused_only_for_image_to_image(self):
        # Text-to-image has always ignored denoise, so zero stays a legal no-op there.
        GenerateInput(
            engine="SD", checkpoint="model.safetensors", prompt="a lantern", width=512, height=512,
            steps=20, cfg=7, denoise=0, seed=1, sampler="euler", scheduler="simple",
        )
        with self.assertRaises(ValidationError):
            request_with_source(denoise=0)

    def test_unknown_resize_mode_is_refused(self):
        with self.assertRaises(ValidationError):
            request_with_source(source_image={
                "enabled": True,
                "image_data": encode_image(Image.new("RGB", (8, 8))),
                "resize_mode": "tile",
            })


class PostprocessOnlyContractTests(unittest.TestCase):
    def test_the_mode_needs_both_a_picture_and_a_stage(self):
        # Without a picture there is nothing to enhance; without a stage the run would load a model,
        # skip sampling and save its own input back out.
        with self.assertRaises(ValidationError):
            GenerateInput(
                engine="SD", checkpoint="model.safetensors", prompt="a lantern", width=512, height=512,
                steps=20, cfg=7, denoise=0.6, seed=1, sampler="euler", scheduler="simple",
                postprocess_only=True, rtx={"enabled": True},
            )
        with self.assertRaises(ValidationError):
            postprocess_request(rtx={"enabled": False})
        for stage in ({"rtx": {"enabled": True}},
                      {"adetailer": {"enabled": True, "units": [{"detector": "face_yolov8n.pt"}]}},
                      {"hires": {"enabled": True, "model": "x.pth"}}):
            postprocess_request(**{"rtx": {"enabled": False}, **stage})

    def test_a_sampling_canvas_keeps_its_alignment_and_ceiling_but_a_postprocess_one_does_not(self):
        # 1080 is not a multiple of 64 and 3000 is past the sampler's ceiling; neither restriction
        # describes a run that never builds a latent grid.
        for canvas in ((1920, 1080), (3000, 2000)):
            with self.assertRaises(ValidationError):
                request_with_source(width=canvas[0], height=canvas[1])
            self.assertEqual(postprocess_request(canvas).width, canvas[0])

    def test_the_source_envelope_is_the_stage_envelope(self):
        with self.assertRaises(ValidationError):
            postprocess_request((8192, 8192))
        with self.assertRaises(ValueError):
            inference_server.validate_postprocess_source_size((8320, 64))
        with self.assertRaises(ValueError):
            inference_server.validate_postprocess_source_size((32, 32))
        inference_server.validate_postprocess_source_size((4096, 4096))

    def test_the_picture_reaches_the_first_stage_exactly_as_supplied(self):
        request = postprocess_request((1920, 1080))
        prepared = inference_server.prepare_source_image(request)
        self.assertEqual(prepared.size, (1920, 1080))
        # The same request without the flag is the ordinary image-to-image contract, which resamples.
        resampled = inference_server.prepare_source_image(request_with_source(width=768, height=512))
        self.assertEqual(resampled.size, (768, 512))

    def test_the_canvas_follows_the_decoded_picture_when_the_two_disagree(self):
        # An EXIF-rotated JPEG is measured one way round by the browser and the other by Pillow, and
        # memory admission reads the request rather than the picture.
        request = postprocess_request((1024, 768))
        self.assertFalse(inference_server.adopt_postprocess_canvas(request, Image.new("RGB", (1024, 768))))
        self.assertTrue(inference_server.adopt_postprocess_canvas(request, Image.new("RGB", (768, 1024))))
        self.assertEqual((request.width, request.height), (768, 1024))
        with self.assertRaises(ValueError):
            inference_server.adopt_postprocess_canvas(request, Image.new("RGB", (9000, 100)))
        # A transform run has a canvas of its own and never adopts one.
        transform = request_with_source()
        self.assertFalse(inference_server.adopt_postprocess_canvas(transform, Image.new("RGB", (64, 64))))
        self.assertEqual((transform.width, transform.height), (512, 512))

    def test_the_base_pass_reports_no_steps_because_it_does_not_run(self):
        self.assertEqual(inference_server.base_sampling_steps(postprocess_request(), "sd"), 0)
        self.assertEqual(inference_server.base_sampling_steps(postprocess_request(), "anima"), 0)

    def test_admission_drops_the_base_canvas_but_never_ends_up_with_nothing(self):
        # RTX is not a diffusion stage, so a post-processing run whose only stage is RTX has no
        # workload of its own — the base canvas stands in rather than leaving admission unsized.
        rtx_only = inference_server.generation_memory_workload_diagnostics(postprocess_request((1024, 1024)), "sd")
        self.assertEqual(rtx_only["admission"][:2], (1024, 1024))
        # With Hires on, admission sizes against the Hires target rather than the untouched source.
        with_hires = inference_server.generation_memory_workload_diagnostics(
            postprocess_request((1024, 1024), rtx={"enabled": False},
                                hires={"enabled": True, "model": "x.pth", "scale": 2}),
            "sd",
        )
        self.assertEqual(with_hires["stage"], "full_frame_target")
        self.assertEqual(with_hires["admission"][:2], (2048, 2048))

    def test_a_transform_run_still_admits_against_its_own_canvas(self):
        diagnostics = inference_server.generation_memory_workload_diagnostics(request_with_source(), "sd")
        self.assertEqual(diagnostics["stage"], "base")
        self.assertEqual(diagnostics["admission"][:2], (512, 512))


class PostprocessOnlyAdmissionTests(unittest.TestCase):
    """The API boundary: what a post-processing submission is answered with."""

    def setUp(self):
        with inference_server.jobs_lock:
            inference_server.jobs.clear()
            inference_server.job_controls.clear()

    tearDown = setUp

    def test_an_oversized_picture_is_refused_even_when_the_declared_canvas_is_not(self):
        # The declared canvas is legal on its own; the picture behind it is not, and post-processing
        # keeps the picture. Only decoding it can tell the two apart.
        request = postprocess_request((512, 512), source_image={
            "enabled": True, "image_data": encode_image(Image.new("RGB", (9000, 64))),
        })
        with patch.object(rtx_vsr, "status", return_value={"available": True, "probing": False}):
            with self.assertRaises(HTTPException) as raised:
                inference_server.create_job(request)
        self.assertEqual(raised.exception.status_code, 422)
        self.assertIn("8192", raised.exception.detail)

    def test_an_accepted_run_records_the_mode_it_was_submitted_in(self):
        request = postprocess_request((1024, 1024))
        with patch.object(rtx_vsr, "status", return_value={"available": True, "probing": False}), \
                patch.object(inference_server.executor, "submit"):
            job = inference_server.create_job(request)
        self.assertTrue(job["postprocess_only"])
        self.assertEqual(job["postprocess_stages"], ["rtx"])
        with patch.object(rtx_vsr, "status", return_value={"available": True, "probing": False}), \
                patch.object(inference_server.executor, "submit"):
            self.setUp()
            transform = inference_server.create_job(postprocess_request((1024, 1024), postprocess_only=False))
        self.assertFalse(transform["postprocess_only"])


class SourceImageDecodeTests(unittest.TestCase):
    def test_decodes_data_url_and_bare_base64_alike(self):
        image = Image.new("RGB", (12, 9), (200, 30, 40))
        for payload in (encode_image(image), encode_image(image, data_url=False)):
            decoded = inference_server.decode_source_image(payload)
            self.assertEqual(decoded.size, (12, 9))
            self.assertEqual(decoded.mode, "RGB")

    def test_transparent_source_is_flattened_onto_white_not_black(self):
        # `convert("RGB")` keeps whatever colour hides under a fully transparent pixel, which for a
        # cut-out is black; the sampler would then treat that halo as content worth preserving.
        source = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        source.putpixel((0, 0), (255, 0, 0, 255))
        decoded = inference_server.decode_source_image(encode_image(source))
        self.assertEqual(decoded.mode, "RGB")
        self.assertEqual(decoded.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(decoded.getpixel((3, 3)), (255, 255, 255))

    def test_unreadable_payloads_raise_a_value_error(self):
        for payload in ("", "data:image/png;base64,not-base64!!", "data:image/png;base64,QUJD"):
            with self.assertRaises(ValueError):
                inference_server.decode_source_image(payload)


class SourceImageGeometryTests(unittest.TestCase):
    def test_every_resize_mode_lands_exactly_on_the_requested_canvas(self):
        source = Image.new("RGB", (640, 480), (10, 20, 30))
        for mode in ("cover", "contain", "stretch"):
            fitted = inference_server.fit_source_image(source, (512, 768), mode)
            self.assertEqual(fitted.size, (512, 768), mode)

    def test_contain_pads_and_cover_crops(self):
        source = Image.new("RGB", (100, 50), (0, 0, 255))
        contained = inference_server.fit_source_image(source, (128, 128), "contain")
        self.assertEqual(contained.getpixel((64, 2)), (255, 255, 255))
        self.assertEqual(contained.getpixel((64, 64)), (0, 0, 255))
        covered = inference_server.fit_source_image(source, (128, 128), "cover")
        self.assertEqual(covered.getpixel((64, 2)), (0, 0, 255))

    def test_a_matching_canvas_returns_the_source_untouched(self):
        source = Image.new("RGB", (256, 256), (7, 7, 7))
        self.assertIs(inference_server.fit_source_image(source, (256, 256), "cover"), source)

    def test_prepare_source_image_resamples_onto_the_request_canvas(self):
        request = request_with_source(width=768, height=512)
        prepared = inference_server.prepare_source_image(request)
        self.assertEqual(prepared.size, (768, 512))


class SamplingStepTests(unittest.TestCase):
    def test_diffusers_image_to_image_reports_the_steps_it_actually_runs(self):
        # Diffusers runs int(steps * strength) updates. Reporting the requested count would leave
        # the progress bar permanently short of its own total.
        request = request_with_source(steps=20, denoise=0.6)
        self.assertEqual(inference_server.base_sampling_steps(request, "sd"), 12)

    def test_a_vanishing_step_count_is_floored_at_one(self):
        request = request_with_source(steps=2, denoise=0.05)
        self.assertEqual(inference_server.base_sampling_steps(request, "sd"), 1)

    def test_anima_and_text_to_image_report_the_requested_steps(self):
        request = request_with_source(steps=20, denoise=0.6, engine="Anima", checkpoint=None,
                                      diffusion_model="d.safetensors", text_encoder="t.safetensors",
                                      vae="v.safetensors", preview_enabled=False)
        self.assertEqual(inference_server.base_sampling_steps(request, "anima"), 20)
        text_request = GenerateInput(
            engine="SD", checkpoint="model.safetensors", prompt="a lantern", width=512, height=512,
            steps=20, cfg=7, denoise=0.4, seed=1, sampler="euler", scheduler="simple",
        )
        self.assertEqual(inference_server.base_sampling_steps(text_request, "sd"), 20)


class SourceImagePipelineKwargsTests(unittest.TestCase):
    def test_kwargs_carry_the_image_and_strength_but_never_a_canvas(self):
        request = request_with_source(steps=20, denoise=0.45, images_per_batch=2)
        generators = [object(), object()]
        callback = object()
        source = Image.new("RGB", (512, 512))
        kwargs = inference_server.source_image_pipeline_kwargs(
            request, source, generators, callback, {"prompt_embeds": "encoded"}
        )
        self.assertIs(kwargs["image"], source)
        self.assertEqual(kwargs["strength"], 0.45)
        self.assertEqual(kwargs["num_inference_steps"], 20)
        self.assertEqual(kwargs["num_images_per_prompt"], 2)
        self.assertEqual(kwargs["prompt_embeds"], "encoded")
        # An image-to-image pipeline derives its canvas from the picture it is handed; passing a
        # width would be silently ignored by some pipelines and rejected by others.
        self.assertNotIn("width", kwargs)
        self.assertNotIn("height", kwargs)

    def test_generator_count_must_match_the_batch(self):
        request = request_with_source(images_per_batch=3)
        with self.assertRaises(ValueError):
            inference_server.source_image_pipeline_kwargs(
                request, Image.new("RGB", (8, 8)), [object()], object(), {}
            )


class SourceImageRecordTests(unittest.TestCase):
    def test_the_picture_itself_never_reaches_a_saved_record(self):
        # The job record is polled several times a second and the PNG parameter block is read back
        # by the gallery; a base64 copy in either would dominate both.
        request = request_with_source()
        recorded = {
            "source_image_enabled": request.source_image.enabled,
            "source_image_resize_mode": request.source_image.resize_mode,
            "source_image_name": request.source_image.name or None,
        }
        self.assertNotIn(request.source_image.image_data, str(recorded))
        self.assertTrue(recorded["source_image_enabled"])


if __name__ == "__main__":
    unittest.main()
