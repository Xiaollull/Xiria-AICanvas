"""P9: the built-in Triton kernels reproduce the PyTorch port bit for bit, and step aside safely."""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from .. import comfy_quant as cq
    from .. import comfy_quant_triton as kt
except ImportError:
    import comfy_quant as cq
    import comfy_quant_triton as kt

GPU = (
    torch.cuda.is_available()
    and kt.triton is not None
    and tuple(torch.cuda.get_device_capability()) >= kt.MIN_CAPABILITY
)


def port(fn, *args, **kwargs):
    with cq.backend_override("port"):
        return fn(*args, **kwargs)


def weights(n, k, device, seed=0):
    generator = torch.Generator().manual_seed(seed)
    w = (torch.randn(n, k, generator=generator) * 0.05).to(device)
    rotated = cq.rotate_weight(w, cq.build_hadamard(256, device=device), 256)
    q8, s8 = cq.quantize_int8_rowwise(rotated)
    q4, s4 = cq.quantize_int4_rowwise(rotated)
    return q8, s8, q4, s4


def activations(rows, k, dtype, device, seed=1):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, k, generator=generator) * 1.5
    x[:, ::53] *= 25  # the outlier channels ConvRot exists for
    return x.to(device=device, dtype=dtype)


@unittest.skipUnless(GPU, "needs a CUDA device of SM 8.0+ and Triton")
class KernelTests(unittest.TestCase):
    device = torch.device("cuda")

    def test_int8_layouts_match_the_port_bit_for_bit(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for rows in (1, 5, 33, 257):
                x = activations(rows, 512, dtype, self.device)
                q8, s8, _, _ = weights(72, 512, self.device)
                bias = torch.randn(72, device=self.device).to(dtype)
                scalar = torch.tensor(0.013, device=self.device)
                for label, args in (
                    ("convrot", (x, q8, s8, bias, dtype, True, 256)),
                    ("per-row", (x, q8, s8, None, dtype)),
                    ("scalar", (x, q8, scalar, bias, dtype)),
                ):
                    with self.subTest(dtype=dtype, rows=rows, layout=label):
                        self.assertTrue(torch.equal(port(cq.int8_linear, *args), kt.int8_linear(*args)))

    def test_plain_int8_takes_any_depth(self):
        x = activations(9, 1000, torch.bfloat16, self.device)
        q8, s8 = cq.quantize_int8_rowwise(torch.randn(40, 1000, device=self.device))
        self.assertTrue(torch.equal(port(cq.int8_linear, x, q8, s8), kt.int8_linear(x, q8, s8)))

    def test_w4a4_matches_the_port_bit_for_bit(self):
        for dtype in (torch.bfloat16, torch.float16):
            for rows in (1, 33, 300):
                x = activations(rows, 768, dtype, self.device)
                _, _, q4, s4 = weights(96, 768, self.device)
                bias = torch.randn(96, device=self.device).to(dtype)
                for linear_dtype in ("int4", "int8"):
                    with self.subTest(dtype=dtype, rows=rows, linear_dtype=linear_dtype):
                        args = (x, q4, s4, bias, 256, 64, linear_dtype)
                        self.assertTrue(torch.equal(port(cq.convrot_w4a4_linear, *args), kt.convrot_w4a4_linear(*args)))

    def test_leading_dimensions_and_an_empty_batch(self):
        q8, s8, q4, s4 = weights(72, 512, self.device)
        x = activations(6, 512, torch.bfloat16, self.device).reshape(2, 3, 512)
        self.assertEqual(tuple(kt.int8_linear(x, q8, s8, None, None, True, 256).shape), (2, 3, 72))
        self.assertEqual(tuple(kt.convrot_w4a4_linear(x, q4, s4, None, 256).shape), (2, 3, 72))
        empty = torch.empty(0, 512, device=self.device, dtype=torch.bfloat16)
        self.assertEqual(tuple(kt.int8_linear(empty, q8, s8, None, None, True, 256).shape), (0, 72))

    def test_the_layer_dispatches_to_the_kernels_with_the_ports_result(self):
        self.assertTrue(kt.status(self.device)["available"])
        x = activations(40, 512, torch.bfloat16, self.device)
        q8, s8, _, _ = weights(72, 512, self.device)
        with mock.patch.object(cq, "kitchen_ops", return_value=None), \
                mock.patch.object(kt, "int8_linear", wraps=kt.int8_linear) as spy:
            got = cq.int8_linear(x, q8, s8, None, None, True, 256)
        spy.assert_called_once()
        self.assertTrue(torch.equal(got, port(cq.int8_linear, x, q8, s8, None, None, True, 256)))

    def test_a_kernel_failure_hands_the_layer_back_to_the_port(self):
        x = activations(40, 512, torch.bfloat16, self.device)
        q8, s8, _, _ = weights(72, 512, self.device)
        index = torch.cuda.current_device()
        try:
            with mock.patch.object(cq, "kitchen_ops", return_value=None), \
                    mock.patch.object(kt, "int8_linear", side_effect=RuntimeError("PTX assembly failed")):
                got = cq.int8_linear(x, q8, s8, None, None, True, 256)
            self.assertTrue(torch.equal(got, port(cq.int8_linear, x, q8, s8, None, None, True, 256)))
            self.assertEqual(kt.status(self.device), {"available": False, "reason": "RuntimeError: PTX assembly failed"})
        finally:
            kt._STATE["failed"].pop(index, None)
        self.assertTrue(kt.status(self.device)["available"])

    def test_running_out_of_memory_is_not_mistaken_for_a_broken_kernel(self):
        x = activations(40, 512, torch.bfloat16, self.device)
        q8, s8, _, _ = weights(72, 512, self.device)
        with mock.patch.object(cq, "kitchen_ops", return_value=None), \
                mock.patch.object(kt, "int8_linear", side_effect=torch.OutOfMemoryError("CUDA out of memory")):
            with self.assertRaises(torch.OutOfMemoryError):
                cq.int8_linear(x, q8, s8, None, None, True, 256)
        self.assertTrue(kt.status(self.device)["available"])


def identical(expected, got):
    """Bit for bit, the sign of zero included; a NaN only has to be a NaN in the same place (an
    overflowed scale times a zero value is one on both sides, with no promise about its payload)."""
    if expected.shape != got.shape or expected.dtype != got.dtype:
        return False
    nan = torch.isnan(expected)
    if not torch.equal(nan, torch.isnan(got)):
        return False
    view = {torch.float32: torch.int32, torch.bfloat16: torch.int16, torch.float16: torch.int16}[expected.dtype]
    return torch.equal(expected[~nan].view(view), got[~nan].view(view))


def random_bytes(shape, generator, low=0, high=256, forbid=()):
    values = torch.randint(low, high, shape, generator=generator, dtype=torch.int32)
    for byte in forbid:
        values[values == byte] = 0x30
    return values.to(torch.uint8)


def nvfp4_layer(rows, storage_cols, device, seed=2):
    """Random packed E2M1 bytes (every code) and E4M3 block scales in the tiled layout, zero and
    subnormal scales included, NaN scales left out (the port's NaN is not bit-comparable)."""
    generator = torch.Generator().manual_seed(seed)
    padded_rows, padded_blocks = cq.swizzled_scale_shape(rows, storage_cols * 2 // cq.NVFP4_GROUP_SIZE)
    qdata = random_bytes((rows, storage_cols), generator)
    scales = random_bytes((padded_rows, padded_blocks), generator, forbid=(0x7F, 0xFF)).view(torch.float8_e4m3fn)
    return qdata.to(device), scales.to(device)


def mxfp8_layer(rows, storage_cols, device, seed=3):
    generator = torch.Generator().manual_seed(seed)
    padded_rows, padded_blocks = cq.swizzled_scale_shape(rows, storage_cols // cq.MXFP8_GROUP_SIZE)
    qdata = random_bytes((rows, storage_cols), generator, forbid=(0x7F, 0xFF)).view(torch.float8_e4m3fn)
    # Every E8M0 byte: 0 is a zero scale, 1 reaches float32 subnormals, 255 is infinity.
    scales = random_bytes((padded_rows * padded_blocks,), generator)
    return qdata.to(device), scales.to(device)


@unittest.skipUnless(GPU, "needs a CUDA device of SM 8.0+ and Triton")
class DecodeTests(unittest.TestCase):
    device = torch.device("cuda")
    dtypes = (torch.bfloat16, torch.float16, torch.float32)

    def test_nvfp4_decodes_like_the_port_bit_for_bit(self):
        # (rows, packed columns, layer columns): whole tiles; padded rows and blocks with trailing
        # columns trimmed; a single row; 17 blocks (padded to 20).
        for rows, storage_cols, cols in ((128, 64, 128), (200, 40, 72), (1, 8, 16), (300, 136, 270)):
            qdata, scales = nvfp4_layer(rows, storage_cols, self.device)
            for global_scale in (0.0123, 3.0e-5, 700.0):  # the last overflows float16 to inf, on both sides
                scale = torch.tensor(global_scale, dtype=torch.float32, device=self.device)
                for dtype in self.dtypes:
                    with self.subTest(rows=rows, cols=cols, global_scale=global_scale, dtype=dtype):
                        args = (qdata, scale, scales, dtype, rows, cols)
                        self.assertTrue(identical(port(cq.dequantize_nvfp4, *args), kt.dequantize_nvfp4(*args)))
        qdata, scales = nvfp4_layer(200, 40, self.device)
        as_bytes = scales.view(torch.uint8)  # files store the scales as E4M3 or as raw bytes
        scale = torch.tensor(0.02, device=self.device)
        self.assertTrue(identical(port(cq.dequantize_nvfp4, qdata, scale, as_bytes, torch.bfloat16, 200, 80),
                                  kt.dequantize_nvfp4(qdata, scale, as_bytes, torch.bfloat16, 200, 80)))

    def test_mxfp8_decodes_like_the_port_bit_for_bit(self):
        for rows, storage_cols, cols in ((200, 128, 100), (256, 64, 64), (33, 96, 90)):
            qdata, scales = mxfp8_layer(rows, storage_cols, self.device)
            for dtype in self.dtypes:
                with self.subTest(rows=rows, cols=cols, dtype=dtype):
                    args = (qdata, scales, dtype, rows, cols)
                    self.assertTrue(identical(port(cq.dequantize_mxfp8, *args), kt.dequantize_mxfp8(*args)))

    def test_the_layer_decode_dispatches_to_the_kernel_with_the_ports_result(self):
        self.assertTrue(kt.decode_status(self.device)["available"])
        qdata, scales = nvfp4_layer(200, 40, self.device)
        scale = torch.tensor(0.0123, device=self.device)
        with mock.patch.object(kt, "dequantize_nvfp4", wraps=kt.dequantize_nvfp4) as spy:
            got = cq.dequantize_nvfp4(qdata, scale, scales, torch.bfloat16, 200, 72)
        spy.assert_called_once()
        self.assertTrue(torch.equal(got, port(cq.dequantize_nvfp4, qdata, scale, scales, torch.bfloat16, 200, 72)))

    def test_a_decoder_failure_hands_the_layer_back_to_the_port_and_leaves_the_integer_kernels_on(self):
        qdata, scales = mxfp8_layer(64, 64, self.device)
        index = torch.cuda.current_device()
        self.assertTrue(kt.decode_status(self.device)["available"])  # verified before the fault is planted
        try:
            with mock.patch.object(kt, "dequantize_mxfp8", side_effect=RuntimeError("decoder broke")):
                got = cq.dequantize_mxfp8(qdata, scales, torch.bfloat16, 64, 64)
            self.assertTrue(torch.equal(got, port(cq.dequantize_mxfp8, qdata, scales, torch.bfloat16, 64, 64)))
            self.assertEqual(kt.decode_status(self.device), {"available": False, "reason": "RuntimeError: decoder broke"})
            self.assertTrue(kt.status(self.device)["available"])
        finally:
            kt._DECODE_STATE["failed"].pop(index, None)
        self.assertTrue(kt.decode_status(self.device)["available"])

    def test_running_out_of_memory_is_not_mistaken_for_a_broken_decoder(self):
        qdata, scales = nvfp4_layer(64, 32, self.device)
        scale = torch.tensor(0.5, device=self.device)
        self.assertTrue(kt.decode_status(self.device)["available"])
        with mock.patch.object(kt, "dequantize_nvfp4", side_effect=torch.OutOfMemoryError("CUDA out of memory")):
            with self.assertRaises(torch.OutOfMemoryError):
                cq.dequantize_nvfp4(qdata, scale, scales, torch.bfloat16, 64, 64)
        self.assertTrue(kt.decode_status(self.device)["available"])


class AvailabilityTests(unittest.TestCase):
    def test_the_switch_turns_the_decoders_off_too(self):
        with mock.patch.dict(os.environ, {"XIRAI_QUANT_TRITON": "0"}):
            self.assertEqual(kt.decode_status()["reason"], "disabled by XIRAI_QUANT_TRITON=0")
            self.assertFalse(kt.decode_peek()["available"])

    def test_cpu_weights_are_decoded_by_the_port(self):
        call = mock.Mock()
        self.assertIsNone(cq._decoded_on_triton(torch.zeros(4, 8, dtype=torch.uint8), torch.bfloat16, call))
        call.assert_not_called()

    def test_the_model_status_names_the_decoder_without_compiling(self):
        with mock.patch.object(cq, "supports_nvfp4_compute", return_value=False):
            with mock.patch.object(kt, "decode_peek", return_value={"available": None, "reason": "verified on first use"}):
                info = cq.execution_info(cq.LayerQuantConfig("nvfp4"))
                self.assertEqual((info["path"], info["decoder"]), ("emulated", "built-in Triton"))
                self.assertIn("built-in Triton decoder", info["text"])
            with mock.patch.object(kt, "decode_peek", return_value={"available": False, "reason": "no CUDA device"}):
                self.assertIn("ported kernel decoder", cq.execution_path(cq.LayerQuantConfig("nvfp4")))
                self.assertEqual(cq.quantization_capabilities()["triton_decoders"],
                                 {"available": False, "reason": "no CUDA device"})
        self.assertNotIn("decoder", cq.execution_info(cq.LayerQuantConfig("float8_e4m3fn")))

    def test_the_switch_turns_the_kernels_off(self):
        with mock.patch.dict(os.environ, {"XIRAI_QUANT_TRITON": "0"}):
            self.assertEqual(kt.status()["reason"], "disabled by XIRAI_QUANT_TRITON=0")
            self.assertFalse(kt.peek()["available"])
            self.assertIsNone(cq.triton_kernels(torch.device("cuda")))

    def test_without_triton_the_port_runs(self):
        with mock.patch.object(kt, "triton", None):
            self.assertIn("not installed", kt.status()["reason"])
            self.assertFalse(kt.peek()["available"])

    def test_cpu_activations_never_leave_the_port(self):
        self.assertIsNone(cq._accelerated(torch.zeros(2, 4)))

    def test_the_model_status_names_the_kernel_without_compiling(self):
        config = cq.LayerQuantConfig("int8_tensorwise", convrot=True)
        with mock.patch.object(cq, "kitchen_ops", return_value=None), \
                mock.patch.object(cq, "_cuda_capability", return_value=(8, 9)):
            with mock.patch.object(kt, "peek", return_value={"available": None, "reason": "verified on first use"}):
                self.assertIn("built-in Triton", cq.execution_path(config))
            with mock.patch.object(kt, "peek", return_value={"available": False, "reason": "no CUDA device"}):
                self.assertIn("ported kernel", cq.execution_path(config))
                report = cq.quantization_capabilities()
                self.assertEqual(report["triton_kernels"], {"available": False, "reason": "no CUDA device"})


if __name__ == "__main__":
    unittest.main()
