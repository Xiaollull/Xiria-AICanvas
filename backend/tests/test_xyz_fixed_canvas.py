"""GPU-free checks of the EXISTING authoritative planner's fixed original canvas contract."""
import copy
import unittest

from backend.tests.test_xyz_planner import baseline, axis, compile, member, evidence
from backend.xyz_planner import CAPABILITIES


class XYZFixedCanvasTests(unittest.TestCase):
    def test_all_engines_2x2_repeat_preview_and_submit_keep_original_canvas_frozen(self):
        for engine in CAPABILITIES['engines']:
            with self.subTest(engine=engine):
                draft = baseline(engine)
                draft['baseline']['parameters'].update(width=1024, height=1408)
                draft['axes'] = dict(x=axis('steps', ['20', '24']), y=axis('cfg', ['4', '6']))
                draft['repeat_count'] = 2
                before = copy.deepcopy(draft)
                preview = compile(draft)
                self.assertEqual(preview.errors, [])
                self.assertEqual(preview.total_cells, 8)
                self.assertEqual({(c.request['width'], c.request['height']) for c in preview.cells}, {(1024, 1408)})
                captured = []
                submitted = compile(draft, resolve_seeds=True, random_uint64=lambda: 12,
                                    validate_request=lambda r: captured.append(copy.deepcopy(r)))
                self.assertEqual(submitted.errors, [])
                self.assertEqual({(r['width'], r['height']) for r in captured}, {(1024, 1408)})
                self.assertEqual(draft, before)
                draft['baseline']['parameters'].update(width=768, height=512)
                self.assertEqual({(c.request['width'], c.request['height']) for c in submitted.cells}, {(1024, 1408)})
                self.assertTrue(all(c.request['postprocess_chain'] == [] for c in submitted.cells))

    def test_raw_dimension_axes_are_refused_not_ignored(self):
        for field in ['width', 'height', 'baseline.width', 'canvas', 'size']:
            with self.subTest(field=field):
                draft = baseline()
                draft['axes']['x'] = axis(field, ['1024', '1408'])
                plan = compile(draft)
                self.assertTrue(plan.errors)
                self.assertEqual(plan.cells, [])

    def test_component_and_lora_profile_dimension_injection_is_refused(self):
        samples = [
            ('components', dict(engine='SD', checkpoint='checkpoint.safetensors', width=1024)),
            ('components', dict(engine='SD', checkpoint='checkpoint.safetensors', height=1408)),
            ('lora_profile', dict(mode='no_loras', members=[], test_targets=[], width=1024)),
            ('lora_profile', dict(mode='full', members=[dict(member(), height=1408)], test_targets=[])),
        ]
        for field, value in samples:
            with self.subTest(field=field, value=value):
                draft = baseline(); draft['axes']['x'] = axis(field, [value])
                plan = compile(draft, lora_registry=evidence([member()]))
                self.assertTrue(plan.errors)
                self.assertEqual(plan.cells, [])

    def test_postprocess_inheritance_preserves_original_canvas_and_chain(self):
        draft = baseline(); draft['baseline']['parameters'].update(width=1024, height=1408)
        chain = [dict(type='resample', enabled=True, scale=1.5, seed_mode='inherit')]
        draft['baseline']['parameters']['postprocess_chain'] = chain
        self.assertEqual(compile(draft).cells[0].request['postprocess_chain'], [])
        draft['inherit_postprocess'] = True
        plan = compile(draft)
        self.assertEqual(plan.errors, [])
        self.assertEqual((plan.cells[0].request['width'], plan.cells[0].request['height']), (1024, 1408))
        self.assertEqual(plan.cells[0].request['postprocess_chain'][0]['scale'], 1.5)
        self.assertEqual(draft['baseline']['parameters']['postprocess_chain'], chain)


if __name__ == '__main__':
    unittest.main()
