import ast
import copy
import json
import unittest
from pathlib import Path

from backend.xyz_planner import CAPABILITIES, PlanningError, compile_xyz, plan_record
from backend.xyz_schema import XYZExperimentInput

ROOT = Path(__file__).parents[2]


def baseline(engine='SD'):
    p = dict(engine=engine, prompt='group words\n\ncat, blue sky', negative_prompt='bad',
             width=512, height=512, steps=20, cfg=6.5, seed='18446744073709551615',
             sampler='euler', scheduler='simple', guidance='none')
    p.update({key: key + '.safetensors' for key in CAPABILITIES['engines'][engine]['model_fields']})
    if engine in {'Flux', 'Flux2'}:
        p['negative_prompt'] = ''
    return dict(schema_version=1, kind='xyz_draft', engine=engine,
                baseline=dict(parameters=p, prompt_without_groups='cat, blue sky', group_prompt='group words', loras=[]),
                axes={}, repeat_count=1, requested_concurrency=1)


def axis(field, values, **kwargs):
    return dict(field=field, values=[dict(value=value, label=f'v{i}') for i, value in enumerate(values)], **kwargs)


def member(index=1, weight='1'):
    return dict(value=f'folder/lora-{index}.safetensors', weight=weight, precision=4,
                identitySchema=1, instanceId=f'00000000-0000-4000-8000-{index:012d}',
                contentHash=f'{index:064x}', contentRevision=1)


def evidence(members):
    return dict(schemaVersion=1, instances={m['instanceId']: {**m, 'engine': 'SD', 'status': 'present'} for m in members})


def compile(draft, **kwargs):
    p = draft['baseline']['parameters']
    catalog = {(draft['engine'], k, v) for k, v in p.items() if k in CAPABILITIES['engines'][draft['engine']]['model_fields']}
    return compile_xyz(draft, model_catalog=kwargs.pop('model_catalog', catalog), **kwargs)


class XYZPlannerTests(unittest.TestCase):
    def test_six_engine_defaults_and_static_readiness(self):
        for engine in CAPABILITIES['engines']:
            with self.subTest(engine=engine):
                plan = compile(baseline(engine))
                self.assertEqual(plan.errors, [])
                self.assertTrue(plan.static_only)
                self.assertFalse(plan.executor_ready)
                self.assertEqual(plan.cells[0].request['seed'], '18446744073709551615')
                self.assertNotIn('source_image', plan.cells[0].request)
                self.assertEqual(plan.cells[0].request['images_per_batch'], 1)

    def test_vocabulary_matches_native_source_and_strict_wire_unchanged(self):
        tree = ast.parse((ROOT/'backend/anima_sampling.py').read_text(encoding='utf-8'))
        constants = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in {'ANIMA_SAMPLERS', 'ANIMA_SCHEDULERS'}}
        self.assertEqual(list(constants['ANIMA_SAMPLERS']), CAPABILITIES['samplers'])
        self.assertEqual(list(constants['ANIMA_SCHEDULERS']), CAPABILITIES['schedulers'])
        tree = ast.parse((ROOT/'backend/inference_server.py').read_text(encoding='utf-8'))
        model = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GenerateInput')
        fields = {n.target.id for n in model.body if isinstance(n, ast.AnnAssign)}
        for engine in CAPABILITIES['engines']:
            request = compile(baseline(engine)).cells[0].request
            self.assertFalse(set(request) - fields)
        self.assertNotIn('axes', fields)
        self.assertNotIn('xyz', fields)

    def test_dimensions_order_and_coordinate_stability(self):
        d = baseline(); d['axes'] = dict(x=axis('steps', ['10', '20']), y=axis('cfg', ['1', '2']), z=axis('prompt', ['cat, blue', 'dog, red']))
        d['repeat_count'] = 2; d['requested_concurrency'] = 10
        plan = compile(d)
        self.assertEqual(plan.total_cells, 16)
        self.assertEqual(plan.dimensions, dict(x=2, y=2, z=2))
        self.assertEqual([c.cell_id for c in plan.cells[:5]], ['0:0:0:0', '1:0:0:0', '0:1:0:0', '1:1:0:0', '0:0:0:1'])
        self.assertEqual(plan.cells[0].request['prompt'], 'cat, blue')

    def test_decimal_range_without_float_accumulation(self):
        d = baseline(); d['axes']['x'] = dict(field='cfg', range=dict(start='0.1', end='0.3', step='0.1'))
        plan = compile(d)
        self.assertEqual([c.request['cfg'] for c in plan.cells], [0.1, 0.2, 0.3])
        d['axes']['x']['range'] = dict(start='0.3', end='0.1', step='-0.1')
        self.assertEqual([c.request['cfg'] for c in compile(d).cells], [0.3, 0.2, 0.1])

    def test_bad_ranges_are_not_truncated(self):
        for values in [('0', '1', '0'), ('1', '0', '1'), ('0', '1000', '1'), ('NaN', '1', '1')]:
            d = baseline(); d['axes']['x'] = dict(field='cfg', range=dict(zip(['start', 'end', 'step'], values)))
            self.assertTrue(compile(d).errors)

    def test_uint64_seed_axis_and_repeat_policy_no_overflow(self):
        d = baseline(); d['axes']['x'] = axis('seed', ['0', '18446744073709551615']); d['repeat_count'] = 3
        plan = compile(d)
        self.assertEqual([c.request['seed'] for c in plan.cells], ['0', '18446744073709551615'] * 3)
        d['sample_seeds'] = ['1', '2', '3']
        with self.assertRaisesRegex(PlanningError, 'Seed axis wins'):
            compile(d)

    def test_seed_bounds_error_has_coordinate(self):
        for value in ['18446744073709551616', '-1', '1.5', 18446744073709551615]:
            d = baseline(); d['axes']['x'] = axis('seed', ['0', value])
            plan = compile(d)
            self.assertEqual(plan.errors[0].coordinate.x_index, 1)
            self.assertEqual(plan.errors[0].code, 'seed')

    def test_preview_never_draws_random_and_samples_shared(self):
        d = baseline(); d['baseline']['seed_mode'] = 'random'; d['repeat_count'] = 2; d['axes']['x'] = axis('steps', ['1', '2'])
        def fail(): raise AssertionError('preview RNG called')
        plan = compile(d, random_uint64=fail)
        self.assertEqual(plan.seed_status, 'unresolved')
        self.assertEqual([c.request['seed'] for c in plan.cells], ['unresolved:sample:0'] * 2 + ['unresolved:sample:1'] * 2)

    def test_submit_draw_once_freeze_python_int_and_json_strings(self):
        d = baseline(); d['baseline']['seed_mode'] = 'random'; d['repeat_count'] = 2; d['axes']['x'] = axis('steps', ['1', '2'])
        draws = []
        def rng(): draws.append(len(draws)); return 18446744073709551615 - draws[-1]
        validated = []
        plan = compile(d, resolve_seeds=True, random_uint64=rng, validate_request=lambda request: validated.append(copy.deepcopy(request)))
        self.assertEqual(len(draws), 2)
        self.assertEqual(plan.seed_status, 'resolved')
        self.assertIs(type(plan.cells[0].request['seed']), int)
        record = plan_record(plan)
        self.assertEqual(record['cells'][0]['request']['seed'], '18446744073709551615')
        self.assertEqual(len(validated), 8)
        self.assertEqual(plan.cells[0].request['seed'], plan.cells[1].request['seed'])

    def test_explicit_sample_list_and_fixed_baseline(self):
        d = baseline(); d['repeat_count'] = 2; d['sample_seeds'] = ['0', '18446744073709551615']
        self.assertEqual([c.request['seed'] for c in compile(d).cells], d['sample_seeds'])
        d['sample_seeds'] = ['1']
        with self.assertRaises(PlanningError): compile(d)

    def test_failed_preflight_and_validator_consume_no_rng(self):
        d = baseline(); d['axes']['x'] = axis('steps', ['0']); calls = []
        plan = compile(d, resolve_seeds=True, random_uint64=lambda: calls.append(1), validate_request=lambda _: None)
        self.assertTrue(plan.errors); self.assertEqual(calls, [])
        d['axes'] = {}; d['baseline']['seed_mode'] = 'random'
        def fail(_): raise ValueError('not admitted')
        with self.assertRaises(ValueError): compile(d, resolve_seeds=True, random_uint64=lambda: calls.append(1), validate_request=fail)
        self.assertEqual(calls, [])

    def test_postprocess_off_and_random_seeds_frozen_per_stage_cell(self):
        d = baseline(); d['axes']['x'] = axis('steps', ['1', '2'])
        d['baseline']['parameters']['postprocess_chain'] = [dict(id='r1', type='resample', enabled=True, seed_mode='random')]
        self.assertEqual(compile(d).cells[0].request['postprocess_chain'], [])
        d['inherit_postprocess'] = True
        plan = compile(d)
        self.assertNotEqual(plan.cells[0].request['postprocess_chain'][0]['seed'], plan.cells[1].request['postprocess_chain'][0]['seed'])
        calls = []
        def rng(): calls.append(1); return len(calls)
        plan = compile(d, resolve_seeds=True, random_uint64=rng, validate_request=lambda _: None)
        self.assertEqual(len(calls), 2)
        self.assertEqual(plan.cells[0].request['postprocess_chain'][0]['seed_mode'], 'fixed')

    def test_no_mutation_of_baseline_groups_or_axis_values(self):
        d = baseline(); d['baseline']['loras'] = [member()]; d['baseline']['group_snapshots'] = [dict(name='g', members=[member()])]
        d['axes']['x'] = axis('lora_weight', ['0', '1.2345'], target=member()['instanceId'])
        before = copy.deepcopy(d)
        compile(d, lora_registry=evidence([member()]))
        self.assertEqual(before, d)

    def test_dual_weight_axes_update_existing_order_and_zero_disables(self):
        d = baseline(); members = [member(1), member(2)]; d['baseline']['loras'] = members
        d['axes'] = dict(x=axis('lora_weight', ['0', '1.2345'], target=members[0]['instanceId']), y=axis('lora_weight', ['-5', '5'], target=members[1]['instanceId']))
        plan = compile(d, lora_registry=evidence(members))
        self.assertFalse(plan.errors)
        self.assertEqual(plan.cells[0].request['loras'], [dict(path=members[1]['value'], weight=-5.0)])
        self.assertEqual([e['instanceId'] for e in plan.cells[1].lora_references], [m['instanceId'] for m in members])

    def test_profiles_keep_background_full_no_test_and_no_loras(self):
        d = baseline(); a, b = member(1), member(2); d['baseline']['loras'] = [a, b]
        profiles = [dict(mode='keep_background', members=[{**a, 'weight': '-2'}], test_targets=[a['instanceId']]),
                    dict(mode='full', members=[b]), dict(mode='no_test', test_targets=[a['instanceId']]), dict(mode='no_loras')]
        d['axes']['x'] = axis('lora_profile', profiles)
        plan = compile(d, lora_registry=evidence([a, b]))
        self.assertFalse(plan.errors)
        self.assertEqual([len(c.request['loras']) for c in plan.cells], [2, 1, 1, 0])
        self.assertEqual(plan.cells[0].request['loras'][0]['weight'], -2)

    def test_group_words_fixed_or_explicit_substitution_once(self):
        d = baseline(); d['axes']['x'] = axis('lora_profile', [dict(mode='full', members=[], preset_prompt='group words'), dict(mode='full', members=[], preset_prompt='new words', includePresetText=True)])
        plan = compile(d)
        self.assertEqual(plan.cells[0].request['prompt'], d['baseline']['parameters']['prompt'])
        self.assertEqual(plan.cells[1].request['prompt'], 'new words\n\ncat, blue sky')
        self.assertNotIn('group words', plan.cells[1].request['prompt'])

    def test_model_candidates_replace_slot_without_double_append(self):
        d = baseline(); a, b, c = member(1), member(2), member(3); d['baseline']['loras'] = [a, c]
        d['axes']['x'] = axis('lora_model', [b], target=a['instanceId'])
        plan = compile(d, lora_registry=evidence([a, b, c]))
        self.assertFalse(plan.errors)
        self.assertEqual([e['instanceId'] for e in plan.cells[0].lora_references], [b['instanceId'], c['instanceId']])

    def test_multiplier_precision_overflow_not_clamped(self):
        d = baseline(); a = member(1, '1.2345'); d['axes']['x'] = axis('lora_multiplier', ['1', '5'], snapshot=dict(mode='full', members=[a]))
        plan = compile(d, lora_registry=evidence([a]))
        self.assertEqual(plan.cells[0].request['loras'][0]['weight'], 1.2345)
        self.assertEqual(plan.errors[0].coordinate.x_index, 1)
        d['axes']['x']['values'] = [dict(value='0.12345')]
        self.assertEqual(compile(d, lora_registry=evidence([a])).errors[0].code, 'precision')

    def test_duplicate_same_instance_same_field_and_combo_conflicts(self):
        for fields in [('steps', 'steps'), ('prompt', 'prompt_replace'), ('lora_weight', 'lora_weight'), ('lora_profile', 'lora_weight')]:
            d = baseline(); d['baseline']['loras'] = [member()]
            d['axes'] = dict(x=axis(fields[0], ['1'], target=member()['instanceId']), y=axis(fields[1], ['1'], target=member()['instanceId']))
            plan = compile(d, lora_registry=evidence([member()]))
            self.assertEqual(plan.errors[0].code, 'axis-conflict')
            self.assertEqual(plan.errors[0].coordinate.x_index, 0)

    def test_duplicate_content_paths_and_profile_members_refused(self):
        a, b = member(1), member(2)
        for entries in [[a, a], [a, {**b, 'contentHash': a['contentHash']}]]:
            d = baseline(); d['baseline']['loras'] = entries
            self.assertTrue(compile(d, lora_registry=evidence(entries)).errors)
        d = baseline(); d['axes']['x'] = axis('lora_profile', [dict(mode='full', members=[a, a])])
        self.assertEqual(compile(d, lora_registry=evidence([a])).errors[0].code, 'duplicate-lora')

    def test_lora_identity_content_revision_wrong_engine_and_missing(self):
        for key, value in [('contentRevision', 2), ('contentHash', 'f'*64), ('engine', 'Flux'), ('status', 'missing'), ('value', 'moved.safetensors')]:
            d = baseline(); a = member(); d['baseline']['loras'] = [a]; registry = evidence([a]); registry['instances'][a['instanceId']][key] = value
            self.assertEqual(compile(d, lora_registry=registry).errors[0].code, 'lora-identity')

    def test_lora_limit_precision_and_weight_bounds(self):
        for entries in [[member(i) for i in range(1, 18)], [member(1, '5.0001')], [member(1, '0.12345')]]:
            d = baseline(); d['baseline']['loras'] = entries
            self.assertTrue(compile(d, lora_registry=evidence(entries)).errors)

    def test_inline_lora_fails_closed_not_silent_double_fusion(self):
        d = baseline(); d['baseline']['parameters']['prompt'] = 'cat <lora:folder/a:1>'
        self.assertEqual(compile(d).errors[0].code, 'inline-admission-pending')

    def test_exact_replace_matches_prompt_not_model_name(self):
        d = baseline(); d['axes']['x'] = axis('prompt_replace', [dict(find='cat', replace='dog, red', expected_matches=1)])
        plan = compile(d)
        self.assertEqual(plan.cells[0].replacement_matches, dict(prompt=1))
        self.assertIn('dog, red', plan.cells[0].request['prompt'])
        self.assertEqual(plan.cells[0].request['checkpoint'], d['baseline']['parameters']['checkpoint'])
        d['axes']['x']['values'][0]['value']['expected_matches'] = 2
        self.assertEqual(compile(d).errors[0].code, 'replace-count')

    def test_native_clip_denoise_distill_guidance_restrictions(self):
        for engine in CAPABILITIES['engines']:
            fields = ['denoise', 'resample.r1.steps']
            if engine not in {'SD', 'iL'}: fields += ['clip_skip']
            if engine in {'Flux', 'Flux2'}: fields += ['negative_prompt', 'pag.scale']
            for field in fields:
                d = baseline(engine); d['axes']['x'] = axis(field, ['1'])
                self.assertTrue(compile(d).errors, (engine, field))
        for engine, invalid in [('SD', 'cfg_zero_star'), ('iL', 'cfg_zero_star'), ('Krea2', 'pag'), ('Flux', 'pag'), ('Flux2', 'cfg_zero_star')]:
            d = baseline(engine); d['axes']['x'] = axis('guidance', [invalid])
            self.assertTrue(compile(d).errors)

    def test_supported_guidance_and_pag_requires_effect(self):
        for engine, types in [('Anima', ['none', 'pag', 'cfg_zero_star']), ('Krea2', ['none', 'cfg_zero_star'])]:
            d = baseline(engine); d['axes']['x'] = axis('guidance', types)
            self.assertFalse(compile(d).errors)
        d = baseline(); d['axes']['x'] = axis('pag.scale', ['0.3'])
        self.assertEqual(compile(d).errors[0].code, 'ineffective-axis')
        d['baseline']['parameters']['guidance'] = 'pag'
        self.assertFalse(compile(d).errors)

    def test_scheduler_axis_refuses_known_inert_sd_adapters(self):
        for sampler in ['lcm', 'ipndm', 'ipndm_v', 'ddim', 'ddpm']:
            d = baseline(); d['baseline']['parameters']['sampler'] = sampler
            d['axes']['x'] = axis('scheduler', ['normal', 'karras'])
            self.assertTrue(all(e.code == 'ineffective-axis' for e in compile(d).errors))

    def test_replacement_amplification_fails_before_allocation(self):
        d = baseline(); d['baseline']['parameters']['prompt'] = 'a' * 2000
        d['axes']['x'] = axis('prompt_replace', [dict(find='a', replace='b' * 2000, expected_matches=2000)])
        self.assertEqual(compile(d).errors[0].code, 'replace-limit')

    def test_model_components_catalog_engine_kind_and_vae(self):
        d = baseline('Flux'); p = d['baseline']['parameters']; profile = {'engine': 'Flux', **{k: p[k] for k in CAPABILITIES['engines']['Flux']['model_fields']}}
        d['axes']['x'] = axis('components', [profile])
        self.assertFalse(compile(d).errors)
        profile['engine'] = 'Anima'
        self.assertEqual(compile(d).errors[0].code, 'components')
        d = baseline('Anima'); d['axes']['x'] = axis('vae', ['new.safetensors'])
        self.assertEqual(compile(d).errors[0].code, 'model-resource')
        catalog = {(d['engine'], k, v) for k, v in d['baseline']['parameters'].items() if k in CAPABILITIES['engines'][d['engine']]['model_fields']}
        catalog.add(('Anima', 'vae', 'new.safetensors'))
        self.assertFalse(compile(d, model_catalog=catalog).errors)

    def test_cells_body_and_expansion_limits(self):
        d = baseline(); d['axes'] = dict(x=axis('steps', ['1']*33), y=axis('cfg', ['1']*33))
        with self.assertRaisesRegex(PlanningError, '1024'): compile(d)
        d = baseline(); d['baseline']['parameters']['prompt'] = 'a' * 1048576
        with self.assertRaisesRegex(PlanningError, '1 MiB'): compile(d)
        d['baseline']['parameters']['prompt'] = 'a' * 20000; d['repeat_count'] = 32; d['axes']['x'] = axis('steps', ['1']*32)
        with self.assertRaisesRegex(PlanningError, '16 MiB'): compile(d)

    def test_wrong_engine_future_schema_plan_not_draft_and_private_wire(self):
        d = baseline(); d['baseline']['parameters']['engine'] = 'iL'
        with self.assertRaises(PlanningError): compile(d)
        for patch in [dict(schema_version=2), dict(kind='xyz_plan'), dict(private_xyz='bad')]:
            d = {**baseline(), **patch}
            with self.assertRaises(ValueError): XYZExperimentInput.model_validate(d)


if __name__ == '__main__':
    unittest.main()
