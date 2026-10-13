import copy
import json
import unittest
from pathlib import Path

from backend import lora_mix_search as mix
from backend.xyz_planner import CAPABILITIES, compile_xyz
from backend.xyz_schema import XYZExperimentInput

FIXTURE = json.loads((Path(__file__).parent / 'fixtures' / 'lora_mix_parity.json').read_text(encoding='utf-8'))


def uid(n):
    return f'00000000-0000-4000-8000-{n:012d}'


def ref(n, **extra):
    return {'value': f'style/l{n}.safetensors', 'name': f'L{n}', 'category': 'style', 'identitySchema': 1, 'instanceId': uid(n),
            'contentHash': f'{n:064x}', 'contentRevision': 2, **extra}


def global_lora(n, weight='0.7', **extra):
    return {**ref(n), 'weight': weight, 'precision': 4, 'enabled': True, **extra}


def rule(base, low, high, step='0.1', preference='balanced'):
    return {'locked': False, 'base': base, 'min': low, 'max': high, 'step': step, 'preference': preference}


def draft(members=None, candidates=None, *, loras=(), engine='Anima', **mix_extra):
    members = members if members is not None else [{'ref': ref(11), 'rule': rule('0.6', '0.2', '1.0', preference='max')},
                                                   {'ref': ref(12), 'rule': rule('-0.2', '-0.8', '0.6', preference='min')},
                                                   {'ref': ref(13), 'rule': {'locked': True, 'base': '0.5'}}]
    parameters = {'engine': engine, 'diffusion_model': 'd.safetensors', 'text_encoder': 't.safetensors', 'vae': 'v.safetensors', 'prompt': 'raw', 'negative_prompt': '',
                  'steps': 20, 'cfg': 4, 'width': 512, 'height': 512, 'seed': '7', 'sampler': 'euler', 'scheduler': 'simple', 'guidance': 'none',
                  'pag': {'scale': 0.3, 'applied_layers': 'mid'}, 'postprocess_chain': []}
    block = {'version': 1, 'members': members, 'max_displacement': '0.6', 'budget': {'preset': '16', 'custom': ''}, 'include_baseline': True, 'sampling_seed': '1',
             'cursor': 0, 'candidates': candidates if candidates is not None else [], 'next_id': 1, 'proposed_key': '', 'session': {'id': None, 'revision': 0}}
    block.update(mix_extra)
    return {'schema_version': 1, 'kind': 'xyz_draft', 'engine': engine, 'axes': {'x': None, 'y': None, 'z': None}, 'repeat_count': 1, 'sample_seeds': None,
            'requested_concurrency': 1, 'inherit_postprocess': False, 'group_by_resources': False,
            'baseline': {'parameters': parameters, 'loras': list(loras), 'group_snapshots': [], 'prompt_without_groups': 'raw', 'group_prompt': '', 'seed_mode': 'fixed', 'library_revision': 0}, 'mix': block}


def row(cid, a, b, source='manual'):
    return {'id': cid, 'source': source, 'raw': {uid(11): a, uid(12): b}}


def issues_of(payload):
    try:
        mix.validate_recipe(payload)
    except mix.MixError as error:
        return error.issues
    return []


class ParityTests(unittest.TestCase):
    def test_parse_and_format_match_the_javascript_rules(self):
        for case in FIXTURE['parse']:
            units, message = mix.parse_units(case['text'])
            self.assertEqual(message is None, case['ok'], case['text'])
            self.assertEqual(units, case['units'], case['text'])
        for units in range(-50000, 50001, 7):
            self.assertEqual(mix.parse_units(mix.format_units(units))[0], units)
        self.assertEqual([mix.format_units(u) for u in (6000, -2500, 0, 10000, 1, -50000)], ['0.6', '-0.25', '0', '1', '0.0001', '-5'])

    def test_sequences_displacement_and_exact_counts_match(self):
        checked = 0
        for case in FIXTURE['rules']:
            analysis = mix.Analysis(case['members'])
            self.assertEqual(analysis.valid, case['valid'])
            self.assertEqual(analysis.dmax, case['dmax'])
            self.assertEqual(str(analysis.space) if analysis.space is not None else None, case['space'])
            self.assertEqual([[row.sequence.at(i) for i in range(row.sequence.size)] for row in analysis.free], case['sequences'])
            for limit, expected in case['counts'].items():
                self.assertEqual(str(mix.count_space(analysis.free, min(int(limit), analysis.dmax))), expected, limit)
                checked += 1
            if analysis.valid:
                half = case['half_limit']
                got = []
                for vector in mix.enumerate_vectors(analysis.free, half):
                    got.append(list(vector))
                    if len(got) >= 25:
                        break
                self.assertEqual(got, case['first_vectors'])
        self.assertGreater(checked, 100)

    def test_invalid_rule_fields_are_identical(self):
        for case in FIXTURE['invalid']:
            errors, parsed, locked = mix.inspect_rule(case['rule'])
            self.assertEqual(sorted(errors), case['error_fields'], case['rule'])
            self.assertEqual(locked, case['locked'])

    def test_spec_example(self):
        analysis = mix.Analysis(draft()['mix']['members'])
        self.assertEqual((analysis.valid, analysis.dmax, analysis.space), (True, 12000, 135))
        self.assertEqual(mix.count_space(analysis.free, 6000), 77)
        self.assertEqual(mix.count_space(analysis.free, 0), 1)
        wide = mix.Analysis([{'rule': rule('0', '-5', '5', '0.01')} for _ in range(16)])
        self.assertEqual(wide.space, 1001 ** 16)
        self.assertIsNone(mix.count_space(wide.free, 40000))


class ValidationTests(unittest.TestCase):
    def test_a_clean_recipe_is_recomputed_from_raw_text(self):
        payload = draft(candidates=[row('C01', '0.6', '-0.2', 'baseline'), row('C02', '0.8', '-0.4')])
        payload['mix']['candidates'][1]['displacement'] = 999999   # a client's own claim is ignored
        recipe = mix.validate_recipe(payload)
        self.assertEqual([c.id for c in recipe.candidates], ['C01', 'C02'])
        self.assertEqual(recipe.candidates[1].displacement, 4000)
        self.assertEqual(recipe.candidates[1].weights, {uid(11): 8000, uid(12): -4000, uid(13): 5000})
        self.assertEqual((recipe.limit, recipe.slots, recipe.images, recipe.space), (6000, 16, 16, 77))

    def test_every_issue_is_reported_together(self):
        payload = draft(members=[{'ref': {**ref(11), 'contentHash': 'zz'}, 'rule': rule('0.6', '0.8', '1.0')}, {'ref': ref(12), 'rule': rule('0.6', '0.2', '1', '0')}])
        scopes = {(issue['scope'], issue.get('field')) for issue in issues_of(payload)}
        self.assertIn(('members', None), scopes)
        self.assertIn(('rule', 'base'), scopes)
        self.assertIn(('rule', 'step'), scopes)

    def test_candidates_are_validated_not_trusted(self):
        payload = draft(candidates=[row('C01', '0.6', '-0.2'), row('C02', '0.65', '-0.2'), row('C03', '0.6', '-0.2'), row('C04', '1.0', '0.6'), row('C05', '0.', '-0.2'),
                                    {'id': 'C06', 'source': 'manual', 'raw': {uid(11): '0.7'}}])
        text = ' '.join(issue['message'] for issue in issues_of(payload))
        for expected in ('候选序列', '与 C01 完全重复', '总位移 1.2 超过上限 0.6', '需要普通十进制数', '缺少权重'):
            self.assertIn(expected, text)
        self.assertNotIn('C01：', text)

    def test_displacement_budget_seeds_and_limits(self):
        base = lambda **k: draft(candidates=[row('C01', '0.6', '-0.2')], **k)
        self.assertTrue(any(i['scope'] == 'displacement' for i in issues_of(base(max_displacement='1.3'))))
        self.assertEqual(issues_of(base(max_displacement='0')), [])
        for custom in ('', '0', '-4', '3.5', 'abc', '12345678'):
            self.assertTrue(any(i['scope'] == 'budget' for i in issues_of(base(budget={'preset': 'custom', 'custom': custom}))), custom)
        payload = base(budget={'preset': 'custom', 'custom': '5'})
        payload['repeat_count'] = 8
        self.assertTrue(any('少于样本数' in i['message'] for i in issues_of(payload)))
        payload = base(budget={'preset': 'custom', 'custom': '1030'})
        payload['repeat_count'] = 5
        self.assertTrue(any('超出现有执行上限' in i['message'] for i in issues_of(payload)))
        payload = base()
        payload['requested_concurrency'] = 11
        self.assertTrue(any('请求并发' in i['message'] for i in issues_of(payload)))
        self.assertTrue(any(i['scope'] == 'sampling' for i in issues_of(base(sampling_seed='-1'))))
        self.assertTrue(any(i['scope'] == 'sampling' for i in issues_of(base(sampling_seed=str(2 ** 64)))))

    def test_unknown_versions_and_fields_are_refused(self):
        with self.assertRaises(mix.MixError) as caught:
            mix.validate_recipe(draft(version=2))
        self.assertEqual(caught.exception.code, 'version')
        with self.assertRaises(mix.MixError) as caught:
            mix.validate_recipe(draft(sneaky=True))
        self.assertEqual(caught.exception.code, 'recipe')
        with self.assertRaises(mix.MixError):
            mix.validate_recipe({'baseline': {}})

    def test_global_and_test_roles_cannot_overlap(self):
        for kind, global_item in (('instance', global_lora(11, '0')), ('path', global_lora(99, value=ref(11)['value'])), ('content', global_lora(99, contentHash=ref(11)['contentHash']))):
            issues = issues_of(draft(candidates=[row('C01', '0.6', '-0.2')], loras=[global_item]))
            self.assertTrue(any(i['scope'] == 'role' and i.get('kind') == kind for i in issues), kind)
        for dormant in (global_lora(99, contentHash=ref(11)['contentHash'], enabled=False), global_lora(99, '0', contentHash=ref(11)['contentHash'])):
            self.assertEqual(issues_of(draft(candidates=[row('C01', '0.6', '-0.2')], loras=[dormant])), [], 'a never-fused global is not a conflict')

    def test_the_combined_active_limit_counts_nonzero_members(self):
        loras = [global_lora(100 + i) for i in range(15)]
        members = [{'ref': ref(11), 'rule': rule('0.6', '0.2', '1.0')}, {'ref': ref(12), 'rule': rule('-0.2', '-0.8', '0.6')}]
        payload = draft(members=members, candidates=[row('C01', '0.6', '-0.2'), row('C02', '0.6', '0')], loras=loras, max_displacement='1.2')
        issues = issues_of(payload)
        self.assertEqual([i['candidate'] for i in issues if i['scope'] == 'candidate'], ['C01'])
        self.assertIn('超过 16 上限', issues[0]['message'])

    def test_all_locked_is_exactly_one_profile(self):
        members = [{'ref': ref(11), 'rule': {'locked': True, 'base': '0.4'}}, {'ref': ref(12), 'rule': {'locked': True, 'base': '-0.2'}}]
        recipe = mix.validate_recipe(draft(members=members, candidates=[], max_displacement='9'))
        self.assertEqual(len(recipe.candidates), 1)
        self.assertEqual(recipe.candidates[0].weights, {uid(11): 4000, uid(12): -2000})
        self.assertEqual((recipe.limit, recipe.space), (0, 1))

    def test_unknown_candidate_ids_duplicate_ids_and_session_shape(self):
        payload = draft(candidates=[row('C01', '0.6', '-0.2'), row('C01', '0.8', '-0.4')])
        self.assertTrue(any('编号重复' in i['message'] for i in issues_of(payload)))
        self.assertTrue(any(i['scope'] == 'session' for i in issues_of(draft(candidates=[row('C01', '0.6', '-0.2')], session={'id': 'nope', 'revision': 0}))))
        self.assertTrue(any(i['scope'] == 'session' for i in issues_of(draft(candidates=[row('C01', '0.6', '-0.2')], session={'id': None, 'revision': -1}))))


class CompileTests(unittest.TestCase):
    def registry(self, loras, members):
        instances = {item['instanceId']: {**item, 'engine': 'Anima', 'status': 'present'} for item in [*loras, *[m['ref'] for m in members]]}
        return {'schemaVersion': 1, 'instances': instances}

    def test_compile_keeps_the_global_background_and_appends_ordered_tests(self):
        loras = [global_lora(31, '0.5'), global_lora(32, '1.25', precision=2)]
        payload = draft(candidates=[row('C01', '0.6', '-0.2', 'baseline'), row('C02', '0.8', '0')], loras=loras, max_displacement='1.2')
        recipe = mix.validate_recipe(payload)
        xyz = mix.compile_xyz_draft(recipe)
        model = XYZExperimentInput.model_validate(xyz)       # strict DTO: no search-only field may leak
        self.assertEqual(set(xyz), {'schema_version', 'kind', 'engine', 'baseline', 'axes', 'repeat_count', 'sample_seeds', 'requested_concurrency', 'inherit_postprocess', 'group_by_resources'})
        self.assertEqual(xyz['axes']['x']['field'], 'lora_profile')
        self.assertEqual([v['label'] for v in xyz['axes']['x']['values']], ['C01', 'C02'])
        for value in xyz['axes']['x']['values']:
            self.assertEqual(value['value']['mode'], 'keep_background')
            self.assertEqual(value['value']['includePresetText'], False)
            self.assertEqual(value['value']['test_targets'], [uid(11), uid(12), uid(13)])
            self.assertTrue(all(m['precision'] == 4 and m['enabled'] is True for m in value['value']['members']))
        registry = self.registry(loras, payload['mix']['members'])
        catalog = {('Anima', f, v) for f, v in model.baseline.parameters.model_dump().items() if f in CAPABILITIES['engines']['Anima']['model_fields']}
        plan = compile_xyz(model, model_catalog=catalog, lora_registry=registry)
        self.assertEqual(plan.errors, [])
        self.assertEqual(plan.total_cells, 2)
        first, second = plan.cells
        self.assertEqual([e['instanceId'] for e in first.lora_references], [uid(31), uid(32), uid(11), uid(12), uid(13)])
        self.assertEqual([e['weight'] for e in first.lora_references], [0.5, 1.25, 0.6, -0.2, 0.5], 'globals unchanged; tests at their weights')
        self.assertEqual([e['instanceId'] for e in second.lora_references], [uid(31), uid(32), uid(11), uid(13)], 'a zero-weight member is not fused')
        self.assertEqual([e['weight'] for e in second.lora_references][2:], [0.8, 0.5])
        self.assertEqual([item['weight'] for item in first.request['loras']][:2], [0.5, 1.25])
        profile = xyz['axes']['x']['values'][1]['value']
        self.assertEqual([m['weight'] for m in profile['members']], ['0.8', '0', '0.5'], 'the zero weight stays auditable in the frozen profile')
        self.assertEqual(first.labels['x'], 'C01')

    def test_a_global_is_never_cleared_or_scaled(self):
        payload = draft(candidates=[row('C01', '0.6', '-0.2')], loras=[global_lora(31, '0.5')])
        recipe = mix.validate_recipe(payload)
        before = copy.deepcopy(payload['baseline'])
        xyz = mix.compile_xyz_draft(recipe)
        self.assertEqual(xyz['baseline']['loras'], before['loras'])
        self.assertEqual(payload['baseline'], before, 'compilation never edits the recipe')
        self.assertNotIn(xyz['axes']['x']['values'][0]['value']['mode'], ('full', 'no_loras'))

    def test_seeds_and_multi_seed_pass_through_to_the_existing_policy(self):
        payload = draft(candidates=[row('C01', '0.6', '-0.2')])
        payload['repeat_count'] = 3
        payload['sample_seeds'] = ['7', '8', '9']
        payload['mix']['budget'] = {'preset': '16', 'custom': ''}
        recipe = mix.validate_recipe(payload)
        self.assertEqual(recipe.slots, 5)
        self.assertEqual(mix.compile_xyz_draft(recipe)['sample_seeds'], ['7', '8', '9'])
        self.assertEqual(mix.explicit_seeds(recipe), ['7', '8', '9'])
        random_draft = draft(candidates=[row('C01', '0.6', '-0.2')])
        random_draft['baseline']['seed_mode'] = 'random'
        self.assertIsNone(mix.explicit_seeds(mix.validate_recipe(random_draft)))


class ContextTests(unittest.TestCase):
    def recipe(self, **kwargs):
        loras = kwargs.pop('loras', [global_lora(31, '0.5')])
        return mix.validate_recipe(draft(candidates=[row('C01', '0.6', '-0.2'), row('C02', '0.8', '-0.4')], loras=loras, **kwargs))

    def test_context_is_stable_and_every_pixel_input_changes_it(self):
        base = self.recipe()
        context = mix.context_hash(base, ['7'])
        self.assertEqual(context, mix.context_hash(self.recipe(), ['7']))
        self.assertEqual(len(context), 64)
        self.assertNotEqual(context, mix.context_hash(base, ['8']))
        self.assertNotEqual(context, mix.context_hash(base, ['7', '8']))
        changed = self.recipe(loras=[global_lora(31, '0.6')])
        self.assertNotEqual(context, mix.context_hash(changed, ['7']), 'a global weight is part of the context')
        other_prompt = mix.validate_recipe(draft(candidates=[row('C01', '0.6', '-0.2')], loras=[global_lora(31, '0.5')]))
        other_prompt.baseline['parameters']['prompt'] = 'different'
        self.assertNotEqual(context, mix.context_hash(other_prompt, ['7']))
        # The tested range, step, preference, budget and sampling seed are not pixels: same context.
        relaxed = self.recipe(max_displacement='0.4', sampling_seed='99', budget={'preset': '64', 'custom': ''})
        self.assertEqual(context, mix.context_hash(relaxed, ['7']))

    def test_candidate_keys_depend_on_canonical_weights_not_names(self):
        recipe = self.recipe()
        context = mix.context_hash(recipe, ['7'])
        keys = [mix.candidate_key(context, recipe, c.weights) for c in recipe.candidates]
        self.assertEqual(len(set(keys)), 2)
        renamed = self.recipe()
        for c in renamed.candidates:
            c.id = 'Z' + c.id
        self.assertEqual(keys, [mix.candidate_key(context, renamed, c.weights) for c in renamed.candidates])
        self.assertNotEqual(keys[0], mix.candidate_key(mix.context_hash(recipe, ['8']), recipe, recipe.candidates[0].weights))


def fake_record(recipe, *, cells, status, released=False, session='a' * 32, round_number=1, seeds=('7',)):
    context = mix.context_hash(recipe, list(seeds))
    origin = mix.origin_block(recipe, session_id=session, round_number=round_number, context=context, seeds=list(seeds), submission_id='00000000-0000-4000-8000-0000000000aa')
    definition = {'engine': recipe.engine, 'created_at': 100.0 + round_number, 'origin': origin, 'draft': {'repeat_count': len(seeds)}}
    state = {'status': status, 'cells': {cid: {'status': value} for cid, value in cells.items()}, **({'mix_released': {'at': 1}} if released else {})}
    return {'definition': definition, 'state': state}, context


class RoundTests(unittest.TestCase):
    def recipe(self):
        return mix.validate_recipe(draft(candidates=[row('C01', '0.6', '-0.2', 'baseline'), row('C02', '0.8', '-0.4'), row('C03', '1.0', '0')], loras=[global_lora(31)]))

    def test_origin_block_freezes_rules_globals_keys_and_seeds(self):
        record, context = fake_record(self.recipe(), cells={'0:0:0:0': 'pending', '1:0:0:0': 'pending', '2:0:0:0': 'pending'}, status='queued')
        origin = record['definition']['origin']
        self.assertEqual(origin['kind'], mix.ORIGIN_KIND)
        self.assertEqual([c['x_index'] for c in origin['candidates']], [0, 1, 2])
        self.assertEqual(origin['global_loras'][0]['instanceId'], uid(31))
        self.assertEqual(origin['rules']['limit_units'], 6000)
        self.assertEqual(origin['algorithm'], {'name': mix.ALGORITHM, 'sampling_seed': '1', 'cursor': 0})
        self.assertEqual(origin['context_hash'], context)
        json.dumps(origin)   # the block is plain JSON: it is persisted verbatim in the experiment definition

    def test_candidate_states_distinguish_tested_reserved_failed_and_cancelled(self):
        recipe = self.recipe()
        cells = {'0:0:0:0': 'complete', '1:0:0:0': 'pending', '2:0:0:0': 'error'}
        record, _ = fake_record(recipe, cells=cells, status='running')
        self.assertEqual({k: v['state'] for k, v in mix.candidate_states(record['definition'], record['state']).items()}, {'C01': 'tested', 'C02': 'reserved', 'C03': 'reserved'})
        record, _ = fake_record(recipe, cells=cells, status='complete')
        self.assertEqual({k: v['state'] for k, v in mix.candidate_states(record['definition'], record['state']).items()}, {'C01': 'tested', 'C02': 'failed', 'C03': 'failed'})
        record, _ = fake_record(recipe, cells={**cells, '1:0:0:0': 'cancelled'}, status='cancelled')
        self.assertEqual(mix.candidate_states(record['definition'], record['state'])['C02']['state'], 'cancelled')
        record, _ = fake_record(recipe, cells=cells, status='interrupted')
        self.assertEqual(mix.candidate_states(record['definition'], record['state'])['C02']['state'], 'reserved', 'an interrupted round still holds its unexecuted vectors')
        record, _ = fake_record(recipe, cells=cells, status='interrupted', released=True)
        self.assertEqual(mix.candidate_states(record['definition'], record['state'])['C02']['state'], 'failed', 'an explicit release frees them')
        record, _ = fake_record(recipe, cells={'0:0:0:0': 'complete', '1:0:0:0': 'complete', '2:0:0:0': 'missing'}, status='complete')
        self.assertEqual(mix.candidate_states(record['definition'], record['state'])['C03']['state'], 'failed', 'a missing original is not a tested vector')

    def test_multi_sample_candidates_need_every_sample(self):
        recipe = self.recipe()
        cells = {f'{x}:0:0:{s}': 'complete' for x in range(3) for s in range(2)}
        cells['1:0:0:1'] = 'error'
        record, _ = fake_record(recipe, cells=cells, status='complete', seeds=('7', '8'))
        states = mix.candidate_states(record['definition'], record['state'])
        self.assertEqual((states['C01']['state'], states['C02']['state']), ('tested', 'failed'))
        self.assertEqual((states['C02']['complete'], states['C02']['samples']), (1, 2), 'partial success is recorded separately, never as a completed profile')

    def test_used_candidates_follow_the_context_across_sessions(self):
        recipe = self.recipe()
        first, context = fake_record(recipe, cells={'0:0:0:0': 'complete', '1:0:0:0': 'complete', '2:0:0:0': 'failed'}, status='complete', session='a' * 32)
        second, same = fake_record(recipe, cells={'0:0:0:0': 'pending', '1:0:0:0': 'pending', '2:0:0:0': 'pending'}, status='queued', session='b' * 32, round_number=1)
        other_seed, different = fake_record(recipe, cells={'0:0:0:0': 'complete', '1:0:0:0': 'complete', '2:0:0:0': 'complete'}, status='complete', seeds=('8',), session='c' * 32)
        records = {'x1': first, 'x2': second, 'x3': other_seed}
        self.assertEqual(context, same)
        self.assertNotEqual(context, different)
        used = mix.used_candidates(records, 'Anima', context)
        self.assertEqual(len(used), 6, 'both sessions of one context; the other seed is a different context')
        self.assertEqual({entry['session_id'] for entry in used}, {'a' * 32, 'b' * 32})
        self.assertEqual(sorted({e['state'] for e in used}), ['failed', 'reserved', 'tested'])
        self.assertEqual(mix.used_candidates(records, 'SD', context), [])
        self.assertEqual({item['id'] for item in mix.session_summaries(records, 'Anima')}, {'a' * 32, 'b' * 32, 'c' * 32})
        self.assertEqual(mix.session_summaries(records, 'SD'), [])
        self.assertEqual([r[0] for r in mix.session_rounds(records, 'a' * 32)], [1])

    def test_remaining_vectors_exclude_used_and_page(self):
        recipe = self.recipe()
        record, context = fake_record(recipe, cells={'0:0:0:0': 'complete', '1:0:0:0': 'complete', '2:0:0:0': 'complete'}, status='complete')
        origin = record['definition']['origin']
        used = {c['key'] for c in origin['candidates']}
        everything = mix.remaining_vectors(origin, set(), 0, 1000)
        self.assertEqual((everything['total'], len(everything['rows']), everything['complete']), (77, 77, True))
        left = mix.remaining_vectors(origin, used, 0, 1000)
        self.assertEqual(len(left['rows']), 74)
        keys = {mix.sha256(context + '|' + ','.join(f'{iid}:{w}' for iid, w in row['weights'].items())) for row in left['rows']}
        self.assertFalse(keys & used)
        page = mix.remaining_vectors(origin, used, 10, 20)
        self.assertEqual(len(page['rows']), 20)
        self.assertEqual(page['rows'], left['rows'][10:30])
        self.assertFalse(page['complete'])
        self.assertEqual(page['next_offset'], 30)
        self.assertTrue(all(row['displacement'] <= 6000 for row in left['rows']))



class CandidateGridExportTests(unittest.TestCase):
    """The candidate-grid export snapshot: seed placement and the optional numbered LoRA weights, through the real layout."""

    def round_snapshot(self, *, samples=1, seeds=None, weights=None, locked=False, globals_=()):
        members = [{'ref': ref(1, name='style/NK/ashima_v4.safetensors'), 'rule': {**rule('0.5', '0.3', '0.9'), 'locked': locked}},
                   {'ref': ref(2, name='Kiriaina'), 'rule': rule('0.5', '0.2', '1', '0.15')}, {'ref': ref(3), 'rule': rule('0.6', '0.2', '1')}]
        weights = weights or [(5000, 5000, 6000), (9000, 9500, 10000), (3000, 9500, 9000), (7000, 9500, 2000), (8000, 2000, 9000)]
        candidates = [{'id': f'C{i + 1:02d}', 'key': f'{i:064x}', 'source': 'baseline' if i == 0 else 'sampled', 'x_index': i,
                       'displacement': sum(abs(w - b) for w, b in zip(row, (5000, 5000, 6000))),
                       'weights': {uid(n + 1): w for n, w in enumerate(row)}} for i, row in enumerate(weights)]
        seeds = seeds or [str(1000 + sample) for sample in range(samples)]
        cells = [{'id': f'{x}:0:0:{sample}', 'coordinate': [x, 0, 0, sample], 'labels': {}, 'status': 'complete',
                  'seed': seeds[sample] if isinstance(seeds[sample], str) else seeds[sample][x], 'width': 768, 'height': 1280, 'path': None, 'alpha': False}
                 for sample in range(samples) for x in range(len(candidates))]
        snapshot = {'experiment_id': 'e' * 32, 'digest': 'd' * 64, 'engine': 'Anima', 'axes': {}, 'dimensions': {}, 'repeat_count': samples,
                    'baseline': {'width': 768, 'height': 1280, 'steps': 25, 'cfg': 5}, 'created_at': 0, 'cells': cells}
        origin = {'session_id': 's' * 32, 'round': 1, 'context_hash': 'c' * 64, 'algorithm': {}, 'seeds': [s if isinstance(s, str) else s[0] for s in seeds],
                  'rules': {'members': members}, 'global_loras': list(globals_), 'candidates': candidates}
        return snapshot, origin

    def layout(self, snapshot, lora_weights=False, captions=True):
        from backend import xyz_grid
        options = {'scope': 'all', 'page': None, 'mode': 'original', 'percent': 100, 'theme': 'light', 'arrangement': 'vertical',
                   'captions': captions, 'lora_weights': lora_weights}
        return xyz_grid.compute_layout(snapshot, options)

    def test_a_shared_seed_moves_from_every_caption_into_the_summary(self):
        snapshot = mix.candidate_grid(*self.round_snapshot(), 3)
        self.assertIn('种子 1000', snapshot['subtitle'])
        captions = [cell['caption'] for cell in snapshot['cells']]
        self.assertEqual(captions[:2], ['C01 · 基线 · Δ0', 'C02 · Δ1.25'])
        self.assertFalse(any('种子' in caption for caption in captions))
        self.assertEqual(snapshot['search']['seed_placement'], 'summary')
        self.assertNotIn('legend', snapshot, 'LoRA weights are off by default')
        self.assertFalse(any('tokens' in cell for cell in snapshot['cells']))

    def test_seeds_that_differ_inside_a_sample_stay_on_each_image(self):
        snapshot = mix.candidate_grid(*self.round_snapshot(seeds=[['1', '2', '3', '4', '5']]), 3)
        self.assertEqual(snapshot['cells'][1]['caption'], 'C02 · Δ1.25 · 种子 2')
        self.assertNotIn('种子', snapshot['subtitle'])
        self.assertEqual(snapshot['search']['seed_placement'], 'caption')

    def test_several_samples_carry_their_seed_in_the_sample_heading(self):
        snapshot = mix.candidate_grid(*self.round_snapshot(samples=2), 3)
        self.assertEqual(snapshot['sample_labels'], ['样本 1 · 种子 1000', '样本 2 · 种子 1001'])
        layout = self.layout(snapshot)
        headings = [text['text'] for text in layout['ops']['texts'] if text['text'].startswith('样本')]
        self.assertEqual(headings, ['样本 1 · 种子 1000', '样本 2 · 种子 1001'])
        self.assertFalse(any('种子' in cell['caption'] for cell in snapshot['cells']))

    def test_lora_weights_number_the_test_loras_and_each_image_lists_them(self):
        snapshot = mix.candidate_grid(*self.round_snapshot(locked=True, globals_=[global_lora(9, name='BG', weight=0.8)]), 3, lora_weights=True)
        self.assertEqual(snapshot['legend']['items'], ['① ashima_v4（锁定）', '② Kiriaina', '③ L3'], 'file names lose folders and extension')
        self.assertEqual(snapshot['legend']['note'], '全局 LoRA（每格相同）：BG 0.8')
        self.assertEqual(snapshot['cells'][1]['tokens'], ['① 0.9', '② 0.95', '③ 1'])
        self.assertTrue(snapshot['search']['lora_weights'])
        self.assertEqual([item['marker'] for item in snapshot['search']['lora_legend']], ['①', '②', '③'])

    def test_the_weights_share_one_slot_grid_and_the_option_only_adds_height(self):
        off = self.layout(mix.candidate_grid(*self.round_snapshot(), 3))
        on = self.layout(mix.candidate_grid(*self.round_snapshot(), 3, lora_weights=True), lora_weights=True)
        self.assertEqual((on['width'], on['columns'], on['rows']), (off['width'], off['columns'], off['rows']))
        self.assertGreater(on['height'], off['height'])
        cells = [entry for entry in on['ops']['cells'] if entry.get('tokens')]
        self.assertEqual(len(cells), 5)
        # Relative to each cell's slot, every token sits at the same place in every cell: values line up across the grid.
        offsets = {tuple((round(t['xy'][0] - e['slot'][0]), round(t['xy'][1] - e['slot'][1])) for t in e['tokens']) for e in cells}
        self.assertEqual(len(offsets), 1, offsets)
        # Tokens stay inside their column and below the caption line; captions keep their place.
        for entry in cells:
            self.assertGreater(entry['tokens'][0]['xy'][1], entry['caption']['xy'][1])
            self.assertLessEqual(max(t['xy'][0] for t in entry['tokens']) - entry['slot'][0], entry['slot'][2])
        off_caption = {e['cell']: e['caption']['xy'][0] - e['slot'][0] for e in off['ops']['cells'] if e.get('caption')}
        self.assertTrue(all(e['caption']['xy'][0] - e['slot'][0] == off_caption[e['cell']] for e in cells))
        legend = [text['text'] for text in on['ops']['texts'] if text['text'].startswith(('①', '②', '③'))]
        self.assertEqual(legend, ['① ashima_v4', '② Kiriaina', '③ L3'])
        # Without captions the weights still render, directly under the image.
        bare = self.layout(mix.candidate_grid(*self.round_snapshot(), 3, lora_weights=True), lora_weights=True, captions=False)
        self.assertTrue(all(not e.get('caption') and e.get('tokens') for e in bare['ops']['cells'] if e['cell']))


if __name__ == '__main__':
    unittest.main()
