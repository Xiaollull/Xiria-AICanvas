"""QPlan2/P2 service integration: plan, confirm, sessions, reservations, remaining and provenance.

Imports the real inference service against isolated state/output/model folders with a real LoRA registry and
real (tiny) LoRA files, so identity, byte and admission checks are the production ones. The executor is mocked:
no GPU is touched, and no experiment cell is actually generated here.
"""
import copy
import hashlib
import json
import struct
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from backend.lora_file_identity import file_identity
from backend import lora_mix_search as mix


def rule(base, low, high, step='0.1', preference='balanced'):
    return {'locked': False, 'base': base, 'min': low, 'max': high, 'step': step, 'preference': preference}


class MixServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from backend import inference_server
        cls.server = inference_server

    def setUp(self):
        server = self.server
        self.root = Path(tempfile.mkdtemp(prefix='mix-service-'))
        self.state, self.outputs, self.models = self.root / 'state', self.root / 'outputs', self.root / 'models'
        for path in (self.state, self.outputs, self.models):
            path.mkdir()
        (self.models / 'model.safetensors').write_bytes(b'checkpoint placeholder')
        self.patches = [mock.patch.object(server, 'STATE_DIRECTORY', self.state), mock.patch.object(server, 'OUTPUT_DIRECTORY', self.outputs),
                        mock.patch.object(server, 'model_roots', return_value=(self.models, self.models)), mock.patch.object(server, '_xyz_store', None)]
        for patch in self.patches:
            patch.start()
        self.submit = mock.patch.object(server.executor, 'submit')
        self.executor = self.submit.start()
        self.draws = []
        self.rng = mock.patch.object(server, 'secure_random_uint64', side_effect=lambda: self.draws.append(1) or (2 ** 64 - len(self.draws)))
        self.rng.start()
        with server.jobs_lock:
            self.before = set(server.jobs)
        self.refs = {}
        self.make_registry({'G': b'global bytes', 'A': b'adapter A', 'B': b'adapter B', 'C': b'adapter C', 'D': b'adapter D', 'G2': b'global bytes'})

    def tearDown(self):
        server = self.server
        self.rng.stop()
        self.submit.stop()
        for patch in reversed(self.patches):
            patch.stop()
        with server.jobs_lock:
            for job_id in set(server.jobs) - self.before:
                server.jobs.pop(job_id, None)
                server.job_controls.pop(job_id, None)

    # Fixtures ----------------------------------------------------------------------------------

    def make_registry(self, files):
        instances = {}
        header = json.dumps({'__metadata__': {}}).encode()
        for name, payload in files.items():
            path = self.models / f'{name}.safetensors'
            path.write_bytes(struct.pack('<Q', len(header)) + header + payload)
            instance = {'instanceId': str(uuid.uuid4()), 'engine': 'SD', 'rootId': 'local-SD', 'value': path.name, 'history': [], 'status': 'present', 'contentRevision': 1,
                        'contentHash': hashlib.sha256(path.read_bytes()).hexdigest(), 'fsIdentity': file_identity(path)}
            instances[instance['instanceId']] = instance
            self.refs[name] = {'value': path.name, 'name': name, 'category': 'style', 'identitySchema': 1, 'instanceId': instance['instanceId'],
                               'contentHash': instance['contentHash'], 'contentRevision': 1}
        library = self.state / 'lora-library'
        library.mkdir(exist_ok=True)
        (library / 'registry.json').write_text(json.dumps({'schemaVersion': 1, 'revision': 1, 'instances': instances, 'substitutions': []}), encoding='utf-8')
        (library / 'publication.json').write_text(json.dumps({'schemaVersion': 1, 'phase': 'published', 'revision': 1}), encoding='utf-8')

    def global_lora(self, name='G', weight='0.5'):
        return {**self.refs[name], 'weight': weight, 'precision': 4, 'enabled': True}

    def draft(self, candidates=None, members=None, loras=None, session=None, **mix_extra):
        members = members if members is not None else [{'ref': self.refs['A'], 'rule': rule('0.6', '0.2', '1.0', preference='max')},
                                                       {'ref': self.refs['B'], 'rule': rule('-0.2', '-0.8', '0.6', preference='min')},
                                                       {'ref': self.refs['C'], 'rule': {'locked': True, 'base': '0.5'}}]
        ids = [m['ref']['instanceId'] for m in members[:2]]
        candidates = candidates if candidates is not None else [{'id': 'C01', 'source': 'baseline', 'raw': dict(zip(ids, ['0.6', '-0.2']))},
                                                                {'id': 'C02', 'source': 'sampled', 'raw': dict(zip(ids, ['0.8', '-0.4']))},
                                                                {'id': 'C03', 'source': 'sampled', 'raw': dict(zip(ids, ['0.4', '0']))}]
        parameters = {'engine': 'SD', 'checkpoint': 'model.safetensors', 'prompt': 'cat, blue sky', 'negative_prompt': 'bad', 'width': 512, 'height': 512, 'steps': 4, 'cfg': 7,
                      'seed': '18446744073709551615', 'sampler': 'euler', 'scheduler': 'normal', 'guidance': 'none', 'pag': {'scale': 0.3, 'applied_layers': 'mid'}, 'postprocess_chain': []}
        block = {'version': 1, 'members': members, 'max_displacement': '0.6', 'budget': {'preset': '16', 'custom': ''}, 'include_baseline': True, 'sampling_seed': '1', 'cursor': 0,
                 'candidates': candidates, 'next_id': 4, 'proposed_key': '', 'session': session or {'id': None, 'revision': 0}}
        block.update(mix_extra)
        return {'schema_version': 1, 'kind': 'xyz_draft', 'engine': 'SD', 'axes': {'x': None, 'y': None, 'z': None}, 'repeat_count': 1, 'sample_seeds': None,
                'requested_concurrency': 1, 'inherit_postprocess': False, 'group_by_resources': False,
                'baseline': {'parameters': parameters, 'loras': [self.global_lora()] if loras is None else loras, 'group_snapshots': [], 'prompt_without_groups': 'cat, blue sky',
                             'group_prompt': '', 'seed_mode': 'fixed', 'library_revision': 1}, 'mix': block}

    def submission(self, draft=None, submission_id=None, on_error='pause'):
        return {'schema_version': 1, 'kind': 'lora_mix_submit', 'submission_id': submission_id or str(uuid.uuid4()), 'draft': draft or self.draft(), 'on_error': on_error}

    def http(self, revision='1'):
        return SimpleNamespace(headers={'x-xirai-lora-revision': revision})

    def confirm(self, payload=None, http=None):
        return self.server.confirm_mix_round(payload or self.submission(), http or self.http())

    def refused(self, payload, status, http=None):
        jobs_before = set(self.server.jobs)
        with self.assertRaises(self.server.HTTPException) as raised:
            self.server.confirm_mix_round(payload, http or self.http())
        self.assertEqual(raised.exception.status_code, status, raised.exception.detail)
        self.assertEqual(set(self.server.jobs), jobs_before, 'a refusal never queues anything')
        return raised.exception.detail

    def complete(self, experiment_id, candidate_indexes=None):
        store = self.server.xyz_store()
        with store.lock:
            record = store.records[experiment_id]
            for cell_id, cell in record['state']['cells'].items():
                if candidate_indexes is None or int(cell_id.split(':')[0]) in candidate_indexes:
                    cell['status'] = 'complete'
            if all(cell['status'] == 'complete' for cell in record['state']['cells'].values()):
                record['state']['status'] = 'complete'
            store.save(experiment_id)

    # Tests -------------------------------------------------------------------------------------

    def test_capabilities_and_a_clean_plan_draw_nothing(self):
        capabilities = self.server.mix_capabilities()
        self.assertTrue(capabilities['executor_ready'])
        self.assertEqual(capabilities['algorithm'], mix.ALGORITHM)
        plan = self.server.plan_mix_round({'draft': self.draft()}, self.http())
        self.assertTrue(plan['ok'], plan['errors'])
        self.assertEqual((plan['summary']['profiles'], plan['summary']['samples'], plan['summary']['cells'], plan['summary']['new']), (3, 1, 3, 3))
        self.assertEqual(plan['summary']['duplicates'], [])
        self.assertEqual(len(plan['summary']['context_hash']), 64)
        self.assertEqual(plan['summary']['space'], '77')
        self.assertEqual(self.draws, [], 'preview draws no random seed')
        self.assertFalse((self.state / 'xyz-plots').exists() and any((self.state / 'xyz-plots').iterdir()), 'preview writes no record')

    def test_confirm_creates_one_round_with_origin_and_queues_it(self):
        result = self.confirm()
        experiment = result['experiment']
        self.assertEqual((result['session']['revision'], result['replayed']), (1, False))
        self.assertRegex(result['session']['id'], r'^[0-9a-f]{32}$')
        self.assertEqual(experiment['origin']['kind'], 'lora_mix_search')
        self.assertEqual((experiment['origin']['round'], experiment['origin']['candidates']), (1, 3))
        self.assertEqual(experiment['total_cells'], 3)
        self.executor.assert_called_once()
        record = self.server.xyz_store().records[experiment['id']]
        origin = record['definition']['origin']
        self.assertEqual([c['id'] for c in origin['candidates']], ['C01', 'C02', 'C03'])
        self.assertEqual(origin['seeds'], ['18446744073709551615'])
        self.assertEqual(origin['global_loras'][0]['instanceId'], self.refs['G']['instanceId'])
        # The stored experiment is the ordinary XYZ definition: one lora_profile axis, no search field in the strict draft.
        stored = record['definition']['draft']
        self.assertEqual(stored['axes']['x']['field'], 'lora_profile')
        self.assertNotIn('mix', stored)
        cells = record['definition']['plan']['cells']
        self.assertEqual([c['labels']['x'] for c in cells], ['C01', 'C02', 'C03'])
        first = [e['instanceId'] for e in cells[0]['lora_references']]
        self.assertEqual(first, [self.refs[n]['instanceId'] for n in ('G', 'A', 'B', 'C')], 'the global background stays first and unmodified; tests follow in X1 order')
        self.assertEqual([e['weight'] for e in cells[0]['lora_references']], [0.5, 0.6, -0.2, 0.5])
        # The parent job mirrors the origin so the queue can open the right workspace.
        parent = self.server.xyz_parent_fields(self.server.xyz_store().summary(experiment['id'], verify_assets=False))
        self.assertEqual(parent['origin']['kind'], 'lora_mix_search')
        self.assertEqual(parent['kind'], 'xyz_plot')

    def test_a_replayed_click_is_the_same_round_and_a_changed_one_is_refused(self):
        submission = self.submission()
        first = self.confirm(submission)
        again = self.confirm(copy.deepcopy(submission))
        self.assertTrue(again['replayed'])
        self.assertEqual(again['experiment']['id'], first['experiment']['id'])
        self.assertEqual(again['session'], first['session'])
        self.assertEqual(self.executor.call_count, 1)
        self.assertEqual(self.draws, [], 'a fixed seed never draws')
        changed = copy.deepcopy(submission)
        changed['draft']['mix']['candidates'][1]['raw'][self.refs['A']['instanceId']] = '1.0'
        detail = self.refused(changed, 409)
        self.assertIn('同一提交编号', json.dumps(detail, ensure_ascii=False) if not isinstance(detail, str) else detail)

    def test_reserved_and_tested_vectors_are_never_taken_twice(self):
        first = self.confirm()
        detail = self.refused(self.submission(self.draft(session={'id': first['session']['id'], 'revision': 1})), 409)
        self.assertEqual(detail['code'], 'candidate-used')
        self.assertEqual({e['candidate'] for e in detail['errors']}, {'C01', 'C02', 'C03'})
        self.assertTrue(all(e['state'] == 'reserved' for e in detail['errors']))
        self.complete(first['experiment']['id'])
        detail = self.refused(self.submission(self.draft(session={'id': first['session']['id'], 'revision': 1})), 409)
        self.assertTrue(all(e['state'] == 'tested' for e in detail['errors']))
        plan = self.server.plan_mix_round({'draft': self.draft(session={'id': first['session']['id'], 'revision': 1})}, self.http())
        self.assertFalse(plan['ok'])
        self.assertEqual([d['state'] for d in plan['summary']['duplicates']], ['tested'] * 3)
        used = self.server.mix_used({'draft': self.draft()})
        self.assertEqual(sorted((u['candidate_id'], u['state']) for u in used['used']), [('C01', 'tested'), ('C02', 'tested'), ('C03', 'tested')])
        # A continuation takes only fresh vectors and extends the same session.
        ids = [self.refs['A']['instanceId'], self.refs['B']['instanceId']]
        fresh = [{'id': 'C04', 'source': 'sampled', 'raw': dict(zip(ids, ['0.4', '-0.4']))}, {'id': 'C05', 'source': 'sampled', 'raw': dict(zip(ids, ['0.8', '0']))}]
        second = self.confirm(self.submission(self.draft(candidates=fresh, session={'id': first['session']['id'], 'revision': 1})))
        self.assertEqual((second['session']['id'], second['session']['revision']), (first['session']['id'], 2))
        view = self.server.get_mix_session(first['session']['id'])
        self.assertEqual((view['revision'], [r['round'] for r in view['rounds']]), (2, [1, 2]))
        self.assertEqual([c['state'] for c in view['rounds'][0]['candidates']], ['tested'] * 3)
        self.assertEqual([c['state'] for c in view['rounds'][1]['candidates']], ['reserved'] * 2)

    def test_a_stale_session_revision_or_a_second_tab_cannot_take_the_same_slot(self):
        first = self.confirm()
        sid = first['session']['id']
        ids = [self.refs['A']['instanceId'], self.refs['B']['instanceId']]
        one = [{'id': 'C04', 'source': 'manual', 'raw': dict(zip(ids, ['0.4', '-0.4']))}]
        other = [{'id': 'C05', 'source': 'manual', 'raw': dict(zip(ids, ['0.8', '0']))}]
        self.confirm(self.submission(self.draft(candidates=one, session={'id': sid, 'revision': 1})))
        detail = self.refused(self.submission(self.draft(candidates=other, session={'id': sid, 'revision': 1})), 409)
        self.assertEqual(detail['code'], 'stale-session')
        self.assertEqual(detail['revision'], 2)
        self.assertEqual(self.server.get_mix_session(sid)['revision'], 2)

    def test_failed_validation_reserves_and_draws_nothing(self):
        payload = self.submission(self.draft(candidates=[{'id': 'C01', 'source': 'manual', 'raw': {self.refs['A']['instanceId']: '0.65', self.refs['B']['instanceId']: '-0.2'}}]))
        detail = self.refused(payload, 422)
        self.assertTrue(any('候选序列' in e['message'] for e in detail['errors']))
        stale = self.submission(self.draft())
        (self.models / 'A.safetensors').write_bytes(b'tampered bytes after registration')
        detail = self.refused(stale, 422)
        self.assertEqual(detail['code'], 'plan')
        self.assertEqual(self.draws, [])
        self.executor.assert_not_called()
        self.assertEqual(self.server.list_mix_sessions('SD'), {'sessions': []})

    def test_role_overlap_is_refused_by_the_server_too(self):
        draft = self.draft(loras=[self.global_lora('A', '0.5')])
        detail = self.refused(self.submission(draft), 422)
        self.assertTrue(any(e['scope'] == 'role' and e['kind'] == 'instance' for e in detail['errors']))
        # An equal-byte copy of a global (G2) among the test members is an explicit role conflict as well.
        members = [{'ref': self.refs['G2'], 'rule': rule('0.6', '0.2', '1.0')}]
        ids = [self.refs['G2']['instanceId']]
        draft = self.draft(members=members, candidates=[{'id': 'C01', 'source': 'baseline', 'raw': {ids[0]: '0.6'}}])
        detail = self.refused(self.submission(draft), 422)
        self.assertTrue(any(e['scope'] == 'role' and e['kind'] == 'content' for e in detail['errors']))

    def test_random_seeds_are_drawn_once_and_a_failed_confirm_draws_none(self):
        draft = self.draft()
        draft['baseline']['seed_mode'] = 'random'
        draft['repeat_count'] = 2
        draft['mix']['budget'] = {'preset': '16', 'custom': ''}
        self.assertEqual(self.server.mix_used({'draft': draft})['context_hash'], None)
        bad = copy.deepcopy(draft)
        bad['mix']['candidates'][0]['raw'][self.refs['A']['instanceId']] = '0.65'
        self.refused(self.submission(bad), 422)
        self.assertEqual(self.draws, [])
        result = self.confirm(self.submission(draft))
        self.assertGreater(len(self.draws), 0)
        origin = self.server.xyz_store().records[result['experiment']['id']]['definition']['origin']
        self.assertEqual(len(origin['seeds']), 2)
        self.assertEqual(len(set(origin['seeds'])), 2, 'the same seed set is shared by every profile')
        cells = self.server.xyz_store().records[result['experiment']['id']]['definition']['plan']['cells']
        by_sample = {}
        for cell in cells:
            by_sample.setdefault(cell['coordinate']['sample_index'], set()).add(cell['request']['seed'])
        self.assertTrue(all(len(seeds) == 1 for seeds in by_sample.values()), 'each sample uses one seed across all candidates')

    def test_release_frees_only_unexecuted_vectors_of_a_stopped_round(self):
        first = self.confirm()
        store = self.server.xyz_store()
        with self.assertRaises(self.server.HTTPException):
            self.server.release_mix_round(first['experiment']['id'])      # queued/running rounds cannot be released
        self.complete(first['experiment']['id'], candidate_indexes={0})
        with store.lock:
            record = store.records[first['experiment']['id']]
            for cell in record['state']['cells'].values():
                if cell['status'] != 'complete':
                    cell['status'] = 'cancelled'
            record['state']['status'] = 'cancelled'
            store.save(first['experiment']['id'])
        view = self.server.get_mix_session(first['session']['id'])
        self.assertEqual([c['state'] for c in view['rounds'][0]['candidates']], ['tested', 'cancelled', 'cancelled'])
        # Cancelled vectors are available again; the tested one stays used.
        used = {u['candidate_id']: u['state'] for u in self.server.mix_used({'draft': self.draft()})['used']}
        self.assertEqual(used, {'C01': 'tested', 'C02': 'cancelled', 'C03': 'cancelled'})
        ids = [self.refs['A']['instanceId'], self.refs['B']['instanceId']]
        retry = [{'id': 'C02', 'source': 'sampled', 'raw': dict(zip(ids, ['0.8', '-0.4']))}]
        again = self.confirm(self.submission(self.draft(candidates=retry, session={'id': first['session']['id'], 'revision': 1})))
        self.assertEqual(again['session']['revision'], 2)
        # An interrupted round holds its vectors until released explicitly.
        record = store.records[again['experiment']['id']]
        with store.lock:
            record['state']['status'] = 'interrupted'
            store.save(again['experiment']['id'])
        self.assertEqual(self.server.get_mix_session(first['session']['id'])['rounds'][1]['candidates'][0]['state'], 'reserved')
        released = self.server.release_mix_round(again['experiment']['id'])
        self.assertTrue(released['experiment']['id'])
        self.assertEqual(self.server.get_mix_session(first['session']['id'])['rounds'][1]['candidates'][0]['state'], 'failed')

    def test_remaining_space_pages_json_and_csv_and_excludes_used_vectors(self):
        first = self.confirm()
        page = self.server.mix_remaining(first['session']['id'], None, 'json', 0, 10)
        self.assertEqual((page['total'], page['used'], page['complete']), ('77', 3, False))
        self.assertEqual(len(page['rows']), 10)
        self.assertEqual(page['next_offset'], 10)
        everything = self.server.mix_remaining(first['session']['id'], None, 'json', 0, 1000)
        self.assertEqual((len(everything['rows']), everything['complete']), (74, True))
        names = {m['instanceId'] for m in everything['members']}
        self.assertEqual(names, {self.refs[n]['instanceId'] for n in ('A', 'B', 'C')})
        taken = {tuple(sorted(r['weights'].items())) for r in everything['rows']}
        used = {tuple(sorted({iid: mix.format_units(u) for iid, u in c['weights'].items()}.items())) for c in self.server.xyz_store().records[first['experiment']['id']]['definition']['origin']['candidates']}
        self.assertFalse(taken & used)
        csv = self.server.mix_remaining(first['session']['id'], None, 'csv', 0, 5)
        lines = csv.body.decode('utf-8').strip().split('\n')
        self.assertEqual(lines[0], 'displacement,A,B,C')
        self.assertEqual(len(lines), 6)
        self.assertEqual(csv.headers['X-Total'], '77')
        for bad in ((-1, 10), (0, 0), (0, 5000)):
            with self.assertRaises(self.server.HTTPException):
                self.server.mix_remaining(first['session']['id'], None, 'json', *bad)
        with self.assertRaises(self.server.HTTPException):
            self.server.mix_remaining('0' * 32, None, 'json', 0, 10)

    def test_cell_provenance_separates_globals_from_tests(self):
        first = self.confirm()
        block = self.server.mix_cell_provenance(first['experiment']['id'], '1:0:0:0')
        self.assertEqual((block['schema_version'], block['round'], block['candidate_id']), (1, 1, 'C02'))
        self.assertEqual([g['instanceId'] for g in block['global_loras']], [self.refs['G']['instanceId']])
        self.assertEqual([(t['name'], t['weight'], t['locked']) for t in block['test_loras']], [('A', '0.8', False), ('B', '-0.4', False), ('C', '0.5', True)])
        self.assertEqual(block['context_hash'], self.server.xyz_store().records[first['experiment']['id']]['definition']['origin']['context_hash'])
        self.assertEqual(len(block['search_recipe']['rules_digest']), 16)
        self.assertIsNone(self.server.mix_cell_provenance(first['experiment']['id'], 'x:0:0:0'))
        self.assertIsNone(self.server.mix_cell_provenance('0' * 32, '0:0:0:0'))
        json.dumps(block)

    def test_an_unknown_origin_stays_visible_and_is_never_interpreted(self):
        first = self.confirm()
        store = self.server.xyz_store()
        record = store.records[first['experiment']['id']]
        record['definition']['origin'] = {'kind': 'future_mode', 'schema_version': 9, 'secret': 'x'}
        summary = store.summary(first['experiment']['id'], verify_assets=False)
        self.assertEqual(summary['origin'], {'kind': 'future_mode', 'known': False})
        self.assertEqual(self.server.list_mix_sessions('SD'), {'sessions': []})
        self.assertIsNone(self.server.mix_cell_provenance(first['experiment']['id'], '0:0:0:0'))
        record['definition'].pop('origin')
        self.assertIsNone(store.summary(first['experiment']['id'], verify_assets=False)['origin'], 'a plain XYZ experiment has no origin')

    def test_the_xyz_workspace_list_excludes_search_rounds(self):
        first = self.confirm()
        experiment_id = first['experiment']['id']
        everything = self.server.list_xyz_experiments('SD')
        self.assertIn(experiment_id, [item['id'] for item in everything['experiments']], 'the default list is unchanged')
        plain = self.server.list_xyz_experiments('SD', origin='none')
        self.assertNotIn(experiment_id, [item['id'] for item in plain['experiments']], 'XYZ results never show a search round')
        self.assertEqual(plain['total'], 0)
        # Any origin (known or not) belongs to another workspace; an experiment without one is XYZ's own.
        store = self.server.xyz_store()
        store.records[experiment_id]['definition']['origin'] = {'kind': 'future_mode', 'schema_version': 9}
        self.assertEqual(self.server.list_xyz_experiments('SD', origin='none')['experiments'], [])
        store.records[experiment_id]['definition'].pop('origin')
        self.assertEqual([item['id'] for item in self.server.list_xyz_experiments('SD', origin='none')['experiments']], [experiment_id])

    def test_candidate_grid_lora_weights_option_is_explicit_and_recorded(self):
        from backend import xyz_grid
        options, columns = self.server.mix_export_options({'columns': 3})
        self.assertEqual((options['lora_weights'], columns), (False, 3), 'off unless asked')
        options, _columns = self.server.mix_export_options({'lora_weights': True})
        self.assertIs(options['lora_weights'], True)
        for bad in ('yes', 1, None):
            with self.assertRaises(self.server.HTTPException) as raised:
                self.server.mix_export_options({'lora_weights': bad})
            self.assertEqual(raised.exception.status_code, 422)
        with self.assertRaises(self.server.HTTPException):
            self.server.xyz_export_options({'lora_weights': True})  # the XYZ export schema itself stays strict
        first = self.five_candidate_round()
        with mock.patch.object(self.server, 'xyz_grid_snapshot', side_effect=lambda eid: self.synthetic_snapshot(eid)):
            plain = self.server.mix_grid_snapshot(first['experiment']['id'], 3)
            weighted = self.server.mix_grid_snapshot(first['experiment']['id'], 3, True)
        self.assertNotIn('legend', plain)
        members = self.server.xyz_store().records[first['experiment']['id']]['definition']['origin']['rules']['members']
        self.assertEqual([item.split(' ')[0] for item in weighted['legend']['items']], ['①', '②', '③'][:len(members)])
        self.assertEqual(weighted['cells'][0]['tokens'][:2], ['① 0.6', '② -0.2'], 'X1 order; a locked member still lists its fixed weight')
        self.assertEqual(len(weighted['cells'][0]['tokens']), len(members))
        self.assertIn('全局 LoRA（每格相同）', weighted['legend']['note'])
        layout = xyz_grid.compute_layout(weighted, options)
        self.assertTrue(xyz_grid.recipe(options, layout)['lora_weights'])
        self.assertTrue(xyz_grid.build_manifest(weighted, options, layout, [], 'export-1', {})['search']['lora_weights'])

    def test_all_locked_rules_confirm_exactly_one_profile(self):
        members = [{'ref': self.refs['A'], 'rule': {'locked': True, 'base': '0.4'}}, {'ref': self.refs['B'], 'rule': {'locked': True, 'base': '-0.2'}}]
        result = self.confirm(self.submission(self.draft(members=members, candidates=[], max_displacement='9')))
        self.assertEqual(result['experiment']['total_cells'], 1)
        origin = self.server.xyz_store().records[result['experiment']['id']]['definition']['origin']
        self.assertEqual([c['weights'] for c in origin['candidates']], [{self.refs['A']['instanceId']: 4000, self.refs['B']['instanceId']: -2000}])

    def synthetic_snapshot(self, experiment_id):
        """What xyz_grid_snapshot returns for a finished round, without decoding any pixels."""
        store = self.server.xyz_store()
        definition = store.records[experiment_id]['definition']
        cells = []
        for cell_id in definition['order']:
            planned = store.records[experiment_id]['cells'][cell_id]
            c = planned['coordinate']
            cells.append({'id': cell_id, 'coordinate': [c['x_index'], c['y_index'], c['z_index'], c['sample_index']], 'labels': planned['labels'], 'status': 'complete',
                          'seed': str(planned['request']['seed']), 'request': planned['request'], 'lora_references': planned['lora_references'], 'batch': None, 'relative': f'{cell_id}.png',
                          'asset_id': f'asset-{cell_id}', 'width': 512, 'height': 512, 'alpha': False, 'path': None})
        cells[-1]['status'] = 'error'
        return {'experiment_id': experiment_id, 'digest': definition['digest'], 'engine': definition['engine'], 'axes': copy.deepcopy(definition['axes']),
                'dimensions': copy.deepcopy(definition['dimensions']), 'repeat_count': definition['draft'].get('repeat_count', 1), 'baseline': definition['baseline'],
                'created_at': definition['created_at'], 'cells': cells, 'title': 'x', 'subtitle': 'y'}

    def five_candidate_round(self):
        ids = [self.refs['A']['instanceId'], self.refs['B']['instanceId']]
        values = [('0.6', '-0.2', 'baseline'), ('0.8', '-0.4', 'sampled'), ('0.4', '0', 'sampled'), ('1.0', '-0.2', 'sampled'), ('0.6', '0.2', 'manual')]
        rows = [{'id': f'C{i + 1:02d}', 'source': source, 'raw': dict(zip(ids, [a, b]))} for i, (a, b, source) in enumerate(values)]
        return self.confirm(self.submission(self.draft(candidates=rows)))

    def test_candidate_grid_maps_candidates_to_rows_and_columns_not_axes(self):
        from backend import xyz_grid
        first = self.five_candidate_round()
        experiment_id = first['experiment']['id']
        with mock.patch.object(self.server, 'xyz_grid_snapshot', side_effect=lambda eid: self.synthetic_snapshot(eid)):
            snapshot = self.server.mix_grid_snapshot(experiment_id, 3)
            cells = {c['candidate']['id']: c for c in snapshot['cells']}
            self.assertEqual({k: v['coordinate'] for k, v in cells.items()}, {'C01': [0, 0, 0, 0], 'C02': [1, 0, 0, 0], 'C03': [2, 0, 0, 0], 'C04': [0, 1, 0, 0], 'C05': [1, 1, 0, 0]})
            self.assertEqual(cells['C04']['origin_coordinate'], [3, 0, 0, 0], 'the experiment coordinate is kept beside the presentation one')
            self.assertEqual(snapshot['dimensions'], {'x': 3, 'y': 2, 'z': 1})
            self.assertFalse(snapshot['axes']['x'].get('field') or snapshot['axes']['y'].get('field'), 'no fictional X/Y headings')
            self.assertIn('第 1 轮', snapshot['title'])
            self.assertIn('5 个候选 × 1 样本 · 3 列 × 2 行', snapshot['subtitle'])
            self.assertEqual(snapshot['search']['columns'], 3)
            self.assertEqual([c['position'] for c in snapshot['search']['candidates']], [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1]])
            self.assertTrue(cells['C01']['caption'].startswith('C01 · 基线 · Δ0'))
            self.assertIn('Δ0.4', cells['C02']['caption'])
            options = self.server.xyz_export_options({'arrangement': 'vertical'})
            layout = xyz_grid.compute_layout(snapshot, options)
            self.assertEqual((len(layout['columns']), len(layout['rows'])), (3, 2))
            placed_cells = [entry for entry in layout['ops']['cells']]
            self.assertEqual(len(placed_cells), 6, 'a 3x2 slot grid; the sixth slot is simply empty')
            captions = {entry['cell']: entry['caption']['text'] for entry in placed_cells if entry.get('caption') and entry['cell']}
            self.assertTrue(captions['0:0:0:0'].startswith('C01'))
            manifest = xyz_grid.build_manifest(snapshot, options, layout, [], 'export-1', {})
            self.assertEqual(manifest['search']['round'], 1)
            self.assertEqual([c['candidate']['id'] for c in manifest['cells']], ['C01', 'C02', 'C03', 'C04', 'C05'])
            info, overflow = xyz_grid.png_metadata(snapshot, layout, options, 'export-1', 'a' * 64, [])
            self.assertFalse(overflow)
            self.assertIn('context_hash', str(info.chunks if hasattr(info, 'chunks') else info))
            # 1..16 columns only; more columns than candidates clamps to the candidate count (and says so).
            self.assertEqual(self.server.mix_grid_snapshot(experiment_id, 12)['dimensions'], {'x': 5, 'y': 1, 'z': 1})
            self.assertEqual(self.server.mix_grid_snapshot(experiment_id, None)['dimensions'], {'x': 3, 'y': 2, 'z': 1}, 'default is near-square')
            for bad in (0, 17, 2.5, '4'):
                with self.assertRaises(self.server.HTTPException):
                    self.server.mix_export_options({'columns': bad})
            self.assertEqual(self.server.mix_export_options({'columns': 4, 'scope': 'all'})[1], 4)
            # A multi-sample round pages by sample, exactly as XYZ does.
            plain = self.server.create_xyz_experiment({'schema_version': 1, 'kind': 'xyz_submit', 'submission_id': str(uuid.uuid4()),
                                                       'draft': {'schema_version': 1, 'kind': 'xyz_draft', 'engine': 'SD', 'baseline': self.draft()['baseline'] | {'loras': []}, 'axes': {'x': {'field': 'steps', 'values': [{'value': 4, 'label': '4'}]}}, 'repeat_count': 1, 'requested_concurrency': 1},
                                                       'on_error': 'pause'}, self.http())
            with self.assertRaises(self.server.HTTPException) as refused:
                self.server.mix_grid_snapshot(plain['experiment']['id'], 3)
            self.assertEqual(refused.exception.status_code, 422)


if __name__ == '__main__':
    unittest.main()
