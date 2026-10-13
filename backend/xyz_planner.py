"""Bounded, pure XYZ compiler. No filesystem registry scans, random preview draws,
RPCs, queue writes or GPU imports. Catalog/registry evidence is supplied by an
owner; B must still perform Node/Python byte/path/admission under the shared lease.
"""
from __future__ import annotations

import copy
import json
import re
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Callable

from .xyz_schema import (AxisValue, Coordinate, LoraReference, PlannedCell,
                         PlanIssue, XYZExperimentInput, XYZPlan)

# One checked-in capability contract shared with the editor, owned here. Tests
# pin its vocabulary to the runtime declarations without importing torch.
CAPABILITIES = json.loads((Path(__file__).parents[1] / 'src/xyz-capabilities.json').read_text(encoding='utf-8'))
LIMITS = CAPABILITIES['limits']
UINT64_MAX = 2**64 - 1
MODEL_FIELDS = {'checkpoint', 'diffusion_model', 'text_encoder', 'text_encoder_2', 'vae'}
NUMERIC_FIELDS = {'steps', 'cfg', 'clip_skip', 'pag.scale', 'lora_weight', 'lora_multiplier'}
FIELDS = NUMERIC_FIELDS | MODEL_FIELDS | {'components', 'prompt', 'negative_prompt', 'prompt_replace',
                                         'negative_replace', 'sampler', 'scheduler', 'seed', 'guidance',
                                         'lora_model', 'lora_profile'}


class PlanningError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def reject(code, message):
    raise PlanningError(code, message)


def decimal(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        reject('number', 'Expected a finite decimal number')
    text = str(value)
    if len(text) > 64:
        reject('number', 'Decimal representation is too large')
    try:
        number = Decimal(text)
    except InvalidOperation:
        reject('number', 'Expected a finite decimal number')
    if not number.is_finite() or abs(number) > Decimal('1e20') or number.as_tuple().exponent < -16:
        reject('number', 'Decimal must be finite with at most 16 fractional places')
    return number


def bounded(value, minimum, maximum, *, integer=False, places=None):
    number = decimal(value)
    if not minimum <= number <= maximum or integer and number != number.to_integral_value():
        reject('bounds', f'Expected {"integer " if integer else ""}value in {minimum}..{maximum}')
    if places is not None and number != number.quantize(Decimal(10) ** -places):
        reject('precision', f'Weight supports at most {places} fractional places; no rounding applied')
    return number


def uint64(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9]{1,20}', value) or int(value) > UINT64_MAX:
        reject('seed', 'Seed must be a decimal uint64 string (0..18446744073709551615)')
    return str(int(value))


def expand_axis(axis):
    if axis is None:
        return [AxisValue(value=None)]
    if axis.field not in FIELDS:
        reject('unsupported-axis', 'Unsupported axis; t2i denoise and resample.stageID axes are not ready')
    if axis.range:
        if axis.values or axis.field not in NUMERIC_FIELDS:
            reject('range', 'A numeric range and an ordered value list are mutually exclusive')
        start, end, step = map(decimal, (axis.range.start, axis.range.end, axis.range.step))
        if not step or (end - start) * step < 0:
            reject('range', 'Range step must point toward the end and must not be zero')
        with localcontext() as ctx:
            ctx.prec = 64
            count = int((end - start) // step) + 1
            if count > LIMITS['axis_values']:
                reject('axis-limit', 'Range exceeds 256 values; increase step or reduce bounds')
            return [AxisValue(value=format(start + step * i, 'f')) for i in range(count)]
    if not axis.values or len(axis.values) > LIMITS['axis_values']:
        reject('axis-limit', 'An active axis requires 1..256 ordered values')
    return axis.values


def capability_issue(field, cap):
    if field == 'clip_skip' and cap['clip_skip'] is None:
        reject('unsupported-axis', 'Native runtimes do not support CLIP skip')
    if field in {'negative_prompt', 'negative_replace'} and not cap['negative_prompt']:
        reject('unsupported-axis', 'This distilled engine has no negative branch')
    if field in MODEL_FIELDS and field not in cap['model_fields']:
        reject('unsupported-axis', 'This engine does not mount that model component')
    if field == 'pag.scale' and 'pag' not in cap['guidance']:
        reject('unsupported-axis', 'PAG strength has no effect on this engine')


def writes(axis, cap):
    if axis is None:
        return set()
    field = axis.field
    capability_issue(field, cap)
    if field == 'components':
        return set(cap['model_fields'])
    if field in {'lora_weight', 'lora_model'}:
        if not axis.target:
            reject('target', 'LoRA axes require an exact instanceId target')
        return {'lora:' + axis.target}
    if field in {'lora_profile', 'lora_multiplier'}:
        return {'loras:*'}
    return { {'prompt_replace': 'prompt', 'negative_replace': 'negative_prompt'}.get(field, field) }


def checked_loras(entries, engine, registry):
    result, ids, paths, hashes = [], set(), set(), set()
    instances = registry.get('instances', {}) if registry and registry.get('schemaVersion') == 1 else {}
    for raw in entries:
        item = LoraReference.model_validate(raw).model_dump(exclude_none=True)
        # Even disabled duplicates/invalid weights are not silently repaired.
        weight = bounded(item['weight'], -5, 5, places=4)
        if item['precision'] not in {1, 2, 4}:
            reject('precision', 'LoRA precision must be 1, 2 or 4')
        identity = item.get('instanceId')
        if item['value'] in paths or identity and identity in ids:
            reject('duplicate-lora', 'Duplicate LoRA path or instance; select one explicitly')
        paths.add(item['value']); ids.add(identity)
        if not item['enabled'] or weight == 0:
            continue
        current = instances.get(identity)
        if (item.get('identitySchema') != 1 or not identity or not current or
                current.get('engine') != engine or current.get('status') != 'present' or
                current.get('value') != item['value'] or current.get('contentHash') != item.get('contentHash') or
                current.get('contentRevision') != item.get('contentRevision') or item.get('missing')):
            reject('lora-identity', 'LoRA identity/content/revision is missing or changed; explicitly resolve via P10')
        if not re.fullmatch(r'[0-9a-f]{64}', item.get('contentHash', '')):
            reject('lora-identity', 'Full content hash required')
        if item['contentHash'] in hashes:
            reject('duplicate-content', 'Equal-byte LoRA copies require an explicit single selection, not repeated fusion')
        hashes.add(item['contentHash'])
        item['weight'] = float(weight)
        result.append(item)
    if len(result) > 16:
        reject('lora-limit', 'At most 16 active LoRAs per cell; no truncation applied')
    return result


def upsert(entries, member, target=None):
    result = copy.deepcopy(entries)
    matches = [i for i, entry in enumerate(result) if entry.get('instanceId') == (target or member.get('instanceId'))]
    if len(matches) > 1:
        reject('duplicate-lora', 'Background contains a duplicate target')
    if target and not matches:
        reject('target', 'Target is not in the frozen baseline; no implicit mount')
    if matches:
        result[matches[0]] = copy.deepcopy(member)
    else:
        result.append(copy.deepcopy(member))
    return result


def profile(entries, value):
    if not isinstance(value, dict) or set(value) - {'mode', 'members', 'test_targets', 'preset_prompt', 'includePresetText', 'name', 'id'}:
        reject('profile', 'Invalid LoRA profile snapshot')
    mode = value.get('mode')
    members, targets = value.get('members', []), value.get('test_targets', [])
    if not isinstance(members, list) or not isinstance(targets, list) or any(not isinstance(t, str) or not t for t in targets):
        reject('profile', 'Profile requires ordered members and exact test instance IDs')
    if len(targets) != len(set(targets)):
        reject('profile', 'Duplicate test targets')
    if mode == 'no_loras':
        if members or targets:
            reject('profile', 'No LoRAs is an explicit empty full list')
        return []
    if mode == 'full':
        return copy.deepcopy(members)
    if mode not in {'keep_background', 'no_test'} or mode == 'no_test' and members:
        reject('profile', 'Select keep_background, full, no_test or no_loras explicitly')
    # Keep slots for members also used by the test (update, not append); remove
    # declared test members absent from this candidate. Other background remains.
    member_ids = [m.get('instanceId') for m in members]
    if len(member_ids) != len(set(member_ids)):
        reject('duplicate-lora', 'Duplicate profile members')
    result = [copy.deepcopy(e) for e in entries if e.get('instanceId') not in targets or e.get('instanceId') in member_ids]
    for member in members:
        result = upsert(result, member)
    return result


def apply_axis(request, entries, axis, value, baseline, matches):
    field = axis.field
    if field in {'lora_profile', 'lora_multiplier'}:
        snapshot = value if field == 'lora_profile' else axis.snapshot
        if snapshot is None:
            reject('profile', 'Multiplier requires an explicit frozen full/profile snapshot')
        entries = profile(entries, snapshot)
        if field == 'lora_multiplier':
            multiplier = bounded(value, -5, 5)
            for entry in entries:
                entry['weight'] = str(bounded(decimal(entry['weight']) * multiplier, -5, 5, places=4))
        if snapshot.get('includePresetText') is True:
            text = snapshot.get('preset_prompt', '')
            if not isinstance(text, str):
                reject('profile', 'Preset text must be a string')
            request['prompt'] = '\n\n'.join(part for part in [text, baseline.prompt_without_groups] if part)
        return entries
    if field in {'lora_weight', 'lora_model'}:
        target = next((e for e in entries if e.get('instanceId') == axis.target), None)
        if target is None:
            reject('target', 'Target is not in the frozen baseline')
        member = copy.deepcopy(target) if field == 'lora_weight' else LoraReference.model_validate(value).model_dump(exclude_none=True)
        if field == 'lora_weight':
            member['weight'] = str(bounded(value, -5, 5, places=4)); member['enabled'] = True
        return upsert(entries, member, axis.target)
    if field == 'components':
        if not isinstance(value, dict) or value.get('engine') != request['engine'] or set(value) != {'engine', *CAPABILITIES['engines'][request['engine']]['model_fields']}:
            reject('components', 'Component profile must explicitly name this engine and every required component')
        request.update({k: v for k, v in value.items() if k != 'engine'})
    elif field in {'prompt_replace', 'negative_replace'}:
        key = 'prompt' if field == 'prompt_replace' else 'negative_prompt'
        if not isinstance(value, dict) or set(value) != {'find', 'replace', 'expected_matches'} or not isinstance(value['find'], str) or not value['find'] or not isinstance(value['replace'], str):
            reject('replace', 'Exact replacement requires nonempty find, replacement text and expected_matches')
        count = request[key].count(value['find'])
        if type(value['expected_matches']) is not int or count != value['expected_matches'] or not count:
            reject('replace-count', 'Exact replacement match count differs from the frozen prompt')
        if len(request[key]) + count * (len(value['replace']) - len(value['find'])) > LIMITS['body_bytes']:
            reject('replace-limit', 'Replacement output exceeds the per-cell prompt budget; reduce replacement text/matches')
        matches[key] = count
        request[key] = request[key].replace(value['find'], value['replace'])
    elif field == 'pag.scale':
        request['pag']['scale'] = float(bounded(value, 0, 5))
    else:
        request[field] = value
    return entries


def validate_parameters(request, cap, catalog):
    if not isinstance(request['prompt'], str) or not request['prompt'].strip():
        reject('prompt', 'Positive prompt must not be empty')
    if not isinstance(request['negative_prompt'], str) or not cap['negative_prompt'] and request['negative_prompt']:
        reject('negative', 'Negative prompt is unsupported or invalid')
    if any(request.get(k) is not None for k in MODEL_FIELDS - set(cap['model_fields'])):
        reject('components', 'Unexpected component for this engine')
    for field in cap['model_fields']:
        value = request.get(field)
        if not isinstance(value, str) or not value or (request['engine'], field, value) not in catalog:
            reject('model-resource', f'{field} is missing from this engine/kind catalog (static check only)')
    if request['engine'] == 'Flux' and request['text_encoder'] == request['text_encoder_2']:
        reject('components', 'Flux requires different CLIP-L and T5-XXL encoders')
    for edge in ['width', 'height']:
        bounded(request[edge], 64, cap['max_edge'], integer=True)
        if request[edge] % 64:
            reject('canvas', 'Canvas edges must be divisible by 64')
    request['steps'] = int(bounded(request['steps'], 1, 100, integer=True))
    request['cfg'] = float(bounded(request['cfg'], 0, 30))
    bounded(request['denoise'], 0, 1)
    if request.get('clip_skip') is not None:
        if cap['clip_skip'] is None:
            reject('clip-skip', 'Native runtimes do not support CLIP skip')
        request['clip_skip'] = int(bounded(request['clip_skip'], *cap['clip_skip'], integer=True))
    if request['guidance'] not in cap['guidance']:
        reject('guidance', 'Guidance type is not implemented by this engine')
    if request['sampler'] not in CAPABILITIES['samplers'] or request['scheduler'] not in CAPABILITIES['schedulers']:
        reject('sampling', 'Unsupported sampler or scheduler')
    if not isinstance(request['pag'], dict) or set(request['pag']) != {'scale', 'applied_layers'} or request['pag']['applied_layers'] not in {'mid', 'all'}:
        reject('pag', 'Invalid PAG parameters')
    request['pag']['scale'] = float(bounded(request['pag']['scale'], 0, 5))


def compile_xyz(payload, *, model_catalog=(), lora_registry=None, resolve_seeds=False,
                random_uint64: Callable[[], int] | None = None, validate_request=None):
    """Internal compile, not execution admission. Preview consumes NO randomness.

    resolve_seeds=True is for B's submit-only owner, NEVER preview. It requires
    the existing strict request validator, to validate inherited stages/output
    rules; no public endpoint currently invokes this function.
    """
    raw = payload.model_dump(mode='json') if isinstance(payload, XYZExperimentInput) else payload
    if len(json.dumps(raw, ensure_ascii=False, allow_nan=False).encode()) > LIMITS['body_bytes']:
        reject('body-limit', 'XYZ body exceeds 1 MiB; reduce values or prompt/snapshot size')
    draft = XYZExperimentInput.model_validate(raw)
    if draft.baseline.parameters.engine != draft.engine:
        reject('engine', 'Baseline engine differs from the experiment')
    for value, key in [(draft.repeat_count, 'repeat_count'), (draft.requested_concurrency, 'requested_concurrency')]:
        if not 1 <= value <= LIMITS[key]:
            reject('bounds', f'{key} must be 1..{LIMITS[key]}')
    if resolve_seeds and (random_uint64 is None or validate_request is None):
        reject('submit-owner', 'Resolved compile requires submit RNG and strict GenerateInput validator')
    cap = CAPABILITIES['engines'][draft.engine]
    axes = {key: draft.axes.get(key) for key in ['x', 'y', 'z']}
    values, errors = {}, []
    for key, axis in axes.items():
        try:
            values[key] = expand_axis(axis)
            if axis:
                writes(axis, cap)
        except PlanningError as error:
            errors.append(PlanIssue(code=error.code, message=str(error), axis=key))
    if errors:
        return XYZPlan(engine=draft.engine, total_cells=0, dimensions={}, cells=[], errors=errors, seed_status='unresolved', requested_concurrency=draft.requested_concurrency)
    dimensions = {key: len(value) for key, value in values.items()}
    total = dimensions['x'] * dimensions['y'] * dimensions['z'] * draft.repeat_count
    if total > LIMITS['total_cells']:
        reject('cell-limit', 'Experiment exceeds 1024 cells; reduce axis values or repeats')
    # Conservative upper bound BEFORE expansion: payload includes every possible
    # prompt/profile and baseline. No potentially gigabyte cell allocation.
    if (len(json.dumps(raw, ensure_ascii=False).encode()) + 2048) * total > LIMITS['plan_bytes']:
        reject('plan-limit', 'Conservative expanded-plan budget exceeds 16 MiB; reduce cells or snapshot size')
    conflict = None
    occupied = set()
    for axis in axes.values():
        fields = writes(axis, cap)
        if occupied & fields or 'loras:*' in occupied and any(f.startswith('lora:') for f in fields) or 'loras:*' in fields and any(f.startswith('lora:') for f in occupied):
            conflict = 'Multiple axes write the same field/instance or a profile conflicts with weight/model overrides'
        occupied |= fields
    if 'loras:*' in occupied and any(a and a.field in {'prompt', 'prompt_replace'} for a in axes.values()):
        # A profile may opt into text per value. Refuse indeterminate ownership.
        if any(a and (a.field == 'lora_multiplier' and a.snapshot and a.snapshot.get('includePresetText') or a.field == 'lora_profile' and any(isinstance(v.value, dict) and v.value.get('includePresetText') for v in a.values)) for a in axes.values()):
            conflict = 'Preset text and a positive prompt axis both write the actual prompt'
    seed_axis = any(a and a.field == 'seed' for a in axes.values())
    if seed_axis and draft.sample_seeds is not None:
        reject('seed-policy', 'Seed axis wins for all repeats; remove the unused sample seed list')
    if draft.sample_seeds is not None and len(draft.sample_seeds) != draft.repeat_count:
        reject('seed-policy', 'Explicit sample seed list must match repeat_count')
    base_seed = uint64(draft.baseline.parameters.seed)
    samples = [uint64(s) for s in draft.sample_seeds] if draft.sample_seeds is not None else [
        base_seed if i == 0 and draft.baseline.seed_mode == 'fixed' else f'unresolved:sample:{i}' for i in range(draft.repeat_count)]
    cells = []
    # Authoritative input catalogs, not health's engine-ready flags or filenames.
    catalog = set(model_catalog)
    for z in range(dimensions['z']):
        for sample in range(draft.repeat_count):
            for y in range(dimensions['y']):
                for x in range(dimensions['x']):
                    coord = Coordinate(x_index=x, y_index=y, z_index=z, sample_index=sample)
                    request = draft.baseline.parameters.model_dump(exclude_none=True)
                    entries = [e.model_dump(exclude_none=True) for e in draft.baseline.loras]
                    matches, labels = {}, {}
                    try:
                        if conflict:
                            reject('axis-conflict', conflict)
                        for key, index in [('x', x), ('y', y), ('z', z)]:
                            axis, value = axes[key], values[key][index]
                            labels[key] = value.label or (str(value.value) if axis else '—')
                            if axis:
                                if value.value is None or isinstance(value.value, str) and not value.value.strip():
                                    reject('empty-value', 'Axis contains an empty value')
                                entries = apply_axis(request, entries, axis, value.value, draft.baseline, matches)
                        if any(a and a.field == 'pag.scale' for a in axes.values()) and request['guidance'] != 'pag':
                            reject('ineffective-axis', 'PAG strength axis requires PAG guidance in this cell')
                        if draft.engine in {'SD', 'iL'} and request['sampler'] in {'lcm', 'ipndm', 'ipndm_v', 'ddim', 'ddpm'} and any(a and a.field == 'scheduler' for a in axes.values()):
                            reject('ineffective-axis', 'This SD/iL sampler adapter does not implement a general sigma scheduler sweep')
                        validate_parameters(request, cap, catalog)
                        entries = checked_loras(entries, draft.engine, lora_registry)
                        # Inline resolution is P10's existing owner. Stage A fails closed
                        # instead of reimplementing basename guessing/duplicate fusion.
                        if re.search(r'<\s*lora\s*:', request['prompt'] + request['negative_prompt'], re.I):
                            reject('inline-admission-pending', 'Inline LoRA requires B P10 preflight; use explicit instance-bound mounts for this plan')
                        request['seed'] = uint64(request['seed']) if seed_axis else samples[sample]
                        request['postprocess_chain'] = request['postprocess_chain'] if draft.inherit_postprocess else []
                        # The chain position is the unit's identity, exactly as
                        # resolve_chain_seeds keys it; the strict ResampleUnitInput
                        # wire has no id field. A unit is enabled only when it says
                        # so (the wire default is disabled).
                        for position, stage in enumerate(request['postprocess_chain']):
                            if stage.get('type') == 'resample' and stage.get('enabled') is True:
                                if stage.get('seed_mode', 'inherit') == 'random':
                                    stage['seed'] = f'unresolved:post:{x}:{y}:{z}:{sample}:{position}'
                                elif stage.get('seed_mode') == 'fixed':
                                    stage['seed'] = uint64(str(stage.get('seed', '')))
                                else:
                                    stage['seed'] = request['seed']
                        request['loras'] = [{'path': e['value'], 'weight': e['weight']} for e in entries]
                        request.update(images_per_batch=1, batch_count=1)
                        cells.append(PlannedCell(cell_id=f'{x}:{y}:{z}:{sample}', coordinate=coord, labels=labels,
                            request=request, lora_references=entries, seed_status='unresolved' if any('unresolved:' in str(v) for v in [request['seed'], request['postprocess_chain']]) else 'resolved', replacement_matches=matches))
                    except (PlanningError, ValueError, TypeError, KeyError) as error:
                        errors.append(PlanIssue(code=getattr(error, 'code', 'invalid-value'), message=str(error) if isinstance(error, PlanningError) else 'Invalid typed axis/profile value', coordinate=coord))
    plan = XYZPlan(engine=draft.engine, total_cells=total, dimensions=dimensions, cells=cells, errors=errors,
                   seed_status='unresolved' if errors or any(c.seed_status == 'unresolved' for c in cells) else 'resolved', requested_concurrency=draft.requested_concurrency)
    if len(plan.model_dump_json().encode()) > LIMITS['plan_bytes']:
        reject('plan-limit', 'Expanded plan exceeds 16 MiB; reduce cells or prompt/snapshot size')
    # Preflight ALL cells first. Failed submit draws nothing. B will run this
    # inside complete admission, then save the returned definition exactly once.
    if resolve_seeds and not errors:
        symbols = {}
        def resolve(value):
            if isinstance(value, str) and value.startswith('unresolved:'):
                if value not in symbols:
                    drawn = random_uint64()
                    if type(drawn) is not int or not 0 <= drawn <= UINT64_MAX:
                        reject('rng', 'Submit RNG must return a uint64 Python int')
                    symbols[value] = str(drawn)
                return symbols[value]
            return value
        # Validate before drawing too; random symbols use legal placeholders.
        for cell in cells:
            trial = copy.deepcopy(cell.request)
            trial['seed'] = 0 if str(trial['seed']).startswith('unresolved:') else int(trial['seed'])
            for stage in trial['postprocess_chain']:
                if stage.get('type') == 'resample' and stage.get('enabled') is True:
                    stage['seed_mode'] = 'fixed'
                    stage['seed'] = '0' if str(stage['seed']).startswith('unresolved:') else str(int(stage['seed']))
            validate_request(trial)
        for cell in cells:
            cell.request['seed'] = resolve(cell.request['seed'])
            for stage in cell.request['postprocess_chain']:
                if stage.get('type') == 'resample' and stage.get('enabled') is True:
                    stage['seed'] = resolve(stage['seed']); stage['seed_mode'] = 'fixed'
            # The base seed executes as a Python int (GenerateInput.seed). A
            # resample unit's seed stays the decimal string ResampleUnitInput
            # requires; JSON consumers always read recorded decimal strings.
            cell.request['seed'] = int(cell.request['seed'])
            for stage in cell.request['postprocess_chain']:
                if stage.get('type') == 'resample' and 'seed' in stage:
                    stage['seed'] = str(int(stage['seed']))
            validate_request(cell.request)
            cell.seed_status = 'resolved'
    plan.seed_status = 'unresolved' if errors or any(c.seed_status == 'unresolved' for c in cells) else 'resolved'
    if len(plan.model_dump_json().encode()) > LIMITS['plan_bytes']:
        reject('plan-limit', 'Expanded plan exceeds 16 MiB; reduce cells or prompt/snapshot size')
    return plan


def plan_record(plan):
    """Lossless JSON-facing record; separate from strict GenerateInput."""
    record = plan.model_dump(mode='json')
    for cell in record['cells']:
        cell['request']['seed'] = str(cell['request']['seed'])
        for stage in cell['request']['postprocess_chain']:
            if 'seed' in stage:
                stage['seed'] = str(stage['seed'])
    return record
