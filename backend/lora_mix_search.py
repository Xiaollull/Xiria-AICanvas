"""QPlan2/P2 LoRA mix search: the authoritative server contract.

Pure module: no torch, FastAPI, RNG, clock or filesystem. It mirrors `src/lora-mix-rules.js` exactly (integer
weight units of 0.0001), validates a whole search recipe without trusting any client-derived count, status or
identity, compiles accepted candidates into the *existing* XYZ `lora_profile` / `keep_background` request, and
derives the state of every candidate from the XYZ experiment records. A search round is nothing but an XYZ
experiment carrying an `origin` block; the experiment record is the single recovery source.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any, Iterator

from typing import Literal

from pydantic import BaseModel, ConfigDict

from .xyz_planner import LIMITS

SCHEMA_VERSION = 1
ALGORITHM = 'lora-mix-stratified-fps-v1'
ORIGIN_KIND = 'lora_mix_search'
UNIT = 10000
MAX_UNITS = 5 * UNIT
PREFERENCES = ('balanced', 'min', 'max')
ACTIVE_LIMIT = 16
WORK_LIMIT = 4_000_000
BUDGET_PRESETS = ('16', '32', '64', '128')
INSTANCE_ID = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$')
CONTENT_HASH = re.compile(r'^[0-9a-f]{64}$')
DECIMAL = re.compile(r'^(-?)(\d+(?:\.\d+)?|\.\d+)$')
SESSION_ID = re.compile(r'^[0-9a-f]{32}$')
ENUMERATE_BUDGET = 1_000_000
MIX_KEYS = {'version', 'members', 'max_displacement', 'budget', 'include_baseline', 'sampling_seed', 'cursor', 'candidates', 'next_id', 'proposed_key', 'session', 'favorites'}
ACTIVE_STATUSES = frozenset({'queued', 'running', 'pausing', 'paused', 'cancelling'})


class MixError(ValueError):
    """One refusal with a machine code; `issues` carries every problem found, not just the first."""

    def __init__(self, code, message, status=422, issues=None):
        super().__init__(message)
        self.code, self.status, self.issues = code, status, issues or [{'code': code, 'message': message}]


# ---------------------------------------------------------------------------------------------------------
# Exact integer rules (mirror of src/lora-mix-rules.js)
# ---------------------------------------------------------------------------------------------------------

def parse_units(text):
    """Exact decimal text -> units, or (None, message). Never rounds, clamps or accepts partial input."""
    if isinstance(text, bool) or not isinstance(text, str):
        return None, '需要十进制数'
    raw = text.strip()
    if not raw:
        return None, '不能为空'
    if re.search(r'e', raw, re.I) or not DECIMAL.match(raw):
        return None, '需要普通十进制数（例如 0.6 或 -0.25）'
    match = DECIMAL.match(raw)
    sign, body = match.group(1), match.group(2)
    whole, _, fraction = body.partition('.')
    if len(fraction) > 4:
        return None, '最多 4 位小数；不会四舍五入'
    whole_value = int((whole.lstrip('0') or '0'))
    if whole_value > 5:
        return None, '权重需在 −5 到 5 之间'
    units = whole_value * UNIT + int((fraction or '').ljust(4, '0') or '0')
    if units > MAX_UNITS:
        return None, '权重需在 −5 到 5 之间'
    return (-units if sign and units else units), None


def format_units(units):
    sign = '-' if units < 0 else ''
    absolute = abs(units)
    whole, fraction = divmod(absolute, UNIT)
    text = str(fraction).rjust(4, '0').rstrip('0')
    return f'{sign}{whole}{"." + text if text else ""}'


class Sequence:
    """min, min+step, ... <= max, then the base as a deduplicated extra option (never an off-grid max)."""

    def __init__(self, base, low, high, step):
        self.base, self.min, self.max, self.step = base, low, high, step
        self.grid_count = (high - low) // step + 1
        self.base_on_grid = (base - low) % step == 0
        self.base_index = (base - low) // step if self.base_on_grid else -(-(base - low) // step)
        self.size = self.grid_count + (0 if self.base_on_grid else 1)
        self.first = low
        self.last = max(low + (self.grid_count - 1) * step, base)
        self.max_distance = max(abs(self.first - base), abs(self.last - base))

    def at(self, index):
        if not isinstance(index, int) or index < 0 or index >= self.size:
            raise IndexError('sequence index out of range')
        if self.base_on_grid or index < self.base_index:
            return self.min + index * self.step
        return self.base if index == self.base_index else self.min + (index - 1) * self.step

    def index_of(self, units):
        if units == self.base:
            return self.base_index
        if units < self.min or units > self.max or (units - self.min) % self.step:
            return -1
        grid = (units - self.min) // self.step
        return grid if self.base_on_grid or grid < self.base_index else grid + 1


def inspect_rule(rule):
    """Field-keyed errors plus the parsed integers. Locked members only need a valid base."""
    errors, parsed = {}, {}
    if not isinstance(rule, dict):
        return {'base': '规则必须是对象'}, None, False

    def read(field, label):
        units, message = parse_units(rule.get(field))
        if message:
            errors[field] = f'{label}{message}'
        else:
            parsed[field] = units
    locked = rule.get('locked') is True
    read('base', '基础权重')
    if locked:
        return errors, (None if errors else {'base': parsed['base'], 'locked': True}), True
    read('min', '最小值'); read('max', '最大值'); read('step', '步长')
    if rule.get('preference') not in PREFERENCES:
        errors['preference'] = '偏好需为 均衡 / 偏向最小 / 偏向最大'
    if 'step' not in errors and parsed['step'] <= 0:
        errors['step'] = '步长必须大于 0'
    if 'min' not in errors and 'max' not in errors and parsed['min'] > parsed['max']:
        errors['max'] = '最小值不能大于最大值'
    if not any(key in errors for key in ('base', 'min', 'max')) and not (parsed['min'] <= parsed['base'] <= parsed['max']):
        errors['base'] = '基础权重必须位于最小值与最大值之间'
    return errors, (None if errors else {**parsed, 'locked': False}), False


class Row:
    def __init__(self, index, member, errors, parsed, locked):
        self.index, self.member, self.errors, self.parsed, self.locked = index, member, errors, parsed, locked
        self.preference = (member.get('rule') or {}).get('preference', 'balanced')
        self.sequence = Sequence(parsed['base'], parsed['min'], parsed['max'], parsed['step']) if parsed and not locked else None


class Analysis:
    def __init__(self, members):
        self.rows = []
        for index, member in enumerate(members):
            errors, parsed, locked = inspect_rule(member.get('rule'))
            self.rows.append(Row(index, member, errors, parsed, locked))
        self.free = [row for row in self.rows if row.sequence]
        self.valid = bool(self.rows) and all(not row.errors for row in self.rows)
        self.dmax = sum(row.sequence.max_distance for row in self.free) if self.valid else None
        self.space = None
        if self.valid:
            self.space = 1
            for row in self.free:
                self.space *= row.sequence.size


def displacement_of(free, vector):
    return sum(abs(vector[i] - free[i].sequence.base) for i in range(len(free)))


def inspect_vector(free, vector, limit):
    if len(vector) != len(free):
        return False, '向量长度与解锁成员不一致'
    for i, row in enumerate(free):
        if not isinstance(vector[i], int) or isinstance(vector[i], bool):
            return False, '存在无效权重'
        if row.sequence.index_of(vector[i]) < 0:
            return False, '不在该成员的候选序列中（未取整、未截断）'
    distance = displacement_of(free, vector)
    if limit is not None and distance > limit:
        return False, f'总位移 {format_units(distance)} 超过上限 {format_units(limit)}'
    return True, None


def count_space(free, limit, work_limit=WORK_LIMIT):
    """Feasible vector count under displacement <= limit: exact, or None when too large to count."""
    if not free:
        return 1
    space = 1
    for row in free:
        space *= row.sequence.size
    dmax = sum(row.sequence.max_distance for row in free)
    if limit is None or limit >= dmax:
        return space
    tables = []
    for row in free:
        table = {}
        for i in range(row.sequence.size):
            distance = abs(row.sequence.at(i) - row.sequence.base)
            if distance <= limit:
                table[distance] = table.get(distance, 0) + 1
        tables.append(sorted(table.items()))
    if sum(len(table) for table in tables) * (limit + 1) > work_limit:
        return None
    ways = [0] * (limit + 1)
    ways[0] = 1
    for table in tables:
        following = [0] * (limit + 1)
        for used, amount in enumerate(ways):
            if not amount:
                continue
            for distance, count in table:
                if used + distance <= limit:
                    following[used + distance] += amount * count
        ways = following
    return sum(ways)


def enumerate_vectors(free, limit) -> Iterator[tuple]:
    """Every legal vector in sequence order (depth first); the caller bounds how many it consumes."""
    current = [0] * len(free)

    def visit(position, used):
        if position == len(free):
            yield tuple(current)
            return
        sequence = free[position].sequence
        for i in range(sequence.size):
            value = sequence.at(i)
            distance = abs(value - sequence.base)
            if limit is not None and used + distance > limit:
                continue
            current[position] = value
            yield from visit(position + 1, used + distance)
    yield from visit(0, 0)


# ---------------------------------------------------------------------------------------------------------
# Recipe validation
# ---------------------------------------------------------------------------------------------------------

def effective(item):
    """Same activity rule as xyz_planner.checked_loras: disabled or zero weight is never fused."""
    try:
        return item.get('enabled', True) is not False and float(item.get('weight', 1)) != 0
    except (TypeError, ValueError):
        return True


def strong_reference(ref):
    return (isinstance(ref, dict) and ref.get('identitySchema') == 1 and isinstance(ref.get('instanceId'), str) and bool(INSTANCE_ID.match(ref['instanceId']))
            and isinstance(ref.get('contentHash'), str) and bool(CONTENT_HASH.match(ref['contentHash']))
            and isinstance(ref.get('contentRevision'), int) and not isinstance(ref.get('contentRevision'), bool) and ref['contentRevision'] >= 1
            and isinstance(ref.get('value'), str) and 0 < len(ref['value']) <= 500)


def label(item):
    return item.get('name') or item.get('value') or '未命名 LoRA'


class Candidate:
    def __init__(self, raw, free_units, weights, displacement):
        self.id, self.source, self.raw, self.units, self.weights, self.displacement = raw['id'], raw['source'], raw['raw'], free_units, weights, displacement


class Recipe:
    """A fully validated search recipe. Every number below was recomputed here, never read from the client."""

    def __init__(self):
        self.issues = []

    def add(self, scope, message, **extra):
        self.issues.append({'code': scope, 'scope': scope, 'message': message, **extra})


class MixSubmitInput(BaseModel):
    """The confirmation envelope. The browser picks `submission_id` once per click, so a replay is one round, not two."""
    model_config = ConfigDict(extra='forbid', strict=True)
    schema_version: Literal[1] = 1
    kind: Literal['lora_mix_submit'] = 'lora_mix_submit'
    submission_id: str
    draft: dict
    on_error: Literal['pause', 'continue'] = 'pause'


def validate_recipe(draft, limits=None, *, require_candidates=True):
    """Parse and check a wire draft. Raises MixError carrying every issue; returns a Recipe when clean.
    `require_candidates=False` is for read-only context queries that have no preview yet."""
    limits = limits or LIMITS
    recipe = Recipe()
    if not isinstance(draft, dict) or not isinstance(draft.get('mix'), dict) or not isinstance(draft.get('baseline'), dict):
        raise MixError('recipe', '搜索草稿需要 baseline 与 mix 块')
    mix, baseline = draft['mix'], draft['baseline']
    if mix.get('version') != SCHEMA_VERSION:
        raise MixError('version', f'未知的搜索规则版本 {mix.get("version")!r}；服务端不解读它', 422)
    extra = set(mix) - MIX_KEYS
    if extra:
        raise MixError('recipe', f'搜索规则含未知字段：{sorted(extra)}')
    if 'favorites' in mix and not (isinstance(mix['favorites'], list) and all(isinstance(item, str) for item in mix['favorites'])):
        raise MixError('recipe', '收藏格式无效')
    members = mix.get('members')
    if not isinstance(members, list) or not all(isinstance(m, dict) and isinstance(m.get('ref'), dict) and isinstance(m.get('rule'), dict) for m in members):
        raise MixError('recipe', '测试成员格式无效')
    recipe.engine = draft.get('engine')
    recipe.baseline, recipe.mix, recipe.draft = baseline, mix, draft
    recipe.repeat_count, recipe.concurrency = draft.get('repeat_count'), draft.get('requested_concurrency')
    analysis = recipe.analysis = Analysis(members)
    if not members:
        recipe.add('members', 'X1：至少选择 1 个测试 LoRA')
    ids = []
    for index, member in enumerate(members):
        ref = member['ref']
        if not strong_reference(ref):
            recipe.add('members', f'「{label(ref)}」缺少强实例 / 内容修订，不能作为测试成员', member=index)
        ids.append(ref.get('instanceId'))
        for field, message in analysis.rows[index].errors.items():
            recipe.add('rule', message, member=index, field=field)
    if len(set(ids)) != len(ids):
        recipe.add('members', '测试成员中有重复实例')
    if len({m['ref'].get('value') for m in members}) != len(members):
        recipe.add('members', '测试成员中有重复路径')
    globals_ = [item for item in baseline.get('loras', []) if isinstance(item, dict)]
    for index, member in enumerate(members):
        row, ref = analysis.rows[index], member['ref']
        never_active = row.locked and row.parsed is not None and row.parsed['base'] == 0
        for item in globals_:
            kind = None
            if item.get('instanceId') and item.get('instanceId') == ref.get('instanceId'):
                kind = 'instance'
            elif item.get('value') == ref.get('value'):
                kind = 'path'
            elif not never_active and item.get('contentHash') and item.get('contentHash') == ref.get('contentHash') and effective(item):
                kind = 'content'
            if kind:
                recipe.add('role', f'测试 LoRA「{label(ref)}」与全局 LoRA「{label(item)}」冲突（{kind}）；全局背景与测试成员不能重叠', member=index, kind=kind)
    recipe.limit = None
    if analysis.valid and not analysis.free:
        recipe.limit = 0
    elif analysis.valid:
        units, message = parse_units(mix.get('max_displacement'))
        if message:
            recipe.add('displacement', f'最大总位移{message}')
        elif units < 0 or units > analysis.dmax:
            recipe.add('displacement', f'最大总位移超过可达上限 {format_units(analysis.dmax)}；不会自动截断')
        else:
            recipe.limit = units
    # Budget: images = profile slots x samples; nothing is truncated.
    budget = mix.get('budget') if isinstance(mix.get('budget'), dict) else {}
    text = budget.get('custom') if budget.get('preset') == 'custom' else budget.get('preset')
    images = int(text.strip()) if isinstance(text, str) and re.fullmatch(r'\d{1,7}', text.strip()) else None
    seeds = recipe.repeat_count
    recipe.images, recipe.slots = images, 0
    if images is None or images < 1:
        recipe.add('budget', '图片预算需要是正整数')
    elif not isinstance(seeds, int) or isinstance(seeds, bool) or not 1 <= seeds <= limits['repeat_count']:
        recipe.add('budget', f'重复样本必须为 1–{limits["repeat_count"]} 整数')
    else:
        recipe.slots = images // seeds
        if recipe.slots < 1:
            recipe.add('budget', f'预算 {images} 张少于样本数 {seeds}；至少需要每个配置 1 组样本')
        elif recipe.slots > limits['axis_values'] or images > limits['total_cells']:
            recipe.add('budget', f'超出现有执行上限（每轴 {limits["axis_values"]} 项、总格数 {limits["total_cells"]}）；不会截断样本')
    if not isinstance(recipe.concurrency, int) or isinstance(recipe.concurrency, bool) or not 1 <= recipe.concurrency <= limits['requested_concurrency']:
        recipe.add('budget', f'请求并发必须为 1–{limits["requested_concurrency"]} 整数；不会自动调整')
    sampling = mix.get('sampling_seed')
    if not isinstance(sampling, str) or not re.fullmatch(r'\d{1,20}', sampling) or int(sampling) > 2**64 - 1:
        recipe.add('sampling', '采样种子必须是 uint64 十进制文本')
    # Candidates: re-derived from raw text. A client's displacement, status or key is never read.
    raw_candidates = mix.get('candidates')
    recipe.candidates = []
    if not isinstance(raw_candidates, list) or not all(isinstance(c, dict) and isinstance(c.get('id'), str) and c.get('source') in ('baseline', 'sampled', 'manual') and isinstance(c.get('raw'), dict) for c in raw_candidates):
        recipe.add('candidates', '候选格式无效')
        raw_candidates = []
    active_globals = sum(1 for item in globals_ if effective(item))
    seen, seen_ids = {}, set()
    for raw in raw_candidates:
        if raw['id'] in seen_ids:
            recipe.add('candidate', f'{raw["id"]}：候选编号重复', candidate=raw['id'])
        seen_ids.add(raw['id'])
        problems, units = [], []
        if analysis.valid:
            for row in analysis.free:
                iid = row.member['ref'].get('instanceId')
                text = raw['raw'].get(iid)
                value, message = parse_units(text) if text is not None else (None, '缺少权重')
                if message:
                    problems.append(message); units.append(None)
                    continue
                if row.sequence.index_of(value) < 0:
                    problems.append('不在该成员的候选序列中（不取整、不截断）')
                units.append(value)
        if analysis.valid and all(value is not None for value in units):
            ok, message = inspect_vector(analysis.free, units, recipe.limit)
            distance = displacement_of(analysis.free, units)
            if not ok and '超过上限' in (message or ''):
                problems.append(message)
            key = tuple(units)
            if key in seen:
                problems.append(f'与 {seen[key]} 完全重复')
            else:
                seen[key] = raw['id']
            weights = {}
            for row in analysis.rows:
                iid = row.member['ref'].get('instanceId')
                weights[iid] = row.parsed['base'] if row.locked else units[analysis.free.index(row)]
            active_members = sum(1 for value in weights.values() if value != 0)
            if active_globals + active_members > ACTIVE_LIMIT:
                problems.append(f'全局 {active_globals} + 测试 {active_members} 个启用且非零的 LoRA 超过 {ACTIVE_LIMIT} 上限；不会截断')
            if not problems:
                recipe.candidates.append(Candidate(raw, tuple(units), weights, distance))
        if problems:
            recipe.add('candidate', f'{raw["id"]}：{"；".join(problems)}', candidate=raw['id'])
    if require_candidates and analysis.valid and analysis.free and not raw_candidates:
        recipe.add('candidates', '尚无候选；请先生成候选预览')
    if recipe.slots and len(raw_candidates) > recipe.slots:
        recipe.add('budget', f'候选 {len(raw_candidates)} 个超过预算可容纳的 {recipe.slots} 个配置；请删除多余候选，不会自动截断')
    if analysis.valid and not analysis.free and not recipe.issues:
        # All locked: exactly one profile (the fixed bases), no preview rows needed.
        weights = {row.member['ref']['instanceId']: row.parsed['base'] for row in analysis.rows}
        recipe.candidates = [Candidate({'id': 'C01', 'source': 'baseline', 'raw': {}}, (), weights, 0)]
    session = mix.get('session') if isinstance(mix.get('session'), dict) else {}
    recipe.session_id, recipe.expected_revision = session.get('id'), session.get('revision', 0)
    if recipe.session_id is not None and not (isinstance(recipe.session_id, str) and SESSION_ID.match(recipe.session_id)):
        recipe.add('session', '搜索会话编号格式无效')
    if not isinstance(recipe.expected_revision, int) or isinstance(recipe.expected_revision, bool) or recipe.expected_revision < 0:
        recipe.add('session', '搜索会话修订无效')
    if recipe.candidates and recipe.repeat_count and len(recipe.candidates) * recipe.repeat_count > limits['total_cells']:
        recipe.add('budget', f'总格数超过 {limits["total_cells"]}；不会截断样本')
    if recipe.issues:
        raise MixError('recipe', f'{len(recipe.issues)} 个问题需要先修正；未预留向量、未抽取种子', 422, recipe.issues)
    recipe.space = count_space(analysis.free, recipe.limit)
    return recipe


# ---------------------------------------------------------------------------------------------------------
# Compilation to the existing XYZ request (no extra field ever reaches a strict DTO)
# ---------------------------------------------------------------------------------------------------------

def test_member(member, units):
    ref = member['ref']
    entry = {key: copy.deepcopy(ref[key]) for key in ('value', 'name', 'category', 'identitySchema', 'instanceId', 'contentHash', 'contentRevision', 'originalReference') if key in ref}
    entry.update(weight=format_units(units), precision=4, enabled=True)
    return entry


def profile_for(recipe, candidate):
    members = [test_member(row.member, candidate.weights[row.member['ref']['instanceId']]) for row in recipe.analysis.rows]
    return {'mode': 'keep_background', 'members': members, 'test_targets': [m['instanceId'] for m in members],
            'includePresetText': False, 'name': candidate.id, 'id': candidate.id}


def compile_xyz_draft(recipe, candidates=None):
    """The one XYZ axis a round needs. Globals stay the frozen background; they are never cleared or scaled."""
    chosen = candidates if candidates is not None else recipe.candidates
    draft = recipe.draft
    return {
        'schema_version': 1, 'kind': 'xyz_draft', 'engine': recipe.engine, 'baseline': copy.deepcopy(recipe.baseline),
        'axes': {'x': {'field': 'lora_profile', 'values': [{'value': profile_for(recipe, c), 'label': c.id} for c in chosen]}, 'y': None, 'z': None},
        'repeat_count': recipe.repeat_count, 'sample_seeds': copy.deepcopy(draft.get('sample_seeds')),
        'requested_concurrency': recipe.concurrency, 'inherit_postprocess': bool(draft.get('inherit_postprocess', False)),
        'group_by_resources': bool(draft.get('group_by_resources', False)),
    }


# ---------------------------------------------------------------------------------------------------------
# Context and candidate keys
# ---------------------------------------------------------------------------------------------------------

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def sha256(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def context_of(recipe, seeds):
    """Everything that changes the pixels except the tested weights (instantaneous free VRAM is not part of it)."""
    p = recipe.baseline.get('parameters', {})
    fields = ('checkpoint', 'diffusion_model', 'text_encoder', 'text_encoder_2', 'vae')
    return {
        'v': 1, 'engine': recipe.engine, 'models': {key: p.get(key) for key in fields},
        'prompt': p.get('prompt'), 'negative_prompt': p.get('negative_prompt'), 'group_prompt': recipe.baseline.get('group_prompt', ''),
        'sampling': {key: p.get(key) for key in ('steps', 'cfg', 'denoise', 'sampler', 'scheduler', 'guidance', 'pag', 'clip_skip')},
        'canvas': [p.get('width'), p.get('height')],
        'postprocess': p.get('postprocess_chain') if recipe.draft.get('inherit_postprocess') else [],
        'output_template': p.get('output_template'),
        'seeds': list(seeds),
        'globals': [{k: item.get(k) for k in ('instanceId', 'contentHash', 'contentRevision', 'value', 'weight', 'precision', 'enabled')} for item in recipe.baseline.get('loras', [])],
        'tests': [{k: row.member['ref'].get(k) for k in ('instanceId', 'contentHash', 'contentRevision')} for row in recipe.analysis.rows],
    }


def context_hash(recipe, seeds):
    return sha256(canonical(context_of(recipe, seeds)))


def candidate_key(context, recipe, weights):
    return sha256(context + '|' + ','.join(f'{row.member["ref"]["instanceId"]}:{weights[row.member["ref"]["instanceId"]]}' for row in recipe.analysis.rows))


def explicit_seeds(recipe):
    """The image seed list when it is already decided (never a random draw); None when a random seed will be drawn."""
    seeds = recipe.draft.get('sample_seeds')
    if isinstance(seeds, list) and len(seeds) == recipe.repeat_count and all(isinstance(s, str) and s.isdigit() for s in seeds):
        return [str(int(s)) for s in seeds]
    parameters = recipe.baseline.get('parameters', {})
    seed = parameters.get('seed')
    if recipe.baseline.get('seed_mode', 'fixed') == 'fixed' and recipe.repeat_count == 1 and isinstance(seed, str) and seed.isdigit():
        return [str(int(seed))]
    return None


# ---------------------------------------------------------------------------------------------------------
# Rounds are XYZ experiments with an origin block
# ---------------------------------------------------------------------------------------------------------

def origin_block(recipe, *, session_id, round_number, context, seeds, submission_id):
    candidates = []
    for index, candidate in enumerate(recipe.candidates):
        candidates.append({'id': candidate.id, 'key': candidate_key(context, recipe, candidate.weights), 'source': candidate.source,
                           'weights': dict(candidate.weights), 'displacement': candidate.displacement, 'x_index': index})
    return {
        'kind': ORIGIN_KIND, 'schema_version': SCHEMA_VERSION, 'session_id': session_id, 'round': round_number, 'submission_id': submission_id,
        'context_hash': context, 'seeds': list(seeds), 'algorithm': {'name': ALGORITHM, 'sampling_seed': recipe.mix['sampling_seed'], 'cursor': recipe.mix.get('cursor', 0)},
        'rules': {'members': copy.deepcopy(recipe.mix['members']), 'max_displacement': recipe.mix['max_displacement'], 'limit_units': recipe.limit,
                  'dmax_units': recipe.analysis.dmax, 'include_baseline': recipe.mix.get('include_baseline', True), 'budget': copy.deepcopy(recipe.mix['budget']),
                  'slots': recipe.slots, 'images': recipe.images},
        'global_loras': copy.deepcopy(recipe.baseline.get('loras', [])), 'candidates': candidates,
    }


def candidate_states(definition, state):
    """tested / reserved / failed / cancelled for every candidate of one round, from the experiment's own state."""
    origin = definition.get('origin') or {}
    if origin.get('kind') != ORIGIN_KIND:
        return {}
    cells, status = state.get('cells', {}), state.get('status')
    released = bool(state.get('mix_released'))
    samples = definition.get('draft', {}).get('repeat_count', 1)
    result = {}
    for candidate in origin.get('candidates', []):
        statuses = [(cells.get(f'{candidate["x_index"]}:0:0:{s}') or {}).get('status', 'missing') for s in range(samples)]
        if statuses and all(item == 'complete' for item in statuses):
            kind = 'tested'
        elif status in ACTIVE_STATUSES and not released:
            kind = 'reserved'
        elif status == 'interrupted' and not released and any(item in ('pending', 'running') for item in statuses):
            kind = 'reserved'
        elif any(item == 'cancelled' for item in statuses) or status == 'cancelled':
            kind = 'cancelled'
        else:
            kind = 'failed'
        result[candidate['id']] = {'state': kind, 'cells': statuses, 'complete': sum(item == 'complete' for item in statuses), 'samples': samples}
    return result


def used_candidates(records, engine, context):
    """Vectors tested or reserved under one context, across every session. `failed`/`cancelled` stay available."""
    used = []
    for experiment_id, record in records.items():
        definition = record['definition']
        origin = definition.get('origin') or {}
        if origin.get('kind') != ORIGIN_KIND or definition.get('engine') != engine or origin.get('context_hash') != context:
            continue
        states = candidate_states(definition, record['state'])
        for candidate in origin['candidates']:
            used.append({'experiment_id': experiment_id, 'session_id': origin['session_id'], 'round': origin['round'], 'candidate_id': candidate['id'],
                         'key': candidate['key'], 'weights': candidate['weights'], 'state': states[candidate['id']]['state']})
    return used


def session_rounds(records, session_id):
    rounds = [(record['definition']['origin']['round'], experiment_id, record) for experiment_id, record in records.items()
              if (record['definition'].get('origin') or {}).get('kind') == ORIGIN_KIND and record['definition']['origin'].get('session_id') == session_id]
    return sorted(rounds, key=lambda item: item[0])


def session_summaries(records, engine=None):
    sessions = {}
    for experiment_id, record in records.items():
        definition = record['definition']
        origin = definition.get('origin') or {}
        if origin.get('kind') != ORIGIN_KIND or (engine and definition.get('engine') != engine):
            continue
        entry = sessions.setdefault(origin['session_id'], {'id': origin['session_id'], 'engine': definition['engine'], 'rounds': 0, 'created_at': definition['created_at'], 'updated_at': definition['created_at'], 'latest_experiment': None})
        entry['rounds'] += 1
        entry['created_at'] = min(entry['created_at'], definition['created_at'])
        if definition['created_at'] >= entry['updated_at']:
            entry['updated_at'], entry['latest_experiment'] = definition['created_at'], experiment_id
    return sorted(sessions.values(), key=lambda item: item['updated_at'], reverse=True)


def remaining_vectors(origin, used_keys, offset=0, limit=200):
    """A bounded page of legal, not-yet-used vectors of one round's rules. Counts are exact; the list is lazy."""
    analysis = Analysis(origin['rules']['members'])
    if not analysis.valid:
        raise MixError('rules', '该回合的规则无法读取', 409)
    limit_units = origin['rules']['limit_units']
    context = origin['context_hash']
    total = count_space(analysis.free, limit_units)
    rows, scanned = [], 0
    skipped = 0
    for vector in enumerate_vectors(analysis.free, limit_units):
        scanned += 1
        if scanned > ENUMERATE_BUDGET:
            return {'total': total, 'rows': rows, 'complete': False, 'reason': '枚举工作量已达上限；这不代表空间已穷尽'}
        weights = {}
        for row in analysis.rows:
            iid = row.member['ref']['instanceId']
            weights[iid] = row.parsed['base'] if row.locked else vector[analysis.free.index(row)]
        key = sha256(context + '|' + ','.join(f'{row.member["ref"]["instanceId"]}:{weights[row.member["ref"]["instanceId"]]}' for row in analysis.rows))
        if key in used_keys:
            continue
        if skipped < offset:
            skipped += 1
            continue
        rows.append({'weights': weights, 'displacement': displacement_of(analysis.free, vector)})
        if len(rows) >= limit:
            return {'total': total, 'rows': rows, 'complete': False, 'next_offset': offset + len(rows)}
    return {'total': total, 'rows': rows, 'complete': True}


# ---------------------------------------------------------------------------------------------------------------------
# Candidate-grid export snapshot. The XYZ grid renderer draws it; this only remaps the round to rows x columns and decides
# what each caption, the summary and (optionally) the LoRA legend say.

def lora_marker(index):
    """①–⑳ for the first 20 members (the bundled font has them); a test set is capped at 16 active LoRAs anyway."""
    return chr(0x2460 + index) if 0 <= index < 20 else f'({index + 1})'


def lora_label(ref):
    """The library name, or the file name without folders and extension."""
    name = str(ref.get('name') or ref.get('value') or '').strip()
    if '/' in name or '\\' in name or name.lower().endswith('.safetensors'):
        name = re.split(r'[\\/]', name)[-1]
        name = re.sub(r'\.safetensors$', '', name, flags=re.IGNORECASE)
    return name or '—'


def candidate_grid(snapshot, origin, columns=None, *, lora_weights=False, created=None):
    """Lay a search round out as rows x columns of candidates.

    The image seed is shared by every candidate of a sample, so it moves out of the per-image caption into the summary
    (one sample) or the sample heading (several); a sample whose cells disagree keeps the seed per image. With
    `lora_weights`, a numbered legend of the test LoRAs (and a note of the fixed globals) follows the summary and every
    image lists its weights by the same numbers."""
    import math
    candidates = origin['candidates']
    count = len(candidates)
    columns = min(columns or max(1, math.ceil(math.sqrt(count))), count)
    rows = math.ceil(count / columns)
    samples = snapshot.get('repeat_count', 1) or 1
    by_index = {item['x_index']: item for item in candidates}
    seeds = {}
    for cell in snapshot['cells']:
        seeds.setdefault(cell['coordinate'][3], set()).add(str(cell.get('seed')))
    shared = {sample: next(iter(values)) for sample, values in seeds.items() if len(values) == 1 and next(iter(values)) not in ('', 'None')}
    members = origin['rules']['members']
    member_ids = [member['ref']['instanceId'] for member in members]
    for cell in snapshot['cells']:
        x, _y, _z, sample = cell['coordinate']
        item = by_index[x]
        displacement = format_units(item['displacement'])
        cell['origin_coordinate'] = list(cell['coordinate'])
        cell['coordinate'] = [x % columns, x // columns, 0, sample]
        cell['candidate'] = {'id': item['id'], 'key': item['key'], 'source': item['source'], 'displacement': displacement,
                             'weights': {iid: format_units(units) for iid, units in item['weights'].items()}}
        caption = f"{item['id']}{' · 基线' if item['source'] == 'baseline' else ''} · Δ{displacement}"
        cell['caption'] = caption if sample in shared else f"{caption} · 种子 {cell.get('seed')}"
        if lora_weights:
            cell['tokens'] = [f'{lora_marker(index)} {format_units(item["weights"][iid])}' for index, iid in enumerate(member_ids) if iid in item['weights']]
    complete = sum(cell['status'] == 'complete' for cell in snapshot['cells'])
    snapshot['axes'] = {'x': {'field': None, 'labels': [''] * columns}, 'y': {'field': None, 'labels': [''] * rows}, 'z': {'field': None, 'labels': ['']}}
    snapshot['dimensions'] = {'x': columns, 'y': rows, 'z': 1}
    baseline = snapshot['baseline']
    seed_text = ''
    if len(shared) == samples and samples == 1:
        seed_text = f'种子 {shared[0]}'
    elif shared:
        snapshot['sample_labels'] = [f'样本 {sample + 1}' + (f' · 种子 {shared[sample]}' if sample in shared else '') for sample in range(samples)]
        seed_text = '种子见各样本标题'
    snapshot['title'] = f"LoRA 混合搜索 · {snapshot['engine']} · 第 {origin['round']} 轮"
    snapshot['subtitle'] = ' · '.join(part for part in [
        f"{count} 个候选 × {samples} 样本 · {columns} 列 × {rows} 行", f"{complete}/{len(snapshot['cells'])} 格完成",
        f"全局 LoRA {len(origin['global_loras'])} 个 · 测试 LoRA {len(members)} 个",
        f"{baseline.get('width')}×{baseline.get('height')} · {baseline.get('steps')} 步 · CFG {baseline.get('cfg')}", seed_text,
        f"会话 {origin['session_id'][:8]} · 条件 {origin['context_hash'][:8]}" + (f' · {created}' if created else '')] if part)
    if lora_weights:
        globals_ = [f"{lora_label(item)} {item.get('weight')}{'（停用）' if item.get('enabled') is False else ''}" for item in origin['global_loras']]
        snapshot['legend'] = {
            'items': [f"{lora_marker(index)} {lora_label(member['ref'])}{'（锁定）' if member['rule'].get('locked') else ''}" for index, member in enumerate(members)],
            'note': ('全局 LoRA（每格相同）：' + '、'.join(globals_)) if globals_ else None}
    snapshot['search'] = {
        'session_id': origin['session_id'], 'round': origin['round'], 'context_hash': origin['context_hash'], 'columns': columns, 'rows': rows, 'candidates_count': count,
        'algorithm': origin['algorithm'], 'rules': origin['rules'], 'global_loras': origin['global_loras'], 'seeds': origin['seeds'],
        'seed_placement': 'summary' if seed_text and samples == 1 else 'sample-heading' if shared else 'caption', 'lora_weights': bool(lora_weights),
        'lora_legend': [{'marker': lora_marker(index), 'instanceId': iid, 'name': lora_label(member['ref'])} for index, (iid, member) in enumerate(zip(member_ids, members))],
        'candidates': [{'id': item['id'], 'key': item['key'], 'source': item['source'], 'displacement': format_units(item['displacement']),
                        'position': [item['x_index'] % columns, item['x_index'] // columns],
                        'weights': {iid: format_units(units) for iid, units in item['weights'].items()}} for item in candidates],
    }
    return snapshot
