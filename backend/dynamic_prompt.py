"""Clean-room dynamic prompt plans and identity-safe UNet conditioning leases.

Schedules ``[a:b:n]``, alternation ``[a|b]`` and positive ``AND`` follow what
Forge f2.0.1 observably does; the P4.1 differential oracle runs Forge's own
schedule code beside this module. Nothing is copied from it. Every prompt is
rendered at the breakpoints of one sampling pass, the states of one prompt are
encoded together (Forge normalises emphasis over that batch), and composable
``AND`` arithmetic happens on UNet predictions rather than on CLIP embeddings.
"""

from __future__ import annotations

import math
import re
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from types import MappingProxyType
from typing import Mapping

import torch

try:
    from .forge_prompt import (
        CHUNK_LENGTH,
        ConditioningPlan,
        EncodedBranch,
        PreparedConditioning,
        encode_forge_texts,
        parse_prompt_attention,
        prepare_prompt_conditioning,
    )
except ImportError:
    from forge_prompt import (
        CHUNK_LENGTH,
        ConditioningPlan,
        EncodedBranch,
        PreparedConditioning,
        encode_forge_texts,
        parse_prompt_attention,
        prepare_prompt_conditioning,
    )


MAX_AND_TERMS = 16
MAX_DYNAMIC_STATES = 256
_AND_RE = re.compile(r"\bAND\b")
# A positive term may end in `:weight` — an unsigned-exponent-free decimal.
# Anything else (`:inf`, `:1e2`, `:x`) stays part of the term's text.
_TERM_WEIGHT = re.compile(r"\s*:\s*([-+]?(?:\d+\.?|\d*\.\d+))\s*\Z")
# The schedule threshold closes its bracket: optional spaces, a signed number
# (integer, decimal or exponent form, ASCII digits only), optional spaces, `]`.
_SCHEDULE_NUMBER = re.compile(r"\s*([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)\s*\]")
_STRUCTURE = frozenset("[]():|")


class PromptLeaseError(RuntimeError):
    pass


class PromptPlanLimitError(ValueError):
    pass


class _ParseFailure(Exception):
    pass


@dataclass(frozen=True)
class _Literal:
    text: str


@dataclass(frozen=True)
class _Sequence:
    parts: tuple[object, ...]


@dataclass(frozen=True)
class _Schedule:
    before: object
    after: object
    number: str


@dataclass(frozen=True)
class _Alternation:
    options: tuple[object, ...]


@dataclass(frozen=True)
class PromptTimeline:
    """The step horizon one sampling pass evaluates prompt schedules against.

    ``steps`` is the pass's requested steps times the denoiser calls each step
    makes (two for Heun and DPM2), which is what Forge counts. A second pass
    (Hires / resample after a base pass) continues the first pass's timeline:
    integer thresholds lose ``int_offset`` — the first pass's horizon — and
    fractional ones lose ``float_offset``, 1.0.
    """

    steps: int
    int_offset: int = 0
    float_offset: float = 0.0

    def __post_init__(self):
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps <= 0:
            raise PromptLeaseError("a prompt timeline needs a positive integer step horizon")
        if isinstance(self.int_offset, bool) or not isinstance(self.int_offset, int) or self.int_offset < 0:
            raise PromptLeaseError("a prompt timeline offset must be a non-negative integer")
        if not math.isfinite(float(self.float_offset)) or self.float_offset < 0:
            raise PromptLeaseError("a prompt timeline fractional offset must be finite and non-negative")


def calls_per_step(pipeline) -> int:
    order = getattr(getattr(pipeline, "scheduler", None), "order", 1)
    return order if isinstance(order, int) and order >= 1 else 1


def sampling_timeline(pipeline, steps: int, *, continues: PromptTimeline | None = None) -> PromptTimeline:
    """The timeline for a pass of ``steps`` on ``pipeline``'s configured scheduler."""

    horizon = int(steps) * calls_per_step(pipeline)
    if continues is None:
        return PromptTimeline(horizon)
    return PromptTimeline(horizon, int_offset=continues.steps, float_offset=1.0)


def _sequence(parts):
    parts = tuple(parts)
    if not parts:
        return _Literal("")
    if len(parts) == 1:
        return parts[0]
    return _Sequence(parts)


class _ScheduleParser:
    """Recursive reading of the observable Forge schedule language.

    Inside any group a construct must be well formed or the group fails. At the
    top level a group that fails leaves its opening bracket as literal text and
    parsing continues after it, so one malformed bracket does not disable a
    later schedule. A ``|`` outside an alternation, or a backslash that cannot
    escape the next character, makes the whole prompt static text.
    """

    def __init__(self, text: str):
        self.text = text
        self.length = len(text)

    def _at(self, position, character):
        return position < self.length and self.text[position] == character

    def _plain_end(self, position):
        text = self.text
        while position < self.length:
            character = text[position]
            if character == "\\":
                if position + 1 < self.length and text[position + 1] != "\n":
                    position += 2
                    continue
                break
            if character in _STRUCTURE:
                break
            position += 1
        return position

    def parse(self):
        parts = []
        position = 0
        while position < self.length:
            character = self.text[position]
            if character in "([":
                try:
                    node, position = self._group(position)
                except _ParseFailure:
                    parts.append(_Literal(character))
                    position += 1
                else:
                    parts.append(node)
                continue
            if character in "]):":
                parts.append(_Literal(character))
                position += 1
                continue
            end = self._plain_end(position)
            if end == position:
                raise _ParseFailure()
            parts.append(_Literal(self.text[position:end]))
            position = end
        return _sequence(parts)

    def _prompt(self, position):
        parts = []
        while position < self.length:
            character = self.text[position]
            if character in "([":
                node, position = self._group(position)
                parts.append(node)
                continue
            if character in "]):|":
                break
            end = self._plain_end(position)
            if end == position:
                raise _ParseFailure()
            parts.append(_Literal(self.text[position:end]))
            position = end
        return _sequence(parts), position

    def _group(self, position):
        if self.text[position] == "(":
            first, position = self._prompt(position + 1)
            if self._at(position, ")"):
                return _sequence((_Literal("("), first, _Literal(")"))), position + 1
            if self._at(position, ":"):
                second, position = self._prompt(position + 1)
                if self._at(position, ")"):
                    return _sequence((_Literal("("), first, _Literal(":"), second, _Literal(")"))), position + 1
            raise _ParseFailure()
        first, position = self._prompt(position + 1)
        if self._at(position, "]"):
            return _sequence((_Literal("["), first, _Literal("]"))), position + 1
        if self._at(position, "|"):
            options = [first]
            while self._at(position, "|"):
                option, position = self._prompt(position + 1)
                options.append(option)
            if self._at(position, "]"):
                return _Alternation(tuple(options)), position + 1
            raise _ParseFailure()
        if self._at(position, ":"):
            number = _SCHEDULE_NUMBER.match(self.text, position + 1)
            if number is not None:
                return _Schedule(_Literal(""), first, number.group(1)), number.end()
            second, position = self._prompt(position + 1)
            if self._at(position, ":"):
                number = _SCHEDULE_NUMBER.match(self.text, position + 1)
                if number is not None:
                    return _Schedule(first, second, number.group(1)), number.end()
        raise _ParseFailure()


def _is_dynamic(node) -> bool:
    if isinstance(node, (_Schedule, _Alternation)):
        return True
    if isinstance(node, _Sequence):
        return any(_is_dynamic(part) for part in node.parts)
    return False


def _threshold(node: _Schedule, timeline: PromptTimeline) -> int:
    value = float(node.number)
    if "." in node.number:
        value = (value - timeline.float_offset) * timeline.steps
    else:
        value = value - timeline.int_offset
    if not math.isfinite(value):
        raise PromptPlanLimitError("prompt schedule threshold is not a finite number")
    return min(timeline.steps, int(value))


def _breakpoints(node, timeline: PromptTimeline) -> tuple[int, ...]:
    points = {timeline.steps}
    alternates = False
    pending = [node]
    while pending:
        current = pending.pop()
        if isinstance(current, _Schedule):
            when = _threshold(current, timeline)
            if when >= 1:
                points.add(when)
            pending.extend((current.before, current.after))
        elif isinstance(current, _Alternation):
            alternates = True
            pending.extend(current.options)
        elif isinstance(current, _Sequence):
            pending.extend(current.parts)
    if alternates:
        points.update(range(1, timeline.steps + 1))
    return tuple(sorted(points))


def _render(node, step: int, timeline: PromptTimeline) -> str:
    if isinstance(node, _Literal):
        return node.text
    if isinstance(node, _Sequence):
        return "".join(_render(part, step, timeline) for part in node.parts)
    if isinstance(node, _Schedule):
        selected = node.before if step <= _threshold(node, timeline) else node.after
        return _render(selected, step, timeline)
    if isinstance(node, _Alternation):
        return _render(node.options[(step - 1) % len(node.options)], step, timeline)
    raise TypeError(f"unknown dynamic prompt node: {type(node).__name__}")


@dataclass(frozen=True)
class TermTimeline:
    """A term's text at each breakpoint of one pass.

    Invocation ``i`` (0-based) uses the first breakpoint ``>= i``; an index past
    the last breakpoint falls back to the first, as Forge's lookup does.
    """

    breakpoints: tuple[int, ...]
    texts: tuple[str, ...]

    def text_at(self, call_index: int) -> str:
        if isinstance(call_index, bool) or not isinstance(call_index, int) or call_index < 0:
            raise PromptLeaseError("dynamic prompt call_index must be a non-negative integer")
        for point, text in zip(self.breakpoints, self.texts):
            if call_index <= point:
                return text
        return self.texts[0]


@dataclass(frozen=True)
class PromptTerm:
    syntax: object
    weight: float = 1.0
    fallback: bool = False
    explicit_weight: bool = False
    source: str = ""

    def timeline(self, timeline: PromptTimeline) -> TermTimeline:
        points = _breakpoints(self.syntax, timeline)
        return TermTimeline(points, tuple(_render(self.syntax, point, timeline) for point in points))


@dataclass(frozen=True)
class PromptSyntaxPlan:
    """Immutable syntax for one positive or negative prompt."""

    source: str
    terms: tuple[PromptTerm, ...]
    dynamic: bool
    composable: bool
    fallback: bool

    def state_for(self, call_index: int, timeline: PromptTimeline):
        return tuple((term.timeline(timeline).text_at(call_index), term.weight) for term in self.terms)

    def unique_texts(self, timeline: PromptTimeline) -> int:
        return sum(len(set(term.timeline(timeline).texts)) for term in self.terms)


@dataclass(frozen=True)
class ConditioningSyntaxPlan:
    positive: PromptSyntaxPlan
    negative: PromptSyntaxPlan

    @property
    def dynamic(self):
        return self.positive.dynamic or self.negative.dynamic

    @property
    def composable(self):
        return self.positive.composable

    def validate(self, timeline: PromptTimeline):
        """Render every state of this pass, bound the encoder work, parse attention."""

        states = self.positive.unique_texts(timeline) + self.negative.unique_texts(timeline)
        if states > MAX_DYNAMIC_STATES:
            raise PromptPlanLimitError(
                f"positive and negative prompts require more than {MAX_DYNAMIC_STATES} "
                "encoded text states; simplify schedules or alternations"
            )
        for term in (*self.positive.terms, *self.negative.terms):
            for text in set(term.timeline(timeline).texts):
                parse_prompt_attention(text)
        return states


@dataclass(frozen=True)
class EncodedTerm:
    branch: EncodedBranch
    weight: float
    text: str


@dataclass(frozen=True)
class EncodedConditionState:
    positive: tuple[EncodedTerm, ...]
    negative: EncodedBranch
    call_index: int


@dataclass(frozen=True)
class EncodedTermPlan:
    weight: float
    timeline: TermTimeline
    encodings: Mapping[str, EncodedBranch]

    def term_at(self, call_index: int) -> EncodedTerm:
        text = self.timeline.text_at(call_index)
        return EncodedTerm(self.encodings[text], self.weight, text)


@dataclass(frozen=True)
class EncodedPromptPlan:
    family: str
    syntax: ConditioningSyntaxPlan
    timeline: PromptTimeline
    positive: tuple[EncodedTermPlan, ...]
    negative: EncodedTermPlan
    emphasis: str = "forge"

    @property
    def composable(self):
        return self.syntax.composable

    def state_for(self, call_index: int) -> EncodedConditionState:
        return EncodedConditionState(
            tuple(term.term_at(call_index) for term in self.positive),
            self.negative.term_at(call_index).branch,
            call_index,
        )


def _parse_term(text: str):
    try:
        return _ScheduleParser(text).parse(), False
    except _ParseFailure:
        return _Literal(text), True


def static_prompt_plan(text: str, *, allow_and: bool = True) -> PromptSyntaxPlan:
    """Compile schedule, alternation and, for positives, ``AND`` terms.

    A positive prompt splits on the word ``AND``; each term may end in
    ``:weight``. A term whose brackets cannot be read at all is static text,
    exactly like a prompt without dynamic syntax.
    """

    if not isinstance(text, str):
        raise TypeError("prompt text must be a string")
    raw_terms = _AND_RE.split(text) if allow_and else [text]
    if len(raw_terms) > MAX_AND_TERMS:
        raise PromptPlanLimitError(f"positive prompts support at most {MAX_AND_TERMS} AND terms")
    terms = []
    for raw in raw_terms:
        body, weight, explicit = raw, 1.0, False
        if allow_and:
            match = _TERM_WEIGHT.search(raw)
            if match is not None:
                body = raw[:match.start()]
                weight = float(match.group(1))
                explicit = True
            # CLIP tokenises surrounding whitespace away; stripping keeps term
            # identity (and therefore encoding reuse) independent of spacing.
            body = body.strip()
        syntax, fallback = _parse_term(body)
        terms.append(PromptTerm(syntax, weight, fallback, explicit, body))
    dynamic = any(_is_dynamic(term.syntax) for term in terms)
    composable = allow_and and (len(terms) > 1 or any(term.explicit_weight for term in terms))
    return PromptSyntaxPlan(text, tuple(terms), dynamic, composable, any(term.fallback for term in terms))


def compile_conditioning_syntax(prompt: str, negative_prompt: str, timelines=()) -> ConditioningSyntaxPlan:
    """Compile both prompts; validate every state of each given pass timeline."""

    plan = ConditioningSyntaxPlan(
        static_prompt_plan(prompt, allow_and=True),
        static_prompt_plan(negative_prompt, allow_and=False),
    )
    if not plan.dynamic:
        # A static prompt has a single state whatever the timeline.
        timelines = (PromptTimeline(1),)
    for timeline in timelines:
        plan.validate(timeline)
    return plan


def prompt_plan_audit(plan: ConditioningSyntaxPlan, timeline: PromptTimeline | None = None) -> dict:
    audit = {
        "mode": "dynamic" if plan.dynamic or plan.composable else "static",
        "positive_terms": len(plan.positive.terms),
        "schedule_or_alternation": plan.dynamic,
        "and_composition": plan.composable,
        "static_fallback": plan.positive.fallback or plan.negative.fallback,
        "limits": {"terms": MAX_AND_TERMS, "states": MAX_DYNAMIC_STATES},
    }
    if timeline is not None:
        audit.update({
            "timeline_steps": timeline.steps,
            "timeline_continues": timeline.int_offset > 0 or timeline.float_offset > 0,
            "positive_states": plan.positive.unique_texts(timeline),
            "negative_states": plan.negative.unique_texts(timeline),
        })
    return audit


def _conditioning_device(pipeline):
    return getattr(pipeline, "_execution_device", torch.device("cuda"))


def _encode_term(pipeline, family, term: PromptTerm, timeline: PromptTimeline, device, *, negative, weight, clip_skip=None):
    schedule = term.timeline(timeline)
    # One Forge batch: every breakpoint's text, repeats included, so the
    # emphasis mean counts each state as often as Forge's batch does.
    branches = encode_forge_texts(pipeline, family, schedule.texts, device, negative=negative, clip_skip=clip_skip)
    encodings = {}
    for text, branch in zip(schedule.texts, branches):
        if text not in encodings:
            encodings[text] = EncodedBranch(
                branch.cross_attention.clone(),
                None if branch.pooled is None else branch.pooled.clone(),
                branch.diagnostics,
            )
    return EncodedTermPlan(weight, schedule, MappingProxyType(encodings))


@torch.inference_mode()
def prepare_dynamic_prompt_conditioning(
    pipeline, family: str, prompt: str, negative_prompt: str, *, emphasis="forge", timeline: PromptTimeline | None = None,
    clip_skip: int | None = None,
):
    """Encode one pass's immutable prompt plan.

    Static prompts keep the verified static kernel. Dynamic prompts need the
    pass ``timeline``; composable prompts without schedules have one state.
    ``clip_skip`` is the job's effective layer, the same for every positive
    term, the negative and every schedule state.
    """

    if emphasis != "forge":
        return prepare_prompt_conditioning(pipeline, family, prompt, negative_prompt, emphasis=emphasis, clip_skip=clip_skip)
    syntax = compile_conditioning_syntax(prompt, negative_prompt)
    if not syntax.dynamic and not syntax.composable:
        return prepare_prompt_conditioning(pipeline, family, prompt, negative_prompt, emphasis=emphasis, clip_skip=clip_skip)
    if timeline is None:
        if syntax.dynamic:
            raise PromptLeaseError("dynamic prompts need the sampling pass's step timeline")
        timeline = PromptTimeline(1)
    syntax.validate(timeline)
    device = _conditioning_device(pipeline)
    # Forge encodes identical AND terms once; each term keeps its own weight.
    shared = {}
    positive = []
    for term in syntax.positive.terms:
        if term.source not in shared:
            shared[term.source] = _encode_term(pipeline, family, term, timeline, device, negative=False, weight=1.0, clip_skip=clip_skip)
        encoded = shared[term.source]
        positive.append(EncodedTermPlan(term.weight, encoded.timeline, encoded.encodings))
    negative = _encode_term(
        pipeline, family, syntax.negative.terms[0], timeline, device, negative=True, weight=1.0, clip_skip=clip_skip
    )
    plan = EncodedPromptPlan(family, syntax, timeline, tuple(positive), negative)
    initial = plan.state_for(0)
    first = initial.positive[0].branch
    values = {
        "prompt_embeds": first.cross_attention[:, :CHUNK_LENGTH],
        "negative_prompt_embeds": initial.negative.cross_attention[:, :CHUNK_LENGTH],
    }
    if family == "sdxl":
        values.update({
            "pooled_prompt_embeds": first.pooled,
            "negative_pooled_prompt_embeds": initial.negative.pooled,
        })
    return PreparedConditioning(values, plan)


def _repeat(tensor, count, *, like=None):
    if tensor is None:
        return None
    result = tensor.to(
        device=like.device if like is not None else tensor.device,
        dtype=like.dtype if like is not None else tensor.dtype,
    )
    return result.expand(count, *result.shape[1:])


def _slice_batch(value, start, end, total):
    if torch.is_tensor(value) and value.ndim and value.shape[0] == total:
        return value[start:end]
    if isinstance(value, dict):
        return {key: _slice_batch(item, start, end, total) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_slice_batch(item, start, end, total) for item in value)
    if isinstance(value, list) and len(value) == total:
        return value[start:end]
    return value


def _prepend_first_group(value, group_size, existing_size):
    if torch.is_tensor(value) and value.ndim and value.shape[0] == existing_size:
        return torch.cat((value[:group_size], value), dim=0)
    if isinstance(value, dict):
        return {key: _prepend_first_group(item, group_size, existing_size) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_prepend_first_group(item, group_size, existing_size) for item in value)
    if isinstance(value, list) and len(value) == existing_size:
        return value[:group_size] + value
    return value


def _repeat_batch_tree(value, group_size, repeats):
    if torch.is_tensor(value) and value.ndim and value.shape[0] == group_size:
        return torch.cat((value,) * repeats, dim=0)
    if isinstance(value, dict):
        return {key: _repeat_batch_tree(item, group_size, repeats) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_repeat_batch_tree(item, group_size, repeats) for item in value)
    if isinstance(value, list) and len(value) == group_size:
        return value * repeats
    return value


def _output_tensor(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, tuple):
        return output[0]
    if isinstance(output, dict):
        return output["sample"]
    sample = getattr(output, "sample", None)
    if torch.is_tensor(sample):
        return sample
    raise TypeError("UNet prompt lease requires tensor, tuple, mapping, or sample output")


def _with_output_tensor(template, tensor):
    if torch.is_tensor(template):
        return tensor
    if isinstance(template, tuple):
        return (tensor, *template[1:])
    if isinstance(template, dict):
        result = dict(template)
        result["sample"] = tensor
        return type(template)(result) if type(template) is not dict else result
    if is_dataclass(template):
        values = {field.name: getattr(template, field.name) for field in fields(template)}
        values["sample"] = tensor
        return type(template)(**values)
    try:
        return type(template)(sample=tensor)
    except TypeError as error:
        raise TypeError("UNet prompt lease cannot rebuild its output type") from error


def _compose(reference, predictions, weights, *, scale=1.0):
    result = reference
    for prediction, weight in zip(predictions, weights):
        result = result + (prediction - reference) * (float(weight) * float(scale))
    if not torch.isfinite(result).all():
        raise PromptLeaseError("AND prediction composition produced NaN or infinity")
    return result


class StaticConditioningLease:
    """Replace Diffusers dummy conditioning only for one pipeline invocation."""

    def __init__(
        self,
        pipeline,
        prepared: PreparedConditioning,
        guidance_scale: float,
        pag_scale: float = 0.0,
    ):
        if not isinstance(prepared, PreparedConditioning) or prepared.plan is None:
            raise PromptLeaseError("a Forge conditioning plan is required")
        self.pipeline = pipeline
        self.unet = getattr(pipeline, "unet", None)
        if self.unet is None:
            raise PromptLeaseError("pipeline has no UNet to lease")
        self.plan = prepared.plan
        self.guidance_scale = float(guidance_scale)
        self.do_cfg = self.guidance_scale > 1.0
        self.do_pag = float(pag_scale or 0.0) > 0.0
        self.token = uuid.uuid4().hex
        lock = getattr(pipeline, "_xiriai_prompt_lease_lock", None)
        if lock is None:
            lock = threading.RLock()
            pipeline._xiriai_prompt_lease_lock = lock
        self.lock = lock
        self.original_forward = None
        self.wrapper = None
        self.serial_wrapper = None
        self.call_index = 0
        self._entered = False
        self._sentinels = []

    def __enter__(self):
        self.lock.acquire()
        try:
            if getattr(self.pipeline, "_xiriai_prompt_lease_token", None) is not None:
                raise PromptLeaseError("pipeline already has an active prompt lease")
            self.pipeline._xiriai_prompt_lease_token = self.token
            self._bind()
            self._install_sentinel()
            return self
        except Exception:
            self._remove_sentinel()
            if self.wrapper is not None and self.unet.forward is self.wrapper:
                self.unet.forward = self.serial_wrapper or self.original_forward
            if getattr(self.pipeline, "_xiriai_prompt_lease_token", None) == self.token:
                delattr(self.pipeline, "_xiriai_prompt_lease_token")
            self.lock.release()
            raise

    def begin_invocation(self, _active_pipeline=None):
        """A pipeline call (or its retry after OOM) starts the pass's timeline over."""

        self.call_index = 0

    def _install_sentinel(self):
        """Refuse any UNet call that did not pass through the lease.

        Module hooks survive ``forward`` being reassigned, which is exactly what
        displaces the lease: Diffusers' lazy-prefetch group offloading removes
        its own hook after the first forward and restores the forward it saw at
        registration. A call that reaches the UNet around the lease would sample
        from the one-block placeholder tensors, so it fails instead.
        """

        register_pre = getattr(self.unet, "register_forward_pre_hook", None)
        register_post = getattr(self.unet, "register_forward_hook", None)
        if register_pre is None or register_post is None:
            return

        def arm(_module, _args):
            self._entered = False

        def verify(_module, _args, _output):
            if not self._entered:
                raise PromptLeaseError(
                    "the UNet was called around the Forge conditioning lease; refusing to sample "
                    "from placeholder conditioning"
                )

        self._sentinels = [register_pre(arm), register_post(verify)]

    def _remove_sentinel(self):
        for handle in self._sentinels:
            handle.remove()
        self._sentinels = []

    def _bind(self):
        current = self.unet.forward
        serial = getattr(self.unet, "_xiriai_sequential_batch_forward", None)
        self.serial_wrapper = None
        if current is serial:
            self.serial_wrapper = serial
            current = getattr(self.unet, "_xiriai_full_batch_forward", None)
            if current is None:
                raise PromptLeaseError("serial CFG wrapper has no recoverable full-batch forward")
        if self.wrapper is not None and current is self.wrapper:
            # Something re-wrapped the UNet on top of this lease; never lease the lease.
            current = self.original_forward
        self.original_forward = current

        def leased_forward(sample, *args, **kwargs):
            if getattr(self.pipeline, "_xiriai_prompt_lease_token", None) != self.token:
                raise PromptLeaseError("stale prompt lease invocation")
            self._entered = True
            try:
                output = self._forward(sample, args, kwargs)
                self.call_index += 1
            except BaseException:
                # OOM escalation rebuilds Accelerate hooks after this exception
                # leaves the UNet. Remove this lease first so the new hook never
                # captures a stale wrapper as its _old_forward.
                self.suspend()
                raise
            if self.unet.forward is not self.wrapper:
                # A hook run by this very call restored an older forward (Diffusers'
                # lazy prefetch does so after the first forward). Re-seat on the
                # forward the module wears now, before the next step calls it.
                self._bind()
            return output

        self.wrapper = leased_forward
        self.unet.forward = leased_forward

    def suspend(self):
        if self.unet.forward is self.wrapper:
            self.unet.forward = self.serial_wrapper or self.original_forward

    def ensure_bound(self):
        """Rebind after an OOM residency transition replaced the hook layer."""

        if getattr(self.pipeline, "_xiriai_prompt_lease_token", None) != self.token:
            raise PromptLeaseError("prompt lease is no longer active")
        if self.unet.forward is self.wrapper:
            return
        self._bind()

    def _branch_kwargs(self, kwargs, start, end, total, branch, count, *, encoder_positional=False):
        result = {key: _slice_batch(value, start, end, total) for key, value in kwargs.items()}
        if not encoder_positional:
            incoming = kwargs.get("encoder_hidden_states")
            result["encoder_hidden_states"] = _repeat(branch.cross_attention, count, like=incoming)
        added = result.get("added_cond_kwargs")
        if isinstance(added, dict) and branch.pooled is not None:
            added = dict(added)
            incoming_pooled = kwargs.get("added_cond_kwargs", {}).get("text_embeds")
            added["text_embeds"] = _repeat(branch.pooled, count, like=incoming_pooled)
            result["added_cond_kwargs"] = added
        return result

    def _call(self, sample, args, kwargs, start, end, branch):
        total = sample.shape[0]
        count = end - start
        branch_args = [_slice_batch(value, start, end, total) for value in args]
        encoder_positional = len(branch_args) >= 2 and "encoder_hidden_states" not in kwargs
        if encoder_positional:
            branch_args[1] = _repeat(branch.cross_attention, count, like=args[1])
        branch_kwargs = self._branch_kwargs(
            kwargs, start, end, total, branch, count, encoder_positional=encoder_positional
        )
        return self.original_forward(sample[start:end], *branch_args, **branch_kwargs)

    def _homogeneous_pag_call(self, sample, args, kwargs, start, size, branch, groups):
        total = sample.shape[0]
        group = sample[start:start + size]
        repeated_sample = torch.cat((group,) * groups, dim=0)
        branch_args = [_slice_batch(value, start, start + size, total) for value in args]
        branch_args = [_repeat_batch_tree(value, size, groups) for value in branch_args]
        encoder_positional = len(branch_args) >= 2 and "encoder_hidden_states" not in kwargs
        if encoder_positional:
            branch_args[1] = _repeat(branch.cross_attention, size * groups, like=args[1])
        branch_kwargs = self._branch_kwargs(
            kwargs, start, start + size, total, branch, size, encoder_positional=encoder_positional
        )
        branch_kwargs = {
            key: _repeat_batch_tree(value, size, groups) for key, value in branch_kwargs.items()
        }
        if not encoder_positional:
            branch_kwargs["encoder_hidden_states"] = _repeat(
                branch.cross_attention, size * groups, like=kwargs.get("encoder_hidden_states")
            )
        if isinstance(branch_kwargs.get("added_cond_kwargs"), dict) and branch.pooled is not None:
            branch_kwargs["added_cond_kwargs"]["text_embeds"] = _repeat(
                branch.pooled,
                size * groups,
                like=kwargs.get("added_cond_kwargs", {}).get("text_embeds"),
            )
        return self.original_forward(repeated_sample, *branch_args, **branch_kwargs)

    def _positive_pag_triplet(self, sample, args, kwargs, size, branch):
        total = sample.shape[0]
        pair = sample[size:]
        triplet = torch.cat((pair[:size], pair), dim=0)
        branch_args = [
            _prepend_first_group(_slice_batch(value, size, total, total), size, size * 2)
            for value in args
        ]
        encoder_positional = len(branch_args) >= 2 and "encoder_hidden_states" not in kwargs
        if encoder_positional:
            branch_args[1] = _repeat(branch.cross_attention, size * 3, like=args[1])
        branch_kwargs = self._branch_kwargs(
            kwargs, size, total, total, branch, size * 2, encoder_positional=encoder_positional
        )
        branch_kwargs = {
            key: _prepend_first_group(value, size, size * 2) for key, value in branch_kwargs.items()
        }
        if not encoder_positional:
            branch_kwargs["encoder_hidden_states"] = _repeat(
                branch.cross_attention, size * 3, like=kwargs.get("encoder_hidden_states")
            )
        if isinstance(branch_kwargs.get("added_cond_kwargs"), dict) and branch.pooled is not None:
            branch_kwargs["added_cond_kwargs"]["text_embeds"] = _repeat(
                branch.pooled,
                size * 3,
                like=kwargs.get("added_cond_kwargs", {}).get("text_embeds"),
            )
        return self.original_forward(triplet, *branch_args, **branch_kwargs)

    def _forward(self, sample, args, kwargs):
        if isinstance(self.plan, EncodedPromptPlan):
            state = self.plan.state_for(self.call_index)
            if self.plan.composable:
                return self._forward_composable(sample, args, kwargs, state)
            return self._forward_static(sample, args, kwargs, state.positive[0].branch, state.negative)
        if not isinstance(self.plan, ConditioningPlan):
            raise PromptLeaseError("unknown Forge conditioning plan type")
        return self._forward_static(sample, args, kwargs, self.plan.positive, self.plan.negative)

    def _forward_static(self, sample, args, kwargs, positive, negative):
        total = int(sample.shape[0])
        groups = 3 if self.do_pag and self.do_cfg else 2 if (self.do_pag or self.do_cfg) else 1
        if total < groups or total % groups:
            raise PromptLeaseError(f"UNet batch {total} is incompatible with the expected {groups}-way guidance layout")
        size = total // groups
        if groups == 1:
            return self._call(sample, args, kwargs, 0, total, positive)
        if self.do_pag and not self.do_cfg:
            branch_args = list(args)
            encoder_positional = len(branch_args) >= 2 and "encoder_hidden_states" not in kwargs
            if encoder_positional:
                branch_args[1] = _repeat(positive.cross_attention, total, like=args[1])
            branch_kwargs = self._branch_kwargs(
                kwargs, 0, total, total, positive, total, encoder_positional=encoder_positional
            )
            return self.original_forward(sample, *branch_args, **branch_kwargs)
        if self.do_cfg and not self.do_pag:
            uncond = self._call(sample, args, kwargs, 0, size, negative)
            cond = self._call(sample, args, kwargs, size, total, positive)
            return _with_output_tensor(uncond, torch.cat((_output_tensor(uncond), _output_tensor(cond)), dim=0))

        # Diffusers 0.38 PAG processors hard-split three ways. Unequal U/C
        # sequence lengths therefore use homogeneous U and legal [C,C,P] calls.
        u_output = self._homogeneous_pag_call(sample, args, kwargs, 0, size, negative, 3)
        cp_output = self._positive_pag_triplet(sample, args, kwargs, size, positive)
        joined = torch.cat((_output_tensor(u_output)[:size], _output_tensor(cp_output)[size:]), dim=0)
        return _with_output_tensor(u_output, joined)

    def _forward_composable(self, sample, args, kwargs, state: EncodedConditionState):
        total = int(sample.shape[0])
        groups = 3 if self.do_pag and self.do_cfg else 2 if self.do_pag or self.do_cfg else 1
        if total < groups or total % groups:
            raise PromptLeaseError(f"UNet batch {total} is incompatible with the expected {groups}-way guidance layout")
        size = total // groups
        weights = [term.weight for term in state.positive]

        if not self.do_pag and self.do_cfg:
            uncond_output = self._call(sample, args, kwargs, 0, size, state.negative)
            uncond = _output_tensor(uncond_output)
            predictions = [
                _output_tensor(self._call(sample, args, kwargs, size, total, term.branch))
                for term in state.positive
            ]
            composed = _compose(uncond, predictions, weights)
            return _with_output_tensor(uncond_output, torch.cat((uncond, composed), dim=0))

        if not self.do_pag:
            uncond_output = self._call(sample, args, kwargs, 0, total, state.negative)
            uncond = _output_tensor(uncond_output)
            predictions = [
                _output_tensor(self._call(sample, args, kwargs, 0, total, term.branch))
                for term in state.positive
            ]
            target = _compose(uncond, predictions, weights, scale=self.guidance_scale)
            return _with_output_tensor(uncond_output, target)

        if self.do_cfg:
            uncond_output = self._homogeneous_pag_call(sample, args, kwargs, 0, size, state.negative, 3)
            uncond = _output_tensor(uncond_output)[:size]
            cond_predictions = []
            perturbed_predictions = []
            for term in state.positive:
                output = _output_tensor(self._positive_pag_triplet(sample, args, kwargs, size, term.branch))
                cond_predictions.append(output[size:size * 2])
                perturbed_predictions.append(output[size * 2:])
            composed_cond = _compose(uncond, cond_predictions, weights)
            composed_perturbed = _compose(uncond, perturbed_predictions, weights)
            joined = torch.cat((uncond, composed_cond, composed_perturbed), dim=0)
            return _with_output_tensor(uncond_output, joined)

        # PAG without outer CFG receives [C,P]. Build U independently, then
        # return slots whose outer PAG formula adds the composable PAG delta to
        # the requested (possibly zero) CFG target.
        uncond_output = self._homogeneous_pag_call(sample, args, kwargs, 0, size, state.negative, 2)
        uncond = _output_tensor(uncond_output)[:size]
        cond_predictions = []
        perturbed_predictions = []
        template = uncond_output
        for term in state.positive:
            output = self._forward_static_pag_pair(sample, args, kwargs, term.branch)
            template = output
            tensor = _output_tensor(output)
            cond_predictions.append(tensor[:size])
            perturbed_predictions.append(tensor[size:])
        target = _compose(uncond, cond_predictions, weights, scale=self.guidance_scale)
        pag_delta = torch.zeros_like(target)
        for cond, perturbed, weight in zip(cond_predictions, perturbed_predictions, weights):
            pag_delta = pag_delta + (cond - perturbed) * float(weight)
        perturbed_target = target - pag_delta
        if not torch.isfinite(perturbed_target).all():
            raise PromptLeaseError("AND PAG prediction composition produced NaN or infinity")
        return _with_output_tensor(template, torch.cat((target, perturbed_target), dim=0))

    def _forward_static_pag_pair(self, sample, args, kwargs, positive):
        total = int(sample.shape[0])
        branch_args = list(args)
        encoder_positional = len(branch_args) >= 2 and "encoder_hidden_states" not in kwargs
        if encoder_positional:
            branch_args[1] = _repeat(positive.cross_attention, total, like=args[1])
        branch_kwargs = self._branch_kwargs(
            kwargs, 0, total, total, positive, total, encoder_positional=encoder_positional
        )
        return self.original_forward(sample, *branch_args, **branch_kwargs)

    def __exit__(self, _type, _value, _traceback):
        try:
            self._remove_sentinel()
            if self.unet.forward is self.wrapper:
                self.unet.forward = self.serial_wrapper or self.original_forward
            if getattr(self.pipeline, "_xiriai_prompt_lease_token", None) == self.token:
                delattr(self.pipeline, "_xiriai_prompt_lease_token")
        finally:
            self.lock.release()
        return False


@contextmanager
def static_conditioning_lease(pipeline, prepared, guidance_scale, pag_scale=0.0):
    if not isinstance(prepared, PreparedConditioning) or prepared.plan is None:
        yield None
        return
    with StaticConditioningLease(pipeline, prepared, guidance_scale, pag_scale) as lease:
        yield lease


__all__ = [
    "ConditioningSyntaxPlan", "EncodedConditionState", "EncodedPromptPlan", "MAX_AND_TERMS",
    "MAX_DYNAMIC_STATES", "PromptLeaseError", "PromptPlanLimitError", "PromptSyntaxPlan", "PromptTimeline",
    "StaticConditioningLease", "TermTimeline", "calls_per_step", "compile_conditioning_syntax",
    "prepare_dynamic_prompt_conditioning", "prompt_plan_audit", "sampling_timeline",
    "static_conditioning_lease", "static_prompt_plan",
]
