"""Separate XYZ planning contract. Does not import inference or expose a route."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, StrictInt

Engine = Literal['SD', 'iL', 'Anima', 'Flux', 'Flux2', 'Krea2']


class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class LoraReference(Contract):
    # P10's actual field names, not an alternative instance/identity scheme.
    value: str
    weight: str | float | int = 1
    precision: StrictInt = 4
    enabled: bool = True
    identitySchema: StrictInt | None = None
    instanceId: str | None = None
    contentHash: str | None = None
    contentRevision: StrictInt | None = None
    originalReference: dict[str, Any] | None = None
    name: str | None = None
    category: str | None = None
    groupId: str | None = None
    missing: bool = False


class GenerationParameters(Contract):
    engine: Engine
    checkpoint: str | None = None
    diffusion_model: str | None = None
    text_encoder: str | None = None
    text_encoder_2: str | None = None
    vae: str | None = None
    prompt: str
    negative_prompt: str = ''
    clip_skip: StrictInt | None = None
    width: StrictInt
    height: StrictInt
    steps: StrictInt
    cfg: str | float | int
    denoise: str | float | int = 1
    seed: str  # JSON never rounds uint64 through a double.
    sampler: str
    scheduler: str
    guidance: str = 'none'
    pag: dict[str, Any] = Field(default_factory=lambda: {'scale': 0.3, 'applied_layers': 'mid'})
    output_template: str | None = None
    background_removal_model: str | None = None
    postprocess_chain: list[dict[str, Any]] = Field(default_factory=list)


class Baseline(Contract):
    parameters: GenerationParameters
    seed_mode: Literal['fixed', 'random'] = 'fixed'
    loras: list[LoraReference] = Field(default_factory=list)
    # Already composed, WITHOUT the group's prefix. Include-preset-text substitutes
    # that prefix once; default planning uses parameters.prompt byte-for-byte.
    prompt_without_groups: str
    group_prompt: str = ''
    group_snapshots: list[dict[str, Any]] = Field(default_factory=list)
    library_revision: StrictInt = 0


class AxisValue(Contract):
    value: Any
    label: str = ''


class DecimalRange(Contract):
    start: str
    end: str
    step: str


class Axis(Contract):
    field: str
    values: list[AxisValue] = Field(default_factory=list)
    range: DecimalRange | None = None
    target: str | None = None  # exact instanceId for weight/model candidates
    snapshot: dict[str, Any] | None = None  # combination multiplier's fixed profile


class XYZExperimentInput(Contract):
    schema_version: Literal[1] = 1
    kind: Literal['xyz_draft'] = 'xyz_draft'
    engine: Engine
    baseline: Baseline
    axes: dict[Literal['x', 'y', 'z'], Axis | None] = Field(default_factory=dict)
    repeat_count: StrictInt = 1
    sample_seeds: list[str] | None = None
    requested_concurrency: StrictInt = 1
    inherit_postprocess: bool = False
    # Phase C: execution order only (fewer model/LoRA switches); coordinates never change.
    group_by_resources: bool = False


class XYZSubmitInput(Contract):
    """Phase B submission envelope. The browser chooses `submission_id` once per
    click, so a replayed POST resolves to the same experiment instead of a second."""
    schema_version: Literal[1] = 1
    kind: Literal['xyz_submit'] = 'xyz_submit'
    submission_id: str
    draft: XYZExperimentInput
    on_error: Literal['pause', 'continue'] = 'pause'


class XYZContinueInput(Contract):
    """Explicit continuation of an interrupted/stopped experiment. Completed
    originals are never requeued; failed cells only when named or `failed`."""
    retry: Literal['none', 'failed'] | list[str] = 'none'
    # Applies to this explicitly created continuation only; frozen requests never change.
    requested_concurrency: StrictInt | None = None


class XYZExportPage(Contract):
    z_index: StrictInt = 0
    sample_index: StrictInt = 0


class XYZExportInput(Contract):
    """Phase D whole-grid export. `original` places sources 1:1 (percent must be 100);
    `scaled` resizes each source once to percent (1–100, integer; 100 = no filtering)."""
    scope: Literal['all', 'page'] = 'all'
    page: XYZExportPage | None = None
    mode: Literal['original', 'scaled'] = 'original'
    percent: StrictInt = 100
    theme: Literal['light', 'dark'] = 'light'
    arrangement: Literal['vertical', 'horizontal', 'grid'] = 'vertical'
    captions: bool = True


class XYZDecisionInput(Contract):
    action: Literal['retry', 'skip', 'stop']


class Coordinate(Contract):
    x_index: int
    y_index: int
    z_index: int
    sample_index: int


class PlanIssue(Contract):
    code: str
    message: str
    coordinate: Coordinate | None = None
    axis: str | None = None


class PlannedCell(Contract):
    cell_id: str
    coordinate: Coordinate
    labels: dict[str, str]
    request: dict[str, Any]
    lora_references: list[dict[str, Any]]
    seed_status: Literal['resolved', 'unresolved']
    replacement_matches: dict[str, int] = Field(default_factory=dict)


class XYZPlan(Contract):
    schema_version: Literal[1] = 1
    kind: Literal['xyz_plan'] = 'xyz_plan'
    engine: Engine
    total_cells: int
    dimensions: dict[str, int]
    cells: list[PlannedCell]
    errors: list[PlanIssue]
    seed_status: Literal['resolved', 'unresolved']
    requested_concurrency: int
    executor_ready: Literal[False] = False
    static_only: Literal[True] = True
    seed_policy: str = 'Seed axis wins for every repeat (identical seed); otherwise one sample seed list shared by all XYZ combinations. No offsets.'
