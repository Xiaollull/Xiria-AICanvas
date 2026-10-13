"""Private Node IPC projection. Reuses the planner; never admits model paths or draws seeds."""
from .xyz_schema import XYZExperimentInput
from .xyz_planner import CAPABILITIES, compile_xyz


def effective_references(payload, registry):
    draft = XYZExperimentInput.model_validate(payload)
    fields = CAPABILITIES['engines'][draft.engine]['model_fields']
    parameters = draft.baseline.parameters.model_dump()
    # Declared catalog is ONLY for projection; the HTTP/execution owners still validate real models.
    catalog = {(draft.engine, field, parameters[field]) for field in fields if isinstance(parameters.get(field), str)}
    for axis in draft.axes.values():
        if axis is None:
            continue
        for entry in axis.values:
            if axis.field in fields and isinstance(entry.value, str):
                catalog.add((draft.engine, axis.field, entry.value))
            elif axis.field == 'components' and isinstance(entry.value, dict):
                catalog.update((draft.engine, field, entry.value[field]) for field in fields if isinstance(entry.value.get(field), str))
    plan = compile_xyz(draft, model_catalog=catalog, lora_registry=registry)
    if plan.errors:
        return {'errors': [item.model_dump(mode='json') for item in plan.errors[:200]], 'error_count': len(plan.errors)}
    references = {}
    for cell in plan.cells:
        for item in cell.lora_references:
            # Include the complete frozen identity, not just the instance: conflicting snapshots
            # must never be deduplicated into an apparently valid first reference.
            key = tuple(item.get(field) for field in ('value', 'instanceId', 'contentHash', 'contentRevision', 'identitySchema'))
            references.setdefault(key, item)
    if len(references) > 256:
        return {'errors': [{'code': 'lora-limit', 'message': 'Too many XYZ LoRA references'}], 'error_count': 1}
    return {'references': list(references.values()), 'errors': [], 'error_count': 0}
