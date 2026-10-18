"""Compile current-browser captured bodies using current strict XYZ + actual GenerateInput AST.
No inference import, formal listener, model files or GPU. Writes only into the supplied evidence root.
"""
import ast, builtins, copy, hashlib, json, re, sys, types, time, math
from pathlib import Path
from typing import Annotated, Any, Literal, Union
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
# GenerateInput's MOSS layout block validates through the pure layout compiler (json/math only, no inference).
from backend import moss_layout
from backend.xyz_planner import CAPABILITIES, compile_xyz, plan_record
from backend.xyz_schema import LoraReference, XYZExperimentInput
from backend.output_paths import normalize_output_template
from PIL import Image

tree=ast.parse((ROOT/'backend/inference_server.py').read_text(encoding='utf-8'));owners={}
for n in tree.body:
    if isinstance(n,(ast.ClassDef,ast.FunctionDef)):owners[n.name]=n
    elif isinstance(n,ast.Assign):
        for t in n.targets:
            if isinstance(t,ast.Name):owners[t.id]=n
required=set();pending=['GenerateInput'];unresolved=set()
while pending:
    name=pending.pop()
    if name in required:continue
    if name not in owners:unresolved.add(name);continue
    required.add(name);n=owners[name]
    local={x.id for x in ast.walk(n) if isinstance(x,ast.Name) and isinstance(x.ctx,(ast.Store,ast.Param))}|{x.arg for x in ast.walk(n) if isinstance(x,ast.arg)}
    pending.extend({x.id for x in ast.walk(n) if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Load)}-local-set(dir(builtins)))
module=types.ModuleType('_xyz_seven_current_schema');sys.modules[module.__name__]=module;env=module.__dict__;env.update({k:globals()[k] for k in unresolved if k in globals()});constants={}
def constant(name,expr):
    if name in constants:return constants[name]
    try:v=ast.literal_eval(expr[name])
    except ValueError:
        assert all(isinstance(n,(ast.BinOp,ast.Constant,ast.Mult,ast.Add,ast.Sub,ast.Pow,ast.Name,ast.Load)) for n in ast.walk(expr[name]))
        for n in ast.walk(expr[name]):
            if isinstance(n,ast.Name):constant(n.id,expr)
        v=eval(compile(ast.Expression(expr[name]),'current-constant','eval'),{'__builtins__':{}},constants)
    constants[name]=v;return v
for file in ['anima_pipeline.py','anima_sampling.py','flux_pipeline.py','flux_sampling.py','flux2_pipeline.py','flux2_sampling.py','krea2_pipeline.py','krea2_sampling.py','moss_model.py','moss_sampling.py','guidance.py','rtx_vsr.py']:
    expr={t.id:n.value for n in ast.parse((ROOT/'backend'/file).read_text(encoding='utf-8')).body if isinstance(n,ast.Assign) for t in n.targets if isinstance(t,ast.Name)}
    for name in expr.keys()&(unresolved|{'MAX_OUTPUT_EDGE','MAX_OUTPUT_PIXELS'}):constant(name,expr)
env.update(constants);env['rtx_vsr']=types.SimpleNamespace(MAX_OUTPUT_EDGE=constants['MAX_OUTPUT_EDGE'],MAX_OUTPUT_PIXELS=constants['MAX_OUTPUT_PIXELS'])
bg=ast.parse((ROOT/'backend/background_removal.py').read_text(encoding='utf-8'));nodes=[n for n in bg.body if isinstance(n,ast.FunctionDef) and n.name=='parse_prompt_directives']
for n in bg.body:
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {v.id for v in ast.walk(nodes[0]) if isinstance(v,ast.Name)} for t in n.targets):nodes.insert(0,n)
exec(compile(ast.Module(body=nodes,type_ignores=[]),'current-directives','exec'),env)
nodes=sorted({id(owners[n]):owners[n] for n in required}.values(),key=lambda n:n.lineno)
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[])),'current-schema','exec'),env)
GenerateInput=env['GenerateInput'];GenerateInput.model_rebuild()
folder=Path(sys.argv[1]);out=folder/f'current-compiler-{int(time.time()*1000)}';out.mkdir()
weight_inputs=json.loads((folder/'weight-multiaxis-wire.json').read_text(encoding='utf-8'))['cases'] if '--weight-multiaxis' in sys.argv else []
weight_cases={c['name']:c for c in weight_inputs}
assert len(weight_cases)==len(weight_inputs),'Evidence case names must be unique; never silently drop numeric negatives'
cases={name:c['wire'] for name,c in weight_cases.items()} if weight_cases else {case['name']:case['wire'] for case in json.loads((folder/'numeric-positive-wire.json').read_text(encoding='utf-8'))} if '--numeric-positive' in sys.argv else json.loads((folder/'captured-wire.json').read_text(encoding='utf-8'))
results=[];plans={}
def references(v):
    found=[]
    if isinstance(v,dict):
        if 'instanceId' in v and 'contentHash' in v and 'value' in v:found.append(v)
        for child in v.values():found.extend(references(child))
    elif isinstance(v,list):
        for child in v:found.extend(references(child))
    return found
for name,draft in cases.items():
    before=copy.deepcopy(draft);XYZExperimentInput.model_validate(draft)
    assert 'seed_source' not in draft['baseline'];assert all(not a or 'editor_source' not in a for a in draft['axes'].values())
    p=draft['baseline']['parameters'];catalog={(draft['engine'],k,p[k]) for k in CAPABILITIES['engines'][draft['engine']]['model_fields']}
    registry=weight_cases[name]['registry'] if weight_cases else {'schemaVersion':1,'instances':{m['instanceId']:{**m,'engine':draft['engine'],'status':'present'} for m in references(draft)}}
    captures=[];draws=[]
    def rng():
        value=9000000000000000000+len(draws);draws.append(str(value));return value
    def validate(request):captures.append(GenerateInput.model_validate(request).model_dump(mode='json'))
    static=compile_xyz(draft,model_catalog=catalog,lora_registry=registry)
    plan=compile_xyz(draft,model_catalog=catalog,lora_registry=registry,resolve_seeds=True,random_uint64=rng,validate_request=validate)
    expected=weight_cases[name]['expected'] if weight_cases else {}
    if expected.get('error'):
        assert static.errors and plan.errors and all(e.code==expected['error'] for e in [*static.errors,*plan.errors]),plan.errors
        assert not captures and not draws and draft==before
        if expected.get('partialCells') is not None:assert len(plan.cells)==expected['partialCells']
        else:assert not plan.cells
        (out/f'{name}.json').write_text(json.dumps({'plan':plan_record(plan),'strictCaptures':captures,'draws':draws},ensure_ascii=False,indent=2),encoding='utf-8')
        results.append({'case':name,'expectedError':expected['error'],'errors':len(plan.errors),'cells':len(plan.cells),'strictCalls':0,'randomDraws':0});continue
    assert not static.errors,static.errors
    assert not plan.errors,plan.errors
    assert draft==before;assert all((c.request['width'],c.request['height'])==(p['width'],p['height']) for c in plan.cells)
    for sample in range(draft['repeat_count']):
        if not any(a and a['field']=='seed' for a in draft['axes'].values()):assert len({c.request['seed'] for c in plan.cells if c.coordinate.sample_index==sample})==1
    if name=='custom-steps':assert [c.request['steps'] for c in plan.cells]==[15,23,28]
    if name=='range-steps':assert [c.request['steps'] for c in plan.cells]==[12,14,16,18,20]
    if name=='range-cfg':assert sorted({c.request['cfg'] for c in plan.cells})==[4,4.5,5,5.5,6,6.5];assert plan.total_cells==18
    if name.startswith('seed-'):
        if name=='seed-全随机':assert len(draws)==3
        else:assert [str(c.request['seed']) for c in plan.cells]==draft['sample_seeds'];assert not draws
    if name in {'lora-add','external-weight','external-multiplier'}:
        for c in plan.cells:
            assert len(c.lora_references)==6
            assert [(m['instanceId'],m['weight']) for m in c.lora_references[:5]]==[(m['instanceId'],float(m['weight'])) for m in draft['baseline']['loras'][:5]]
            assert c.request['prompt']==p['prompt']
    if name=='lora-replace':assert len(plan.cells[0].lora_references)==5;assert [m['instanceId'] for m in plan.cells[0].lora_references[1:]]==[m['instanceId'] for m in draft['baseline']['loras'][1:]]
    if name=='profile-preset':assert plan.cells[0].request['prompt'].count('group preset')==1
    if name=='profile-no-loras':assert not plan.cells[0].lora_references
    if weight_cases:
        from backend.xyz_planner import expand_axis
        from decimal import Decimal
        assert plan.total_cells==expected['cells']==len(plan.cells)
        ui=weight_cases[name]['uiBaseline']
        assert draft['baseline']['loras'][:len(ui['loras'])]==ui['loras']
        assert draft['baseline']['group_prompt']==ui['group_prompt'] and draft['baseline']['group_snapshots']==ui['group_snapshots']
        parsed=XYZExperimentInput.model_validate(draft)
        axes={key:expand_axis(axis) for key,axis in parsed.axes.items() if axis}
        for key,values in expected.get('axisValues',{}).items():assert [float(Decimal(str(e.value))) for e in axes[key]]==values
        if weight_cases[name].get('submitDTO'):assert weight_cases[name]['submitDTO']['draft']==draft
        coordinates=set()
        for cell in plan.cells:
            coordinates.add((cell.coordinate.x_index,cell.coordinate.y_index,cell.coordinate.z_index,cell.coordinate.sample_index))
            members=[LoraReference.model_validate(m).model_dump(exclude_none=True) for m in draft['baseline']['loras']]
            numeric={'steps':p['steps'],'cfg':p['cfg']}
            for key in ['x','y','z']:
                axis=parsed.axes[key]
                if not axis:continue
                assert axis.field in {'lora_weight','steps','cfg'}
                index=getattr(cell.coordinate,key+'_index');value=expected.get('axisValues',{}).get(key,[e.value for e in axes[key]])[index]
                if axis.field=='lora_weight':
                    member=next(m for m in members if m['instanceId']==axis.target);member['weight']=float(Decimal(str(value)));member['enabled']=True
                else:numeric[axis.field]=float(Decimal(str(value)))
            active=[]
            for member in members:
                if member['enabled'] and Decimal(str(member['weight']))!=0:
                    member['weight']=float(Decimal(str(member['weight'])));active.append(member)
            assert cell.lora_references==active,(name,cell.cell_id,cell.lora_references,active)
            assert cell.request['loras']==[{'path':m['value'],'weight':m['weight']} for m in active]
            assert type(cell.request['steps']) is int and cell.request['steps']==numeric['steps']
            assert cell.request['cfg']==numeric['cfg']
            assert cell.request['prompt']==p['prompt'] and cell.request['negative_prompt']==p['negative_prompt']
        import itertools
        assert coordinates==set(itertools.product(*(range(len(axes[k])) if k in axes else range(1) for k in ['x','y','z']),range(draft['repeat_count'])))
        if expected.get('axisValues'):
            weights=sorted({float(Decimal(str(v))) for k,values in expected['axisValues'].items() if parsed.axes[k].field=='lora_weight' for v in values if Decimal(str(v))!=0})
        else:weights=expected.get('weights',[0.2,0.5,0.8])
        assert sorted({m['weight'] for c in plan.cells for m in c.lora_references if m['instanceId'] in {a.target for a in parsed.axes.values() if a and a.field=='lora_weight'}})==weights
    record=plan_record(plan);plans[name]=record;(out/f'{name}.json').write_text(json.dumps({'plan':record,'strictCaptures':captures,'draws':draws},ensure_ascii=False,indent=2),encoding='utf-8')
    results.append({'case':name,'cells':plan.total_cells,'strictCalls':len(captures),'randomDraws':len(draws)})
assert 'torch' not in sys.modules
report={'status':'PASS','cases':results,'sourceSchemaHash':hashlib.sha256((ROOT/'backend/inference_server.py').read_bytes()).hexdigest(),'sourceSHA':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['src/xyz-editor-source.js','src/xyz-local-validation.js','src/xyz-draft.js','src/xyz-prompt-source.js','backend/xyz_planner.py','backend/xyz_schema.py','scripts/xyz-seven.compile.py','scripts/xyz-weight-multiaxis.browser-checks.mjs','scripts/xyz-full-ui-stage2.fixture.jsx','scripts/xyz-full-ui-stage2.browser.mjs','scripts/xyz-weight-multiaxis.test.mjs']},'torchImported':False};(out/'RESULT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({'out':str(out),**report},ensure_ascii=False))
