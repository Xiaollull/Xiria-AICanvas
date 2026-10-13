import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import zipfile

from PIL import Image
from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend import wd14, tagger_adapters as adapters, tagger_store as store, tagger_runtime_install as layer
from backend.tagger_registry import Artifact, ENGINES, MODELS, validate_selection, resource_budget


class TaggerTests(unittest.TestCase):
    def test_selection_legacy_defaults_and_category_snapshots(self):
        self.assertEqual(wd14.Options().engine, "wd")
        self.assertEqual(wd14.Options().device, "cpu")
        self.assertEqual(wd14._effective(wd14.Options())["general"], .35)
        for engine in ("cl", "camie", "pixai"):
            params = wd14._effective(wd14.Options(engine=engine, model=ENGINES[engine]["default"]))
            self.assertEqual(params["general"], ENGINES[engine]["defaults"]["general"])
        for args in [("wd","gpu","pixai-tagger-v1.0"),("pixai","onnx","pixai-tagger-v1.0"),("evil","cpu",wd14.DEFAULT)]:
            with self.assertRaises(ValueError): validate_selection(*args, wd14.REGISTRY)
        self.assertEqual(len(wd14.REGISTRY),16)
        self.assertEqual(resource_budget("pixai-tagger-v1.0")["required_mib"],4416)

    def test_decoder_preserves_non_general_categories_and_rating(self):
        params=adapters.parameters("pixai","pixai-tagger-v1.0","cpu",limit=50)
        result=adapters.decode([.9,.8,.4,.3,.6],[("artist_name","artist"),("style_name","style"),("series","copyright"),("hero","character"),("rating:g","rating")],params)
        self.assertEqual([row["category"] for row in result["tags"]],["artist","style","copyright","character"])
        self.assertEqual(result["rating"],{"general":.6})
        for invalid in ([float("nan")]*5,[2]*5,[.5]*4):
            with self.assertRaises(ValueError): adapters.decode(invalid,[("x","general")]*5,params)

    def test_camie_rgb_normalization_padding_and_pixai_white_alpha_float_pad(self):
        import numpy as np
        data=adapters.camie_input(Image.new("RGB",(512,256),(255,0,0)))
        self.assertEqual(data.shape,(1,3,512,512))
        np.testing.assert_allclose(data[0,:,0,0],(np.array([124,116,104],np.float32)/255-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32))
        np.testing.assert_allclose(data[0,:,256,256],(np.array([1,0,0],np.float32)-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32))
        pix=adapters.pixai_input(Image.new("RGBA",(1008,504),(0,0,0,0)))
        self.assertEqual(tuple(pix.shape),(1,3,1008,1008))
        self.assertTrue((pix[0,:,0,0]==-1).all())
        self.assertTrue((pix[0,:,500,500]==1).all())

    def test_complete_manifest_hashes_marker_and_tamper_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); folder=root/"camie-tagger-v2";folder.mkdir()
            artifacts=[]
            for name,content in [("metadata.json",b"{}"),("model.onnx",b"\x08\x01")]:
                (folder/name).write_bytes(content)
                artifacts.append(Artifact(name,name,len(content),hashlib.sha256(content).hexdigest()))
            spec={**MODELS["camie-tagger-v2"],"artifacts":tuple(artifacts)}
            with patch.dict(MODELS,{"camie-tagger-v2":spec}):
                self.assertTrue(store.installed(root,"camie-tagger-v2",lambda p:p))
                (folder/".wd14-installing").write_text("user evidence")
                self.assertFalse(store.installed(root,"camie-tagger-v2",lambda p:p))
                (folder/".wd14-installing").unlink()
                (folder/"model.onnx").write_bytes(b"\x08\x02")
                self.assertFalse(store.installed(root,"camie-tagger-v2",lambda p:p))

    def test_three_artifact_publish_rollback_unknown_files_and_cancel(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(wd14,"ROOT",Path(directory)):
            folder=Path(directory)/"pixai-tagger-v1.0";folder.mkdir()
            stage=folder/".wd14-stage-ab12_cd3";stage.mkdir()
            targets={name:folder/name for name in ["config.json","preprocessor_config.json","model.safetensors"]}
            temporary={name:stage/(name+".part") for name in targets}
            job={"id":"a"*32,"cancel":threading.Event(),"state":"running","publish_started":False,"logs":[]}
            (folder/"user-note.json").write_text("keep")
            for name in targets:
                targets[name].write_bytes(b"old");temporary[name].write_bytes(b"new");wd14._record_stage_file(job,temporary[name])
            original=wd14.os.replace
            def failure(source,target):
                if Path(source)==temporary["model.safetensors"]: raise OSError("publish failure")
                return original(source,target)
            with patch.object(wd14.os,"replace",side_effect=failure):
                with self.assertRaises(OSError):wd14._publish_artifacts(job,targets,temporary)
            for path in targets.values():self.assertEqual(path.read_bytes(),b"old")
            self.assertFalse((folder/".wd14-installing").exists())
            self.assertEqual((folder/"user-note.json").read_text(),"keep")
            job["cancel"].set()
            self.assertFalse(wd14._publish_artifacts(job,targets,temporary))

    def test_fixed_runtime_wheel_receipt_cannot_authorize_modified_or_extra_python(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/layer.LAYER_ID;folder.mkdir();site=folder/"site-packages";site.mkdir()
            wheel=folder/layer.WHEEL_NAME
            with zipfile.ZipFile(wheel,"w") as archive:archive.writestr("onnxruntime/__init__.py",b"SAFE = True\n")
            (site/"onnxruntime").mkdir();(site/"onnxruntime/__init__.py").write_bytes(b"SAFE = True\n")
            receipt={"schema":1,"version":layer.VERSION,"python_abi":"cp312","platform":"win_amd64","cuda_major":13,"wheel_sha256":hashlib.sha256(wheel.read_bytes()).hexdigest(),"python":str(Path(layer.sys.executable).resolve())}
            (folder/"receipt.json").write_text(json.dumps(receipt))
            with patch.object(layer,"ROOT",root), patch.object(layer,"supported",return_value=True), patch.object(layer,"WHEEL_SIZE",wheel.stat().st_size), patch.object(layer,"WHEEL_SHA",receipt["wheel_sha256"]):
                self.assertEqual(layer.verify_layer(),site)
                (site/"pwn.py").write_text("raise RuntimeError('must not import')")
                with self.assertRaisesRegex(ValueError,"未知文件"):layer.verify_layer()
                (site/"pwn.py").unlink()
                (site/"onnxruntime/__init__.py").write_bytes(b"EVIL = True\n")
                with self.assertRaises(ValueError):layer.verify_layer()

    def test_runtime_http_explicit_confirmation_unknown_arguments_remote_and_engine_mismatch(self):
        app=FastAPI();app.include_router(wd14.router)
        with TestClient(app,base_url="http://127.0.0.1:19001",client=("127.0.0.1",1234)) as client:
            headers={"Origin":"http://127.0.0.1:19001"}
            for payload in [{"confirm":False},{"confirm":True,"url":"https://evil.example"},{"confirm":True,"pipargs":["evil"]}]:
                self.assertEqual(client.post("/api/inference/wd14/runtime/install",json=payload,headers=headers).status_code,422)
            self.assertEqual(client.post("/api/inference/wd14/runtime/install",json={"confirm":True},headers={"Origin":"https://evil.example"}).status_code,403)
            self.assertEqual(client.post("/api/inference/wd14/tag?engine=wd&model=pixai-tagger-v1.0",content=b"x",headers=headers).status_code,422)
            self.assertEqual(client.post("/api/inference/wd14/download/unknown",headers=headers).status_code,422)

    def test_cpu_provider_and_gpu_missing_never_fallback(self):
        with patch("onnxruntime.get_available_providers",return_value=["CPUExecutionProvider"]), patch("onnxruntime.InferenceSession") as create:
            with self.assertRaisesRegex(ValueError,"不会回退 CPU"):adapters.create_onnx(Path("model.onnx"),"gpu")
            create.assert_not_called()

    def test_cl_dynamic_nchw_graph_logits_and_camie_refined_sigmoid_once(self):
        import numpy as np
        from types import SimpleNamespace
        content=io.BytesIO();Image.new("RGB",(20,10),(255,0,0)).save(content,format="PNG")
        for engine,count,size in [("cl",51213,448),("camie",70527,512)]:
            with tempfile.TemporaryDirectory() as directory:
                folder=Path(directory);(folder/"model.onnx").write_bytes(b"\x08\x01")
                rows=[("general","rating"),("sensitive","rating"),("questionable","rating"),("explicit","rating")]+[(f"tag_{i}","general") for i in range(count-4)]
                logits=np.full((1,count),-5,dtype=np.float32);logits[0,4]=5
                session=unittest.mock.Mock()
                session.get_inputs.return_value=[SimpleNamespace(name="input",shape=["batch",3,"height","width"] if engine=="cl" else ["batch",3,512,512])]
                session.get_outputs.return_value=[SimpleNamespace(name="logits",shape=["batch",count])] if engine=="cl" else [SimpleNamespace(name="initial",shape=["batch",count]),SimpleNamespace(name="refined",shape=["batch",count]),SimpleNamespace(name="candidates",shape=["batch",256])]
                session.get_providers.return_value=["CPUExecutionProvider"]
                session.run.return_value=[logits]
                with patch.object(adapters,"read_labels",return_value=rows),patch.object(adapters,"create_onnx",return_value=session):
                    params=adapters.parameters(engine,ENGINES[engine]["default"],"cpu")
                    result=adapters.infer(content.getvalue(),params,folder,"cpu")
                    self.assertEqual(len(result["tags"]),1)
                    self.assertAlmostEqual(result["rating"]["general"],.0066928509,places=7)
                    self.assertEqual(session.run.call_args.args[0],["logits" if engine=="cl" else "refined"])
                    self.assertEqual(tuple(session.run.call_args.args[1]["input"].shape),(1,3,size,size))

    def test_gpu_generation_admission_serializes_and_never_evicts_cache(self):
        from backend import inference_server as server
        from concurrent.futures import ThreadPoolExecutor
        entered,release,generated=threading.Event(),threading.Event(),threading.Event()
        sentinel=object();errors=[]
        def operation():entered.set();self.assertTrue(release.wait(5));return "tagged"
        with ThreadPoolExecutor(max_workers=1) as executor, patch.object(server,"executor",executor), patch.object(server,"jobs",{}), \
             patch.object(server,"shutdown_requested",False), patch.object(server,"loaded_pipeline",sentinel), patch.object(server,"_tagger_gpu_admission",threading.Lock()):
            def invoke():
                try:server.run_tagger_gpu(operation)
                except Exception as error:errors.append(error)
            thread=threading.Thread(target=invoke);thread.start();self.assertTrue(entered.wait(5))
            with server.jobs_lock:
                server.jobs["generation"]={"status":"queued"}
                future=executor.submit(generated.set)
            self.assertFalse(generated.wait(.05))
            with self.assertRaises(ValueError):server.run_tagger_gpu(lambda:None)
            release.set();thread.join(5);future.result(timeout=5)
            self.assertTrue(generated.is_set());self.assertFalse(errors);self.assertIs(server.loaded_pipeline,sentinel)
            for status in ("queued","running","paused","cancelling"):
                server.jobs["generation"]["status"]=status
                with self.assertRaisesRegex(ValueError,"生图"):server.run_tagger_gpu(lambda:None)

    def test_all_legacy_ids_have_fixed_official_complete_manifests(self):
        self.assertEqual(set(wd14.REGISTRY),set(wd14.WD_MODELS))
        for model in wd14.REGISTRY:
            repo,revision,files=wd14._metadata(model)
            self.assertEqual(repo,wd14.REGISTRY[model][0]);self.assertEqual(len(revision),40)
            self.assertEqual(set(files),{"model.onnx","selected_tags.csv"})
            self.assertEqual(len(files["model.onnx"][1]),64)

    def test_isolated_worker_ipc_is_utf8_on_cp936_hosts(self):
        import subprocess,sys
        command=[sys.executable,"-I","-B",str(Path(wd14.__file__).with_name("tagger_worker.py"))]
        result=subprocess.run(command,input=json.dumps({"params":{"engine":"unknown","device":"cpu","model":"unknown"}}),
                              text=True,encoding="utf-8",errors="strict",capture_output=True,timeout=30)
        self.assertEqual(result.returncode,0)
        self.assertIn("未知反推引擎或设备",json.loads(result.stdout)["error"])

    def test_worker_oom_is_readable_and_never_retries_cpu(self):
        from backend import tagger_resources as resource
        from types import SimpleNamespace
        result=SimpleNamespace(returncode=0,stdout=json.dumps({"error":"CUDA out of memory (controlled refusal fixture)"}),stderr="")
        params=adapters.parameters("wd",wd14.DEFAULT,"gpu")
        with patch("torch.cuda.is_available",return_value=True),patch("torch.cuda.mem_get_info",return_value=(8*1024**3,8*1024**3)), \
             patch.object(resource.subprocess,"run",return_value=result) as launch:
            with self.assertRaisesRegex(ValueError,"CUDA out of memory"):resource.worker(b"fixture",params,Path("fixture-model-root"))
            self.assertEqual(launch.call_count,1)
            self.assertEqual(json.loads(launch.call_args.kwargs["input"])["params"]["device"],"gpu")
        self.assertFalse(resource._worker_lock.locked())


if __name__=="__main__":unittest.main()
