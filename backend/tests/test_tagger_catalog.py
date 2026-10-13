import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import zipfile

from backend.tagger_catalog_cache import DisplayCache
from backend import tagger_runtime_install as layer, tagger_store as store, tagger_resources as resource, wd14
from backend.tagger_registry import Artifact, WD_MODELS, MODELS
from fastapi import FastAPI
from fastapi.testclient import TestClient


def ready(check, timeout=3):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        result = check()
        if not result.get("pending"):
            return result
        time.sleep(.01)
    raise AssertionError("display verification did not settle")


class CatalogTests(unittest.TestCase):
    def test_force_write_gate_before_any_display_or_validator(self):
        app=FastAPI();app.include_router(wd14.router)
        with patch.object(resource,"snapshot") as sample,patch.object(layer,"display_status") as runtime,patch.object(store,"display_status") as model:
            for peer,host,origin,proof in [("10.0.0.42","10.0.0.1","http://10.0.0.1:19109",""),
                                            ("127.0.0.1","127.0.0.1","http://evil.example:19109",""),
                                            ("127.0.0.1","127.0.0.1",None,""),
                                            ("127.0.0.1","127.0.0.1","http://10.0.0.1:19109","trusted-proxy-fixture")]:
                with patch.dict(wd14.os.environ,{"XIRAI_WD14_PROXY_PROOF":"trusted-proxy-fixture"}),TestClient(app,base_url=f"http://{host}:19109",client=(peer,12345)) as client:
                    headers={"Origin":origin} if origin else {}
                    if proof:headers["x-xirai-wd14-proxy-proof"]=proof
                    for value in ('true','1','on','yes'):
                        self.assertEqual(client.get(f"/api/inference/wd14/models?refresh={value}",headers=headers).status_code,403)
                        self.assertEqual(client.post(f"/api/inference/wd14/models?refresh={value}",headers=headers).status_code,403)
                    self.assertEqual(client.get("/api/inference/wd14/models?refresh=false&refresh=true",headers=headers).status_code,422)
            sample.assert_not_called();runtime.assert_not_called();model.assert_not_called()

    def test_ordinary_remote_catalog_and_local_force_contract(self):
        app=FastAPI();app.include_router(wd14.router)
        sample={"available":False,"reason":"fixture","free":0,"total":0,"cuda_major":"13","busy":False}
        with patch.object(resource,"snapshot",return_value=sample),patch.object(layer,"display_status",return_value={"installed":False,"pending":True,"reason":"pending"}) as runtime, \
             patch.object(store,"display_status",return_value={"installed":False,"verification":"pending","pending":True}):
            with TestClient(app,base_url="http://10.0.0.1:19109",client=("10.0.0.42",12345)) as client:
                r=client.get("/api/inference/wd14/models");self.assertEqual(r.status_code,200);self.assertEqual(len(r.json()["models"]),19)
                self.assertEqual(runtime.call_args.kwargs,{"refresh":False})
            with TestClient(app,base_url="http://127.0.0.1:19109",client=("127.0.0.1",12345)) as client:
                for method in (client.get,client.post):
                    r=method("/api/inference/wd14/models?refresh=true",headers={"Origin":"http://127.0.0.1:19109"})
                    self.assertEqual(r.status_code,200);self.assertEqual(runtime.call_args.kwargs,{"refresh":True})
    def test_singleflight_pending_generation_refresh_ttl_and_bounded_slots(self):
        now=[0.0];cache=DisplayCache(ttl=5,max_entries=2,workers=1,clock=lambda:now[0])
        self.addCleanup(cache.executor.shutdown,wait=True)
        entered,release=threading.Event(),threading.Event();calls=[]
        def verify():calls.append(1);entered.set();release.wait(2);return {"installed":True,"reason":""}
        self.assertFalse(cache.get("one",(1,),verify)["installed"]);self.assertTrue(entered.wait(1))
        for _ in range(10):self.assertTrue(cache.get("one",(1,),verify,force=True)["pending"])
        self.assertEqual(len(calls),1)
        # Stale successful completion cannot mark a newer fingerprint installed.
        self.assertTrue(cache.get("one",(2,),lambda:{"installed":False,"reason":"changed"})["pending"])
        release.set();result=ready(lambda:cache.get("one",(2,),lambda:{"installed":False,"reason":"changed"}))
        self.assertFalse(result["installed"])
        result=ready(lambda:cache.get("one",(2,),lambda:{"installed":True},force=False));self.assertFalse(result["installed"])
        cache.get("one",(2,),lambda:{"installed":True},force=True)
        self.assertTrue(ready(lambda:cache.get("one",(2,),lambda:{"installed":True}))["installed"])
        now[0]=6;self.assertTrue(cache.get("one",(2,),lambda:{"installed":False})["pending"])
        self.assertFalse(ready(lambda:cache.get("one",(2,),lambda:{"installed":False}))["installed"])

    def test_model_digest_force_recheck_and_marker_delete_same_size_mutation(self):
        cache=DisplayCache();self.addCleanup(cache.executor.shutdown,wait=True)
        with tempfile.TemporaryDirectory() as directory,patch.object(store,"DISPLAY_CACHE",cache):
            root=Path(directory);folder=root/"test";folder.mkdir();path=folder/"model.onnx";path.write_bytes(b"good")
            spec={"revision":"a"*40,"artifacts":(Artifact("model.onnx","model.onnx",4,hashlib.sha256(b"good").hexdigest()),)}
            check=lambda:store.display_status(root,"test",lambda p:p,spec)
            self.assertTrue(ready(check)["installed"])
            path.write_bytes(b"evil")
            self.assertFalse(check()["installed"]);self.assertFalse(ready(check)["installed"])
            path.write_bytes(b"good");self.assertTrue(ready(check)["installed"])
            (folder/".wd14-installing").write_text("user marker")
            self.assertFalse(check()["installed"]);self.assertEqual(check()["verification"],"unavailable")
            (folder/".wd14-installing").unlink();path.unlink();self.assertFalse(check()["installed"])
            path.write_bytes(b"good");self.assertTrue(ready(check)["installed"])
            stamp=path.stat().st_mtime_ns
            path.write_bytes(b"evil")
            # Explicit force does not rely on a caller's supplied receipt or old digest hint.
            store.display_status(root,"test",lambda p:p,spec,refresh=True)
            self.assertFalse(ready(check)["installed"])
            self.assertFalse(store.installed(root,"test",lambda p:p,spec))

    def test_runtime_cached_display_is_not_worker_authorization(self):
        cache=DisplayCache();self.addCleanup(cache.executor.shutdown,wait=True)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/layer.LAYER_ID;site=folder/"site-packages";site.mkdir(parents=True)
            wheel=folder/layer.WHEEL_NAME
            content=b"SAFE = True\n"
            with zipfile.ZipFile(wheel,"w") as archive:archive.writestr("onnxruntime/__init__.py",content)
            (site/"onnxruntime").mkdir();module=site/"onnxruntime/__init__.py";module.write_bytes(content)
            sha=hashlib.sha256(wheel.read_bytes()).hexdigest()
            receipt={"schema":1,"version":layer.VERSION,"python_abi":"cp312","platform":"win_amd64","cuda_major":13,"wheel_sha256":sha,"python":str(Path(layer.sys.executable).resolve())}
            (folder/"receipt.json").write_text(json.dumps(receipt))
            with patch.object(layer,"ROOT",root),patch.object(layer,"supported",return_value=True),patch.object(layer,"WHEEL_SHA",sha),patch.object(layer,"WHEEL_SIZE",wheel.stat().st_size),patch.object(layer,"DISPLAY_CACHE",cache):
                self.assertTrue(ready(layer.display_status)["installed"])
                # Even a deliberately stale positive DISPLAY result cannot bypass full import gate.
                old_fingerprint=layer._display_fingerprint()
                module.write_bytes(b"EVIL = True\n")
                with patch.object(layer,"_display_fingerprint",return_value=old_fingerprint):
                    self.assertTrue(layer.display_status()["installed"])
                    with self.assertRaises(ValueError):layer.verify_layer()
                self.assertFalse(ready(layer.display_status)["installed"])
                module.write_bytes(content);self.assertTrue(ready(layer.display_status)["installed"])
                (site/"unexpected.py").write_text("raise Exception('never import')")
                self.assertFalse(ready(layer.display_status)["installed"])
                (site/"unexpected.py").unlink();self.assertTrue(ready(layer.display_status)["installed"])
                receipt["wheel_sha256"]="0"*64;(folder/"receipt.json").write_text(json.dumps(receipt))
                self.assertFalse(ready(layer.display_status)["installed"])
                receipt["wheel_sha256"]=sha;(folder/"receipt.json").write_text(json.dumps(receipt));self.assertTrue(ready(layer.display_status)["installed"])
                wheel.write_bytes(b"x"*wheel.stat().st_size);self.assertFalse(ready(layer.display_status)["installed"])

    def test_model_link_and_runtime_link_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);target=root/"outside";target.write_bytes(b"good")
            folder=root/"model";folder.mkdir();link=folder/"model.onnx"
            try:link.symlink_to(target)
            except OSError:self.skipTest("Windows symlink privilege unavailable")
            spec={"revision":"a"*40,"artifacts":(Artifact("model.onnx","model.onnx",4,hashlib.sha256(b"good").hexdigest()),)}
            self.assertFalse(store.display_status(root,"model",lambda p:p,spec)["installed"])
            cache=DisplayCache();self.addCleanup(cache.executor.shutdown,wait=True)
            layer_folder=root/layer.LAYER_ID;layer_folder.mkdir();(layer_folder/"site-packages").mkdir()
            (layer_folder/layer.WHEEL_NAME).symlink_to(target)
            (layer_folder/"receipt.json").write_text("{}")
            with patch.object(layer,"ROOT",root),patch.object(layer,"supported",return_value=True),patch.object(layer,"WHEEL_SIZE",4),patch.object(layer,"DISPLAY_CACHE",cache):
                self.assertFalse(layer.display_status()["installed"])

    def test_publish_after_cached_missing_is_immediately_rechecked(self):
        cache=DisplayCache();self.addCleanup(cache.executor.shutdown,wait=True)
        with tempfile.TemporaryDirectory() as directory,patch.object(store,"DISPLAY_CACHE",cache),patch.object(wd14,"ROOT",Path(directory)):
            root=Path(directory);folder=root/"model";folder.mkdir()
            spec={"revision":"a"*40,"artifacts":(Artifact("model.onnx","model.onnx",4,hashlib.sha256(b"good").hexdigest()),)}
            check=lambda:store.display_status(root,"model",wd14._safe_directory,spec)
            self.assertFalse(check()["installed"])
            stage=folder/".wd14-stage-ab12_cd3";stage.mkdir();part=stage/"model.onnx.part";part.write_bytes(b"good")
            job={"id":"a"*32,"cancel":threading.Event(),"state":"running","publish_started":False,"logs":[]}
            wd14._record_stage_file(job,part)
            self.assertTrue(wd14._publish_artifacts(job,{"model.onnx":folder/"model.onnx"},{"model.onnx":part}))
            self.assertTrue(ready(check)["installed"])

    def test_catalog_fixed_offline_and_fresh_resource_sample_once(self):
        empty_layer={"installed":False,"pending":True,"reason":"verifying"}
        sample={"available":True,"reason":"","free":8*1024**3,"total":8*1024**3,"cuda_major":"13","busy":False}
        with patch.object(resource,"snapshot",return_value=sample) as sampled,patch.object(layer,"display_status",return_value=empty_layer) as checked, \
             patch.object(store,"display_status",return_value={"installed":False,"verification":"pending","pending":True,"reason":"verifying"}), \
             patch("requests.Session.request",side_effect=AssertionError("no online catalog")):
            result=wd14.models();self.assertEqual(len(result["models"]),19);self.assertEqual(len(result["engines"]),4)
            sampled.assert_called_once();checked.assert_called_once()
            self.assertEqual({row["id"] for row in result["models"]},set(WD_MODELS)|set(MODELS))
        valid={"installed":True,"reason":""}
        with patch.object(resource,"GPU_BUSY",return_value=False),patch("torch.cuda.is_available",return_value=True),patch("torch.cuda.mem_get_info",return_value=(8*1024**3,8*1024**3)):
            first=resource.snapshot();self.assertTrue(resource.capability(wd14.DEFAULT,"wd",first,valid)["available"])
            with patch("torch.cuda.mem_get_info",return_value=(1,8*1024**3)):
                current=resource.snapshot();self.assertFalse(resource.capability(wd14.DEFAULT,"wd",current,valid)["available"])
            with patch.object(resource,"GPU_BUSY",return_value=True):
                current=resource.snapshot();self.assertFalse(resource.capability(wd14.DEFAULT,"wd",current,valid)["available"])


if __name__=="__main__":unittest.main()
