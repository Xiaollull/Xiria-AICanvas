import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from backend.benchmark_lease import MAX_LEASE_BYTES, MAX_TTL_SECONDS, _default_inspector, create_lease, validate_lease


class BenchmarkLeaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.state, self.output = self.root / "state", self.root / "output"
        self.script = self.root / "state-cache" / "benchmark" / "harness.py"
        self.script.parent.mkdir(parents=True); self.script.write_text("# test", encoding="utf-8")
        self.python = self.root / ".venv" / "Scripts" / "python.exe"; self.python.parent.mkdir(parents=True); self.python.write_bytes(b"")
        self.now = datetime.now(timezone.utc)
        self.path, self.nonce = create_lease(self.state, self.output, "workspace", self.script, 60, 77, self.python)
        self.inspector = lambda _pid: {"executable": str(self.python), "command_line": [str(self.python), "-u", str(self.script)], "cwd": str(self.root)}

    def tearDown(self): self.temp.cleanup()
    def valid(self, **kwargs):
        nonce = kwargs.pop("nonce", self.nonce)
        options = {"parent_pid": 77, "now": self.now, "inspector": self.inspector}; options.update(kwargs)
        return validate_lease(self.path, nonce, self.state, self.output, "workspace", self.root, **options)
    def payload(self): return json.loads(self.path.read_text(encoding="utf-8"))
    def write(self, value): self.path.write_text(json.dumps(value), encoding="utf-8")
    def fresh(self): self.path, self.nonce = create_lease(self.state, self.output, "workspace", self.script, 60, 77, self.python)
    def assert_rejected(self, **kwargs):
        result = self.valid(**kwargs); self.assertFalse(result["valid"]); self.assertTrue(result["reason"].startswith("lease_")); return result

    def test_valid_relative_script_and_flags(self):
        self.assertTrue(self.valid()["valid"])
        # A parent that spells its script relative to its own cwd and carries
        # interpreter flags is the ordinary case, not a mismatch.
        inspector = lambda _pid: {"executable": str(self.python), "command_line": [str(self.python), "-u", "state-cache/benchmark/harness.py"], "cwd": str(self.root)}
        self.assertTrue(self.valid(inspector=inspector)["valid"])

    def test_executable_case_follows_the_host_filesystem(self):
        # `Path.resolve()` folds case on Windows and not on POSIX, and that is
        # exactly right: on Windows a retyped path is the same file, while on
        # Linux `/srv/PY` and `/srv/py` are two different binaries. Accepting a
        # differently-cased executable there would bind the lease to a parent
        # that is not the one it was minted for.
        inspector = lambda _pid: {"executable": str(self.python).upper(), "command_line": [str(self.python).upper(), "-u", str(self.script)], "cwd": str(self.root)}
        result = self.valid(inspector=inspector)
        if os.name == "nt":
            self.assertTrue(result["valid"])
        else:
            self.assertFalse(result["valid"])
            self.assertEqual(result["reason"], "lease_parent_process")

    def test_missing_empty_malformed_oversize_and_path_escape_fail(self):
        self.path.unlink(); self.assertFalse(self.valid()["valid"])
        self.path.write_text("", encoding="utf-8"); self.assertFalse(self.valid()["valid"])
        self.path.write_text("{", encoding="utf-8"); self.assertFalse(self.valid()["valid"])
        self.path.write_bytes(b"x" * (MAX_LEASE_BYTES + 1)); self.assertFalse(self.valid()["valid"])
        outside = self.root / "outside.json"; outside.write_text("{}", encoding="utf-8")
        self.assertFalse(validate_lease(outside, self.nonce, self.state, self.output, "workspace", self.root, parent_pid=77, now=self.now, inspector=self.inspector)["valid"])

    def test_schema_version_purpose_hash_and_identity_fail_closed(self):
        for key, value in (("extra", 1), ("purpose", "wrong"), ("version", 2), ("parent_pid", "77"), ("nonce_sha256", "bad")):
            payload = self.payload(); payload[key] = value; self.write(payload); self.assertFalse(self.valid()["valid"], key)
            self.path, self.nonce = create_lease(self.state, self.output, "workspace", self.script, 60, 77, self.python)
        payload = self.payload(); del payload["workspace_id"]; self.write(payload); self.assertFalse(self.valid()["valid"])
        self.path, self.nonce = create_lease(self.state, self.output, "workspace", self.script, 60, 77, self.python)
        self.assertFalse(validate_lease(self.path, self.nonce, self.state / "other", self.output, "workspace", self.root, parent_pid=77, now=self.now, inspector=self.inspector)["valid"])
        self.assertFalse(validate_lease(self.path, self.nonce, self.state, self.output / "other", "workspace", self.root, parent_pid=77, now=self.now, inspector=self.inspector)["valid"])
        self.assertFalse(validate_lease(self.path, self.nonce, self.state, self.output, "other", self.root, parent_pid=77, now=self.now, inspector=self.inspector)["valid"])

    def test_each_required_json_field_missing_fails_closed(self):
        required = ("version", "purpose", "parent_pid", "parent_executable", "parent_script", "state_root", "output_root", "workspace_id", "nonce_sha256", "expires_at_utc")
        for key in required:
            payload = self.payload(); del payload[key]; self.write(payload)
            self.assertEqual(self.assert_rejected()["reason"], "lease_schema", key)
            self.fresh()

    def test_each_string_field_empty_and_wrong_type_fails_closed(self):
        string_fields = ("purpose", "parent_executable", "parent_script", "state_root", "output_root", "workspace_id", "nonce_sha256", "expires_at_utc")
        for key in string_fields:
            for value in ("", 7):
                payload = self.payload(); payload[key] = value; self.write(payload)
                self.assertEqual(self.assert_rejected()["reason"], "lease_schema", f"{key}={value!r}")
                self.fresh()

    def test_parent_pid_timestamp_and_nonce_shapes_fail_closed(self):
        for value in ("77", None, 7.0, True):
            payload = self.payload(); payload["parent_pid"] = value; self.write(payload)
            self.assertEqual(self.assert_rejected()["reason"], "lease_schema", f"stored parent_pid={value!r}"); self.fresh()
        for value in ("not-a-date", "2026-08-02T00:00:00", self.now.isoformat(), (self.now - timedelta(seconds=1)).isoformat(), (self.now + timedelta(seconds=MAX_TTL_SECONDS + 1)).isoformat()):
            payload = self.payload(); payload["expires_at_utc"] = value; self.write(payload)
            self.assertEqual(self.assert_rejected()["reason"], "lease_expiry", f"expiry={value!r}"); self.fresh()
        for value in (None, 1, "", "wrong"):
            self.assertEqual(self.assert_rejected(nonce=value)["reason"], "lease_nonce", f"nonce={value!r}")
        for value in ("f" * 63, "z" * 64):
            payload = self.payload(); payload["nonce_sha256"] = value; self.write(payload)
            self.assertEqual(self.assert_rejected()["reason"], "lease_schema", f"hash={value!r}"); self.fresh()
        for value in ("77", True): self.assertEqual(self.assert_rejected(parent_pid=value)["reason"], "lease_parent_pid")
        self.assertEqual(self.assert_rejected(now="now")["reason"], "lease_expiry")

    def test_injected_resolver_rejects_lease_and_script_reparse_escapes(self):
        outside = self.root / "outside"; outside.mkdir(); escaped_lease = outside / "lease.json"; escaped_lease.write_text(self.path.read_text(encoding="utf-8"), encoding="utf-8")
        def lease_escape(value):
            return escaped_lease if Path(value).name == self.path.name else Path(value).resolve()
        self.assertEqual(self.assert_rejected(resolver=lease_escape)["reason"], "lease_path")
        escaped_script = outside / "harness.py"; escaped_script.write_text("# outside", encoding="utf-8")
        def script_escape(value):
            return escaped_script if Path(value) == self.script else Path(value).resolve()
        self.assertEqual(self.assert_rejected(resolver=script_escape)["reason"], "lease_parent_binding")

    def test_injected_normal_resolver_preserves_valid_lease(self):
        resolver = lambda value: Path(value).expanduser().resolve()
        self.assertTrue(self.valid(resolver=resolver)["valid"])

    def test_ttl_expiry_and_revalidation_after_content_delete(self):
        with self.assertRaises(ValueError): create_lease(self.state, self.output, "workspace", self.script, 0, 77, self.python)
        with self.assertRaises(ValueError): create_lease(self.state, self.output, "workspace", self.script, 901, 77, self.python)
        self.assertFalse(self.valid(now=self.now + timedelta(seconds=61))["valid"])
        self.assertTrue(self.valid()["valid"])
        payload = self.payload(); payload["workspace_id"] = "changed"; self.write(payload)
        self.assertFalse(self.valid()["valid"])
        self.path.unlink(); self.assertFalse(self.valid()["valid"])

    def test_parent_pid_executable_node_system_script_and_token_rejections(self):
        self.assertFalse(self.valid(parent_pid=78)["valid"])
        node = lambda _p: {"executable": "node.exe", "command_line": ["node.exe", str(self.script)]}
        self.assertFalse(validate_lease(self.path, self.nonce, self.state, self.output, "workspace", self.root, parent_pid=77, now=self.now, inspector=node)["valid"])
        system = lambda _p: {"executable": str(self.python), "command_line": ["C:/Python/python.exe", str(self.script)]}
        self.assertFalse(validate_lease(self.path, self.nonce, self.state, self.output, "workspace", self.root, parent_pid=77, now=self.now, inspector=system)["valid"])
        missing = lambda _p: {"executable": str(self.python), "command_line": [str(self.python), "-u", "other.py"], "cwd": str(self.root)}
        self.assertFalse(validate_lease(self.path, self.nonce, self.state, self.output, "workspace", self.root, parent_pid=77, now=self.now, inspector=missing)["valid"])
        outside = self.root / "outside.py"; outside.write_text("# no", encoding="utf-8")
        payload = self.payload(); payload["parent_script"] = str(outside); self.write(payload)
        self.assertFalse(self.valid()["valid"])

    def test_nonpy_missing_script_and_symlink_when_supported(self):
        payload = self.payload(); payload["parent_script"] = str(self.script.with_suffix(".txt")); self.write(payload); self.assertFalse(self.valid()["valid"])
        self.path, self.nonce = create_lease(self.state, self.output, "workspace", self.script, 60, 77, self.python); self.script.unlink(); self.assertFalse(self.valid()["valid"])
        target = self.root / "outside.py"; target.write_text("# target", encoding="utf-8"); link = self.root / "state-cache" / "benchmark" / "link.py"
        try: link.symlink_to(target)
        except OSError: self.skipTest("symlink unsupported")
        self.path, self.nonce = create_lease(self.state, self.output, "workspace", link, 60, 77, self.python)
        self.assertFalse(self.valid()["valid"])

    def test_windows_venv_child_direct_parent_identity_validates_then_lease_deletes(self):
        parent = self.root / "state-cache" / "benchmark" / "parent.py"
        repo = Path(__file__).resolve().parents[1]
        parent.write_text(f'''import json, os, subprocess, sys
from pathlib import Path
sys.path.insert(0, {str(repo)!r})
from backend.benchmark_lease import create_lease, _default_inspector
root = Path({str(self.root)!r}); state=root / "state"; output=root / "output"
identity = _default_inspector(os.getpid())
lease, nonce = create_lease(state, output, "workspace", Path(__file__), 60, parent_pid=os.getpid(), parent_executable=identity['executable'])
child = """import json,os,sys; from pathlib import Path; sys.path.insert(0,{str(repo)!r}); from backend.benchmark_lease import validate_lease; root,path,nonce,state,output=sys.argv[1:]; print(json.dumps({{'pid':os.getpid(),'ppid':os.getppid(),'result':validate_lease(path,nonce,state,output,'workspace',root)}}))"""
env=os.environ.copy(); env["__PYVENV_LAUNCHER__"] = sys.executable
base = getattr(sys, "_base_executable", sys.executable)
child_result = subprocess.run([base, "-c", child, str(root), str(lease), nonce, str(state), str(output)], env=env, capture_output=True, text=True, check=True)
record=json.loads(child_result.stdout); print(json.dumps({{'stored_parent_pid':json.loads(lease.read_text())['parent_pid'],'stored_parent_executable':json.loads(lease.read_text())['parent_executable'],'parent_identity':identity,'child':record,'lease_path':str(lease)}})); lease.unlink()
''', encoding="utf-8")
        completed = subprocess.run([sys.executable, str(parent)], capture_output=True, text=True, check=True)
        result = json.loads(completed.stdout)
        self.assertTrue(result["child"]["result"]["valid"], result)
        self.assertEqual(result["child"]["ppid"], result["stored_parent_pid"])
        self.assertFalse(Path(result["lease_path"]).exists())
