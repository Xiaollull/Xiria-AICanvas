"""The service runs as a script (`python backend/inference_server.py`), not as `backend.inference_server`.

Its gallery import block is executed here exactly as written, in a fresh interpreter laid out the way
Vite starts the service. Before the fix its fallback loaded a top-level `gallery` module whose relative
P10 imports (`.lora_relink_store`) failed, so every gallery request answered 500 while the package-mode
unit tests stayed green. Found by the QPlan2/P1 Phase E runtime check (adding an XYZ cell to the gallery).
"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"


class GalleryScriptModeTests(unittest.TestCase):
    def test_the_service_import_block_opens_the_gallery_when_run_as_a_script(self):
        source = (BACKEND / "inference_server.py").read_text(encoding="utf-8")
        start = source.index("try:\n    from .gallery import (")
        fallback = source.index("except ImportError:", start)
        end = source.index("\n    )\n", fallback) + len("\n    )\n")
        block = source[start:end]
        self.assertNotIn("\ntry:", block[1:], "gallery import block not isolated")
        work = Path(tempfile.mkdtemp(prefix="gallery-script-mode-"))
        program = work / "service_imports.py"
        program.write_text("\n".join([
            "import sys",
            # Script mode: the script's own folder first; the project root is what inference_server adds.
            f"sys.path[:0] = [{str(BACKEND)!r}, {str(ROOT)!r}]",
            block,
            f"store = GalleryStore({str(work / 'state')!r})",
            "print(type(store).__module__, 'gallery' in sys.modules, sorted(name for name in sys.modules if name.startswith('backend.')))",
        ]), encoding="utf-8")
        result = subprocess.run([sys.executable, str(program)], cwd=str(BACKEND), capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        module, top_level_gallery, loaded = result.stdout.split(" ", 2)
        self.assertEqual(module, "backend.gallery")
        self.assertEqual(top_level_gallery, "False", "a second, top-level copy of the gallery module would not share exception classes")
        self.assertIn("backend.lora_relink_store", loaded)
        self.assertIn("backend.lora_library_gate", loaded)


if __name__ == "__main__":
    unittest.main()
