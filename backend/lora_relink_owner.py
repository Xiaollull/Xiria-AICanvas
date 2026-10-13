"""Private UTF-8 stdin/stdout Gallery-owner bridge, never registered as HTTP."""
import json
import sys
from pathlib import Path
from .gallery import GalleryStore


def main():
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    request = json.load(sys.stdin)
    if request["action"] == "xyz-references":
        # Stdin owner IPC only, NOT an HTTP endpoint or a caller-authorizing protocol hash.
        from .lora_library_gate import library_admission, verify_node_lease
        from .xyz_projection import effective_references
        verify_node_lease(request['stateDirectory'], request.get('token'))
        with library_admission(request['stateDirectory'], delegated=request.get('token')):
            registry_path = Path(request['stateDirectory']) / 'lora-library' / 'registry.json'
            registry = json.loads(registry_path.read_text(encoding='utf-8')) if registry_path.exists() else None
            result = effective_references(request['draft'], registry)
        print(json.dumps(result, ensure_ascii=False))
        return
    if request["action"] == "identity":
        from .lora_file_identity import file_identity
        print(json.dumps(file_identity(request["file"])))
        return
    if request["action"] == "list" and not (Path(request["stateDirectory"]) / "gallery.sqlite3").exists():
        print(json.dumps({"cards": [], "collections": []}))
        return
    store = GalleryStore(request["stateDirectory"], relink_token=request["token"])
    result = store.list_gallery() if request["action"] == "list" else store.relink_transaction(request["action"], request.get("operationId"), request.get("patches", []))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
