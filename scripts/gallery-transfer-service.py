"""PRIVATE CPU-only HTTP fixture for the gallery import/export browser check.

The actual service gallery routes on private state and output roots, seeded with synthetic
collections. Requires a pre-created owned evidence directory; no app lifespan, model or GPU.
"""
import argparse
import base64
import io
import json
import os
import tempfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True)
parser.add_argument("--port", type=int, required=True)
args = parser.parse_args()
root = Path(args.root).resolve()
approved = Path(os.environ["LOCALAPPDATA"]) / "Temp" / "opencode"
if approved.resolve() not in root.parents or not (root / "OWNED-FIXTURE.json").is_file():
    raise RuntimeError("Unowned fixture root refused")
tempfile.tempdir = str(root)
os.environ["XIRAI_STATE_DIR"] = str(root / "state")
os.environ["XIRAI_OUTPUT_DIR"] = str(root / "outputs")
os.environ["XIRAI_WD14_PROXY_PROOF"] = "gallery-transfer-private-proof"

from fastapi import FastAPI  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from backend import inference_server as server  # noqa: E402


def picture(index, image_format="PNG", size=(320, 420)):
    hue = (index * 47) % 255
    image = Image.new("RGB", size, (30 + hue // 3, 40 + (hue * 2) % 160, 70 + (255 - hue) // 3))
    draw = ImageDraw.Draw(image)
    for step in range(0, size[1], 24):
        draw.rectangle([0, step, size[0], step + 10], fill=(60 + (step + hue) % 150, 50 + step % 120, 90 + hue % 100))
    draw.ellipse([size[0] // 4, size[1] // 4, size[0] * 3 // 4, size[1] * 3 // 5], fill=(230, 200 - hue % 80, 140))
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    mime = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}[image_format]
    return f"data:{mime};base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


store = server.gallery_store()
if not store.list_gallery()["collections"]:
    store.create_collection("人像 portraits", "夹具：两张卡片，含 PNG 与 JPEG")
    store.create_cards("人像 portraits", [
        {"title": "window light", "settings": {"model": "iL", "positive": "portrait, window light", "loras": [{"value": "style/A.safetensors", "weight": 0.7}]},
         "images": [{"data_url": picture(1), "name": "window-1.png"}, {"data_url": picture(2, "JPEG"), "name": "window-2.jpg"}]},
        {"title": "text only card", "settings": {"positive": "no image here"}, "images": []},
    ])
    store.create_collection("landscapes")
    store.create_cards("landscapes", [{"title": "hill", "settings": {"positive": "hill"}, "images": [{"data_url": picture(3, "WEBP"), "name": "hill.webp"}]}])
    store.create_collection("一个名字很长的收藏夹，用来检查导出列表里的截断与对齐是否正常显示", "长名称与简介：简介也应该在一行内截断，不撑开对话框的宽度。")
    store.create_cards("一个名字很长的收藏夹，用来检查导出列表里的截断与对齐是否正常显示", [{"title": "long", "settings": {}, "images": [{"data_url": picture(4), "name": "long.png"}]}])
    for number in range(1, 9):
        collection = f"studies-{number:02d}"
        store.create_collection(collection)
        store.create_cards(collection, [{"title": f"study {number}", "settings": {}, "images": [{"data_url": picture(10 + number), "name": f"study-{number}.png"}]}])

info = {"state": str(server.STATE_DIRECTORY), "outputs": str(server.OUTPUT_DIRECTORY), "proof": os.environ["XIRAI_WD14_PROXY_PROOF"]}
(root / "fixture-info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

app = FastAPI()
app.router.routes.extend(route for route in server.app.routes if getattr(route, "path", "").startswith("/api/inference/gallery"))


@app.get("/fixture-info")
def fixture_info():
    return info


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
