import contextlib
import json
import sys
from pathlib import Path


def main():
    payload = json.loads(sys.stdin.read())
    model_path = Path(payload["model"])
    image_path = Path(payload["image"])
    mask_directory = Path(payload["mask_directory"])
    confidence = float(payload["confidence"])

    with contextlib.redirect_stdout(sys.stderr):
        from PIL import Image
        from ultralytics import YOLO

        image = Image.open(image_path).convert("RGB")
        model = YOLO(str(model_path))
        result = model.predict(image, conf=confidence, device="cpu", verbose=False)[0]

    names = result.names
    boxes = result.boxes.xyxy.cpu().tolist() if result.boxes is not None else []
    confidences = result.boxes.conf.cpu().tolist() if result.boxes is not None else []
    classes = result.boxes.cls.cpu().tolist() if result.boxes is not None else []
    segmentation = result.masks.data.cpu() if result.masks is not None else None
    detections = []
    for index, (box, score, class_id) in enumerate(zip(boxes, confidences, classes)):
        mask_name = None
        if segmentation is not None and index < len(segmentation):
            mask_name = f"mask-{index:02d}.png"
            mask = Image.fromarray((segmentation[index].numpy() * 255).astype("uint8"), mode="L")
            mask.resize(image.size, Image.Resampling.BILINEAR).save(mask_directory / mask_name)
        detections.append({
            "box": box,
            "confidence": score,
            "class_name": names.get(int(class_id), str(int(class_id))),
            "mask": mask_name,
        })
    print(json.dumps({"detections": detections}, ensure_ascii=True))


if __name__ == "__main__":
    main()
