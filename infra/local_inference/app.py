from __future__ import annotations

import base64
import io
import json
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from huggingface_hub import hf_hub_download
from PIL import Image
from pydantic import BaseModel
from transformers import CLIPModel, CLIPProcessor
from ultralytics import YOLO


class TilePayload(BaseModel):
    x: int
    y: int
    width: int
    height: int


class DetectRequest(BaseModel):
    image_b64: str
    tile: TilePayload


class EmbedRequest(BaseModel):
    image_b64: str


CROPANDWEED_SCHEMA_LABELS = [
    "maize",
    "sugar_beet",
    "soy",
    "sunflower",
    "potato",
    "pea",
    "bean",
    "pumpkin",
    "weed",
]


def _detector_model_name() -> str:
    return os.getenv("AFKVISION_DETECTOR_MODEL", "yolov8n.pt")


def _detector_model_repo() -> str:
    return os.getenv("AFKVISION_DETECTOR_MODEL_REPO", "").strip()


def _detector_model_file() -> str:
    return os.getenv("AFKVISION_DETECTOR_MODEL_FILE", "best.pt").strip() or "best.pt"


def _embedding_model_name() -> str:
    return os.getenv("AFKVISION_EMBEDDING_MODEL", "openai/clip-vit-base-patch32")


def _detector_confidence() -> float:
    return float(os.getenv("AFKVISION_DETECTOR_CONFIDENCE", "0.2"))


def _detector_image_size() -> int:
    return int(os.getenv("AFKVISION_DETECTOR_IMAGE_SIZE", "1024"))


def _class_schema_name() -> str:
    return os.getenv("AFKVISION_CLASS_SCHEMA", "cropandweed_9").strip() or "cropandweed_9"


def _schema_labels() -> list[str]:
    raw = os.getenv("AFKVISION_SCHEMA_LABELS", "").strip()
    if raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    return CROPANDWEED_SCHEMA_LABELS


def _allowed_labels() -> set[str]:
    raw = os.getenv("AFKVISION_ALLOWED_LABELS", "").strip()
    if not raw:
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def _label_aliases() -> dict[str, str]:
    raw = os.getenv("AFKVISION_LABEL_ALIASES_JSON", "").strip()
    if not raw:
        if _class_schema_name() == "cropandweed_9":
            return {
                "Maize": "maize",
                "Sugar Beet": "sugar_beet",
                "Soy": "soy",
                "Sunflower": "sunflower",
                "Potato": "potato",
                "Pea": "pea",
                "Bean": "bean",
                "Pumpkin": "pumpkin",
                "Weed": "weed",
            }
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(key).strip(): str(value).strip() for key, value in payload.items() if str(key).strip()}


def _resolve_detector_model() -> tuple[str, str]:
    repo_id = _detector_model_repo()
    if repo_id:
        filename = _detector_model_file()
        local_path = hf_hub_download(repo_id=repo_id, filename=filename)
        return local_path, f"{repo_id}:{filename}"
    return _detector_model_name(), _detector_model_name()


def _decode_image(image_b64: str) -> Image.Image:
    try:
        raw = base64.b64decode(image_b64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="image_b64 is not valid base64.") from exc
    return Image.open(io.BytesIO(raw)).convert("RGB")


@asynccontextmanager
async def lifespan(app: FastAPI):
    torch.set_num_threads(max(1, int(os.getenv("OMP_NUM_THREADS", "4"))))
    detector_model_path, detector_model_name = _resolve_detector_model()
    app.state.detector_model_name = detector_model_name
    app.state.embedding_model_name = _embedding_model_name()
    app.state.class_schema = _class_schema_name()
    app.state.schema_labels = _schema_labels()
    app.state.allowed_labels = _allowed_labels()
    app.state.label_aliases = _label_aliases()
    app.state.detector = YOLO(detector_model_path)
    app.state.embedding_processor = CLIPProcessor.from_pretrained(app.state.embedding_model_name)
    app.state.embedding_model = CLIPModel.from_pretrained(app.state.embedding_model_name)
    app.state.embedding_model.eval()
    app.state.ready = True
    yield


app = FastAPI(title="AFK Vision Local Inference", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "ready": bool(getattr(app.state, "ready", False)),
        "detector_model": getattr(app.state, "detector_model_name", None),
        "embedding_model": getattr(app.state, "embedding_model_name", None),
        "class_schema": getattr(app.state, "class_schema", None),
        "schema_labels": getattr(app.state, "schema_labels", []),
        "allowed_labels": sorted(getattr(app.state, "allowed_labels", set())),
    }


@app.post("/detect")
def detect(payload: DetectRequest) -> dict[str, object]:
    image = _decode_image(payload.image_b64)
    result = app.state.detector.predict(
        source=image,
        device="cpu",
        imgsz=_detector_image_size(),
        conf=_detector_confidence(),
        verbose=False,
    )[0]

    detections = []
    names = result.names
    boxes = result.boxes
    if boxes is None:
        return {"detections": detections}

    for box in boxes:
        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        confidence = float(box.conf[0].item())
        label = str(names[int(box.cls[0].item())])
        label = app.state.label_aliases.get(label, label)
        allowed_labels = app.state.allowed_labels
        if allowed_labels and label not in allowed_labels:
            continue

        x1 = max(0.0, min(float(payload.tile.width), x1))
        y1 = max(0.0, min(float(payload.tile.height), y1))
        x2 = max(0.0, min(float(payload.tile.width), x2))
        y2 = max(0.0, min(float(payload.tile.height), y2))
        if x2 <= x1 or y2 <= y1:
            continue

        detections.append(
            {
                "label": label,
                "confidence": round(confidence, 4),
                "bbox": {
                    "x1": round(payload.tile.x + x1, 2),
                    "y1": round(payload.tile.y + y1, 2),
                    "x2": round(payload.tile.x + x2, 2),
                    "y2": round(payload.tile.y + y2, 2),
                },
                "tile_bbox": {
                    "tile_x": payload.tile.x,
                    "tile_y": payload.tile.y,
                    "tile_width": payload.tile.width,
                    "tile_height": payload.tile.height,
                },
            }
        )
    return {"detections": detections}


@app.post("/embed")
def embed(payload: EmbedRequest) -> dict[str, object]:
    image = _decode_image(payload.image_b64)
    inputs = app.state.embedding_processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        features = app.state.embedding_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return {"embedding": [float(value) for value in features[0].cpu().tolist()]}
