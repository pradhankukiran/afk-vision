from __future__ import annotations

import io

import numpy as np
import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from PIL import Image, ImageDraw

from vision import api_views, tasks
from vision.models import Detection, EmbeddingRecord, ImageAsset, InferenceRun, Project
from vision.services import DetectionCandidate
from vision.tasks import process_inference_run


class FakeDetectorProvider:
    def detect(self, image_bytes: bytes, tile) -> list[DetectionCandidate]:
        if tile.x != 0 or tile.y != 0:
            return []
        return [
            DetectionCandidate(
                label="disease_hotspot",
                confidence=0.91,
                bbox={"x1": 180.0, "y1": 180.0, "x2": 520.0, "y2": 520.0},
                tile_bbox={"tile_x": tile.x, "tile_y": tile.y, "tile_width": tile.width, "tile_height": tile.height},
            ),
            DetectionCandidate(
                label="irrigation_anomaly",
                confidence=0.84,
                bbox={"x1": 820.0, "y1": 320.0, "x2": 1180.0, "y2": 720.0},
                tile_bbox={"tile_x": tile.x, "tile_y": tile.y, "tile_width": tile.width, "tile_height": tile.height},
            ),
            DetectionCandidate(
                label="pest_cluster",
                confidence=0.73,
                bbox={"x1": 1400.0, "y1": 860.0, "x2": 1820.0, "y2": 1220.0},
                tile_bbox={"tile_x": tile.x, "tile_y": tile.y, "tile_width": tile.width, "tile_height": tile.height},
            ),
        ]


class FakeExplanationProvider:
    def explain(self, image_bytes: bytes, label: str, confidence: float) -> tuple[str, str]:
        state = "verified" if confidence >= 0.8 else "flagged"
        return state, f"{label} explanation"


class FakeEmbeddingProvider:
    def embed_image(self, image_bytes: bytes) -> list[float]:
        seed = float(len(image_bytes) % 11)
        return [seed + float(index) for index in range(1, 513)]


class BrokenExplanationProvider:
    def explain(self, image_bytes: bytes, label: str, confidence: float) -> tuple[str, str]:
        raise RuntimeError("ollama model is still installing")


@pytest.fixture(autouse=True)
def provider_stubs(monkeypatch):
    monkeypatch.setattr(tasks, "get_detector_provider", lambda: FakeDetectorProvider())
    monkeypatch.setattr(tasks, "get_explanation_provider", lambda: FakeExplanationProvider())
    monkeypatch.setattr(tasks, "get_embedding_provider", lambda: FakeEmbeddingProvider())
    monkeypatch.setattr(api_views, "validate_runtime_configuration", lambda: {
        "detector": "yolov8n.pt",
        "explainer": "moondream",
        "embedding": "openai/clip-vit-base-patch32",
    })
    monkeypatch.setattr(api_views, "get_embedding_provider", lambda: FakeEmbeddingProvider())


@pytest.mark.django_db
def test_inference_pipeline_creates_detections_and_embeddings(settings):
    settings.CELERY_TASK_ALWAYS_EAGER = True
    project = Project.objects.create(name="Demo")
    upload = SimpleUploadedFile("demo.png", make_demo_image_bytes(), content_type="image/png")
    image_asset = ImageAsset.objects.create(project=project, image=upload)
    run = InferenceRun.objects.create(project=project, image=image_asset)

    process_inference_run.apply(args=[run.id]).get()

    run.refresh_from_db()
    assert run.status == "succeeded"
    assert Detection.objects.filter(run=run).count() >= 3
    assert EmbeddingRecord.objects.count() == Detection.objects.count()


@pytest.mark.django_db
def test_api_project_upload_and_similar_search(client, settings):
    settings.CELERY_TASK_ALWAYS_EAGER = True
    response = client.post("/api/projects/", {"name": "API Demo", "description": "demo"})
    assert response.status_code == 201
    project_id = response.json()["id"]

    upload_response = client.post(
        f"/api/projects/{project_id}/images/",
        {"image": SimpleUploadedFile("demo.png", make_demo_image_bytes(), content_type="image/png")},
    )
    assert upload_response.status_code == 201
    image_id = upload_response.json()["id"]

    run_response = client.post(
        f"/api/projects/{project_id}/runs/",
        data={"image_id": image_id},
        content_type="application/json",
    )
    assert run_response.status_code == 201
    run_id = run_response.json()["id"]

    run = InferenceRun.objects.get(pk=run_id)
    process_inference_run.apply(args=[run.id]).get()

    detection = Detection.objects.filter(run=run).first()
    assert detection is not None

    results = client.post("/api/search/similar/", {"detection_id": detection.id, "limit": 3})
    assert results.status_code == 200
    assert "results" in results.json()


@pytest.mark.django_db
def test_inference_pipeline_continues_when_explainer_is_unavailable(monkeypatch):
    monkeypatch.setattr(tasks, "get_detector_provider", lambda: FakeDetectorProvider())
    monkeypatch.setattr(tasks, "get_explanation_provider", lambda: BrokenExplanationProvider())
    monkeypatch.setattr(tasks, "get_embedding_provider", lambda: FakeEmbeddingProvider())

    project = Project.objects.create(name="Explainer Offline")
    upload = SimpleUploadedFile("demo.png", make_demo_image_bytes(), content_type="image/png")
    image_asset = ImageAsset.objects.create(project=project, image=upload)
    run = InferenceRun.objects.create(project=project, image=image_asset)

    process_inference_run.apply(args=[run.id]).get()

    detections = Detection.objects.filter(run=run)
    assert detections.exists()
    assert all(detection.verification_state == "not_run" for detection in detections)


def make_demo_image_bytes() -> bytes:
    width, height = 2048, 1536
    rng = np.random.default_rng(7)
    base = np.zeros((height, width, 3), dtype=np.uint8)
    base[:, :, 0] = rng.integers(30, 60, size=(height, width))
    base[:, :, 1] = rng.integers(90, 150, size=(height, width))
    base[:, :, 2] = rng.integers(35, 70, size=(height, width))

    image = Image.fromarray(base, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((180, 180, 520, 520), fill=(214, 96, 80))
    draw.rectangle((820, 320, 1180, 720), fill=(75, 210, 255))
    draw.rectangle((1400, 860, 1820, 1220), fill=(244, 176, 0))

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
