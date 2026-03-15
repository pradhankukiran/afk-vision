from __future__ import annotations

import json

from vision.providers import (
    LocalInferenceDetectorProvider,
    LocalInferenceEmbeddingProvider,
    OllamaExplanationProvider,
    validate_runtime_configuration,
)
from vision.services import Tile


class StubResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_local_detector_provider_parses_detection_payload(monkeypatch):
    def fake_urlopen(req, timeout=0):
        assert req.full_url.endswith("/detect")
        return StubResponse(
            {
                "detections": [
                    {
                        "label": "truck",
                        "confidence": 0.88,
                        "bbox": {"x1": 10, "y1": 20, "x2": 120, "y2": 220},
                        "tile_bbox": {"tile_x": 0, "tile_y": 0, "tile_width": 512, "tile_height": 512},
                    }
                ]
            }
        )

    monkeypatch.setattr("vision.providers.request.urlopen", fake_urlopen)

    detections = LocalInferenceDetectorProvider().detect(b"image-bytes", Tile(x=0, y=0, width=512, height=512))
    assert len(detections) == 1
    assert detections[0].label == "truck"
    assert detections[0].bbox["x2"] == 120.0


def test_ollama_explanation_provider_parses_json_response(monkeypatch):
    def fake_urlopen(req, timeout=0):
        assert req.full_url.endswith("/api/generate")
        return StubResponse(
            {
                "response": json.dumps(
                    {
                        "verification_state": "verified",
                        "explanation": "The crop is consistent with the proposed label.",
                    }
                )
            }
        )

    monkeypatch.setattr("vision.providers.request.urlopen", fake_urlopen)

    state, explanation = OllamaExplanationProvider().explain(b"crop-bytes", "truck", 0.88)
    assert state == "verified"
    assert "consistent" in explanation


def test_local_embedding_provider_parses_embedding(monkeypatch):
    def fake_urlopen(req, timeout=0):
        assert req.full_url.endswith("/embed")
        return StubResponse({"embedding": [0.1, 0.2, 0.3]})

    monkeypatch.setattr("vision.providers.request.urlopen", fake_urlopen)

    vector = LocalInferenceEmbeddingProvider().embed_image(b"crop-bytes")
    assert vector == [0.1, 0.2, 0.3]


def test_validate_runtime_configuration_checks_local_services(monkeypatch):
    def fake_urlopen(req, timeout=0):
        if req.full_url.endswith("/health"):
            return StubResponse({"ready": True})
        if req.full_url.endswith("/api/tags"):
            return StubResponse({"models": [{"name": "moondream:latest"}]})
        raise AssertionError(f"Unexpected URL {req.full_url}")

    monkeypatch.setattr("vision.providers.request.urlopen", fake_urlopen)

    versions = validate_runtime_configuration()
    assert versions["detector"] == "yolov8n.pt"
    assert versions["detector_repo"] == "NvMayMay/weedblaster-vision-yolov8s"
    assert versions["detector_file"] == "best.pt"
    assert versions["explainer"] == "moondream"
    assert versions["embedding"] == "openai/clip-vit-base-patch32"
    assert versions["class_schema"] == "cropandweed_9"
    assert versions["schema_labels"] == [
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
    assert versions["explainer_status"] == "ready"


def test_validate_runtime_configuration_allows_pending_ollama_model(monkeypatch):
    def fake_urlopen(req, timeout=0):
        if req.full_url.endswith("/health"):
            return StubResponse({"ready": True})
        if req.full_url.endswith("/api/tags"):
            return StubResponse({"models": []})
        raise AssertionError(f"Unexpected URL {req.full_url}")

    monkeypatch.setattr("vision.providers.request.urlopen", fake_urlopen)

    versions = validate_runtime_configuration()
    assert versions["class_schema"] == "cropandweed_9"
    assert versions["explainer_status"] == "pending_install"
