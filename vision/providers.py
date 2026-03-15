from __future__ import annotations

import base64
import json
import os
from abc import ABC, abstractmethod
from typing import Any
from urllib import error, request

from .services import DetectionCandidate, Tile


LOCAL_INFERENCE_DEFAULT_URL = "http://127.0.0.1:8091"
OLLAMA_DEFAULT_URL = "http://127.0.0.1:11434"
DEFAULT_DETECTOR_MODEL = "yolov8n.pt"
DEFAULT_EMBEDDING_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_OLLAMA_MODEL = "moondream"
DEFAULT_DETECTOR_MODEL_REPO = "NvMayMay/weedblaster-vision-yolov8s"
DEFAULT_DETECTOR_MODEL_FILE = "best.pt"
DEFAULT_CLASS_SCHEMA = "cropandweed_9"
DEFAULT_SCHEMA_LABELS = "maize,sugar_beet,soy,sunflower,potato,pea,bean,pumpkin,weed"


class ProviderConfigurationError(RuntimeError):
    """Raised when a required runtime provider configuration is missing."""


class DetectorProvider(ABC):
    @abstractmethod
    def detect(self, image_bytes: bytes, tile: Tile) -> list[DetectionCandidate]:
        raise NotImplementedError


class ExplanationProvider(ABC):
    @abstractmethod
    def explain(self, image_bytes: bytes, label: str, confidence: float) -> tuple[str, str]:
        raise NotImplementedError


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_image(self, image_bytes: bytes) -> list[float]:
        raise NotImplementedError


def _env_or_default(name: str, default: str) -> str:
    return os.getenv(name, "").strip() or default


def _base_url(name: str, default: str) -> str:
    return _env_or_default(name, default).rstrip("/")


def _csv_env_or_default(name: str, default: str) -> list[str]:
    raw = _env_or_default(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _decode_json_response(response) -> dict[str, Any]:
    payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Provider response must be a JSON object.")
    return payload


def _http_get_json(url: str) -> dict[str, Any]:
    req = request.Request(url, method="GET")
    return _send_request(req)


def _http_post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return _send_request(req)


def _send_request(req: request.Request) -> dict[str, Any]:
    try:
        with request.urlopen(req, timeout=180) as response:
            return _decode_json_response(response)
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "ignore").strip()
        message = f"{req.full_url} returned HTTP {exc.code}."
        if detail:
            message = f"{message} {detail[:300]}"
        raise RuntimeError(message) from exc
    except error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise ProviderConfigurationError(f"Could not reach {req.full_url}: {reason}") from exc


class LocalInferenceDetectorProvider(DetectorProvider):
    def __init__(self) -> None:
        self.base_url = _base_url("AFKVISION_INFERENCE_BASE_URL", LOCAL_INFERENCE_DEFAULT_URL)

    def detect(self, image_bytes: bytes, tile: Tile) -> list[DetectionCandidate]:
        body = _http_post_json(
            f"{self.base_url}/detect",
            {
                "image_b64": base64.b64encode(image_bytes).decode("ascii"),
                "tile": {"x": tile.x, "y": tile.y, "width": tile.width, "height": tile.height},
            },
        )
        detections = body.get("detections", [])
        if not isinstance(detections, list):
            raise RuntimeError("Detector response must include a detections list.")

        results: list[DetectionCandidate] = []
        for item in detections:
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox", {})
            if not isinstance(bbox, dict):
                continue
            results.append(
                DetectionCandidate(
                    label=str(item.get("label", "")).strip(),
                    confidence=float(item.get("confidence", 0.0)),
                    bbox={
                        "x1": float(bbox["x1"]),
                        "y1": float(bbox["y1"]),
                        "x2": float(bbox["x2"]),
                        "y2": float(bbox["y2"]),
                    },
                    tile_bbox=item.get(
                        "tile_bbox",
                        {
                            "tile_x": tile.x,
                            "tile_y": tile.y,
                            "tile_width": tile.width,
                            "tile_height": tile.height,
                        },
                    ),
                )
            )
        return results


class OllamaExplanationProvider(ExplanationProvider):
    def __init__(self) -> None:
        self.base_url = _base_url("AFKVISION_OLLAMA_BASE_URL", OLLAMA_DEFAULT_URL)
        self.model_name = _env_or_default("AFKVISION_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)

    def explain(self, image_bytes: bytes, label: str, confidence: float) -> tuple[str, str]:
        prompt = (
            "You are validating an aerial-imagery detection crop. "
            f"The detector labeled this crop as '{label}' with confidence {confidence:.2f}. "
            "Respond with JSON containing two keys: verification_state and explanation. "
            "verification_state must be either 'verified' or 'flagged'. "
            "explanation must be one short operator-facing sentence."
        )
        body = _http_post_json(
            f"{self.base_url}/api/generate",
            {
                "model": self.model_name,
                "prompt": prompt,
                "images": [base64.b64encode(image_bytes).decode("ascii")],
                "format": {
                    "type": "object",
                    "properties": {
                        "verification_state": {"type": "string", "enum": ["verified", "flagged"]},
                        "explanation": {"type": "string"},
                    },
                    "required": ["verification_state", "explanation"],
                },
                "options": {"temperature": 0},
                "stream": False,
                "keep_alive": "30m",
            },
        )
        raw = str(body.get("response", "")).strip()
        if not raw:
            raise RuntimeError("Ollama did not return an explanation payload.")

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return "flagged", raw[:500]

        state = str(payload.get("verification_state", "flagged")).strip().lower()
        if state not in {"verified", "flagged"}:
            state = "flagged"
        explanation = str(payload.get("explanation", "")).strip()
        if not explanation:
            explanation = "The local VLM returned no explanation text."
        return state, explanation


class LocalInferenceEmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        self.base_url = _base_url("AFKVISION_INFERENCE_BASE_URL", LOCAL_INFERENCE_DEFAULT_URL)

    def embed_image(self, image_bytes: bytes) -> list[float]:
        body = _http_post_json(
            f"{self.base_url}/embed",
            {"image_b64": base64.b64encode(image_bytes).decode("ascii")},
        )

        vector = body.get("embedding")
        if not isinstance(vector, list):
            raise RuntimeError("Embedding response must include an embedding vector.")
        return [float(value) for value in vector]


def get_detector_provider() -> DetectorProvider:
    return LocalInferenceDetectorProvider()


def get_explanation_provider() -> ExplanationProvider:
    return OllamaExplanationProvider()


def get_embedding_provider() -> EmbeddingProvider:
    return LocalInferenceEmbeddingProvider()


def get_runtime_model_versions() -> dict[str, str]:
    return {
        "detector": _env_or_default("AFKVISION_DETECTOR_MODEL", DEFAULT_DETECTOR_MODEL),
        "detector_repo": _env_or_default("AFKVISION_DETECTOR_MODEL_REPO", DEFAULT_DETECTOR_MODEL_REPO),
        "detector_file": _env_or_default("AFKVISION_DETECTOR_MODEL_FILE", DEFAULT_DETECTOR_MODEL_FILE),
        "explainer": _env_or_default("AFKVISION_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        "embedding": _env_or_default("AFKVISION_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        "class_schema": _env_or_default("AFKVISION_CLASS_SCHEMA", DEFAULT_CLASS_SCHEMA),
        "schema_labels": _csv_env_or_default("AFKVISION_SCHEMA_LABELS", DEFAULT_SCHEMA_LABELS),
    }


def validate_runtime_configuration(require_explainer: bool = False) -> dict[str, str]:
    versions = get_runtime_model_versions()

    inference_health = _http_get_json(
        f"{_base_url('AFKVISION_INFERENCE_BASE_URL', LOCAL_INFERENCE_DEFAULT_URL)}/health"
    )
    if not inference_health.get("ready"):
        raise ProviderConfigurationError("Local inference service is up but not ready.")

    ollama_base_url = _base_url("AFKVISION_OLLAMA_BASE_URL", OLLAMA_DEFAULT_URL)
    model_name = versions["explainer"]
    try:
        tags = _http_get_json(f"{ollama_base_url}/api/tags")
        models = tags.get("models", [])
        available_names = {
            str(item.get("model") or item.get("name")).strip()
            for item in models
            if isinstance(item, dict)
        }
    except ProviderConfigurationError:
        available_names = set()
        versions["explainer_status"] = "unreachable"
        if require_explainer:
            raise
        return versions

    if model_name in available_names or f"{model_name}:latest" in available_names:
        versions["explainer_status"] = "ready"
        return versions

    versions["explainer_status"] = "pending_install"
    if require_explainer:
        raise ProviderConfigurationError(
            f"Ollama model '{model_name}' is not installed. Run `docker compose up ollama-pull` first."
        )

    return versions
