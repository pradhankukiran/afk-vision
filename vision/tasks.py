from __future__ import annotations

import io
import logging
import os
from pathlib import Path

from celery import shared_task
from django.core.files.base import ContentFile
from django.db import transaction
from django.utils import timezone
from PIL import Image

from .models import Detection, EmbeddingRecord, InferenceRun, ProjectStatus, RunStatus, VerificationState
from .providers import get_detector_provider, get_embedding_provider, get_explanation_provider
from .services import crop_image, iter_tiles, load_image_bytes, merge_candidates, normalize_bbox

logger = logging.getLogger(__name__)
EXPLANATION_CONFIDENCE_THRESHOLD = float(os.getenv("AFKVISION_EXPLANATION_CONFIDENCE_THRESHOLD", "0.40"))
EXPLANATION_MAX_DETECTIONS = int(os.getenv("AFKVISION_EXPLANATION_MAX_DETECTIONS", "1"))


@shared_task(bind=True, max_retries=3, autoretry_for=(Exception,), retry_backoff=5)
def process_inference_run(self, run_id: int) -> dict[str, str | int]:
    run = InferenceRun.objects.select_related("project", "image").get(pk=run_id)
    image_asset = run.image
    detector = get_detector_provider()
    explainer = get_explanation_provider()
    embedder = get_embedding_provider()

    run.status = RunStatus.RUNNING
    run.current_stage = "extract_metadata"
    run.progress = 10
    run.started_at = timezone.now()
    run.save(update_fields=["status", "current_stage", "progress", "started_at"])

    image_bytes = load_image_bytes(image_asset.image)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image_asset.width, image_asset.height = image.size
    image_asset.save(update_fields=["width", "height"])

    run.current_stage = "tiling"
    run.progress = 25
    run.save(update_fields=["current_stage", "progress"])

    candidates = []
    tiles = iter_tiles(image.width, image.height)
    for tile in tiles:
        tile_image = image.crop((tile.x, tile.y, tile.x + tile.width, tile.y + tile.height))
        payload = io.BytesIO()
        tile_image.save(payload, format="PNG")
        candidates.extend(detector.detect(payload.getvalue(), tile))

    run.current_stage = "merge"
    run.progress = 55
    run.save(update_fields=["current_stage", "progress"])

    merged = merge_candidates(candidates)

    run.current_stage = "verify"
    run.progress = 75
    run.save(update_fields=["current_stage", "progress"])

    with transaction.atomic():
        Detection.objects.filter(run=run).delete()
        created = []
        explained_count = 0
        for candidate in merged:
            crop = crop_image(image, candidate.bbox)
            crop_bytes = io.BytesIO()
            crop.save(crop_bytes, format="PNG")
            crop_payload = crop_bytes.getvalue()
            should_explain = (
                candidate.confidence < EXPLANATION_CONFIDENCE_THRESHOLD
                and explained_count < EXPLANATION_MAX_DETECTIONS
            )
            if should_explain:
                try:
                    verification_state, explanation = explainer.explain(
                        crop_payload,
                        candidate.label,
                        candidate.confidence,
                    )
                    explained_count += 1
                except Exception as exc:
                    logger.warning("VLM verification skipped for detection %s: %s", candidate.label, exc)
                    verification_state = VerificationState.NOT_RUN
                    explanation = "VLM verification unavailable for this detection."
            else:
                verification_state = VerificationState.NOT_RUN
                explanation = ""
            detection = Detection.objects.create(
                image=image_asset,
                run=run,
                label=candidate.label,
                confidence=round(candidate.confidence, 4),
                pixel_bbox=candidate.bbox,
                normalized_bbox=normalize_bbox(candidate.bbox, image.width, image.height),
                tile_bbox=candidate.tile_bbox,
                verification_state=verification_state,
                explanation=explanation,
            )
            crop_name = f"{Path(image_asset.filename).stem}-{detection.id}.png"
            detection.crop.save(crop_name, ContentFile(crop_payload), save=True)
            created.append(detection)

        run.current_stage = "embed"
        run.progress = 90
        run.save(update_fields=["current_stage", "progress"])

        for detection in created:
            detection.crop.open("rb")
            try:
                embedding = embedder.embed_image(detection.crop.read())
                EmbeddingRecord.objects.update_or_create(
                    detection=detection,
                    defaults={"embedding": embedding},
                )
            finally:
                detection.crop.close()

    run.status = RunStatus.SUCCEEDED
    run.current_stage = "completed"
    run.progress = 100
    run.completed_at = timezone.now()
    run.metrics = {
        "tiles_processed": len(tiles),
        "raw_candidates": len(candidates),
        "merged_detections": len(merged),
    }
    run.save(update_fields=["status", "current_stage", "progress", "completed_at", "metrics"])

    project = run.project
    project.status = ProjectStatus.READY
    project.classes = sorted({detection.label for detection in Detection.objects.filter(run=run)})
    project.save(update_fields=["status", "classes", "updated_at"])

    return {"run_id": run.id, "detections": len(merged)}
