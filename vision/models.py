from __future__ import annotations

from pathlib import Path

from django.db import models
from pgvector.django import VectorField


class ProjectStatus(models.TextChoices):
    CREATED = "created", "Created"
    PROCESSING = "processing", "Processing"
    READY = "ready", "Ready"
    FAILED = "failed", "Failed"


class RunStatus(models.TextChoices):
    QUEUED = "queued", "Queued"
    RUNNING = "running", "Running"
    SUCCEEDED = "succeeded", "Succeeded"
    FAILED = "failed", "Failed"


class ReviewState(models.TextChoices):
    PENDING = "pending", "Pending"
    CONFIRMED = "confirmed", "Confirmed"
    REJECTED = "rejected", "Rejected"
    RELABELED = "relabeled", "Relabeled"


class VerificationState(models.TextChoices):
    NOT_RUN = "not_run", "Not run"
    VERIFIED = "verified", "Verified"
    FLAGGED = "flagged", "Flagged"


class Project(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    status = models.CharField(max_length=32, choices=ProjectStatus.choices, default=ProjectStatus.CREATED)
    classes = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:
        return self.name


class ImageAsset(models.Model):
    project = models.ForeignKey(Project, related_name="images", on_delete=models.CASCADE)
    image = models.ImageField(upload_to="images/")
    width = models.PositiveIntegerField(default=0)
    height = models.PositiveIntegerField(default=0)
    exif_metadata = models.JSONField(default=dict, blank=True)
    georeference = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    @property
    def filename(self) -> str:
        return Path(self.image.name).name

    def __str__(self) -> str:
        return self.filename


class InferenceRun(models.Model):
    project = models.ForeignKey(Project, related_name="runs", on_delete=models.CASCADE)
    image = models.ForeignKey(ImageAsset, related_name="runs", on_delete=models.CASCADE)
    status = models.CharField(max_length=32, choices=RunStatus.choices, default=RunStatus.QUEUED)
    current_stage = models.CharField(max_length=64, default="queued")
    progress = models.PositiveIntegerField(default=0)
    error_message = models.TextField(blank=True)
    model_versions = models.JSONField(default=dict, blank=True)
    metrics = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self) -> str:
        return f"{self.project.name} / {self.image.filename} / {self.current_stage}"


class Detection(models.Model):
    image = models.ForeignKey(ImageAsset, related_name="detections", on_delete=models.CASCADE)
    run = models.ForeignKey(InferenceRun, related_name="detections", on_delete=models.CASCADE)
    label = models.CharField(max_length=128)
    confidence = models.FloatField()
    pixel_bbox = models.JSONField(default=dict)
    normalized_bbox = models.JSONField(default=dict)
    geo_bbox = models.JSONField(default=dict, blank=True)
    tile_bbox = models.JSONField(default=dict, blank=True)
    review_state = models.CharField(max_length=32, choices=ReviewState.choices, default=ReviewState.PENDING)
    verification_state = models.CharField(max_length=32, choices=VerificationState.choices, default=VerificationState.NOT_RUN)
    explanation = models.TextField(blank=True)
    reviewer_notes = models.TextField(blank=True)
    crop = models.ImageField(upload_to="crops/", blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:
        return f"{self.label} ({self.confidence:.2f})"


class EmbeddingRecord(models.Model):
    detection = models.OneToOneField(Detection, related_name="embedding_record", on_delete=models.CASCADE)
    embedding = VectorField(dimensions=512)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return f"Embedding for {self.detection_id}"
