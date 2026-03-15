from django.contrib import admin

from .models import Detection, EmbeddingRecord, ImageAsset, InferenceRun, Project


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ("name", "status", "created_at")


@admin.register(ImageAsset)
class ImageAssetAdmin(admin.ModelAdmin):
    list_display = ("project", "filename", "width", "height", "created_at")


@admin.register(InferenceRun)
class InferenceRunAdmin(admin.ModelAdmin):
    list_display = ("project", "image", "status", "current_stage", "progress", "created_at")


@admin.register(Detection)
class DetectionAdmin(admin.ModelAdmin):
    list_display = ("label", "confidence", "review_state", "verification_state", "created_at")


@admin.register(EmbeddingRecord)
class EmbeddingRecordAdmin(admin.ModelAdmin):
    list_display = ("detection", "created_at")
