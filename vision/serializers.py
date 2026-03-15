from rest_framework import serializers

from .models import Detection, ImageAsset, InferenceRun, Project


class ProjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = Project
        fields = [
            "id",
            "name",
            "description",
            "status",
            "classes",
            "created_at",
            "updated_at",
        ]


class ImageAssetSerializer(serializers.ModelSerializer):
    image_url = serializers.SerializerMethodField()

    class Meta:
        model = ImageAsset
        fields = [
            "id",
            "project",
            "image_url",
            "filename",
            "width",
            "height",
            "exif_metadata",
            "georeference",
            "created_at",
        ]

    def get_image_url(self, obj: ImageAsset) -> str:
        request = self.context.get("request")
        if not request:
            return obj.image.url
        return request.build_absolute_uri(obj.image.url)


class InferenceRunSerializer(serializers.ModelSerializer):
    class Meta:
        model = InferenceRun
        fields = [
            "id",
            "project",
            "image",
            "status",
            "current_stage",
            "progress",
            "error_message",
            "model_versions",
            "metrics",
            "created_at",
            "started_at",
            "completed_at",
        ]


class DetectionSerializer(serializers.ModelSerializer):
    crop_url = serializers.SerializerMethodField()

    class Meta:
        model = Detection
        fields = [
            "id",
            "image",
            "run",
            "label",
            "confidence",
            "pixel_bbox",
            "normalized_bbox",
            "geo_bbox",
            "tile_bbox",
            "review_state",
            "verification_state",
            "explanation",
            "reviewer_notes",
            "crop_url",
            "created_at",
            "updated_at",
        ]

    def get_crop_url(self, obj: Detection) -> str | None:
        if not obj.crop:
            return None
        request = self.context.get("request")
        if not request:
            return obj.crop.url
        return request.build_absolute_uri(obj.crop.url)
