from __future__ import annotations

from pgvector.django import CosineDistance
from rest_framework import generics, parsers, status, viewsets
from rest_framework.decorators import action
from rest_framework.request import Request
from rest_framework.response import Response

from .models import Detection, EmbeddingRecord, ImageAsset, InferenceRun, Project, ReviewState
from .providers import (
    ProviderConfigurationError,
    get_embedding_provider,
    validate_runtime_configuration,
)
from .serializers import DetectionSerializer, ImageAssetSerializer, InferenceRunSerializer, ProjectSerializer
from .tasks import process_inference_run
from .services import cosine_similarity


class ProjectViewSet(viewsets.ModelViewSet):
    queryset = Project.objects.order_by("-created_at")
    serializer_class = ProjectSerializer

    @action(detail=True, methods=["post"], parser_classes=[parsers.MultiPartParser, parsers.FormParser])
    def images(self, request: Request, pk: int | None = None) -> Response:
        project = self.get_object()
        upload = request.FILES.get("image")
        if upload is None:
            return Response({"detail": "image is required"}, status=status.HTTP_400_BAD_REQUEST)

        image_asset = ImageAsset.objects.create(project=project, image=upload)
        serializer = ImageAssetSerializer(image_asset, context={"request": request})
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    @action(detail=True, methods=["post"])
    def runs(self, request: Request, pk: int | None = None) -> Response:
        project = self.get_object()
        image_id = request.data.get("image_id")
        image = generics.get_object_or_404(ImageAsset, pk=image_id, project=project)
        try:
            model_versions = validate_runtime_configuration()
        except ProviderConfigurationError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
        project.status = "processing"
        project.save(update_fields=["status", "updated_at"])
        run = InferenceRun.objects.create(
            project=project,
            image=image,
            model_versions=model_versions,
        )
        process_inference_run.delay(run.id)
        serializer = InferenceRunSerializer(run)
        return Response(serializer.data, status=status.HTTP_201_CREATED)


class ImageDetectionsView(generics.ListAPIView):
    serializer_class = DetectionSerializer

    def get_queryset(self):
        image_id = self.kwargs["image_id"]
        return Detection.objects.filter(image_id=image_id).order_by("-confidence", "-created_at")


class DetectionDetailView(generics.UpdateAPIView):
    queryset = Detection.objects.all()
    serializer_class = DetectionSerializer
    http_method_names = ["patch"]

    def partial_update(self, request, *args, **kwargs):
        detection = self.get_object()
        review_state = request.data.get("review_state")
        notes = request.data.get("reviewer_notes", "")
        label = request.data.get("label")

        if review_state in ReviewState.values:
            detection.review_state = review_state
        if label:
            detection.label = label
        detection.reviewer_notes = notes
        detection.save(update_fields=["review_state", "label", "reviewer_notes", "updated_at"])
        return Response(self.get_serializer(detection).data)


class RunDetailView(generics.RetrieveAPIView):
    queryset = InferenceRun.objects.select_related("image", "project")
    serializer_class = InferenceRunSerializer


class SimilarSearchView(generics.GenericAPIView):
    parser_classes = [parsers.MultiPartParser, parsers.FormParser]

    def post(self, request: Request) -> Response:
        detection_id = request.data.get("detection_id")
        limit = min(int(request.data.get("limit", 5)), 10)

        if detection_id:
            detection = generics.get_object_or_404(Detection, pk=detection_id)
            base_embedding = detection.embedding_record.embedding
        else:
            upload = request.FILES.get("image")
            if upload is None:
                return Response({"detail": "Provide detection_id or image"}, status=status.HTTP_400_BAD_REQUEST)
            try:
                base_embedding = get_embedding_provider().embed_image(upload.read())
            except ProviderConfigurationError as exc:
                return Response({"detail": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)

        queryset = EmbeddingRecord.objects.select_related("detection", "detection__image")
        if detection_id:
            queryset = queryset.exclude(detection_id=detection_id)

        try:
            matches = queryset.order_by(CosineDistance("embedding", base_embedding))[:limit]
            results = [
                {
                    "detection_id": record.detection_id,
                    "image_id": record.detection.image_id,
                    "label": record.detection.label,
                    "confidence": record.detection.confidence,
                    "crop_url": request.build_absolute_uri(record.detection.crop.url) if record.detection.crop else None,
                }
                for record in matches
            ]
        except Exception:
            # Fall back for non-Postgres test scenarios.
            scored = sorted(
                queryset,
                key=lambda record: cosine_similarity(base_embedding, record.embedding),
                reverse=True,
            )[:limit]
            results = [
                {
                    "detection_id": record.detection_id,
                    "image_id": record.detection.image_id,
                    "label": record.detection.label,
                    "confidence": record.detection.confidence,
                    "crop_url": request.build_absolute_uri(record.detection.crop.url) if record.detection.crop else None,
                }
                for record in scored
            ]

        return Response({"results": results})
