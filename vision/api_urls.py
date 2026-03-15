from rest_framework.routers import DefaultRouter
from django.urls import path

from .api_views import DetectionDetailView, ImageDetectionsView, ProjectViewSet, RunDetailView, SimilarSearchView

router = DefaultRouter()
router.register("projects", ProjectViewSet, basename="project")

urlpatterns = [
    *router.urls,
    path("images/<int:image_id>/detections/", ImageDetectionsView.as_view(), name="image-detections"),
    path("detections/<int:pk>/", DetectionDetailView.as_view(), name="detection-detail"),
    path("runs/<int:pk>/", RunDetailView.as_view(), name="run-detail"),
    path("search/similar/", SimilarSearchView.as_view(), name="similar-search"),
]
