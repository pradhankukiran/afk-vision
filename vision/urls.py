from django.urls import path

from . import views

app_name = "vision"

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("projects/<int:pk>/", views.project_detail, name="project-detail"),
    path("projects/<int:pk>/images/<int:image_id>/run/", views.launch_run, name="launch-run"),
    path("projects/<int:pk>/detections/<int:detection_id>/review/", views.update_detection, name="update-detection"),
]
