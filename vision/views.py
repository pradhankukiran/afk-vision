from __future__ import annotations

from django.contrib import messages
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_POST

from .forms import ProjectForm, UploadImageForm
from .models import Detection, ImageAsset, InferenceRun, Project, ReviewState
from .providers import ProviderConfigurationError, validate_runtime_configuration
from .tasks import process_inference_run


def dashboard(request: HttpRequest) -> HttpResponse:
    if request.method == "POST":
        form = ProjectForm(request.POST)
        if form.is_valid():
            project = form.save()
            return redirect("vision:project-detail", pk=project.pk)
    else:
        form = ProjectForm()

    projects = Project.objects.order_by("-created_at")
    return render(request, "vision/dashboard.html", {"form": form, "projects": projects})


def project_detail(request: HttpRequest, pk: int) -> HttpResponse:
    project = get_object_or_404(
        Project.objects.prefetch_related("images", "runs", "runs__detections", "images__detections"),
        pk=pk,
    )

    if request.method == "POST" and "upload_image" in request.POST:
        upload_form = UploadImageForm(request.POST, request.FILES)
        if upload_form.is_valid():
            image_asset = ImageAsset.objects.create(project=project, image=upload_form.cleaned_data["image"])
            messages.success(request, f"Uploaded scouting image {image_asset.filename}.")
            return redirect("vision:project-detail", pk=pk)
    else:
        upload_form = UploadImageForm()

    selected_image = project.images.order_by("-created_at").first()
    if request.GET.get("image"):
        selected_image = get_object_or_404(ImageAsset, pk=request.GET["image"], project=project)

    latest_run = project.runs.order_by("-created_at").first()
    detections = selected_image.detections.order_by("-confidence") if selected_image else Detection.objects.none()
    selected_detection = detections.first() if selected_image else None
    if selected_image and request.GET.get("detection"):
        selected_detection = get_object_or_404(Detection, pk=request.GET["detection"], image=selected_image)

    return render(
        request,
        "vision/project_detail.html",
        {
            "project": project,
            "selected_image": selected_image,
            "latest_run": latest_run,
            "detections": detections,
            "selected_detection": selected_detection,
            "upload_form": upload_form,
        },
    )


@require_POST
def launch_run(request: HttpRequest, pk: int, image_id: int) -> HttpResponse:
    project = get_object_or_404(Project, pk=pk)
    image = get_object_or_404(ImageAsset, pk=image_id, project=project)
    try:
        model_versions = validate_runtime_configuration()
    except ProviderConfigurationError as exc:
        messages.error(request, str(exc))
        return redirect(f"{reverse('vision:project-detail', kwargs={'pk': pk})}?image={image_id}")
    if model_versions.get("explainer_status") != "ready":
        messages.warning(
            request,
            "Crop and weed detection are live. VLM verification will start once the local Ollama model finishes installing.",
        )
    project.status = "processing"
    project.save(update_fields=["status", "updated_at"])
    run = InferenceRun.objects.create(
        project=project,
        image=image,
        model_versions=model_versions,
    )
    process_inference_run.delay(run.id)
    messages.info(request, "Scouting run queued.")
    return redirect(f"{reverse('vision:project-detail', kwargs={'pk': pk})}?image={image_id}")


@require_POST
def update_detection(request: HttpRequest, pk: int, detection_id: int) -> HttpResponse:
    project = get_object_or_404(Project, pk=pk)
    detection = get_object_or_404(Detection, pk=detection_id, image__project=project)
    review_state = request.POST.get("review_state", ReviewState.PENDING)
    if review_state in ReviewState.values:
        detection.review_state = review_state
    detection.reviewer_notes = request.POST.get("reviewer_notes", "")
    detection.save(update_fields=["review_state", "reviewer_notes", "updated_at"])
    return redirect(
        f"{reverse('vision:project-detail', kwargs={'pk': pk})}?image={detection.image_id}&detection={detection.id}"
    )
