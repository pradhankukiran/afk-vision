from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from django.core.files.base import ContentFile
from django.core.management.base import BaseCommand
from PIL import Image, ImageDraw, ImageFilter

from vision.models import ImageAsset, Project


class Command(BaseCommand):
    help = "Create a synthetic AFK Vision demo project with one crop-and-weed scouting image."

    def handle(self, *args, **options):
        project, _ = Project.objects.get_or_create(
            name="Monsoon East Demo",
            defaults={"description": "Synthetic aerial field with anomaly patches for AFK Vision demos."},
        )

        if project.images.exists():
            self.stdout.write(self.style.WARNING("Demo project already has images; skipping image generation."))
            return

        canvas = self._generate_image()
        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")
        image_name = "monsoon-east-demo.png"
        image_asset = ImageAsset(project=project)
        image_asset.image.save(image_name, ContentFile(buffer.getvalue()), save=False)
        image_asset.save()
        image_asset.width, image_asset.height = canvas.size
        image_asset.save(update_fields=["width", "height"])
        self.stdout.write(self.style.SUCCESS(f"Created demo project {project.pk} with image {image_asset.filename}."))

    def _generate_image(self) -> Image.Image:
        width, height = 4096, 3072
        rng = np.random.default_rng(42)
        base = np.zeros((height, width, 3), dtype=np.uint8)
        base[:, :, 0] = rng.integers(42, 74, size=(height, width))
        base[:, :, 1] = rng.integers(92, 148, size=(height, width))
        base[:, :, 2] = rng.integers(38, 70, size=(height, width))

        image = Image.fromarray(base, mode="RGB").filter(ImageFilter.GaussianBlur(radius=2.5))
        draw = ImageDraw.Draw(image, "RGBA")
        draw.rectangle((430, 420, 1180, 1090), fill=(222, 104, 79, 245))
        draw.rectangle((1860, 700, 2680, 1410), fill=(91, 212, 255, 255))
        draw.rectangle((2940, 1180, 3580, 1760), fill=(252, 179, 3, 240))
        draw.rectangle((1460, 2020, 2140, 2600), fill=(230, 214, 156, 238))
        draw.rectangle((760, 1880, 1120, 2240), fill=(222, 104, 79, 200))
        return image
