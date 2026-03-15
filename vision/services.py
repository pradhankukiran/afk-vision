from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from django.conf import settings
from PIL import Image


@dataclass(slots=True)
class Tile:
    x: int
    y: int
    width: int
    height: int


@dataclass(slots=True)
class DetectionCandidate:
    label: str
    confidence: float
    bbox: dict[str, float]
    tile_bbox: dict[str, int]


def iter_tiles(width: int, height: int, tile_size: int | None = None, overlap: int | None = None) -> list[Tile]:
    tile_size = tile_size or settings.AFKVISION_TILE_SIZE
    overlap = overlap or settings.AFKVISION_TILE_OVERLAP
    step = max(1, tile_size - overlap)

    x_positions = _axis_positions(width, tile_size, step)
    y_positions = _axis_positions(height, tile_size, step)

    return [
        Tile(
            x=x,
            y=y,
            width=min(tile_size, width - x),
            height=min(tile_size, height - y),
        )
        for y in y_positions
        for x in x_positions
    ]


def _axis_positions(length: int, tile_size: int, step: int) -> list[int]:
    if length <= tile_size:
        return [0]

    positions: list[int] = []
    cursor = 0
    while cursor + tile_size < length:
        positions.append(cursor)
        cursor += step
    positions.append(length - tile_size)
    return sorted(set(positions))


def normalize_bbox(bbox: dict[str, float], width: int, height: int) -> dict[str, float]:
    return {
        "x1": round(bbox["x1"] / width, 6),
        "y1": round(bbox["y1"] / height, 6),
        "x2": round(bbox["x2"] / width, 6),
        "y2": round(bbox["y2"] / height, 6),
    }


def merge_candidates(candidates: Iterable[DetectionCandidate], iou_threshold: float = 0.35) -> list[DetectionCandidate]:
    grouped: dict[str, list[DetectionCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.label, []).append(candidate)

    merged: list[DetectionCandidate] = []
    for group in grouped.values():
        ordered = sorted(group, key=lambda item: item.confidence, reverse=True)
        while ordered:
            current = ordered.pop(0)
            overlaps = [item for item in ordered if iou(current.bbox, item.bbox) >= iou_threshold]
            ordered = [item for item in ordered if iou(current.bbox, item.bbox) < iou_threshold]
            if overlaps:
                merged.append(_weighted_merge([current, *overlaps]))
            else:
                merged.append(current)
    return merged


def _weighted_merge(items: list[DetectionCandidate]) -> DetectionCandidate:
    total = sum(item.confidence for item in items)
    bbox = {
        "x1": sum(item.bbox["x1"] * item.confidence for item in items) / total,
        "y1": sum(item.bbox["y1"] * item.confidence for item in items) / total,
        "x2": sum(item.bbox["x2"] * item.confidence for item in items) / total,
        "y2": sum(item.bbox["y2"] * item.confidence for item in items) / total,
    }
    return DetectionCandidate(
        label=items[0].label,
        confidence=max(item.confidence for item in items),
        bbox=bbox,
        tile_bbox=items[0].tile_bbox,
    )


def iou(a: dict[str, float], b: dict[str, float]) -> float:
    left = max(a["x1"], b["x1"])
    top = max(a["y1"], b["y1"])
    right = min(a["x2"], b["x2"])
    bottom = min(a["y2"], b["y2"])
    if right <= left or bottom <= top:
        return 0.0

    intersection = (right - left) * (bottom - top)
    area_a = max(1.0, (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]))
    area_b = max(1.0, (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]))
    union = area_a + area_b - intersection
    return intersection / union


def crop_image(image: Image.Image, bbox: dict[str, float]) -> Image.Image:
    return image.crop((bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot_product = sum(left * right for left, right in zip(a, b))
    norm_a = sum(value * value for value in a) ** 0.5
    norm_b = sum(value * value for value in b) ** 0.5
    denom = norm_a * norm_b
    if denom == 0:
        return 0.0
    return float(dot_product / denom)


def load_image_bytes(field_file) -> bytes:
    field_file.open("rb")
    try:
        return field_file.read()
    finally:
        field_file.close()
