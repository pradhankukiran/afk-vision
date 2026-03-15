from vision.services import DetectionCandidate, iter_tiles, iou, merge_candidates, normalize_bbox


def test_iter_tiles_covers_edges():
    tiles = iter_tiles(4000, 3000, tile_size=1024, overlap=128)
    assert tiles[0].x == 0
    assert tiles[0].y == 0
    assert max(tile.x + tile.width for tile in tiles) == 4000
    assert max(tile.y + tile.height for tile in tiles) == 3000


def test_merge_candidates_collapses_overlap():
    merged = merge_candidates(
        [
            DetectionCandidate(
                label="disease_hotspot",
                confidence=0.8,
                bbox={"x1": 100, "y1": 100, "x2": 200, "y2": 200},
                tile_bbox={},
            ),
            DetectionCandidate(
                label="disease_hotspot",
                confidence=0.7,
                bbox={"x1": 105, "y1": 105, "x2": 210, "y2": 210},
                tile_bbox={},
            ),
        ]
    )
    assert len(merged) == 1
    assert 100 <= merged[0].bbox["x1"] <= 105
    assert 200 <= merged[0].bbox["x2"] <= 210


def test_normalize_bbox():
    normalized = normalize_bbox({"x1": 200, "y1": 100, "x2": 600, "y2": 300}, width=1000, height=500)
    assert normalized == {"x1": 0.2, "y1": 0.2, "x2": 0.6, "y2": 0.6}
