<p align="center">
  <img src="https://img.icons8.com/color/96/visual-search.png" alt="AFK Vision Logo" width="80" />
</p>

<h1 align="center">AFK Vision</h1>

<p align="center">
  <strong>Local-first crop & weed detection workbench powered by YOLOv8, CLIP, and Moondream</strong>
</p>

<p align="center">
  <a href="https://github.com/pradhankukiran/afk-vision/blob/main/LICENSE"><img src="https://img.shields.io/github/license/pradhankukiran/afk-vision?style=flat-square&color=blue" alt="License"></a>
  <a href="https://github.com/pradhankukiran/afk-vision/releases"><img src="https://img.shields.io/github/v/release/pradhankukiran/afk-vision?style=flat-square&color=green" alt="Release"></a>
  <a href="https://github.com/pradhankukiran/afk-vision/stargazers"><img src="https://img.shields.io/github/stars/pradhankukiran/afk-vision?style=flat-square" alt="Stars"></a>
  <a href="https://github.com/pradhankukiran/afk-vision/issues"><img src="https://img.shields.io/github/issues/pradhankukiran/afk-vision?style=flat-square" alt="Issues"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Django-092E20?style=for-the-badge&logo=django&logoColor=white" alt="Django">
  <img src="https://img.shields.io/badge/DRF-A30000?style=for-the-badge&logo=django&logoColor=white" alt="DRF">
  <img src="https://img.shields.io/badge/Celery-37814A?style=for-the-badge&logo=celery&logoColor=white" alt="Celery">
  <img src="https://img.shields.io/badge/YOLOv8-111111?style=for-the-badge&logo=yolo&logoColor=white" alt="YOLOv8">
  <img src="https://img.shields.io/badge/CLIP-412991?style=for-the-badge&logo=openai&logoColor=white" alt="CLIP">
  <img src="https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white" alt="Ollama">
  <img src="https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white" alt="PostgreSQL">
  <img src="https://img.shields.io/badge/pgvector-4169E1?style=for-the-badge&logo=postgresql&logoColor=white" alt="pgvector">
  <img src="https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=redis&logoColor=white" alt="Redis">
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
</p>

---

## What is AFK Vision?

AFK Vision is a Django-based crop-and-weed scouting workbench built around a fully local inference path. It takes large field images, tiles them, runs CPU weed/crop detection with YOLOv8, verifies low-confidence findings with a local VLM (Moondream via Ollama), stores CLIP embeddings in pgvector for similarity search, and exposes a reviewer UI plus REST APIs.

### Key Features

- **Tiled Detection** -- Splits large field images into tiles and runs YOLOv8 inference on CPU
- **VLM Verification** -- Low-confidence detections are verified by Moondream through Ollama
- **Similarity Search** -- CLIP embeddings stored in pgvector for visual similarity queries
- **Reviewer UI** -- Built-in web interface for reviewing and correcting detections
- **Async Processing** -- Celery + Redis for background task queuing
- **Fully Local** -- No cloud APIs required, everything runs on your machine

---

## Architecture

```text
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  Django App  │────>│  Celery      │────>│  Inference     │
│  (DRF API)  │     │  (Redis)     │     │  (YOLOv8+CLIP) │
└──────┬──────┘     └──────────────┘     └───────────────┘
       │                                         │
       v                                         v
┌──────────────┐                         ┌───────────────┐
│  PostgreSQL  │                         │  Ollama       │
│  + pgvector  │                         │  (Moondream)  │
└──────────────┘                         └───────────────┘
```

| Component | Stack |
|---|---|
| **Web / API** | Django, Django REST Framework |
| **Task Queue** | Celery, Redis |
| **Detector** | YOLOv8 (`weedblaster-vision-yolov8s`) |
| **Embeddings** | OpenAI CLIP (`clip-vit-base-patch32`) |
| **Explainer** | Moondream via Ollama |
| **Storage** | PostgreSQL + pgvector |
| **Infra** | Docker Compose, uv |

---

## Detection Taxonomy

Built around the CropAndWeed / WeedBlaster scouting schema (`CropsOrWeed9`):

`maize` | `sugar_beet` | `soy` | `sunflower` | `potato` | `pea` | `bean` | `pumpkin` | `weed`

> Best suited for close field imagery and robot/top-down scouting views, not broad orthomosaic disease mapping.

---

## Getting Started

### Prerequisites

- Docker + Docker Compose
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- curl

### Quick Start

```bash
git clone https://github.com/pradhankukiran/afk-vision.git
cd afk-vision
cp .env.example .env
./run.sh
```

That script starts PostgreSQL, Redis, Ollama, the inference container, installs Python deps, runs migrations, and launches the Django server + Celery worker.

Open [localhost:8001](http://localhost:8001).

> First launch downloads YOLOv8, CLIP, and Ollama models -- expect startup to take a while.

### Manual Setup

```bash
# Start infrastructure
docker compose up -d db redis inference ollama
docker compose up ollama-pull

# Install deps + migrate
uv sync
uv run python manage.py migrate

# Start server
uv run python manage.py runserver 8001

# In a second terminal — start worker
uv run celery -A config worker -l info
```

### Companion Scripts

| Script | Description |
|---|---|
| `./run.sh` | Start everything |
| `./status.sh` | Check service status |
| `./stop.sh` | Stop all services |

---

## Environment Variables

### Core

| Variable | Default | Description |
|---|---|---|
| `AFKVISION_INFERENCE_BASE_URL` | `http://127.0.0.1:8091` | Inference container URL |
| `AFKVISION_DETECTOR_MODEL` | `yolov8n.pt` | Detector model name |
| `AFKVISION_DETECTOR_MODEL_REPO` | `NvMayMay/weedblaster-vision-yolov8s` | HuggingFace model repo |
| `AFKVISION_DETECTOR_MODEL_FILE` | `best.pt` | Model weights filename |
| `AFKVISION_EMBEDDING_MODEL` | `openai/clip-vit-base-patch32` | CLIP embedding model |
| `AFKVISION_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama endpoint |
| `AFKVISION_OLLAMA_MODEL` | `moondream` | VLM model for verification |
| `AFKVISION_CLASS_SCHEMA` | `cropandweed_9` | Detection class schema |
| `AFKVISION_SCHEMA_LABELS` | `maize,sugar_beet,...,weed` | Comma-separated class labels |

### Tuning

| Variable | Default | Description |
|---|---|---|
| `AFKVISION_DETECTOR_CONFIDENCE` | `0.2` | Detection confidence threshold |
| `AFKVISION_DETECTOR_IMAGE_SIZE` | `1024` | Inference image size |
| `AFKVISION_INFERENCE_THREADS` | `4` | CPU inference threads |
| `AFKVISION_TILE_SIZE` | `1024` | Tile size in pixels |
| `AFKVISION_TILE_OVERLAP` | `128` | Tile overlap in pixels |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/projects/` | Create a project |
| `POST` | `/api/projects/{id}/images/` | Upload images |
| `POST` | `/api/projects/{id}/runs/` | Start detection run |
| `GET` | `/api/runs/{id}/` | Get run status |
| `GET` | `/api/images/{id}/detections/` | List detections for image |
| `PATCH` | `/api/detections/{id}/` | Update a detection |
| `POST` | `/api/search/similar/` | Similarity search via CLIP |

---

## License

[MIT](LICENSE)
