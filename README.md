# AFK Vision

AFK Vision is a Django/DRF/Celery crop-and-weed scouting workbench built around a fully local inference path. It takes large field images, tiles them, runs CPU weed/crop detection, verifies selected low-confidence findings with a local VLM, stores CLIP embeddings in `pgvector`, and exposes a reviewer UI plus APIs.

## Runtime

- Detector: `NvMayMay/weedblaster-vision-yolov8s:best.pt` in a local CPU inference container
- Explainer: `moondream` through `Ollama`
- Embeddings: `openai/clip-vit-base-patch32` in the same local inference container
- Queueing: `Celery + Redis`
- Storage/indexing: `PostgreSQL + pgvector`
- Target schema: CropAndWeed `CropsOrWeed9`

## CropAndWeed demo path

The app is now wired for a CropAndWeed/WeedBlaster scouting taxonomy:

- `maize`
- `sugar_beet`
- `soy`
- `sunflower`
- `potato`
- `pea`
- `bean`
- `pumpkin`
- `weed`

The default detector path now points at the public WeedBlaster YOLOv8 weights from Hugging Face. This is a far better agriculture demo than generic COCO YOLO, but it is still best suited to close field imagery and robot/top-down scouting views, not broad orthomosaic disease mapping.

## Local setup

```bash
cd afk-vision
cp .env.example .env
docker compose up -d db redis inference ollama
docker compose up ollama-pull
uv sync
uv run python manage.py migrate
uv run python manage.py runserver 8001
```

In a second terminal:

```bash
cd afk-vision
uv run celery -A config worker -l info
```

Open `http://127.0.0.1:8001/`.

The first launch downloads the YOLO, CLIP, and Ollama models, so expect startup to take a while.

## One-command startup

If the machine already has `Docker`, `docker compose`, `curl`, and `uv`, a reviewer can bring up the full local suite with:

```bash
cd afk-vision
./run.sh
```

That script:

- starts `PostgreSQL`, `Redis`, `Ollama`, and the local `inference` container
- installs Python dependencies with `uv`
- runs Django migrations
- starts the Django server on `8001`
- starts the Celery worker

Companion commands:

```bash
./status.sh
./stop.sh
```

## Environment

The defaults in `.env.example` are local-first and work with the compose file:

```bash
AFKVISION_INFERENCE_BASE_URL=http://127.0.0.1:8091
AFKVISION_DETECTOR_MODEL=yolov8n.pt
AFKVISION_DETECTOR_MODEL_REPO=NvMayMay/weedblaster-vision-yolov8s
AFKVISION_DETECTOR_MODEL_FILE=best.pt
AFKVISION_EMBEDDING_MODEL=openai/clip-vit-base-patch32
AFKVISION_OLLAMA_BASE_URL=http://127.0.0.1:11434
AFKVISION_OLLAMA_MODEL=moondream
AFKVISION_CLASS_SCHEMA=cropandweed_9
AFKVISION_SCHEMA_LABELS=maize,sugar_beet,soy,sunflower,potato,pea,bean,pumpkin,weed
AFKVISION_ALLOWED_LABELS=
AFKVISION_LABEL_ALIASES_JSON=
```

You can tune CPU behavior with:

```bash
AFKVISION_DETECTOR_CONFIDENCE=0.2
AFKVISION_DETECTOR_IMAGE_SIZE=1024
AFKVISION_INFERENCE_THREADS=4
AFKVISION_TILE_SIZE=1024
AFKVISION_TILE_OVERLAP=128
```

## Local inference service

The Dockerized local inference service lives in [infra/local_inference/app.py](/home/kiran/AFK/afk-vision/infra/local_inference/app.py) and exposes:

- `GET /health`
- `POST /detect`
- `POST /embed`

The detector returns global-image bounding boxes by remapping YOLO tile detections back into the original image space.

For a custom agriculture model, the most relevant detector env vars are:

- `AFKVISION_DETECTOR_MODEL_REPO`
- `AFKVISION_DETECTOR_MODEL_FILE`
- `AFKVISION_CLASS_SCHEMA`
- `AFKVISION_SCHEMA_LABELS`
- `AFKVISION_ALLOWED_LABELS`
- `AFKVISION_LABEL_ALIASES_JSON`

`AFKVISION_ALLOWED_LABELS` is optional and should usually stay empty until you want to hard-filter the detector output. The built-in default alias map normalizes WeedBlaster class names like `Sugar Beet` to `sugar_beet`.

## Optional demo seed

```bash
uv run python manage.py seed_demo
```

That generates a synthetic field image you can upload to verify the local pipeline without hunting for sample imagery first.

## API surface

- `POST /api/projects/`
- `POST /api/projects/{id}/images/`
- `POST /api/projects/{id}/runs/`
- `GET /api/runs/{id}/`
- `GET /api/images/{id}/detections/`
- `PATCH /api/detections/{id}/`
- `POST /api/search/similar/`
