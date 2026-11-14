from contextlib import asynccontextmanager
from pathlib import Path
from io import BytesIO
import hashlib
import json
import os
import time

import numpy as np
import torch
from PIL import Image, ImageOps
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from transformers import (
    AutoImageProcessor,
    AutoModel,
    BlipForQuestionAnswering,
    BlipProcessor,
    CLIPModel,
    CLIPProcessor,
)

import retrieval


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
REFERENCE_DIR = DATASET_DIR / "reference_images"   # the searchable (bounded) index
VAL_DIR = DATASET_DIR / "val_queries"              # used ONLY to calibrate UNKNOWN
TEST_DIR = DATASET_DIR / "test_queries"            # used ONLY for final evaluation

# Choose the embedding model with an environment variable:
#   PowerShell:  $env:QA_MODEL = "dinov2"      (default: clip)
# Each model keeps its OWN index, thresholds and evaluation results, because
# similarity scores from different models are not comparable.
MODEL_NAMES = {
    "clip": "openai/clip-vit-base-patch32",
    "dinov2": "facebook/dinov2-base",
}
MODEL_KEY = os.environ.get("QA_MODEL", "clip").strip().lower()
if MODEL_KEY not in MODEL_NAMES:
    raise SystemExit(f"QA_MODEL must be one of {sorted(MODEL_NAMES)}, got '{MODEL_KEY}'")
EMBED_MODEL_NAME = MODEL_NAMES[MODEL_KEY]
EMBEDDINGS_DIR = BASE_DIR / "embeddings" / MODEL_KEY

# AI category recognition: CLIP zero-shot over the labels in backend/labels.txt
LABELS_PATH = BASE_DIR / "labels.txt"
CATEGORY_MODEL_NAME = MODEL_NAMES["clip"]
DEFAULT_LABELS = ["shoe", "handbag", "pen", "bottle", "building", "painting"]

THRESHOLDS_PATH = EMBEDDINGS_DIR / "thresholds.json"
EVAL_PATHS = [EMBEDDINGS_DIR / "evaluation_test.json",
              EMBEDDINGS_DIR / "evaluation_val.json"]

for folder in (REFERENCE_DIR, VAL_DIR, TEST_DIR, EMBEDDINGS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

VQA_MODEL_NAME = "Salesforce/blip-vqa-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@asynccontextmanager
async def lifespan(app):
    # Re-use the saved index if it still matches the dataset folder.
    load_index_from_disk()
    yield


app = FastAPI(title="QA Visual Retrieval and VQA API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/reference-images",
    StaticFiles(directory=str(REFERENCE_DIR)),
    name="reference-images",
)

embed_model = None
embed_processor = None
cat_model = None
cat_processor = None
_label_cache = None
vqa_model = None
vqa_processor = None
indexed_vectors = None
image_records = []


def load_embedding_model():
    global embed_model, embed_processor

    if embed_model is None or embed_processor is None:
        if MODEL_KEY == "clip":
            embed_processor = CLIPProcessor.from_pretrained(EMBED_MODEL_NAME)
            embed_model = CLIPModel.from_pretrained(EMBED_MODEL_NAME)
        else:  # dinov2
            embed_processor = AutoImageProcessor.from_pretrained(EMBED_MODEL_NAME)
            embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME)
        embed_model.to(DEVICE)
        embed_model.eval()

    return embed_model, embed_processor


def load_vqa_model():
    """Load BLIP only when the user first asks a question about an image."""
    global vqa_model, vqa_processor

    if vqa_model is None or vqa_processor is None:
        vqa_processor = BlipProcessor.from_pretrained(VQA_MODEL_NAME)
        vqa_model = BlipForQuestionAnswering.from_pretrained(VQA_MODEL_NAME)
        vqa_model.to(DEVICE)
        vqa_model.eval()

    return vqa_model, vqa_processor


def image_embedding(image: Image.Image) -> np.ndarray:
    current_model, current_processor = load_embedding_model()

    # exif_transpose: phone photos are often stored rotated.
    image = ImageOps.exif_transpose(image).convert("RGB")
    inputs = current_processor(images=image, return_tensors="pt")
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    with torch.no_grad():
        if MODEL_KEY == "clip":
            features = current_model.get_image_features(**inputs)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
        else:  # dinov2: the [CLS] token is the global image descriptor
            features = current_model(**inputs).last_hidden_state[:, 0]
        features = features / features.norm(dim=-1, keepdim=True)

    return features[0].cpu().numpy().astype(np.float32)


# --------------------------------------------- AI category recognition --
def category_labels():
    try:
        lines = [x.strip() for x in LABELS_PATH.read_text("utf-8").splitlines()]
        labels = [x for x in lines if x and not x.startswith("#")]
    except OSError:
        labels = []
    return labels or DEFAULT_LABELS


def labels_hash():
    return hashlib.sha1("\n".join(category_labels()).encode()).hexdigest()[:12]


def load_category_model():
    """CLIP names what is in the picture. If CLIP is also the embedding model it is shared."""
    global cat_model, cat_processor
    if MODEL_KEY == "clip":
        return load_embedding_model()
    if cat_model is None or cat_processor is None:
        cat_processor = CLIPProcessor.from_pretrained(CATEGORY_MODEL_NAME)
        cat_model = CLIPModel.from_pretrained(CATEGORY_MODEL_NAME)
        cat_model.to(DEVICE)
        cat_model.eval()
    return cat_model, cat_processor


def _normalise(features):
    if hasattr(features, "pooler_output"):
        features = features.pooler_output
    return features / features.norm(dim=-1, keepdim=True)


def clip_image_vector(image: Image.Image) -> np.ndarray:
    model, processor = load_category_model()
    image = ImageOps.exif_transpose(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
    with torch.no_grad():
        features = _normalise(model.get_image_features(**inputs))
    return features[0].cpu().numpy().astype(np.float32)


def label_text_features():
    """(labels, text embeddings) for the prompts 'a photo of a <label>'."""
    global _label_cache
    labels = tuple(category_labels())
    if _label_cache is None or _label_cache[0] != labels:
        model, processor = load_category_model()
        prompts = [f"a photo of a {label}" for label in labels]
        inputs = processor(text=prompts, return_tensors="pt", padding=True)
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
        with torch.no_grad():
            features = _normalise(model.get_text_features(**inputs))
        _label_cache = (labels, features.cpu().numpy().astype(np.float32))
    return _label_cache


def embed_and_classify(image: Image.Image):
    """Return (retrieval embedding, category probabilities, labels)."""
    vector = image_embedding(image)
    category_vector = vector if MODEL_KEY == "clip" else clip_image_vector(image)
    labels, text = label_text_features()
    probs = retrieval.softmax(100.0 * (text @ category_vector))
    return vector, probs, labels


# ------------------------------------------------------------------ index --
def _set_index(vectors, records):
    global indexed_vectors, image_records
    indexed_vectors = vectors
    image_records = records


def rebuild_index():
    vectors, records, prob_list, index_labels = [], [], [], []

    for path in retrieval.list_images(REFERENCE_DIR):
        try:
            with Image.open(path) as image:
                vector, probs, labels = embed_and_classify(image)
            record = retrieval.parse_image_path(path, REFERENCE_DIR)
            record["image_url"] = f"/reference-images/{record['relpath']}"
            vectors.append(vector)
            records.append(record)
            prob_list.append(probs)
            index_labels = labels
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}")

    # One AI category per OBJECT: average the category scores of all its photos, so
    # photos of the same object can never end up in different categories.
    totals, counts = {}, {}
    for rec, probs in zip(records, prob_list):
        iid = rec["instance_id"]
        totals[iid] = totals.get(iid, 0) + probs
        counts[iid] = counts.get(iid, 0) + 1
    for rec in records:
        mean = totals[rec["instance_id"]] / counts[rec["instance_id"]]
        best = int(np.argmax(mean))
        rec["category"] = index_labels[best]
        rec["category_conf"] = round(float(mean[best]), 3)

    matrix = np.vstack(vectors) if vectors else np.empty((0, 1), dtype=np.float32)
    _set_index(matrix, records)

    np.save(EMBEDDINGS_DIR / "vectors.npy", matrix)
    (EMBEDDINGS_DIR / "metadata.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8")
    (EMBEDDINGS_DIR / "index_info.json").write_text(
        json.dumps({"model": EMBED_MODEL_NAME, "count": len(records),
                    "labels": labels_hash()}),
        encoding="utf-8")
    return len(records)


def load_index_from_disk() -> bool:
    """True if a saved index exists AND matches the images now on disk."""
    try:
        info = json.loads((EMBEDDINGS_DIR / "index_info.json").read_text("utf-8"))
        records = json.loads((EMBEDDINGS_DIR / "metadata.json").read_text("utf-8"))
        vectors = np.load(EMBEDDINGS_DIR / "vectors.npy")
        on_disk = {p.relative_to(REFERENCE_DIR).as_posix()
                   for p in retrieval.list_images(REFERENCE_DIR)}
        if (info.get("model") != EMBED_MODEL_NAME
                or info.get("labels") != labels_hash()
                or any("category" not in r for r in records)
                or len(records) != len(vectors)
                or on_disk != {r["relpath"] for r in records}):
            return False
    except Exception:
        return False
    _set_index(vectors, records)
    return True


def ensure_index():
    if indexed_vectors is None and not load_index_from_disk():
        rebuild_index()


# -------------------------------------------------------------- endpoints --
@app.get("/")
def home():
    return {
        "message": "QA Visual Retrieval and VQA API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "embedding_model": EMBED_MODEL_NAME,
        "indexed_images": len(image_records),
        "indexed_instances": len({r["instance_id"] for r in image_records}),
        "domains": sorted({r["domain"] for r in image_records}),
        "thresholds_calibrated": bool(
            retrieval.load_thresholds(THRESHOLDS_PATH).get("calibrated")),
        "embedding_model_loaded": embed_model is not None,
        "vqa_model_loaded": vqa_model is not None,
    }


@app.post("/index/rebuild")
def index_rebuild():
    try:
        count = rebuild_index()
        return {
            "message": "Reference index rebuilt",
            "indexed_images": count,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/thresholds")
def thresholds():
    return retrieval.load_thresholds(THRESHOLDS_PATH)


@app.get("/evaluation")
def evaluation():
    """Metrics written by:  python backend/evaluate.py"""
    for path in EVAL_PATHS:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise HTTPException(
        status_code=404,
        detail="No evaluation results yet. Run: python backend/evaluate.py",
    )


@app.post("/search")
async def search(
    file: UploadFile = File(...),
    domain: str = Form("all"),
    top_k: int = Form(5),
    category_filter: str = Form("true"),
):
    if top_k < 1 or top_k > 50:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 50")

    ensure_index()

    if not image_records:
        raise HTTPException(
            status_code=400,
            detail=(
                "No reference images are indexed. Add images to "
                "backend/dataset/reference_images/<domain>/ and rebuild the index."
            ),
        )

    # Real domain filtering: the domain is the sub-folder of the reference image.
    domain = domain.strip().lower()
    mask = None
    if domain != "all":
        mask = np.array([r["domain"] == domain for r in image_records])
        if not mask.any():
            available = sorted({r["domain"] for r in image_records})
            raise HTTPException(
                status_code=400,
                detail=f"No reference images in domain '{domain}'. Available: {available}",
            )

    contents = await file.read()
    load_embedding_model()  # keep one-time model loading out of the latency figure
    try:
        t0 = time.perf_counter()
        with Image.open(BytesIO(contents)) as image:
            query_vector, cat_probs, labels = embed_and_classify(image)
        t1 = time.perf_counter()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a supported image.",
        )

    thr = retrieval.load_thresholds(THRESHOLDS_PATH)
    domain_mask = mask

    # AI step: name what is in the picture, then (optionally) search only that kind of item.
    order = np.argsort(-cat_probs)
    category = {
        "label": labels[int(order[0])],
        "confidence": round(float(cat_probs[order[0]]), 3),
        "top3": [{"label": labels[int(i)], "confidence": round(float(cat_probs[i]), 3)}
                 for i in order[:3]],
        "filter_applied": False,
        "note": "",
    }
    wanted = category_filter.strip().lower() in ("true", "1", "yes", "on")
    if wanted and category["confidence"] >= thr["category_min_conf"]:
        same = np.array([r.get("category") == category["label"] for r in image_records])
        combined = same if mask is None else (mask & same)
        if combined.any():
            mask = combined
            category["filter_applied"] = True
        else:
            category["note"] = (f"No '{category['label']}' items in the index, "
                                "so nearest items of any type are shown.")
    elif wanted:
        category["note"] = "Category not confident enough, so no category filter was used."

    scores = indexed_vectors @ query_vector
    candidates = retrieval.rank_instances(
        scores, image_records, mask, limit=max(top_k, 2))
    decision = retrieval.decide(candidates, thr)
    t2 = time.perf_counter()

    # Show only related items; hide weaker ones (e.g. a bag when you search a shoe).
    related = retrieval.filter_related(candidates, thr)[:top_k]
    hidden_unrelated = len(candidates[:top_k]) - len(related)
    results = [{**item, "rank": rank} for rank, item in enumerate(related, start=1)]

    scope_records = (image_records if domain_mask is None
                     else [r for r, keep in zip(image_records, domain_mask) if keep])
    return {
        "query_filename": file.filename,
        "domain": domain,
        "decision": decision,
        "category": category,
        "results": results,
        "hidden_unrelated": max(hidden_unrelated, 0),
        "indexed_images": len(scope_records),
        "indexed_instances": len({r["instance_id"] for r in scope_records}),
        "latency_ms": {
            "embedding": round((t1 - t0) * 1000, 1),
            "search": round((t2 - t1) * 1000, 2),
            "total": round((t2 - t0) * 1000, 1),
        },
        "message": (
            "Scores are cosine similarities between CLIP embeddings, not "
            "probabilities. UNKNOWN means the evidence was not strong enough."
        ),
    }


@app.post("/ask")
async def ask_about_image(
    file: UploadFile = File(...),
    question: str = Form(...),
):
    """Answer a natural-language question about the uploaded image using BLIP VQA."""
    cleaned_question = question.strip()

    if not cleaned_question:
        raise HTTPException(status_code=400, detail="Please enter a question.")

    if len(cleaned_question) > 500:
        raise HTTPException(
            status_code=400,
            detail="Please keep the question under 500 characters.",
        )

    contents = await file.read()
    try:
        with Image.open(BytesIO(contents)) as image:
            image = image.convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a supported image.",
        )

    try:
        current_model, current_processor = load_vqa_model()
        inputs = current_processor(
            images=image,
            text=cleaned_question,
            return_tensors="pt",
        )
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

        with torch.no_grad():
            output = current_model.generate(**inputs, max_new_tokens=30)

        answer = current_processor.decode(
            output[0],
            skip_special_tokens=True,
        ).strip()

        return {
            "question": cleaned_question,
            "answer": answer or "I couldn't determine an answer from this image.",
            "model": VQA_MODEL_NAME,
            "device": DEVICE,
            "message": (
                "AI-generated visual answer. Verify details if accuracy is important."
            ),
        }
    except HTTPException:
        raise
    except Exception as exc:
        print(f"VQA request failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail=(
                "The visual question-answering model could not process this request. "
                "Check the backend terminal for details. If GPU memory is insufficient, "
                "close other GPU-heavy applications or run the model on CPU."
            ),
        )
