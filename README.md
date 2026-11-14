# QA - Universal Image-Instance Search with an UNKNOWN state

CLIP (or DINOv2) embeddings + cosine retrieval + validation-calibrated UNKNOWN decision + retrieval evaluation.
(The optional BLIP "Ask about this image" feature from the previous version is unchanged.)

## What the system does
1. Embed the uploaded query photo with CLIP (ViT-B/32), L2-normalised.
2. Rank reference instances by their best-matching photo (cosine similarity), optionally within one domain.
3. Decide: **MATCH** only if `top score >= min_score` AND `top score - runner-up >= min_gap`;
   otherwise **UNKNOWN**. Both thresholds are tuned on the validation queries.
4. Show candidates, query-vs-reference photos, scores, gap, thresholds, the reason, and latency.

## Setup (Windows / VS Code, from the project root)
```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements.txt
```

## Workflow
1. Build the dataset: see `backend/dataset/README.md`.
2. Start the API (leave running): `python -m uvicorn main:app --app-dir backend --reload`
   and open `index.html` with Live Server. The index is built on the first search
   (or `POST /index/rebuild` in `/docs`) and cached in `backend/embeddings/<model>/`.
3. In a second terminal (venv active):
   ```powershell
   python backend/evaluate.py --calibrate   # tunes UNKNOWN thresholds on val_queries
   python backend/evaluate.py               # final metrics on test_queries
   ```
   Refresh the website: the Evaluation section shows P@K, mP@K, Recall@K, Success@K, unknown rejection,
   false acceptance, latency, and per-domain / per-tag results. `per_query_test.csv` lists every query.

## AI category recognition (Google-Lens style)
Before searching, CLIP names what is in the picture (zero-shot, using the prompts
"a photo of a <label>" for every label in `backend/labels.txt`). The website shows
"AI recognised: shoe (92% confidence)" and, with the checkbox on, lists ONLY items of that type.
* Edit `backend/labels.txt` so it contains the kinds of objects in YOUR dataset (one word per concept).
  After editing, the index rebuilds automatically on the next search (every reference photo gets a category).
* If the AI is less than `category_min_conf` (default 0.40, in `thresholds.json`) sure, no category filter is used.
* If the index has no item of the recognised type, the nearest items of any type are shown, with a note.
* Works with both embedding models: with DINOv2, CLIP is loaded additionally just to name the category.
* `related_margin` / `related_floor` in `thresholds.json` additionally hide weakly similar items.
* `evaluate.py` measures the embedding retrieval without the category filter; there is no ground-truth
  category for your photos, so category accuracy is not part of those metrics.

## Choosing / comparing embedding models (CLIP vs DINOv2)
Default is CLIP. DINOv2 is usually stronger for exact-instance matching. Switch with an environment
variable in the SAME terminal that starts the API and runs the evaluation:
```powershell
$env:QA_MODEL = "dinov2"      # or "clip"
python -m uvicorn main:app --app-dir backend --reload
```
Each model keeps its own index, thresholds and results in `backend/embeddings/<model>/`, because
similarity scores from different models are not comparable. Calibrate every model separately:
```powershell
$env:QA_MODEL="clip";   python backend/evaluate.py --calibrate; python backend/evaluate.py
$env:QA_MODEL="dinov2"; python backend/evaluate.py --calibrate; python backend/evaluate.py
python backend/compare_models.py          # side-by-side table for your report
```
DINOv2 downloads about 350 MB on first use. The website always shows the model the API was started with.

## History page (second interface)
`history.html` lists every image you searched, with the AI's decision (MATCH / UNKNOWN), the recognised
category, the scores and thresholds, the time taken and the result photos. Open it from the "History"
link in the top bar. You can filter by Match / Unknown or by text, delete single entries, clear everything,
and download the list as a CSV (useful for your report).
* History is stored in the BROWSER (localStorage, last 50 searches, small thumbnails of your query images).
  It is not sent to the server, so it is private to that browser and is lost if you clear site data.
* Searches are saved automatically after each successful search on the Search page.

## Starter dataset and landmark builder
* `backend/dataset/reference_images/` already holds 5 sample objects (2 shoes, 1 backpack in `fashion/`,
  2 pens in `products/`), ONE photo each. That is enough to try the app, but not to evaluate it: to test
  a search, upload a DIFFERENT photo of a shoe / pen / bag (never the same file), and add more photos of
  each object (`shoe001__side.jpg`, `shoe001__top.jpg` ...) before running `evaluate.py`.
* The Unsplash Lite zip contains only a table of photo links. Build a real multi-photo landmarks set from it
  (nothing is unpacked, the download is about 5 MB):
  ```powershell
  python backend/build_landmark_dataset.py --zip "C:\path\unsplash-research-dataset-lite-latest.zip" --dry-run
  python backend/build_landmark_dataset.py --zip "C:\path\unsplash-research-dataset-lite-latest.zip"
  ```
  It creates 12 known landmarks (3 reference photos each, 1 validation and up to 2 test photos) plus unknown
  landmarks for the UNKNOWN tests, and writes photo credits to `dataset/landmarks_credits.csv`.
  Some places overlap (e.g. Moraine Lake and Banff National Park), which makes retrieval harder, as in real life.
  Afterwards the zip can be deleted to free disk space.
* All photos of one object get the same AI category (their category scores are averaged).

## Deploying online
Vercel cannot run the backend (torch + transformers are over 5 GB; Vercel's limit is 500 MB), so deploy in two parts:
1. **Backend -> Hugging Face Space (Docker).** Create a Space with SDK "Docker". Upload everything inside
   `backend/` (including `dataset/` with your reference images, `labels.txt`, `Dockerfile`) to the Space, and
   rename `HF_SPACE_README.md` to `README.md` there. Wait for the build; the Space gives you an
   `https://<user>-<space>.hf.space` address. Check `<address>/health`.
2. **Frontend -> Vercel.** Edit `config.js`: `window.QA_API_BASE = "https://<user>-<space>.hf.space";`
   then commit and push. `vercel.json` makes Vercel deploy only the static page and ignore `backend/`.
Free Spaces sleep when idle, so the first request after a break is slow.

## Metric definitions
* **P@K** relevant photos in top-K / K. **mP@K** relevant in top-K / min(K, #relevant) (Google-challenge style).
  **Recall@K** relevant in top-K / #relevant. **Success@K** at least one relevant in top-K.
  Computed on queries whose instance is in the index.
* **Unknown rejection rate**: unknown-identity queries answered UNKNOWN / all unknown-identity queries.
* **False acceptance rate**: queries accepted although they should not be (unknown identity or wrong top-1)
  / all queries that should not be accepted.
* **Accept precision**: correct accepted answers / accepted answers. **False rejection rate**: correct top-1
  answers turned into UNKNOWN / correct top-1 answers. **Coverage**: share of queries answered with a MATCH.

## Honest limits
* Cosine similarity is not a probability; thresholds are only as good as the validation data.
* Until you run `--calibrate`, default thresholds are used and the website says "NOT calibrated".
* Report the metrics your own dataset produces; do not quote numbers from examples.
