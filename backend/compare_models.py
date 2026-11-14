"""Print CLIP vs DINOv2 side by side (needs evaluation_test.json for each).

    $env:QA_MODEL="clip";   python backend/evaluate.py --calibrate; python backend/evaluate.py
    $env:QA_MODEL="dinov2"; python backend/evaluate.py --calibrate; python backend/evaluate.py
    python backend/compare_models.py
"""
import json
from pathlib import Path

root = Path(__file__).resolve().parent / "embeddings"
reports = {}
for key in ("clip", "dinov2"):
    path = root / key / "evaluation_test.json"
    if path.exists():
        reports[key] = json.loads(path.read_text(encoding="utf-8"))
if not reports:
    raise SystemExit("No evaluation_test.json found for any model. See the docstring.")

rows = [("P@1", "retrieval", "P@1"), ("P@5", "retrieval", "P@5"),
        ("mP@5", "retrieval", "mP@5"), ("Recall@5", "retrieval", "Recall@5"),
        ("Unknown rejection", "decision", "unknown_rejection_rate"),
        ("False acceptance", "decision", "false_acceptance_rate"),
        ("Accept precision", "decision", "accept_precision"),
        ("Coverage", "decision", "coverage")]
print(f"{'metric':<20}" + "".join(f"{k:>10}" for k in reports))
for label, section, key in rows:
    cells = "".join(
        f"{'n/a':>10}" if r[section][key] is None else f"{r[section][key]:>10.3f}"
        for r in reports.values())
    print(f"{label:<20}{cells}")
print(f"{'embed ms/query':<20}" + "".join(f"{r['latency_ms']['embedding']:>10}" for r in reports.values()))
