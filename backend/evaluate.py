"""Evaluate retrieval quality and the UNKNOWN rule.

Run from the project root with the virtual environment active:

    python backend/evaluate.py --calibrate   # 1) tune thresholds on dataset/val_queries
    python backend/evaluate.py               # 2) final report on dataset/test_queries

Never calibrate on test_queries: the test split must stay unseen.
A query counts as UNKNOWN-identity when its instance_id has no photo in
reference_images/ (so no correct answer exists in the index).
"""
import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone

import numpy as np
from PIL import Image

import main as app
import retrieval

KS = retrieval.DEFAULT_KS


def sha1(path):
    return hashlib.sha1(path.read_bytes()).hexdigest()


def embed_folder(root):
    vectors, records, paths, seconds = [], [], [], []
    for path in retrieval.list_images(root):
        try:
            with Image.open(path) as image:
                t0 = time.perf_counter()
                vector = app.image_embedding(image)
                seconds.append(time.perf_counter() - t0)
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}")
            continue
        vectors.append(vector)
        records.append(retrieval.parse_image_path(path, root))
        paths.append(path)
    return vectors, records, paths, seconds


def split_checks(split, q_paths, q_records):
    """Warnings that protect the 'query vs reference are disjoint' rule."""
    warnings = []
    ref_paths = retrieval.list_images(app.REFERENCE_DIR)
    ref_hashes = {sha1(p): p.name for p in ref_paths}
    dupes = [p.name for p in q_paths if sha1(p) in ref_hashes]
    if dupes:
        warnings.append(
            f"{len(dupes)} {split} query photo(s) are byte-identical to a reference "
            f"photo (e.g. {dupes[0]}). Use different photographs.")

    other = app.TEST_DIR if split == "val" else app.VAL_DIR
    other_paths = retrieval.list_images(other)
    other_hashes = {sha1(p) for p in other_paths}
    shared = [p.name for p in q_paths if sha1(p) in other_hashes]
    if shared:
        warnings.append(
            f"{len(shared)} photo(s) appear in BOTH val and test (e.g. {shared[0]}).")

    ref_ids = {r["instance_id"] for r in app.image_records}
    mine = {r["instance_id"] for r in q_records} - ref_ids
    theirs = {retrieval.parse_image_path(p, other)["instance_id"]
              for p in other_paths} - ref_ids
    if mine & theirs:
        warnings.append(
            f"{len(mine & theirs)} unknown identity(ies) are used in both val and test "
            f"(e.g. {sorted(mine & theirs)[0]}). Keep unknown identities disjoint.")
    if not mine:
        warnings.append(
            f"The {split} set has NO unknown-identity queries, so unknown rejection "
            "and false acceptance cannot be measured properly.")
    return warnings


def timing_ms(values_s):
    if not values_s:
        return {"mean": None, "p95": None}
    ms = np.array(values_s) * 1000
    return {"mean": round(float(ms.mean()), 2),
            "p95": round(float(np.percentile(ms, 95)), 2)}


def fmt(value):
    return "  n/a " if value is None else f"{value:6.3f}"


def print_report(report):
    s, thr = report, report["thresholds"]
    print("\n" + "=" * 64)
    print(f"QA evaluation - split: {s['split']}   model: {s['model']}")
    print(f"Index: {s['index']['photos']} photos, {s['index']['instances']} instances")
    print(f"Queries: {s['decision']['n_queries']} "
          f"({s['retrieval']['n_queries']} known, "
          f"{s['decision']['n_unknown_queries']} unknown-identity)")
    print("=" * 64)
    print(f"{'K':>4} {'P@K':>7} {'mP@K':>7} {'Recall@K':>9} {'Success@K':>10}")
    for k in KS:
        r = s["retrieval"]
        print(f"{k:>4} {fmt(r[f'P@{k}'])} {fmt(r[f'mP@{k}'])} "
              f"{fmt(r[f'Recall@{k}'])}  {fmt(r[f'Success@{k}'])}")
    d = s["decision"]
    print(f"\nUNKNOWN rule: min_score={thr['min_score']}  min_gap={thr['min_gap']}  "
          f"calibrated={thr.get('calibrated')}")
    for key in ("unknown_rejection_rate", "false_acceptance_rate",
                "accept_precision", "false_rejection_rate", "coverage"):
        print(f"  {key:<24} {fmt(d[key])}")
    print(f"\nLatency (ms/query): embedding {s['latency_ms']['embedding']}, "
          f"search {s['latency_ms']['search']}")
    for w in s["warnings"]:
        print("WARNING:", w)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--calibrate", action="store_true",
                        help="tune the UNKNOWN thresholds on val_queries and save them")
    parser.add_argument("--split", choices=["val", "test"],
                        help="default: val with --calibrate, otherwise test")
    parser.add_argument("--max-far", type=float, default=0.05,
                        help="target false acceptance rate for calibration (default 0.05)")
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild the reference index first")
    args = parser.parse_args()

    split = args.split or ("val" if args.calibrate else "test")
    if args.calibrate and split != "val":
        parser.error("calibrate on the validation split only, never on test")
    query_dir = app.VAL_DIR if split == "val" else app.TEST_DIR

    if args.rebuild:
        app.rebuild_index()
    else:
        app.ensure_index()
    if not app.image_records:
        raise SystemExit("No reference images. Fill backend/dataset/reference_images/<domain>/")

    vectors, q_records, q_paths, embed_s = embed_folder(query_dir)
    if not vectors:
        raise SystemExit(f"No query images found in {query_dir}")
    q_matrix = np.vstack(vectors)
    ref = app.indexed_vectors

    rows = retrieval.build_rows(q_matrix @ ref.T, q_records, app.image_records, KS)

    if args.calibrate:
        thr = retrieval.calibrate(rows, args.max_far)
        previous = retrieval.load_thresholds(app.THRESHOLDS_PATH)
        for key in ("related_margin", "related_floor", "category_min_conf"):  # keep hand-tuned display settings
            thr[key] = previous[key]
        thr["model"] = app.EMBED_MODEL_NAME
        retrieval.save_thresholds(app.THRESHOLDS_PATH, thr)
        print(f"Saved thresholds to {app.THRESHOLDS_PATH}")
        if not thr["target_met"]:
            print(f"WARNING: could not reach false acceptance <= {args.max_far}; "
                  "saved the lowest achievable. Add more/better validation data.")
    else:
        thr = retrieval.load_thresholds(app.THRESHOLDS_PATH)
        if not thr.get("calibrated"):
            print("WARNING: thresholds are NOT calibrated (using defaults). "
                  "Run: python backend/evaluate.py --calibrate")

    search_s = []
    for vector in q_matrix:
        t0 = time.perf_counter()
        retrieval.rank_instances(ref @ vector, app.image_records, None, 5)
        search_s.append(time.perf_counter() - t0)

    summary = retrieval.summarize(rows, thr, KS)
    domains = {}
    for r in app.image_records:
        domains[r["domain"]] = domains.get(r["domain"], 0) + 1
    report = {
        "split": split,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": app.EMBED_MODEL_NAME,
        "index": {"photos": len(app.image_records),
                  "instances": len({r["instance_id"] for r in app.image_records}),
                  "photos_per_domain": domains},
        "thresholds": {k: thr.get(k) for k in
                       ("min_score", "min_gap", "calibrated", "max_far_target")},
        **summary,
        "latency_ms": {"embedding": timing_ms(embed_s)["mean"],
                       "search": timing_ms(search_s)["mean"],
                       "search_p95": timing_ms(search_s)["p95"]},
        "warnings": split_checks(split, q_paths, q_records),
    }

    (app.EMBEDDINGS_DIR / f"evaluation_{split}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    csv_path = app.EMBEDDINGS_DIR / f"per_query_{split}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "domain", "variation", "true_instance", "known",
                         "top_instance", "top_score", "second_score", "correct_top1"])
        for r in rows:
            writer.writerow([r["file"], r["domain"], r["variation"], r["instance_id"],
                             r["known"], r["top_instance"], round(r["top_score"], 4),
                             "" if r["second_score"] is None else round(r["second_score"], 4),
                             r["correct_top1"]])

    print_report(report)
    print(f"\nSaved: evaluation_{split}.json and per_query_{split}.csv in {app.EMBEDDINGS_DIR}")


if __name__ == "__main__":
    main()
