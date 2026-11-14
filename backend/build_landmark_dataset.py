"""Build the LANDMARKS part of the dataset from the Unsplash Lite metadata zip.

The zip contains only a table of photo links, not the photos. This script reads
photos.tsv000 straight from the zip (nothing is unpacked), picks landmarks that have
several photos by different photographers, and downloads small copies (400 px wide).

    python backend/build_landmark_dataset.py --zip "C:\\path\\unsplash-research-dataset-lite-latest.zip" --dry-run
    python backend/build_landmark_dataset.py --zip "C:\\path\\unsplash-research-dataset-lite-latest.zip"

Per KNOWN landmark: 3 photos -> reference_images, 1 -> val_queries, the rest (max 2) -> test_queries.
UNKNOWN landmarks (no reference photo): some -> val_queries, different ones -> test_queries.
Photo credits are written to dataset/landmarks_credits.csv. Respect the Unsplash TERMS.md in the zip.
"""
import argparse
import csv
import io
import random
import re
import time
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

DATASET = Path(__file__).resolve().parent / "dataset"


def slug(text, n=18):
    return re.sub(r"[^a-z0-9]", "", text.lower())[:n] or "place"


def confidence(raw):
    try:
        v = float(raw)
    except ValueError:
        return 0.0
    return v / 100 if v > 1 else v      # the column mixes 0-1 and 0-100 scales


def load_rows(zip_path):
    with zipfile.ZipFile(zip_path) as z, z.open("photos.tsv000") as f:
        return list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"), delimiter="\t"))


def pick(rows, n_known, n_unknown, min_conf, seed):
    by_name = defaultdict(list)
    for r in rows:
        name = r["ai_primary_landmark_name"].strip()
        if name and confidence(r["ai_primary_landmark_confidence"]) >= min_conf:
            by_name[name].append(r)

    # "Yosemite National Park" / "... Half Dome" / "... Valley" are the same area under
    # different names; keeping them would create near-identical "different" objects.
    families = defaultdict(set)
    for name in by_name:
        families[name.split(",")[0].strip().lower()].add(name)
    clean = {n: p for n, p in by_name.items()
             if len(families[n.split(",")[0].strip().lower()]) == 1}

    rng = random.Random(seed)
    known_pool = sorted((n for n, p in clean.items() if len(p) >= 5), key=lambda n: (-len(clean[n]), n))
    known = known_pool[:n_known]
    unknown_pool = sorted((n for n, p in clean.items() if len(p) >= 2 and n not in known),
                          key=lambda n: (-len(clean[n]), n))
    unknown = unknown_pool[: 2 * n_unknown]          # first half -> val, second half -> test

    plan = []                                       # (split, instance_id, tag, row)
    for i, name in enumerate(known, 1):
        photos = clean[name][:]
        rng.shuffle(photos)
        iid = f"lm{i:02d}{slug(name)}"
        for k, r in enumerate(photos[:3], 1):
            plan.append(("reference_images", iid, f"ref{k}", r))
        plan.append(("val_queries", iid, "val1", photos[3]))
        for k, r in enumerate(photos[4:6], 1):
            plan.append(("test_queries", iid, f"test{k}", r))
    for j, name in enumerate(unknown):
        split = "val_queries" if j < n_unknown else "test_queries"
        iid = f"lmu{j + 1:02d}{slug(name)}"
        for k, r in enumerate(clean[name][:2], 1):
            plan.append((split, iid, f"unk{k}", r))
    return known, unknown, plan


def download(url, dest, width):
    req = urllib.request.Request(f"{url}?w={width}&q=75&fm=jpg&fit=max",
                                 headers={"User-Agent": "QA-student-project"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                dest.write_bytes(resp.read())
            return True
        except Exception as exc:
            print(f"  retry {attempt + 1} for {dest.name}: {exc}")
            time.sleep(1.5)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--zip", required=True, help="path to unsplash-research-dataset-lite-latest.zip")
    ap.add_argument("--known", type=int, default=12, help="landmarks that are in the index (default 12)")
    ap.add_argument("--unknown", type=int, default=5, help="unknown landmarks per split (default 5)")
    ap.add_argument("--min-confidence", type=float, default=0.6)
    ap.add_argument("--width", type=int, default=400, help="download width in px (small = less disk)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="only show the selection, download nothing")
    args = ap.parse_args()

    known, unknown, plan = pick(load_rows(args.zip), args.known, args.unknown, args.min_confidence, args.seed)
    print(f"Known landmarks ({len(known)}): {', '.join(known)}")
    print(f"Unknown landmarks ({len(unknown)}): {', '.join(unknown)}")
    counts = defaultdict(int)
    for split, *_ in plan:
        counts[split] += 1
    print("Photos:", dict(counts))
    if len(known) < args.known:
        print(f"Only {len(known)} landmarks have 5+ usable photos; lower --min-confidence for more.")
    if args.dry_run:
        return

    credits = []
    ok = 0
    for split, iid, tag, r in plan:
        folder = DATASET / split / "landmarks"
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / f"{iid}__{tag}.jpg"
        if dest.exists() or download(r["photo_image_url"], dest, args.width):
            ok += 1
            credits.append([dest.name, r["photo_id"], r["photo_url"],
                            f'{r["photographer_first_name"]} {r["photographer_last_name"]}'.strip()])
        else:
            print(f"  FAILED: {dest.name}")
    with open(DATASET / "landmarks_credits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "unsplash_photo_id", "unsplash_url", "photographer"])
        w.writerows(credits)
    print(f"Done: {ok}/{len(plan)} photos saved under {DATASET}")


if __name__ == "__main__":
    main()
