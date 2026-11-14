"""Retrieval, UNKNOWN decision and evaluation logic (pure numpy, no torch needed).

Dataset naming convention (used for reference images AND query images):

    <root>/<domain>/<instance_id>__<variation>.<ext>

    e.g. reference_images/fashion/shoe001__front.jpg
         test_queries/fashion/shoe001__side_blur.jpg

* domain      = first sub-folder (fashion, products, landmarks, artwork)
* instance_id = text before "__"  (photos of the same object share it)
* variation   = text after "__"   (free tags separated by "_", e.g. side_blur)
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_KS = (1, 3, 5, 10)
# Used only until you run:  python backend/evaluate.py --calibrate
# related_margin / related_floor control which results are SHOWN (Google-Lens style):
# an item is listed only if its score is within `related_margin` of the best score
# AND at least `related_floor`. Tune them by looking at your own scores.
DEFAULT_THRESHOLDS = {
    "min_score": 0.85, "min_gap": 0.0, "calibrated": False,
    "related_margin": 0.10, "related_floor": 0.0,
    # category filter is applied only if the AI is at least this confident
    "category_min_conf": 0.40,
}


# ----------------------------------------------------------------- dataset --
def list_images(root):
    return sorted(
        p for p in Path(root).rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def parse_image_path(path, root):
    rel = Path(path).relative_to(root)
    domain = rel.parts[0].lower() if len(rel.parts) > 1 else "uncategorized"
    instance_id, _, variation = Path(path).stem.partition("__")
    return {
        "instance_id": instance_id,
        "variation": variation,
        "domain": domain,
        "filename": Path(path).name,
        "relpath": rel.as_posix(),
    }


# -------------------------------------------------------------- thresholds --
def load_thresholds(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return {**DEFAULT_THRESHOLDS, **data}
    except (OSError, ValueError):
        return dict(DEFAULT_THRESHOLDS)


def save_thresholds(path, thresholds):
    Path(path).write_text(json.dumps(thresholds, indent=2), encoding="utf-8")


# ------------------------------------------------------- ranking + UNKNOWN --
def rank_instances(scores, records, mask=None, limit=5):
    """Best reference photo per instance, sorted by cosine similarity."""
    order = np.argsort(-scores)
    photo_counts = Counter(
        r["instance_id"] for i, r in enumerate(records)
        if mask is None or mask[i]
    )
    seen, out = set(), []
    for idx in order:
        if mask is not None and not mask[idx]:
            continue
        record = records[idx]
        iid = record["instance_id"]
        if iid in seen:
            continue
        seen.add(iid)
        out.append({
            **record,
            "similarity": float(scores[idx]),
            "reference_photos": photo_counts[iid],
        })
        if len(out) >= limit:
            break
    return out


def softmax(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = np.exp(z - z.max())
    return z / z.sum()


def filter_related(candidates, thr):
    """Keep only candidates that are visually close to the best one, so a query
    of a shoe lists shoes and not whatever else happens to be in the index.
    The best candidate is always kept."""
    if not candidates:
        return candidates
    cutoff = max(candidates[0]["similarity"] - thr["related_margin"],
                 thr["related_floor"])
    return [c for i, c in enumerate(candidates)
            if i == 0 or c["similarity"] >= cutoff]


def check_thresholds(top, second, thr):
    gap = None if second is None else top - second
    score_ok = top >= thr["min_score"]
    gap_ok = gap is None or gap >= thr["min_gap"]
    return score_ok, gap_ok, gap


def decide(candidates, thr):
    """MATCH only if the best score AND its lead over the runner-up are
    both strong enough; otherwise UNKNOWN with a human-readable reason."""
    base = {
        "min_score": thr["min_score"],
        "min_gap": thr["min_gap"],
        "calibrated": bool(thr.get("calibrated")),
    }
    if not candidates:
        return {**base, "status": "UNKNOWN", "top_score": None,
                "second_score": None, "score_gap": None,
                "reason": "No reference images are available in this scope."}

    top = candidates[0]["similarity"]
    second = candidates[1]["similarity"] if len(candidates) > 1 else None
    score_ok, gap_ok, gap = check_thresholds(top, second, thr)

    if score_ok and gap_ok:
        status = "MATCH"
        reason = (f"Best score {top:.3f} meets the {thr['min_score']:.3f} "
                  "acceptance threshold")
        if gap is not None:
            reason += (f" and leads the next instance by {gap:.3f} "
                       f"(needs at least {thr['min_gap']:.3f})")
        reason += "."
    else:
        status = "UNKNOWN"
        parts = []
        if not score_ok:
            parts.append(f"the best score {top:.3f} is below the "
                         f"{thr['min_score']:.3f} acceptance threshold")
        if not gap_ok:
            parts.append(f"it leads the runner-up by only {gap:.3f} "
                         f"(needs at least {thr['min_gap']:.3f})")
        reason = "Insufficient evidence: " + " and ".join(parts) + "."

    return {**base, "status": status, "reason": reason, "top_score": top,
            "second_score": second, "score_gap": gap}


# -------------------------------------------------------------- evaluation --
def build_rows(sims, q_records, ref_records, ks=DEFAULT_KS):
    """One row per query. sims = query_vectors @ reference_vectors.T"""
    ref_ids = np.array([r["instance_id"] for r in ref_records])
    counts = Counter(ref_ids.tolist())
    rows = []
    for i, q in enumerate(q_records):
        s = sims[i]
        order = np.argsort(-s)
        ranked = ref_ids[order]
        iid = q["instance_id"]
        n_rel = counts.get(iid, 0)
        top_iid = str(ranked[0])
        others = np.flatnonzero(ranked != top_iid)
        rows.append({
            "file": q["relpath"],
            "domain": q["domain"],
            "variation": q["variation"],
            "instance_id": iid,
            "known": n_rel > 0,          # instance exists in the index
            "n_relevant": n_rel,
            "top_instance": top_iid,
            "top_score": float(s[order[0]]),
            "second_score": float(s[order[others[0]]]) if others.size else None,
            "correct_top1": top_iid == iid,
            "rel": {k: int((ranked[:k] == iid).sum()) for k in ks},
        })
    return rows


def _mean(values):
    return round(float(np.mean(values)), 4) if values else None


def retrieval_metrics(rows, ks=DEFAULT_KS):
    """Ranking quality on queries whose instance IS in the index.

    P@K        relevant photos in top-K / K
    mP@K       relevant in top-K / min(K, #relevant)   (Google challenge style)
    Recall@K   relevant in top-K / #relevant
    Success@K  1 if at least one relevant photo is in the top-K
    """
    known = [r for r in rows if r["known"]]
    out = {"n_queries": len(known)}
    for k in ks:
        out[f"P@{k}"] = _mean([r["rel"][k] / k for r in known])
        out[f"mP@{k}"] = _mean(
            [r["rel"][k] / min(k, r["n_relevant"]) for r in known])
        out[f"Recall@{k}"] = _mean(
            [r["rel"][k] / r["n_relevant"] for r in known])
        out[f"Success@{k}"] = _mean([1.0 if r["rel"][k] else 0.0 for r in known])
    return out


def decision_metrics(rows, thr):
    """Behaviour of the accept / UNKNOWN rule on ALL queries."""
    n = len(rows)
    acc = acc_ok = acc_bad = 0
    n_unknown = rej_unknown = 0
    n_correct = false_reject = should_reject = 0
    for r in rows:
        s_ok, g_ok, _ = check_thresholds(r["top_score"], r["second_score"], thr)
        accepted = s_ok and g_ok
        correct = r["known"] and r["correct_top1"]
        if not r["known"]:
            n_unknown += 1
        if correct:
            n_correct += 1
        else:
            should_reject += 1          # unknown query or wrong top-1
        if accepted:
            acc += 1
            if correct:
                acc_ok += 1
            else:
                acc_bad += 1
        else:
            if not r["known"]:
                rej_unknown += 1
            if correct:
                false_reject += 1

    def ratio(a, b):
        return round(a / b, 4) if b else None

    return {
        "n_queries": n,
        "n_unknown_queries": n_unknown,
        "accepted": acc,
        "accepted_correct": acc_ok,
        "accepted_wrong": acc_bad,
        # share of unknown-identity queries correctly answered UNKNOWN
        "unknown_rejection_rate": ratio(rej_unknown, n_unknown),
        # share of "should not be accepted" queries that were accepted anyway
        "false_acceptance_rate": ratio(acc_bad, should_reject),
        # share of accepted answers that are right
        "accept_precision": ratio(acc_ok, acc),
        # share of correct top-1 answers wrongly turned into UNKNOWN
        "false_rejection_rate": ratio(false_reject, n_correct),
        "coverage": ratio(acc, n),
    }


def _group(rows, keys_of):
    groups = defaultdict(list)
    for r in rows:
        for key in keys_of(r):
            groups[key].append(r)
    return groups


def summarize(rows, thr, ks=DEFAULT_KS):
    keep = ("n_queries", "P@1", "P@5", "Success@1", "Recall@5")

    def short(rs):
        full = retrieval_metrics(rs, ks)
        return {key: full.get(key) for key in keep}

    by_domain = {
        d: short(rs)
        for d, rs in sorted(_group(rows, lambda r: [r["domain"]]).items())
    }
    by_variation = {
        v: short(rs)
        for v, rs in sorted(_group(
            rows, lambda r: [t for t in r["variation"].split("_") if t]).items())
    }
    return {
        "retrieval": retrieval_metrics(rows, ks),
        "by_domain": by_domain,
        "by_variation": by_variation,
        "decision": decision_metrics(rows, thr),
    }


def calibrate(rows, max_far=0.05, gaps=(0.0, 0.01, 0.02, 0.03, 0.05)):
    """Pick (min_score, min_gap) on VALIDATION rows: the most correct
    acceptances while keeping false_acceptance_rate <= max_far."""
    pts = sorted({r["top_score"] for r in rows})
    cands = ([pts[0] - 1e-3]
             + [(a + b) / 2 for a, b in zip(pts, pts[1:])]
             + [pts[-1] + 1e-3])
    best_ok = best_any = None
    for g in gaps:
        for t in cands:
            thr = {"min_score": t, "min_gap": g}
            m = decision_metrics(rows, thr)
            far = m["false_acceptance_rate"] or 0.0
            if far <= max_far:
                key = (m["accepted_correct"], -far, -g, t)
                if best_ok is None or key > best_ok[0]:
                    best_ok = (key, thr, m)
            key_any = (-far, m["accepted_correct"], -g, t)
            if best_any is None or key_any > best_any[0]:
                best_any = (key_any, thr, m)
    target_met = best_ok is not None
    _, thr, m = best_ok if target_met else best_any
    return {
        "min_score": round(float(thr["min_score"]), 4),
        "min_gap": float(thr["min_gap"]),
        "calibrated": True,
        "max_far_target": max_far,
        "target_met": target_met,
        "calibrated_on_queries": len(rows),
        "validation_metrics": m,
    }
