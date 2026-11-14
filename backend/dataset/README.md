# Dataset layout and rules

```
dataset/
  reference_images/<domain>/<instance_id>__<variation>.jpg   <- the bounded, searchable index
  val_queries/<domain>/...                                   <- tune the UNKNOWN threshold ONLY
  test_queries/<domain>/...                                  <- final evaluation ONLY
```

* `<domain>`: `fashion`, `products`, `landmarks` or `artwork` (must match the website dropdown).
* `<instance_id>`: the identity. Photos of the same object share it, e.g. `shoe001`. Keep it unique across domains.
* `<variation>`: free tags joined by `_`, e.g. `front`, `side`, `top`, `side_blur`, `dark`.
  Metrics are reported per tag, so tag viewpoint and quality honestly.

## Rules that satisfy "identity-disjoint query/reference with controlled variation"

1. A query photo must never be the same photograph as a reference photo (`evaluate.py` hashes files and warns).
2. Reference index: 3-5 photos per instance, at least 3 domains, roughly 100-300 instances in total (bounded).
3. Known queries: new photos of instances that ARE in the index (different viewpoint / lighting / quality).
4. Unknown queries: photos of instances that have NO reference photo. Their `instance_id` simply does not
   exist in `reference_images/`. Use at least 20-30% unknown queries in both val and test.
5. val and test must use different photographs, and different unknown identities.
6. Quality variation: run `python backend/make_degraded_queries.py backend/dataset/test_queries`
   to add blur / low-res / dark / noise / jpeg copies of your QUERY photos (never of reference photos).
   Viewpoint variation must come from real photographs.
