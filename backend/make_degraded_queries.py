"""Create controlled image-quality variants of QUERY photos (blur, low-res, dark,
noise, jpeg). Never run this on reference_images (that would leak the same
photograph into the index).

    python backend/make_degraded_queries.py backend/dataset/test_queries
    python backend/make_degraded_queries.py backend/dataset/val_queries --tags blur dark
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

import retrieval


def blur(im):
    return im.filter(ImageFilter.GaussianBlur(radius=4))


def lowres(im):
    w, h = im.size
    scale = 64 / min(w, h)
    small = im.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    return small.resize((w, h))


def dark(im):
    return ImageEnhance.Brightness(im).enhance(0.35)


def noise(im):
    arr = np.asarray(im, dtype=np.float32)
    arr += np.random.default_rng(0).normal(0, 25, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def jpeg(im):
    return im  # the low quality is applied when saving


EFFECTS = {"blur": blur, "lowres": lowres, "dark": dark, "noise": noise, "jpeg": jpeg}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path, help="val_queries or test_queries folder")
    parser.add_argument("--tags", nargs="+", default=list(EFFECTS), choices=list(EFFECTS))
    args = parser.parse_args()

    if "reference_images" in args.src.resolve().parts:
        raise SystemExit("Refusing to degrade reference images.")

    made = 0
    for path in retrieval.list_images(args.src):
        instance, _, variation = path.stem.partition("__")
        if any(tag in variation.split("_") for tag in EFFECTS):
            continue  # already a degraded copy
        with Image.open(path) as source:
            base = ImageOps.exif_transpose(source).convert("RGB")
        for tag in args.tags:
            new_variation = f"{variation}_{tag}" if variation else tag
            out = path.with_name(f"{instance}__{new_variation}.jpg")
            EFFECTS[tag](base).save(out, quality=15 if tag == "jpeg" else 92)
            made += 1
    print(f"Created {made} degraded query images in {args.src}")


if __name__ == "__main__":
    main()
