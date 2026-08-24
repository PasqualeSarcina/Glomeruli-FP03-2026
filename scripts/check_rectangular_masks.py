"""
Count glomerulus masks that are rectangles rather than polygons.

Motivation: a qualitative check surfaced a patch whose ground-truth mask is a
perfect axis-aligned rectangle. If some annotations are bounding boxes instead
of outlines, they inflate the glomerulus area (a circle fills only pi/4 ~ 79% of
its box), teach the network wrong boundaries and penalise it at evaluation.

Test used: fill ratio = glomerulus pixels / area of their bounding box.
  * a disc/ellipse  -> ~0.79
  * a rectangle     -> ~1.00
Clipping does NOT confound this test: a disc cut by a straight patch border keeps a
fill ratio of ~0.79 (a half-disc has area pi*r^2/2 inside a 2*r^2 box), so a boxy
fill ratio means a boxy annotation whether or not the component touches an edge.
Border contact is therefore reported as extra information, not as an exemption.

    python scripts/check_rectangular_masks.py data/dataset
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

RECT_THRESHOLD = 0.95   # fill ratio above which a component is called rectangular
MIN_PIXELS = 2000       # ignore specks: their fill ratio is noise


def components(mask: np.ndarray):
    """Yield (slice_y, slice_x, pixel_count) per connected component (4-connectivity)."""
    try:
        from scipy import ndimage
    except ImportError:
        sys.exit("needs scipy (available in the `glomeruli` env)")
    lab, n = ndimage.label(mask)
    for sy, sx in ndimage.find_objects(lab):
        yield sy, sx, int((lab[sy, sx] > 0).sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--splits", nargs="*", default=["train", "validation", "test"])
    args = ap.parse_args()

    per_slide = Counter()
    rect_files, total, clipped = [], 0, 0

    for split in args.splits:
        mask_dir = args.dataset / split / "mask"
        if not mask_dir.is_dir():
            continue
        for p in sorted(mask_dir.glob("*.png")):
            if "_reinhard_" in p.name:
                continue
            m = np.array(Image.open(p))
            if m.ndim == 3:
                m = m[..., 0]
            m = (m > 0).astype(np.uint8)
            if not m.any():
                continue
            h, w = m.shape
            for sy, sx, count in components(m):
                if count < MIN_PIXELS:
                    continue
                total += 1
                box = (sy.stop - sy.start) * (sx.stop - sx.start)
                fill = count / box if box else 0.0
                if fill <= RECT_THRESHOLD:
                    continue
                if (sy.start == 0 or sx.start == 0 or sy.stop >= h or sx.stop >= w):
                    clipped += 1          # counted as rectangular too, just also on an edge
                rect_files.append((p.name, fill, count))
                per_slide[p.name.split("_")[0]] += 1

    print(f"glomerulus components inspected : {total}")
    print(f"rectangular (fill > {RECT_THRESHOLD})        : {len(rect_files)}"
          + (f"  = {100 * len(rect_files) / total:.1f}% of components" if total else ""))
    print(f"  of which also touching an edge: {clipped}")
    if per_slide:
        print("\nby slide:")
        for slide, n in per_slide.most_common():
            print(f"  {slide}: {n}")
        print("\nexamples:")
        for name, fill, count in sorted(rect_files, key=lambda r: -r[1])[:15]:
            print(f"  {name}  fill={fill:.3f}  px={count}")


if __name__ == "__main__":
    main()
