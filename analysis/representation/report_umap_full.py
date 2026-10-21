"""The report appendix's full UMAP: every screened representation, in three parts.

    python analysis/representation/report_umap_full.py
        -> report/figures/fig_read_umap_full_{1,2,3}.png

visualise.py drew all 15 representations as one image (figures/umap.png), one
row each: coloured by label, by the motif's positive rate, and by read depth.
At 15 rows it is too tall for a page, so this cuts it into three images of
five rows. Nothing is recomputed: the rows are the same sites and UMAP
settings as the report's four-panel figure (report_umap.py). Regenerating it
from scratch needs every representation's site vectors, two of which
(set_masked, contrastive) exist only on Ronin.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "figures" / "umap.png"
OUT = HERE.parents[1] / "report" / "figures"
ROWS = 15
PARTS = 3


def main() -> None:
    image = Image.open(SOURCE)
    width, height = image.size
    row = height // ROWS
    assert row * ROWS == height, f"{SOURCE} is not {ROWS} equal rows ({height} px)"
    per_part = ROWS // PARTS
    for part in range(PARTS):
        top = part * per_part * row
        crop = image.crop((0, top, width, top + per_part * row))
        path = OUT / f"fig_read_umap_full_{part + 1}.png"
        crop.save(path)
        print(f"wrote {path} (rows {part * per_part + 1}-{(part + 1) * per_part})")


if __name__ == "__main__":
    main()
