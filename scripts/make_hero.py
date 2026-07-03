"""Rebuild docs/media/hero.png — a 2x2 collage of real gallery clips.

Each tile is the first frame of an actual recorded run (regenerate those
with the examples/*.py --record scripts first). Run:
    python scripts/make_hero.py
"""

import os

import numpy as np
from PIL import Image, ImageSequence

MEDIA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "docs", "media")
# a spread across the pillars: manipulation, worlds, real assets, dynamics
TILES = ["franka_cube_grasp", "procedural_room", "real_assets", "collision_tower"]
TW, TH = 480, 300


def first_frame(name):
    im = Image.open(os.path.join(MEDIA, name + ".webp"))
    frame = next(ImageSequence.Iterator(im)).convert("RGB")
    return frame.resize((TW, TH), Image.LANCZOS)


def main():
    grid = Image.new("RGB", (TW * 2, TH * 2), (12, 12, 12))
    for k, name in enumerate(TILES):
        grid.paste(first_frame(name), ((k % 2) * TW, (k // 2) * TH))
    out = os.path.join(MEDIA, "hero.png")
    grid.save(out, optimize=True)
    print(f"wrote {out}  ({os.path.getsize(out)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
