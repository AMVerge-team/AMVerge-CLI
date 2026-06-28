"""Image crop, rotate, and flip utility."""
from __future__ import annotations

import os
from dataclasses import dataclass

from PIL import Image, ImageOps, ImageSequence


@dataclass
class CropData:
    x: float
    y: float
    width: float
    height: float
    rotation: float = 0.0
    flip_h: bool = False
    flip_v: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "CropData":
        return cls(
            x=d["x"], y=d["y"],
            width=d["width"], height=d["height"],
            rotation=d.get("rotation", 0.0),
            flip_h=d.get("flip_h", False),
            flip_v=d.get("flip_v", False),
        )


def _transform_frame(img: Image.Image, crop: CropData) -> Image.Image:
    if crop.rotation != 0:
        img = img.rotate(-crop.rotation, expand=True)
    if crop.flip_h:
        img = ImageOps.mirror(img)
    if crop.flip_v:
        img = ImageOps.flip(img)
    x, y, w, h = round(crop.x), round(crop.y), round(crop.width), round(crop.height)
    return img.crop((x, y, x + w, y + h))


def _save(img: Image.Image, path: str) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        img.save(path, "JPEG", quality=95)
    elif ext == ".png":
        if img.mode == "P":
            img = img.convert("RGBA")
        img.save(path, "PNG", optimize=True)
    elif ext == ".gif":
        if img.mode not in ("P", "L"):
            img = img.convert("P", palette=Image.ADAPTIVE)
        img.save(path, "GIF", optimize=True)
    else:
        img.save(path)


def crop_image(source_path: str, dest_path: str, crop: CropData) -> None:
    """Crop, rotate, and flip an image (or animated GIF) and save to dest_path."""
    with Image.open(source_path) as img:
        if img.format == "GIF" and getattr(img, "is_animated", False):
            frames = [_transform_frame(f.copy(), crop) for f in ImageSequence.Iterator(img)]
            frames[0].save(
                dest_path,
                save_all=True,
                append_images=frames[1:],
                optimize=True,
                duration=img.info.get("duration", 100),
                loop=img.info.get("loop", 0),
                disposal=2,
            )
        else:
            _save(_transform_frame(img, crop), dest_path)
