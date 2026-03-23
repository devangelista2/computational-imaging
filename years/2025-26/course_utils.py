from pathlib import Path

import numpy as np
import torch
from IPython.display import display
from PIL import Image


def course_root() -> Path:
    here = Path.cwd().resolve()
    for base in (here, *here.parents):
        if (base / "_config.yml").exists() and (base / "imgs").exists():
            return base
    raise FileNotFoundError(f"Could not locate the course root from {here}")


def asset(name: str) -> Path:
    return course_root() / "imgs" / name


def gray(name: str, size=None) -> Image.Image:
    img = Image.open(asset(name)).convert("L")
    if size is None:
        return img
    if isinstance(size, int):
        return img.resize((size, size))
    return img.resize(size)


def rgb(name: str, size=None) -> Image.Image:
    img = Image.open(asset(name)).convert("RGB")
    if size is None:
        return img
    if isinstance(size, int):
        return img.resize((size, size))
    return img.resize(size)


def show(img: Image.Image, width=None) -> None:
    if width is None:
        display(img)
        return
    ratio = width / img.width
    height = max(1, int(img.height * ratio))
    display(img.resize((width, height)))


def array(img: Image.Image, dtype=np.float32, scale=True):
    arr = np.asarray(img, dtype=dtype)
    if scale:
        arr = arr / 255.0
    return arr


def tensor(img: Image.Image, dtype=torch.float32, batch=False):
    x = torch.tensor(array(img), dtype=dtype)
    x = x.unsqueeze(0)
    if batch:
        x = x.unsqueeze(0)
    return x
