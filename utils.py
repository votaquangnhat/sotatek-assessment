import math
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision import transforms

def _to_pil_rgb(
    image: Union[str, np.ndarray, Image.Image],
) -> Image.Image:
    if isinstance(image, str):
        image = Image.open(image)
    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] == 4:
            image = Image.fromarray(arr, mode="RGBA")
        else:
            image = Image.fromarray(arr[..., :3], mode="RGB")
    if not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported image type: {type(image)}")
    if image.mode == "RGBA":
        bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        bg.alpha_composite(image)
        image = bg.convert("RGB")
    else:
        image = image.convert("RGB")
    return image

def crop_to_foreground(image: Image.Image) -> Image.Image:
    """Crops the image to the bounding box of the foreground (non-white) pixels."""
    bbox = image.getbbox()
    if bbox is None:
        print("Warning: no foreground detected, using original image")
        return image
    return image.crop(bbox)

def turn_to_binary(image: Image.Image, threshold: int = 220) -> Image.Image:
    gray = image.convert("L")
    binary = gray.point(lambda x: 255 if x > threshold else 0)
    return binary

def fit_to_patch_grid(image: Image.Image, patch_size: int, option: str = "crop") -> tuple[Image.Image, Tuple[int, int]]:
    new_w = math.floor(image.width / patch_size) * patch_size
    new_h = math.floor(image.height / patch_size) * patch_size

    if new_w == 0 or new_h == 0:
        raise ValueError(
            f"Image is too small after cropping: {image.size}. "
            f"Need at least {patch_size}x{patch_size} pixels in the foreground."
        )
    if option == "crop":
        image = image.crop((0, 0, new_w, new_h))
    elif option == "resize":
        image = image.resize((new_w, new_h), Image.Resampling.NEAREST)
    else:
        raise ValueError(f"Invalid option: {option}. Must be 'crop' or 'resize'.")
    
    grid_size = (image.size[0] // patch_size, image.size[1] // patch_size) # [w, h]
    return image, grid_size

def resize_with_scale(image: Image.Image, scale: float) -> Image.Image:
    w, h = image.size
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    new_image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
    return new_image

def save_heatmap(
    heatmap: torch.Tensor,
    save_path: str,
    title: str | None = None,
):
    """
    Saves a heatmap tensor as a PNG image.

    Args:
        heatmap:
            Shape: outH x outW
        save_path:
            Example: "heatmap0.png"
        title:
            Optional plot title
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    hm = heatmap.detach().cpu().float()

    plt.figure(figsize=(8, 6))
    plt.imshow(hm, cmap="hot")
    plt.colorbar(label="Average cosine similarity")

    if title is not None:
        plt.title(title)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved heatmap to {save_path}")
    plt.close()