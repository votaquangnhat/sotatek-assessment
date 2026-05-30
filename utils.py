import math
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision import transforms

import cv2
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

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


def display_overlay(mask: np.ndarray, background: Image.Image, alpha=0.5, output_path="overlay.png"):
    """ overplay is the mask, background is the original image"""

    # Convert to RGBA
    overlay = Image.fromarray(mask).convert("RGBA")
    background = background.convert("RGBA")

    # Make white transparent, black -> blue
    pixels = overlay.load()
    for y in range(overlay.height):
        for x in range(overlay.width):
            r, g, b, a = pixels[x, y]

            # assuming binary image: black foreground, white background
            if r < 128:  # black
                pixels[x, y] = (0, 0, 0, 0)  # fully transparent
            else:        # white
                pixels[x, y] = (0, 0, 255, 128)  # semi-transparent blue

    # Scale overlay up to match background size
    overlay = overlay.resize(background.size, Image.Resampling.NEAREST)

    # Put overlay on top
    result = Image.alpha_composite(background, overlay)

    result.save(output_path)
    return result


def find_roi(
    features_np: np.ndarray,
    image_size: tuple, #[w, h]
    patch_size=14,
    n_clusters=3,
    center_xy=None,
    min_component_area=5,
    keep_top_k_components=None,
    margin_tokens=0,
    l2_normalize=True,
    connectivity=8,
) -> Tuple[Tuple[int, int, int, int], Dict]:
    """
    KMeans center-cluster ROI, but ignore outlier islands.

    Steps:
    1. KMeans directly on DINO features
    2. Get center token's label
    3. Make mask of all tokens with that label
    4. Remove small connected components
    5. Take bbox of remaining tokens
    6. Convert bbox back to image coordinates by * patch_size

    Args:
        features: [H_tokens, W_tokens, D]
        image: PIL.Image
        patch_size: usually 14 for DINOv2 ViT/14
        n_clusters: KMeans cluster count
        center_xy: optional pixel-space point (x, y), default image center
        min_component_area: remove connected components smaller than this token count
        keep_top_k_components:
            - None: keep all components >= min_component_area
            - int: keep only top-k largest valid components
        margin_tokens: expand bbox by this many tokens
        l2_normalize: normalize features before KMeans

    Returns:
        roi: (x1, y1, x2, y2)
        debug: dict
    """

    h_tokens, w_tokens, d = features_np.shape
    flat_features = features_np.reshape(-1, d).astype(np.float32)

    if l2_normalize:
        flat_features = normalize(flat_features, norm="l2", axis=1)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=0,
        n_init=10,
    )

    flat_labels = kmeans.fit_predict(flat_features)
    labels = flat_labels.reshape(h_tokens, w_tokens)

    # Center token
    if center_xy is None:
        cx_token = w_tokens // 2
        cy_token = h_tokens // 2
    else:
        cx, cy = center_xy
        cx_token = int(cx / patch_size)
        cy_token = int(cy / patch_size)
        cx_token = np.clip(cx_token, 0, w_tokens - 1)
        cy_token = np.clip(cy_token, 0, h_tokens - 1)

    center_label = labels[cy_token, cx_token]

    # Mask of selected cluster
    cluster_mask = (labels == center_label).astype(np.uint8)

    # Connected components on token grid
    num_cc, cc_map, stats, _ = cv2.connectedComponentsWithStats(
        cluster_mask,
        connectivity=connectivity,
    )

    valid_components = []

    for cc_id in range(1, num_cc):
        area = stats[cc_id, cv2.CC_STAT_AREA]

        if area >= min_component_area:
            valid_components.append((cc_id, area))

    # Fallback: if filtering removed everything, use original cluster mask
    if len(valid_components) == 0:
        clean_mask = cluster_mask.copy()
    else:
        valid_components = sorted(
            valid_components,
            key=lambda x: x[1],
            reverse=True,
        )

        if keep_top_k_components is not None:
            valid_components = valid_components[:keep_top_k_components]

        clean_mask = np.zeros_like(cluster_mask)

        for cc_id, area in valid_components:
            clean_mask[cc_map == cc_id] = 1

    ys, xs = np.where(clean_mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, image_size[0], image_size[1]), {
            "labels": labels,
            "center_label": center_label,
            "cluster_mask": cluster_mask,
            "clean_mask": clean_mask,
            "center_token": (cx_token, cy_token),
        }

    x_min_token = max(0, xs.min() - margin_tokens)
    x_max_token = min(w_tokens - 1, xs.max() + margin_tokens)
    y_min_token = max(0, ys.min() - margin_tokens)
    y_max_token = min(h_tokens - 1, ys.max() + margin_tokens)

    x1 = int(x_min_token * patch_size)
    y1 = int(y_min_token * patch_size)
    x2 = int((x_max_token + 1) * patch_size)
    y2 = int((y_max_token + 1) * patch_size)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_size[0], x2)
    y2 = min(image_size[1], y2)

    roi = (x1, y1, x2, y2)

    debug = {
        "labels": labels,
        "center_label": center_label,
        "cluster_mask": cluster_mask,
        "clean_mask": clean_mask,
        "center_token": (cx_token, cy_token),
        "token_bbox": (
            x_min_token,
            y_min_token,
            x_max_token,
            y_max_token,
        ),
        "valid_components": valid_components,
    }

    return roi, debug
