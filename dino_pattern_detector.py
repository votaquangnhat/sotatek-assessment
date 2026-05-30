from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter
from torchvision import transforms
from utils import save_heatmap, crop_to_foreground, turn_to_binary, fit_to_patch_grid, find_roi, resize_with_scale
import matplotlib.pyplot as plt
import math
import time

# for effectiveness and efficiency balance, grid size of the pattern shouldn't be too small or too large
# therefore, the larger side of the pattern grid is set to be LONGER_GRID_SIZE
# this number is founded by empirical tests
LONGER_GRID_SIZE = 5
TOKEN_MASK_METHOD = "v2" # v1: any foreground pixel in patch, v2: resize + threshold
TEST_PATTERN_NO = 3
DEVICE = "cpu"
_SCALE_LIST = [1.5]# [x / 10 for x in range(5, 21)]

ROTATE_ANGLES = [Image.Transpose.ROTATE_90]

SCALE_LIST = [x*LONGER_GRID_SIZE / 5 for x in _SCALE_LIST]

class ImageWrapper:
    """
    Image wrapper for drawing image and pattern image.
    Given image will be:
    1. basically a binary image (still has 3 channels, for dinov2 input)
    2. cropped to foreground
    3. always divisible by patch size (either by cropping or resizing, controlled by option)
    4. grid_size has a side which always is LONGER_GRID_SIZE if pattern_grid_size_check is True
    """
    def __init__(
            self, 
            image: Image.Image, 
            threshold: int = 220,
            patch_size: int = 14, 
            resize_option: str = "crop",
            pattern_grid_size_check: bool = False, # if true, force the longer size of the image fit LONGER_GRID_SIZE
            ):
        
        binary = turn_to_binary(image, threshold=threshold)
        binary = crop_to_foreground(binary)
        image = binary.convert("RGB")

        if pattern_grid_size_check:
            w, h = image.size
            if w > h:
                new_w = patch_size * LONGER_GRID_SIZE
                new_h = patch_size * math.floor(LONGER_GRID_SIZE * h / w + 0.5)
            else:
                new_h = patch_size * LONGER_GRID_SIZE
                new_w = patch_size * math.floor(LONGER_GRID_SIZE * w / h + 0.5)
            
            ## need to pad to square
            image = image.resize((new_w, new_h), resample=Image.Resampling.NEAREST)
            
        binary = turn_to_binary(image, threshold=threshold)
        binary = crop_to_foreground(binary)
        binary, grid_size, is_originally_square = fit_to_patch_grid(
            image=binary,
            patch_size=patch_size,
            option=resize_option,
            make_square=pattern_grid_size_check,
        )

        if pattern_grid_size_check and max(grid_size) != LONGER_GRID_SIZE:
            raise ValueError(f"Grid size is not correct: {grid_size}, image size: {image.size}, patch size: {patch_size}")

        self.binary = binary
        self.image = binary.convert("RGB")
        self.image_size = binary.size # [w, h]
        self.grid_size = grid_size # [w, h]
        self.patch_size = patch_size
        self.resize_option = resize_option
        self.threshold = threshold
        self.is_originally_square = is_originally_square

        if TOKEN_MASK_METHOD == "v1":
            self.get_token_mask = self.get_token_mask_v1
        else:
            self.get_token_mask = self.get_token_mask_v2

    def get_token_mask_v1(self) -> torch.Tensor:
        arr = np.array(self.binary) == 0  # shape: [H, W]

        grid_h, grid_w = self.grid_size[::-1]

        token_mask = arr.reshape(
            grid_h, self.patch_size,
            grid_w, self.patch_size
        ).any(axis=(1, 3)) # a token is true if any pixel in its patch is foreground

        return torch.from_numpy(token_mask).bool()
    
    def get_token_mask_v2(self) -> torch.Tensor:
        resized_image = self.binary.resize(self.grid_size, resample=Image.Resampling.LANCZOS)
        resized_image = turn_to_binary(resized_image, threshold=self.threshold)
        mask_binary = np.array(resized_image) == 0 # black is True, white is False, shape [h,w]
        return torch.from_numpy(mask_binary).bool()
    
    def get_token_mask(self) -> torch.Tensor:
        return self.get_token_mask_v1() if TOKEN_MASK_METHOD == "v1" else self.get_token_mask_v2()


class DinoPatternDetector:
    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = torch.hub.load(
            repo_or_dir="facebookresearch/dinov2",
            model=model_name,
        ).to(self.device)
        self.model.eval()

        patch_size = getattr(self.model, "patch_size", 14)
        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]
        self.patch_size = int(patch_size)

        # reminder transforms in dino repo is: resize + to_tensor + normalize
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(
        self,
        image_wrapper: ImageWrapper,
    ) -> torch.Tensor:
        
        image_tensor = self.transform(image_wrapper.image)
        grid_w, grid_h = image_wrapper.grid_size

        with torch.inference_mode():
            image_batch = image_tensor.unsqueeze(0).to(self.device)
            tokens = self.model.get_intermediate_layers(image_batch)[0].squeeze()
            features = tokens.reshape(grid_h, grid_w, -1).contiguous() # tensor so [h,w]
            print(f"Tokens shape: {tokens.shape}, Features shape: {features.shape}")
        
        return features.to(self.device)

    def compute_similarity_heatmap(
        self,
        drawing_features: torch.Tensor,
        query_features: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Sliding feature-space matching.

        Returns:
            heatmap:
                Shape: H - qH + 1, W - qW + 1
        """

        H, W, D = drawing_features.shape
        qH, qW, qD = query_features.shape

        if qD != D:
            raise ValueError(f"Feature dim mismatch: drawing D={D}, query D={qD}")

        if qH > H or qW > W:
            return None

        # 1 x D x H x W
        drawing = drawing_features.permute(2, 0, 1).unsqueeze(0)
        query_feat = query_features.permute(2, 0, 1).unsqueeze(0)

        drawing = F.normalize(drawing, p=2, dim=1)
        query_feat = F.normalize(query_feat, p=2, dim=1)

        # 1 x 1 x qH x qW
        mask = query_mask.float().unsqueeze(0).unsqueeze(0)

        # Conv2D kernel: 1 x D x qH x qW
        kernel = query_feat * mask

        denom = mask.sum().clamp(min=1.0)

        # 1 x 1 x outH x outW
        heatmap = F.conv2d(drawing, kernel) / denom

        return heatmap.squeeze(0).squeeze(0)

    def get_candidates_from_heatmap(
        self,
        heatmap: torch.Tensor,
        score_threshold: Optional[float] = None,
        threshold_percentile: float = 99.5,
        max_candidates: int = 300,
        peak_kernel_size: int = 3,
    ) -> List[Tuple[int, int, float]]:
        """
        Convert heatmap peaks into candidate top-left token locations.

        """
        if heatmap is None or heatmap.numel() == 0:
            return []

        hm = heatmap.detach()

        if score_threshold is None:
            flat = hm.flatten()
            threshold = torch.quantile(flat, threshold_percentile / 100.0)
        else:
            threshold = torch.tensor(score_threshold, device=hm.device)

        if peak_kernel_size % 2 == 0:
            peak_kernel_size += 1

        pooled = F.max_pool2d(
            hm.unsqueeze(0).unsqueeze(0),
            kernel_size=peak_kernel_size,
            stride=1,
            padding=peak_kernel_size // 2,
        ).squeeze(0).squeeze(0)

        peaks = (hm >= pooled - 1e-6) & (hm >= threshold) # find local maximum and above threshold

        ys, xs = torch.where(peaks)

        if len(xs) == 0:
            return []

        scores = hm[ys, xs]

        order = torch.argsort(scores, descending=True)
        order = order[:max_candidates]

        candidates = []
        for idx in order:
            x = int(xs[idx].item())
            y = int(ys[idx].item())
            s = float(scores[idx].item())
            candidates.append((x, y, s))

        return candidates

    def nms(
        self,
        boxes: Sequence[Sequence[float]],
        scores: Sequence[float],
        iou_threshold: float = 0.3,
    ) -> List[int]:
        """
        Pure NumPy NMS.

        Args:
            boxes:
                xyxy boxes: [x1, y1, x2, y2]

            scores:
                confidence scores

        Returns:
            kept indices
        """
        if len(boxes) == 0:
            return []

        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        order = scores.argsort()[::-1]

        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(int(i))

            if order.size == 1:
                break

            rest = order[1:]

            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])

            inter_w = np.maximum(0, xx2 - xx1)
            inter_h = np.maximum(0, yy2 - yy1)
            inter = inter_w * inter_h

            union = areas[i] + areas[rest] - inter + 1e-6
            iou = inter / union

            order = rest[iou <= iou_threshold]

        return keep


    def detect(
        self,
        pattern_image: Image.Image,
        drawing_image: Image.Image,
        roi_coords: tuple,
        drawing_scales: Sequence[float] = (1.0,),
        score_threshold: Optional[float] = None,
        threshold_percentile: float = 99.5,
        nms_iou_threshold: float = 0.3,
        max_detections: int = 100,
        debug: bool = False,
    ) -> Tuple[List[Dict], Image.Image]:
        """
        Run pattern detection by resizing the drawing instead of resizing the pattern.

        The pattern is resized once by ImageWrapper

        For every drawing_scale:
            original drawing -> resized drawing
            detect boxes in resized drawing coordinates
            map boxes back to original drawing coordinates by / drawing_scale
        """
        roi_image = drawing_image.crop(roi_coords)
        original_w, original_h = roi_image.size

        # Pattern is processed once.
        original_pattern_wrapper = ImageWrapper(
            pattern_image,
            patch_size=self.patch_size,
            resize_option="resize",
            pattern_grid_size_check=True
        )

        # create all pattern variances by rotation
        pattern_image_wrappers = [original_pattern_wrapper]

        print("Creating rotated pattern variants...")
        for angle in ROTATE_ANGLES:
            rotated_image = original_pattern_wrapper.image.transpose(method=angle)
            pattern_image_wrappers.append(ImageWrapper(
                rotated_image,
                patch_size=self.patch_size,
                resize_option="resize",
                pattern_grid_size_check=True
            ))


        
        all_boxes = []
        all_scores = []
        all_meta = []

        for pattern_wrapper in pattern_image_wrappers:

            pattern_feature = self.extract_features(pattern_wrapper)
            query_mask = pattern_wrapper.get_token_mask().to(self.device)

            q_grid_w, q_grid_h = pattern_wrapper.grid_size
            query_w, query_h = pattern_wrapper.image_size

            for drawing_scale in drawing_scales:
                # Resize original drawing directly.
                # Do not resize drawing_wrapper.image, because that may already be cropped / binarized.
                start = time.time()
                scaled_w = max(1, int(round(original_w * drawing_scale)))
                scaled_h = max(1, int(round(original_h * drawing_scale)))

                scaled_drawing = roi_image.resize(
                    (scaled_w, scaled_h),
                    resample=Image.Resampling.BICUBIC,
                )

                curr_drawing_wrapper = ImageWrapper(
                    scaled_drawing,
                    patch_size=self.patch_size,
                    resize_option="crop",
                )

                d_grid_w, d_grid_h = curr_drawing_wrapper.grid_size

                # Query bigger than current drawing; skip.
                if q_grid_h > d_grid_h or q_grid_w > d_grid_w:
                    continue

                drawing_feature = self.extract_features(curr_drawing_wrapper)

                print(f"Extracted features for drawing scale {drawing_scale:.2f} in {time.time() - start:.2f} seconds")
                start = time.time()
                heatmap = self.compute_similarity_heatmap(
                    drawing_features=drawing_feature,
                    query_features=pattern_feature,
                    query_mask=query_mask,
                )
                print(f"Computed heatmap for drawing scale {drawing_scale:.2f} in {time.time() - start:.2f} seconds")

                if heatmap is None:
                    continue

                if debug:
                    save_heatmap(
                        heatmap,
                        save_path=f"heatmaps/heatmap_drawing_scale_{drawing_scale:.2f}.png",
                        title=f"Drawing scale {drawing_scale:.2f}",
                    )

                start = time.time()
                peak_kernel = max(
                    3,
                    int(min(q_grid_h, q_grid_w) // 2) * 2 + 1,
                )

                candidates = self.get_candidates_from_heatmap(
                    heatmap=heatmap,
                    score_threshold=score_threshold,
                    threshold_percentile=threshold_percentile,
                    max_candidates=300,
                    peak_kernel_size=peak_kernel,
                )
                print(f"Got {len(candidates)} candidates for drawing scale {drawing_scale:.2f} in {time.time() - start:.2f} seconds")
                start = time.time()
                if debug:
                    print(f"Drawing scale: {drawing_scale}")
                    print(f"Drawing grid: {d_grid_w} x {d_grid_h}")
                    print(f"Query grid: {q_grid_w} x {q_grid_h}")
                    print(f"Candidates len: {len(candidates)}")

                for token_x, token_y, score in candidates:
                    # Coordinates in the resized drawing.
                    x1_scaled = token_x * self.patch_size
                    y1_scaled = token_y * self.patch_size
                    x2_scaled = x1_scaled + query_w
                    y2_scaled = y1_scaled + query_h

                    # Map back to original drawing coordinates.
                    x1 = int(round(x1_scaled / drawing_scale))
                    y1 = int(round(y1_scaled / drawing_scale))
                    x2 = int(round(x2_scaled / drawing_scale))
                    y2 = int(round(y2_scaled / drawing_scale))

                    # Clamp to original image size.
                    x1 = max(0, min(original_w - 1, x1))
                    y1 = max(0, min(original_h - 1, y1))
                    x2 = max(0, min(original_w, x2))
                    y2 = max(0, min(original_h, y2))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    all_boxes.append([x1, y1, x2, y2])
                    all_scores.append(float(score))
                    all_meta.append(
                        {
                            "drawing_scale": float(drawing_scale),
                            "query_grid_size": [int(q_grid_w), int(q_grid_h)],
                            "drawing_grid_size": [int(d_grid_w), int(d_grid_h)],
                        }
                    )


        end = time.time()
        print(f"Measure time: {end - start:.2f} seconds")
        keep = self.nms(
            all_boxes,
            all_scores,
            iou_threshold=nms_iou_threshold,
        )
        print(f"Total candidates before NMS: {len(all_boxes)}, after NMS: {len(keep)}")
        keep = keep[:max_detections]

        detections = []

        roi_x1, roi_y1, _, _ = roi_coords
        for idx in keep:
            x1, y1, x2, y2 = all_boxes[idx]
            score = float(all_scores[idx])

            # add roi offset
            x1 += roi_x1
            x2 += roi_x1
            y1 += roi_y1
            y2 += roi_y1

            detections.append(
                {
                    "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                    "xyxy": [int(x1), int(y1), int(x2), int(y2)],
                    "score": score,
                    **all_meta[idx],
                }
            )

        detections = sorted(detections, key=lambda d: d["score"], reverse=True)

        
        visualization = self.draw_detections(drawing_image, detections)

        return detections, visualization


    def draw_detections(
        self,
        drawing_image: Image.Image,
        detections: List[Dict],
        box_width: int = 3,
    ) -> Image.Image:
        image = drawing_image.copy()
        draw = ImageDraw.Draw(image)

        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det["xyxy"]
            score = det["score"]
            drawing_scale = det.get("drawing_scale", 1.0)

            draw.rectangle(
                [x1, y1, x2, y2],
                outline=(255, 0, 0),
                width=box_width,
            )

            label = f"{score:.3f}\nds={drawing_scale:.2f}\nidx={idx}"
            text_y = max(0, y1 + 6)
            draw.text((x1 + 4, text_y), label, fill=(255, 0, 0))

        return image

if __name__ == "__main__":
    detector = DinoPatternDetector(model_name="dinov2_vits14", device=DEVICE)

    pattern_image = Image.open(f"examples/pattern{TEST_PATTERN_NO}.png")
    drawing_image = Image.open("examples/drawing.png")

    image_wrapper = ImageWrapper(resize_with_scale(drawing_image, scale=0.22), patch_size=14, resize_option="resize")

    scaled_roi_coords, _ = find_roi(
        features_np=detector.extract_features(image_wrapper).cpu().numpy(),
        image_size=image_wrapper.image_size,
        patch_size=detector.patch_size,
        n_clusters=3,
        min_component_area=8,
        keep_top_k_components=None,
        margin_tokens=0,
        connectivity=4,
    )

    roi_coords = (np.array(scaled_roi_coords) / 0.22).astype(int).tolist()
    efficiency = (roi_coords[2] - roi_coords[0]) * (roi_coords[3] - roi_coords[1]) / (drawing_image.size[0] * drawing_image.size[1])
    print(f"ROI efficiency: {efficiency:.2f}")

    roi = drawing_image.crop(roi_coords)
    #roi = drawing_image

    print(f"ROI coords: {roi_coords}")

    roi.save("roi.png")

    start_time = time.time()
    detections, viz = detector.detect(
        pattern_image=pattern_image,
        drawing_image=drawing_image,\
        roi_coords=roi_coords,
        drawing_scales=SCALE_LIST,
        threshold_percentile=90.0,
        nms_iou_threshold=0.05,
        max_detections=10,
        debug=True,
    )
    end_time = time.time()
    print(f"Detection time: {end_time - start_time:.2f} seconds")
    viz.save("output.png")