from __future__ import annotations

import json
import time
from functools import lru_cache
from typing import Sequence

import gradio as gr
import gradio.themes as gr_themes
from PIL import Image, ImageDraw

from dino_pattern_detector import DinoPatternDetector, ImageWrapper, SCALE_LIST
from utils import find_roi, resize_with_scale


DEFAULT_MODEL_NAME = "dinov2_vits14"
DEFAULT_DEVICE = None
ROI_PREVIEW_SCALE = 0.22


def _default_scale_text() -> str:
    return ", ".join(f"{scale:.2f}" for scale in SCALE_LIST) or "1.00"


def _parse_scales(text: str) -> list[float]:
    values: list[float] = []

    for part in text.replace("\n", ",").split(","):
        token = part.strip()
        if not token:
            continue

        try:
            scale = float(token)
        except ValueError as exc:
            raise gr.Error(f"Invalid drawing scale value: {token}") from exc

        if scale <= 0:
            raise gr.Error(f"Drawing scales must be positive, got {scale}")

        values.append(scale)

    if not values:
        raise gr.Error("Please provide at least one drawing scale.")

    return values


def _draw_roi(image: Image.Image, roi: Sequence[int]) -> Image.Image:
    preview = image.copy().convert("RGB")
    draw = ImageDraw.Draw(preview)
    x1, y1, x2, y2 = roi
    draw.rectangle([x1, y1, x2, y2], outline=(0, 180, 255), width=4)
    return preview


@lru_cache(maxsize=1)
def get_detector() -> DinoPatternDetector:
    return DinoPatternDetector(model_name=DEFAULT_MODEL_NAME, device=DEFAULT_DEVICE)


def detect_pattern(
    pattern_image: Image.Image,
    drawing_image: Image.Image,
    use_auto_roi: bool,
    roi_preview_scale: float,
    roi_n_clusters: int,
    roi_min_component_area: int,
    roi_margin_tokens: int,
    roi_keep_top_k_components: int,
    drawing_scales_text: str,
    threshold_percentile: float,
    score_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
    debug: bool,
):
    if pattern_image is None:
        raise gr.Error("Please upload a pattern image.")
    if drawing_image is None:
        raise gr.Error("Please upload a drawing image.")

    detector = get_detector()
    drawing_scales = _parse_scales(drawing_scales_text)
    roi_n_clusters = int(roi_n_clusters)
    roi_min_component_area = int(roi_min_component_area)
    roi_margin_tokens = int(roi_margin_tokens)
    roi_keep_top_k_components = int(roi_keep_top_k_components)
    max_detections = int(max_detections)

    roi_coords = [0, 0, drawing_image.width, drawing_image.height]
    roi_preview = drawing_image
    roi_status = "ROI: full drawing"

    if use_auto_roi:
        try:
            preview_image = resize_with_scale(drawing_image, scale=roi_preview_scale)
            preview_wrapper = ImageWrapper(
                preview_image,
                patch_size=detector.patch_size,
                resize_option="resize",
            )
            roi_small, _ = find_roi(
                features_np=detector.extract_features(preview_wrapper).cpu().numpy(),
                image_size=preview_wrapper.image_size,
                patch_size=detector.patch_size,
                n_clusters=roi_n_clusters,
                min_component_area=roi_min_component_area,
                keep_top_k_components=(
                    None if roi_keep_top_k_components <= 0 else roi_keep_top_k_components
                ),
                margin_tokens=roi_margin_tokens,
                connectivity=4,
            )

            roi_coords = [int(round(coord / roi_preview_scale)) for coord in roi_small]
            roi_coords[0] = max(0, min(drawing_image.width - 1, roi_coords[0]))
            roi_coords[1] = max(0, min(drawing_image.height - 1, roi_coords[1]))
            roi_coords[2] = max(1, min(drawing_image.width, roi_coords[2]))
            roi_coords[3] = max(1, min(drawing_image.height, roi_coords[3]))

            if roi_coords[2] <= roi_coords[0] or roi_coords[3] <= roi_coords[1]:
                roi_coords = [0, 0, drawing_image.width, drawing_image.height]
                roi_status = "ROI fallback: full drawing"
            else:
                roi_status = f"ROI: {roi_coords}"
                roi_preview = _draw_roi(drawing_image, roi_coords)
        except Exception as exc:
            roi_coords = [0, 0, drawing_image.width, drawing_image.height]
            roi_preview = drawing_image
            roi_status = f"ROI fallback: {exc}"

    score_threshold_value = score_threshold if score_threshold > 0 else None

    start_time = time.time()
    detections, visualization = detector.detect(
        pattern_image=pattern_image,
        drawing_image=drawing_image,
        roi_coords=tuple(roi_coords),
        drawing_scales=drawing_scales,
        score_threshold=score_threshold_value,
        threshold_percentile=threshold_percentile,
        nms_iou_threshold=nms_iou_threshold,
        max_detections=max_detections,
        debug=debug,
    )
    runtime_seconds = time.time() - start_time

    result = {
        "num_detections": len(detections),
        "detections": detections,
        "runtime_seconds": round(runtime_seconds, 3),
        "model": DEFAULT_MODEL_NAME,
        "device": detector.device,
        "drawing_scales": drawing_scales,
        "roi_coords": roi_coords,
        "roi_status": roi_status,
    }

    return visualization, json.dumps(result, indent=2), roi_preview, roi_status


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="DINO Pattern Detector Demo - Sotatek Assessment - Võ Tá Quang Nhật",
        theme=gr_themes.Soft(primary_hue="blue", secondary_hue="slate"),
        css="""
        .gradio-container {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        }
        .hero {
            padding: 0.5rem 0 0.25rem 0;
        }
        .hero h1 {
            margin: 0;
            font-size: 2rem;
            letter-spacing: -0.03em;
            color: #f8fafc;
        }
        .hero p {
            margin: 0.4rem 0 0 0;
            color: #cbd5e1;
        }
        """,
    ) as demo:
        gr.HTML(
            """
            <div class="hero">
              <h1>DINO Pattern Detector</h1>
              <p>Upload a cropped pattern and a technical drawing to search for similar symbols with frozen DINOv2 features.</p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                pattern_input = gr.Image(label="Pattern image", type="pil")
                drawing_input = gr.Image(label="Drawing image", type="pil")


            with gr.Column(scale=1):
                gr.Examples(
                    examples=[
                        ["examples/pattern1.png", "examples/drawing.png"],
                        #["examples/pattern2.png", "examples/drawing.png"],
                        ["examples/pattern3.png", "examples/drawing.png"],
                    ],
                    inputs=[pattern_input, drawing_input],
                    label="Examples",
                )

                with gr.Accordion("Tuning", open=False):
                    use_auto_roi = gr.Checkbox(value=True, label="Use auto ROI proposal")
                    roi_preview_scale = gr.Slider(
                        minimum=0.1,
                        maximum=0.5,
                        value=ROI_PREVIEW_SCALE,
                        step=0.01,
                        label="ROI preview scale",
                    )
                    roi_n_clusters = gr.Slider(
                        minimum=2,
                        maximum=8,
                        value=3,
                        step=1,
                        label="ROI clusters",
                    )
                    roi_min_component_area = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=8,
                        step=1,
                        label="ROI min component area",
                    )
                    roi_margin_tokens = gr.Slider(
                        minimum=0,
                        maximum=10,
                        value=0,
                        step=1,
                        label="ROI margin tokens",
                    )
                    roi_keep_top_k_components = gr.Slider(
                        minimum=0,
                        maximum=10,
                        value=0,
                        step=1,
                        label="ROI keep top-k components (0 = all)",
                    )
                    drawing_scales_text = gr.Textbox(
                        value=_default_scale_text(),
                        label="Drawing scales",
                        placeholder="0.8, 1.0, 1.2",
                    )
                    threshold_percentile = gr.Slider(
                        minimum=80.0,
                        maximum=99.9,
                        value=90.0,
                        step=0.1,
                        label="Heatmap threshold percentile",
                    )
                    score_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.0,
                        step=0.001,
                        label="Absolute score threshold (0 = disabled)",
                    )
                    nms_iou_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=0.9,
                        value=0.05,
                        step=0.01,
                        label="NMS IoU threshold",
                    )
                    max_detections = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=10,
                        step=1,
                        label="Max detections",
                    )
                    debug = gr.Checkbox(value=False, label="Debug mode (save heatmaps to disk)")

                run_button = gr.Button("Run detection", variant="primary")

        with gr.Row():
            output_image = gr.Image(label="Detection result", type="pil")
            roi_image = gr.Image(label="ROI preview", type="pil")

        with gr.Row():
            output_json = gr.Code(label="Detections JSON", language="json")
            status = gr.Textbox(label="Status", interactive=False)

        run_button.click(
            fn=detect_pattern,
            inputs=[
                pattern_input,
                drawing_input,
                use_auto_roi,
                roi_preview_scale,
                roi_n_clusters,
                roi_min_component_area,
                roi_margin_tokens,
                roi_keep_top_k_components,
                drawing_scales_text,
                threshold_percentile,
                score_threshold,
                nms_iou_threshold,
                max_detections,
                debug,
            ],
            outputs=[output_image, output_json, roi_image, status],
        )

    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.queue(max_size=8).launch(server_name="0.0.0.0")