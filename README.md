# DINO Pattern Detector — SOTATEK AI Home Assessment

A zero-shot pattern detection demo for finding repeated symbols in black-and-white technical drawings.

The system takes two images as input:

1. a cropped **pattern image**,
2. a full **technical drawing image**,

and returns bounding boxes for regions in the drawing that are visually similar to the query pattern.

This project uses frozen DINOv2 features, cosine-similarity heatmaps, candidate peak selection, and Non-Maximum Suppression (NMS). No training or fine-tuning is required when the input pattern changes.

---

## Demo

HuggingFace Space:

```text
https://huggingface.co/spaces/votaquangnhat/sotatek-assessment-hf
````

GitHub repository:

```text
https://github.com/votaquangnhat/sotatek-assessment
```

---

## Main idea

Instead of using classical pixel-level template matching, this project compares the query pattern and the drawing in DINOv2 feature space.

The pipeline is:

1. Convert the pattern and drawing to binary line-art images.
2. Crop foreground regions and align images to the DINOv2 patch grid.
3. Optionally estimate an ROI in the drawing to reduce the search area.
4. Extract frozen DINOv2 patch features.
5. Slide the pattern feature window over the drawing feature map.
6. Compute a cosine-similarity heatmap.
7. Select local heatmap peaks as candidate detections.
8. Convert candidates back to pixel bounding boxes.
9. Apply NMS and return the final detections.


## Installation

### Requirements

* Python 3.12+
* `uv`
* PyTorch
* Gradio
* OpenCV
* scikit-learn
* torchvision
* PIL / Pillow

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies with `uv`:

```bash
uv sync
```

If `uv` is not installed, install it first:

```bash
pip install uv
```

---

## Run the Gradio demo locally

```bash
uv run app.py
```

Then open:

```text
http://localhost:7860
```

The interface allows the user to upload a pattern image and a drawing image, run detection, view the result image, inspect the ROI preview, and read the detection JSON output.

---

## Run example inference

```bash
uv run dino_pattern_detector.py
```

The script uses images from the `examples/` folder.

It may save:

* `output.png`: detection visualization,
* `roi.png`: proposed ROI image,
* `heatmaps/`: saved heatmaps when debug mode is enabled.

---

## Output format

The detector returns a list of detections. Each detection contains:

```json
{
  "bbox": [x, y, w, h],
  "xyxy": [x1, y1, x2, y2],
  "score": 0.0,
  "drawing_scale": 1.0,
  "query_grid_size": [w, h],
  "drawing_grid_size": [w, h]
}
```

Where:

* `bbox` is the bounding box in `[x, y, width, height]` format,
* `xyxy` is the bounding box in `[x1, y1, x2, y2]` format,
* `score` is the average cosine similarity score,
* `drawing_scale` is the scale used during matching.

---

## Project structure

```text
.
├── app.py                    # Gradio demo
├── dino_pattern_detector.py  # Main detection pipeline
├── utils.py                  # Preprocessing, ROI, heatmap utilities
├── examples/                 # Example pattern and drawing images
├── heatmaps/                 # Debug heatmaps, generated when debug=True
├── pyproject.toml            # Project dependencies
└── README.md
```

---

## Notes

The first run may take some time because the DINOv2 model is downloaded automatically through `torch.hub`.

GPU is used automatically if available. If no GPU is available, the system runs on CPU, but inference can be slow, especially on HuggingFace Spaces.

The current default parameters were selected empirically from the provided examples.

---

## Known limitations and unfinished parts

This project is a functional prototype, but several parts are not fully solved yet.

1. **Scale search is still manual.**
   The current system uses pre-selected drawing scales. This works for some examples, but it is not robust enough for all pattern sizes.

2. **The system does not reliably detect the exact number of pattern occurrences.**
   It mainly returns the top-scoring candidate boxes. If the threshold is too low, false positives may appear. If the threshold is too high, some true patterns may be missed.

3. **Non-square patterns are difficult.**
   The current preprocessing resizes the pattern into a fixed token grid. This helped keep the feature window small, but it does not work well when the pattern width-to-height ratio is far from 1.

4. **Very thin or very simple patterns are difficult.**
   Since DINOv2 works at patch level, very thin line structures may not be represented clearly enough in the token features.

5. **The ROI proposal is heuristic.**
   The ROI is estimated by KMeans clustering on preview DINO features. It can reduce the search area, but it may fail on drawings with unusual layouts.

6. **Runtime on CPU is slow.**
   Computing heatmaps for multiple scales can take a long time on CPU, especially on HuggingFace Spaces.

7. **The model is not trained specifically for technical drawings.**
   DINOv2 is trained as a general visual feature extractor, mostly on natural images. It is useful for zero-shot matching, but it may not be optimal for line-based technical drawings.

---

## Future improvements

Given more time, I would improve the system in the following directions:

1. Add automatic scale search using a scale pyramid.
2. Improve handling of non-square patterns.
3. Add better preprocessing for scanned drawings, such as adaptive thresholding, denoising, and line thickening.
4. Cache drawing features when searching for multiple patterns in the same drawing.
5. Batch multiple scales to improve runtime.
6. Try other feature extractors instead of relying only on DINOv2.
