# DINO Pattern Detector — Sotatek Assessment

Lightweight demo for finding repeated symbols in technical drawings using frozen DINOv2 features.

**Installation**
- Requirements: Python 3.8+.
- Create and activate a virtualenv, then use `uv` to sync/install:
```bash
python -m venv .venv
source .venv/bin/activate
uv sync
```

**Run (demo)**
```bash
uv run app.py
```
Open http://localhost:7860 to use the Gradio interface.

**Example inference**
```bash
uv run dino_pattern_detector.py
```
The example script uses images in `examples/` and will save `output.png`, `roi.png`, and heatmaps under `heatmaps/` (when `debug=True`).

**Notes**
- Device: GPU is used automatically if available; otherwise CPU (slower).
