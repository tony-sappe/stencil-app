# Stencil Creator

Turn phone photos of artwork into clean, single-color vector files suitable for screen printing stencils.

The project provides both an interactive GUI (recommended) and command-line tools. It uses OpenCV for preprocessing (denoising, adaptive thresholding, cleanup) followed by [vtracer](https://github.com/visioncortex/vtracer) for high-quality vectorization.

## Features

- **Live preview GUI** — adjust settings and see the resulting vector update in real time
- Tuned preprocessing for noisy phone-camera photos
- Export **SVG**, **EPS**, or **PDF** from the save dialog (EPS/PDF require [Inkscape](https://inkscape.org/) on your PATH)
- Controls mapped to OpenCV and vtracer parameters with intuitive labels

## Installation

```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.9+
- The packages listed in `requirements.txt` (pure pip installs, no system dependencies for the GUI):
  - `dearpygui`
  - `numpy`
  - `opencv-python`
  - `pillow`
  - `vtracer`

**Dear PyGui** is used for the GUI (replaced the previous Tkinter version). It is a single `pip install dearpygui` with no Tcl/Tk or other system library requirements — it works cleanly even in pyenv / custom Python setups on macOS.

No extra steps are needed beyond `pip install -r requirements.txt`.

## Usage

### GUI (Recommended)

The GUI uses **Dear PyGui** for a clean, modern interface (no Tkinter).

```bash
python stencil_app.py
```

1. Click **Load Photo** and select an image (JPG, PNG, etc. — HEIC works if `pillow-heif` is also installed). The full image is processed; exports use the same pixel dimensions.
2. Adjust the sliders. Previews update live (throttled during rapid drags for stability; a final update is forced when you release the mouse so you always see the exact final settings).
3. Use the **Invert Image** checkbox if needed.
4. Click **Save Vector** and pick a filename ending in `.svg`, `.eps`, or `.pdf`.
5. Use **Open SVG** to quickly view the last saved (or current live) vector in your default viewer.

#### Understanding the three previews

The GUI shows three panes. They are **not** three versions of the same image type — each pane uses a different pipeline.

| Pane | What you are seeing | Use it for |
|------|---------------------|------------|
| **Original photo** | Your loaded image (full width and height) | Reference for the source photo |
| **Stencil bitmap** (center) | The **exact** black/white bitmap from OpenCV preprocessing (0/255 pixels), thumbnailed for display | Tuning threshold, denoise, blob removal, line thickening — this is what **vtracer receives** |
| **Traced SVG preview** (right) | SVG rasterized at **up to 2048 px wide** (same aspect as the image), then shown in the pane | Tuning trace settings; closer to export than before, but still a thumbnail |

**Click any preview** to open a **full-window modal** (export-width raster for the trace view; 1:1-style view for the stencil bitmap).

**Why the small right pane used to look much thicker than the center**

Even with **Preset: max trace fidelity**, the right pane is **not** a pixel copy of the center pane:

1. **Different source** — Center is a raw bitmap; right is **filled SVG paths** drawn by a renderer.
2. **Rasterization** — Anti-aliased PNG export plus binarization can still thicken strokes slightly (the app uses a high threshold to reduce this).
3. **Thumbnail** — All three panes are letterboxed into ~470 px boxes; use **click to enlarge** for an export-width trace view.
4. **Tracing** — vtracer builds filled regions, not single-pixel strokes.

**What to trust for screen printing**

- **Center (stencil bitmap)** — Ground truth for the binary stencil before tracing.
- **Saved SVG / EPS / PDF** — Ground truth for export. The SVG root `width` and `height` match the loaded image pixel dimensions (same aspect ratio as the original).
- **Right thumbnail** — Quick check while tuning; **click it** for a larger trace raster, or judge the **PDF** for final line weight.

#### Controls

| Control | Group | Maps to | Effect | Range |
|---------|-------|---------|--------|-------|
| Noise reduction | Preprocess | OpenCV `fastNlMeansDenoising` **`h`** | Stronger denoising before threshold | 0 – 40 |
| Blur before threshold (σ) | Preprocess | `GaussianBlur` **sigma** (kernel fixed 5×5) | Smoothing before adaptive threshold | 0.0 – 5.0 |
| Threshold bias (C) | Preprocess | `adaptiveThreshold` **`C`** | Higher → lighter image (more white) | -20 – 20 |
| Local threshold window | Preprocess | `adaptiveThreshold` **`blockSize`** (forced odd ≥ 3) | Larger → smoother local threshold | 3 – 31 |
| Remove blobs smaller than (px) | Preprocess | Connected-components area filter | Drops tiny black/white specks before tracing | 10 – 100 |
| Thicken stencil lines (px) | Preprocess | Morphological dilation on ink | Thicker printable lines | 0.0 – 12.0 |
| Invert Image | Preprocess | `bitwise_not` | Swap black/white | — |
| Preset: max trace fidelity | Vector | Preset | Sets vtracer to minimum simplification (does not make right pane pixel-identical to center) | — |
| Curve type | Vector | vtracer **`mode`** | Spline / polygon / pixel-accurate paths | combo |
| Path fidelity | Vector | vtracer **`length_threshold`** [3.5, 10] | Higher slider → more detail (inverse map) | 0.35 – 1.0 |
| Corner smoothing (°) | Vector | vtracer **`corner_threshold`** | Higher → straighter paths (fewer corners) | 0 – 180 |
| Join straight segments (°) | Vector | vtracer **`splice_threshold`** | Angle threshold for merging segments | 0 – 180 |
| Tracer: ignore specks (px) | Vector | vtracer **`filter_speckle`** | Second-pass speck removal in tracer | 0 – 80 |
| Curve fit iterations | Vector | vtracer **`max_iterations`** | More iterations for spline fitting | 1 – 20 |
| SVG path decimal places | Vector | vtracer **`path_precision`** | Coordinate precision in exported SVG | 1 – 12 |

Sliders update the live vector preview continuously (drag may feel slow on very large/complex photos).

**Tip:** Lower path fidelity + higher “remove blobs” = simpler, bolder stencils that are easier to expose and print.

### Command Line

Two CLI scripts are included for scripting or when you don't need the GUI.

**stencil_creator2.py** (recommended — similar vtracer usage, fewer knobs than the GUI):

```bash
python stencil_creator2.py photo.jpg \
    --detail 0.95 \
    --denoise 18 \
    --min-area 30 \
    --output my-stencil
```

**stencil_creator.py** (older version using the `Vectorizer` class directly):

```bash
python stencil_creator.py photo.jpg --detail 0.65 --output my-stencil
```

Both accept the same main flags:
- `--detail` (float)
- `--denoise` (int)
- `--blur` (float)
- `--threshold` (int)
- `--min-area` (int)
- `--invert` (flag)
- `--output` (base name for output files)

The CLI does not expose all GUI vector options (`max_iterations`, `path_precision`, curve type, etc.).

## Optional: Inkscape for EPS and PDF Export

Screen print shops often prefer EPS; PDF is handy for proofs and Illustrator workflows.

Install Inkscape:

```bash
# macOS
brew install inkscape
```

In the GUI, choose `.eps` or `.pdf` in the save dialog. SVG is written by vtracer; other formats are converted with Inkscape.

## How It Works

1. Load full image (native width × height) → convert to grayscale
2. Denoise (`fastNlMeansDenoising`)
3. Mild Gaussian blur (σ from slider; 5×5 kernel)
4. Adaptive thresholding (Gaussian C, binary)
5. Morphological open (3×3) to remove tiny noise
6. Remove small connected components (blob size slider)
7. Optional line thickening (dilation)
8. Vector trace with vtracer in binary mode
9. Optional EPS/PDF conversion via Inkscape

The GUI runs preprocessing + vectorization after you stop moving a slider (or on mouse release). Changing only **vector** sliders re-traces from the cached center bitmap without re-running OpenCV. The **right preview** is for tuning trace settings; judge final line weight from the **exported file** or the **center** bitmap.

## Tips for Screen Printing

- Take the photo in even, diffuse light with the artwork flat.
- Higher contrast originals vectorize more cleanly.
- Simpler is usually better — aggressive simplification often produces more printable stencils.
- After exporting, you can open the SVG in Illustrator, Affinity, or Inkscape for any final manual cleanup.

## License

Public domain / use as you like. No warranty.

---

Built with OpenCV + vtracer + Dear PyGui.