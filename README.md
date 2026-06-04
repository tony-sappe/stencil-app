# Stencil Creator

Turn phone photos of artwork into clean, single-color vector files suitable for screen printing stencils.

The project provides both an interactive GUI (recommended) and command-line tools. It uses OpenCV for preprocessing (denoising, adaptive thresholding, cleanup) followed by [vtracer](https://github.com/visioncortex/vtracer) for high-quality vectorization.

## Features

- **Live preview GUI** — adjust settings and see the resulting vector update in real time
- Tuned preprocessing for noisy phone-camera photos
- Exports **SVG** (always) + **EPS** (when Inkscape is available)
- Simple controls for detail level, noise removal, threshold, speckle filtering, and corner simplification

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

1. Click **Load Photo** and select an image (JPG, PNG, etc. — HEIC works if `pillow-heif` is also installed).
2. Adjust the sliders. Previews update live (throttled during rapid drags for stability; a final update is forced when you release the mouse so you always see the exact final settings).
3. Use the **Invert Image** checkbox if needed.
4. Click **Save Vector** to export.
   - Always produces a `.svg`
   - Also produces a `.eps` if Inkscape is found in your PATH
5. Use **Open SVG** to quickly view the last saved (or current live) vector in your default viewer.

#### Controls

| Control              | Group        | Effect                                              | Typical Range |
|----------------------|--------------|-----------------------------------------------------|---------------|
| Denoise Strength     | Preprocess   | Removes phone camera noise (fastNlMeans)            | 0 – 40        |
| Blur Radius          | Preprocess   | Gentle smoothing before thresholding                | 0.0 – 5.0     |
| Threshold Offset     | Preprocess   | Bias for adaptive threshold (higher = more white)   | -20 – 20      |
| Adaptive Block Size  | Preprocess   | Neighborhood size for adaptive threshold (odd >=3 per OpenCV) | 3 – 31 (odd) |
| Min Area (speckle)   | Preprocess   | Removes small connected components (both colors)    | 10 – 100      |
| Line Thickening      | Preprocess   | Dilate the final binary (makes lines more durable)  | 0 – 3         |
| Invert Image         | Preprocess   | Swap black/white                                    | —             |
| Detail Level         | Vector       | Curve simplification (maps to vtracer length_threshold in [3.5, 10]; higher = more detail) | 0.35 – 1.0 |
| Corner Threshold     | Vector       | Minimum angle (deg) considered a corner (vtracer 0-180) | 0 – 180 |
| Splice Threshold     | Vector       | Splicing threshold (treated as deg in vtracer)      | 0 – 180       |

Sliders update the live vector preview continuously (drag may feel slow on very large/complex photos).

**Tip:** Lower detail + higher min-area = simpler, bolder stencils that are easier to expose and print.

### Command Line

Two CLI scripts are included for scripting or when you don't need the GUI.

**stencil_creator2.py** (recommended — matches the GUI's vtracer settings):

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

## Optional: Inkscape for EPS Export

Screen print shops often prefer (or require) EPS files.

Install Inkscape:

```bash
# macOS
brew install inkscape
```

Once installed, both the GUI and CLI scripts will automatically generate an `.eps` file next to the `.svg`.

## How It Works

1. Load image → convert to grayscale
2. Denoise (`fastNlMeansDenoising`)
3. Mild Gaussian blur
4. Adaptive thresholding (preserves some density variation)
5. Morphological open to remove tiny noise
6. Vector trace with vtracer in binary mode (`filter_speckle`, `corner_threshold`, `length_threshold`)
7. Optional EPS conversion via Inkscape

The GUI runs the full preprocessing + vectorization pipeline on every slider release so you can see the final stencil result, not just the intermediate binary image.

## Tips for Screen Printing

- Take the photo in even, diffuse light with the artwork flat.
- Higher contrast originals vectorize more cleanly.
- Simpler is usually better — aggressive simplification often produces more printable stencils.
- After exporting, you can open the SVG in Illustrator, Affinity, or Inkscape for any final manual cleanup.

## License

Public domain / use as you like. No warranty.

---

Built with OpenCV + vtracer + Dear PyGui.
