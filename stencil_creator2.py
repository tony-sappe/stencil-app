import argparse
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import vtracer  # Fixed import

def preprocess_phone_photo(input_path, config):
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Failed to load {input_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=config['denoise_strength'])
    blurred = cv2.GaussianBlur(denoised, (5, 5), config['blur_radius'])

    # Adaptive threshold preserves density variations for "opacity" feel
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, config['threshold_offset']
    )

    if config['invert']:
        thresh = cv2.bitwise_not(thresh)

    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    temp_path = str(Path(input_path).with_suffix('.preprocessed.png'))
    cv2.imwrite(temp_path, cleaned)
    return temp_path

def image_to_vector(input_path, output_base, config):
    preprocessed = preprocess_phone_photo(input_path, config)

    # vtracer single-color/binary mode for screen printing
    svg_path = output_base + ".svg"
    vtracer.convert_image_to_svg_py(
        preprocessed,
        svg_path,
        colormode='binary',           # single color
        filter_speckle=config['min_area'],
        corner_threshold=config['corner_threshold'],
        length_threshold=config['detail_level'] * 10,  # scale to vtracer range
        max_iterations=10
    )

    # EPS export (standard for print)
    eps_path = output_base + ".eps"
    try:
        subprocess.run(['inkscape', svg_path, '--export-filename', eps_path],
                       check=True, capture_output=True, text=True)
        print(f"EPS generated: {eps_path}")
    except FileNotFoundError:
        print("Inkscape not found. Install it for EPS/AI export.")
    except subprocess.CalledProcessError as e:
        print(f"EPS export failed: {e.stderr}")

    os.remove(preprocessed)
    return svg_path, eps_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phone photo → clean vector for screen printing")
    parser.add_argument("input", help="Input photo path")
    parser.add_argument("--detail", type=float, default=0.65, help="0.0-1.0 (lower = simpler, bolder outlines)")
    parser.add_argument("--denoise", type=int, default=15, help="Higher removes more phone noise")
    parser.add_argument("--blur", type=float, default=1.5)
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--min-area", type=int, default=25)
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--output", default=None, help="Output base name")

    args = parser.parse_args()

    config = {
        "detail_level": args.detail,
        "denoise_strength": args.denoise,
        "blur_radius": args.blur,
        "threshold_offset": args.threshold,
        "min_area": args.min_area,
        "invert": args.invert,
        "corner_threshold": 60
    }

    output_base = args.output or str(Path(args.input).with_suffix(''))
    svg, eps = image_to_vector(args.input, output_base, config)
    print(f"Vectorization complete: {svg}")
