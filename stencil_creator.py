import argparse
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

from PIL import Image, ImageFilter, ImageOps
from vtracer import Vectorizer  # pip install vtracer


def preprocess_phone_photo(input_path, config):
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("Failed to load image")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise (phone photos are noisy)
    denoised = cv2.fastNlMeansDenoising(gray, h=config['denoise_strength'])

    # Mild blur to reduce junk while keeping outlines
    blurred = cv2.GaussianBlur(denoised, (5, 5), config['blur_radius'])

    # Adaptive threshold for varying "opacity" feel via density
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, config['threshold_offset']
    )

    # Optional invert
    if config['invert']:
        thresh = cv2.bitwise_not(thresh)

    # Remove small noise specks
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Save temp for vtracer
    temp_path = str(Path(input_path).with_suffix('.preprocessed.png'))
    cv2.imwrite(temp_path, cleaned)
    return temp_path


def image_to_vector(input_path, output_base, config):
    preprocessed = preprocess_phone_photo(input_path, config)

    vectorizer = Vectorizer(
        mode="binary",  # single "color"
        color_precision=0,  # not needed
        filter_speckle=config['min_area'],
        corner_threshold=config['corner_threshold'],
        length_threshold=config['detail_level'],
        max_iterations=10,
        hierarchical=0,
        splice_threshold=10
    )

    with open(preprocessed, 'rb') as f:
        svg_data = vectorizer.vectorize(f.read())

    # Save SVG
    svg_path = output_base + ".svg"
    with open(svg_path, 'w') as f:
        f.write(svg_data.decode('utf-8'))

    # Convert to EPS (common for print RIPs)
    eps_path = output_base + ".eps"
    try:
        subprocess.run(['inkscape', svg_path, '--export-filename', eps_path],
                       check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Inkscape not found for EPS export. Install it or use online converter.")

    # Optional AI (Illustrator) via EPS rename or further conversion if needed
    ai_path = output_base + ".ai"
    try:
        # Rough but often works for basic cases
        subprocess.run(['inkscape', svg_path, '--export-filename', ai_path],
                       check=True, capture_output=True)
    except Exception:
        pass  # AI is proprietary, EPS is usually sufficient

    os.remove(preprocessed)  # cleanup
    return svg_path, eps_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input photo path")
    parser.add_argument("--detail", type=float, default=0.7, help="0.0-1.0, lower = simpler/cleaner outlines")
    parser.add_argument("--denoise", type=int, default=15, help="Denoise strength (higher removes more noise)")
    parser.add_argument("--blur", type=float, default=1.5)
    parser.add_argument("--threshold", type=int, default=5, help="Adaptive threshold offset")
    parser.add_argument("--min-area", type=int, default=20)
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--output", default=None)

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
    print(f"Done: {svg} (and EPS)")
