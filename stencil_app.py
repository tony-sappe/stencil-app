#!/usr/bin/env python3
"""
stencil_app.py - Phone Photo → Screen Print Vector (Dear PyGui)
Brand new implementation per specification.md + prior conversation.

Core flow:
- Load image (robust DPG file dialog, PIL + cv2 fallback for HEIC etc.)
- Preprocess (denoise, blur, adaptive thresh, optional invert, open morph) -> binary for tracing
- vtracer.convert_image_to_svg_py (binary; preprocess owns speckle cleanup; vector sliders control curve fidelity)
- 3 live previews: orig photo, exact preproc binary (vtracer input), rasterized SVG preview (approximate); debounced updates
- Realtime slider/checkbox callbacks update everything
- Export SVG, EPS, or PDF (user choice; EPS/PDF via Inkscape when installed)
- Temps live in a private per-process directory and are removed on exit
- No Tkinter

This version uses documented DPG dynamic textures (np arrays, not python lists) + slider debounce (process only after you stop moving or release) + mouse-release force for reliable live previews without unnecessary work while dragging.
"""

import argparse
import atexit
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import dearpygui.dearpygui as dpg
import vtracer
from PIL import Image, ImageOps

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass


def _verbose_log(msg: str, *, verbose: bool) -> None:
    if verbose:
        print(msg)


# vtracer length_threshold: lower = shorter segments = closer to the binary input
_LENGTH_MIN = 3.5
_LENGTH_MAX = 10.0
_DETAIL_MIN = 0.35
_DETAIL_MAX = 1.0

# SVG→PNG rasterization: use image width (capped) so previews stay close to export/PDF.
_INLINE_RASTER_MAX = 2048
_MODAL_RASTER_MAX = 2048
_INLINE_TEXTURE_SIZE = 512
# Luma below this becomes ink after the raster is composited onto white.
# 128 matches stroke width; a higher cutoff fattens anti-aliased edges.
_PREVIEW_BINARIZE_THRESHOLD = 128
# child_window(border=True) shrinks the interior; pad outer size so image buttons fit without scrollbars
_PREVIEW_CHILD_PAD_X = 30
_PREVIEW_CHILD_PAD_Y = 16
# Pillow raises DecompressionBombError above 2x MAX_IMAGE_PIXELS. OpenCV's own
# cap is ~2^30 px, so a bomb Pillow refuses must not be handed to cv2.imread.
_PIL_PIXEL_LIMIT = int(Image.MAX_IMAGE_PIXELS or 89_478_485)
_MAX_IMAGE_PIXELS = _PIL_PIXEL_LIMIT * 2


def _make_work_dir() -> Path:
    """Mode-0700 directory unique to this process. Callers must delete it."""
    path = Path(tempfile.mkdtemp(prefix="stencil_creator_"))
    path.chmod(0o700)
    return path


def _composite_on_white(pil: Image.Image) -> Image.Image:
    """Straight-alpha images composite onto white. RGB-only convert("L") drops alpha."""
    if pil.mode == "P" and ("transparency" in pil.info or "A" in pil.getbands()):
        pil = pil.convert("RGBA")
    if pil.mode in ("RGBA", "LA"):
        rgba = pil.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba)
    return pil


def _scale_to_u8(arr: np.ndarray, full_scale: float) -> np.ndarray:
    scaled = np.rint(arr.astype(np.float64) * (255.0 / float(full_scale)))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _pil_to_gray_u8(pil: Image.Image) -> np.ndarray:
    """Grayscale uint8. Wide modes are scaled, not clipped at 255."""
    pil = _composite_on_white(pil)
    if pil.mode in ("I;16", "I;16B", "I;16L", "I;16N"):
        return _scale_to_u8(np.array(pil), 65535.0)
    if pil.mode == "I":
        arr = np.array(pil)
        peak = float(arr.max()) if arr.size else 0.0
        if peak <= 255.0:
            return np.clip(arr, 0, 255).astype(np.uint8)
        full = 65535.0 if peak <= 65535.0 else max(peak, 1.0)
        return _scale_to_u8(arr, full)
    if pil.mode == "F":
        arr = np.array(pil, dtype=np.float64)
        peak = float(np.nanmax(arr)) if arr.size else 1.0
        full = 1.0 if peak <= 1.0 else (65535.0 if peak <= 65535.0 else max(peak, 1.0))
        return _scale_to_u8(np.nan_to_num(arr, nan=full), full)
    if pil.mode != "L":
        pil = pil.convert("L")
    return np.array(pil, dtype=np.uint8)


def _load_raster(path):
    """Open a source image. Returns (RGBA on white, gray uint8).

    Raises Image.DecompressionBombError when the pixel count is past Pillow's
    hard limit. Other failures propagate so the caller can try OpenCV.
    """
    pil = ImageOps.exif_transpose(Image.open(path))
    width, height = pil.size
    if width * height > _MAX_IMAGE_PIXELS:
        raise Image.DecompressionBombError(
            f"{width}×{height} exceeds {_MAX_IMAGE_PIXELS} pixels"
        )
    return _composite_on_white(pil).convert("RGBA"), _pil_to_gray_u8(pil)


def _cv_image_to_gray_u8(img: np.ndarray) -> Optional[np.ndarray]:
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        bgra = img
        if bgra.dtype == np.uint16:
            bgra = _scale_to_u8(bgra, 65535.0)
        gray = cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
        alpha = bgra[:, :, 3].astype(np.float32) / 255.0
        return np.clip(np.rint(gray * alpha + 255.0 * (1.0 - alpha)), 0, 255).astype(np.uint8)
    if img.dtype == np.uint16:
        img = _scale_to_u8(img, 65535.0)
    if img.ndim == 2:
        if img.dtype == np.uint8:
            return img
        return _scale_to_u8(img, float(np.max(img) or 1))
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return None


class StencilApp:
    TRACE_MODES = ("Smooth (spline)", "Polygon", "Pixel-accurate")
    SAVE_EXTENSIONS = (".svg", ".eps", ".pdf")

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.input_path = None
        self.current_preprocessed = None
        self.last_svg = None
        self._orig_pil = None  # cached for faster repeated updates

        # Private 0700 directory so two copies of the app, and a planted symlink
        # in the shared temp dir, cannot clobber each other's files.
        self.temp_dir = _make_work_dir()
        self._temps_cleaned = False
        atexit.register(self._cleanup_temps)
        self.preproc_temp = str(self.temp_dir / "preproc.png")
        self.preproc_staging = str(self.temp_dir / "preproc_next.png")
        self.svg_temp = str(self.temp_dir / "vector.svg")
        self.svg_staging = str(self.temp_dir / "vector_next.svg")
        self.vec_preview_temp = str(self.temp_dir / "vecprev.png")
        self.vec_preview_modal_temp = str(self.temp_dir / "vecprev_modal.png")

        self.texture_size = _INLINE_TEXTURE_SIZE
        self.preview_size = 470  # preview image button size (letterboxed; preserves aspect ratio)
        self.preview_child_w = self.preview_size + _PREVIEW_CHILD_PAD_X
        self.preview_child_h = self.preview_size + _PREVIEW_CHILD_PAD_Y
        self._preview_cache = {"orig": None, "proc": None, "vec": None}
        self._image_dims = (0, 0)  # (width, height) of processed bitmap
        self._modal_tex_size = (0, 0)  # (width, height) of modal_texture buffer

        self._dirty = False

        # Debounce for sliders: don't start heavy processing (preprocess + vtracer) until
        # the user has stopped moving the slider for this long (or releases the mouse).
        self._last_slider_change = 0.0
        self._debounce_delay = 0.30  # seconds

        self._last_config = None
        self._last_binary = None
        self._busy = False

        self._build_ui()

    def _create_checkerboard_texture_data(self, size, square=16):
        """High-contrast checker (numpy for compact mem)."""
        data = np.zeros((size, size, 4), dtype=np.float32)
        data[..., 3] = 1.0
        for yy in range(0, size, square):
            for xx in range(0, size, square):
                val = 0.12 if ((xx // square + yy // square) % 2 == 0) else 0.55
                data[yy:yy + square, xx:xx + square, :3] = val
        return data

    def _pil_to_texture_data(self, pil_img, size=None, resample=Image.LANCZOS):
        """Letterbox on a square canvas. Returns np.float32 (h, w, 4)."""
        if size is None:
            size = self.texture_size
        canvas = Image.new("RGBA", (size, size), (25, 25, 25, 255))
        pil_img = pil_img.convert("RGBA")
        pil_img.thumbnail((size, size), resample)
        x = (size - pil_img.width) // 2
        y = (size - pil_img.height) // 2
        canvas.paste(pil_img, (x, y))
        return np.array(canvas, dtype=np.float32) / 255.0

    def _fit_pil_to_texture_data(self, pil_img, canvas_w, canvas_h, resample=Image.LANCZOS):
        """Letterbox on a rectangular canvas (for the zoom modal)."""
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (25, 25, 25, 255))
        pil_img = pil_img.convert("RGBA")
        pil_img.thumbnail((canvas_w, canvas_h), resample)
        x = (canvas_w - pil_img.width) // 2
        y = (canvas_h - pil_img.height) // 2
        canvas.paste(pil_img, (x, y))
        return np.array(canvas, dtype=np.float32) / 255.0

    @staticmethod
    def _svg_export_width(img_width: int, max_dim: int) -> int:
        """Inkscape --export-width: match bitmap width up to max_dim."""
        if img_width <= 0:
            return max_dim
        return int(min(img_width, max_dim))

    def _build_ui(self):
        dpg.create_context()

        # Dynamic textures (documented for live/set_value updates) created early.
        with dpg.texture_registry(tag="tex_reg"):
            checker = self._create_checkerboard_texture_data(self.texture_size)
            for ttag in ("orig_texture", "proc_texture", "vec_texture"):
                dpg.add_dynamic_texture(
                    width=self.texture_size,
                    height=self.texture_size,
                    default_value=checker,
                    tag=ttag,
                )
            dpg.add_dynamic_texture(
                width=4,
                height=4,
                default_value=self._create_checkerboard_texture_data(4),
                tag="modal_tex_stub",
            )
            dpg.add_dynamic_texture(
                width=4,
                height=4,
                default_value=self._create_checkerboard_texture_data(4),
                tag="modal_texture",
            )
        _verbose_log("[INFO] Dynamic texture registry + 3 checker placeholders created.", verbose=self.verbose)

        dpg.create_viewport(
            title="Photo → Image Vector",
            width=1540,
            height=880,
            resizable=True,
        )
        dpg.setup_dearpygui()
        _verbose_log("[INFO] Viewport + DPG setup done.", verbose=self.verbose)

        with dpg.theme(tag="preview_image_btn_theme"):
            with dpg.theme_component(dpg.mvImageButton):
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 0, 0, 0), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(
                    dpg.mvThemeCol_ButtonHovered, (50, 60, 75, 90), category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_ButtonActive, (70, 90, 110, 120), category=dpg.mvThemeCat_Core
                )

        with dpg.window(tag="main_win"):
            with dpg.group(horizontal=True):
                dpg.add_button(label="Load Photo", callback=self.load_image, width=110)
                dpg.add_button(label="Save Vector", callback=self.save_vector, width=110)
                dpg.add_button(label="Open SVG", callback=self.open_svg, width=110)

            # Fixed-height panels: width 690, tall enough for controls but not full-window stretch.
            settings_col_w = 690
            settings_panel_h = 340
            settings_slider_w = 360
            with dpg.group(horizontal=True, tag="settings_row"):
                with dpg.child_window(
                    tag="settings_col_preprocess",
                    width=settings_col_w,
                    height=settings_panel_h,
                    border=False,
                    no_scrollbar=True,
                ):
                    with dpg.collapsing_header(label="Preprocessing settings", default_open=True):
                        dpg.add_slider_int(
                            label="Noise reduction",
                            tag="slider_denoise",
                            default_value=15,
                            min_value=0,
                            max_value=40,
                            callback=self.update_preview,
                            width=settings_slider_w,
                        )
                        dpg.add_slider_float(
                            label="Blur before threshold (σ)",
                            tag="slider_blur",
                            default_value=1.5,
                            min_value=0.0,
                            max_value=5.0,
                            callback=self.update_preview,
                            width=settings_slider_w,
                        )
                        dpg.add_slider_int(
                            label="Threshold bias (C)",
                            tag="slider_threshold",
                            default_value=5,
                            min_value=-20,
                            max_value=20,
                            callback=self.update_preview,
                            width=settings_slider_w,
                        )
                        dpg.add_slider_int(
                            label="Local threshold window",
                            tag="slider_block_size",
                            default_value=11,
                            min_value=3,
                            max_value=31,
                            callback=self.update_preview,
                            width=settings_slider_w,
                        )
                        dpg.add_slider_int(
                            label="Remove blobs smaller than (px²)",
                            tag="slider_min_area",
                            default_value=25,
                            min_value=10,
                            max_value=100,
                            callback=self.update_preview,
                            width=settings_slider_w,
                        )
                        dpg.add_slider_float(
                            label="Thicken stencil lines (px)",
                            tag="slider_line_width",
                            default_value=0.0,
                            min_value=0.0,
                            max_value=12.0,
                            format="%.1f",
                            callback=self.update_preview,
                            width=settings_slider_w,
                        )
                        dpg.add_checkbox(
                            label="Invert Image",
                            tag="check_invert",
                            default_value=False,
                            callback=self.update_preview,
                        )

                with dpg.child_window(
                    tag="settings_col_vector",
                    width=settings_col_w,
                    height=settings_panel_h,
                    border=False,
                    no_scrollbar=True,
                ):
                    with dpg.collapsing_header(label="Vector settings", default_open=True):
                        dpg.add_text(
                            "Trace raster uses image width (up to 2048px) — click preview to zoom.",
                            color=(160, 180, 200),
                        )
                        dpg.add_slider_float(
                            label="Path fidelity",
                            tag="slider_detail",
                            default_value=1.0,
                            min_value=_DETAIL_MIN,
                            max_value=_DETAIL_MAX,
                            callback=self.update_preview,
                            width=settings_slider_w,
                        )
                        dpg.add_slider_int(
                            label="Corner smoothing (°)",
                            tag="slider_corner",
                            default_value=30,
                            min_value=0,
                            max_value=180,
                            callback=self.update_preview,
                            width=settings_slider_w,
                        )
                        dpg.add_slider_int(
                            label="Join straight segments (°)",
                            tag="slider_splice_threshold",
                            default_value=45,
                            min_value=0,
                            max_value=180,
                            callback=self.update_preview,
                            width=settings_slider_w,
                        )
                        dpg.add_slider_int(
                            label="Tracer: ignore specks (px)",
                            tag="slider_vector_speckle",
                            default_value=0,
                            min_value=0,
                            max_value=80,
                            callback=self.update_preview,
                            width=settings_slider_w,
                        )
                        dpg.add_slider_int(
                            label="Curve fit iterations",
                            tag="slider_max_iterations",
                            default_value=10,
                            min_value=1,
                            max_value=20,
                            callback=self.update_preview,
                            width=settings_slider_w,
                        )
                        dpg.add_slider_int(
                            label="SVG path decimal places",
                            tag="slider_path_precision",
                            default_value=8,
                            min_value=1,
                            max_value=12,
                            callback=self.update_preview,
                            width=settings_slider_w,
                        )
                        dpg.add_button(
                            label="Preset: max trace fidelity",
                            callback=self._apply_match_preprocess_preset,
                            width=280,
                        )
                        dpg.add_combo(
                            label="Curve type",
                            tag="combo_trace_mode",
                            items=list(self.TRACE_MODES),
                            default_value="Polygon",
                            callback=self.update_preview,
                            width=220,
                        )

            # Labels above bordered preview boxes (images fill the boxes exactly)
            with dpg.group(horizontal=True):
                with dpg.group():
                    dpg.add_text("Original photo")
                    dpg.add_text(
                        "Full image — SVG size matches this",
                        color=(160, 180, 200),
                    )
                    with dpg.child_window(
                        tag="col_orig",
                        border=True,
                        width=self.preview_child_w,
                        height=self.preview_child_h,
                        no_scrollbar=True,
                    ):
                        pass

                with dpg.group():
                    dpg.add_text("Stencil bitmap")
                    dpg.add_text(
                        "Exact OpenCV binary (vtracer input)",
                        color=(160, 180, 200),
                    )
                    with dpg.child_window(
                        tag="col_proc",
                        border=True,
                        width=self.preview_child_w,
                        height=self.preview_child_h,
                        no_scrollbar=True,
                    ):
                        pass

                with dpg.group():
                    dpg.add_text("Traced SVG preview", color=(140, 220, 140))
                    dpg.add_text(
                        "Higher-res raster — click to enlarge",
                        color=(160, 200, 160),
                    )
                    with dpg.child_window(
                        tag="col_vec",
                        border=True,
                        width=self.preview_child_w,
                        height=self.preview_child_h,
                        no_scrollbar=True,
                    ):
                        pass

            dpg.add_text(
                "Click any preview to open a full-window view (export-sized for trace).",
                color=(140, 160, 180),
            )
            dpg.add_spacer(height=4)
            with dpg.group(horizontal=True):
                dpg.add_text(tag="loading_indicator", default_value="", color=(255, 180, 100))
                dpg.add_text(
                    tag="status_text",
                    default_value=(
                        "Load a photo to begin. Center = exact stencil bitmap; "
                        "right = higher-res trace raster. Click a preview to zoom."
                    ),
                )

        with dpg.window(
            tag="preview_modal",
            label="Preview",
            modal=True,
            show=False,
            no_resize=True,
            width=920,
            height=720,
        ):
            dpg.add_text("", tag="modal_caption")
            dpg.add_image(
                tag="modal_image",
                texture_tag="modal_texture",
                width=800,
                height=600,
            )
            dpg.add_button(label="Close", callback=self._close_preview_modal, width=120)

        # File dialogs (robust path extraction in callbacks to handle .* filter etc.)
        with dpg.file_dialog(
            tag="load_dialog",
            show=False,
            callback=self._on_load_callback,
            width=780,
            height=580,
            directory_selector=False,
        ):
            dpg.add_file_extension(".*")
            dpg.add_file_extension(".jpg", color=(0, 220, 100, 255))
            dpg.add_file_extension(".jpeg", color=(0, 220, 100, 255))
            dpg.add_file_extension(".png", color=(0, 220, 100, 255))
            dpg.add_file_extension(".bmp", color=(0, 220, 100, 255))
            dpg.add_file_extension(".webp", color=(0, 220, 100, 255))
            dpg.add_file_extension(".tiff", color=(0, 220, 100, 255))

        with dpg.file_dialog(
            tag="save_dialog",
            show=False,
            callback=self._on_save_callback,
            width=780,
            height=580,
            directory_selector=False,
            default_filename="stencil.svg",
        ):
            dpg.add_file_extension(".svg", color=(100, 180, 255, 255))
            dpg.add_file_extension(".eps", color=(255, 200, 100, 255))
            dpg.add_file_extension(".pdf", color=(255, 120, 120, 255))

        with dpg.handler_registry(tag="global_mouse_handlers"):
            dpg.add_mouse_release_handler(callback=self._on_global_mouse_release)

        dpg.set_primary_window("main_win", True)
        dpg.set_exit_callback(self._on_exit)
        dpg.show_viewport()
        _verbose_log("[INFO] Viewport shown.", verbose=self.verbose)

        # Schedule adding the (texture-bound) image widgets once DPG is fully up.
        def _initial_add_preview_images(sender, app_data, user_data):
            _verbose_log("[INFO] Adding initial preview images...", verbose=self.verbose)
            for col_tag, img_tag, tex_tag, kind in [
                ("col_orig", "img_orig", "orig_texture", "orig"),
                ("col_proc", "img_proc", "proc_texture", "proc"),
                ("col_vec", "img_vec", "vec_texture", "vec"),
            ]:
                if dpg.does_item_exist(img_tag):
                    dpg.delete_item(img_tag)
                dpg.add_image_button(
                    texture_tag=tex_tag,
                    tag=img_tag,
                    width=self.preview_size,
                    height=self.preview_size,
                    parent=col_tag,
                    callback=self._on_preview_button,
                    user_data=kind,
                    background_color=(0, 0, 0, 0),
                )
                dpg.bind_item_theme(img_tag, "preview_image_btn_theme")
            _verbose_log("[INFO] Initial preview images added.", verbose=self.verbose)

        dpg.set_frame_callback(2, _initial_add_preview_images)
        _verbose_log("[INFO] Initial preview image add scheduled for frame 2.", verbose=self.verbose)

        # Start the debounce/idle poller that will trigger processing after the user stops
        # moving a slider for _debounce_delay seconds.
        self._schedule_debounce_check()

    def get_config(self):
        return {
            "detail_level": dpg.get_value("slider_detail"),
            "denoise_strength": dpg.get_value("slider_denoise"),
            "blur_radius": dpg.get_value("slider_blur"),
            "threshold_offset": dpg.get_value("slider_threshold"),
            "block_size": dpg.get_value("slider_block_size"),
            "min_area": dpg.get_value("slider_min_area"),
            "line_width_px": float(dpg.get_value("slider_line_width")),
            "corner_threshold": dpg.get_value("slider_corner"),
            "splice_threshold": dpg.get_value("slider_splice_threshold"),
            "trace_mode": dpg.get_value("combo_trace_mode"),
            "vector_speckle": dpg.get_value("slider_vector_speckle"),
            "max_iterations": dpg.get_value("slider_max_iterations"),
            "path_precision": dpg.get_value("slider_path_precision"),
            "invert": dpg.get_value("check_invert"),
        }

    def _length_threshold_from_detail(self, detail_level: float) -> float:
        """Map UI fidelity (high = match binary) to vtracer length_threshold (low = more segments)."""
        span = _DETAIL_MAX - _DETAIL_MIN
        if span <= 0:
            return _LENGTH_MIN
        t = (_DETAIL_MAX - detail_level) / span
        return _LENGTH_MIN + t * (_LENGTH_MAX - _LENGTH_MIN)

    def _trace_mode_to_vtracer(self, label: str) -> str:
        return {
            "Smooth (spline)": "spline",
            "Polygon": "polygon",
            "Pixel-accurate": "none",
        }.get(label, "polygon")

    def _vtracer_kwargs(self, config: dict) -> dict:
        """Build vtracer args. Preprocess already applies Min Area; default vector speckle is 0."""
        return {
            "colormode": "binary",
            "filter_speckle": int(config.get("vector_speckle", 0)),
            "mode": self._trace_mode_to_vtracer(config.get("trace_mode", "Polygon")),
            "corner_threshold": int(config["corner_threshold"]),
            "length_threshold": self._length_threshold_from_detail(float(config["detail_level"])),
            "splice_threshold": int(config.get("splice_threshold", 45)),
            "max_iterations": int(config.get("max_iterations", 10)),
            "path_precision": int(config.get("path_precision", 8)),
        }

    def _apply_match_preprocess_preset(self, sender=None, app_data=None, user_data=None):
        """Apply vtracer settings for minimum simplification (right preview remains rasterized SVG)."""
        dpg.set_value("slider_detail", _DETAIL_MAX)
        dpg.set_value("slider_corner", 0)
        dpg.set_value("slider_splice_threshold", 180)
        dpg.set_value("combo_trace_mode", "Pixel-accurate")
        dpg.set_value("slider_vector_speckle", 0)
        dpg.set_value(
            "status_text",
            "Max-fidelity preset applied — release slider or wait for update "
            "(right pane still rasterized SVG, not a copy of center).",
        )
        self.update_preview()

    def _binarize_preview_pil(self, pil_img: Image.Image) -> Image.Image:
        """Ink is luma below the cutoff. 128 keeps stroke width; higher fattens AA."""
        arr = np.array(pil_img.convert("L"), dtype=np.uint8)
        return Image.fromarray(
            np.where(arr < _PREVIEW_BINARIZE_THRESHOLD, 0, 255).astype(np.uint8)
        )

    def _load_inkscape_png_as_gray(self, png_path: str) -> Optional[Image.Image]:
        try:
            # Inkscape writes straight-alpha coverage. convert("L") ignores it
            # and paints every covered pixel as solid ink.
            return _composite_on_white(Image.open(png_path)).convert("L")
        except Exception:
            return None

    def _on_preview_button(self, sender=None, app_data=None, user_data=None):
        if not self.input_path or not user_data:
            return
        self._open_preview_modal(user_data)

    def _modal_canvas_size(self):
        try:
            vw = dpg.get_viewport_client_width()
            vh = dpg.get_viewport_client_height()
        except Exception:
            vw, vh = 1200, 800
        return max(320, int(vw * 0.88)), max(280, int(vh * 0.78))

    def _replace_modal_texture(self, data: np.ndarray):
        ch, cw = int(data.shape[0]), int(data.shape[1])
        if self._modal_tex_size == (cw, ch) and dpg.does_item_exist("modal_texture"):
            dpg.set_value("modal_texture", data)
            dpg.configure_item("modal_image", width=cw, height=ch)
            return
        if dpg.does_item_exist("modal_texture"):
            # Image must stop referencing the texture before delete (DPG keeps the alias otherwise).
            dpg.configure_item("modal_image", texture_tag="modal_tex_stub")
            dpg.delete_item("modal_texture")
        dpg.add_dynamic_texture(cw, ch, data, tag="modal_texture", parent="tex_reg")
        self._modal_tex_size = (cw, ch)
        dpg.configure_item("modal_image", texture_tag="modal_texture", width=cw, height=ch)

    def _open_preview_modal(self, kind, sender=None, app_data=None, user_data=None):
        if not self.input_path:
            return
        pw, ph = self._image_dims
        dim_note = f"{pw}×{ph} px" if pw and ph else ""
        titles = {
            "orig": "Original photo",
            "proc": "Stencil bitmap (vtracer input)",
            "vec": "Traced SVG (export-width raster)",
        }
        pil = None
        resample = Image.LANCZOS

        if kind == "orig":
            pil = self._preview_cache.get("orig")
        elif kind == "proc":
            pil = self._preview_cache.get("proc")
            resample = Image.NEAREST
        elif kind == "vec":
            svg_path = self.last_svg
            if svg_path and os.path.isfile(svg_path) and pw > 0:
                export_w = self._svg_export_width(pw, _MODAL_RASTER_MAX)
                ok, _err = self._render_svg_to_png(
                    svg_path,
                    self.vec_preview_modal_temp,
                    export_width=export_w,
                    timeout=90,
                )
                if ok:
                    raw = self._load_inkscape_png_as_gray(self.vec_preview_modal_temp)
                    if raw is not None:
                        pil = self._binarize_preview_pil(raw)
            if pil is None:
                pil = self._preview_cache.get("vec")

        if pil is None:
            dpg.set_value("status_text", "Nothing to show — load a photo and wait for preview.")
            return

        canvas_w, canvas_h = self._modal_canvas_size()
        data = self._fit_pil_to_texture_data(pil, canvas_w, canvas_h, resample=resample)
        self._replace_modal_texture(data)
        cap = titles.get(kind, "Preview")
        if dim_note:
            cap = f"{cap}  —  {dim_note}"
        dpg.set_value("modal_caption", cap)
        dpg.show_item("preview_modal")
        try:
            dpg.focus_item("preview_modal")
        except Exception:
            pass

    def _close_preview_modal(self, sender=None, app_data=None, user_data=None):
        if dpg.does_item_exist("preview_modal"):
            dpg.hide_item("preview_modal")

    def preprocess(self, img_path, config):
        """PIL-first load (HEIC if pillow-heif registered) then cv2 binary prep."""
        gray = None
        try:
            _display, gray = _load_raster(img_path)
        except Image.DecompressionBombError:
            return None
        except Exception:
            gray = None

        if gray is None:
            try:
                raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            except Exception:
                return None
            if raw is None or raw.shape[0] * raw.shape[1] > _MAX_IMAGE_PIXELS:
                return None
            gray = _cv_image_to_gray_u8(raw)

        if gray is None:
            return None

        denoised = cv2.fastNlMeansDenoising(gray, h=config["denoise_strength"])
        # sigma 0 means "derive from the 5×5 kernel" in OpenCV (~1.1), which is
        # stronger than a small positive sigma. The slider's 0 is no blur.
        sigma = float(config["blur_radius"])
        if sigma > 0:
            blurred = cv2.GaussianBlur(denoised, (5, 5), sigma)
        else:
            blurred = denoised

        block_size = int(config.get("block_size", 11))
        block_size = max(3, block_size | 1)  # must be odd and >=3
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, config["threshold_offset"]
        )

        if config["invert"]:
            thresh = cv2.bitwise_not(thresh)

        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # Pre-filter small connected components using the Min Area param *before* handing to vtracer.
        # This prevents "overflow" panics in visioncortex's cluster builder (which has a hard
        # limit on number of clusters, ~65k labels) on high-res or speckly images.
        min_area = int(config.get("min_area", 25))
        cleaned = self._filter_small_components(cleaned, min_area)

        cleaned = self._thicken_ink_lines(cleaned, float(config.get("line_width_px", 0.0)))

        return cleaned

    def _thicken_ink_lines(self, binary: np.ndarray, width_px: float) -> np.ndarray:
        """Thicken the strokes. Those are whichever color covers less of the image.

        Invert runs before this, so a black line that becomes white is still the
        smaller region and still grows.
        """
        if width_px <= 0 or binary is None:
            return binary
        # Radius tracks the slider. The old max(3, round|1) kernel kept
        # 0.1 through 3.4 on the same 3×3 ellipse.
        radius = max(1, int(float(width_px) + 0.5))
        ksize = radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        dark = int(np.count_nonzero(binary == 0))
        # Tie (equal area) keeps the usual black ink.
        if dark <= binary.size - dark:
            ink = (binary == 0).astype(np.uint8) * 255
            ink = cv2.dilate(ink, kernel, iterations=1)
            return np.where(ink > 0, 0, 255).astype(np.uint8)
        ink = (binary == 255).astype(np.uint8) * 255
        ink = cv2.dilate(ink, kernel, iterations=1)
        return np.where(ink > 0, 255, 0).astype(np.uint8)

    def _filter_small_components(self, binary: np.ndarray, min_area: int) -> np.ndarray:
        """Remove small connected components (8-way) of *either* color below min_area by flipping them.
        Handles both small 255 specks and small 0 specks. Preserves large structures exactly.
        This is the key guard against visioncortex cluster "overflow" panics (internal label limit).
        """
        if min_area <= 1 or binary is None:
            return binary
        cleaned = binary.copy()
        # 1. small 255 (white) specks
        num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                cleaned[labels == i] = 0
        # 2. small 0 (black) specks - by inverting
        inv = cv2.bitwise_not(binary)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                cleaned[labels == i] = 255
        return cleaned

    def _flush_preview_update(self):
        """Run preprocess + vector now (initial load or slider release)."""
        if not self.input_path:
            return
        self._dirty = False
        self._do_update_preview()

    def _run_subprocess(self, args, timeout, check=True):
        """Run args in their own process group so a timeout kills helpers too."""
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                proc.kill()
            try:
                proc.communicate(timeout=1)
            except Exception:
                pass
            raise
        if check and proc.returncode != 0:
            raise subprocess.CalledProcessError(
                proc.returncode, args, output=out, stderr=err
            )
        return out, err

    def _inkscape_export(self, svg_path, out_path, export_type=None, timeout=60):
        """Convert SVG to another vector format via Inkscape. Returns (ok, error_message)."""
        out_path = str(out_path)
        if export_type is None:
            export_type = Path(out_path).suffix.lstrip(".").lower()
        try:
            self._run_subprocess(
                [
                    "inkscape",
                    svg_path,
                    f"--export-type={export_type}",
                    "--export-filename",
                    out_path,
                ],
                timeout=timeout,
            )
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                return True, None
            return False, "export produced no output file"
        except FileNotFoundError:
            return False, "inkscape not found in PATH"
        except subprocess.CalledProcessError as e:
            err = (e.stderr or e.stdout or str(e)).strip()
            return False, err[:240] if err else "inkscape export failed"
        except subprocess.TimeoutExpired:
            return False, "inkscape export timed out"

    def _render_svg_to_png(self, svg_path, png_path, export_width=380, timeout=30):
        """Rasterize SVG with Inkscape. Returns (ok, error_message).

        Quick Look is not used: its thumbnail is a corner speck, not the stencil,
        and a filename search in the temp dir can rename the preprocess PNG.
        """
        export_width = max(64, int(export_width))
        try:
            self._run_subprocess(
                [
                    "inkscape",
                    svg_path,
                    "--export-type=png",
                    "--export-filename",
                    png_path,
                    f"--export-width={export_width}",
                ],
                timeout=timeout,
            )
        except FileNotFoundError:
            return False, (
                "Right preview unavailable — install Inkscape to rasterize SVG."
            )
        except subprocess.TimeoutExpired:
            return False, "Inkscape timed out while rasterizing the trace."
        except subprocess.CalledProcessError as e:
            err = (e.stderr or e.stdout or str(e)).strip()
            detail = (err or "export failed")[:180]
            return False, f"Inkscape could not rasterize the trace: {detail}"
        if not os.path.isfile(png_path) or os.path.getsize(png_path) == 0:
            return False, "Inkscape did not write a preview PNG."
        return True, ""

    def _trace_to_svg(self, src, dest, config):
        """Run vtracer. Return None on success, or the error (panics are BaseException)."""
        try:
            vtracer.convert_image_to_svg_py(src, dest, **self._vtracer_kwargs(config))
        except (KeyboardInterrupt, SystemExit, GeneratorExit):
            raise
        except BaseException as exc:
            return exc
        return None

    def _discard_staging(self):
        for path in (getattr(self, "preproc_staging", None), getattr(self, "svg_staging", None)):
            try:
                if path and os.path.isfile(path):
                    os.unlink(path)
            except Exception:
                pass

    def _rearm_if_config_drifted(self, sampled: dict) -> None:
        """A long trace can drop queued slider callbacks. The widget value is already new."""
        try:
            latest = self.get_config()
        except Exception:
            return
        if any(latest.get(key) != sampled.get(key) for key in sampled):
            self._dirty = True
            self._last_slider_change = time.time()

    def _preview_matches_sliders(self) -> bool:
        if not self._last_config or not self.current_preprocessed:
            return False
        if not os.path.isfile(self.current_preprocessed):
            return False
        try:
            live = self.get_config()
        except Exception:
            return False
        return all(live.get(key) == self._last_config.get(key) for key in self._last_config)

    def update_preview(self, sender=None, app_data=None, user_data=None):
        """Entry point from *every* slider/checkbox change.

        We only *mark* that an update is needed + record the time of the last change.
        We never start the expensive preprocess + vtracer work while the user is
        actively dragging.

        Real processing is triggered later by:
        - the global mouse release handler (immediate final update on release)
        - the idle debounce poller (after _debounce_delay of no further changes)

        This gives the requested "don't start processing until I stop moving the slider" behavior.
        """
        if not self.input_path:
            return
        self._last_slider_change = time.time()
        self._dirty = True
        # Do NOT call _do_update_preview() here (and do not show "Processing" yet).
        # The triggers below will start the work (and set the indicator).

    def _do_update_preview(self):
        """Heavy work: preprocess + vtracer + render to texture data.

        Files, caches, and textures commit together only after the trace succeeds.
        A failed trace leaves the previous preview in place.
        """
        if self._busy:
            self._dirty = True
            self._last_slider_change = time.time()
            return
        if not self.input_path:
            return
        self._busy = True
        sampled = None
        try:
            dpg.set_value("loading_indicator", "⏳ Processing...")
            dpg.set_value("status_text", "Processing…")
            try:
                dpg.split_frame()
            except Exception:
                pass
            config = self.get_config()
            sampled = dict(config)
            preprocess_keys = [
                "denoise_strength", "blur_radius", "threshold_offset", "invert",
                "min_area", "block_size", "line_width_px",
            ]
            vector_keys = [
                "detail_level", "corner_threshold", "splice_threshold",
                "trace_mode", "vector_speckle", "max_iterations", "path_precision",
            ]
            input_changed = self.input_path != getattr(self, "_last_input_path", None)
            preprocess_config_changed = (
                self._last_config is None
                or any(config[k] != self._last_config.get(k, None) for k in preprocess_keys)
            )
            vector_config_changed = (
                self._last_config is None
                or any(config[k] != self._last_config.get(k, None) for k in vector_keys)
            )
            do_preprocess = (
                self._last_binary is None
                or input_changed
                or preprocess_config_changed
                or not os.path.isfile(self.preproc_temp)
            )
            do_vector = (
                do_preprocess
                or vector_config_changed
                or not (self.last_svg and os.path.isfile(self.svg_temp))
            )

            processed = self._last_binary
            trace_png = self.preproc_temp
            if do_preprocess:
                processed = self.preprocess(self.input_path, config)
                if processed is None:
                    dpg.set_value(
                        "status_text",
                        "Failed to load/process image. Preview left unchanged.",
                    )
                    return
                if not cv2.imwrite(self.preproc_staging, processed):
                    dpg.set_value(
                        "status_text",
                        "Could not write the stencil bitmap. Preview left unchanged.",
                    )
                    return
                trace_png = self.preproc_staging

            svg_for_preview = self.svg_temp
            traced_new = False
            if do_vector:
                err = self._trace_to_svg(trace_png, self.svg_staging, config)
                if err is not None:
                    self._discard_staging()
                    dpg.set_value(
                        "status_text",
                        f"Vectorize failed (overflow or bad params): {err}. "
                        "Try ↑ Min Area (speckle), ↑ Denoise, or adjust Threshold. "
                        "Preview left unchanged.",
                    )
                    return
                svg_for_preview = self.svg_staging
                traced_new = True

            if self._orig_pil is not None and not input_changed:
                orig_pil = self._orig_pil
            else:
                try:
                    orig_pil, _gray = _load_raster(self.input_path)
                except Image.DecompressionBombError:
                    self._discard_staging()
                    dpg.set_value(
                        "status_text",
                        "Image is too large to open. Downscale it and try again.",
                    )
                    return
                except Exception:
                    orig_pil = self._orig_pil
            ph, pw = processed.shape[:2]

            vec_pil = None
            export_w = self._svg_export_width(pw, _INLINE_RASTER_MAX)
            rendered, render_err = self._render_svg_to_png(
                svg_for_preview,
                self.vec_preview_temp,
                export_width=export_w,
                timeout=min(90, 20 + export_w // 80),
            )
            if rendered:
                raw = self._load_inkscape_png_as_gray(self.vec_preview_temp)
                if raw is not None:
                    vec_pil = self._binarize_preview_pil(raw)

            if vec_pil is None:
                note = render_err or (
                    "Right preview unavailable — install Inkscape to rasterize SVG."
                )
                dpg.set_value("status_text", f"{note} Showing the stencil bitmap on the right.")
                vec_pil = Image.fromarray(processed).convert("L")
            else:
                lt = self._length_threshold_from_detail(config["detail_level"])
                dpg.set_value(
                    "status_text",
                    f"Output {pw}×{ph} px ({config['trace_mode']}, length {lt:.1f}). "
                    f"Trace preview rasterized at {export_w}px wide — click to enlarge.",
                )

            proc_pil = Image.fromarray(processed).convert("L")
            if orig_pil is not None:
                dpg.set_value("orig_texture", self._pil_to_texture_data(orig_pil))
            dpg.set_value(
                "proc_texture",
                self._pil_to_texture_data(proc_pil, resample=Image.NEAREST),
            )
            dpg.set_value("vec_texture", self._pil_to_texture_data(vec_pil))

            if do_preprocess:
                os.replace(self.preproc_staging, self.preproc_temp)
                self.current_preprocessed = self.preproc_temp
                self._last_binary = processed
            if traced_new:
                os.replace(self.svg_staging, self.svg_temp)
                self.last_svg = self.svg_temp
            self._last_input_path = self.input_path
            if orig_pil is not None:
                self._orig_pil = orig_pil
            self._image_dims = (pw, ph)
            self._preview_cache["orig"] = orig_pil.copy() if orig_pil else None
            self._preview_cache["proc"] = proc_pil.copy()
            self._preview_cache["vec"] = vec_pil.copy()
            self._last_config = dict(config)
        except Exception as e:
            self._discard_staging()
            dpg.set_value(
                "status_text",
                f"Preview error: {e}. Previous preview left unchanged.",
            )
        finally:
            self._discard_staging()
            self._busy = False
            try:
                dpg.set_value("loading_indicator", "")
            except Exception:
                pass
            if sampled is not None:
                self._rearm_if_config_drifted(sampled)

    def load_image(self):
        dpg.show_item("load_dialog")

    def _extract_path(self, app_data, require_exists=True):
        """Absolute dialog paths only.

        Dear PyGui's `.*` filter rewrites file_path_name to `name.*`, and the
        selection key is the basename. isfile() on that basename opens a
        different file in the process cwd. Ignore both.
        """
        if not isinstance(app_data, dict):
            return None
        current_path = app_data.get("current_path") or ""
        candidates = []

        selections = app_data.get("selections") or {}
        if isinstance(selections, dict):
            for value in selections.values():
                if isinstance(value, str) and os.path.isabs(value):
                    candidates.append(value)

        file_path_name = app_data.get("file_path_name")
        if (
            isinstance(file_path_name, str)
            and os.path.isabs(file_path_name)
            and not file_path_name.endswith(".*")
        ):
            candidates.append(file_path_name)

        file_name = app_data.get("file_name")
        if (
            isinstance(file_name, str)
            and file_name
            and os.path.isabs(current_path)
            and not file_name.endswith(".*")
            and os.sep not in file_name
            and "/" not in file_name
            and "\\" not in file_name
        ):
            candidates.append(os.path.join(current_path, file_name))

        if require_exists:
            for cand in candidates:
                if os.path.isfile(cand):
                    return cand
            return None
        for cand in candidates:
            if cand:
                return cand
        return None

    def _on_load_callback(self, sender, app_data, user_data):
        dpg.hide_item("load_dialog")
        path = self._extract_path(app_data, require_exists=True)
        if not path:
            dpg.set_value("status_text", "Could not open that file. Choose the image again.")
            return
        try:
            orig, _gray = _load_raster(path)
        except Image.DecompressionBombError:
            dpg.set_value(
                "status_text",
                "Image is too large to open. Downscale it and try again.",
            )
            return
        except Exception as exc:
            dpg.set_value("status_text", f"Could not open image: {exc}")
            return

        # Keep the previous preview until the new trace commits. A failure
        # rolls the path back so Save cannot export the previous bitmap
        # under the new photo's name.
        prev_path = self.input_path
        prev_orig = self._orig_pil
        self.input_path = path
        self._orig_pil = orig
        width, height = orig.size
        dpg.set_value(
            "status_text",
            f"Loaded: {Path(path).name} ({width}×{height} px) — updating previews...",
        )
        self.update_preview()
        if self._dirty:
            self._flush_preview_update()
        if self._last_input_path != path or self._last_config is None:
            self.input_path = prev_path
            self._orig_pil = prev_orig

    def save_vector(self):
        if not self.input_path and (
            not self.current_preprocessed or not os.path.isfile(self.current_preprocessed)
        ):
            dpg.set_value("status_text", "Load a photo first")
            return
        dpg.show_item("save_dialog")

    def _on_save_callback(self, sender, app_data, user_data):
        dpg.hide_item("save_dialog")
        save_path = self._extract_path(app_data, require_exists=False)
        if not save_path:
            dpg.set_value("status_text", "Save cancelled — no file name.")
            return
        if self._dirty and self.input_path:
            self._flush_preview_update()
        if not self._preview_matches_sliders():
            dpg.set_value(
                "status_text",
                "Preview does not match these settings, so nothing was saved.",
            )
            return
        suffix = Path(save_path).suffix.lower()
        if suffix not in self.SAVE_EXTENSIONS:
            save_path = str(Path(save_path).with_suffix(".svg"))
        self._do_save(save_path)

    def _do_save(self, save_path):
        config = dict(self._last_config)
        out_path = Path(save_path)
        fmt = out_path.suffix.lower()
        svg_dest = str(out_path) if fmt == ".svg" else self.svg_temp

        try:
            err = self._trace_to_svg(self.current_preprocessed, svg_dest, config)
            if err is not None:
                dpg.set_value("status_text", f"Vectorize failed: {err}")
                return
        except Exception as exc:
            dpg.set_value("status_text", f"Vectorize failed: {exc}")
            return

        self.last_svg = svg_dest

        if fmt == ".svg":
            dpg.set_value("status_text", f"Saved SVG: {save_path}")
            return

        export_type = fmt.lstrip(".")
        ok, err = self._inkscape_export(svg_dest, save_path, export_type=export_type)
        if ok:
            dpg.set_value("status_text", f"Saved {export_type.upper()}: {save_path}")
        else:
            dpg.set_value(
                "status_text",
                f"Vector traced to SVG but {export_type.upper()} export failed: {err}",
            )

    def open_svg(self):
        target = self.last_svg
        if not target or not os.path.isfile(target):
            dpg.set_value("status_text", "No SVG available yet")
            return
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", target], check=False)
            elif os.name == "nt":
                os.startfile(target)
            else:
                subprocess.run(["xdg-open", target], check=False)
            dpg.set_value("status_text", f"Opened {Path(target).name}")
        except Exception as exc:
            dpg.set_value("status_text", f"Could not open SVG: {exc}")

    def _on_global_mouse_release(self, sender=None, app_data=None, user_data=None):
        """Force an immediate preview update when the user releases the mouse after dragging a slider."""
        if getattr(self, "_dirty", False) and self.input_path:
            self._flush_preview_update()

    def _schedule_debounce_check(self):
        """Schedule the next idle debounce poll.
        We use DPG's frame callbacks (no threads) to check every ~100-150ms whether the
        user has stopped moving a slider long enough to trigger processing.
        """
        try:
            current_frame = dpg.get_frame_count()
            dpg.set_frame_callback(current_frame + 8, self._debounce_check)
        except Exception:
            # Can happen very early or during shutdown; just ignore.
            pass

    def _debounce_check(self, sender=None, app_data=None, user_data=None):
        """Polled periodically. If the user has not touched any slider for >= _debounce_delay
        and we have a pending update, kick off the real work.
        This (plus the mouse release handler) implements "don't process until I stop moving".
        """
        if getattr(self, "_dirty", False) and self.input_path:
            if time.time() - self._last_slider_change >= self._debounce_delay:
                self._dirty = False
                self._do_update_preview()
        # Keep the poller alive
        self._schedule_debounce_check()

    def _cleanup_temps(self):
        if getattr(self, "_temps_cleaned", False):
            return
        self._temps_cleaned = True
        work = getattr(self, "temp_dir", None)
        if work is not None:
            shutil.rmtree(work, ignore_errors=True)

    def _on_exit(self):
        self._cleanup_temps()

    def run(self):
        try:
            dpg.start_dearpygui()
        finally:
            self._cleanup_temps()
            try:
                dpg.destroy_context()
            except Exception:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phone photo → screen print vector (Dear PyGui)")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print startup diagnostics to the console",
    )
    args = parser.parse_args()
    app = StencilApp(verbose=args.verbose)
    app.run()
