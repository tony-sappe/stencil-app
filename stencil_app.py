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
- Temps isolated to /tmp, cleaned on exit
- No Tkinter

This version uses documented DPG dynamic textures (np arrays, not python lists) + slider debounce (process only after you stop moving or release) + mouse-release force for reliable live previews without unnecessary work while dragging.
"""

import argparse
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import dearpygui.dearpygui as dpg
import vtracer
from PIL import Image


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
_PREVIEW_BINARIZE_THRESHOLD = 220  # higher = thinner lines (less AA fattening)
# child_window(border=True) shrinks the interior; pad outer size so image buttons fit without scrollbars
_PREVIEW_CHILD_PAD_X = 30
_PREVIEW_CHILD_PAD_Y = 16


class StencilApp:
    TRACE_MODES = ("Smooth (spline)", "Polygon", "Pixel-accurate")
    SAVE_EXTENSIONS = (".svg", ".eps", ".pdf")

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.input_path = None
        self.current_preprocessed = None
        self.last_svg = None
        self._orig_pil = None  # cached for faster repeated updates

        # All temps in system tmp to avoid polluting cwd or source dirs
        self.temp_dir = Path(tempfile.gettempdir())
        self.preproc_temp = str(self.temp_dir / "stencil_creator_preproc.png")
        self.svg_temp = str(self.temp_dir / "stencil_creator_vector.svg")
        self.vec_preview_temp = str(self.temp_dir / "stencil_creator_vecprev.png")
        self.vec_preview_modal_temp = str(self.temp_dir / "stencil_creator_vecprev_modal.png")

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
                            label="Remove blobs smaller than (px)",
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
        """Re-threshold rasterized SVG; high threshold avoids bloating anti-aliased edges."""
        arr = np.array(pil_img.convert("L"), dtype=np.uint8)
        return Image.fromarray(
            np.where(arr < _PREVIEW_BINARIZE_THRESHOLD, 0, 255).astype(np.uint8)
        )

    def _load_inkscape_png_as_gray(self, png_path: str) -> Optional[Image.Image]:
        try:
            return Image.open(png_path).convert("L")
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
            if self.last_svg and os.path.exists(self.svg_temp) and pw > 0:
                export_w = self._svg_export_width(pw, _MODAL_RASTER_MAX)
                if self._render_svg_to_png(
                    self.svg_temp,
                    self.vec_preview_modal_temp,
                    export_width=export_w,
                    timeout=90,
                ):
                    pil = self._load_inkscape_png_as_gray(self.vec_preview_modal_temp)
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
        """PIL-first load (broad formats incl. HEIC if pillow-heif installed) then cv2 binary prep."""
        gray = None
        try:
            pil = Image.open(img_path)
            if pil.mode in ("RGBA", "LA", "P"):
                pil = pil.convert("L")
            elif pil.mode != "L":
                pil = pil.convert("L")
            gray = np.array(pil, dtype=np.uint8)
        except Exception:
            img = cv2.imread(str(img_path))
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if gray is None:
            return None

        denoised = cv2.fastNlMeansDenoising(gray, h=config["denoise_strength"])
        blurred = cv2.GaussianBlur(denoised, (5, 5), config["blur_radius"])

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
        """Expand dark ink (0) so higher line width = thicker stencil lines on screen."""
        if width_px <= 0 or binary is None:
            return binary
        # Kernel diameter from requested width (odd, at least 3px for visible effect).
        ksize = max(3, int(round(width_px)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        ink = (binary == 0).astype(np.uint8) * 255
        ink = cv2.dilate(ink, kernel, iterations=1)
        return np.where(ink > 0, 0, 255).astype(np.uint8)

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

    def _inkscape_export(self, svg_path, out_path, export_type=None, timeout=60):
        """Convert SVG to another vector format via Inkscape. Returns (ok, error_message)."""
        out_path = str(out_path)
        if export_type is None:
            export_type = Path(out_path).suffix.lstrip(".").lower()
        try:
            subprocess.run(
                [
                    "inkscape",
                    svg_path,
                    f"--export-type={export_type}",
                    "--export-filename",
                    out_path,
                ],
                check=True,
                capture_output=True,
                text=True,
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
        """Best-effort SVG→PNG. export_width should match bitmap width (capped) for faithful preview.
        Prefers inkscape (consistent with EPS export). Falls back to macOS qlmanage.
        Returns True if png_path now exists with useful content.
        """
        export_width = max(64, int(export_width))
        # 1. inkscape (preferred)
        try:
            subprocess.run(
                [
                    "inkscape",
                    svg_path,
                    "--export-type=png",
                    "--export-filename", png_path,
                    f"--export-width={export_width}",
                ],
                check=True,
                capture_output=True,
                timeout=timeout,
            )
            if os.path.exists(png_path):
                return True
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass

        # 2. qlmanage (macOS built-in)
        if os.name == "posix":
            try:
                out_dir = str(Path(png_path).parent)
                expected = str(Path(out_dir) / (Path(svg_path).name + ".png"))
                thumb = min(export_width, 1024)
                subprocess.run(
                    ["qlmanage", "-t", "-s", str(thumb), "-o", out_dir, svg_path],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                for _ in range(12):
                    if os.path.exists(expected):
                        os.replace(expected, png_path)
                        return True
                    time.sleep(0.04)
                # last resort: newest plausible thumb in dir
                for f in sorted(Path(out_dir).glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True):
                    name = f.name.lower()
                    if any(k in name for k in ("stencil", "vector", "thumb", "preview")):
                        os.replace(str(f), png_path)
                        return True
            except Exception:
                pass
        return False

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
        Triggered only after debounce (user stopped moving slider) or mouse release.
        Skips preprocess when only vector settings change; re-traces from the same binary PNG.
        """
        dpg.set_value("loading_indicator", "⏳ Processing...")
        dpg.set_value("status_text", "Processing…")
        try:
            config = self.get_config()
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
                or any(
                    config[k] != self._last_config.get(k, None)
                    for k in preprocess_keys
                )
            )
            vector_config_changed = (
                self._last_config is None
                or any(
                    config[k] != self._last_config.get(k, None)
                    for k in vector_keys
                )
            )
            do_preprocess = (
                self._last_binary is None
                or input_changed
                or preprocess_config_changed
            )
            do_vector = do_preprocess or vector_config_changed

            if do_preprocess:
                processed = self.preprocess(self.input_path, config)
                if processed is None:
                    dpg.set_value("status_text", "Failed to load/process image")
                    dpg.set_value("loading_indicator", "")
                    return
                cv2.imwrite(self.preproc_temp, processed)
                self.current_preprocessed = self.preproc_temp
                self._last_binary = processed
            else:
                # Reuse previous binary (png is still valid)
                processed = self._last_binary
                if processed is None or not os.path.exists(self.preproc_temp):
                    # fallback
                    processed = self.preprocess(self.input_path, config)
                    if processed is None:
                        dpg.set_value("status_text", "Failed to load/process image")
                        dpg.set_value("loading_indicator", "")
                        return
                    cv2.imwrite(self.preproc_temp, processed)
                    self.current_preprocessed = self.preproc_temp
                    self._last_binary = processed

            self._last_input_path = self.input_path

            if not do_vector and self.last_svg and os.path.exists(self.svg_temp):
                pass  # reuse existing SVG
            else:
                try:
                    vtracer.convert_image_to_svg_py(
                        self.preproc_temp,
                        self.svg_temp,
                        **self._vtracer_kwargs(config),
                    )
                except Exception as ve:
                    dpg.set_value(
                        "status_text",
                        f"Vectorize failed (overflow or bad params): {ve}. "
                        "Try ↑ Min Area (speckle), ↑ Denoise, or adjust Threshold. Preview not updated.",
                    )
                    dpg.set_value("loading_indicator", "")
                    return
                self.last_svg = self.svg_temp

            if self._orig_pil is not None:
                orig_pil = self._orig_pil
            else:
                orig_pil = Image.open(self.input_path)
            ph, pw = processed.shape[:2]

            vec_pil = None
            export_w = self._svg_export_width(pw, _INLINE_RASTER_MAX)
            if self.last_svg and self._render_svg_to_png(
                self.svg_temp,
                self.vec_preview_temp,
                export_width=export_w,
                timeout=min(90, 20 + export_w // 80),
            ):
                raw = self._load_inkscape_png_as_gray(self.vec_preview_temp)
                if raw is not None:
                    vec_pil = self._binarize_preview_pil(raw)

            if vec_pil is None:
                dpg.set_value(
                    "status_text",
                    "Right preview unavailable — install Inkscape to rasterize SVG "
                    "(showing center bitmap as fallback)"
                )
                vec_pil = Image.fromarray(processed).convert("L")
            else:
                lt = self._length_threshold_from_detail(config["detail_level"])
                dpg.set_value(
                    "status_text",
                    f"Output {pw}×{ph} px ({config['trace_mode']}, length {lt:.1f}). "
                    f"Trace preview rasterized at {export_w}px wide — click to enlarge.",
                )

            proc_pil = Image.fromarray(processed).convert("L")
            self._image_dims = (pw, ph)
            self._preview_cache["orig"] = orig_pil.copy() if orig_pil else None
            self._preview_cache["proc"] = proc_pil.copy()
            self._preview_cache["vec"] = vec_pil.copy() if vec_pil is not None else None

            orig_data = self._pil_to_texture_data(orig_pil)
            proc_data = self._pil_to_texture_data(proc_pil, resample=Image.NEAREST)
            vec_data = self._pil_to_texture_data(vec_pil)

            dpg.set_value("orig_texture", orig_data)
            dpg.set_value("proc_texture", proc_data)
            dpg.set_value("vec_texture", vec_data)

            # Clear loading icon now that vector is ready
            dpg.set_value("loading_indicator", "")

            # Remember for next time (to skip preprocess on pure vector changes)
            self._last_config = dict(config)

        except Exception as e:
            dpg.set_value("status_text", f"Preview error: {e}. Try different settings or reload image.")
            dpg.set_value("loading_indicator", "")

    def load_image(self):
        dpg.show_item("load_dialog")

    def _extract_path(self, app_data, require_exists=True):
        """Robust extraction that survived the '.*' filter mangling bug in earlier DPG file_dialog usage."""
        if not isinstance(app_data, dict):
            return None
        candidates = []

        p = app_data.get("file_path_name")
        if isinstance(p, str) and p:
            candidates.append(p)

        selections = app_data.get("selections") or {}
        current_path = app_data.get("current_path") or ""
        for k, v in selections.items():
            for cand in (k, v):
                if isinstance(cand, str) and cand:
                    candidates.append(cand)
                    if current_path:
                        joined = os.path.join(current_path, cand.lstrip(os.sep + "/\\"))
                        candidates.append(joined)

        fname = app_data.get("file_name")
        if isinstance(fname, str) and fname and current_path:
            joined = os.path.join(current_path, fname.lstrip(os.sep + "/\\"))
            candidates.append(joined)

        for cand in candidates:
            if isinstance(cand, str) and cand:
                if not require_exists:
                    return cand
                if os.path.isfile(cand):
                    return cand

        for cand in candidates:
            if isinstance(cand, str) and cand:
                return cand
        return None

    def _on_load_callback(self, sender, app_data, user_data):
        dpg.hide_item("load_dialog")
        path = self._extract_path(app_data, require_exists=True)
        if path:
            self.input_path = path
            self._last_config = None
            self._last_binary = None
            self._last_slider_change = 0.0
            self._last_input_path = None
            try:
                self._orig_pil = Image.open(path).convert("RGBA")
                w, h = self._orig_pil.size
                dpg.set_value("orig_texture", self._pil_to_texture_data(self._orig_pil))
            except Exception:
                self._orig_pil = None

            dim = ""
            if self._orig_pil is not None:
                dim = f" ({self._orig_pil.size[0]}×{self._orig_pil.size[1]} px)"
            dpg.set_value("status_text", f"Loaded: {Path(path).name}{dim} — updating previews...")
            self.update_preview()  # marks _dirty + _last_slider_change

            # For the *initial* preview after loading an image we want it right away,
            # not waiting for the debounce timer.
            if self._dirty:
                self._flush_preview_update()

    def save_vector(self):
        if not self.current_preprocessed or not os.path.exists(self.current_preprocessed):
            dpg.set_value("status_text", "Load a photo first")
            return
        dpg.show_item("save_dialog")

    def _on_save_callback(self, sender, app_data, user_data):
        dpg.hide_item("save_dialog")
        save_path = self._extract_path(app_data, require_exists=False)
        if not save_path:
            return
        suffix = Path(save_path).suffix.lower()
        if suffix not in self.SAVE_EXTENSIONS:
            save_path = str(Path(save_path).with_suffix(".svg"))
        self._do_save(save_path)

    def _do_save(self, save_path):
        config = self.get_config()
        out_path = Path(save_path)
        fmt = out_path.suffix.lower()
        svg_dest = str(out_path) if fmt == ".svg" else self.svg_temp

        try:
            vtracer.convert_image_to_svg_py(
                self.current_preprocessed,
                svg_dest,
                **self._vtracer_kwargs(config),
            )
        except Exception as e:
            dpg.set_value("status_text", f"Vectorize failed: {e}")
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
        if (not target or not os.path.exists(target)) and os.path.exists(self.svg_temp):
            target = self.svg_temp
        if not target or not os.path.exists(target):
            dpg.set_value("status_text", "No SVG available yet")
            return
        try:
            if os.name == "posix":
                subprocess.run(["open", target])
            else:
                os.startfile(target)
            dpg.set_value("status_text", f"Opened {Path(target).name}")
        except Exception as e:
            dpg.set_value("status_text", f"Could not open SVG: {e}")

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
        for p in (
            getattr(self, "preproc_temp", None),
            getattr(self, "svg_temp", None),
            getattr(self, "vec_preview_temp", None),
            getattr(self, "vec_preview_modal_temp", None),
        ):
            try:
                if p and os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass

    def _on_exit(self):
        self._cleanup_temps()

    def run(self):
        dpg.start_dearpygui()
        dpg.destroy_context()


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
