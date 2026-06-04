#!/usr/bin/env python3
"""
stencil_app.py - Phone Photo → Screen Print Vector (Dear PyGui)
Brand new implementation per specification.md + prior conversation.

Core flow:
- Load image (robust DPG file dialog, PIL + cv2 fallback for HEIC etc.)
- Preprocess (denoise, blur, adaptive thresh, optional invert, open morph) -> binary for tracing
- vtracer.convert_image_to_svg_py (binary, live params for detail/min_area/corner)
- 3 live previews via DPG dynamic textures + set_value (orig color, preproc binary, vector raster); uses np arrays + slider debounce to avoid unnecessary work while dragging
- Realtime slider/checkbox callbacks update everything
- Export SVG (always) + EPS (if inkscape in PATH)
- Temps isolated to /tmp, cleaned on exit
- No Tkinter

This version uses documented DPG dynamic textures (np arrays, not python lists) + slider debounce (process only after you stop moving or release) + mouse-release force for reliable live previews without unnecessary work while dragging.
"""

import os
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import dearpygui.dearpygui as dpg
import vtracer
from PIL import Image


class StencilApp:
    def __init__(self):
        self.input_path = None
        self.current_preprocessed = None
        self.last_svg = None
        self._orig_pil = None  # cached for faster repeated updates

        # All temps in system tmp to avoid polluting cwd or source dirs
        self.temp_dir = Path(tempfile.gettempdir())
        self.preproc_temp = str(self.temp_dir / "stencil_creator_preproc.png")
        self.svg_temp = str(self.temp_dir / "stencil_creator_vector.svg")
        self.vec_preview_temp = str(self.temp_dir / "stencil_creator_vecprev.png")
        self.crop_temp = str(self.temp_dir / "stencil_creator_crop.png")

        self.texture_size = 256   # internal texture res (displayed at preview_size; smaller = less mem/CPU per update)
        self.preview_size = 470  # larger to span full window width evenly as squares (with borders)

        # Throttling to prevent memory blowup from rapid slider drags (vtracer + large texture data churn)
        self._last_update_ts = 0.0
        self._dirty = False

        # Debounce for sliders: don't start heavy processing (preprocess + vtracer) until
        # the user has stopped moving the slider for this long (or releases the mouse).
        self._last_slider_change = 0.0
        self._debounce_delay = 0.30  # seconds

        self._last_config = None
        self._last_binary = None
        self._last_crop = None

        # For square crop selection on original
        self.full_pil = None
        self.crop_rect = None  # (left, top, side) in full_pil coords or None
        self.process_input = None  # the (possibly cropped) path for preprocess
        self._orig_drawlist_tag = "orig_drawlist"
        self._sel_rect_tag = "sel_rect"
        self._orig_img_rect = (0.0, 0.0, 0.0, 0.0)  # (ox, oy, dw, dh) in drawlist space
        self._orig_scale = 1.0

        # drag state
        self._is_dragging = False
        self._resize_corner = None  # 0=tl,1=tr,2=br,3=bl or -1=move
        self._drag_start_rect = None
        self._drag_start_mouse = None
        self._last_drag_end_ts = 0.0

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

    def _pil_to_texture_data(self, pil_img, size=None):
        """Center thumbnail on fixed dark square canvas. Returns np.float32 (h,w,4) — DPG accepts directly (huge mem win vs list of Py floats)."""
        if size is None:
            size = self.texture_size
        canvas = Image.new("RGBA", (size, size), (25, 25, 25, 255))
        pil_img = pil_img.convert("RGBA")
        pil_img.thumbnail((size, size), Image.LANCZOS)
        x = (size - pil_img.width) // 2
        y = (size - pil_img.height) // 2
        canvas.paste(pil_img, (x, y))
        arr = np.array(canvas, dtype=np.float32) / 255.0
        return arr  # shaped ndarray, not .tolist() — compact + DPG friendly

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
            print("[INFO] Dynamic texture registry + 3 checker placeholders created.")

        dpg.create_viewport(
            title="Phone Photo → Screen Print Vector",
            width=1450,
            height=880,
            resizable=True,
        )
        dpg.setup_dearpygui()
        print("[INFO] Viewport + DPG setup done.")

        with dpg.window(tag="main_win"):
            with dpg.group(horizontal=True):
                dpg.add_button(label="Load Photo", callback=self.load_image, width=110)
                dpg.add_button(label="Save Vector", callback=self.save_vector, width=110)
                dpg.add_button(label="Open SVG", callback=self.open_svg, width=110)
                dpg.add_button(label="Reset Crop", callback=self._reset_crop, width=95)

            dpg.add_separator()

            with dpg.collapsing_header(
                label="Preprocessing settings (affect binary image for tracing — Min Area now also pre-filters specks)",
                default_open=True,
            ):
                dpg.add_slider_int(
                    label="Denoise Strength",
                    tag="slider_denoise",
                    default_value=15,
                    min_value=0,
                    max_value=40,
                    callback=self.update_preview,
                    width=380,
                )
                dpg.add_slider_float(
                    label="Blur Radius",
                    tag="slider_blur",
                    default_value=1.5,
                    min_value=0.0,
                    max_value=5.0,
                    callback=self.update_preview,
                    width=380,
                )
                dpg.add_slider_int(
                    label="Threshold Offset",
                    tag="slider_threshold",
                    default_value=5,
                    min_value=-20,
                    max_value=20,
                    callback=self.update_preview,
                    width=380,
                )
                dpg.add_slider_int(
                    label="Adaptive Block Size",
                    tag="slider_block_size",
                    default_value=11,
                    min_value=3,
                    max_value=31,
                    callback=self.update_preview,
                    width=380,
                )
                dpg.add_slider_int(
                    label="Min Area (speckle filter)",
                    tag="slider_min_area",
                    default_value=25,
                    min_value=10,
                    max_value=100,
                    callback=self.update_preview,
                    width=380,
                )
                dpg.add_slider_int(
                    label="Line Thickening",
                    tag="slider_line_thickness",
                    default_value=0,
                    min_value=0,
                    max_value=3,
                    callback=self.update_preview,
                    width=380,
                )
                dpg.add_checkbox(
                    label="Invert Image",
                    tag="check_invert",
                    default_value=False,
                    callback=self.update_preview,
                )

            with dpg.collapsing_header(
                label="Vector settings (only affect final paths — reuse preprocessed binary for faster updates)",
                default_open=True,
            ):
                dpg.add_slider_float(
                    label="Detail Level",
                    tag="slider_detail",
                    default_value=0.95,
                    min_value=0.35,
                    max_value=1.0,
                    callback=self.update_preview,
                    width=380,
                )
                dpg.add_slider_int(
                    label="Corner Threshold",
                    tag="slider_corner",
                    default_value=30,
                    min_value=0,
                    max_value=180,
                    callback=self.update_preview,
                    width=380,
                )
                dpg.add_slider_int(
                    label="Splice Threshold",
                    tag="slider_splice_threshold",
                    default_value=45,
                    min_value=0,
                    max_value=180,
                    callback=self.update_preview,
                    width=380,
                )

            dpg.add_separator()

            dpg.add_text(
                "Preview boxes below show checkerboard at launch (proves texture system). "
                "Load an image — previews update after you stop moving a slider (debounced) or release the mouse. ⏳ shows while processing. "
                "If input isn't square, drag the yellow square overlay on Original to pick the exact region (final output is always square).",
                color=(200, 200, 140),
            )

            # Labels above bordered preview boxes (images fill the boxes exactly)
            with dpg.group(horizontal=True):
                with dpg.group():
                    dpg.add_text("Original (drag yellow square to crop)")
                    with dpg.child_window(
                        tag="col_orig", border=True,
                        width=self.preview_size + 2, height=self.preview_size + 2
                    ):
                        dpg.add_drawlist(tag=self._orig_drawlist_tag, width=self.preview_size, height=self.preview_size)

                with dpg.group():
                    dpg.add_text("Preprocessed (binary input to vectorizer)")
                    with dpg.child_window(
                        tag="col_proc", border=True,
                        width=self.preview_size + 2, height=self.preview_size + 2
                    ):
                        pass

                with dpg.group():
                    dpg.add_text("Vector Result (live)", color=(140, 220, 140))
                    with dpg.child_window(
                        tag="col_vec", border=True,
                        width=self.preview_size + 2, height=self.preview_size + 2
                    ):
                        pass

            dpg.add_spacer(height=4)
            dpg.add_text(tag="crop_info_text", default_value="")
            dpg.add_text(tag="status_text", default_value="Load a photo to begin. Changes to sliders/checkbox update the vector preview live.")
            dpg.add_text(tag="loading_indicator", default_value="", color=(255, 180, 100))

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
            dpg.add_file_extension(".*")

        # Handler registry for global mouse events.
        # We put both the slider debounce release *and* the crop drag handlers here.
        # (mouse_*_down/drag/release are only compatible with normal handler_registry,
        # not item_handler_registry. The crop callbacks use manual rect/hit tests
        # + the drawlist's screen bounds to decide if a click applies to the overlay.)
        with dpg.handler_registry(tag="global_mouse_handlers"):
            dpg.add_mouse_release_handler(callback=self._on_global_mouse_release)
            dpg.add_mouse_down_handler(button=0, callback=self._on_mouse_down)
            dpg.add_mouse_drag_handler(button=0, callback=self._on_mouse_drag)
            dpg.add_mouse_release_handler(button=0, callback=self._on_mouse_release_crop)
            # Fallback move handler: helps with trackpads / devices where "mouse down"
            # events report button in odd ways (e.g. [0, delta] or repeated). While left
            # button is held (per is_mouse_button_down) and we are not yet dragging,
            # we can latch onto the crop rect if the cursor is over it.
            dpg.add_mouse_move_handler(callback=self._on_mouse_move)
        print("[CROP] mouse handlers (down/drag/release + move fallback) registered (global handler_registry)")

        dpg.set_primary_window("main_win", True)
        dpg.set_exit_callback(self._on_exit)
        dpg.show_viewport()
        print("[INFO] Viewport shown.")

        # Schedule adding the (texture-bound) image widgets once DPG is fully up.
        # Using frame callback avoids early binding issues seen in prior attempts.
        def _initial_add_preview_images(sender, app_data, user_data):
            print("[INFO] Adding initial preview images (bound to dynamic checker textures)...")
            for col_tag, img_tag, tex_tag in [
                ("col_orig", "img_orig", "orig_texture"),
                ("col_proc", "img_proc", "proc_texture"),
                ("col_vec", "img_vec", "vec_texture"),
            ]:
                if dpg.does_item_exist(img_tag):
                    dpg.delete_item(img_tag)
                if col_tag == "col_orig":
                    self._setup_orig_drawlist()
                    continue
                dpg.add_image(
                    tag=img_tag,
                    texture_tag=tex_tag,
                    width=self.preview_size,
                    height=self.preview_size,
                    parent=col_tag,
                )
            print("[INFO] Initial preview images added. Checkerboards should be visible in the three bordered boxes.")

        dpg.set_frame_callback(2, _initial_add_preview_images)
        print("[INFO] Initial preview image add scheduled for frame 2.")

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
            "line_thickness": dpg.get_value("slider_line_thickness"),
            "corner_threshold": dpg.get_value("slider_corner"),
            "splice_threshold": dpg.get_value("slider_splice_threshold"),
            "invert": dpg.get_value("check_invert"),
        }

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

        # Optional line thickening (dilation) after cleaning. Makes stencil lines more durable for screen printing.
        thickness = int(config.get("line_thickness", 0))
        if thickness > 0:
            kernel = np.ones((3, 3), np.uint8)
            cleaned = cv2.dilate(cleaned, kernel, iterations=thickness)

        return cleaned

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

    def _apply_crop(self):
        """Apply current crop_rect to create the process_input (cropped square png)."""
        if not self.full_pil or not self.crop_rect:
            self.process_input = self.input_path
            return
        x, y, s = self.crop_rect
        try:
            cropped = self.full_pil.crop((x, y, x + s, y + s))
            cropped.save(self.crop_temp)
            self.process_input = self.crop_temp
        except Exception:
            self.process_input = self.input_path

    def _compute_display_rect(self):
        if not self.full_pil:
            ps = float(self.preview_size)
            self._orig_img_rect = (0.0, 0.0, ps, ps)
            self._orig_scale = 1.0
            return
        ps = float(self.preview_size)
        pw, ph = self.full_pil.size
        scale = min(ps / pw, ps / ph) if pw > 0 and ph > 0 else 1.0
        dw = pw * scale
        dh = ph * scale
        ox = (ps - dw) / 2.0
        oy = (ps - dh) / 2.0
        self._orig_img_rect = (ox, oy, dw, dh)
        self._orig_scale = scale

    def _point_in_img_rect(self, lx, ly):
        ox, oy, ow, oh = self._orig_img_rect
        return ox <= lx <= ox + ow and oy <= ly <= oy + oh

    def _display_to_image(self, lx, ly):
        ox, oy, ow, oh = self._orig_img_rect
        if self._orig_scale <= 0:
            return 0, 0
        ix = (lx - ox) / self._orig_scale
        iy = (ly - oy) / self._orig_scale
        return ix, iy

    def _update_selection_visual(self):
        if not dpg.does_item_exist(self._orig_drawlist_tag):
            return
        if not self.crop_rect or not self.full_pil:
            if dpg.does_item_exist(self._sel_rect_tag):
                try:
                    dpg.delete_item(self._sel_rect_tag)
                except:
                    pass
            dpg.set_value("crop_info_text", "")
            return
        if not hasattr(self, "_orig_scale") or self._orig_scale <= 0:
            self._compute_display_rect()
        ox, oy, dw, dh = self._orig_img_rect
        scale = self._orig_scale
        if scale <= 0:
            return
        x, y, s = self.crop_rect
        d_x = ox + x * scale
        d_y = oy + y * scale
        d_s = s * scale
        pmin = [d_x, d_y]
        pmax = [d_x + d_s, d_y + d_s]
        if dpg.does_item_exist(self._sel_rect_tag):
            dpg.configure_item(self._sel_rect_tag, pmin=pmin, pmax=pmax)
        else:
            dpg.draw_rectangle(
                pmin=pmin,
                pmax=pmax,
                color=(255, 255, 0, 255),
                thickness=2.0,
                fill=(255, 255, 0, 50),
                parent=self._orig_drawlist_tag,
                tag=self._sel_rect_tag
            )
        # live crop size info (updates during drag too)
        pw, ph = self.full_pil.size
        dpg.set_value("crop_info_text", f"Crop region: {s}×{s} px  (from original {pw}×{ph})")

    def _setup_orig_drawlist(self):
        if not dpg.does_item_exist(self._orig_drawlist_tag):
            return
        # clear previous
        for ch in dpg.get_item_children(self._orig_drawlist_tag, 1) or []:
            try:
                dpg.delete_item(ch)
            except:
                pass
        ps = self.preview_size
        # bg: the full image view (via the orig_texture which is updated from full_pil)
        dpg.draw_image(
            texture_tag="orig_texture",
            pmin=[0, 0],
            pmax=[ps, ps],
            parent=self._orig_drawlist_tag,
            tag="orig_bg_image_draw"
        )
        self._compute_display_rect()
        if self.crop_rect and self.full_pil:
            self._update_selection_visual()

    def _try_begin_crop_drag(self, mpos):
        """Shared logic to test if a mouse position (global) is over the photo content
        and over the current crop rect (or its handles). If so, set drag state and return True.
        Used by both the down handler and the move fallback (for trackpad compatibility).
        """
        if not dpg.does_item_exist(self._orig_drawlist_tag) or not self.full_pil or not self.crop_rect:
            return False
        rmin = dpg.get_item_rect_min(self._orig_drawlist_tag)
        try:
            rmax = dpg.get_item_rect_max(self._orig_drawlist_tag)
        except Exception:
            ps = self.preview_size
            rmax = [rmin[0] + ps, rmin[1] + ps]
        margin = 3
        if not (rmin[0] - margin <= mpos[0] <= rmax[0] + margin and
                rmin[1] - margin <= mpos[1] <= rmax[1] + margin):
            return False
        lx = mpos[0] - rmin[0]
        ly = mpos[1] - rmin[1]
        if not self._point_in_img_rect(lx, ly):
            return False
        ix, iy = self._display_to_image(lx, ly)
        cx, cy, cs = self.crop_rect
        handle = 10.0 / self._orig_scale if self._orig_scale > 0 else 10.0
        corners = [(0, 0), (cs, 0), (cs, cs), (0, cs)]
        hit_corner = None
        for ci in range(4):
            ccx = cx + corners[ci][0]
            ccy = cy + corners[ci][1]
            if abs(ix - ccx) <= handle and abs(iy - ccy) <= handle:
                hit_corner = ci
                break
        if hit_corner is not None or (cx <= ix <= cx + cs and cy <= iy <= cy + cs):
            self._resize_corner = hit_corner if hit_corner is not None else -1
            self._is_dragging = True
            self._drag_start_rect = (cx, cy, cs)
            self._drag_start_mouse = (ix, iy)
            dpg.set_value("status_text", "✥ Dragging crop (move/resize yellow square) — release mouse to update previews")
            if hit_corner is not None:
                print(f"[CROP] drag START corner={hit_corner} at img=({ix:.1f},{iy:.1f}) handle_px={handle:.1f}")
            else:
                print(f"[CROP] drag START move inside at img=({ix:.1f},{iy:.1f}) crop=({cx},{cy},{cs})")
            return True
        return False

    def _end_crop_drag(self):
        """Clear all crop drag state. Call on release or when button is no longer down."""
        self._is_dragging = False
        self._resize_corner = None
        self._drag_start_rect = None
        self._drag_start_mouse = None
        self._last_drag_end_ts = time.time()
        # Clear any temporary drag status message if present
        try:
            current = dpg.get_value("status_text") or ""
            if "Dragging crop" in current or "✥" in current:
                dpg.set_value("status_text", "")
        except Exception:
            pass

    def _on_mouse_down(self, sender, app_data, user_data):
        # Robust button extraction.
        # On some systems (esp. macOS trackpad), app_data for mouse_down_handler can be
        # an int or a list/tuple like [button, delta_or_pressure].
        if isinstance(app_data, (list, tuple)) and len(app_data) > 0:
            button = app_data[0]
        else:
            button = app_data

        if button != 0:  # left button only
            # Only log the ignore for non-left to avoid too much spam, but show raw for debug
            if not isinstance(app_data, (list, tuple)) or app_data[0] != 0:
                print(f"[CROP] MOUSE_DOWN ignored non-left: raw_app_data={app_data}")
            return

        if not dpg.does_item_exist(self._orig_drawlist_tag):
            print("[CROP]   -> drawlist item does not exist!")
            return
        if not self.full_pil or not self.crop_rect:
            print(f"[CROP]   -> no image loaded yet (full_pil={self.full_pil is not None}, crop_rect={self.crop_rect})")
            return
        if self._orig_scale <= 0:
            self._compute_display_rect()
        mpos = dpg.get_mouse_pos(local=False)
        print(f"[CROP] MOUSE_DOWN left: raw_app_data={app_data}, mouse_global={mpos}")
        # To avoid spamming on clicks elsewhere, only evaluate the crop hit test for
        # mouse positions that could plausibly be over the left Original preview column.
        if mpos[0] > 550:
            return
        if self._try_begin_crop_drag(mpos):
            return
        # If we got here, the click was over the left preview area but did not hit the
        # current yellow crop rect or its handles.
        # Re-compute local coords just for a helpful diagnostic message.
        rmin = dpg.get_item_rect_min(self._orig_drawlist_tag)
        lx = mpos[0] - rmin[0]
        ly = mpos[1] - rmin[1]
        if not self._point_in_img_rect(lx, ly):
            print(f"[CROP] down: over drawlist but outside photo content area (lx={lx:.1f}, ly={ly:.1f})")
        else:
            ix, iy = self._display_to_image(lx, ly)
            cx, cy, cs = self.crop_rect
            print(f"[CROP] down: over photo but outside current crop rect (ix={ix:.1f},iy={iy:.1f}) crop=({cx},{cy},{cs})")

    def _on_mouse_drag(self, sender, app_data, user_data):
        # Extra safety for devices (trackpads) where release events or button state
        # can be delayed or reported oddly: if we think we're dragging but the button
        # is no longer down, force end the drag immediately. Do this before logging.
        if self._is_dragging and not dpg.is_mouse_button_down(0):
            self._end_crop_drag()
            return
        if self._is_dragging:
            print(f"[CROP] DRAG handler (while crop active), app_data={app_data}")
        if not self._is_dragging or not self.crop_rect or not self.full_pil:
            return
        if self._drag_start_mouse is None or self._drag_start_rect is None:
            self._end_crop_drag()
            return
        if not dpg.does_item_exist(self._orig_drawlist_tag):
            self._end_crop_drag()
            return
        mpos = dpg.get_mouse_pos(local=False)
        rmin = dpg.get_item_rect_min(self._orig_drawlist_tag)
        lx = mpos[0] - rmin[0]
        ly = mpos[1] - rmin[1]
        ix, iy = self._display_to_image(lx, ly)
        pw, ph = self.full_pil.size
        ix = max(0.0, min(float(pw), ix))
        iy = max(0.0, min(float(ph), iy))
        dx = ix - self._drag_start_mouse[0]
        dy = iy - self._drag_start_mouse[1]
        ox, oy, os = self._drag_start_rect
        if self._resize_corner == -1:  # move
            nx = max(0.0, min(ox + dx, pw - os))
            ny = max(0.0, min(oy + dy, ph - os))
            self.crop_rect = (int(nx), int(ny), os)
        else:
            ci = self._resize_corner
            ns = os
            nx, ny = ox, oy
            if ci == 0:  # tl
                ns = max(5.0, min(ox + os - ix, oy + os - iy, pw, ph))
                nx = ox + os - ns
                ny = oy + os - ns
            elif ci == 1:  # tr
                ns = max(5.0, min(ix - ox, oy + os - iy, pw, ph))
                nx = ox
                ny = oy + os - ns
            elif ci == 2:  # br
                ns = max(5.0, min(ix - ox, iy - oy, pw, ph))
                nx = ox
                ny = oy
            elif ci == 3:  # bl
                ns = max(5.0, min(ox + os - ix, iy - oy, pw, ph))
                nx = ox + os - ns
                ny = oy
            ns = max(5.0, min(ns, pw - nx, ph - ny))
            nx = max(0.0, min(nx, pw - ns))
            ny = max(0.0, min(ny, ph - ns))
            self.crop_rect = (int(nx), int(ny), int(ns))
        self._apply_crop()
        self._update_selection_visual()
        self._last_slider_change = time.time()
        self._dirty = True
        # Occasional console feedback so user can see drag is alive even if visual is subtle
        self._drag_dbg = getattr(self, "_drag_dbg", 0) + 1
        if self._drag_dbg % 8 == 0:
            print(f"[CROP] drag live -> {self.crop_rect} (mouse local lx={lx:.0f} ly={ly:.0f})")

    def _on_mouse_release_crop(self, sender, app_data, user_data):
        if self._is_dragging:
            print(f"[CROP] RELEASE while crop drag active (app_data={app_data})")
            print("[CROP] mouse release while dragging crop")
            self._end_crop_drag()
            current = dpg.get_value("status_text") or ""
            if "Dragging crop" in current:
                dpg.set_value("status_text", "Crop drag released — applying change...")
            # existing global release will force if dirty (which will run _do_update_preview and overwrite status)

    def _on_mouse_move(self, sender, app_data, user_data):
        """Fallback for starting a crop drag on trackpads / devices where the mouse_down
        events report button values in non-standard ways (e.g. [0, small_delta] or repeated
        while "pressed").
        If the left button is physically down (per is_mouse_button_down) and we are not
        already in a drag, and the cursor is over the crop rect/handles in the photo area,
        we latch the drag state here. The normal _on_mouse_drag will then take over.
        """
        if self._is_dragging:
            if not dpg.is_mouse_button_down(0):
                self._end_crop_drag()
            return
        if not dpg.is_mouse_button_down(0):
            return
        # Cooldown after a release to prevent the fallback from immediately re-latching
        # on flaky trackpad button-up reporting (is_mouse_button_down staying True briefly).
        if time.time() - getattr(self, '_last_drag_end_ts', 0) < 0.15:
            return
        mpos = dpg.get_mouse_pos(local=False)
        # Only consider positions that could be over the left preview to avoid work/spam
        if mpos[0] > 550:
            return
        if self._try_begin_crop_drag(mpos):
            print("[CROP] drag latched via mouse_move + is_mouse_button_down(0) fallback")

    def _reset_crop(self, sender=None, app_data=None, user_data=None):
        """Reset crop to the largest centered square (full width or height)."""
        if not self.full_pil or not self.input_path:
            return
        w, h = self.full_pil.size
        side = min(w, h)
        self.crop_rect = ((w - side) // 2, (h - side) // 2, side)
        self._end_crop_drag()
        self._apply_crop()
        self._compute_display_rect()
        self._update_selection_visual()
        self._last_crop = None
        dpg.set_value("status_text", "Crop reset to centered max square — updating...")
        self._do_update_preview()

    def _render_svg_to_png(self, svg_path, png_path, size=380):
        """Best-effort SVG->PNG for the vector preview pane.
        Prefers inkscape (consistent with EPS export). Falls back to macOS qlmanage.
        Returns True if png_path now exists with useful content.
        """
        # 1. inkscape (preferred)
        try:
            subprocess.run(
                [
                    "inkscape",
                    svg_path,
                    "--export-type=png",
                    "--export-filename", png_path,
                    f"--export-width={size}",
                ],
                check=True,
                capture_output=True,
                timeout=15,
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
                subprocess.run(
                    ["qlmanage", "-t", "-s", str(size), "-o", out_dir, svg_path],
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
        Skips full preprocess if only 'vector-only' settings (detail, corner) changed.
        """
        dpg.set_value("loading_indicator", "⏳ Processing...")
        dpg.set_value("status_text", "Processing ⏳ ... (may lag on complex images)")
        try:
            config = self.get_config()
            preprocess_keys = ["denoise_strength", "blur_radius", "threshold_offset", "invert", "min_area", "block_size", "line_thickness"]
            do_preprocess = (
                self._last_config is None or
                any(config[k] != self._last_config.get(k, None) for k in preprocess_keys) or
                self.crop_rect != getattr(self, "_last_crop", None)
            )

            if do_preprocess:
                proc_in = self.process_input or self.input_path
                processed = self.preprocess(proc_in, config)
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
                    proc_in = self.process_input or self.input_path
                    processed = self.preprocess(proc_in, config)
                    if processed is None:
                        dpg.set_value("status_text", "Failed to load/process image")
                        dpg.set_value("loading_indicator", "")
                        return
                    cv2.imwrite(self.preproc_temp, processed)
                    self.current_preprocessed = self.preproc_temp
                    self._last_binary = processed

            self._last_crop = self.crop_rect

            # Vector step — the params that must affect the live right-hand pane.
            # Wrapped to catch visioncortex "overflow" (too many clusters from speckly binary on high-res images).
            try:
                vtracer.convert_image_to_svg_py(
                    self.preproc_temp,
                    self.svg_temp,
                    colormode="binary",
                    filter_speckle=config["min_area"],
                    corner_threshold=config["corner_threshold"],
                    length_threshold=max(3.5, min(10.0, config["detail_level"] * 10)),
                    splice_threshold=config.get("splice_threshold", 45),
                    max_iterations=10,
                )
            except Exception as ve:
                # The Rust panic prints to stderr before this; we turn it into a nice status.
                dpg.set_value(
                    "status_text",
                    f"Vectorize failed (overflow or bad params): {ve}. "
                    "Try ↑ Min Area (speckle), ↑ Denoise, or adjust Threshold. Preview not updated."
                )
                dpg.set_value("loading_indicator", "")
                return  # keep previous textures / last good state
            self.last_svg = self.svg_temp

            orig_pil = self.full_pil if self.full_pil is not None else (self._orig_pil if self._orig_pil is not None else Image.open(self.input_path))

            vec_pil = None
            render_size = self.texture_size  # render smaller for mem, we resample to tex anyway
            if self._render_svg_to_png(self.svg_temp, self.vec_preview_temp, size=render_size):
                if os.path.exists(self.vec_preview_temp):
                    vec_pil = Image.open(self.vec_preview_temp)

            if vec_pil is None:
                dpg.set_value(
                    "status_text",
                    "Live preview (vector raster used fallback — install inkscape for best vector preview)"
                )
                vec_pil = Image.fromarray(processed).convert("L")
            else:
                dpg.set_value("status_text", "Live vector preview updated.")

            # Push fresh pixel data into the (already-bound) dynamic textures.
            # Now using compact np arrays (no .tolist()), so far lower per-update RAM.
            orig_data = self._pil_to_texture_data(orig_pil)
            proc_data = self._pil_to_texture_data(Image.fromarray(processed))
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
            self._last_crop = None
            self.process_input = self.input_path

            # setup crop selection (centered square by default)
            try:
                self.full_pil = Image.open(path).convert("RGBA")
                self._orig_pil = self.full_pil  # for full context texture
                w, h = self.full_pil.size
                side = min(w, h)
                self.crop_rect = ((w - side) // 2, (h - side) // 2, side)
                self._end_crop_drag()
                self._apply_crop()
                self._compute_display_rect()
                self._update_selection_visual()
                self._last_crop = self.crop_rect

                # immediately show the full photo + selection box in left pane
                if self.full_pil:
                    odata = self._pil_to_texture_data(self.full_pil)
                    dpg.set_value("orig_texture", odata)
            except Exception:
                self.full_pil = None
                self._orig_pil = None
                self.crop_rect = None
                self.process_input = self.input_path
                self._end_crop_drag()
                if dpg.does_item_exist(self._sel_rect_tag):
                    try:
                        dpg.delete_item(self._sel_rect_tag)
                    except:
                        pass
                dpg.set_value("crop_info_text", "")

            dpg.set_value("status_text", f"Loaded: {Path(path).name} — updating previews...")
            self.update_preview()  # marks _dirty + _last_slider_change

            # For the *initial* preview after loading an image we want it right away,
            # not waiting for the debounce timer.
            if self._dirty:
                self._dirty = False
                self._last_update_ts = time.time()
                self._do_update_preview()

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
        if not str(save_path).lower().endswith(".svg"):
            save_path = str(Path(save_path).with_suffix(".svg"))
        self._do_save(save_path)

    def _do_save(self, save_path):
        config = self.get_config()
        try:
            vtracer.convert_image_to_svg_py(
                self.current_preprocessed,
                save_path,
                colormode="binary",
                filter_speckle=config["min_area"],
                corner_threshold=config["corner_threshold"],
                length_threshold=max(3.5, min(10.0, config["detail_level"] * 10)),
                splice_threshold=config.get("splice_threshold", 45),
                max_iterations=10,
            )
        except Exception as e:
            dpg.set_value("status_text", f"Vectorize failed: {e}")
            return

        self.last_svg = save_path

        eps_path = Path(save_path).with_suffix(".eps")
        eps_ok = False
        try:
            subprocess.run(
                ["inkscape", save_path, "--export-filename", str(eps_path)],
                check=True,
                capture_output=True,
            )
            eps_ok = True
        except Exception:
            pass

        msg = f"Saved SVG: {save_path}"
        if eps_ok:
            msg += f"  (+EPS: {eps_path})"
        else:
            msg += "  (EPS not generated — inkscape not found in PATH)"
        dpg.set_value("status_text", msg)

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
        """Force an immediate preview update when the user releases the mouse after dragging a slider.
        This (together with the idle poller) implements "don't start heavy processing until the user stops moving the slider".
        """
        was_dragging_crop = getattr(self, "_is_dragging", False)
        if getattr(self, "_dirty", False) and self.input_path:
            self._dirty = False
            self._last_update_ts = time.time()
            self._do_update_preview()
        if was_dragging_crop or self._is_dragging:
            self._end_crop_drag()
            print("[CROP] global release cleared crop drag state")
            current = dpg.get_value("status_text") or ""
            if "Dragging crop" in current:
                dpg.set_value("status_text", "Crop drag released — applying change...")

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
                self._last_update_ts = time.time()
                self._do_update_preview()
        # Keep the poller alive
        self._schedule_debounce_check()

    def _cleanup_temps(self):
        for p in (
            getattr(self, "preproc_temp", None),
            getattr(self, "svg_temp", None),
            getattr(self, "vec_preview_temp", None),
            getattr(self, "crop_temp", None),
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
    app = StencilApp()
    app.run()
