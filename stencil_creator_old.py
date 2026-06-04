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


class VectorGUI:
    def __init__(self):
        self.input_path = None
        self.current_preprocessed = None
        self.last_svg = None

        # Managed temp files in /tmp to avoid cluttering user's image folders
        self.temp_dir = Path(tempfile.gettempdir())
        self.preproc_temp = str(self.temp_dir / "stencil_creator_preproc.png")
        self.svg_temp = str(self.temp_dir / "stencil_creator_vector.svg")
        self.vec_preview_temp = str(self.temp_dir / "stencil_creator_vecprev.png")

        self.texture_size = 400
        self.preview_size = 380

        self._build_ui()

    def _create_checkerboard_texture_data(self, size, square=32):
        """Create a simple checkerboard so the placeholder is clearly visible and we can verify textures are working."""
        data = []
        for y in range(size):
            for x in range(size):
                if ((x // square) + (y // square)) % 2 == 0:
                    data.extend([0.1, 0.1, 0.1, 1.0])   # very dark
                else:
                    data.extend([0.6, 0.6, 0.6, 1.0])   # much lighter gray
        return data

    def _replace_texture(self, tag, data):
        """Delete existing texture (if any) and re-create it under the same tag with new data.
        This is a reliable way to force Dear PyGui to display fresh image data.
        """
        print(f"[DEBUG] Replacing texture {tag} (len data={len(data)})")
        if dpg.does_item_exist(tag):
            dpg.delete_item(tag)
        dpg.add_raw_texture(
            width=self.texture_size,
            height=self.texture_size,
            default_value=data,
            format=dpg.mvFormat_Float_rgba,
            tag=tag,
            parent="tex_reg",
        )
        print(f"[DEBUG] Texture {tag} re-added.")

    def _refresh_preview_column(self, col_tag, img_tag, tex_tag):
        """Delete the existing image widget (if any) and re-add it to the column group.
        This forces Dear PyGui to bind the (newly replaced) texture.
        """
        print(f"[DEBUG] Refreshing preview column {col_tag} (re-adding {img_tag} with {tex_tag})")
        print(f"[DEBUG]   Does texture {tex_tag} exist at this moment? {dpg.does_item_exist(tex_tag)}")
        if dpg.does_item_exist(img_tag):
            dpg.delete_item(img_tag)
        dpg.add_image(
            tag=img_tag,
            texture_tag=tex_tag,
            width=self.preview_size,
            height=self.preview_size,
            parent=col_tag,
        )
        print(f"[DEBUG] Image {img_tag} re-added to {col_tag}.")

    def _build_ui(self):
        dpg.create_context()

        # Create textures as early as possible using the standard 'with' pattern.
        # This is the most commonly working way in Dear PyGui examples.
        with dpg.texture_registry(tag="tex_reg"):
            init_data = self._create_checkerboard_texture_data(self.texture_size)
            for ttag in ("orig_texture", "proc_texture", "vec_texture"):
                dpg.add_raw_texture(
                    width=self.texture_size,
                    height=self.texture_size,
                    default_value=init_data,
                    format=dpg.mvFormat_Float_rgba,
                    tag=ttag,
                )
            print("[DEBUG] Texture registry and 3 placeholder textures created (checkerboard).")

        dpg.create_viewport(
            title="Phone Photo → Screen Print Vector",
            width=1420,
            height=880,
            resizable=True,
        )
        dpg.setup_dearpygui()
        print("[DEBUG] Viewport and DPG setup done.")

        # Main UI window
        with dpg.window(tag="main_win"):
            # Top action bar
            with dpg.group(horizontal=True):
                dpg.add_button(label="Load Photo", callback=self.load_image, width=110)
                dpg.add_button(label="Save Vector", callback=self.save_vector, width=110)
                dpg.add_button(label="Open SVG", callback=self.open_svg, width=110)

            dpg.add_separator()

            # Controls
            with dpg.collapsing_header(
                label="Adjust Settings (changes update previews live — may lag on complex images while dragging)",
                default_open=True,
            ):
                dpg.add_slider_float(
                    label="Detail Level",
                    tag="slider_detail",
                    default_value=0.65,
                    min_value=0.1,
                    max_value=1.0,
                    callback=self.update_preview,
                    width=380,
                )
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
                    label="Min Area",
                    tag="slider_min_area",
                    default_value=25,
                    min_value=5,
                    max_value=100,
                    callback=self.update_preview,
                    width=380,
                )
                dpg.add_slider_int(
                    label="Corner Threshold",
                    tag="slider_corner",
                    default_value=60,
                    min_value=30,
                    max_value=120,
                    callback=self.update_preview,
                    width=380,
                )
                dpg.add_checkbox(
                    label="Invert Image",
                    tag="check_invert",
                    default_value=False,
                    callback=self.update_preview,
                )

            dpg.add_separator()

            dpg.add_text("Preview boxes (bordered) should appear immediately. Content (checkerboard before load, images after) is added shortly after launch in the scheduled refresh. Report what you see inside the bordered areas.", color=(255, 200, 100))

            # Preview "boxes" using child windows so the area is clearly visible even if the image content fails to render.
            # Images (the actual texture content) are added dynamically in the refresh after everything is ready.
            with dpg.group(horizontal=True):
                with dpg.child_window(tag="col_orig", border=True, width=self.preview_size + 10, height=self.preview_size + 30):
                    dpg.add_text("Original")
                with dpg.child_window(tag="col_proc", border=True, width=self.preview_size + 10, height=self.preview_size + 30):
                    dpg.add_text("Preprocessed (tracing input)")
                with dpg.child_window(tag="col_vec", border=True, width=self.preview_size + 10, height=self.preview_size + 30):
                    dpg.add_text("Vector Result (live)", color=(140, 220, 140))

            dpg.add_spacer(height=6)
            dpg.add_text(tag="status_text", default_value="Load a photo to begin. Sliders update the vector preview live.")

        # Hidden file dialogs
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

        dpg.set_primary_window("main_win", True)
        dpg.set_exit_callback(self._on_exit)
        dpg.show_viewport()
        print("[DEBUG] Viewport shown.")

        # Schedule the initial preview refresh for frame 2 (after the first render) to give DPG time to be fully ready.
        def _initial_refresh(sender, app_data, user_data):
            print("[DEBUG] Running scheduled initial preview column refresh (frame callback)...")
            self._refresh_preview_column("col_orig", "img_orig", "orig_texture")
            self._refresh_preview_column("col_proc", "img_proc", "proc_texture")
            self._refresh_preview_column("col_vec", "img_vec", "vec_texture")
            print("[DEBUG] Scheduled initial preview refresh done.")

        dpg.set_frame_callback(2, _initial_refresh)
        print("[DEBUG] Initial refresh scheduled for frame 2.")

    def get_config(self):
        return {
            "detail_level": dpg.get_value("slider_detail"),
            "denoise_strength": dpg.get_value("slider_denoise"),
            "blur_radius": dpg.get_value("slider_blur"),
            "threshold_offset": dpg.get_value("slider_threshold"),
            "min_area": dpg.get_value("slider_min_area"),
            "invert": dpg.get_value("check_invert"),
            "corner_threshold": dpg.get_value("slider_corner"),
        }

    def preprocess(self, img_path, config):
        """Load via PIL first (broader format support incl. HEIC if pillow-heif present)
        then process to binary for vectorization."""
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

        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, config["threshold_offset"]
        )

        if config["invert"]:
            thresh = cv2.bitwise_not(thresh)

        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        return cleaned

    def _pil_to_texture_data(self, pil_img, size=None):
        if size is None:
            size = self.texture_size
        canvas = Image.new("RGBA", (size, size), (25, 25, 25, 255))
        pil_img = pil_img.convert("RGBA")
        pil_img.thumbnail((size, size), Image.LANCZOS)
        x = (size - pil_img.width) // 2
        y = (size - pil_img.height) // 2
        canvas.paste(pil_img, (x, y))
        arr = np.array(canvas, dtype=np.float32) / 255.0
        return arr.ravel().tolist()

    def _render_svg_to_png(self, svg_path, png_path, size=380):
        """Render SVG to PNG for live preview. Tries inkscape first (matches export),
        then macOS qlmanage (built-in, no deps). Returns True on success."""
        # inkscape (best, same renderer as EPS)
        try:
            subprocess.run(
                [
                    "inkscape",
                    svg_path,
                    "--export-type=png",
                    "--export-filename",
                    png_path,
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

        # macOS Quick Look (no install needed)
        if os.name == "posix":
            try:
                out_dir = str(Path(png_path).parent)
                expected_thumb = str(Path(out_dir) / (Path(svg_path).name + ".png"))
                subprocess.run(
                    ["qlmanage", "-t", "-s", str(size), "-o", out_dir, svg_path],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                # qlmanage thumbnail write can lag a little behind process exit; poll briefly
                for _ in range(12):
                    if os.path.exists(expected_thumb):
                        os.replace(expected_thumb, png_path)
                        return True
                    time.sleep(0.04)
                # fallback: pick any fresh png in out_dir that looks like a thumb
                for f in sorted(
                    Path(out_dir).glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True
                ):
                    if "stencil" in f.name or "vector" in f.name or "thumb" in f.name.lower():
                        os.replace(str(f), png_path)
                        return True
            except Exception:
                pass
        return False

    def update_preview(self, sender=None, app_data=None, user_data=None):
        print("[DEBUG] update_preview called")
        if not self.input_path:
            print("[DEBUG] No input_path, skipping")
            return

        try:
            config = self.get_config()
            processed = self.preprocess(self.input_path, config)
            if processed is None:
                dpg.set_value("status_text", "Failed to load/process image")
                return

            # Write preprocessed binary for vtracer (and for re-use on Save)
            cv2.imwrite(self.preproc_temp, processed)
            self.current_preprocessed = self.preproc_temp

            # Vectorize (this is what makes detail/min_area/corner affect live preview)
            vtracer.convert_image_to_svg_py(
                self.preproc_temp,
                self.svg_temp,
                colormode="binary",
                filter_speckle=config["min_area"],
                corner_threshold=config["corner_threshold"],
                length_threshold=int(config["detail_level"] * 10),
                max_iterations=10,
            )
            self.last_svg = self.svg_temp

            # Prepare display images
            orig_pil = Image.open(self.input_path)

            vec_pil = None
            if self._render_svg_to_png(self.svg_temp, self.vec_preview_temp, size=self.preview_size):
                if os.path.exists(self.vec_preview_temp):
                    vec_pil = Image.open(self.vec_preview_temp)

            if vec_pil is None:
                dpg.set_value(
                    "status_text",
                    "Preview updated (vector raster unavailable — install inkscape for accurate vector preview)",
                )
                vec_pil = Image.fromarray(processed)
            else:
                dpg.set_value("status_text", "Live vector preview updated.")

            # Update the three DPG textures by deleting and re-adding them with fresh data.
            # This is more reliable than set_value() in some Dear PyGui versions/contexts.
            orig_data = self._pil_to_texture_data(orig_pil)
            self._replace_texture("orig_texture", orig_data)

            proc_pil = Image.fromarray(processed)
            proc_data = self._pil_to_texture_data(proc_pil)
            self._replace_texture("proc_texture", proc_data)

            vec_data = self._pil_to_texture_data(vec_pil)
            self._replace_texture("vec_texture", vec_data)

            print("[DEBUG] All textures replaced. Now refreshing image widgets...")

            # Delete and re-add the image widgets inside their column groups.
            # This is the most reliable way to force the new textures to be displayed.
            self._refresh_preview_column("col_orig", "img_orig", "orig_texture")
            self._refresh_preview_column("col_proc", "img_proc", "proc_texture")
            self._refresh_preview_column("col_vec", "img_vec", "vec_texture")

        except Exception as e:
            dpg.set_value("status_text", f"Preview error: {e}")

    def load_image(self):
        dpg.show_item("load_dialog")

    def _extract_path(self, app_data, require_exists=True):
        """Robustly extract the selected file path from a DPG file_dialog callback.
        Handles differences across versions and the '.*' filter gotcha that can
        cause the extension to be replaced by '*' in the returned name.
        """
        if not isinstance(app_data, dict):
            return None

        candidates = []

        # 1. Preferred explicit key
        p = app_data.get("file_path_name")
        if isinstance(p, str) and p:
            candidates.append(p)

        # 2. selections dict (common structure: {basename: full_path} or reverse)
        selections = app_data.get("selections") or {}
        current_path = app_data.get("current_path") or ""
        for k, v in selections.items():
            for cand in (k, v):
                if isinstance(cand, str) and cand:
                    candidates.append(cand)
                    if current_path:
                        joined = os.path.join(current_path, cand.lstrip(os.sep + "/\\"))
                        candidates.append(joined)

        # 3. current_path + file_name
        fname = app_data.get("file_name")
        if isinstance(fname, str) and fname and current_path:
            joined = os.path.join(current_path, fname.lstrip(os.sep + "/\\"))
            candidates.append(joined)

        # Now pick the best candidate
        for cand in candidates:
            if isinstance(cand, str) and cand:
                if not require_exists:
                    return cand  # for save dialogs we may be creating a new file
                if os.path.isfile(cand):
                    return cand

        # As a last resort return the first non-empty string we saw (may still be bad)
        for cand in candidates:
            if isinstance(cand, str) and cand:
                return cand
        return None

    def _on_load_callback(self, sender, app_data, user_data):
        print("[DEBUG] Load dialog callback fired")
        dpg.hide_item("load_dialog")  # close dialog so main window can refresh cleanly
        path = self._extract_path(app_data, require_exists=True)
        if path:
            self.input_path = path
            dpg.set_value("status_text", f"Loaded: {Path(path).name} — updating preview...")
            self.update_preview()

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
                length_threshold=int(config["detail_level"] * 10),
                max_iterations=10,
            )
        except Exception as e:
            dpg.set_value("status_text", f"Vectorize failed: {e}")
            return

        self.last_svg = save_path

        # Try EPS export if inkscape present (standard for screen print RIPs)
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
            # Allow opening the live (unsaved) vector for inspection
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

    def _cleanup_temps(self):
        for p in (
            getattr(self, "preproc_temp", None),
            getattr(self, "svg_temp", None),
            getattr(self, "vec_preview_temp", None),
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
    app = VectorGUI()
    app.run()
