# SPEC: stencil_app.py — Phone Photo → Screen Print Vector (Dear PyGui)

Target: AI agent reimplementation without source or runtime access. Implement `stencil_app.py` as single file `StencilApp` class + `if __name__ == "__main__"`. No Tkinter. No threads for debounce (use DPG frame callbacks).

## ENTRY

```
python stencil_app.py [-v|--verbose]
```

- `-v`: print `[INFO] ...` startup diagnostics only; no debug spam in normal mode.
- `StencilApp(verbose=False).__init__` → `_build_ui()`; `run()` → `dpg.start_dearpygui()` then `dpg.destroy_context()`.

## DEPS (requirements.txt)

```
dearpygui
numpy
opencv-python
pillow
vtracer
```

Optional runtime (not pip): `inkscape` on PATH for SVG→PNG previews, EPS/PDF export, modal trace zoom. macOS fallback: `qlmanage` for SVG→PNG only.

## MODULE CONSTANTS

```python
_LENGTH_MIN = 3.5
_LENGTH_MAX = 10.0
_DETAIL_MIN = 0.35
_DETAIL_MAX = 1.0
_INLINE_RASTER_MAX = 2048      # inkscape --export-width cap for inline vec preview
_MODAL_RASTER_MAX = 2048       # same for modal vec preview
_INLINE_TEXTURE_SIZE = 512     # dynamic texture W/H for 3 inline previews
_PREVIEW_BINARIZE_THRESHOLD = 220  # L<220 → 0 else 255 on rasterized SVG preview
_PREVIEW_CHILD_PAD_X = 30      # child_window border eats interior; outer W = preview_size + this
_PREVIEW_CHILD_PAD_Y = 16      # outer H = preview_size + this
```

## CLASS: StencilApp

### Class attrs

```python
TRACE_MODES = ("Smooth (spline)", "Polygon", "Pixel-accurate")
SAVE_EXTENSIONS = (".svg", ".eps", ".pdf")
```

### Instance state (__init__)

| Field | Purpose |
|-------|---------|
| `verbose` | bool |
| `input_path` | str\|None loaded image path |
| `current_preprocessed` | str\|None path to preproc PNG temp |
| `last_svg` | str\|None path to last SVG |
| `_orig_pil` | PIL RGBA cache of loaded photo |
| `temp_dir` | `Path(tempfile.gettempdir())` |
| `preproc_temp` | `{temp_dir}/stencil_creator_preproc.png` |
| `svg_temp` | `{temp_dir}/stencil_creator_vector.svg` |
| `vec_preview_temp` | `{temp_dir}/stencil_creator_vecprev.png` |
| `vec_preview_modal_temp` | `{temp_dir}/stencil_creator_vecprev_modal.png` |
| `texture_size` | 512 |
| `preview_size` | 470 (image_button W/H) |
| `preview_child_w` | preview_size + 30 |
| `preview_child_h` | preview_size + 16 |
| `_preview_cache` | dict `orig`/`proc`/`vec` → PIL copies for modal |
| `_image_dims` | `(pw, ph)` processed bitmap size |
| `_modal_tex_size` | `(cw, ch)` current modal_texture buffer |
| `_dirty` | bool pending preview recompute |
| `_last_slider_change` | float monotonic time |
| `_debounce_delay` | 0.30 seconds |
| `_last_config` | dict\|None last get_config() after successful preview |
| `_last_binary` | np.ndarray\|None last preprocess output |
| `_last_input_path` | set in _do_update_preview |

No square crop. Full input image dimensions flow to SVG export (vtracer from full-frame binary).

---

## PIPELINE (data flow)

```
input_path
  → preprocess(config) → uint8 binary H×W (0=ink, 255=paper)
  → cv2.imwrite(preproc_temp)
  → vtracer.convert_image_to_svg_py(preproc_temp, svg_temp, **vtracer_kwargs)
  → inkscape rasterize svg → PNG → optional binarize → vec preview
  → 3× letterbox to 512² textures + cache PIL for modal
```

**Three preview semantics (critical):**

| kind | Source | Display resample | Trust for tuning |
|------|--------|------------------|------------------|
| orig | Full photo RGBA→texture | LANCZOS | Reference photo |
| proc | Exact preprocess binary 0/255 | NEAREST | Ground truth vtracer **input** |
| vec | SVG rasterized at min(pw,2048) px wide, then binarized | LANCZOS (+ binarize) | Approximate trace look; modal = export-width raster |

Right pane ≠ pixel copy of center (filled SVG paths + AA + rasterization).

---

## preprocess(img_path, config) → np.ndarray | None

1. **Load gray uint8:**
   - PIL `Image.open`; modes RGBA/LA/P → L; else non-L → L; `np.array`.
   - On PIL fail: `cv2.imread` + `COLOR_BGR2GRAY`.
   - Return None if both fail.

2. `cv2.fastNlMeansDenoising(gray, h=config["denoise_strength"])`

3. `cv2.GaussianBlur(denoised, (5, 5), config["blur_radius"])` — kernel **fixed** 5×5; slider is **sigma** only.

4. `block_size = max(3, int(config["block_size"]) | 1)` — force odd ≥3.

5. `cv2.adaptiveThreshold(blurred, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, block_size, config["threshold_offset"])`

6. If `config["invert"]`: `cv2.bitwise_not`

7. `cv2.morphologyEx(..., MORPH_OPEN, 3×3 ones, iterations=1)`

8. `_filter_small_components(cleaned, config["min_area"])`:
   - 8-connectivity on binary: remove components area < min_area (flip small white to 0).
   - On inverted binary: remove small black specks (flip to 255).
   - Guards vtracer cluster overflow (~65k labels).

9. `_thicken_ink_lines(cleaned, config["line_width_px"])`:
   - If width_px ≤ 0: return unchanged.
   - `ksize = max(3, int(round(width_px)) | 1)` odd.
   - Ellipse kernel dilate on ink mask `(binary==0)`, write back 0/255.

Return final binary.

---

## get_config() → dict keys

| key | DPG tag | type |
|-----|---------|------|
| detail_level | slider_detail | float |
| denoise_strength | slider_denoise | int |
| blur_radius | slider_blur | float |
| threshold_offset | slider_threshold | int |
| block_size | slider_block_size | int |
| min_area | slider_min_area | int |
| line_width_px | slider_line_width | float |
| corner_threshold | slider_corner | int |
| splice_threshold | slider_splice_threshold | int |
| trace_mode | combo_trace_mode | str (TRACE_MODES label) |
| vector_speckle | slider_vector_speckle | int |
| max_iterations | slider_max_iterations | int |
| path_precision | slider_path_precision | int |
| invert | check_invert | bool |

---

## vtracer mapping

```python
def _length_threshold_from_detail(detail_level):
    span = _DETAIL_MAX - _DETAIL_MIN  # 0.65
    t = (_DETAIL_MAX - detail_level) / span
    return _LENGTH_MIN + t * (_LENGTH_MAX - _LENGTH_MIN)
    # detail=1.0 → 3.5 (more segments, higher fidelity)
    # detail=0.35 → 10.0 (simpler paths)

def _trace_mode_to_vtracer(label):
    {"Smooth (spline)": "spline", "Polygon": "polygon", "Pixel-accurate": "none"}.get(label, "polygon")

def _vtracer_kwargs(config):
    return {
        "colormode": "binary",
        "filter_speckle": int(config.get("vector_speckle", 0)),
        "mode": _trace_mode_to_vtracer(config["trace_mode"]),
        "corner_threshold": int(config["corner_threshold"]),
        "length_threshold": _length_threshold_from_detail(float(config["detail_level"])),
        "splice_threshold": int(config.get("splice_threshold", 45)),
        "max_iterations": int(config.get("max_iterations", 10)),
        "path_precision": int(config.get("path_precision", 8)),
    }
```

Call: `vtracer.convert_image_to_svg_py(preproc_temp, svg_temp, **kwargs)`.

**Preset button** `_apply_match_preprocess_preset`: set slider_detail=1.0, slider_corner=0, slider_splice_threshold=180, combo_trace_mode="Pixel-accurate", slider_vector_speckle=0; status message; `update_preview()`.

Preprocess **min_area** is separate from vtracer **vector_speckle** (default speckle 0).

---

## SVG → PNG: _render_svg_to_png(svg, png, export_width, timeout) → bool

`export_width = max(64, int(export_width))`.

1. **inkscape** (preferred):
   ```
   inkscape <svg> --export-type=png --export-filename <png> --export-width=<export_width>
   ```
   check=True, capture_output, timeout. Return True if png exists.

2. **qlmanage** (posix only, macOS):
   `qlmanage -t -s min(export_width,1024) -o <dir> <svg>`
   Poll 12×0.04s for `{svg_basename}.png` in out_dir → `os.replace` to png_path.
   Else newest `*.png` in dir with name containing stencil|vector|thumb|preview.

Return False if no file.

`_svg_export_width(img_width, max_dim)`: `min(img_width, max_dim)` if img_width>0 else max_dim.

Inline vec: `export_w = _svg_export_width(pw, _INLINE_RASTER_MAX)`; timeout `min(90, 20 + export_w // 80)`.

`_binarize_preview_pil(pil_L)`: gray array; pixels `< 220` → 0 else 255.

`_load_inkscape_png_as_gray(path)`: PIL open convert L or None.

---

## Inkscape vector export: _inkscape_export(svg, out, export_type=None, timeout=60) → (ok, err)

```
inkscape <svg> --export-type=<type> --export-filename <out>
```

type from extension if None. ok if file exists and size>0. Errors: not found, CalledProcessError (stderr[:240]), TimeoutExpired.

---

## Texture / PIL helpers

**Checker placeholder** `_create_checkerboard_texture_data(size, square=16)`:
- `np.zeros((size,size,4), float32)`, alpha=1.
- Alternating squares RGB 0.12 / 0.55.

**`_pil_to_texture_data(pil, size=texture_size, resample=LANCZOS)`**:
- RGBA canvas `(size,size)` fill `(25,25,25,255)`.
- thumbnail fit inside square; center paste.
- Return `np.float32 (h,w,4) / 255.0` — **not** ravel list; DPG dynamic texture accepts ndarray.

**`_fit_pil_to_texture_data(pil, canvas_w, canvas_h, resample)`**: same letterbox on rectangle for modal.

**Inline preview update:** `dpg.set_value("orig_texture"|"proc_texture"|"vec_texture", data)` — never delete/recreate inline textures after init.

---

## LIVE UPDATE / DEBOUNCE

**`update_preview()`** (all slider/checkbox/combo callbacks + preset + load after mark):
- If no `input_path`: return.
- `_last_slider_change = time.time()`; `_dirty = True`.
- Do **not** call `_do_update_preview()` here.

**Triggers for `_do_update_preview()`:**
1. `_on_global_mouse_release` if `_dirty` and input_path → `_flush_preview_update()` (clears dirty, runs heavy).
2. `_debounce_check` frame callback every ~8 frames: if `_dirty` and `time.time() - _last_slider_change >= 0.30` → clear dirty, `_do_update_preview()`.
3. `_on_load_callback`: after `update_preview()`, if `_dirty` → `_flush_preview_update()` immediate (no wait on first load).

**`_schedule_debounce_check`**: `set_frame_callback(current_frame + 8, _debounce_check)`; end of `_debounce_check` reschedules itself.

**`_do_update_preview()` optimization:**
- `preprocess_keys`: denoise_strength, blur_radius, threshold_offset, invert, min_area, block_size, line_width_px
- `vector_keys`: detail_level, corner_threshold, splice_threshold, trace_mode, vector_speckle, max_iterations, path_precision
- `do_preprocess` if no `_last_binary`, input changed, or preprocess_keys changed vs `_last_config`
- `do_vector` if do_preprocess or vector_keys changed
- If not do_vector and last_svg exists: reuse svg_temp
- Else vtracer; on exception set status overflow hint, clear loading_indicator, return
- Build orig_pil from `_orig_pil` cache or `Image.open(input_path)`
- Vec: render svg at export_w; binarize; fallback proc_pil + status if no inkscape
- Cache `_preview_cache` copies; `set_value` three textures; `_last_config = dict(config)`; clear loading_indicator

UI during processing: `loading_indicator` = "⏳ Processing..."; `status_text` = "Processing…".

---

## DPG UI LAYOUT

### Viewport

- title: `"Photo → Image Vector"`
- width 1540, height 880, resizable

### texture_registry tag=`tex_reg`

| tag | size | notes |
|-----|------|-------|
| orig_texture, proc_texture, vec_texture | 512×512 | checker init |
| modal_tex_stub | 4×4 | detach target before modal texture delete |
| modal_texture | 4×4 | recreated on modal open |

### Theme tag=`preview_image_btn_theme` (mvImageButton)

- FramePadding 0,0
- FrameBorderSize 0
- Button (0,0,0,0); ButtonHovered (50,60,75,90); ButtonActive (70,90,110,120)

### window tag=`main_win` (primary)

**Row 1** horizontal buttons: Load Photo (`load_image` 110w), Save Vector (`save_vector`), Open SVG (`open_svg`).

**Row 2** `settings_row` horizontal:

**`settings_col_preprocess`** child_window 690×340, border=False, no_scrollbar:
- collapsing_header "Preprocessing settings" default_open
- slider_denoise: int 0-40 default 15 label "Noise reduction"
- slider_blur: float 0-5 default 1.5 label "Blur before threshold (σ)"
- slider_threshold: int -20..20 default 5 label "Threshold bias (C)"
- slider_block_size: int 3-31 default 11 label "Local threshold window"
- slider_min_area: int 10-100 default 25 label "Remove blobs smaller than (px)"
- slider_line_width: float 0-12 default 0 format "%.1f" label "Thicken stencil lines (px)"
- check_invert: "Invert Image" default False

**`settings_col_vector`** child_window 690×340, border=False, no_scrollbar:
- collapsing_header "Vector settings" default_open
- hint text color (160,180,200): trace raster up to 2048px, click to zoom
- slider_detail: float 0.35-1.0 default 1.0 label "Path fidelity"
- slider_corner: int 0-180 default 30 label "Corner smoothing (°)"
- slider_splice_threshold: int 0-180 default 45 label "Join straight segments (°)"
- slider_vector_speckle: int 0-80 default 0 label "Tracer: ignore specks (px)"
- slider_max_iterations: int 1-20 default 10 label "Curve fit iterations"
- slider_path_precision: int 1-12 default 8 label "SVG path decimal places"
- button "Preset: max trace fidelity" → preset callback
- combo_trace_mode: TRACE_MODES default "Polygon"

All sliders width 360, callback `update_preview`.

**Row 3** horizontal 3 preview columns (group each):

| column | title texts | child tag | image tag | texture | user_data |
|--------|-------------|-----------|-----------|---------|-----------|
| 1 | "Original photo" + subtitle | col_orig | img_orig | orig_texture | orig |
| 2 | "Stencil bitmap" + subtitle | col_proc | img_proc | proc_texture | proc |
| 3 | "Traced SVG preview" + subtitle green tint | col_vec | img_vec | vec_texture | vec |

Each `child_window`: border=True, width=preview_child_w, height=preview_child_h, **no_scrollbar=True**, empty at build.

Hint: "Click any preview to open a full-window view..."

**Status row:** `loading_indicator` (color 255,180,100), `status_text` default load hint.

### Preview widgets (frame callback 2)

`set_frame_callback(2, _initial_add_preview_images)`:
- For each column: delete img if exists; `add_image_button(texture_tag, tag, preview_size×preview_size, parent=col, callback=_on_preview_button, user_data=kind, background_color=(0,0,0,0))`
- `bind_item_theme(img, preview_image_btn_theme)`

**Must use add_image_button not add_image** — plain Image does not receive clicks.

### Modal window tag=`preview_modal`

- modal=True, show=False, no_resize, 920×720
- modal_caption text
- modal_image → modal_texture, initial 800×600
- Close button → `_close_preview_modal`

**`_open_preview_modal(kind)`** kind ∈ orig|proc|vec:
- pil from `_preview_cache`; proc resample NEAREST; vec: re-render svg at `_MODAL_RASTER_MAX` width to vec_preview_modal_temp if possible else cache
- canvas = `(max(320, vw*0.88), max(280, vh*0.78))` from viewport client size
- `_fit_pil_to_texture_data` → `_replace_modal_texture`
- caption with title + `{pw}×{ph} px`; show + focus modal

**`_replace_modal_texture(data)`** — DPG alias gotcha:
- If same (cw,ch) as `_modal_tex_size`: `set_value("modal_texture", data)` + configure modal_image w/h
- Else: `configure_item(modal_image, texture_tag="modal_tex_stub")` then `delete_item("modal_texture")` then `add_dynamic_texture(cw,ch,data,tag=modal_texture,parent=tex_reg)` then point modal_image back

### file_dialog load_dialog

show=False, callback `_on_load_callback`, 780×580, directory_selector=False, extensions .* .jpg .jpeg .png .bmp .webp .tiff

### file_dialog save_dialog

show=False, callback `_on_save_callback`, default_filename stencil.svg, extensions .svg .eps .pdf

### handler_registry global_mouse_handlers

`add_mouse_release_handler` → `_on_global_mouse_release`

### Lifecycle

`set_primary_window("main_win")`, `set_exit_callback(_on_exit)`, `show_viewport`, `_schedule_debounce_check()` at end of _build_ui.

---

## FILE DIALOG PATH: _extract_path(app_data, require_exists=True)

From dict `app_data`:
- candidates: file_path_name; selections keys/values; join(current_path, selection); join(current_path, file_name)
- If require_exists: return first candidate that `os.path.isfile`
- Else return first candidate string
- Else return any candidate; else None

Load: require_exists=True. Save: False; if suffix not in SAVE_EXTENSIONS force `.svg`.

---

## SAVE: _do_save(save_path)

1. `vtracer.convert_image_to_svg_py(current_preprocessed, svg_dest, **kwargs)` where svg_dest = save_path if .svg else svg_temp
2. `last_svg = svg_dest`
3. If .svg: status Saved SVG
4. Else `_inkscape_export(svg_dest, save_path, export_type)` → status saved or SVG ok but export failed

**save_vector**: require current_preprocessed exists else status "Load a photo first".

---

## OPEN SVG: open_svg()

target = last_svg or svg_temp if exists; posix `open` else `os.startfile`; status messages.

---

## LOAD: _on_load_callback

hide load_dialog; path = extract; set input_path; reset _last_config, _last_binary, _last_slider_change, _last_input_path; load _orig_pil RGBA + immediate orig_texture set_value; status with dimensions; update_preview + _flush_preview_update if dirty.

---

## CLEANUP: _on_exit → _cleanup_temps

unlink preproc_temp, svg_temp, vec_preview_temp, vec_preview_modal_temp if exist.

---

## DPG IMPLEMENTATION RULES (do not deviate)

1. **Textures:** `add_dynamic_texture` + `set_value` for inline 3 previews. Do not delete/recreate on each update.
2. **Previews:** `add_image_button` + direct callback + user_data kind. Not `add_image` + item_handler_registry (clicks won't fire on Image).
3. **Child window sizing:** bordered child inner area < outer; pad +30/+16 and no_scrollbar or scrollbars appear with 470×470 button inside 472×472 box.
4. **Modal texture:** always detach modal_image to modal_tex_stub before delete_item(modal_texture).
5. **Debounce:** never run vtracer on every slider drag frame; mouse release + 300ms idle poller.
6. **Incremental recompute:** vector-only slider changes skip preprocess if binary unchanged.
7. **No Tk.** No raw_texture/ravel list pattern from older port.
8. **Full image:** no crop UI; SVG dimensions match full input bitmap pw×ph.

---

## STATUS MESSAGES (representative)

- Load: `Loaded: {name} ({w}×{h} px) — updating previews...`
- Success: `Output {pw}×{ph} px ({trace_mode}, length {lt:.1f}). Trace preview rasterized at {export_w}px wide — click to enlarge.`
- Vec preview fail: install Inkscape hint; fallback center bitmap
- vtracer fail: overflow / min area / denoise hint
- Save/export inkscape errors with truncated stderr

---

## OUT OF SCOPE FOR THIS SPEC

- CLI scripts (`stencil_creator.py`, `stencil_creator2.py`) — separate tools, fewer knobs.
- AI export format.
- README / human docs.

END SPEC.