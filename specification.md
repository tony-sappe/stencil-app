# SPEC FOR AI: stencil_creator_app.py (Dear PyGui port, current broken state)
# Goal: GUI tool "Phone Photo → Screen Print Vector"
# Load image, realtime sliders for params, live 3 previews (orig color, preproc binary, vector raster from SVG), export SVG + optional EPS (inkscape).
# Use conversation history + this file content for continuation. Dense, AI-only format. All key details, code, history, bugs included. No fluff.

## CORE REQUIREMENTS (from original query + conv)
- Open image file (jpg/png/etc, handle heic via PIL fallback).
- Preprocess: PIL load to gray, cv2.fastNlMeansDenoising(h=denoise), GaussianBlur((5,5), blur), adaptiveThreshold(Gaussian,11, threshold_offset), optional invert, morphologyEx(MORPH_OPEN,3x3,1).
- Vector: vtracer.convert_image_to_svg_py( preproc_png, svg, colormode='binary', filter_speckle=min_area, corner_threshold, length_threshold=detail*10, max_iterations=10 ).
- Previews: 3 side-by-side, realtime on param change (sliders release or live drag).
- Export: Save SVG (user path), try inkscape svg --export-filename eps.
- Open SVG via subprocess 'open' (posix) or startfile.
- Temps: /tmp/stencil_creator_*.png / .svg (cleanup on exit).
- Params/sliders (exact current defaults/ranges):
  detail: float 0.1-1.0 default 0.65
  denoise: int 0-40 d15
  blur: float 0-5 d1.5
  threshold: int -20-20 d5
  min_area: int 5-100 d25
  corner: int 30-120 d60
  invert: bool
- DPG GUI (user chose over Tk, no _tkinter).
- File dialogs: DPG add_file_dialog (robust _extract_path for file_path_name/selections/* filter bug fixed).
- Live update: update_preview on every slider/checkbox callback + load.
- Status: dpg text updates.
- EPS optional, graceful.

## CURRENT IMPLEMENTATION (from stencil_creator_app.py full read)
class VectorGUI:
  __init__: paths, texture_size=400, preview_size=380, _build_ui()
  _create_checkerboard... : builds list floats 0-1 checker (high contrast 32px squares, dark 0.1 / light 0.6) for placeholder.
  _replace_texture(tag, data): if exists delete, add_raw_texture(size, data, Float_rgba, parent="tex_reg", tag). Debug prints.
  _refresh_preview_column(col, img, tex): if img exists delete, add_image(tag, texture_tag=tex, size, parent=col). Debug + "does texture exist?"
  _build_ui():
    create_context()
    with texture_registry("tex_reg"): create 3 raw_textures placeholder checker. print DEBUG created.
    create_viewport(1420x880 resizable)
    setup_dearpygui()
    with window("main_win"):
      group h: 3 buttons (load/save/open callbacks)
      separator
      collapsing_header "Adjust Settings...": 6 sliders (float/int with tags/callback=update_preview/width=380) + checkbox invert.
      separator
      text hint about boxes
      group h: 3 child_window (border=True, w=preview+10, h=preview+30, tags col_*) with only add_text label (no initial image)
      spacer
      text status_text
    file_dialog load (show=False, callback _on_load, exts)
    file_dialog save (default stencil.svg, callback)
    set_primary main_win
    set_exit _on_exit
    show_viewport
    print
    set_frame_callback(2, _initial_refresh)  # calls 3 _refresh_preview_column
    print scheduled
  get_config: dpg.get_value all 7 params
  preprocess: exact as history (PIL gray fallback cv2, denoise, blur, thresh, invert, morph open)
  _pil_to_texture_data(pil, size=400): RGBA canvas dark, thumbnail center paste, np float32/255 ravel tolist()
  _render_svg_to_png: inkscape first (export png width), then posix qlmanage -t -s poll+replace + glob fallback for thumb. Returns bool
  update_preview: print debug, if no path return; config, processed=preprocess, write preproc_temp, current= , vtracer to svg_temp (binary, speckle=min, corner, len=detail*10), last=svg; orig_pil=Image.open; vec_pil = render to vec_preview or fallback processed pil; status; 3x _pil_to + _replace_texture; print; 3x _refresh; except status error
  load_image: show_item load_dialog
  _extract_path(app_data, require_exists): robust candidates from file_path_name, selections (k/v), current+fname; prefer exists if require (for load); fallback any str. (fixed * ext bug)
  _on_load: print, hide_dialog, path=extract(require=True), set input, status, update_preview()
  save_vector: if no preproc status; show save_dialog
  _on_save: hide, path=extract(False), ensure .svg, _do_save
  _do_save: config, vtracer to save_path, last=; eps try inkscape, status msg with +EPS or note
  open_svg: target=last or svg_temp; if none status; subprocess open or startfile; status
  _cleanup_temps: unlink 3 temps
  _on_exit: cleanup
  run: start_dearpygui, destroy_context
if __name__: VectorGUI().run()

## DPG SPECIFICS / LAYOUT (current)
- No Tk anywhere.
- Textures: raw Float_rgba, 400x400, placeholder checker then real via pil_to.
- Update strategy (after many fails): replace_texture (del+add_raw), refresh (del image + add_image to child). Frame callback delay for initial.
- Previews: 3 bordered child_windows (h layout) as "boxes"; images added post-viewport in scheduled frame 2 + on updates.
- Sliders live callback (drag may lag noted in label).
- File ext handling in dialogs.
- Debug prints everywhere for texture exist, refresh, update calls, etc.
- Child windows for visibility of areas even on failure.


## EXACT PARAMS / LOGIC TO PRESERVE
- Sliders exact as above + callbacks.
- vtracer call exact (incl length*10).
- preprocess exact (incl PIL gray, fixed 5x5 blur, 11 block thresh, 1 iter open).
- _pil_to: always 400 square canvas dark25, thumbnail LANCZOS center, float32/255 ravel list.
- _render: exact inkscape then ql (poll 12*0.04, expected .png, glob fallback "stencil|vector|thumb").
- Save: vtracer + inkscape eps, status msg.
- Paths: 3 /tmp fixed names.
- DPG: viewport size, tags (sliders, status, dialogs, cols, imgs, texs, tex_reg), child bordered, frame cb, prints (keep for debug?).
- No Tk, no argparse leftover, etc.

## FILES / DEPS
- specification.md (this), stencil_creator_app.py (current, rename to _old after), requirements (dearpygui + numpy opencv pillow vtracer).

END SPEC. Include full logic from reads above. Use to recreate/fix without losing details.