#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AAA1 Media Studio: High-Performance Diffusers Dashboard & Sequential Flipbook Studio

Features:
- Full Diffusers Generation Dashboard (txt2img, img2img, inpaint)
- Automatic aspect ratio routing for all generated images:
    * Portrait  (height > width)  -> static/novel_images0/
    * Landscape (width > height) -> static/novel_images1/
    * Square    (width == height) -> static/novel_images2/
- High-Performance Sequential Flipbook Player & Monitor (/flipbook)
- In-browser MP4 Movie Maker via make_video.py (/api/export_video)
- Generated Video Gallery (/videos)
"""

import os
import sys
import glob
import json
import re
import time
import shutil
import inspect
import signal
import threading
from datetime import datetime
from io import BytesIO

from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    send_from_directory,
    Response,
    redirect,
    url_for,
)
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont

from engine import GenerationEngine
from make_video import create_video_from_directory

# Base paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
DEMOS_DIR = os.path.join(STATIC_DIR, "demos")
HISTORY_FILE = os.path.join(BASE_DIR, "history.json")
LOG_FILE_PATH = os.path.join(STATIC_DIR, "app_log.txt")

# Novel Image Directories sorted by aspect ratio
NOVEL_PORTRAIT_DIR = os.path.join(STATIC_DIR, "novel_images0")   # Portrait (height > width)
NOVEL_LANDSCAPE_DIR = os.path.join(STATIC_DIR, "novel_images1")  # Landscape (width > height)
NOVEL_SQUARE_DIR = os.path.join(STATIC_DIR, "novel_images2")     # Square (width == height)

# Ensure required directories exist
for d in [OUTPUTS_DIR, UPLOADS_DIR, DEMOS_DIR, NOVEL_PORTRAIT_DIR, NOVEL_LANDSCAPE_DIR, NOVEL_SQUARE_DIR]:
    os.makedirs(d, exist_ok=True)

# Ensure log file exists
if not os.path.exists(LOG_FILE_PATH):
    try:
        with open(LOG_FILE_PATH, "w", encoding="utf-8") as f:
            f.write("")
    except Exception:
        pass

# Initialize Flask & Diffusers Engine
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["TEMPLATES_AUTO_RELOAD"] = True
engine = GenerationEngine()

generation_lock = threading.Lock()
current_task = {
    "is_busy": False,
    "last_result": None
}


# --------------------------------------------------
# LOGGING & UTILITY FUNCTIONS
# --------------------------------------------------

def logit(*args):
    """
    Lightweight file-based logging utility for debugging and diagnostics.
    """
    try:
        timestr = datetime.now().strftime("%A_%b-%d-%Y_%H-%M-%S")
        frame = inspect.stack()[1]
        filename = os.path.basename(frame.filename)
        lineno = frame.lineno

        parts = []
        for arg in args:
            if isinstance(arg, (list, tuple, set)):
                parts.append(" ".join(map(str, arg)))
            else:
                parts.append(str(arg))

        message_str = " ".join(parts)
        log_message = f"{timestr} - File: {filename}, Line: {lineno}: {message_str}\n"

        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(log_message)
    except Exception as e:
        print(f"[LOGIT ERROR] {e}")


def natural_sort_key(s):
    """Sort strings containing numbers naturally (e.g. 1, 2, 10 instead of 1, 10, 2)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", str(s))]


def load_history():
    """Loads generation history from JSON."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history_item(meta):
    """Saves a single generation item into history.json (keeps newest 100)."""
    history = load_history()
    history.insert(0, meta)
    history = history[:100]
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logit(f"Error saving history item: {e}")


def route_image_by_aspect_ratio(meta):
    """
    Automatically routes a generated image into the correct novel_images directory:
    - Portrait  (height > width)  -> static/novel_images0/
    - Landscape (width > height) -> static/novel_images1/
    - Square    (width == height) -> static/novel_images2/
    Also updates a local generations.json in that folder for metadata inspection.
    """
    try:
        filepath = meta.get("filepath")
        if not filepath or not os.path.exists(filepath):
            return None

        # Inspect dimensions from meta or the image file directly
        width = meta.get("width")
        height = meta.get("height")
        if not width or not height:
            with Image.open(filepath) as im:
                width, height = im.size

        width = int(width)
        height = int(height)

        if height > width:
            dest_dir = NOVEL_PORTRAIT_DIR
            dir_name = "novel_images0"
            aspect_type = "Portrait"
        elif width > height:
            dest_dir = NOVEL_LANDSCAPE_DIR
            dir_name = "novel_images1"
            aspect_type = "Landscape"
        else:
            dest_dir = NOVEL_SQUARE_DIR
            dir_name = "novel_images2"
            aspect_type = "Square"

        filename = os.path.basename(filepath)
        dest_filepath = os.path.join(dest_dir, filename)

        # Copy to the corresponding novel_images directory
        shutil.copy2(filepath, dest_filepath)

        # Record metadata into generations.json in the target directory
        gen_json_path = os.path.join(dest_dir, "generations.json")
        try:
            with open(gen_json_path, "w", encoding="utf-8") as gf:
                json.dump({
                    "prompt": meta.get("prompt", ""),
                    "seed": meta.get("seed", "---"),
                    "model": meta.get("model", "---"),
                    "last_file": filename,
                    "updated_at": datetime.now().isoformat(),
                    "width": width,
                    "height": height,
                    "aspect": aspect_type,
                }, gf, indent=2)
        except Exception:
            pass

        meta["novel_dir"] = dir_name
        meta["novel_url"] = f"/static/{dir_name}/{filename}"
        meta["aspect_type"] = aspect_type

        log_msg = f"[✓] Routed {filename} ({width}x{height} {aspect_type}) -> static/{dir_name}/"
        print(log_msg)
        logit(log_msg)

        return dest_filepath
    except Exception as e:
        err_msg = f"Error routing image by aspect ratio: {e}"
        print(f"[!] {err_msg}")
        logit(err_msg)
        return None


def sync_existing_outputs():
    """
    On startup, routes any existing images in outputs/ that are not yet sorted
    into the novel_images directories.
    """
    valid_exts = (".png", ".jpg", ".jpeg", ".webp")
    if not os.path.exists(OUTPUTS_DIR):
        return

    routed_count = 0
    try:
        for fname in os.listdir(OUTPUTS_DIR):
            if fname.lower().endswith(valid_exts):
                fpath = os.path.join(OUTPUTS_DIR, fname)
                # Check if file is already present in any novel directory
                already_in_novel = any(
                    os.path.exists(os.path.join(nd, fname))
                    for nd in [NOVEL_PORTRAIT_DIR, NOVEL_LANDSCAPE_DIR, NOVEL_SQUARE_DIR]
                )
                if not already_in_novel:
                    try:
                        with Image.open(fpath) as im:
                            w, h = im.size
                        route_image_by_aspect_ratio({
                            "filepath": fpath,
                            "filename": fname,
                            "width": w,
                            "height": h,
                            "prompt": "",
                            "model": "diffusers",
                            "seed": "---"
                        })
                        routed_count += 1
                    except Exception:
                        pass
        if routed_count > 0:
            print(f"[*] Initialized & sorted {routed_count} existing image(s) from outputs/ into novel_images folders.")
    except Exception as e:
        logit(f"Startup sync error: {e}")


# --------------------------------------------------
# NOVEL IMAGES FLIPBOOK HELPERS
# --------------------------------------------------

def get_novel_image_directories():
    """
    Discovers all directories under static/ that match novel_images*.
    Returns a list of directory objects sorted naturally.
    """
    pattern = os.path.join(STATIC_DIR, "novel_images*")
    dir_paths = glob.glob(pattern)
    directories = []
    valid_exts = (".png", ".jpg", ".jpeg", ".webp")

    # Friendly aspect ratio badges
    aspect_map = {
        "novel_images0": "Portrait (Height > Width)",
        "novel_images1": "Landscape (Width > Height)",
        "novel_images2": "Square (Equal)",
    }

    type_labels = {
        "cfg": "CFG Sweep",
        "slerp": "SLERP Morph",
        "models": "Models Comparison",
        "generations": "Generations",
        "generic": "Image Sequence",
    }

    for dp in dir_paths:
        if not os.path.isdir(dp):
            continue
        dname = os.path.basename(dp)
        try:
            img_files = [
                f for f in os.listdir(dp)
                if f.lower().endswith(valid_exts) and not f.startswith(".")
            ]
            count = len(img_files)
        except Exception:
            count = 0

        # Detect metadata type
        meta_type = "generic"
        if os.path.exists(os.path.join(dp, "cfg.json")):
            meta_type = "cfg"
        elif os.path.exists(os.path.join(dp, "slerp.json")):
            meta_type = "slerp"
        elif os.path.exists(os.path.join(dp, "models.json")):
            meta_type = "models"
        elif os.path.exists(os.path.join(dp, "generations.json")):
            meta_type = "generations"

        # Display aspect ratio label if one of the standard folders
        if dname in aspect_map:
            type_label = aspect_map[dname]
        else:
            type_label = type_labels.get(meta_type, "Images")

        try:
            mtime = os.path.getmtime(dp)
        except Exception:
            mtime = 0

        directories.append({
            "name": dname,
            "count": count,
            "meta_type": meta_type,
            "type_label": type_label,
            "mtime": mtime,
        })

    directories.sort(key=lambda d: natural_sort_key(d["name"]))
    return directories


def get_directory_data(dirname, sort_by="name", filter_prefix=""):
    """
    Retrieves all images and metadata for a specified novel_images directory.
    """
    if not re.match(r"^novel_images[\w-]*$", dirname):
        return {"success": False, "error": f"Invalid directory name: {dirname}"}

    dir_path = os.path.join(STATIC_DIR, dirname)
    if not os.path.isdir(dir_path):
        return {"success": False, "error": f"Directory '{dirname}' does not exist"}

    valid_exts = (".png", ".jpg", ".jpeg", ".webp")
    try:
        raw_files = [
            f for f in os.listdir(dir_path)
            if f.lower().endswith(valid_exts) and not f.startswith(".")
        ]
    except Exception as e:
        return {"success": False, "error": str(e)}

    if filter_prefix:
        fp_low = filter_prefix.lower()
        raw_files = [f for f in raw_files if fp_low in f.lower()]

    if sort_by == "time":
        raw_files.sort(
            key=lambda f: os.path.getmtime(os.path.join(dir_path, f)),
            reverse=True,
        )
    else:
        raw_files.sort(key=natural_sort_key)

    # Parse metadata if present
    meta_info = {
        "type": "generic",
        "seed": "---",
        "model": "---",
        "prompt": "",
        "prompt2": "",
        "total_runs": len(raw_files),
        "last_completed_val": "---",
    }
    file_meta_map = {}

    for candidate in ["cfg.json", "slerp.json", "models.json", "generations.json"]:
        json_path = os.path.join(dir_path, candidate)
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as jf:
                    jdata = json.load(jf)

                if candidate == "cfg.json" and "runs" in jdata:
                    meta_info["type"] = "cfg"
                    meta_info["seed"] = jdata.get("seed", "---")
                    meta_info["model"] = jdata.get("model", "---")
                    meta_info["prompt"] = jdata.get("prompt", "")
                    meta_info["total_runs"] = len(jdata.get("runs", []))
                    last_cfg = jdata.get("last_completed_cfg")
                    if last_cfg is not None:
                        meta_info["last_completed_val"] = f"{float(last_cfg):.3f}"
                    for r in jdata.get("runs", []):
                        fn = r.get("file")
                        if fn:
                            cfg_val = (
                                f"{float(r['guidance_scale']):.3f}"
                                if r.get("guidance_scale") is not None
                                else "---"
                            )
                            file_meta_map[fn] = {
                                "label": f"CFG: {cfg_val}",
                                "meta": f"CFG {cfg_val} • Step {r.get('steps', '---')}",
                                "cfg": r.get("guidance_scale"),
                                "step": r.get("steps"),
                                "seed": r.get("seed"),
                                "model": r.get("model"),
                                "prompt": r.get("prompt"),
                            }
                    break

                elif candidate == "slerp.json":
                    meta_info["type"] = "slerp"
                    meta_info["seed"] = jdata.get("seed", "---")
                    meta_info["model"] = jdata.get("model", "---")
                    meta_info["prompt"] = jdata.get("prompt1", "")
                    meta_info["prompt2"] = jdata.get("prompt2", "")
                    meta_info["total_runs"] = jdata.get(
                        "total_frames", len(jdata.get("frames", []))
                    )
                    last_t = jdata.get("last_completed_t")
                    if last_t is not None:
                        meta_info["last_completed_val"] = f"{float(last_t):.3f}"
                    for fr in jdata.get("frames", []):
                        fn = fr.get("file")
                        if fn:
                            t_val = (
                                f"{float(fr.get('t', 0.0)):.3f}"
                                if fr.get("t") is not None
                                else "0.000"
                            )
                            file_meta_map[fn] = {
                                "label": f"t = {t_val}",
                                "meta": f"t = {t_val} • Frame #{fr.get('index', 0)}",
                                "t": fr.get("t"),
                                "index": fr.get("index"),
                            }
                    break

                elif candidate == "models.json" and "runs" in jdata:
                    meta_info["type"] = "models"
                    meta_info["seed"] = jdata.get("seed", "---")
                    meta_info["model"] = jdata.get("model", "---")
                    meta_info["prompt"] = jdata.get("prompt", "")
                    meta_info["total_runs"] = len(jdata.get("runs", []))
                    for r in jdata.get("runs", []):
                        fn = r.get("file")
                        if fn:
                            m_clean = (r.get("model") or "").split(".")[0]
                            cfg_val = (
                                f"{float(r['guidance_scale']):.1f}"
                                if r.get("guidance_scale") is not None
                                else ""
                            )
                            file_meta_map[fn] = {
                                "label": m_clean or fn,
                                "meta": f"{m_clean} (CFG {cfg_val})" if cfg_val else m_clean,
                                "model": r.get("model"),
                                "cfg": r.get("guidance_scale"),
                                "step": r.get("steps"),
                                "seed": r.get("seed"),
                            }
                    break

                elif candidate == "generations.json":
                    meta_info["type"] = "generations"
                    if isinstance(jdata, dict):
                        meta_info["prompt"] = jdata.get("prompt", "")
                        meta_info["seed"] = jdata.get("seed", "---")
                        meta_info["model"] = jdata.get("model", "---")
                    break

            except Exception as e:
                logit(f"Error parsing metadata {json_path}: {e}")

    images = []
    for idx, f in enumerate(raw_files):
        fpath = os.path.join(dir_path, f)
        fm = file_meta_map.get(f, {})
        label = fm.get("label", f)
        meta = fm.get("meta", f)
        try:
            mtime = os.path.getmtime(fpath)
        except Exception:
            mtime = 0

        images.append({
            "index": idx,
            "filename": f,
            "path": f"/static/{dirname}/{f}",
            "label": label,
            "meta": meta,
            "mtime": mtime,
            "extra": fm,
        })

    return {
        "success": True,
        "directory": dirname,
        "count": len(images),
        "metadata": meta_info,
        "images": images,
    }


# --------------------------------------------------
# CORE APPLICATION ROUTES
# --------------------------------------------------

@app.route("/")
def index():
    """Main Diffusers Generation Dashboard."""
    return render_template("index.html")


@app.route("/flipbook")
def flipbook():
    """Sequential Flipbook Player & Movie Maker."""
    directories = get_novel_image_directories()
    selected_dir = request.args.get("dir", "").strip()

    valid_dir_names = [d["name"] for d in directories]
    if not selected_dir or selected_dir not in valid_dir_names:
        selected_dir = valid_dir_names[0] if valid_dir_names else "novel_images0"

    return render_template(
        "live_slerp.html",
        directories=directories,
        selected_dir=selected_dir,
    )


@app.route("/videos")
def videos_view():
    """Video gallery view of all generated MP4 movies in static/demos/."""
    videos = []
    if os.path.exists(DEMOS_DIR):
        valid_vid_exts = (".mp4", ".webm", ".mkv")
        try:
            files = [
                f for f in os.listdir(DEMOS_DIR)
                if f.lower().endswith(valid_vid_exts) and os.path.isfile(os.path.join(DEMOS_DIR, f))
            ]
            files.sort(key=lambda f: os.path.getmtime(os.path.join(DEMOS_DIR, f)), reverse=True)
            for f in files:
                fpath = os.path.join(DEMOS_DIR, f)
                videos.append({
                    "filename": f,
                    "url": f"/static/demos/{f}",
                    "size_mb": round(os.path.getsize(fpath) / (1024 * 1024), 2),
                    "mtime": datetime.fromtimestamp(os.path.getmtime(fpath)).strftime("%Y-%m-%d %H:%M:%S")
                })
        except Exception as e:
            logit(f"Error reading videos directory: {e}")

    return render_template("videos.html", videos=videos)


# --------------------------------------------------
# DIFFUSERS GENERATION API ROUTES
# --------------------------------------------------

@app.route("/api/models")
def get_models():
    """Scans and returns available checkpoints, loras, and vaes."""
    checkpoints = engine.scan_models()
    loras = engine.scan_loras()
    vaes = engine.scan_vaes()
    return jsonify({
        "checkpoints": list(checkpoints.values()),
        "loras": list(loras.values()),
        "vaes": list(vaes.values()),
        "active_model": os.path.basename(engine.current_model_path) if engine.current_model_path else None
    })


@app.route("/api/progress")
def get_progress():
    """Returns real-time progress of current diffusion generation."""
    state = engine.progress_state.copy()
    state["is_busy"] = current_task["is_busy"]
    if current_task["last_result"]:
        state["last_result"] = current_task["last_result"]
    return jsonify(state)


@app.route("/api/interrupt", methods=["POST"])
def interrupt():
    """Signals current diffusion pipeline to cancel."""
    engine.interrupt()
    return jsonify({"success": True, "message": "Interrupt signal sent"})


@app.route("/api/generate", methods=["POST"])
def generate():
    """txt2img endpoint: renders image and automatically routes by aspect ratio."""
    if current_task["is_busy"]:
        return jsonify({"success": False, "error": "Engine is currently busy generating an image"}), 429

    params = request.get_json() or {}
    current_task["is_busy"] = True
    current_task["last_result"] = None

    def worker():
        try:
            result = engine.generate_txt2img(params, OUTPUTS_DIR)
            if result.get("success") and "image" in result:
                save_history_item(result["image"])
                route_image_by_aspect_ratio(result["image"])
            current_task["last_result"] = result
        except Exception as e:
            current_task["last_result"] = {"success": False, "error": str(e)}
        finally:
            current_task["is_busy"] = False

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return jsonify({"success": True, "message": "Generation started"})


@app.route("/api/generate_img2img", methods=["POST"])
def generate_img2img():
    """img2img endpoint: renders image and automatically routes by aspect ratio."""
    if current_task["is_busy"]:
        return jsonify({"success": False, "error": "Engine is currently busy"}), 429

    if "init_image" not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"}), 400

    file = request.files["init_image"]
    filename = secure_filename(file.filename) or "input.png"
    filepath = os.path.join(UPLOADS_DIR, f"upload_{int(datetime.now().timestamp())}_{filename}")
    file.save(filepath)

    params = {
        "prompt": request.form.get("prompt", ""),
        "negative_prompt": request.form.get("negative_prompt", ""),
        "steps": int(request.form.get("steps", 5)),
        "cfg_scale": float(request.form.get("cfg_scale", 1.5)),
        "strength": float(request.form.get("strength", 0.65)),
        "width": int(request.form.get("width", 512)),
        "height": int(request.form.get("height", 512)),
        "seed": int(request.form.get("seed", -1)),
        "sampler": request.form.get("sampler", "LCM"),
        "model_path": request.form.get("model_path"),
        "vae_path": request.form.get("vae_path")
    }

    current_task["is_busy"] = True
    current_task["last_result"] = None

    def worker():
        try:
            result = engine.generate_img2img(params, filepath, OUTPUTS_DIR)
            if result.get("success") and "image" in result:
                save_history_item(result["image"])
                route_image_by_aspect_ratio(result["image"])
            current_task["last_result"] = result
        except Exception as e:
            current_task["last_result"] = {"success": False, "error": str(e)}
        finally:
            current_task["is_busy"] = False

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return jsonify({"success": True, "message": "Img2Img started"})


@app.route("/api/generate_inpaint", methods=["POST"])
def generate_inpaint():
    """inpaint endpoint: renders image and automatically routes by aspect ratio."""
    if current_task["is_busy"]:
        return jsonify({"success": False, "error": "Engine is currently busy"}), 429

    init_filepath = None
    if "init_image" in request.files:
        file = request.files["init_image"]
        filename = secure_filename(file.filename) or "init_inpaint.png"
        init_filepath = os.path.join(UPLOADS_DIR, f"inpaint_{int(datetime.now().timestamp())}_{filename}")
        file.save(init_filepath)
    elif request.form.get("init_image_url"):
        url = request.form.get("init_image_url")
        base_fn = os.path.basename(url)
        possible_path = os.path.join(OUTPUTS_DIR, base_fn)
        if os.path.exists(possible_path):
            init_filepath = possible_path
    elif request.form.get("init_image_data"):
        data = request.form.get("init_image_data")
        if "," in data:
            data = data.split(",", 1)[1]
        import base64
        img_bytes = base64.b64decode(data)
        init_filepath = os.path.join(UPLOADS_DIR, f"inpaint_init_{int(datetime.now().timestamp())}.png")
        with open(init_filepath, "wb") as f:
            f.write(img_bytes)

    mask_filepath = None
    if "mask_image" in request.files:
        mfile = request.files["mask_image"]
        mfilename = secure_filename(mfile.filename) or "mask.png"
        mask_filepath = os.path.join(UPLOADS_DIR, f"mask_{int(datetime.now().timestamp())}_{mfilename}")
        mfile.save(mask_filepath)
    elif request.form.get("mask_data"):
        mdata = request.form.get("mask_data")
        if "," in mdata:
            mdata = mdata.split(",", 1)[1]
        import base64
        mbytes = base64.b64decode(mdata)
        mask_filepath = os.path.join(UPLOADS_DIR, f"mask_{int(datetime.now().timestamp())}.png")
        with open(mask_filepath, "wb") as mf:
            mf.write(mbytes)

    if not init_filepath or not os.path.exists(init_filepath):
        return jsonify({"success": False, "error": "No source image provided"}), 400

    if not mask_filepath or not os.path.exists(mask_filepath):
        return jsonify({"success": False, "error": "No mask painted. Please draw a mask over the area you wish to inpaint."}), 400

    params = {
        "prompt": request.form.get("prompt", ""),
        "negative_prompt": request.form.get("negative_prompt", ""),
        "steps": int(request.form.get("steps", 5)),
        "cfg_scale": float(request.form.get("cfg_scale", 1.5)),
        "strength": float(request.form.get("strength", 1.0)),
        "mask_blur": int(request.form.get("mask_blur", 4)),
        "invert_mask": request.form.get("invert_mask", "false").lower() == "true",
        "width": int(request.form.get("width", 512)),
        "height": int(request.form.get("height", 512)),
        "seed": int(request.form.get("seed", -1)),
        "sampler": request.form.get("sampler", "LCM"),
        "model_path": request.form.get("model_path"),
        "vae_path": request.form.get("vae_path")
    }

    current_task["is_busy"] = True
    current_task["last_result"] = None

    def worker():
        try:
            result = engine.generate_inpaint(params, init_filepath, mask_filepath, OUTPUTS_DIR)
            if result.get("success") and "image" in result:
                save_history_item(result["image"])
                route_image_by_aspect_ratio(result["image"])
            current_task["last_result"] = result
        except Exception as e:
            current_task["last_result"] = {"success": False, "error": str(e)}
        finally:
            current_task["is_busy"] = False

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return jsonify({"success": True, "message": "Inpaint started"})


@app.route("/api/gallery")
def get_gallery():
    """Returns generation history."""
    history = load_history()
    return jsonify(history)


@app.route("/api/delete_image", methods=["POST"])
def delete_image():
    """Deletes an image from disk (outputs and novel_images directories) and history."""
    data = request.get_json() or {}
    filename = data.get("filename")
    if not filename:
        return jsonify({"success": False, "error": "No filename provided"}), 400

    safe_filename = os.path.basename(filename)

    # 1. Delete from outputs/
    filepath = os.path.join(OUTPUTS_DIR, safe_filename)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except Exception as e:
            print(f"[!] Error deleting {filepath}: {e}")

    # 2. Delete from novel_images directories if present
    for nd in [NOVEL_PORTRAIT_DIR, NOVEL_LANDSCAPE_DIR, NOVEL_SQUARE_DIR]:
        novel_filepath = os.path.join(nd, safe_filename)
        if os.path.exists(novel_filepath):
            try:
                os.remove(novel_filepath)
            except Exception as e:
                print(f"[!] Error deleting {novel_filepath}: {e}")

    # 3. Remove from history.json
    history = load_history()
    updated_history = [item for item in history if item.get("filename") != safe_filename]
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(updated_history, f, indent=2)
    except Exception as e:
        print(f"[!] Error updating history.json: {e}")

    return jsonify({"success": True, "message": f"Deleted {safe_filename}", "remaining": len(updated_history)})


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    """Serves generated images directly from outputs/."""
    return send_from_directory(OUTPUTS_DIR, filename)


# --------------------------------------------------
# FLIPBOOK & MOVIE MAKER API ROUTES
# --------------------------------------------------

@app.route("/api/directories", methods=["GET"])
def api_directories():
    """API endpoint to list all novel_images* directories."""
    dirs = get_novel_image_directories()
    return jsonify({"success": True, "directories": dirs})


@app.route("/api/images", methods=["GET"])
def api_images():
    """API endpoint to get sequential images and metadata for a directory."""
    dirname = request.args.get("dir", "").strip()
    sort_by = request.args.get("sort", "name")
    filter_prefix = request.args.get("filter", "").strip()

    if not dirname:
        dirs = get_novel_image_directories()
        if dirs:
            dirname = dirs[0]["name"]
        else:
            dirname = "novel_images0"

    data = get_directory_data(dirname, sort_by=sort_by, filter_prefix=filter_prefix)
    return jsonify(data)


@app.route("/api/export_video", methods=["POST"])
def api_export_video():
    """
    API endpoint to generate an MP4 video from the currently viewed novel_images directory.
    Uses make_video.py create_video_from_directory.
    """
    req = request.get_json(force=True, silent=True) or {}
    dirname = req.get("dir", "").strip()
    raw_fps = req.get("fps", 10)
    try:
        fps = max(1, min(60, int(float(raw_fps))))
    except (ValueError, TypeError):
        fps = 10

    pingpong = bool(req.get("pingpong", False))
    raw_loops = req.get("loops", 1)
    try:
        loops = max(1, min(10, int(raw_loops)))
    except (ValueError, TypeError):
        loops = 1

    sort_by = req.get("sort", "name")
    filter_keyword = req.get("filter", "").strip()

    if not dirname:
        dirs = get_novel_image_directories()
        if dirs:
            dirname = dirs[0]["name"]
        else:
            return jsonify({"success": False, "error": "No novel_images directories found"})

    try:
        result = create_video_from_directory(
            dir_name_or_path=dirname,
            fps=fps,
            pingpong=pingpong,
            loops=loops,
            filter_keyword=filter_keyword,
            sort_by=sort_by
        )
        rel_video_path = os.path.relpath(result["output_path"], STATIC_DIR)
        return jsonify({
            "success": True,
            "filename": os.path.basename(result["output_path"]),
            "video_url": f"/static/{rel_video_path}",
            "file_size_mb": round(result["filesize_mb"], 2),
            "fps": result["fps"],
            "frames": result["frames"],
            "duration": round(result["duration"], 2),
            "elapsed": round(result["elapsed"], 2),
        })
    except Exception as e:
        logit(f"Video export error for {dirname}: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route("/favicon.ico")
def favicon():
    """Dynamic favicon generator."""
    size = (32, 32)
    favicon_img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(favicon_img)
    draw.ellipse([(0, 0), size], fill=(255, 255, 0))

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    if not os.path.exists(font_path):
        font_path = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"

    try:
        font = ImageFont.truetype(font_path, 24)
        text = "F"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_pos = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)
        draw.text(text_pos, text, font=font, fill=(255, 0, 0))
    except Exception:
        pass

    buf = BytesIO()
    favicon_img.save(buf, format="ICO")
    buf.seek(0)
    return Response(buf.getvalue(), content_type="image/x-icon")


# --------------------------------------------------
# SERVER STARTUP
# --------------------------------------------------

if __name__ == "__main__":
    def exit_gracefully(signum, frame):
        print("\n[*] Shutting down Media Studio server gracefully...")
        os._exit(0)

    signal.signal(signal.SIGINT, exit_gracefully)
    signal.signal(signal.SIGTERM, exit_gracefully)

    # Sort any existing images in outputs/ into novel_images folders
    sync_existing_outputs()

    print("\n" + "="*65)
    print("  🚀 Media Studio: Diffusers Dashboard, Flipbook & Video Studio")
    print("  🔗 Generator UI:     http://127.0.0.1:7865")
    print("  🎬 Flipbook Player:  http://127.0.0.1:7865/flipbook")
    print("  🎞️ Videos Gallery:   http://127.0.0.1:7865/videos")
    print("  📁 Aspect Routing:")
    print("     - Portrait  -> static/novel_images0/")
    print("     - Landscape -> static/novel_images1/")
    print("     - Square    -> static/novel_images2/")
    print("="*65 + "\n")

    app.run(host="0.0.0.0", port=7865, debug=False, threaded=True)
