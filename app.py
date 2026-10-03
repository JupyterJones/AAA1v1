#!/usr/bin/env python3
import os
import json
import threading
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from engine import GenerationEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
HISTORY_FILE = os.path.join(BASE_DIR, "history.json")

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
engine = GenerationEngine()

generation_lock = threading.Lock()
current_task = {
    "is_busy": False,
    "last_result": None
}

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history_item(meta):
    history = load_history()
    history.insert(0, meta)
    # keep last 100
    history = history[:100]
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/models")
def get_models():
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
    state = engine.progress_state.copy()
    state["is_busy"] = current_task["is_busy"]
    if current_task["last_result"]:
        state["last_result"] = current_task["last_result"]
    return jsonify(state)

@app.route("/api/interrupt", methods=["POST"])
def interrupt():
    engine.interrupt()
    return jsonify({"success": True, "message": "Interrupt signal sent"})

@app.route("/api/generate", methods=["POST"])
def generate():
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
    history = load_history()
    return jsonify(history)

@app.route("/api/delete_image", methods=["POST"])
def delete_image():
    data = request.get_json() or {}
    filename = data.get("filename")
    if not filename:
        return jsonify({"success": False, "error": "No filename provided"}), 400

    safe_filename = os.path.basename(filename)
    filepath = os.path.join(OUTPUTS_DIR, safe_filename)

    # Delete physical file from disk
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except Exception as e:
            print(f"[!] Error deleting {filepath}: {e}")

    # Remove from history.json
    history = load_history()
    updated_history = [item for item in history if item.get("filename") != safe_filename]
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(updated_history, f, indent=2)
    except Exception as e:
        print(f"[!] Error updating history.json: {e}")

    return jsonify({"success": True, "message": f"Deleted {safe_filename}", "remaining": len(updated_history)})

@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUTS_DIR, filename)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  🚀 AAA1v1 WebUI: High-Performance Diffusers Dashboard")
    print("  🔗 http://127.0.0.1:7865")
    print("="*60 + "\n")
    app.run(host="0.0.0.0", port=7865, debug=False, threaded=True)
