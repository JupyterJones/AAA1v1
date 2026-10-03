# ⚡ AAA1v1 WebUI: Fast Diffusers Dashboard

A lightweight, high-performance web interface for Stable Diffusion built on Hugging Face **`diffusers`** and **Flask**. Designed as a responsive, resource-efficient alternative to heavyweight interfaces, featuring the familiar Automatic1111 layout with rapid CPU/LCM acceleration, img2img, interactive inpainting, and zero-crop aspect ratio preservation.

---

## ✨ Features

- **⚡ Fast Inference Engine**: Powered by Hugging Face `diffusers` with attention slicing, VAE tiling, and optional LCM (Latent Consistency Model) scheduling for 4–8 step generations.
- **🎨 txt2img, img2img & Inpainting**:
  - Full text-to-image synthesis.
  - Image-to-image with customizable denoising strength.
  - Interactive HTML5 inpainting canvas with brush sizing, clear mask, undo, and mask inversion.
- **🖼️ Full Aspect Ratio Preservation**: All images (main canvas, thumbnail reels, and gallery cards) display with original aspect ratios (`512x512`, `512x768`, `768x512`, etc.) with zero cropping.
- **📋 Parameter Replication (`<Copy Info>`)**: One-click copying of prompt, negative prompt, steps, CFG, sampler, dimensions, seed, and checkpoint directly into the UI controls.
- **🗑️ Disk & History Management**: Delete unwanted images directly from the gallery or preview bar, automatically wiping the file from disk and purging its entry from `history.json`.
- **📂 Flexible Model Scanning**: Automatically detects checkpoints, LoRAs, and VAEs from local `models/` folders or external drives (e.g., USB SSDs).

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/AAA1v1.git
cd AAA1v1
```

### 2. Create a Virtual Environment (`venv`)
Create and activate an isolated Python virtual environment (Python 3.10, 3.11, or 3.12 recommended):

```bash
# Create the virtual environment named 'venv'
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

> **Note**: To deactivate later, simply run `deactivate`.

---

### 3. Install Dependencies from `requirements.txt`

#### For CPU (Recommended for low-resource or CPU-only setups):
```bash
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

#### For NVIDIA GPU (CUDA acceleration):
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📁 Model Directory Structure

Place your models, LoRAs, and VAEs in the `models/` directory (or edit `engine.py` to point to external hard drives):

```
AAA1v1/
├── models/
│   ├── checkpoints/      <-- Place .safetensors checkpoints here
│   ├── loras/            <-- Place .safetensors LoRAs here
│   └── vae/              <-- Place custom VAEs here
├── outputs/              <-- Generated images are saved here
├── uploads/              <-- Temporary img2img & inpaint uploads
├── app.py                <-- Flask web server
├── engine.py             <-- Diffusers generation backend
├── start.sh              <-- Startup script
└── requirements.txt      <-- Python dependencies
```

*(You can also configure external SSD or custom drive paths inside `engine.py` in the `MODEL_PATHS`, `LORA_PATHS`, and `VAE_PATHS` lists).*

---

## 🏃 Running AAA1v1

### Option 1: Using the Startup Script (Recommended)
Make sure `start.sh` is executable, then run:

```bash
chmod +x start.sh
./start.sh
```
`start.sh` automatically detects your `venv`, applies CPU thread tuning, and launches the server.

### Option 2: Running Directly with Python
```bash
source venv/bin/activate
python3 app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:7865
```

---

## 🖥️ Desktop Shortcut (Linux)

To create a one-click desktop launcher:

1. Copy `AAA1v1.desktop` to your desktop:
   ```bash
   cp AAA1v1.desktop ~/Desktop/
   chmod +x ~/Desktop/AAA1v1.desktop
   ```
2. Right-click the desktop icon and choose **"Allow Launching"** (if prompted by your desktop environment).

---

## ⚙️ Recommended Settings for Fast CPU Generation

- **Model**: Use any SD 1.5 checkpoint with **LCM** in the name (e.g. `dreamshaper_8LCM`, `comiccraftLCM`).
- **Sampler**: `LCM`
- **Sampling Steps**: `4` to `8`
- **CFG Scale**: `1.2` to `2.0`
- **Resolution**: `512 × 512` (Square) or `512 × 768` (Portrait)

---

## 📄 License
Open source under the [MIT License](LICENSE).
