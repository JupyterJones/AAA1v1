# 📖 AAA1v1 User Manual & Operational Instructions

Welcome to **AAA1v1**! This guide walks you through every feature, control, and workflow to help you get the best image generation experience.

---

## 📑 Table of Contents
1. [Interface Overview](#1-interface-overview)
2. [Text-to-Image (txt2img)](#2-text-to-image-txt2img)
3. [Image-to-Image (img2img)](#3-image-to-image-img2img)
4. [Inpainting (Inpaint)](#4-inpainting-inpaint)
5. [Gallery & Parameter Replication (<Copy Info>)](#5-gallery--parameter-replication-copy-info)
6. [Managing Models & LoRAs](#6-managing-models--loras)
7. [CPU Performance & LCM Acceleration](#7-cpu-performance--lcm-acceleration)

---

## 1. Interface Overview

### Top Bar
- **Checkpoint Dropdown**: Selects the active Stable Diffusion model. Checkpoints with `[LCM]` in the name are optimized for ultra-fast generation (4–8 steps).
- **🔄 Refresh Button**: Rescans your model directories and updates the dropdown without needing to restart the server.
- **VAE Dropdown**: Choose between the model's baked VAE or a custom external VAE (e.g. TAESD for fast previews).
- **System Status**: Displays CPU thread allocation and system readiness.

### Navigation Tabs
- **txt2img**: Create new artwork from text prompts.
- **img2img**: Transform or stylize an existing image.
- **Inpaint**: Paint a mask over parts of an image and regenerate only that section.
- **Gallery / History**: Browse, inspect, replicate parameters from, or delete previous generations.

---

## 2. Text-to-Image (txt2img)

### Prompts
- **Prompt**: Describe your scene in detail (subjects, style, lighting, composition).
  - *Tip*: Click **"💡 Prompt Idea"** to inject a high-quality sample prompt.
- **Negative Prompt**: Describe elements you want to avoid (e.g., `blurry, bad anatomy, low quality`).
  - *Tip*: Click **"+ Standard Negative"** to insert a curated negative prompt.

### Key Parameters
- **Sampling Method**:
  - **LCM**: Fastest method! Best for models trained or fine-tuned with Latent Consistency Models. Requires only **4 to 8 steps** with **CFG 1.2 to 2.0**.
  - **Euler a / Euler / DPM++ 2M Karras**: Standard samplers for non-LCM models. Recommended: **20 steps** with **CFG 7.0**.
- **Sampling Steps**: The number of denoising iterations. Higher isn't always better—for LCM models, 5–6 steps produces sharp results in record time.
- **CFG Scale (Guidance)**: How strictly the AI adheres to your prompt.
  - For **LCM**: Keep between **1.2 and 2.0**.
  - For **Standard SD**: Keep between **6.0 and 8.0**.
- **Resolution**:
  - Click the preset buttons for quick standard dimensions:
    - `512 × 512` (Square)
    - `512 × 768` (Portrait)
    - `768 × 512` (Landscape)
- **Seed**:
  - `-1`: Generates a completely new random image every time.
  - Specific number: Reproduces the exact composition and variation.
  - Click **🎲** to randomize, or **♻️** to reload the seed from the last generation.

### Generation & Progress
- Click **"Generate"** to begin. The button turns into an **"Interrupt"** button, and a live progress bar displays the current step, elapsed time, and real-time ETA.
- If you change your mind mid-generation, click **"Interrupt"** to halt processing immediately.

---

## 3. Image-to-Image (img2img)

Transform an existing image into a new style or variation:

1. **Upload Source**:
   - Drag & drop an image into the upload dropzone, click to browse, or click **"🖼️ Send to img2img"** from any completed render.
2. **Prompt**: Enter a prompt describing the changes or style (e.g. `watercolor painting of an astronaut, vibrant`).
3. **Denoising Strength**:
   - `0.20 – 0.40`: Subtle variations, keeps nearly all original details.
   - `0.50 – 0.70`: Balanced transformation (ideal for artistic restyling).
   - `0.75 – 1.00`: Heavy re-imagination, loosely guided by the original shapes and colors.
4. Click **"Generate img2img"**.

---

## 4. Inpainting (Inpaint)

Replace or modify specific elements in an image while leaving the rest completely untouched:

### Step-by-Step Workflow
1. **Load Source Image**:
   - Drag and drop an image into the Inpaint dropzone, or click **"🖌️ Send to Inpaint"** from any generation in txt2img or img2img.
2. **Paint the Mask**:
   - Use the interactive brush to paint over the exact area you want to replace (the mask appears as a translucent orange overlay).
   - Adjust the **Brush Size slider** (5px to 100px) for broad strokes or delicate details.
   - Click **"🧹 Clear Mask"** to erase the painted mask and start over.
   - Click **"↩️ Undo"** to revert your last brush strokes.
   - Check **"Invert"** if you want to keep the masked subject and replace the background instead.
3. **Set the Prompt**:
   - Describe what should fill the masked area (e.g., `golden aviator sunglasses`, `cybernetic arm with glowing blue LEDs`, `blooming red roses`).
4. **Set Parameters**:
   - **Denoising strength**: Set to `1.0` to completely replace the masked region, or `0.6–0.8` to blend new details with the existing background.
   - **Mask blur (feather)**: Defaults to `4px` to softly blend the boundaries so there are no harsh seams.
5. **Generate**: Click **"Generate Inpaint"**.
6. **Iterative / Chained Inpainting**:
   - Once your inpaint completes, click **"🖌️ Send back to Inpaint"** to feed the result back onto the canvas. You can immediately paint a mask over another area and continue refining your image!

---

## 5. Gallery & Parameter Replication (<Copy Info>)

### Full Aspect Ratio Display
All images in the **Recent Images** strip, the main viewer, and the **Gallery** tab preserve their full aspect ratio with zero cropping or distortion.

### Replicating Settings (`<Copy Info>`)
Whenever you find an image you want to recreate or build upon:
- Click the **`📋 <Copy Info>`** button on the preview bar or on any card in the Gallery.
- This immediately:
  1. Populates the **Prompt** and **Negative Prompt**.
  2. Sets **Sampling Steps**, **CFG Scale**, **Width**, and **Height**.
  3. Matches the **Sampler** and original **Seed**.
  4. Selects the original **Checkpoint Model** in the dropdown.
  5. Copies the standard parameter string to your clipboard.
  6. Automatically switches your tab to **txt2img** so you can tweak and generate right away.

### Deleting Images
- Click the **`🗑️ Delete`** button on the viewer or the red **`🗑️`** icon on any Gallery card.
- A confirmation prompt will appear. Upon confirmation, the image file is permanently deleted from the `outputs/` folder and purged from `history.json`.

---

## 6. Managing Models & LoRAs

### Directory Locations
By default, AAA1v1 looks in:
- `models/checkpoints/` (or external USB SSD `/media/jack/USB_USBSSD/models/checkpoints`)
- `models/loras/`
- `models/vae/`

### Adding New Models
1. Copy any Stable Diffusion 1.5 `.safetensors` file into the checkpoints folder.
2. In the top navigation bar, click the **🔄** button to refresh the list without restarting the app.
3. Select your new model from the dropdown.

### Using LoRAs
1. Expand the **"LoRA Networks"** accordion on the txt2img panel.
2. Select your desired LoRA from the dropdown.
3. Adjust the **Weight** slider (typically `0.6` to `0.85`).

---

## 7. CPU Performance & LCM Acceleration

If you are running on a CPU or low-VRAM computer:
- **Always prioritize LCM models**: Standard models require 20–30 steps (often taking 15+ minutes per image on CPU). LCM models converge in just **4 to 6 steps** and complete in roughly **2.5 to 3 minutes**!
- **Keep dimensions at 512 × 512 or 512 × 768**: Generating at 1024×1024 on CPU takes exponentially longer; stick to 512px base resolution for the smoothest experience.
- **Attention Slicing**: AAA1v1 automatically enables attention slicing in PyTorch, reducing peak RAM usage so your system stays snappy.
