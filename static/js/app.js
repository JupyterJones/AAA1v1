document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const checkpointSelect = document.getElementById("checkpoint-select");
  const vaeSelect = document.getElementById("vae-select");
  const loraSelect = document.getElementById("lora-select");
  const refreshModelsBtn = document.getElementById("refresh-models-btn");
  
  const promptInput = document.getElementById("prompt");
  const negativePromptInput = document.getElementById("negative-prompt");
  const generateBtn = document.getElementById("generate-btn");
  const interruptBtn = document.getElementById("interrupt-btn");
  const samplePromptBtn = document.getElementById("sample-prompt-btn");
  const negCleanBtn = document.getElementById("neg-clean-btn");

  const samplerSelect = document.getElementById("sampler-select");
  const stepsSlider = document.getElementById("steps");
  const stepsNum = document.getElementById("steps-num");
  const cfgSlider = document.getElementById("cfg-scale");
  const cfgNum = document.getElementById("cfg-scale-num");
  const widthSlider = document.getElementById("width");
  const widthNum = document.getElementById("width-num");
  const heightSlider = document.getElementById("height");
  const heightNum = document.getElementById("height-num");
  const seedInput = document.getElementById("seed");
  const randomSeedBtn = document.getElementById("random-seed-btn");
  const reuseSeedBtn = document.getElementById("reuse-seed-btn");
  const loraWeightSlider = document.getElementById("lora-weight");
  const loraWeightVal = document.getElementById("lora-weight-val");

  // Output elements
  const progressContainer = document.getElementById("progress-container");
  const progressBar = document.getElementById("progress-bar");
  const progressText = document.getElementById("progress-text");
  const progressEta = document.getElementById("progress-eta");
  const placeholderView = document.getElementById("placeholder-view");
  const mainOutputImg = document.getElementById("main-output-img");
  const imageActions = document.getElementById("image-actions");
  const downloadBtn = document.getElementById("download-btn");
  const sendImg2imgBtn = document.getElementById("send-img2img-btn");
  const copyMetaBtn = document.getElementById("copy-meta-btn");
  const deleteCurrentBtn = document.getElementById("delete-current-btn");
  const clearGalleryBtn = document.getElementById("clear-gallery-btn");
  const metaCard = document.getElementById("meta-card");
  const metaModel = document.getElementById("meta-model");
  const metaSteps = document.getElementById("meta-steps");
  const metaCfg = document.getElementById("meta-cfg");
  const metaSampler = document.getElementById("meta-sampler");
  const metaSeed = document.getElementById("meta-seed");
  const metaTime = document.getElementById("meta-time");
  const thumbnailStrip = document.getElementById("thumbnail-strip");
  const historyCount = document.getElementById("history-count");

  // img2img elements
  const dropzone = document.getElementById("dropzone");
  const dropzoneText = document.getElementById("dropzone-text");
  const img2imgFile = document.getElementById("img2img-file");
  const img2imgPreview = document.getElementById("img2img-preview");
  const i2iPrompt = document.getElementById("i2i-prompt");
  const i2iGenerateBtn = document.getElementById("i2i-generate-btn");
  const i2iOutputImg = document.getElementById("i2i-output-img");
  const denoisingStrength = document.getElementById("denoising-strength");
  const strengthNum = document.getElementById("strength-num");
  const i2iSteps = document.getElementById("i2i-steps");
  const i2iStepsNum = document.getElementById("i2i-steps-num");
  const i2iActions = document.getElementById("i2i-actions");
  const i2iDownloadBtn = document.getElementById("i2i-download-btn");
  const i2iSendInpaintBtn = document.getElementById("i2i-send-inpaint-btn");

  // Inpaint elements
  const sendInpaintBtn = document.getElementById("send-inpaint-btn");
  const inpaintFile = document.getElementById("inpaint-file");
  const inpaintDropzone = document.getElementById("inpaint-dropzone");
  const inpaintCanvasWrapper = document.getElementById("inpaint-canvas-wrapper");
  const inpaintBaseImg = document.getElementById("inpaint-base-img");
  const inpaintMaskCanvas = document.getElementById("inpaint-mask-canvas");
  const brushSizeSlider = document.getElementById("brush-size");
  const brushSizeVal = document.getElementById("brush-size-val");
  const inpaintClearBtn = document.getElementById("inpaint-clear-btn");
  const inpaintUndoBtn = document.getElementById("inpaint-undo-btn");
  const inpaintInvert = document.getElementById("inpaint-invert");

  const inpaintPrompt = document.getElementById("inpaint-prompt");
  const inpaintNegPrompt = document.getElementById("inpaint-neg-prompt");
  const inpaintGenerateBtn = document.getElementById("inpaint-generate-btn");
  const inpaintInterruptBtn = document.getElementById("inpaint-interrupt-btn");

  const inpaintStrength = document.getElementById("inpaint-strength");
  const inpaintStrengthNum = document.getElementById("inpaint-strength-num");
  const inpaintMaskBlur = document.getElementById("inpaint-mask-blur");
  const inpaintMaskBlurNum = document.getElementById("inpaint-mask-blur-num");
  const inpaintSteps = document.getElementById("inpaint-steps");
  const inpaintStepsNum = document.getElementById("inpaint-steps-num");
  const inpaintCfg = document.getElementById("inpaint-cfg");
  const inpaintCfgNum = document.getElementById("inpaint-cfg-num");
  const inpaintSeed = document.getElementById("inpaint-seed");
  const inpaintRandomSeedBtn = document.getElementById("inpaint-random-seed-btn");
  const inpaintReuseSeedBtn = document.getElementById("inpaint-reuse-seed-btn");

  const inpaintPlaceholderView = document.getElementById("inpaint-placeholder-view");
  const inpaintOutputImg = document.getElementById("inpaint-output-img");
  const inpaintImageActions = document.getElementById("inpaint-image-actions");
  const inpaintDownloadBtn = document.getElementById("inpaint-download-btn");
  const inpaintReuseOutputBtn = document.getElementById("inpaint-reuse-output-btn");
  const inpaintSendImg2imgBtn = document.getElementById("inpaint-send-img2img-btn");
  const inpaintCopyMetaBtn = document.getElementById("inpaint-copy-meta-btn");

  const inpaintMetaCard = document.getElementById("inpaint-meta-card");
  const inpaintMetaModel = document.getElementById("inpaint-meta-model");
  const inpaintMetaSteps = document.getElementById("inpaint-meta-steps");
  const inpaintMetaCfg = document.getElementById("inpaint-meta-cfg");
  const inpaintMetaStrength = document.getElementById("inpaint-meta-strength");
  const inpaintMetaSeed = document.getElementById("inpaint-meta-seed");
  const inpaintMetaTime = document.getElementById("inpaint-meta-time");

  let pollInterval = null;
  let lastGeneratedSeed = null;
  let currentActiveMeta = null;
  let currentInpaintMeta = null;

  // Sync dual slider-number inputs
  function bindSliderNum(slider, num) {
    slider.addEventListener("input", () => { num.value = slider.value; });
    num.addEventListener("input", () => { slider.value = num.value; });
  }
  bindSliderNum(stepsSlider, stepsNum);
  bindSliderNum(cfgSlider, cfgNum);
  bindSliderNum(widthSlider, widthNum);
  bindSliderNum(heightSlider, heightNum);
  bindSliderNum(denoisingStrength, strengthNum);
  bindSliderNum(i2iSteps, i2iStepsNum);
  bindSliderNum(inpaintStrength, inpaintStrengthNum);
  bindSliderNum(inpaintMaskBlur, inpaintMaskBlurNum);
  bindSliderNum(inpaintSteps, inpaintStepsNum);
  bindSliderNum(inpaintCfg, inpaintCfgNum);

  loraWeightSlider.addEventListener("input", () => {
    loraWeightVal.textContent = loraWeightSlider.value;
  });

  // Preset Resolution Buttons
  document.querySelectorAll(".preset-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".preset-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const w = btn.getAttribute("data-w");
      const h = btn.getAttribute("data-h");
      widthSlider.value = w; widthNum.value = w;
      heightSlider.value = h; heightNum.value = h;
    });
  });

  // Tabs
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
      btn.classList.add("active");
      const tabId = btn.getAttribute("data-tab");
      document.getElementById(tabId).classList.add("active");
      if (tabId === "gallery-tab") loadFullGallery();
    });
  });

  // Fetch Available Models
  async function fetchModels() {
    try {
      const res = await fetch("/api/models");
      const data = await res.json();

      // Checkpoints
      checkpointSelect.innerHTML = "";
      let defaultSelected = false;
      data.checkpoints.forEach(cp => {
        const opt = document.createElement("option");
        opt.value = cp.path;
        opt.textContent = `${cp.name} (${cp.size_gb} GB)${cp.is_lcm ? " [LCM]" : ""}`;
        if (!defaultSelected && (cp.is_lcm || cp.name.includes("dreamshaper"))) {
          opt.selected = true;
          defaultSelected = true;
        }
        checkpointSelect.appendChild(opt);
      });

      // VAEs
      vaeSelect.innerHTML = '<option value="">Default (Baked VAE)</option>';
      data.vaes.forEach(v => {
        const opt = document.createElement("option");
        opt.value = v.path;
        opt.textContent = v.name;
        vaeSelect.appendChild(opt);
      });

      // LoRAs
      loraSelect.innerHTML = '<option value="">None</option>';
      data.loras.forEach(l => {
        const opt = document.createElement("option");
        opt.value = l.path;
        opt.textContent = l.name;
        loraSelect.appendChild(opt);
      });

      onModelChanged();
    } catch (err) {
      console.error("Error loading models:", err);
    }
  }

  function onModelChanged() {
    const selectedText = checkpointSelect.options[checkpointSelect.selectedIndex]?.text.toLowerCase() || "";
    const isLCM = selectedText.includes("lcm");
    if (isLCM) {
      samplerSelect.value = "LCM";
      stepsSlider.value = 5; stepsNum.value = 5;
      cfgSlider.value = 1.5; cfgNum.value = 1.5;
    } else {
      samplerSelect.value = "Euler a";
      stepsSlider.value = 20; stepsNum.value = 20;
      cfgSlider.value = 7.0; cfgNum.value = 7.0;
    }
  }

  checkpointSelect.addEventListener("change", onModelChanged);
  refreshModelsBtn.addEventListener("click", fetchModels);

  // Prompt helpers
  const promptIdeas = [
    "A cozy rainy coffee shop in Kyoto, neon signs reflecting on wet asphalt, studio ghibli aesthetic, 8k",
    "Futuristic cyberpunk street market, detailed cybernetics, volumetric lighting, unreal engine 5 render",
    "Ancient mystical tree glowing with bioluminescent blue flowers, magical forest, cinematic lighting",
    "Detailed oil painting portrait of an astronaut floating among vibrant cosmic nebula, masterpiece",
    "Retro 1980s synthwave sports car driving towards a wireframe sunset, detailed reflections"
  ];
  samplePromptBtn.addEventListener("click", () => {
    promptInput.value = promptIdeas[Math.floor(Math.random() * promptIdeas.length)];
  });

  negCleanBtn.addEventListener("click", () => {
    negativePromptInput.value = "blurry, low quality, bad anatomy, deformed limbs, distorted face, extra fingers, oversaturated, watermark, signature";
  });

  randomSeedBtn.addEventListener("click", () => {
    seedInput.value = -1;
  });

  reuseSeedBtn.addEventListener("click", () => {
    if (lastGeneratedSeed !== null) {
      seedInput.value = lastGeneratedSeed;
    }
  });

  // Generate txt2img
  generateBtn.addEventListener("click", async () => {
    const payload = {
      prompt: promptInput.value.trim(),
      negative_prompt: negativePromptInput.value.trim(),
      steps: parseInt(stepsSlider.value),
      cfg_scale: parseFloat(cfgSlider.value),
      width: parseInt(widthSlider.value),
      height: parseInt(heightSlider.value),
      seed: parseInt(seedInput.value) || -1,
      sampler: samplerSelect.value,
      model_path: checkpointSelect.value,
      vae_path: vaeSelect.value || null,
      lora_path: loraSelect.value || null,
      lora_scale: parseFloat(loraWeightSlider.value)
    };

    if (!payload.prompt) {
      alert("Please enter a prompt first!");
      return;
    }

    setGeneratingState(true);

    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!data.success) {
        alert(data.error || "Generation request failed");
        setGeneratingState(false);
      } else {
        startProgressPolling();
      }
    } catch (err) {
      console.error(err);
      alert("Failed to communicate with server");
      setGeneratingState(false);
    }
  });

  // Interrupt
  interruptBtn.addEventListener("click", async () => {
    try {
      await fetch("/api/interrupt", { method: "POST" });
    } catch (err) {
      console.error(err);
    }
  });

  function setGeneratingState(isBusy) {
    if (isBusy) {
      generateBtn.classList.add("hidden");
      interruptBtn.classList.remove("hidden");
      progressContainer.classList.remove("hidden");
      progressBar.style.width = "0%";
      progressText.textContent = "Initializing model & pipeline...";
      progressEta.textContent = "ETA: calculating...";
    } else {
      generateBtn.classList.remove("hidden");
      interruptBtn.classList.add("hidden");
      clearInterval(pollInterval);
    }
  }

  function startProgressPolling() {
    clearInterval(pollInterval);
    pollInterval = setInterval(async () => {
      try {
        const res = await fetch("/api/progress");
        const data = await res.json();

        if (data.is_busy) {
          const pct = Math.round(data.progress * 100);
          progressBar.style.width = `${pct}%`;
          progressText.textContent = data.info || `Step ${data.step}/${data.total_steps} (${pct}%)`;
          progressEta.textContent = `ETA: ${data.eta || 0}s (${data.elapsed || 0}s elapsed)`;
        } else {
          // Completed or interrupted
          setGeneratingState(false);
          if (data.last_result) {
            if (data.last_result.success && data.last_result.image) {
              displayGeneratedImage(data.last_result.image);
              loadGalleryThumbnails();
            } else if (data.last_result.error) {
              progressText.textContent = data.last_result.error;
            }
          }
        }
      } catch (err) {
        console.error(err);
      }
    }, 250);
  }

  function displayGeneratedImage(meta) {
    currentActiveMeta = meta;
    lastGeneratedSeed = meta.seed;
    placeholderView.classList.add("hidden");
    mainOutputImg.classList.remove("hidden");
    mainOutputImg.src = `${meta.url}?t=${Date.now()}`;

    imageActions.classList.remove("hidden");
    downloadBtn.href = meta.url;
    downloadBtn.download = meta.filename;

    metaCard.classList.remove("hidden");
    metaModel.textContent = meta.model;
    metaSteps.textContent = meta.steps;
    metaCfg.textContent = meta.cfg_scale;
    metaSampler.textContent = meta.sampler;
    metaSeed.textContent = meta.seed;
    metaTime.textContent = `${meta.render_time}s`;
  }

  // Action Buttons
  sendImg2imgBtn.addEventListener("click", () => {
    if (!currentActiveMeta) return;
    document.querySelector('[data-tab="img2img-tab"]').click();
    img2imgPreview.src = currentActiveMeta.url;
    img2imgPreview.classList.remove("hidden");
    dropzoneText.classList.add("hidden");
    i2iPrompt.value = currentActiveMeta.prompt;
  });

  if (sendInpaintBtn) {
    sendInpaintBtn.addEventListener("click", () => {
      if (!currentActiveMeta) return;
      document.querySelector('[data-tab="inpaint-tab"]').click();
      loadInpaintImage(currentActiveMeta.url);
      inpaintPrompt.value = currentActiveMeta.prompt || "";
      showToast("🖌️ Loaded image into Inpaint canvas");
    });
  }

  // Toast Helper
  let toastTimer = null;
  function showToast(message, duration = 3000) {
    const toast = document.getElementById("toast");
    if (!toast) return;
    toast.textContent = message;
    toast.classList.remove("hidden");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      toast.classList.add("hidden");
    }, duration);
  }

  function escapeHtml(text) {
    if (!text) return "";
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Copy Info into Controls & Clipboard
  function applyMetadataToControls(meta) {
    if (!meta) return;

    // Prompt & Negative prompt
    if (meta.prompt !== undefined) promptInput.value = meta.prompt;
    if (meta.negative_prompt !== undefined) negativePromptInput.value = meta.negative_prompt;

    // Steps
    if (meta.steps !== undefined) {
      stepsSlider.value = meta.steps;
      stepsNum.value = meta.steps;
    }

    // CFG Scale
    if (meta.cfg_scale !== undefined) {
      cfgSlider.value = meta.cfg_scale;
      cfgNum.value = meta.cfg_scale;
    }

    // Dimensions
    if (meta.width !== undefined) {
      widthSlider.value = meta.width;
      widthNum.value = meta.width;
    }
    if (meta.height !== undefined) {
      heightSlider.value = meta.height;
      heightNum.value = meta.height;
    }

    // Highlight matching resolution preset button
    document.querySelectorAll(".preset-btn").forEach(btn => {
      const bw = parseInt(btn.getAttribute("data-w"));
      const bh = parseInt(btn.getAttribute("data-h"));
      if (bw === parseInt(meta.width) && bh === parseInt(meta.height)) {
        btn.classList.add("active");
      } else {
        btn.classList.remove("active");
      }
    });

    // Sampler
    if (meta.sampler) {
      for (let i = 0; i < samplerSelect.options.length; i++) {
        if (samplerSelect.options[i].value.toLowerCase() === meta.sampler.toLowerCase()) {
          samplerSelect.selectedIndex = i;
          break;
        }
      }
    }

    // Seed
    if (meta.seed !== undefined) {
      seedInput.value = meta.seed;
      lastGeneratedSeed = meta.seed;
    }

    // Checkpoint Model
    const targetModel = (meta.model_path || meta.model || "").toLowerCase();
    if (targetModel) {
      for (let i = 0; i < checkpointSelect.options.length; i++) {
        const opt = checkpointSelect.options[i];
        const optVal = opt.value.toLowerCase();
        const optText = opt.text.toLowerCase();
        if (optVal === targetModel || optVal.includes(targetModel) || optText.includes(targetModel)) {
          checkpointSelect.selectedIndex = i;
          break;
        }
      }
    }

    // Also copy formatted string to clipboard
    const text = `${meta.prompt || ""}\nNegative prompt: ${meta.negative_prompt || ""}\nSteps: ${meta.steps}, Sampler: ${meta.sampler}, CFG scale: ${meta.cfg_scale}, Seed: ${meta.seed}, Size: ${meta.width}x${meta.height}, Model: ${meta.model}`;
    try {
      navigator.clipboard.writeText(text);
    } catch (e) {
      console.warn("Clipboard write failed:", e);
    }

    // Switch to txt2img tab
    const txtTabBtn = document.querySelector('[data-tab="txt2img-tab"]');
    if (txtTabBtn) txtTabBtn.click();

    showToast("📋 Copied info into prompt and parameters!");
  }

  // Delete Image from disk & history
  async function deleteImage(filename) {
    if (!filename) return false;
    try {
      const res = await fetch("/api/delete_image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename })
      });
      const data = await res.json();
      if (data.success) {
        if (currentActiveMeta && currentActiveMeta.filename === filename) {
          currentActiveMeta = null;
          mainOutputImg.src = "";
          mainOutputImg.classList.add("hidden");
          imageActions.classList.add("hidden");
          metaCard.classList.add("hidden");
          placeholderView.classList.remove("hidden");
        }
        loadGalleryThumbnails();
        showToast("🗑️ Image deleted from disk & history");
        return true;
      } else {
        alert(data.error || "Failed to delete image");
        return false;
      }
    } catch (err) {
      console.error("Delete error:", err);
      alert("Network error while deleting image");
      return false;
    }
  }

  copyMetaBtn.addEventListener("click", () => {
    if (!currentActiveMeta) return;
    applyMetadataToControls(currentActiveMeta);
  });

  if (deleteCurrentBtn) {
    deleteCurrentBtn.addEventListener("click", async () => {
      if (!currentActiveMeta) return;
      if (!confirm(`Permanently delete this image from disk and history?\n${currentActiveMeta.filename}`)) return;
      await deleteImage(currentActiveMeta.filename);
    });
  }

  if (clearGalleryBtn) {
    clearGalleryBtn.addEventListener("click", () => {
      thumbnailStrip.innerHTML = "";
    });
  }

  // Recent Thumbnails Reel (Full aspect ratio, contain)
  async function loadGalleryThumbnails() {
    try {
      const res = await fetch("/api/gallery");
      const list = await res.json();
      historyCount.textContent = list.length;
      thumbnailStrip.innerHTML = "";
      list.slice(0, 30).forEach(item => {
        const img = document.createElement("img");
        img.className = "thumbnail-item";
        img.src = `${item.url}?t=${Date.now()}`;
        img.alt = item.filename;
        img.title = `Seed: ${item.seed} • ${item.width || 512}x${item.height || 512}\n${item.prompt.substring(0, 50)}...`;
        img.addEventListener("click", () => {
          displayGeneratedImage(item);
          document.querySelectorAll(".thumbnail-item").forEach(t => t.classList.remove("active"));
          img.classList.add("active");
        });
        thumbnailStrip.appendChild(img);
      });
    } catch (err) {
      console.error(err);
    }
  }

  // Full Gallery View (Full aspect ratio, copy & delete buttons)
  async function loadFullGallery() {
    try {
      const res = await fetch("/api/gallery");
      const list = await res.json();
      const grid = document.getElementById("full-gallery-grid");
      grid.innerHTML = "";

      if (list.length === 0) {
        grid.innerHTML = `<div style="grid-column: 1/-1; text-align: center; color: var(--text-muted); padding: 50px 0;">No generated images in history yet.</div>`;
        return;
      }

      list.forEach(item => {
        const card = document.createElement("div");
        card.className = "gallery-card";
        card.setAttribute("data-filename", item.filename);
        card.innerHTML = `
          <button class="gallery-card-copy-btn" title="Copy info into prompt & parameters">📋</button>
          <button class="gallery-card-del-btn" title="Delete image from disk and history">🗑️</button>
          <div class="gallery-card-img-wrap">
            <img src="${item.url}" alt="Output" loading="lazy">
          </div>
          <div class="gallery-card-info">
            <p><strong>Seed:</strong> ${item.seed} | <strong>${item.width || 512}×${item.height || 512}</strong></p>
            <p style="color:#8b949e; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${escapeHtml(item.prompt)}">${escapeHtml(item.prompt)}</p>
          </div>
        `;

        // Click image to view in main preview
        card.querySelector("img").addEventListener("click", () => {
          document.querySelector('[data-tab="txt2img-tab"]').click();
          displayGeneratedImage(item);
        });

        // Copy button
        card.querySelector(".gallery-card-copy-btn").addEventListener("click", (e) => {
          e.stopPropagation();
          applyMetadataToControls(item);
        });

        // Delete button
        card.querySelector(".gallery-card-del-btn").addEventListener("click", async (e) => {
          e.stopPropagation();
          if (!confirm(`Permanently delete this image from disk and history?\n${item.filename}`)) return;
          const ok = await deleteImage(item.filename);
          if (ok) {
            card.remove();
            if (grid.children.length === 0) {
              grid.innerHTML = `<div style="grid-column: 1/-1; text-align: center; color: var(--text-muted); padding: 50px 0;">No generated images in history yet.</div>`;
            }
          }
        });

        grid.appendChild(card);
      });
    } catch (err) {
      console.error(err);
    }
  }

  // Dropzone for img2img
  dropzone.addEventListener("click", () => img2imgFile.click());
  img2imgFile.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      const reader = new FileReader();
      reader.onload = (re) => {
        img2imgPreview.src = re.target.result;
        img2imgPreview.classList.remove("hidden");
        dropzoneText.classList.add("hidden");
      };
      reader.readAsDataURL(e.target.files[0]);
    }
  });

  // Generate img2img
  i2iGenerateBtn.addEventListener("click", async () => {
    if (!img2imgFile.files || !img2imgFile.files[0]) {
      if (!img2imgPreview.src) {
        alert("Please upload an image first!");
        return;
      }
    }

    const formData = new FormData();
    if (img2imgFile.files[0]) {
      formData.append("init_image", img2imgFile.files[0]);
    } else {
      // Fetch current preview as blob
      const blob = await fetch(img2imgPreview.src).then(r => r.blob());
      formData.append("init_image", blob, "input.png");
    }

    formData.append("prompt", i2iPrompt.value.trim());
    formData.append("negative_prompt", negativePromptInput.value.trim());
    formData.append("steps", i2iSteps.value);
    formData.append("cfg_scale", cfgSlider.value);
    formData.append("strength", denoisingStrength.value);
    formData.append("width", widthSlider.value);
    formData.append("height", heightSlider.value);
    formData.append("seed", seedInput.value || -1);
    formData.append("sampler", samplerSelect.value);
    formData.append("model_path", checkpointSelect.value);
    formData.append("vae_path", vaeSelect.value || "");

    i2iGenerateBtn.disabled = true;
    i2iGenerateBtn.textContent = "Processing img2img...";

    try {
      const res = await fetch("/api/generate_img2img", {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      if (data.success) {
        startProgressPolling();
        // Wait for result
        const checkDone = setInterval(async () => {
          const pr = await fetch("/api/progress").then(r => r.json());
          if (!pr.is_busy) {
            clearInterval(checkDone);
            i2iGenerateBtn.disabled = false;
            i2iGenerateBtn.textContent = "Generate img2img";
            if (pr.last_result && pr.last_result.image) {
              i2iOutputImg.src = pr.last_result.image.url;
              i2iOutputImg.classList.remove("hidden");
              loadGalleryThumbnails();
            }
          }
        }, 500);
      } else {
        alert(data.error || "Failed");
        i2iGenerateBtn.disabled = false;
        i2iGenerateBtn.textContent = "Generate img2img";
      }
    } catch (err) {
      console.error(err);
      i2iGenerateBtn.disabled = false;
      i2iGenerateBtn.textContent = "Generate img2img";
    }
  });

  if (i2iSendInpaintBtn) {
    i2iSendInpaintBtn.addEventListener("click", () => {
      if (!i2iOutputImg.src) return;
      document.querySelector('[data-tab="inpaint-tab"]').click();
      loadInpaintImage(i2iOutputImg.src);
      inpaintPrompt.value = i2iPrompt.value || "";
      showToast("🖌️ Loaded img2img output into Inpaint canvas");
    });
  }

  // ==========================================
  // INPAINT MODULE
  // ==========================================
  let isDrawing = false;
  let strokeHistory = [];
  let inpaintSourceFile = null;
  let inpaintSourceUrl = null;
  let lastX = 0;
  let lastY = 0;

  if (brushSizeSlider) {
    brushSizeSlider.addEventListener("input", () => {
      brushSizeVal.textContent = brushSizeSlider.value;
    });
  }

  function getCanvasCoords(e) {
    const rect = inpaintMaskCanvas.getBoundingClientRect();
    const scaleX = inpaintMaskCanvas.width / rect.width;
    const scaleY = inpaintMaskCanvas.height / rect.height;
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY
    };
  }

  function saveStrokeState() {
    if (!inpaintMaskCanvas) return;
    const ctx = inpaintMaskCanvas.getContext("2d");
    if (strokeHistory.length > 25) strokeHistory.shift();
    strokeHistory.push(ctx.getImageData(0, 0, inpaintMaskCanvas.width, inpaintMaskCanvas.height));
  }

  function undoStroke() {
    if (!inpaintMaskCanvas || strokeHistory.length === 0) return;
    const prev = strokeHistory.pop();
    const ctx = inpaintMaskCanvas.getContext("2d");
    ctx.putImageData(prev, 0, 0);
  }

  function clearMask() {
    if (!inpaintMaskCanvas) return;
    const ctx = inpaintMaskCanvas.getContext("2d");
    ctx.clearRect(0, 0, inpaintMaskCanvas.width, inpaintMaskCanvas.height);
    strokeHistory = [];
  }

  function drawDot(x, y) {
    const ctx = inpaintMaskCanvas.getContext("2d");
    const radius = parseInt(brushSizeSlider.value) / 2;
    ctx.fillStyle = "rgba(240, 136, 62, 0.65)";
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
  }

  function startDrawing(e) {
    if (!inpaintBaseImg.src) return;
    e.preventDefault();
    isDrawing = true;
    saveStrokeState();
    const coords = getCanvasCoords(e);
    lastX = coords.x;
    lastY = coords.y;
    drawDot(coords.x, coords.y);
  }

  function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();
    const coords = getCanvasCoords(e);
    const ctx = inpaintMaskCanvas.getContext("2d");
    ctx.strokeStyle = "rgba(240, 136, 62, 0.65)";
    ctx.lineWidth = parseInt(brushSizeSlider.value);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(coords.x, coords.y);
    ctx.stroke();
    lastX = coords.x;
    lastY = coords.y;
  }

  function stopDrawing() {
    isDrawing = false;
  }

  if (inpaintMaskCanvas) {
    inpaintMaskCanvas.addEventListener("mousedown", startDrawing);
    inpaintMaskCanvas.addEventListener("mousemove", draw);
    window.addEventListener("mouseup", stopDrawing);

    inpaintMaskCanvas.addEventListener("touchstart", startDrawing, { passive: false });
    inpaintMaskCanvas.addEventListener("touchmove", draw, { passive: false });
    inpaintMaskCanvas.addEventListener("touchend", stopDrawing);
  }

  if (inpaintClearBtn) inpaintClearBtn.addEventListener("click", clearMask);
  if (inpaintUndoBtn) inpaintUndoBtn.addEventListener("click", undoStroke);

  function loadInpaintImage(src, file = null) {
    inpaintSourceFile = file;
    inpaintSourceUrl = typeof src === "string" ? src : null;

    inpaintBaseImg.onload = () => {
      inpaintDropzone.classList.add("hidden");
      inpaintCanvasWrapper.classList.remove("hidden");
      inpaintMaskCanvas.width = inpaintBaseImg.naturalWidth || 512;
      inpaintMaskCanvas.height = inpaintBaseImg.naturalHeight || 512;
      clearMask();
    };

    inpaintBaseImg.src = typeof src === "string" ? src : URL.createObjectURL(src);
  }

  if (inpaintDropzone) {
    inpaintDropzone.addEventListener("click", () => inpaintFile.click());
  }

  if (inpaintFile) {
    inpaintFile.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        loadInpaintImage(e.target.files[0], e.target.files[0]);
      }
    });
  }

  function exportBinaryMask() {
    if (!inpaintMaskCanvas) return null;
    const w = inpaintMaskCanvas.width;
    const h = inpaintMaskCanvas.height;
    const offCanvas = document.createElement("canvas");
    offCanvas.width = w;
    offCanvas.height = h;
    const offCtx = offCanvas.getContext("2d");

    const imgData = inpaintMaskCanvas.getContext("2d").getImageData(0, 0, w, h);
    const outData = offCtx.createImageData(w, h);
    let paintedCount = 0;

    for (let i = 0; i < imgData.data.length; i += 4) {
      const alpha = imgData.data[i + 3];
      if (alpha > 15) {
        outData.data[i] = 255;
        outData.data[i + 1] = 255;
        outData.data[i + 2] = 255;
        outData.data[i + 3] = 255;
        paintedCount++;
      } else {
        outData.data[i] = 0;
        outData.data[i + 1] = 0;
        outData.data[i + 2] = 0;
        outData.data[i + 3] = 255;
      }
    }

    if (paintedCount === 0) return null;
    offCtx.putImageData(outData, 0, 0);
    return offCanvas.toDataURL("image/png");
  }

  if (inpaintRandomSeedBtn) inpaintRandomSeedBtn.addEventListener("click", () => inpaintSeed.value = -1);
  if (inpaintReuseSeedBtn) {
    inpaintReuseSeedBtn.addEventListener("click", () => {
      if (lastGeneratedSeed !== null) inpaintSeed.value = lastGeneratedSeed;
    });
  }

  if (inpaintGenerateBtn) {
    inpaintGenerateBtn.addEventListener("click", async () => {
      if (!inpaintBaseImg.src || inpaintCanvasWrapper.classList.contains("hidden")) {
        alert("Please load or drop an image into the Inpaint canvas first!");
        return;
      }

      const maskData = exportBinaryMask();
      if (!maskData) {
        alert("Please paint a mask over the area you want to inpaint using the brush!");
        return;
      }

      const promptText = inpaintPrompt.value.trim();
      if (!promptText) {
        alert("Please enter a prompt describing what to inpaint!");
        return;
      }

      const formData = new FormData();
      if (inpaintSourceFile) {
        formData.append("init_image", inpaintSourceFile);
      } else if (inpaintSourceUrl) {
        formData.append("init_image_url", inpaintSourceUrl);
      } else {
        const off = document.createElement("canvas");
        off.width = inpaintBaseImg.naturalWidth || 512;
        off.height = inpaintBaseImg.naturalHeight || 512;
        off.getContext("2d").drawImage(inpaintBaseImg, 0, 0);
        formData.append("init_image_data", off.toDataURL("image/png"));
      }

      formData.append("mask_data", maskData);
      formData.append("prompt", promptText);
      formData.append("negative_prompt", inpaintNegPrompt.value.trim());
      formData.append("steps", inpaintSteps.value);
      formData.append("cfg_scale", inpaintCfg.value);
      formData.append("strength", inpaintStrength.value);
      formData.append("mask_blur", inpaintMaskBlur.value);
      formData.append("invert_mask", inpaintInvert.checked ? "true" : "false");
      formData.append("width", inpaintBaseImg.naturalWidth || 512);
      formData.append("height", inpaintBaseImg.naturalHeight || 512);
      formData.append("seed", inpaintSeed.value || -1);
      formData.append("sampler", samplerSelect.value);
      formData.append("model_path", checkpointSelect.value);
      formData.append("vae_path", vaeSelect.value || "");

      inpaintGenerateBtn.disabled = true;
      inpaintGenerateBtn.textContent = "Processing Inpaint...";

      try {
        const res = await fetch("/api/generate_inpaint", {
          method: "POST",
          body: formData
        });
        const data = await res.json();
        if (data.success) {
          startProgressPolling();

          const checkDone = setInterval(async () => {
            const pr = await fetch("/api/progress").then(r => r.json());
            if (!pr.is_busy) {
              clearInterval(checkDone);
              inpaintGenerateBtn.disabled = false;
              inpaintGenerateBtn.textContent = "Generate Inpaint";

              if (pr.last_result && pr.last_result.image) {
                displayInpaintResult(pr.last_result.image);
                loadGalleryThumbnails();
                showToast("✨ Inpaint completed!");
              } else if (pr.last_result && pr.last_result.error) {
                alert(pr.last_result.error);
              }
            }
          }, 500);
        } else {
          alert(data.error || "Inpaint failed");
          inpaintGenerateBtn.disabled = false;
          inpaintGenerateBtn.textContent = "Generate Inpaint";
        }
      } catch (err) {
        console.error(err);
        alert("Failed to send inpaint request");
        inpaintGenerateBtn.disabled = false;
        inpaintGenerateBtn.textContent = "Generate Inpaint";
      }
    });
  }

  function displayInpaintResult(meta) {
    currentInpaintMeta = meta;
    lastGeneratedSeed = meta.seed;
    inpaintPlaceholderView.classList.add("hidden");
    inpaintOutputImg.classList.remove("hidden");
    inpaintOutputImg.src = `${meta.url}?t=${Date.now()}`;

    inpaintImageActions.classList.remove("hidden");
    inpaintDownloadBtn.href = meta.url;
    inpaintDownloadBtn.download = meta.filename;

    inpaintMetaCard.classList.remove("hidden");
    inpaintMetaModel.textContent = meta.model || "-";
    inpaintMetaSteps.textContent = meta.steps || "-";
    inpaintMetaCfg.textContent = meta.cfg_scale || "-";
    inpaintMetaStrength.textContent = meta.strength !== undefined ? meta.strength : "-";
    inpaintMetaSeed.textContent = meta.seed !== undefined ? meta.seed : "-";
    inpaintMetaTime.textContent = meta.render_time ? `${meta.render_time}s` : "-";
  }

  if (inpaintReuseOutputBtn) {
    inpaintReuseOutputBtn.addEventListener("click", () => {
      if (!currentInpaintMeta) return;
      loadInpaintImage(currentInpaintMeta.url);
      showToast("🖌️ Loaded output back into Inpaint canvas");
    });
  }

  if (inpaintSendImg2imgBtn) {
    inpaintSendImg2imgBtn.addEventListener("click", () => {
      if (!currentInpaintMeta) return;
      document.querySelector('[data-tab="img2img-tab"]').click();
      img2imgPreview.src = currentInpaintMeta.url;
      img2imgPreview.classList.remove("hidden");
      dropzoneText.classList.add("hidden");
      i2iPrompt.value = currentInpaintMeta.prompt || "";
    });
  }

  if (inpaintCopyMetaBtn) {
    inpaintCopyMetaBtn.addEventListener("click", () => {
      if (!currentInpaintMeta) return;
      applyMetadataToControls(currentInpaintMeta);
    });
  }

  // Init
  fetchModels();
  loadGalleryThumbnails();
});
