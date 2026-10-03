#!/usr/bin/env python3
import os
import gc
import time
import glob
import torch
from datetime import datetime
from PIL import Image, ImageFilter, ImageOps
from icecream import ic
# Ensure PyTorch ignores CUDA checks on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    LCMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    AutoencoderTiny,
    AutoencoderKL
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model directories to scan
MODEL_PATHS = [
    "/media/jack/USB_USBSSD/models/checkpoints",
    os.path.join(BASE_DIR, "models", "checkpoints"),
    os.path.join(BASE_DIR, "models")
]
LORA_PATHS = [
    "/media/jack/USB_USBSSD/models/loras",
    os.path.join(BASE_DIR, "models", "loras")
]
VAE_PATHS = [
    "/media/jack/USB_USBSSD/models/vae",
    os.path.join(BASE_DIR, "models", "vae")
]

class GenerationEngine:
    def __init__(self):
        self.current_model_path = None
        self.current_vae_path = None
        self.pipe = None
        self.pipe_img2img = None
        self.pipe_inpaint = None
        self.device = "cpu"
        self.torch_dtype = torch.float32
        self.is_interrupted = False
        self.progress_state = {
            "status": "idle",
            "step": 0,
            "total_steps": 0,
            "progress": 0.0,
            "eta": 0,
            "elapsed": 0.0,
            "info": ""
        }
        
        # Configure CPU thread limits for optimal performance
        torch.set_num_threads(4)
        try:
            torch.set_num_interop_threads(4)
        except Exception:
            pass
        os.environ["OMP_NUM_THREADS"] = "4"
        os.environ["MKL_NUM_THREADS"] = "4"

    def scan_models(self):
        checkpoints = {}
        for path in MODEL_PATHS:
            if os.path.exists(path):
                for f in sorted(glob.glob(os.path.join(path, "**/*.safetensors"), recursive=True) +
                                glob.glob(os.path.join(path, "**/*.ckpt"), recursive=True)):
                    name = os.path.basename(f)
                    if name not in checkpoints:
                        checkpoints[name] = {
                            "name": name,
                            "path": f,
                            "is_lcm": "lcm" in name.lower(),
                            "size_gb": round(os.path.getsize(f) / (1024**3), 2)
                        }
        return checkpoints

    def scan_loras(self):
        loras = {}
        for path in LORA_PATHS:
            if os.path.exists(path):
                for f in sorted(glob.glob(os.path.join(path, "**/*.safetensors"), recursive=True) +
                                glob.glob(os.path.join(path, "**/*.ckpt"), recursive=True)):
                    name = os.path.basename(f)
                    if name not in loras:
                        loras[name] = {"name": name, "path": f}
        return loras

    def scan_vaes(self):
        vaes = {}
        for path in VAE_PATHS:
            if os.path.exists(path):
                for f in sorted(glob.glob(os.path.join(path, "**/*.safetensors"), recursive=True) +
                                glob.glob(os.path.join(path, "**/*.ckpt"), recursive=True)):
                    name = os.path.basename(f)
                    if name not in vaes:
                        vaes[name] = {"name": name, "path": f}
        return vaes

    def load_model(self, model_path, vae_path=None):
        if self.pipe is not None and self.current_model_path == model_path and self.current_vae_path == vae_path:
            return True, "Model already cached in memory"

        self.progress_state["status"] = "loading_model"
        self.progress_state["info"] = f"Loading {os.path.basename(model_path)} into RAM..."
        
        # Free old model from memory
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        if self.pipe_img2img is not None:
            del self.pipe_img2img
            self.pipe_img2img = None
        if self.pipe_inpaint is not None:
            del self.pipe_inpaint
            self.pipe_inpaint = None
        gc.collect()

        file_size_gb = os.path.getsize(model_path) / (1024**3)
        is_large = file_size_gb > 3.5
        dtype = torch.bfloat16 if is_large else torch.float32

        print(f"[*] Loading checkpoint from_single_file: {model_path} ({file_size_gb:.2f} GB, dtype: {dtype})")
        
        try:
            pipe = StableDiffusionPipeline.from_single_file(
                model_path,
                torch_dtype=dtype,
                use_safetensors=True,
                extract_ema=True,
                safety_checker=None,
                requires_safety_checker=False
            )
            pipe = pipe.to(self.device)
            pipe.enable_attention_slicing(1)
            
            # Custom VAE if specified
            if vae_path and os.path.exists(vae_path):
                print(f"[*] Loading custom VAE: {vae_path}")
                try:
                    if "taesd" in vae_path.lower():
                        custom_vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=dtype)
                    else:
                        custom_vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=dtype)
                    pipe.vae = custom_vae.to(self.device)
                except Exception as ve:
                    print(f"[!] Custom VAE error: {ve}")

            if hasattr(pipe, "vae") and pipe.vae is not None:
                if hasattr(pipe.vae, "enable_slicing"):
                    pipe.vae.enable_slicing()
                if hasattr(pipe.vae, "enable_tiling"):
                    pipe.vae.enable_tiling()

            self.pipe = pipe
            self.current_model_path = model_path
            self.current_vae_path = vae_path
            self.progress_state["status"] = "idle"
            self.progress_state["info"] = f"Loaded {os.path.basename(model_path)}"
            return True, f"Loaded {os.path.basename(model_path)}"
        except Exception as e:
            self.progress_state["status"] = "error"
            self.progress_state["info"] = f"Failed to load: {e}"
            print(f"[ERROR] Loading model failed: {e}")
            return False, str(e)

    def set_scheduler_for_pipe(self, pipe_obj, sampler_name, is_lcm=False):
        if pipe_obj is None:
            return
        cfg = pipe_obj.scheduler.config
        if sampler_name == "LCM" or (sampler_name == "Auto" and is_lcm):
            pipe_obj.scheduler = LCMScheduler.from_config(cfg)
        elif sampler_name == "Euler":
            pipe_obj.scheduler = EulerDiscreteScheduler.from_config(cfg)
        elif sampler_name == "Euler a":
            pipe_obj.scheduler = EulerAncestralDiscreteScheduler.from_config(cfg)
        elif sampler_name == "DPM++ 2M Karras":
            pipe_obj.scheduler = DPMSolverMultistepScheduler.from_config(cfg, use_karras_sigmas=True)
        else:
            if is_lcm:
                pipe_obj.scheduler = LCMScheduler.from_config(cfg)
            else:
                pipe_obj.scheduler = EulerAncestralDiscreteScheduler.from_config(cfg)

    def set_scheduler(self, sampler_name, is_lcm=False):
        self.set_scheduler_for_pipe(self.pipe, sampler_name, is_lcm)

    def interrupt(self):
        self.is_interrupted = True
        self.progress_state["status"] = "interrupting"

    def encode_prompt_long(self, pipe_obj, prompt, negative_prompt=""):
        """
        Encode prompts of arbitrary length into 77-token chunks without truncation.
        Concatenates hidden states across chunks so UNet attends to unlimited tokens.
        """
        tokenizer = pipe_obj.tokenizer
        text_encoder = pipe_obj.text_encoder
        device = self.device

        bos_token_id = tokenizer.bos_token_id
        eos_token_id = tokenizer.eos_token_id
        pad_token_id = tokenizer.pad_token_id or eos_token_id

        def tokenize_to_chunks(text):
            if not text or not str(text).strip():
                return [[bos_token_id, eos_token_id] + [pad_token_id] * 75]

            orig_max = tokenizer.model_max_length
            try:
                tokenizer.model_max_length = int(1e9)
                tokens = tokenizer(str(text), truncation=False, add_special_tokens=False)["input_ids"]
            finally:
                tokenizer.model_max_length = orig_max

            if not tokens:
                return [[bos_token_id, eos_token_id] + [pad_token_id] * 75]

            chunks = []
            chunk_size = 75
            for i in range(0, len(tokens), chunk_size):
                chunk = tokens[i : i + chunk_size]
                padded = [bos_token_id] + chunk + [eos_token_id]
                pad_len = 77 - len(padded)
                if pad_len > 0:
                    padded = padded + [pad_token_id] * pad_len
                chunks.append(padded)
            return chunks

        prompt_chunks = tokenize_to_chunks(prompt)
        neg_chunks = tokenize_to_chunks(negative_prompt)

        # Equalize chunk count so prompt and negative prompt have matching sequence length
        num_chunks = max(len(prompt_chunks), len(neg_chunks))
        empty_chunk = [bos_token_id, eos_token_id] + [pad_token_id] * 75

        while len(prompt_chunks) < num_chunks:
            prompt_chunks.append(empty_chunk)
        while len(neg_chunks) < num_chunks:
            neg_chunks.append(empty_chunk)

        if num_chunks > 1:
            print(f"[*] Long prompt detected: split into {num_chunks} chunks ({num_chunks * 77} tokens total)")

        target_dtype = getattr(pipe_obj.unet, "dtype", text_encoder.dtype)

        def encode_chunks(chunks):
            embeds = []
            for c in chunks:
                chunk_tensor = torch.tensor([c], dtype=torch.long, device=device)
                with torch.no_grad():
                    e = text_encoder(chunk_tensor)[0]
                embeds.append(e)
            return torch.cat(embeds, dim=1).to(dtype=target_dtype, device=device)

        prompt_embeds = encode_chunks(prompt_chunks)
        neg_prompt_embeds = encode_chunks(neg_chunks)

        return prompt_embeds, neg_prompt_embeds

    def generate_txt2img(self, params, output_dir):
        prompt = params.get("prompt", "")
        negative_prompt = params.get("negative_prompt", "")
        steps = int(params.get("steps", 5))
        cfg_scale = float(params.get("cfg_scale", 1.5))
        width = int(params.get("width", 512))
        height = int(params.get("height", 512))
        seed = int(params.get("seed", -1))
        sampler = params.get("sampler", "LCM")
        model_path = params.get("model_path")
        vae_path = params.get("vae_path")
        lora_path = params.get("lora_path")
        lora_scale = float(params.get("lora_scale", 0.75))

        if not model_path or not os.path.exists(model_path):
            return {"success": False, "error": "Model file not found"}

        # Load model if not already active
        success, msg = self.load_model(model_path, vae_path)
        if not success:
            return {"success": False, "error": msg}

        # Apply LoRA if provided
        if lora_path and os.path.exists(lora_path):
            try:
                self.pipe.unload_lora_weights()
            except Exception:
                pass
            try:
                self.pipe.load_lora_weights(lora_path)
                self.pipe.fuse_lora(lora_scale=lora_scale)
            except Exception as le:
                print(f"[!] LoRA load error: {le}")

        # Configure scheduler
        is_lcm = "lcm" in os.path.basename(model_path).lower()
        self.set_scheduler(sampler, is_lcm=is_lcm)

        # Handle seed
        if seed == -1 or seed < 0:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(device=self.device).manual_seed(seed)

        self.is_interrupted = False
        self.progress_state = {
            "status": "generating",
            "step": 0,
            "total_steps": steps,
            "progress": 0.0,
            "eta": 0,
            "elapsed": 0.0,
            "info": f"Step 0/{steps}"
        }

        start_time = time.time()

        def step_callback(pipe_obj, step_index, timestep, callback_kwargs):
            if self.is_interrupted:
                raise InterruptedError("Generation canceled by user")
            elapsed = time.time() - start_time
            step = step_index + 1
            progress = round(step / steps, 3)
            time_per_step = elapsed / max(1, step)
            eta = round((steps - step) * time_per_step, 1)
            self.progress_state["step"] = step
            self.progress_state["total_steps"] = steps
            self.progress_state["progress"] = progress
            self.progress_state["elapsed"] = round(elapsed, 1)
            self.progress_state["eta"] = eta
            self.progress_state["info"] = f"Step {step}/{steps} ({int(progress*100)}%) • {round(time_per_step, 1)}s/it • ETA: {eta}s"
            return callback_kwargs

        prompt_embeds, neg_prompt_embeds = self.encode_prompt_long(self.pipe, prompt, negative_prompt)

        try:
            with torch.inference_mode():
                result = self.pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=neg_prompt_embeds,
                    num_inference_steps=steps,
                    guidance_scale=cfg_scale,
                    width=width,
                    height=height,
                    generator=generator,
                    callback_on_step_end=step_callback
                )
            
            image = result.images[0]
            total_time = round(time.time() - start_time, 2)
            
            # Save output image
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gen_{timestamp}_{seed}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath, format="PNG")

            meta = {
                "filename": filename,
                "filepath": filepath,
                "url": f"/outputs/{filename}",
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "sampler": sampler,
                "seed": seed,
                "width": width,
                "height": height,
                "model": os.path.basename(model_path),
                "model_path": model_path,
                "render_time": total_time,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            self.progress_state["status"] = "completed"
            self.progress_state["progress"] = 1.0
            self.progress_state["info"] = f"Finished in {total_time}s"

            return {"success": True, "image": meta}

        except InterruptedError:
            self.progress_state["status"] = "interrupted"
            self.progress_state["info"] = "Interrupted by user"
            return {"success": False, "interrupted": True, "error": "Generation interrupted"}
        except Exception as e:
            self.progress_state["status"] = "error"
            self.progress_state["info"] = f"Error: {e}"
            print(f"[ERROR] Inference error: {e}")
            return {"success": False, "error": str(e)}

    def generate_img2img(self, params, init_image_path, output_dir):
        prompt = params.get("prompt", "")
        negative_prompt = params.get("negative_prompt", "")
        steps = int(params.get("steps", 5))
        cfg_scale = float(params.get("cfg_scale", 1.5))
        strength = float(params.get("strength", 0.65))
        seed = int(params.get("seed", -1))
        sampler = params.get("sampler", "LCM")
        model_path = params.get("model_path")
        vae_path = params.get("vae_path")

        if not model_path or not os.path.exists(model_path):
            return {"success": False, "error": "Model file not found"}

        success, msg = self.load_model(model_path, vae_path)
        if not success:
            return {"success": False, "error": msg}

        is_lcm = "lcm" in os.path.basename(model_path).lower()
        self.set_scheduler(sampler, is_lcm=is_lcm)

        # Build img2img pipeline reusing components from txt2img pipe to save RAM
        img2img_pipe = StableDiffusionImg2ImgPipeline(
            vae=self.pipe.vae,
            text_encoder=self.pipe.text_encoder,
            tokenizer=self.pipe.tokenizer,
            unet=self.pipe.unet,
            scheduler=self.pipe.scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False
        ).to(self.device)
        img2img_pipe.enable_attention_slicing(1)

        init_img = Image.open(init_image_path).convert("RGB")
        width = int(params.get("width", 512))
        height = int(params.get("height", 512))
        init_img = init_img.resize((width, height), Image.Resampling.LANCZOS)

        if seed == -1 or seed < 0:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(device=self.device).manual_seed(seed)

        self.is_interrupted = False
        start_time = time.time()
        self.progress_state = {
            "status": "generating",
            "step": 0,
            "total_steps": steps,
            "progress": 0.0,
            "eta": 0,
            "elapsed": 0.0,
            "info": "Img2Img processing..."
        }

        prompt_embeds, neg_prompt_embeds = self.encode_prompt_long(img2img_pipe, prompt, negative_prompt)

        try:
            with torch.inference_mode():
                result = img2img_pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=neg_prompt_embeds,
                    image=init_img,
                    strength=strength,
                    num_inference_steps=steps,
                    guidance_scale=cfg_scale,
                    generator=generator
                )

            image = result.images[0]
            total_time = round(time.time() - start_time, 2)

            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"i2i_{timestamp}_{seed}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath, format="PNG")

            meta = {
                "filename": filename,
                "filepath": filepath,
                "url": f"/outputs/{filename}",
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "strength": strength,
                "sampler": sampler,
                "seed": seed,
                "width": width,
                "height": height,
                "model": os.path.basename(model_path),
                "model_path": model_path,
                "render_time": total_time,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            self.progress_state["status"] = "completed"
            self.progress_state["progress"] = 1.0
            self.progress_state["info"] = f"Finished in {total_time}s"
            return {"success": True, "image": meta}

        except Exception as e:
            self.progress_state["status"] = "error"
            self.progress_state["info"] = f"Error: {e}"
            return {"success": False, "error": str(e)}

    def generate_inpaint(self, params, init_image_path, mask_image_path, output_dir):
        prompt = params.get("prompt", "")
        negative_prompt = params.get("negative_prompt", "")
        steps = int(params.get("steps", 5))
        cfg_scale = float(params.get("cfg_scale", 1.5))
        strength = float(params.get("strength", 1.0))
        mask_blur = int(params.get("mask_blur", 4))
        invert_mask = bool(params.get("invert_mask", False))
        width = int(params.get("width", 512))
        height = int(params.get("height", 512))
        seed = int(params.get("seed", -1))
        sampler = params.get("sampler", "LCM")
        model_path = params.get("model_path")
        vae_path = params.get("vae_path")

        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()

        if not model_path:
            models = self.scan_models()
            if models:
                comiccraft_match = [m["path"] for m in models.values() if "comiccraft" in m["name"].lower()]
                model_path = comiccraft_match[0] if comiccraft_match else list(models.values())[0]["path"]
            else:
                return {"success": False, "error": "No checkpoints found"}

        is_lcm = "lcm" in os.path.basename(model_path).lower()

        # Load or reuse base model
        if self.pipe is None or self.current_model_path != model_path or self.current_vae_path != vae_path:
            ok, msg = self.load_model(model_path, vae_path)
            if not ok:
                return {"success": False, "error": msg}

        if self.pipe_inpaint is None:
            print("[*] Creating StableDiffusionInpaintPipeline from loaded pipe...")
            try:
                self.pipe_inpaint = StableDiffusionInpaintPipeline.from_pipe(
                    self.pipe,
                    vae=self.pipe.vae,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            except Exception as pe:
                print(f"[!] from_pipe failed ({pe}), falling back to direct instantiation...")
                self.pipe_inpaint = StableDiffusionInpaintPipeline(
                    vae=self.pipe.vae,
                    text_encoder=self.pipe.text_encoder,
                    tokenizer=self.pipe.tokenizer,
                    unet=self.pipe.unet,
                    scheduler=self.pipe.scheduler,
                    safety_checker=None,
                    feature_extractor=None,
                    requires_safety_checker=False
                ).to(self.device)
            self.pipe_inpaint.enable_attention_slicing(1)
        else:
            self.pipe_inpaint.vae = self.pipe.vae

        inpaint_pipe = self.pipe_inpaint
        self.set_scheduler_for_pipe(inpaint_pipe, sampler, is_lcm=is_lcm)

        # Prepare images
        init_img = Image.open(init_image_path).convert("RGB")
        mask_img = Image.open(mask_image_path).convert("L")

        # Ensure dimensions are multiples of 8 for UNet/VAE
        width = (width // 8) * 8
        height = (height // 8) * 8

        # Resize to matching target dimensions
        init_img = init_img.resize((width, height), Image.Resampling.LANCZOS)
        mask_img = mask_img.resize((width, height), Image.Resampling.NEAREST)

        # Invert mask if requested
        if invert_mask:
            mask_img = ImageOps.invert(mask_img)

        # Apply mask blur if specified
        if mask_blur > 0:
            mask_img = mask_img.filter(ImageFilter.GaussianBlur(mask_blur))

        generator = torch.Generator(device=self.device).manual_seed(seed)

        self.is_interrupted = False
        start_time = time.time()
        self.progress_state = {
            "status": "generating",
            "step": 0,
            "total_steps": steps,
            "progress": 0.0,
            "eta": 0,
            "elapsed": 0.0,
            "info": "Inpainting processing..."
        }

        def step_callback(pipe, step_index, timestep, callback_kwargs):
            if self.is_interrupted:
                raise InterruptedError("Generation cancelled")
            elapsed = time.time() - start_time
            current_step = step_index + 1
            progress = current_step / steps
            rate = elapsed / current_step if current_step > 0 else 0
            remaining_steps = steps - current_step
            eta = round(remaining_steps * rate, 1)

            self.progress_state["step"] = current_step
            self.progress_state["total_steps"] = steps
            self.progress_state["progress"] = progress
            self.progress_state["eta"] = eta
            self.progress_state["elapsed"] = round(elapsed, 1)
            self.progress_state["info"] = f"Inpaint Step {current_step}/{steps} ({int(progress*100)}%)"
            return callback_kwargs

        prompt_embeds, neg_prompt_embeds = self.encode_prompt_long(inpaint_pipe, prompt, negative_prompt)

        try:
            with torch.inference_mode():
                result = inpaint_pipe(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=neg_prompt_embeds,
                    image=init_img,
                    mask_image=mask_img,
                    width=width,
                    height=height,
                    strength=strength,
                    num_inference_steps=steps,
                    guidance_scale=cfg_scale,
                    generator=generator,
                    callback_on_step_end=step_callback
                )

            image = result.images[0]
            total_time = round(time.time() - start_time, 2)

            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"inpaint_{timestamp}_{seed}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath, format="PNG")

            meta = {
                "type": "inpaint",
                "filename": filename,
                "filepath": filepath,
                "url": f"/outputs/{filename}",
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "strength": strength,
                "mask_blur": mask_blur,
                "sampler": sampler,
                "seed": seed,
                "width": width,
                "height": height,
                "model": os.path.basename(model_path),
                "model_path": model_path,
                "render_time": total_time,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            self.progress_state["status"] = "completed"
            self.progress_state["progress"] = 1.0
            self.progress_state["info"] = f"Finished in {total_time}s"
            return {"success": True, "image": meta}

        except InterruptedError:
            self.progress_state["status"] = "interrupted"
            self.progress_state["info"] = "Inpaint interrupted by user"
            return {"success": False, "interrupted": True, "error": "Inpaint interrupted"}
        except Exception as e:
            self.progress_state["status"] = "error"
            self.progress_state["info"] = f"Error: {e}"
            return {"success": False, "error": str(e)}
