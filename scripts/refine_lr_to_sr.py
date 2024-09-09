import torch
import os
import time

import numpy as np
from hashlib import md5
def hash_img(img):
    return md5(np.array(img).tobytes()).hexdigest()
def hash_any(obj):
    return md5(str(obj).encode()).hexdigest()



import torch
from PIL import Image
import numpy as np
import time
from diffusers import StableDiffusionPipeline, ControlNetModel
from perflow.src.scheduler_perflow import PeRFlowScheduler

def run_sr_sd(source_pils, scale=4, pipe=None, strength=0.35, prompt="best quality", neg_prompt="blur, lowres, bad anatomy, bad hands, cropped, worst quality", controlnet_conditioning_scale=1.0):
    global SD_cache

    if SD_cache is not None and pipe is None:
        pipe = SD_cache
    elif pipe is None:
        # Initialize the Stable Diffusion pipeline with the refiner

       
        controlnet = ControlNetModel.from_pretrained(
            "/root/Unique3D/perflow/control_v11f1e_sd15_tile/",
            torch_dtype=torch.float16
        )
        
        pipe = StableDiffusionPipeline.from_pretrained(
            "hansyan/perflow-sd15-dreamshaper",
            torch_dtype=torch.float16,
            custom_pipeline="stable_diffusion_controlnet_img2img",
            controlnet=controlnet,
        )
        
        pipe.scheduler = PeRFlowScheduler.from_config(
            pipe.scheduler.config, prediction_type="ddim_eps", num_time_windows=4
        )
        pipe = pipe.to("cuda")

    ret_pils = []
    start = time.time()

    for idx, img_pil in enumerate(source_pils):
        np_in = isinstance(img_pil, np.ndarray)
        if np_in:
            img_pil = Image.fromarray(img_pil)

        # Calculate new dimensions
        width, height = img_pil.size
        new_width, new_height = width * scale, height * scale

        # Resize the input image for conditioning
        condition_image = resize_for_condition_image(img_pil, max(new_width, new_height))

        # Upscale the image
        with torch.no_grad():
            output = pipe(
                prompt=prompt,
                negative_prompt=neg_prompt,
                image=condition_image,
                controlnet_conditioning_image=condition_image,
                width=condition_image.size[0],
                height=condition_image.size[1],
                strength=strength,
                num_inference_steps=4,
                generator=torch.manual_seed(233),
                controlnet_conditioning_scale=controlnet_conditioning_scale,
            ).images[0]

        if np_in:
            ret_pils.append(np.array(output))
        else:
            ret_pils.append(output)

    if SD_cache is None:
        SD_cache = pipe

    print(f"SD upscaling took {time.time() - start:.2f} seconds")
    return ret_pils

# Initialize the global cache
SD_cache = None

def resize_for_condition_image(input_image: Image.Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img

def refine_lr_with_sd(pil_image_list, concept_img_list, control_image_list, prompt_list, pipe=None, strength=0.35, neg_prompt_list="", output_size=(512, 512), controlnet_conditioning_scale=1.):
    print("Refining images using SD")
    start = time.time()
    with torch.no_grad():
        images = pipe(
            image=pil_image_list,
            ip_adapter_image=concept_img_list,
            prompt=prompt_list,
            neg_prompt=neg_prompt_list,
            num_inference_steps=50,
            strength=strength,
            height=output_size[0],
            width=output_size[1],
            control_image=control_image_list,
            guidance_scale=5.0,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=torch.manual_seed(233),
        ).images
    print(f"Refining took {time.time() - start:.2f} seconds")
    return images

SR_cache = None


import os
import subprocess
from PIL import Image
import numpy as np

# def run_sr_fast(source_pils, scale=4, model_name="realesrgan-x4plus"):
#     print("Running super resolution on images using RealESRGAN CLI")
    
#     # Path to the RealESRGAN binary
#     realesrgan_path = "/opt/realesrgan/realesrgan-ncnn-vulkan"
    
#     # Check if the binary exists
#     if not os.path.exists(realesrgan_path):
#         raise FileNotFoundError(f"RealESRGAN binary not found at {realesrgan_path}")
    
#     # Create directories
#     os.makedirs("/intermediate", exist_ok=True)
#     os.makedirs("/intermediate/output", exist_ok=True)
    
#     # Save input images
#     input_paths = []
#     for idx, img_pil in enumerate(source_pils):
#         if isinstance(img_pil, np.ndarray):
#             img_pil = Image.fromarray(img_pil)
#         input_path = f"/intermediate/sr_img_{idx}.png"
#         img_pil.save(input_path)
#         input_paths.append(input_path)
    
#     # Run RealESRGAN CLI
#     cli_command = [
#         realesrgan_path,
#         "-i", "/intermediate",
#         "-o", "/intermediate/output",
#         # "-n", model_name,
#         "-s", str(scale),
#         "-f", "png",
#         "-v"
#     ]
    
#     try:
#         subprocess.run(cli_command, check=True)
#     except subprocess.CalledProcessError as e:
#         print(f"Error running RealESRGAN CLI: {e}")
#         return None
    
#     # Load and return output images
#     ret_pils = []
#     for idx, _ in enumerate(source_pils):
#         output_path = f"/intermediate/output/sr_img_{idx}.png"
#         if os.path.exists(output_path):
#             output_img = Image.open(output_path)
#             if isinstance(source_pils[idx], np.ndarray):
#                 ret_pils.append(np.array(output_img))
#             else:
#                 ret_pils.append(output_img)
#         else:
#             print(f"Output file not found: {output_path}")
    
#     # Clean up intermediate files
#     for path in input_paths:
#         os.remove(path)
#     for path in [f"/intermediate/output/sr_img_{idx}.png" for idx, _ in enumerate(source_pils)]:
#         if os.path.exists(path):
#             os.remove(path)
    
#     return ret_pils

def run_sr_fast(source_pils, scale=4):
    from PIL import Image
    from scripts.upsampler import RealESRGANer
    import numpy as np
    global SR_cache
    if SR_cache is not None:
        upsampler = SR_cache
    else:
        upsampler = RealESRGANer(
            scale=4,
            onnx_path="ckpt/realesrgan-x4.onnx",
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True,
            gpu_id=0,
        )
    ret_pils = []
    for idx, img_pils in enumerate(source_pils):
        np_in = isinstance(img_pils, np.ndarray)
        assert isinstance(img_pils, (Image.Image, np.ndarray))
        img = np.array(img_pils)
        output, _ = upsampler.enhance(img, outscale=scale)
        if np_in:
            ret_pils.append(output)
        else:
            ret_pils.append(Image.fromarray(output))
    if SR_cache is None:
        SR_cache = upsampler
    return ret_pils