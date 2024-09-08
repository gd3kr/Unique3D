import torch
import os

import numpy as np
from hashlib import md5
def hash_img(img):
    return md5(np.array(img).tobytes()).hexdigest()
def hash_any(obj):
    return md5(str(obj).encode()).hexdigest()

def refine_lr_with_sd(pil_image_list, concept_img_list, control_image_list, prompt_list, pipe=None, strength=0.35, neg_prompt_list="", output_size=(512, 512), controlnet_conditioning_scale=1.):
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
    import numpy as np
    from PIL import Image


    print("running super resolution on images")
    # save images to disk
    for idx, img_pils in enumerate(source_pils):
        np_in = isinstance(img_pils, np.ndarray)
        assert isinstance(img_pils, (Image.Image, np.ndarray))
        img = np.array(img_pils)
        if np_in:
            img_pils = Image.fromarray(img)
        try:
            os.makedirs("/intermediate", exist_ok=True)
        except:
            pass
        img_pils.save(f"/intermediate/sr_img_{idx}.png")
    from scripts.upsampler import RealESRGANer
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
