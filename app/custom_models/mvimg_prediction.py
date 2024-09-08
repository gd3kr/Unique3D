import sys
import torch
import gradio as gr
from PIL import Image
import numpy as np
from rembg import remove
from app.utils import change_rgba_bg, rgba_to_rgb
from app.custom_models.utils import load_pipeline
from scripts.all_typing import *
from scripts.utils import session, simple_preprocess

# from perflow.src.utils_perflow import merge_delta_weights_into_unet
from perflow.src.scheduler_perflow import PeRFlowScheduler
from diffusers import UNet2DConditionModel



def merge_delta_weights_into_unet(pipe, delta_weights):
    unet_weights = pipe.unet.state_dict()
    
    # Check for missing keys
    missing_keys = set(delta_weights.keys()) - set(unet_weights.keys())
    if missing_keys:
        print(f"Warning: Keys in delta_weights but not in unet_weights: {missing_keys}")
    
    # Check for extra keys in unet_weights
    extra_keys = set(unet_weights.keys()) - set(delta_weights.keys())
    if extra_keys:
        print(f"Warning: Keys in unet_weights but not in delta_weights: {extra_keys}")
    
    for key in delta_weights.keys():
        if key not in unet_weights:
            print(f"Skipping key {key} as it's not in unet_weights")
            continue
        
        print(f"Processing key: {key}")
        print(f"unet_weights shape: {unet_weights[key].shape}")
        print(f"delta_weights shape: {delta_weights[key].shape}")
        
        dtype = unet_weights[key].dtype
        try:
            unet_weights[key] = (
                unet_weights[key].to(dtype=delta_weights[key].dtype) + 
                delta_weights[key].to(device=unet_weights[key].device)
            )
            unet_weights[key] = unet_weights[key].to(dtype)
        except RuntimeError as e:
            print(f"Error occurred for key: {key}")
            print(f"unet_weights dtype: {unet_weights[key].dtype}")
            print(f"delta_weights dtype: {delta_weights[key].dtype}")
            print(f"Error message: {str(e)}")
            raise
    
    try:
        pipe.unet.load_state_dict(unet_weights, strict=True)
    except RuntimeError as e:
        print(f"Error loading state dict: {str(e)}")
        raise
    
    return pipe

training_config = "app/custom_models/image2mvimage.yaml"
checkpoint_path = "ckpt/img2mvimg/unet_state_dict.pth"
trainer, pipeline = load_pipeline(training_config, checkpoint_path)

# merge delta weights into unet
delta_weights = UNet2DConditionModel.from_pretrained("hansyan/perflow-sd15-delta-weights", torch_dtype=torch.bfloat16, variant="v0-1",).state_dict()
pipeline = merge_delta_weights_into_unet(pipeline, delta_weights)


pipeline.scheduler = PeRFlowScheduler.from_config(pipeline.scheduler.config, prediction_type="diff_eps", num_time_windows=4)



# pipeline.enable_model_cpu_offload()

def predict(img_list: List[Image.Image], guidance_scale=2., **kwargs):
    if isinstance(img_list, Image.Image):
        img_list = [img_list]
    img_list = [rgba_to_rgb(i) if i.mode == 'RGBA' else i for i in img_list]
    ret = []
    for img in img_list:
        images = trainer.pipeline_forward(
            pipeline=pipeline,
            image=img,
            guidance_scale=guidance_scale, 
            **kwargs
        ).images
        ret.extend(images)
    return ret


def run_mvprediction(input_image: Image.Image, remove_bg=True, guidance_scale=1.5, seed=1145):
    if input_image.mode == 'RGB' or np.array(input_image)[..., -1].mean() == 255.:
        # still do remove using rembg, since simple_preprocess requires RGBA image
        print("RGB image not RGBA! still remove bg!")
        remove_bg = True

    if remove_bg:
        input_image = remove(input_image, session=session)

    # make front_pil RGBA with white bg
    input_image = change_rgba_bg(input_image, "white")
    single_image = simple_preprocess(input_image)

    generator = torch.Generator(device="cuda").manual_seed(int(seed)) if seed >= 0 else None

    rgb_pils = predict(
        single_image,
        generator=generator,
        guidance_scale=guidance_scale,
        width=256,
        height=256,
        num_inference_steps=30,
    )

    return rgb_pils, single_image
