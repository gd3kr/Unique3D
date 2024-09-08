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

from perflow.src.utils_perflow import merge_delta_weights_into_unet
from perflow.src.scheduler_perflow import PeRFlowScheduler
from diffusers import UNet2DConditionModel



training_config = "app/custom_models/image2mvimage.yaml"
checkpoint_path = "ckpt/img2mvimg/unet_state_dict.pth"
trainer, pipeline = load_pipeline(training_config, checkpoint_path)

# merge delta weights into unet
delta_weights = UNet2DConditionModel.from_pretrained("hansyan/perflow-sd15-delta-weights", torch_dtype=torch.bfloat16, variant="v0-1",).state_dict()
# pipeline = merge_delta_weights_into_unet(pipeline, delta_weights)

unet_weights = pipeline.unet.state_dict()
# assert unet_weights.keys() == delta_weights.keys()
for key in delta_weights.keys():
    print(key)
    dtype = unet_weights[key].dtype
    unet_weights[key] = unet_weights[key].to(dtype=delta_weights[key].dtype) + delta_weights[key].to(device=unet_weights[key].device)
    unet_weights[key] = unet_weights[key].to(dtype)
pipeline.unet.load_state_dict(unet_weights, strict=True)

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
