import torch
from scripts.sd_model_zoo import load_common_sd15_pipe
from diffusers import StableDiffusionControlNetImg2ImgPipeline, StableDiffusionPipeline
from diffusers import ControlNetModel, StableDiffusionPipeline
from perflow.src.scheduler_perflow import PeRFlowScheduler



class MyModelZoo:
    _pipe_disney_controlnet_lineart_ipadapter_i2i: StableDiffusionControlNetImg2ImgPipeline = None
    _perflow_refine_model = None
    
    # base_model = "runwayml/stable-diffusion-v1-5"
    base_model = "pt-sk/stable-diffusion-1.5"

    def __init__(self, base_model=None) -> None:
        if base_model is not None:
            self.base_model = base_model

    @property
    def pipe_disney_controlnet_tile_ipadapter_i2i(self):
        return self._pipe_disney_controlnet_lineart_ipadapter_i2i
    
    @property
    def perflow_pipe_refine_model(self):
        return self._perflow_refine_model

    def load_perflow_refine_model(self):

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


    
    
    def init_models(self, load_models_onto_gpu=True, enable_sequential_cpu_offload=False):
        self._pipe_disney_controlnet_lineart_ipadapter_i2i = load_common_sd15_pipe(base_model=self.base_model, ip_adapter=True, plus_model=False, controlnet="./ckpt/controlnet-tile",load_model_onto_gpu=load_models_onto_gpu, pipeline_class=StableDiffusionControlNetImg2ImgPipeline, enable_sequential_cpu_offload=enable_sequential_cpu_offload)
        self.load_perflow_refine_model()

model_zoo = MyModelZoo()
