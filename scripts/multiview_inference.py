import os
from PIL import Image
from scripts.mesh_init import build_mesh, calc_w_over_h, fix_border_with_pymeshlab_fast
from scripts.project_mesh import multiview_color_projection
from scripts.refine_lr_to_sr import run_sr_fast, run_sr_sd
from scripts.utils import simple_clean_mesh
from app.utils import simple_remove, split_image
from app.custom_models.normal_prediction import predict_normals
from mesh_reconstruction.recon import reconstruct_stage1
from mesh_reconstruction.refine import run_mesh_refine
from scripts.project_mesh import get_cameras_list
from scripts.utils import from_py3d_mesh, to_pyml_mesh
from pytorch3d.structures import Meshes, join_meshes_as_scene
import numpy as np

def fast_geo(front_normal: Image.Image, back_normal: Image.Image, side_normal: Image.Image, clamp=0., init_type="std"):
    import time
    if front_normal.mode == "RGB":
        front_normal = simple_remove(front_normal, run_sr=False)
    front_normal = front_normal.resize((192, 192))
    if back_normal.mode == "RGB":
        back_normal = simple_remove(back_normal, run_sr=False)
    back_normal = back_normal.resize((192, 192))
    if side_normal.mode == "RGB":
        side_normal = simple_remove(side_normal, run_sr=False)
    side_normal = side_normal.resize((192, 192))
    
    # build mesh with front back projection # ~3s
    side_w_over_h = calc_w_over_h(side_normal)
    mesh_front = build_mesh(front_normal, front_normal, clamp_min=clamp, scale=side_w_over_h, init_type=init_type)
    mesh_back = build_mesh(back_normal, back_normal, is_back=True, clamp_min=clamp, scale=side_w_over_h, init_type=init_type)
    meshes = join_meshes_as_scene([mesh_front, mesh_back])
    meshes = fix_border_with_pymeshlab_fast(meshes, poissson_depth=6, simplification=2000)
    return meshes

def refine_rgb(rgb_pils, front_pil):
    from scripts.refine_lr_to_sr import refine_lr_with_sd
    from scripts.utils import NEG_PROMPT
    from app.utils import make_image_grid
    from app.all_models import model_zoo
    from app.utils import rgba_to_rgb
    rgb_pil = make_image_grid(rgb_pils, rows=2)
    prompt = "4views, multiview"
    neg_prompt = NEG_PROMPT
    control_image = rgb_pil.resize((1024, 1024))
    pipe = model_zoo.pipe_disney_controlnet_tile_ipadapter_i2i.to("cuda")
    refined_rgb = refine_lr_with_sd([rgb_pil], [rgba_to_rgb(front_pil)], [control_image], prompt_list=[prompt], neg_prompt_list=[neg_prompt], pipe=pipe, strength=0.2, output_size=(1024, 1024))[0]
    refined_rgbs = split_image(refined_rgb, rows=2)
    return refined_rgbs

def erode_alpha(img_list):
    out_img_list = []
    for idx, img in enumerate(img_list):
        arr = np.array(img)
        alpha = (arr[:, :, 3] > 127).astype(np.uint8)
        # erode 1px
        import cv2
        alpha = cv2.erode(alpha, np.ones((3, 3), np.uint8), iterations=1)
        alpha = (alpha * 255).astype(np.uint8)
        img = Image.fromarray(np.concatenate([arr[:, :, :3], alpha[:, :, None]], axis=-1))
        out_img_list.append(img)
    return out_img_list
import time
def geo_reconstruct(rgb_pils, normal_pils, front_pil, do_refine=False, predict_normal=True, expansion_weight=0.1, init_type="std"):
    import os
    os.makedirs("/intermediate", exist_ok=True)

    start_time = time.time()
    print(f"Input front_pil shape: {front_pil.size}")
    if front_pil.size[0] <= 512:
        front_pil = run_sr_fast([front_pil])[0]
        print(f"After SR, front_pil shape: {front_pil.size}")
        # front_pil.save("/intermediate/front_pil_sr.png")
    if do_refine:
        print("Refining RGB images...")
        refine_start_time = time.time()
        refined_rgbs = refine_rgb(rgb_pils, front_pil)  # 6s
        print(f"Input rgb_pils shapes: {[rgb.size for rgb in rgb_pils]}")
        print(f"Refined rgb_pils shapes: {[rgb.size for rgb in refined_rgbs]}")
        # for i, rgb in enumerate(refined_rgbs):
            # rgb.save(f"/intermediate/refined_rgb_{i}.png")
        refine_end_time = time.time()
        print(f"Refining RGB images took {refine_end_time - refine_start_time:.2f} seconds")
    else:
        refined_rgbs = [rgb.resize((512, 512), resample=Image.LANCZOS) for rgb in rgb_pils]
        print(f"Input rgb_pils shapes: {[rgb.size for rgb in rgb_pils]}")
        print(f"Resized rgb_pils shapes: {[rgb.size for rgb in refined_rgbs]}")
        # for i, rgb in enumerate(refined_rgbs):
        #     rgb.save(f"/intermediate/resized_rgb_{i}.png")
    # img_list = [front_pil] + run_sr_fast(refined_rgbs[1:])
    img_list = [front_pil] + run_sr_sd(refined_rgbs[1:])
    print(f"img_list shapes: {[img.size for img in img_list]}")
    # for i, img in enumerate(img_list):
    #     img.save(f"/intermediate/img_list_{i}.png")
    
    if predict_normal:
        print("Predicting normals...")
        predict_normal_start_time = time.time()
        rm_normals = predict_normals([img.resize((512, 512), resample=Image.LANCZOS) for img in img_list], guidance_scale=1.5)
        print(f"Input img_list shapes for normal prediction: {[img.size for img in img_list]}")
        print(f"Predicted rm_normals shapes: {[img.size for img in rm_normals]}")
        # for i, img in enumerate(rm_normals):
        #     img.save(f"/intermediate/rm_normal_{i}.png")
        predict_normal_end_time = time.time()
        print(f"Predicting normals took {predict_normal_end_time - predict_normal_start_time:.2f} seconds")
    else:
        rm_normals = simple_remove([img.resize((512, 512), resample=Image.LANCZOS) for img in normal_pils])
        # for i, img in enumerate(rm_normals):
        #     img.save(f"/intermediate/simple_remove_{i}.png")
    # transfer the alpha channel of rm_normals to img_list
    for idx, img in enumerate(rm_normals):
        if idx == 0 and img_list[0].mode == "RGBA":
            temp = img_list[0].resize((2048, 2048))
            rm_normals[0] = Image.fromarray(np.concatenate([np.array(rm_normals[0])[:, :, :3], np.array(temp)[:, :, 3:4]], axis=-1))
            # rm_normals[0].save("/intermediate/rm_normal_0_alpha.png")
            continue
        img_list[idx] = Image.fromarray(np.concatenate([np.array(img_list[idx]), np.array(img)[:, :, 3:4]], axis=-1))
        # img_list[idx].save(f"/intermediate/img_list_{idx}_alpha.png")
    assert img_list[0].mode == "RGBA"
    assert np.mean(np.array(img_list[0])[..., 3]) < 250
    
    img_list = [img_list[0]] + erode_alpha(img_list[1:])
    normal_stg1 = [img.resize((512, 512)) for img in rm_normals]
    if init_type in ["std", "thin"]:
        print("Initializing mesh...")
        init_mesh_start_time = time.time()
        meshes = fast_geo(normal_stg1[0], normal_stg1[2], normal_stg1[1], init_type=init_type)
        init_mesh_end_time = time.time()
        print(f"Initializing mesh took {init_mesh_end_time - init_mesh_start_time:.2f} seconds")
        _ = multiview_color_projection(meshes, rgb_pils, resolution=512, device="cuda", complete_unseen=False, confidence_threshold=0.1)    # just check for validation, may throw error
        vertices, faces, _ = from_py3d_mesh(meshes)
        print("Reconstructing stage 1...")
        reconstruct_stage1_start_time = time.time()
        vertices, faces = reconstruct_stage1(normal_stg1, steps=200, vertices=vertices, faces=faces, start_edge_len=0.1, end_edge_len=0.02, gain=0.05, return_mesh=False, loss_expansion_weight=expansion_weight)
        reconstruct_stage1_end_time = time.time()
        print(f"Reconstructing stage 1 took {reconstruct_stage1_end_time - reconstruct_stage1_start_time:.2f} seconds")
    elif init_type in ["ball"]:
        print("Reconstructing stage 1...")
        reconstruct_stage1_start_time = time.time()
        vertices, faces = reconstruct_stage1(normal_stg1, steps=200, end_edge_len=0.01, return_mesh=False, loss_expansion_weight=expansion_weight)
        reconstruct_stage1_end_time = time.time()
        print(f"Reconstructing stage 1 took {reconstruct_stage1_end_time - reconstruct_stage1_start_time:.2f} seconds")
    print("Refining mesh...")
    refine_mesh_start_time = time.time()
    vertices, faces = run_mesh_refine(vertices, faces, rm_normals, steps=100, start_edge_len=0.02, end_edge_len=0.005, decay=0.99, update_normal_interval=20, update_warmup=5, return_mesh=False, process_inputs=False, process_outputs=False)
    refine_mesh_end_time = time.time()
    print(f"Refining mesh took {refine_mesh_end_time - refine_mesh_start_time:.2f} seconds")
    print("Cleaning mesh...")
    clean_mesh_start_time = time.time()
    meshes = simple_clean_mesh(to_pyml_mesh(vertices, faces), apply_smooth=True, stepsmoothnum=1, apply_sub_divide=True, sub_divide_threshold=0.25).to("cuda")
    clean_mesh_end_time = time.time()
    print(f"Cleaning mesh took {clean_mesh_end_time - clean_mesh_start_time:.2f} seconds")
    print("Projecting color...")
    project_color_start_time = time.time()
    new_meshes = multiview_color_projection(meshes, img_list, resolution=1024, device="cuda", complete_unseen=True, confidence_threshold=0.2, cameras_list = get_cameras_list([0, 90, 180, 270], "cuda", focal=1))
    project_color_end_time = time.time()
    print(f"Projecting color took {project_color_end_time - project_color_start_time:.2f} seconds")
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    return new_meshes
