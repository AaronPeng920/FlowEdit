import cv2
import os
from typing import Any, Callable, Dict, List, Optional, Union
import re
import functools
import pandas as pd
from tqdm import tqdm
import torch
from pipeline_flowedit import FlowEditPipeline

def convert_attention_maps_to_video(
    attention_maps_image_folder: str,
    output_videos_folder: Optional[str] = 'inters/attentions_dynamic',
    read_flag: Optional[str] = "gray",
    fps: Optional[int] = 30,
    by_layer: Optional[bool] = True
):
    """Converting saved attention maps to videos according to layer and mode
    
    Args:
        attention_maps_image_folder: str, attention maps image saved folder
        read_flag: str, 
                * `gray` for gray scale 
                * `color` for color images that ignore the alpha channel
                * `unchanged` for complete image with full alpha channel
                
    Return:
        output_video_names: List[str], saved video path
        output_video_count: int, saved video count
    """
    if read_flag == 'gray':
        flag = cv2.IMREAD_GRAYSCALE
    elif read_flag == 'color':
        flag = cv2.IMREAD_COLOR
    elif read_flag == 'unchanged':
        flag = cv2.IMREAD_UNCHANGED
    else:
        raise ValueError(f"Unsupported read flag of `{read_flag}`.")
    
    # 1. List all image filenames
    fnames = os.listdir(attention_maps_image_folder)
    
    # 2. Sorting
    sorted_fnames = sort_attention_map_inters(fnames, by_layer=by_layer)
    
    # 3. Grouping
    grouped_fnames = group_attention_map_inters(sorted_fnames, by_layer=by_layer)
    
    # 4. Making Video
    output_video_names = []
    for name, image_filenames in tqdm(grouped_fnames.items(), desc="Making Videos"):
        if len(image_filenames) == 0:
            continue
        saved_video_name = os.path.join(output_videos_folder, f"{name}.mp4")
        shape = cv2.imread(os.path.join(attention_maps_image_folder, image_filenames[0]), flag).shape 
        height, width = shape[0], shape[1]
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(saved_video_name, fourcc, fps, (width, height))
        
        for image_filename in image_filenames:
            image_filename = os.path.join(attention_maps_image_folder, image_filename)
            frame = cv2.imread(image_filename)
            video.write(frame)
            
        video.release()
        output_video_names.append(saved_video_name)
    
    return output_video_names, len(output_video_names)

def sort_attention_map_inters(f_names: List[str], by_layer: bool=False):
    def attention_map_inters_sort(f_name1: str, f_name2: str):
        """Sorting Saved attention maps, `select` mode has higher priority than others, 
            then sort in ascending order according to
                1. `mode_type(str)` or `token_id(int)`
                2. `layer_id(int)` if `layer_first = True` otherwise `step_id(int)`
                3. `step_id(int)` if `layer_first = False` otherwise `layer_id(int)`

        Args:
            f_name1: Named in the format of 
                        * `layer{layer_id}_{mode}_{token_id}th_token_step_{step_id}.ext` if `{mode}` is 'select', or
                        * `layer{layer_id}_{mode}_step_{step_id}.ext` otherwise.
            f_name2: Same as above
        """
        if 'token' in f_name1:
            select_mode1 = True
        else:
            select_mode1 = False
        if 'token' in f_name2:
            select_mode2 = True
        else:
            select_mode2 = False
        
        # Different mode
        if select_mode1 and not select_mode2:
            return -1
        if not select_mode1 and select_mode2:
            return 1
        
        # Same mode
        mode = "select" if select_mode1 else "other"

        if mode == "select":
            re_template = re.compile(r"layer(\d+)_select_(\d+)th_token_step_(\d+)\.*")
        else:
            re_template = re.compile(r"layer(\d+)_([a-zA-Z0-9]+)_step_(\d+)\.*")

        match1 = re_template.search(f_name1)
        if match1:
            layer_id1 = eval(match1.group(1))
            token_or_mode_id1 = eval(match1.group(2)) if mode == "select" else match1.group(2)
            step_id1 = eval(match1.group(3))
        else:
            raise ValueError(f"An incorrectly formatted file name of `{f_name1}` was found.")
        match2 = re_template.search(f_name2)
        if match2:
            layer_id2 = eval(match2.group(1))
            token_or_mode_id2 = eval(match2.group(2)) if mode == "select" else match2.group(2)
            step_id2 = eval(match2.group(3))
        else:
            raise ValueError(f"An incorrectly formatted file name of `{f_name2}` was found.")

        if token_or_mode_id1 < token_or_mode_id2:
            return -1
        elif token_or_mode_id1 > token_or_mode_id2:
            return 1
        else:
            if by_layer:
                if layer_id1 < layer_id2:
                    return -1
                elif layer_id1 > layer_id2:
                    return 1
                else:
                    if step_id1 < step_id2:
                        return -1
                    elif step_id1 > step_id2:
                        return 1
                    else:
                        return 0
            else:
                if step_id1 < step_id2:
                    return -1
                elif step_id1 > step_id2:
                    return 1
                else:
                    if layer_id1 < layer_id2:
                        return -1
                    elif layer_id1 > layer_id2:
                        return 1
                    else:
                        return 0
        
    sorted_fnames = sorted(f_names, key=functools.cmp_to_key(attention_map_inters_sort))
    return sorted_fnames


def group_attention_map_inters(f_names: List[str], by_layer: bool=False):
    """Grouping Saved attention map with layer_id and mode
    
    Args:
        f_names: List[str], ensuring that it has sorted by `sort_attention_map_inters()` method 
        
    Return:
        grouped_attention_map_inters: Dict[str: List[str]], group name `layer?_mode` is the key, and list of filenames is the value.
    """
    df = pd.DataFrame({
        "layer_id": pd.Series(dtype='int'),
        "mode": pd.Series(dtype='str'),
        "step_id": pd.Series(dtype='int'),
        "filename": pd.Series(dtype='str')
    })

    for f_name in f_names:
        # `select` mode
        if "token" in f_name:
            re_template = re.compile(r"layer(\d+)_select_(\d+)th_token_step_(\d+)\.*")
            match = re_template.search(f_name)
            if match:
                layer_id = eval(match.group(1))
                mode = match.group(2)
                step_id = eval(match.group(3))
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame([{"layer_id": layer_id, "mode": "select" + mode, "step_id":step_id, "filename": f_name}])
                    ], ignore_index=True
                )
            else:
                raise ValueError(f"An incorrectly `select` mode formatted file name of `{f_name}` was found.")
        # other mode
        else:
            re_template = re.compile(r"layer(\d+)_([a-zA-Z0-9]+)_step_(\d+)\.*")
            match = re_template.search(f_name)
            if match:
                layer_id = eval(match.group(1))
                mode = match.group(2)
                step_id = eval(match.group(3))
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame([{"layer_id": layer_id, "mode": mode, "step_id":step_id, "filename": f_name}])
                    ], ignore_index=True
                )
            else:
                raise ValueError(f"An incorrectly `{mode}` mode formatted file name of `{f_name}` was found.")
    
    if by_layer:
        grouped_df = df.groupby(["layer_id", "mode"])
    else:
        grouped_df = df.groupby(["step_id", "mode"])

    grouped_attention_map_inters = {}
    for name, grouped_data in grouped_df:
        name = f"layer{name[0]}_{name[1]}" if by_layer else f"step{name[0]}_{name[1]}"
        grouped_data = grouped_data['filename'].values.tolist()
        grouped_attention_map_inters[name] = grouped_data
        
    return grouped_attention_map_inters
        
def save_inter_latents_callback(pipe, step_i, timestep, local_kwargs):
    """Saving inter latents of `FlowEditPipeline` at each sampling step."""
    latents = local_kwargs.get("latents", None)
    source_latents = local_kwargs.get("source_latents", None)
             
    if latents is not None:
        scaled_latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        image = pipe.vae.decode(scaled_latents, return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type="pil")[0]
        image.save(f"inters/inter_latents/latents_step_{step_i}.png")
    if source_latents is not None:
        source_scaled_latents = (source_latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        source_image = pipe.vae.decode(source_scaled_latents, return_dict=False)[0]
        source_image = pipe.image_processor.postprocess(source_image, output_type="pil")[0]
        source_image.save(f"inters/inter_latents/source_latents_step_{step_i}.png")
    return local_kwargs

def generate_mask(
    dtype, device,
    mask_type: Optional[str] = "",
    image_resolution: Optional[int] = 512,
    has_text_encoder_3: Optional[bool] = False
):
    """Generating masks according to `mask_type`
    
    Args:
        mask_type: Attention region name in Attention layer
    
    Return:
        masks: torch.Tensor of shape `[L, L]`, `L` is latent resolution
    """
    if mask_type == "":
        return None
    
    clip_l = 77
    t5_l = 256 if has_text_encoder_3 else 77
    latent_l = (image_resolution // 8 // 2) ** 2
    resolution = latent_l + clip_l + t5_l    
    
    shape = (
        resolution,
        resolution
    )
    if mask_type == "all":
        return torch.ones(*shape, dtype=dtype).to(device) * -1 * torch.inf
    
    masks = torch.zeros(*shape, dtype=dtype).to(device)
    mask_types = mask_type.split('+')
    for mask_type in mask_types:
        if mask_type == 'latent2latent':
            masks[:latent_l, :latent_l] = -torch.inf
        elif mask_type == 'latent2clip':
            masks[:latent_l, latent_l:latent_l+clip_l] = -torch.inf
        elif mask_type == 'latent2t5':
            masks[:latent_l, latent_l+clip_l:] = -torch.inf
        elif mask_type == 'clip2latent':
            masks[latent_l:latent_l+clip_l, :latent_l] = -torch.inf
        elif mask_type == 'clip2clip':
            masks[latent_l:latent_l+clip_l, latent_l:latent_l+clip_l] = -torch.inf
        elif mask_type == 'clip2t5':
            masks[latent_l:latent_l+clip_l, latent_l+clip_l:] = -torch.inf
        elif mask_type == 't52latent':
            masks[latent_l+clip_l:, :latent_l] = -torch.inf
        elif mask_type == 't52clip':
            masks[latent_l+clip_l:, latent_l:latent_l+clip_l] = -torch.inf
        elif mask_type == 't52t5':
            masks[latent_l+clip_l:, latent_l+clip_l:] = -torch.inf
        else:
            raise ValueError(f"Unsupprted mask type of `{mask_type}`.")
    return masks
    
    
if __name__ == '__main__':
    saved_video_names, count = convert_attention_maps_to_video(
        "/home/pengzhengwei/projects/FlowEdit/inters/attentions",
        fps=1,
        by_layer=False
    )
    print(count)
