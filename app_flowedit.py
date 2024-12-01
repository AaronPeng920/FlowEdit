import warnings
import gradio as gr
import torch
from pipeline_flowedit import FlowEditPipeline
from attention_processor import register_attention_processor, visualize_attention
import seq_aligner
import abc
import logging
from typing import Any, Callable, Dict, List, Optional, Union
from sklearn.decomposition import PCA
from PIL import Image
import numpy as np
import torch.nn.functional as nnf
import time
from utils import combine_images_with_captions, save_inter_latents_callback, parse_string_to_processor_id
import yaml
import json
import os
import argparse

# Copied and revised from https://github.com/sled-group/InfEdit/blob/main/app_infedit/LocalBlend
class LocalBlend:
    def __init__(
        self, 
        source_thresh: Optional[float] = 0.0,
        target_thresh: Optional[float] = 0.0,
        save_blend_mask: Optional[bool] = False,
        num_steps: Optional[int] = 10,
        start_step: Optional[float] = 0.0,
        end_step: Optional[float] = 0.0
    ):
        self.source_thresh = source_thresh
        self.target_thresh = target_thresh
        self.save_blend_mask = save_blend_mask
        self.start_step = start_step
        self.end_step = end_step
        self.num_steps = num_steps
        self.cur_step = 0
    
    def reset(self):
        self.cur_step = 0
        
    def set_blend(
        self,
        alphas: torch.Tensor,
        target_blend_alphas: torch.Tensor,
        source_blend_alphas: torch.Tensor,
    ):
        self.alphas = alphas
        self.target_blend_alphas = target_blend_alphas
        self.source_blend_alphas = source_blend_alphas
        target_source_blend_alphas = target_blend_alphas.to(torch.bool) & source_blend_alphas.to(torch.bool)
        self.target_source_blend_alphas = target_source_blend_alphas.to(torch.float)
        
    def __get_mask(
        self, 
        attn_map: torch.Tensor,
        word_idxs: torch.Tensor,
        thresh: Optional[float] = 0.0
    ):
        """Extracting attention maps with word_idx and apply thresh to get mask. 
        
        Args:
            attn_map: torch.Tensor of shape `[b, n_layers, L, S]`
            word_idxs: torch.Tensor of shape `[b, L]`
            thresh: float
            
        Return:
            mask: torch.Tensor of shape `[b, 1, 2s, 2s]`, s is equal to S.sqrt()
        """
        bs, n_layers, L, S = attn_map.shape
        s = int(S ** 0.5)
        
        if word_idxs.sum() == 0:
            maps = torch.zeros(bs, 1, 2*s, 2*s).to(attn_map.device, attn_map.dtype)     # zeroTensor, [b, 1, 2s, 2s]
            mask = maps > 0                                                             # boolTensor (False), [b, 1, 2s, 2s]
        else:
            word_idxs = word_idxs.squeeze(dim=0)                            # [L]
            word_idxs = word_idxs > 0                                       # boolTensor, [L]
            extract_maps = attn_map[:, :, word_idxs, :]                     # [b, n_layers, M, S], M is the number of True in word_idxs
            extract_maps = torch.mean(extract_maps, dim=1)                  # [b, M, S]
            extract_maps = torch.mean(extract_maps, dim=1)                  # [b, S]
            maps = extract_maps.reshape(extract_maps.shape[0], 1, s, s)     # [b, 1, s, s]
            maps = nnf.interpolate(maps, size=(2*s, 2*s))                   # [b, 1, 2s, 2s]
            maps = (maps - maps.min()) / (maps.max() - maps.min())          # normalization, [b, 1, 2s, 2s]
            mask = maps > thresh                                            # boolTensor, [b, 1, 2s, 2s]
        return mask, maps


    def __save_blend_mask(self, blend_mask: torch.Tensor, save_name: str):
        blend_mask = blend_mask.squeeze().cpu().numpy()
        blend_mask_gray = (blend_mask * 255).astype(np.uint8)                               # 0 for black, 255 for white
        blend_mask = Image.fromarray(blend_mask_gray, mode='L')  
        blend_mask.save(save_name)
        
    def __call__(
        self,
        step: int,
        target_latent: torch.Tensor,
        source_latent: torch.Tensor,
        attn_store: List[torch.Tensor],
        sigma
    ):
        """Local Blend
        
        Args:
            target_latent: torch.Tensor of shape `[2b, C, s, s]`
            source_latent: torch.Tensor of shape `[2b, C, s, s]`
            attn_store: List[torch.Tensor] of shape `n_layer * [2b, L, S]`, n_layer is processed layers count
        """
        cur_progress = (self.cur_step + 1) * 1.0 / self.num_steps
        if cur_progress >= self.start_step and cur_progress <= self.end_step:
            attn_maps = torch.stack(attn_store, dim=1)                          # [2b, n_layers, L, S]
            target_attn_maps, source_attn_maps = attn_maps.chunk(2, dim=0)      # 2 * [b, n_layers, L, S]
            
            source_blend_mask, source_blend_maps = self.__get_mask(
                source_attn_maps, 
                self.source_blend_alphas, 
                self.source_thresh
            )       # [b, 1, s, s]
            target_blend_mask, target_blend_maps = self.__get_mask(
                target_attn_maps, 
                self.target_blend_alphas, 
                self.target_thresh
            )      # [b, 1, s, s]

            if self.save_blend_mask:
                self.__save_blend_mask(source_blend_mask[0], f"inters/blend_masks/source_blend_mask_step_{step}.png")
                self.__save_blend_mask(target_blend_mask[0], f"inters/blend_masks/target_blend_mask_step_{step}.png")
            if self.target_blend_alphas.sum() != 0:
                blended_latent = torch.where(target_blend_mask, target_latent, source_latent)
            else:
                blended_latent = target_latent
            blended_latent = torch.where(source_blend_mask, source_latent, blended_latent)
            target_latent = blended_latent
        self.cur_step += 1
        return target_latent

# Copied and revised from https://github.com/sled-group/InfEdit/blob/main/app_infedit/AttentionControl
class AttentionControl(abc.ABC):
    def step_callback(self, z_t):
        return z_t

    def between_steps(self, layer_id: int):
        return

    @abc.abstractmethod
    def forward(self, layer_id: int, attn_weight: torch.Tensor, value: torch.Tensor):
        raise NotImplementedError
    
    @abc.abstractmethod
    def process_qkv(self, layer_id: int, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        raise NotImplementedError

    def __call__(self, layer_id: int, attn_weight: torch.Tensor, value: torch.Tensor):
        if self.do_classifier_free_guidance:
            bs = attn_weight.shape[0]
            attn_weight[bs // 2:], value[bs // 2:] = self.forward(layer_id, attn_weight[bs // 2:], value[bs // 2:])
        else:
            attn_weight, value = self.forward(layer_id, attn_weight, value)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.between_steps(layer_id)
            self.cur_att_layer = 0
            self.cur_step += 1
            
        return attn_weight, value

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(
        self, 
        num_inference_steps: int, 
        do_classifier_free_guidance: bool, 
        device: str
    ):
        self.cur_step = 0
        self.num_att_layers = -1
        self.registered_layer_ids = []
        self.cur_att_layer = 0
        self.num_inference_steps = num_inference_steps
        self.do_classifier_free_guidance = do_classifier_free_guidance
        self.device = device

# Copied and revised from https://github.com/sled-group/InfEdit/blob/main/app_infedit/EmptyControl
class EmptyControl(AttentionControl):
    def forward(self, layer_id: int, attn_weight: torch.Tensor, value: torch.Tensor):
        return attn_weight, value
    
    def process_qkv(self, layer_id: int, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        return query, key, value

# Copied and revised from https://github.com/sled-group/InfEdit/blob/main/app_infedit/AttentionStore
class AttentionStore(AttentionControl):
    def __init__(
        self, 
        store_enable_layers: list,
        store_start_step: float,
        store_end_step: float,
        num_inference_steps: int, 
        image_resolution: int,
        has_text_encoder_3: bool,
        visualize_now: bool,
        do_classifier_free_guidance: bool,
        device: str,
        store_mode: Optional[str] = None,
        **kwargs
    ):
        assert store_mode in [
            None, "all", "select",
            "latent2latent", "latent2clip", "latent2t5",
            "clip2latent", "clip2clip", "clip2t5",
            "t52latent", "t52clip", "t52t5"
        ]
        
        super(AttentionStore, self).__init__(
            num_inference_steps=num_inference_steps,
            do_classifier_free_guidance=do_classifier_free_guidance, 
            device=device
        )
        
        self.store_enable_layers = store_enable_layers
        self.store_start_step = store_start_step
        self.store_end_step = store_end_step
        self.store_mode = store_mode
        self.num_inference_steps = num_inference_steps
        self.step_store = self.get_empty_store()
        self.attention_store = []
        self.image_resolution = image_resolution
        self.has_text_encoder_3 = has_text_encoder_3
        self.visualize_now = visualize_now
        self.index = kwargs.get("index", None) 
        if store_mode == 'select' and self.index is None:
            raise ValueError(f"You choose attention apart mode of `{store_mode}`, so you should provide the argument of `index`.")
        
    @staticmethod
    def get_empty_store():
        return []

    def process_qkv(self, layer_id: int, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        return query, key, value
    
    def forward(self, layer_id: int, attn_weight: torch.Tensor, value: torch.Tensor):
        """Saving or visualizing attention maps of the average attention heads.
        
        Args:
            attn_weight: torch.Tensor of shape `[2b, num_heads, S+L, S+L]`, `S` is the image token length, `L` is the text token length
        """
        if self.store_mode is not None:
            stored_attn_weight = self.__get_attention_apart(attn_weight)        # `[2b, L1, L2]`
            if self.visualize_now:
                self.__visualize_attention_now(stored_attn_weight)
            else:
                cur_progress = (self.cur_step + 1) * 1.0 / self.num_inference_steps
                if layer_id in self.store_enable_layers and cur_progress >= self.store_start_step and cur_progress <= self.store_end_step:
                    self.step_store.append(stored_attn_weight)    
        return attn_weight, value
            
    def between_steps(self, layer_id: int):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            cur_progress = (self.cur_step + 1) * 1.0 / self.num_inference_steps
            if cur_progress >= self.store_start_step and cur_progress <= self.store_end_step:
                for i in range(len(self.attention_store)):
                    self.attention_store[i] += self.step_store[i]
        next_progress = (self.cur_step + 1 + 1) * 1.0 / self.num_inference_steps
        if not (next_progress > self.store_end_step and next_progress <= 1.0):
            self.step_store = self.get_empty_store()

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = []
        
    def __get_attention_apart(self, attn_weight: torch.Tensor):
        clip_l = 77
        t5_l = 256 if self.has_text_encoder_3 else 77
        latent_l = (self.image_resolution // 8 // 2) ** 2                               # `// 8` for vae encode, `// 2` for patchify
        
        attn_weight = torch.mean(attn_weight, dim=1)                                    # [b, num_heads, S+L, S+L] -> [b, S+L, S+L]
        
        latent2latent = attn_weight[:, :latent_l, :latent_l]
        latent2clip = attn_weight[:, :latent_l, latent_l:latent_l+clip_l]
        latent2t5 = attn_weight[:, :latent_l, latent_l+clip_l:latent_l+clip_l+t5_l]
        
        clip2latent = attn_weight[:, latent_l:latent_l+clip_l, :latent_l]
        clip2clip = attn_weight[:, latent_l:latent_l+clip_l, latent_l:latent_l+clip_l]
        clip2t5 = attn_weight[:, latent_l:latent_l+clip_l, latent_l+clip_l:latent_l+clip_l+t5_l]
        
        t52latent = attn_weight[:, latent_l+clip_l:latent_l+clip_l+t5_l, :latent_l]
        t52clip = attn_weight[:, latent_l+clip_l:latent_l+clip_l+t5_l, latent_l:latent_l+clip_l]
        t52t5 = attn_weight[:, latent_l+clip_l:latent_l+clip_l+t5_l, latent_l+clip_l:latent_l+clip_l+t5_l]
        
        if self.store_mode == 'all':
            return attn_weight
        elif self.store_mode == 'latent2latent':
            return latent2latent
        elif self.store_mode == 'latent2clip':
            return latent2clip
        elif self.store_mode == 'latent2t5':
            return latent2t5
        elif self.store_mode == 'clip2latent':
            return clip2latent
        elif self.store_mode == 'clip2clip':
            return clip2clip
        elif self.store_mode == 'clip2t5':
            return clip2t5
        elif self.store_mode == 't52latent':
            return t52latent
        elif self.store_mode == 't52clip':
            return t52clip
        elif self.store_mode == 't52t5':
            return t52t5
        elif self.store_mode == 'select':                               
            select_index = self.index
            attn_map = latent2clip[:, :, select_index]
            H = W = int(attn_map.shape[-1] ** 0.5)
            attn_map = attn_map.view(-1, H, W)                         
            return attn_map
    
    def __visualize_attention_now(self, attn_weight: torch.Tensor):
        """Visualize attention map of shape `[2b, L1, L2]` """
        height = attn_weight.shape[1]
        width = attn_weight.shape[0] * attn_weight.shape[2]
        attn_map_images = Image.new("L", (width, height))
        attn_maps = attn_weight.cpu().numpy()
        offset = 0
        for sample_i, attn_map in enumerate(attn_maps):
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())      # normalization
            attn_map_gray = (attn_map * 255).astype(np.uint8)                               # 0 for black, 255 for white
            attn_map_img = Image.fromarray(attn_map_gray, mode='L')  
            save_prefix = f"flowedit_curlayer{self.cur_att_layer}_"
            attn_map_images.paste(attn_map_img, (offset, 0))
            offset += attn_weight.shape[2]
        if self.store_mode == 'select':
            attn_map_images.save(f"inters/flowedit_attentions/{save_prefix}{self.store_mode}_{self.index}th_token_curstep_{self.cur_step}.png")
        else:
            attn_map_images.save(f"inters/flowedit_attentions/{save_prefix}{self.store_mode}_curstep_{self.cur_step}.png")
            
# Copied and revised from https://github.com/sled-group/InfEdit/blob/main/app_infedit/AttentionControlEdit
class AttentionControlEdit(AttentionStore, abc.ABC):
    @abc.abstractmethod
    def replace_attention(self, layer_id:int, attn_source: torch.Tensor, attn_target: torch.Tensor, value: torch.Tensor):
        raise NotImplementedError
    
    def forward(self, layer_id: int, attn_weight: torch.Tensor, value: torch.Tensor):
        """
        Replace target branch attention weight with source branch without considering CFG
        
        Args:
            attn_weight: torch.Tensor of shape `[2b, num_heads, S+L, S+L]`, b is the number of <prompt, source_prompt> pairs
            value: torch.Tensor of shape `[2b, num_heads, S+L, C]`, `C` is the channels of each head 
        """
        target_attn_weight, source_attn_weight = attn_weight.chunk(2, dim=0)        # [b, num_heads, S+L, S+L]
        target_attn_replace = self.replace_attention(
            layer_id, 
            source_attn_weight, 
            target_attn_weight, 
            value
        )                                                                          

        attn_store = torch.cat([target_attn_replace, source_attn_weight], dim=0)    # [2b, num_heads, S+L, S+L]
        super(AttentionControlEdit, self).forward(layer_id, attn_store, value)
        attn_weight = torch.cat([target_attn_replace, source_attn_weight], dim=0)
        return attn_weight, value

    def __init__(
        self, 
        num_inference_steps: Optional[int] = 25,
        cross_replace_enable_layers: Optional[list] = list(range(24)),
        self_process_enable_layers: Optional[list] = list(range(24)),
        cross_start_step: Optional[float] = 0.,
        cross_end_step: Optional[float] = 0.,
        self_start_step: Optional[float] = 0.,
        self_end_step: Optional[float] = 0.,
        store_enable_layers: Optional[list] = list(range(24)),
        store_start_step: Optional[float] = 0.0,
        store_end_step: Optional[float] = 1.0,
        image_resolution: Optional[int] = 512,
        has_text_encoder_3: Optional[bool] = False,
        store_mode: Optional[str] = None,
        visualize_attention_now: Optional[bool] = False,
        do_classifier_free_guidance: Optional[bool] = True,
        device: Optional[str] = "cpu",
        **kwargs
    ):
        super(AttentionControlEdit, self).__init__(
            store_enable_layers=store_enable_layers,
            store_start_step=store_start_step,
            store_end_step=store_end_step,
            num_inference_steps=num_inference_steps, 
            image_resolution=image_resolution,
            has_text_encoder_3=has_text_encoder_3,
            store_mode=store_mode,
            visualize_now=visualize_attention_now,
            do_classifier_free_guidance=do_classifier_free_guidance,
            device=device,
            **kwargs
        )
        
        self.cross_replace_enable_layers = cross_replace_enable_layers
        self.self_process_enable_layers = self_process_enable_layers
        self.cross_start_step = cross_start_step
        self.cross_end_step = cross_end_step
        self.self_start_step = self_start_step
        self.self_end_step = self_end_step

# Copied and revised from https://github.com/sled-group/InfEdit/blob/main/app_infedit/AttentionRefine
class AttentionRefine(AttentionControlEdit):
    def callback_on_step_end(self, pipe, step_i, timestep, local_kwargs):
        if self.local_blend is None:
            return local_kwargs
        else:
            latents = local_kwargs.get("latents", None)
            source_latents = local_kwargs.get("source_latents", None)
            scheduler = pipe.scheduler
            sigma = scheduler.sigmas[scheduler.step_index - 1]
            blended_latent = self.local_blend(step_i, latents, source_latents, self.attention_store, sigma)
            output_kwargs = {"latents": blended_latent}
            return output_kwargs
        
    def replace_attention(self, layer_id: int, attn_source: torch.Tensor, attn_target: torch.Tensor, value: torch.Tensor):
        """
        Replace attn_target weight with attn_source weight with mapper got from Needleman-Wunsch global sequence alignment algorithm

        Args:
            attn_source: torch.Tensor of shape `[b, h, L, L]`, `b` is the number of <prompt, source_prompt> pairs,
                        `L` is sequence length for calculating attention in MM-DiT, which consists 3 parts:
                            * patchified latent length: [image_resolution // 8 (for vae encode) // 2 (for patchify)] ** 2
                            * clip prompt embeds length: 77
                            * t5 prompt embeds length: 256 if provided, else 77
            attn_target: Same as above
        """
        clip_l = 77
        t5_l = 256 if self.has_text_encoder_3 else 77
        latent_l = (self.image_resolution // 8 // 2) ** 2
        b = attn_source.shape[0]

        cur_progress = (self.cur_step + 1) * 1.0 / self.num_inference_steps
        if layer_id in self.cross_replace_enable_layers:
            if cur_progress >= self.cross_start_step and cur_progress <= self.cross_end_step:
                mapper_l = clip_l + t5_l
                source_latent2text = attn_source[:, :, :latent_l, latent_l:latent_l+mapper_l] 
                target_latent2text = attn_target[:, :, :latent_l, latent_l:latent_l+mapper_l] 
                source_text2latent = attn_source[:, :, latent_l:latent_l+mapper_l, :latent_l]
                target_text2latent = attn_target[:, :, latent_l:latent_l+mapper_l, :latent_l]

                attn_replace = attn_target.clone()
                mapped_source_latent2text = source_latent2text[:, :, :, self.mapper].squeeze()
                mapped_source_text2latent = source_text2latent[:, :, self.mapper, :].squeeze()
                
                alphas = self.alphas.transpose(-1, -2)
                attn_replace[:, :, :latent_l, latent_l:latent_l+mapper_l] = mapped_source_latent2text * self.alphas + (1. - self.alphas) * target_latent2text
                attn_replace[:, :, latent_l:latent_l+mapper_l, :latent_l] = mapped_source_text2latent * alphas + (1. - alphas) * target_text2latent

                attn_target = attn_replace
        
        return attn_target

    def process_qkv(self, layer_id: int, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor): 
        clip_l = 77
        t5_l = 256 if self.has_text_encoder_3 else 77
        latent_l = (self.image_resolution // 8 // 2) ** 2

        cur_progress = (self.cur_step + 1) * 1.0 / self.num_inference_steps
        if layer_id in self.self_process_enable_layers:
            if cur_progress >= self.self_start_step and cur_progress <= self.self_end_step:
                target_uncond_value, source_uncond_value, target_cond_value, source_cond_value = value.chunk(4, dim=0)
                target_uncond_value_text = target_uncond_value[:, :, latent_l:, :]
                source_uncond_value_text = source_uncond_value[:, :, latent_l:, :]
                target_cond_value_text = target_cond_value[:, :, latent_l:, :]
                source_cond_value_text = source_cond_value[:, :, latent_l:, :]
                
                value_uncond_text_replace = source_uncond_value_text[:, :, self.mapper, :].squeeze()
                value_cond_text_replace = source_cond_value_text[:, :, self.mapper, :].squeeze()

                alphas = self.alphas.transpose(-1, -2)
                target_uncond_value_text = value_uncond_text_replace * alphas + (1. - alphas) * target_uncond_value_text
                target_cond_value_text = value_cond_text_replace * alphas + (1. - alphas) * target_cond_value_text
                
                target_uncond_value[:, :, latent_l:, :] = target_uncond_value_text
                target_cond_value[:, :, latent_l:, :] = target_cond_value_text
                value = torch.cat([target_uncond_value, source_uncond_value, target_cond_value, source_cond_value], dim=0)    
        return query, key, value
    
    def __init__(
        self, 
        prompts: List[str], 
        prompt_specifiers: List[List[str]], 
        tokenizer,
        text_encoder,
        num_inference_steps: Optional[int] = 25,
        use_local_blend: Optional[bool] = True,
        local_blend: Optional[LocalBlend] = None,
        cross_replace_enable_layers: Optional[list] = list(range(24)),
        self_process_enable_layers: Optional[list] = list(range(24)),
        cross_start_step: Optional[float] = 0.,
        cross_end_step: Optional[float] = 0.,
        self_start_step: Optional[float] = 0.,
        self_end_step: Optional[float] = 0.,
        store_enable_layers: Optional[list] = list(range(24)),
        store_start_step: Optional[float] = 0.0,
        store_end_step: Optional[float] = 1.0,
        image_resolution: Optional[int] = 512,
        has_text_encoder_3: Optional[bool] = False,
        store_mode: Optional[str] = None,
        visualize_attention_now: Optional[bool] = False,
        do_classifier_free_guidance: Optional[bool] = True,
        device: Optional[str] = "cpu",
        torch_dtype: Union[torch.dtype] = torch.float16,
        **kwargs
    ):
        """AttentionRefine Controller

        Args:
            prompts: List[str], `1` source prompt and `n` target prompts. E.g. ["a dog", "a cat", "a blue dog"], 
                        which `n` euqals 2. the first one is source prompt, the last two is target prompts, 
                        and will get 2 edited images. The ultimate goal is to achieve multiple parallel edits of an input image.
                        NOTE: This version of the code only implements the case where `n` equals 1.
            prompt_specifiers: List[List[str]], target blended words (emphasize editing content) and source blended words (emphasize preserving content) 
                        for each editing scene, the length should be `n`. E.g. [["cat", ""], ["blue", "dog"]]
            num_inference_steps: int
            start_step: float in [0, 1], the normalized value representing the denoising progress
            end_step: float in [0, 1], the normalized value representing the denoising progress
        """
        assert not (use_local_blend and local_blend is None)
        
        super(AttentionRefine, self).__init__(
            num_inference_steps=num_inference_steps,
            cross_replace_enable_layers=cross_replace_enable_layers,
            self_process_enable_layers=self_process_enable_layers,
            cross_start_step=cross_start_step,
            cross_end_step=cross_end_step,
            self_start_step=self_start_step,
            self_end_step=self_end_step,
            store_enable_layers=store_enable_layers,
            store_start_step=store_start_step,
            store_end_step=store_end_step,
            image_resolution=image_resolution,
            has_text_encoder_3=has_text_encoder_3,
            store_mode=store_mode,
            visualize_attention_now=visualize_attention_now,
            do_classifier_free_guidance=do_classifier_free_guidance,
            device=device,
            **kwargs
        )

        t5_len = 77 if not has_text_encoder_3 else 77 + 256
        
        (
            mapper,
            alphas,
            ms,                                     # Original alpha
            alpha_e,                                # For editing
            alpha_p,                                # For preserving
            details
        ) = seq_aligner.get_refinement_mapper(
            prompts, 
            prompt_specifiers, 
            tokenizer, 
            text_encoder, 
            device
        )
        self.x_seq_string, self.y_seq_string, original_mapper = details
        
        t5_mapper = torch.arange(mapper.shape[-1], mapper.shape[-1] + t5_len, dtype=mapper.dtype, device=mapper.device).reshape(1, -1).repeat(mapper.shape[0], 1)
        t5_alphas = torch.ones(t5_len).reshape(1, -1).repeat(alphas.shape[0], 1)
        original_mapper = torch.cat([original_mapper, t5_mapper], dim=1)
        mapper = torch.cat([mapper, t5_mapper], dim=1)
        alphas = torch.cat([alphas, t5_alphas], dim=1)
        
        self.original_mapper = original_mapper.to(device)
        self.mapper = mapper.to(device)                                                                 
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1]).to(device).to(torch_dtype)         
        self.ms = ms.reshape(ms.shape[0], 1, 1, ms.shape[1]).to(device).to(torch_dtype)
        
        if use_local_blend:
            self.local_blend = local_blend
            self.alpha_e = alpha_e.to(device).to(torch_dtype)        # [b, 77]
            self.alpha_p = alpha_p.to(device).to(torch_dtype)        # [b, 77]
            alphas = alphas.to(device).to(torch_dtype)          # [b, 77]
            self.local_blend.set_blend(alphas, alpha_e, alpha_p)
        else:
            self.local_blend = None

def inference(
    image: Image,
    source_prompt: str,
    target_prompt: str,
    source_guidance_scale: Optional[float] = 1.5,
    target_guidance_scale: Optional[float] = 3.5,
    num_inference_steps: Optional[int] = 15,
    image_resolution: Optional[int] = 512,
    seed: Optional[int] = None,
    attn_enable_layers: Optional[list] = list(range(24)),
    denoise_model: Optional[bool] = False,
    
    store_enable_layers: Optional[list] = list(range(24)),
    store_start_step: Optional[float] = 0.0,
    store_end_step: Optional[float] = 1.0,
    attention_store_mode: Optional[str] = None, 
    visualize_attention_now: Optional[bool] = False, 
        
    use_local_blend: Optional[bool] = False,
    source_blended_words: Optional[str] = "",
    target_blended_words: Optional[str] = "",
    blend_start_step: Optional[float] = 0.0,
    blend_end_step: Optional[float] = 0.0,
    source_thresh: Optional[float] = 0.0,
    target_thresh: Optional[float] = 0.0,
    save_blend_mask: Optional[bool] = False,
    
    cross_replace_enable_layers: Optional[list] = list(range(24)),
    cross_start_step: Optional[float] = 0.,
    cross_end_step: Optional[float] = 0.,
    
    self_process_enable_layers: Optional[list] = list(range(24)),
    self_start_step: Optional[float] = 0.,
    self_end_step: Optional[float] = 0.,
    
    callback_on_step_end: Optional[Callable] = None,
    
    torch_dtype: Optional[torch.dtype] = torch.float16,
    device: Optional[str] = 'cuda',
    **kwargs
):
    image = image.resize([image_resolution, image_resolution])
    
    if seed is not None:
        generator = torch.manual_seed(seed)

    else:
        generator = None
    if use_local_blend:
        local_blend = LocalBlend(
            source_thresh=source_thresh, 
            target_thresh=target_thresh, 
            save_blend_mask=save_blend_mask,
            num_steps=num_inference_steps,
            start_step=blend_start_step,
            end_step=blend_end_step
        )
    else:
        local_blend = None
    
    controller = AttentionRefine(
        prompts=[source_prompt, target_prompt],
        prompt_specifiers=[[target_blended_words, source_blended_words]],
        tokenizer=pipe.tokenizer,
        text_encoder=pipe.text_encoder,
        num_inference_steps=num_inference_steps,
        
        use_local_blend=use_local_blend,
        local_blend=local_blend,
        
        
        cross_replace_enable_layers=cross_replace_enable_layers,
        self_process_enable_layers=self_process_enable_layers,
        cross_start_step=cross_start_step,
        cross_end_step=cross_end_step,
        self_start_step=self_start_step,
        self_end_step=self_end_step,
        
        store_enable_layers=store_enable_layers,
        store_start_step=store_start_step,
        store_end_step=store_end_step,
        
        image_resolution=image_resolution,
        has_text_encoder_3=False,
        store_mode=attention_store_mode,
        visualize_attention_now=visualize_attention_now,
        do_classifier_free_guidance=True,
        device=device,
        torch_dtype=torch_dtype,
        **kwargs
    )
    def attn_processor_filter(attn_processor_i, attn_processor_name):
        return True if attn_processor_i in attn_enable_layers else False
    
    registered_attn_processor_names, registered_attn_processors_count = register_attention_processor(pipe.transformer, controller, attn_processor_filter)

    if use_local_blend:
        callback_on_step_end = controller.callback_on_step_end
        if visualize_attention_now is not False or attention_store_mode != "clip2latent":
            visualize_attention_now = False
            attention_store_mode = "clip2latent"
            warnings.warn("You use local blend, so FlowEdit reset your config of `visualize_attention_now` to `False`, \
                        `attention_store_mode` to `clip2latent` and `callback_on_step_end` to local blend.")
    
    start_time = time.time()
    result, source = pipe(
        prompt=target_prompt,
        source_prompt=source_prompt,
        image=image,
        guidance_scale=target_guidance_scale,
        source_guidance_scale=source_guidance_scale,
        num_inference_steps=num_inference_steps,
        height=image_resolution,
        width=image_resolution,
        generator=generator,
        denoise_model=denoise_model,
        return_source=True,
        callback_on_step_end=callback_on_step_end
    )
    end_time = time.time()
    duration = format(end_time - start_time, ".3f")
        
    result_images = result.images
    source_images = source.images

    return result_images[0], source_images[0], controller, duration

def edit(
    source_image,
    source_prompt,
    source_guidance_scale,
    source_blend_words,
    source_blend_thresh,
    
    target_prompt,
    target_guidance_scale,
    target_blend_words,
    target_blend_thresh,
    
    attn_layers,
    store_layers,
    cross_layers,
    self_layers,
    
    store_start_step,
    store_end_step,
    cross_start_step,
    cross_end_step,
    self_start_step,
    self_end_step,
    blend_start_step,
    blend_end_step,
    
    seed,
    num_inference_steps
):
    image_resolution = 512
    denoise_model = False
    attn_store_mode = "clip2latent"
    visualize_attention = False
    use_local_blend = True
    save_blend_mask = False
    callback_on_step_end = None
    torch_dtype = torch.float16
    device = "cuda:5"
    result_image, source_image, controller, duration = inference(
        source_image, source_prompt, target_prompt, source_guidance_scale, target_guidance_scale, int(num_inference_steps), image_resolution,
        int(seed), attn_layers, denoise_model, store_layers, store_start_step, store_end_step, attn_store_mode, visualize_attention, use_local_blend,
        source_blend_words, target_blend_words, blend_start_step, blend_end_step, source_blend_thresh, target_blend_thresh, save_blend_mask,
        cross_layers, cross_start_step, cross_end_step, self_layers, self_start_step, self_end_step, callback_on_step_end, torch_dtype, device
    )
    return result_image

if __name__ == '__main__':
    model_id_or_path = "stabilityai/stable-diffusion-3-medium"
    torch_dtype = torch.float16 
    device = "cuda:0"
    
    pipe = FlowEditPipeline.from_pretrained(
        model_id_or_path,
        text_encoder_3=None,
        tokenizer_3=None,
        torch_dtype=torch_dtype
    ).to(device)
    
    with gr.Blocks() as flowedit_demo:
        gr.HTML("<center><h1>FlowEdit</h1></center>")
        with gr.Column():
            with gr.Group():
                source_image = gr.Image(label="Source image", type="pil")
                result_image = gr.Image(label="Edited image", type="pil")
                
        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("Setting"):
                    with gr.Group():
                        with gr.Row():
                            source_prompt = gr.Textbox(label="Source prompt", value="", interactive=True)
                            source_guidance_scale = gr.Slider(label="Source guidance scale", value=1.5, minimum=1, maximum=10, interactive=True)
                        with gr.Row():
                            source_blend_words = gr.Textbox(label="Source blend words", value="", interactive=True)
                            source_blend_thresh = gr.Number(label="Source blend thresh", value=0.4, minimum=0.0, maximum=1.0, step=0.05, interactive=True)
                            
                    with gr.Group():
                        with gr.Row():
                            target_prompt = gr.Textbox(label="Target prompt", value="", interactive=True)
                            target_guidance_scale = gr.Slider(label="Target guidance scale", value=3.5, minimum=1, maximum=10, interactive=True)
                        with gr.Row():
                            target_blend_words = gr.Textbox(label="Target blend words", value="", interactive=True)
                            target_blend_thresh = gr.Number(label="Target blend thresh", value=0.2, minimum=0.0, maximum=1.0, step=0.05, interactive=True)
                        
                    with gr.Group():
                        with gr.Row():
                            seed = gr.Number(label="Seed", value=1823, interactive=True)
                            num_inference_steps = gr.Number(label="Inference steps", value=15, interactive=True)
                    
                    inference_btn1 = gr.Button(value="Edit")
                    
                with gr.TabItem("Advanced Setting"):
                    with gr.Accordion(label="MM-DiT layers control strategy", open=False):
                        registered_attn_processors = gr.Dropdown(label="AttnProcessor layers", value=list(range(0, 24)), choices=list(range(0,24)), multiselect=True, interactive=True)
                        attn_store_layers = gr.Dropdown(label="AttnStore layers", value=list(range(0, 24)), choices=list(range(0,24)), multiselect=True, interactive=True)
                        cross_replace_layers = gr.Dropdown(label="Cross-Attn replace layers", value=list(range(8, 24)), choices=list(range(0,24)), multiselect=True, interactive=True)
                        self_process_layers = gr.Dropdown(label="Self-Attn process layers", value=list(range(0, 24)), choices=list(range(0,24)), multiselect=True, interactive=True)
                    
                    with gr.Group():
                        with gr.Column():
                            with gr.Row():
                                store_start_step = gr.Number(label="Store start", value=0.0, minimum=0.0, maximum=1.0, step=0.05, interactive=True)
                                store_end_step = gr.Number(label="Store end", value=1.0, minimum=0.0, maximum=1.0, step=0.05, interactive=True)
                                
                            with gr.Row():
                                cross_start_step = gr.Number(label="Cross start", value=0.0, minimum=0.0, maximum=1.0, step=0.05, interactive=True)
                                cross_end_step = gr.Number(label="Cross end", value=0.5, minimum=0.0, maximum=1.0, step=0.05, interactive=True)
                                
                            with gr.Row():
                                self_start_step = gr.Number(label="Self start", value=0.0, minimum=0.0, maximum=1.0, step=0.05, interactive=True)
                                self_end_step = gr.Number(label="Self end", value=0.0, minimum=0.0, maximum=1.0, step=0.05, interactive=True)
                            
                            with gr.Row():
                                blend_start_step = gr.Number(label="Blend start", value=0.6, minimum=0.0, maximum=1.0, step=0.05, interactive=True)
                                blend_end_step = gr.Number(label="Blend end", value=1.0, minimum=0.0, maximum=1.0, step=0.05, interactive=True)
                    inference_btn2 = gr.Button(value="Edit")
                    
            inference_btn1.click(
                edit, 
                inputs = [source_image, source_prompt, source_guidance_scale, source_blend_words, source_blend_thresh, target_prompt, target_guidance_scale,
                    target_blend_words, target_blend_thresh, registered_attn_processors, attn_store_layers, cross_replace_layers, self_process_layers,
                    store_start_step, store_end_step, cross_start_step, cross_end_step, self_start_step, self_end_step, blend_start_step, blend_end_step,
                    seed, num_inference_steps],  
                outputs = [result_image]
            )
            inference_btn2.click(
                edit, 
                inputs = [source_image, source_prompt, source_guidance_scale, source_blend_words, source_blend_thresh, target_prompt, target_guidance_scale,
                    target_blend_words, target_blend_thresh, registered_attn_processors, attn_store_layers, cross_replace_layers, self_process_layers,
                    store_start_step, store_end_step, cross_start_step, cross_end_step, self_start_step, self_end_step, blend_start_step, blend_end_step,
                    seed, num_inference_steps],  
                outputs = [result_image]
            )
            
        examples = gr.Examples(
            examples=[
                ["images/cat.jpg", "a opened eyes cat sitting on wooden floor", 1.0, "", 0.4, "a closed eyes cat sitting on wooden floor", 3,
                "eyes", 0.2, list(range(0, 24)), list(range(0,24)), list(range(8, 24)), list(range(0, 24)), 
                0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 1823, 15],
                ["images/rose.jpg", "a red rose in the dark", 1.5, "", 0.4, "a blue rose in the dark", 3.5,
                "rose", 0.07, list(range(0, 24)), list(range(0,24)), list(range(8, 24)), list(range(0, 24)), 
                0.0, 1.0, 0.0, 0.8, 0.0, 0.0, 0.6, 1.0, 1823, 15],
                ["images/lion.jpg", "a lion looking to the left is standing on the grass", 1.5, "", 0.4, "a lion looking ahead is standing on the grass", 3.5, 
                "lion", 0.07, list(range(0, 24)), list(range(0,24)), list(range(8, 24)), list(range(0, 24)),
                0.0, 1.0, 0.0, 0.3, 0.0, 0.0, 0.6, 1.0, 0, 15],
                ["images/flower.jpg", "white flowers on a tree branch with blue sky background", 1.5, "", 0.4, "an oil painting of white flowers on a tree branch with blue sky background", 3.5, 
                "", 0.2, list(range(0, 24)), list(range(0,24)), list(range(8, 24)), list(range(0, 24)),
                0.0, 1.0, 0.0, 0.7, 0.0, 0.0, 0.7, 1.0, 1823, 15],
            ],
            inputs = [source_image, source_prompt, source_guidance_scale, source_blend_words, source_blend_thresh, target_prompt, target_guidance_scale,
                    target_blend_words, target_blend_thresh, registered_attn_processors, attn_store_layers, cross_replace_layers, self_process_layers,
                    store_start_step, store_end_step, cross_start_step, cross_end_step, self_start_step, self_end_step, blend_start_step, blend_end_step,
                    seed, num_inference_steps]
        )
            

    flowedit_demo.launch()