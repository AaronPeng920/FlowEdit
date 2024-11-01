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

logging.basicConfig(
    level=logging.DEBUG, 
    datefmt='%Y/%m/%d %H:%M:%S', 
    format='%(asctime)s - [%(levelname)s] %(message)s', 
    filename="logs/flow_edit_process.log", 
    filemode='w'
)
logger = logging.getLogger("FlowEdit")

MODEL_ID_OR_PATH = "/data/pengzhengwei/checkpoints/stablediffusion/v3"
TORCH_DTYPE = torch.float16
DEVICE = "cuda:5"

pipe = FlowEditPipeline.from_pretrained(
        MODEL_ID_OR_PATH,
        text_encoder_3=None,
        tokenizer_3=None,
        torch_dtype=TORCH_DTYPE
    ).to(DEVICE)
logger.info(f"Load FlowEditPipeline from `{MODEL_ID_OR_PATH}` successfully, inferencing with `{TORCH_DTYPE}` on `{DEVICE}`.")

# Copied and revised from https://github.com/sled-group/InfEdit/blob/main/app_infedit/AttentionControl
class AttentionControl(abc.ABC):
    def step_callback(self, z_t):
        return z_t

    def between_steps(self):
        return

    @abc.abstractmethod
    def forward(self, attn_weight: torch.Tensor, value: torch.Tensor):
        raise NotImplementedError

    def __call__(self, attn_weight: torch.Tensor, value: torch.Tensor):
        if self.do_classifier_free_guidance:
            bs = attn_weight.shape[0]
            # attn_weight[:bs // 2], value[:bs // 2] = self.forward(attn_weight[:bs // 2], value[:bs // 2])
            attn_weight[bs // 2:], value[bs // 2:] = self.forward(attn_weight[bs // 2:], value[bs // 2:])
        else:
            attn_weight, value = self.forward(attn_weight, value)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            logger.info(f"Complete the {self.cur_step + 1}/{self.num_inference_steps} step.")
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn_weight, value

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(
        self, 
        num_inference_steps: int = 25, 
        do_classifier_free_guidance: bool = True, 
        device: str = "cpu"
    ):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.num_inference_steps = num_inference_steps
        self.do_classifier_free_guidance = do_classifier_free_guidance
        self.device = device

# Copied and revised from https://github.com/sled-group/InfEdit/blob/main/app_infedit/EmptyControl
class EmptyControl(AttentionControl):
    def forward(self, attn_weight: torch.Tensor, value: torch.Tensor):
        return attn_weight, value

# Copied and revised from https://github.com/sled-group/InfEdit/blob/main/app_infedit/AttentionStore
class AttentionStore(AttentionControl):
    @staticmethod
    def get_empty_store():
        return []

    def forward(self, attn_weight: torch.Tensor, value: torch.Tensor):
        if self.mode is not None:
            stored_attn_weight = self.__get_attention_apart(attn_weight)
            if self.visualize_now:
                self.__visualize_attention_now(stored_attn_weight)
            else:
                self.step_store.append(stored_attn_weight)
        return attn_weight, value
            
    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for i in range(len(self.attention_store)):
                self.attention_store[i] += self.step_store[i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = [attn_store / self.cur_step for attn_store in self.attention_store]
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = []

    def __get_attention_apart(self, attn_weight: torch.Tensor):
        clip_l = 77
        t5_l = 256 if self.has_text_encoder_3 else 77
        latent_l = (self.image_resolution // 8 // 2) ** 2                               # `// 8` for vae encode, `// 2` for patchify
        
        attn_weight = torch.mean(attn_weight, dim=1)                                    # [B, num_heads, L, L] -> [B, L, L]
        
        latent2latent = attn_weight[:, :latent_l, :latent_l]
        latent2clip = attn_weight[:, :latent_l, latent_l:latent_l+clip_l]
        latent2t5 = attn_weight[:, :latent_l, latent_l+clip_l:]
        
        clip2latent = attn_weight[:, latent_l:latent_l+clip_l, :latent_l]
        clip2clip = attn_weight[:, latent_l:latent_l+clip_l, latent_l:latent_l+clip_l]
        clip2t5 = attn_weight[:, latent_l:latent_l+clip_l, latent_l+clip_l:]
        
        t52latent = attn_weight[:, latent_l+clip_l:, :latent_l]
        t52clip = attn_weight[:, latent_l+clip_l:, latent_l:latent_l+clip_l]
        t52t5 = attn_weight[:, latent_l+clip_l:, latent_l+clip_l:]
        
        if self.mode == 'all':
            return attn_weight
        elif self.mode == 'latent2latent':
            return latent2latent
        elif self.mode == 'latent2clip':
            return latent2clip
        elif self.mode == 'latent2t5':
            return latent2t5
        elif self.mode == 'clip2latent':
            return clip2latent
        elif self.mode == 'clip2clip':
            return clip2clip
        elif self.mode == 'clip2t5':
            return clip2t5
        elif self.mode == 't52latent':
            return t52latent
        elif self.mode == 't52clip':
            return t52clip
        elif self.mode == 't52t5':
            return t52t5
        elif self.mode == 'select':                                      # for latent2clip
            select_index = self.index
            attn_map = latent2clip[:, :, select_index]
            H = W = int(attn_map.shape[-1] ** 0.5)
            attn_map = attn_map.view(-1, H, W)
            return attn_map
        else:
            raise ValueError(f"Unsupport attention apart mode of `{self.mode}`.")
    
    def __visualize_attention_now(self, attn_weight: torch.Tensor):
        """Visualize attention map of shape `[B, L, S]` or `[B, H, W]` if select mode right now."""
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
        if self.mode == 'select':
            attn_map_images.save(f"inters/flowedit_attentions/{save_prefix}{self.mode}_{self.index}th_token_curstep_{self.cur_step}.png")
        else:
            attn_map_images.save(f"inters/flowedit_attentions/{save_prefix}{self.mode}_curstep_{self.cur_step}.png")
        
        
    def __init__(
        self,
        num_inference_steps: Optional[int] = 25, 
        image_resolution: Optional[int] = 512,
        has_text_encoder_3: Optional[bool] = False,
        mode: Optional[str] = None,
        visualize_now: Optional[bool] = False,
        do_classifier_free_guidance: Optional[bool] = True,
        device: Optional[str] = "cpu",
        **kwargs
    ):
        assert mode in [
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
        self.num_inference_steps = num_inference_steps
        self.step_store = self.get_empty_store()
        self.attention_store = []
        self.image_resolution = image_resolution
        self.has_text_encoder_3 = has_text_encoder_3
        self.mode = mode
        self.visualize_now = visualize_now
        self.index = kwargs.get("index", None) 
        if mode == 'select' and self.index is None:
            raise ValueError(f"You choose attention apart mode of `{mode}`, so you should provide the argument of ")
        
# Copied and revised from https://github.com/sled-group/InfEdit/blob/main/app_infedit/AttentionControlEdit
class AttentionControlEdit(AttentionStore, abc.ABC):
    @abc.abstractmethod
    def replace_attention(self, attn_source, attn_target):
        raise NotImplementedError
    
    def forward(self, attn_weight: torch.Tensor, value: torch.Tensor):
        """
        Replace target branch attention weight with source branch without considering CFG
        
        Args:
            attn_weight: torch.Tensor of shape `[2b, h, L, L]`, b is the number of <prompt, source_prompt> pairs
            value: torch.Tensor of shape `[2b, h, L, C // h]`
        """
        target_attn_weight, source_attn_weight = attn_weight.chunk(2, dim=0)        # [b, h, L, L]
        target_attn_replace = self.replace_attention(source_attn_weight, target_attn_weight)
        attn_store = torch.cat([target_attn_replace, source_attn_weight], dim=0)    # [2b, h, L, L]
        super(AttentionControlEdit, self).forward(attn_store, value)
        attn_weight = torch.cat([target_attn_replace, source_attn_weight], dim=0)
        return attn_weight, value

    def __init__(
        self, 
        num_inference_steps: Optional[int] = 25,
        start_step: Optional[float] = 0.,
        end_step: Optional[float] = 1.,
        image_resolution: Optional[int] = 512,
        has_text_encoder_3: Optional[bool] = False,
        mode: Optional[str] = None,
        do_classifier_free_guidance: Optional[bool] = True,
        visualize_attention_now: Optional[bool] = False,
        device: Optional[str] = "cpu",
        **kwargs
    ):
        super(AttentionControlEdit, self).__init__(
            num_inference_steps=num_inference_steps,
            image_resolution=image_resolution,
            has_text_encoder_3=has_text_encoder_3,
            mode=mode,
            do_classifier_free_guidance=do_classifier_free_guidance,
            visualize_now=visualize_attention_now,
            device=device,
            **kwargs
        )
        self.num_inference_steps = num_inference_steps
        self.start_step = start_step
        self.end_step = end_step

# Copied and revised from https://github.com/sled-group/InfEdit/blob/main/app_infedit/AttentionRefine
class AttentionRefine(AttentionControlEdit):
    def replace_attention(self, attn_source, attn_target):
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
        
        source_latent2clip = attn_source[:, :, :latent_l, latent_l:latent_l+clip_l] 
        target_latent2clip = attn_target[:, :, :latent_l, latent_l:latent_l+clip_l] 

        attn_replace = attn_target.clone()
        mapped_source_latent2clip = source_latent2clip[:, :, :, self.mapper].squeeze()
        attn_replace[:, :, :latent_l, latent_l:latent_l+clip_l] = mapped_source_latent2clip * self.alphas + (1. - self.alphas) * target_latent2clip
        cur_progress = (self.cur_step + 1) * 1.0 / self.num_inference_steps
        if cur_progress >= self.start_step and cur_progress <= self.end_step:
            logger.info(f"Replace attention at {self.cur_att_layer}th layer.")
            attn_target[:b] = attn_replace
            
        # source_latent2clip = attn_source[:, :, :latent_l, latent_l:] 
        # target_latent2clip = attn_target[:, :, :latent_l, latent_l:] 

        # attn_replace = attn_target.clone()
        # mapper = self.mapper.repeat(1, 2)
        # alphas =self.alphas.repeat(1, 1, 1, 2)
        # mapped_source_latent2clip = source_latent2clip[:, :, :, mapper].squeeze()
        # attn_replace[:, :, :latent_l, latent_l:] = mapped_source_latent2clip * alphas + (1. - alphas) * target_latent2clip
        
        return attn_target

    def process_qkv(
        self,
        query,
        key,
        value
    ):
        # clip_l = 77
        # t5_l = 256 if self.has_text_encoder_3 else 77
        # latent_l = (self.image_resolution // 8 // 2) ** 2
        
        # pca = PCA(n_components=1)
        # query_average_heads = torch.mean(value, dim=1).cpu().numpy()
        # for i, q in enumerate(query_average_heads):
        #     pca.fit(q)
        #     embs = pca.transform(q).squeeze()
        #     embs_latent = embs[:latent_l]
        #     h = w = int(latent_l ** 0.5)
        #     embs_latent = embs_latent.reshape(h, w)
        #     embs_map = (embs_latent - embs_latent.min()) / (embs_latent.max() - embs_latent.min())      
        #     embs_map = (embs_map >= 0.5).astype(np.uint8)
        #     embs_map_gray = (embs_map * 255).astype(np.uint8)                               
        #     embs_map_img = Image.fromarray(embs_map_gray, mode='L')  
        #     embs_map_img.save(f"inters/queries/layer{self.cur_att_layer}_step_{self.cur_step}_batch_{i}.png")
        
    
            
        return query, key, value
    
    def __init__(
        self, 
        prompts: List[str], 
        prompt_specifiers: List[List[str]], 
        tokenizer,
        text_encoder,
        num_inference_steps: int,
        start_step: Optional[float] = 0.,
        end_step: Optional[float] = 1.,
        image_resolution: Optional[int] = 512,
        has_text_encoder_3: Optional[bool] = False,
        mode: Optional[str] = None,
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
            start_step: float in [0, 1], the normalized value representing the denoising progress, 
                        after that attention replacement begins.
            end_step: float in [0, 1], the normalized value representing the denoising progress, 
                        after that attention replacement ends.
        """
        super(AttentionRefine, self).__init__(
            num_inference_steps=num_inference_steps,
            start_step=start_step,
            end_step=end_step,
            image_resolution=image_resolution,
            has_text_encoder_3=has_text_encoder_3,
            mode=mode,
            visualize_attention_now=visualize_attention_now,
            do_classifier_free_guidance=do_classifier_free_guidance,
            **kwargs
        )

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
        x_seq_string, y_seq_string, original_mapper = details
        
        self.original_mapper = original_mapper.to(device)
        self.mapper = mapper.to(device)         # [n, S], S if the max_seq_length of tokenizer
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1]).to(device).to(torch_dtype)
        self.ms = ms.reshape(ms.shape[0], 1, 1, ms.shape[1]).to(device).to(torch_dtype)

        logger.info(f"=========== Initializing AttentionRefine controller. ===========")
        logger.info(f"source prompt is `{prompts[0]}`.")
        logger.info(f"target prompt is `{prompts[1]}`.")
        logger.info(f"source blend is `{prompt_specifiers[0][1]}`.")
        logger.info(f"target blend is `{prompt_specifiers[0][0]}`.")
        logger.info(f"source encoded sequence is `{x_seq_string}`.")
        logger.info(f"target encoded sequence is `{y_seq_string}`.")
        logger.info(f"original mapper is {original_mapper}.")
        logger.info(f"final mapper is `{mapper}`.")
        logger.info(f"final alphas is `{alphas}`.")
        logger.info(f"total inference steps is `{num_inference_steps}`.")
        logger.info(f"attention replacement start step is {start_step}.")
        logger.info(f"attention replacement end step is {end_step}.")
        logger.info(f"visualize attention mode is `{mode}`.")
        logger.info(f"set visualize attention now to `{visualize_attention_now}`.")
        logger.info(f"extra args are `{kwargs}`.")
        logger.info(f"================================================================")


def inference(
    image: Image,
    source_prompt: str,
    target_prompt: str,
    source_blended_words: Optional[str] = "",
    target_blended_words: Optional[str] = "",
    source_guidance_scale: Optional[float] = 7.0,
    target_guidance_scale: Optional[float] = 7.0,
    num_inference_steps: Optional[int] = 25,
    start_step: Optional[float] = 0.,
    end_step: Optional[float] = 1.,
    image_resolution: Optional[int] = 512,
    seed: Optional[int] = None,
    attention_processor_filter: Optional[Callable] = None,
    attention_visualize_mode: Optional[str] = None, 
    visualize_attention_now: Optional[bool] = False, 
    denoise_model: Optional[bool] = False,
    callback_on_step_end: Optional[Callable] = None,
    **kwargs
):
    generator = torch.manual_seed(seed) if seed is not None else None
    logger.info(f"Set random seed `{seed}`.")
    if callback_on_step_end is not None:
        logger.info(f"Set step end callback `{callback_on_step_end.__name__}`")
        
    controller = AttentionRefine(
        prompts=[source_prompt, target_prompt],
        prompt_specifiers=[[target_blended_words, source_blended_words]],
        tokenizer=pipe.tokenizer,
        text_encoder=pipe.text_encoder,
        num_inference_steps=num_inference_steps,
        start_step=start_step,
        end_step=end_step,
        image_resolution=image_resolution,
        has_text_encoder_3=False,
        mode=attention_visualize_mode,
        visualize_attention_now=visualize_attention_now,
        do_classifier_free_guidance=True,
        device=DEVICE,
        torch_dtype=TORCH_DTYPE,
        **kwargs
    )

    registered_attn_processor_names, registered_attn_processors_count = register_attention_processor(pipe.transformer, controller, attention_processor_filter)
    logger.info(f"Registered {registered_attn_processors_count} {controller.__class__.__name__} controllers, namely {registered_attn_processor_names}.")

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
    
    result_images = result.images
    source_images = source.images

    return result_images[0], source_images[0], controller

if __name__ == '__main__':
    from utils import save_inter_latents_callback
    
    source_prompt = "the apple is on the desk"
    target_prompt = "the banana is on the desk"
    source_blended_words = ""
    target_blended_words = ""
    source_guidance_scale = 7.0
    target_guidance_scale = 7.0
    num_inference_steps = 4
    start_step = 0.0
    end_step = 0.0
    image_resolution = 512
    seed = 1234
    image = Image.open("assets/apple.png").convert('RGB').resize([image_resolution, image_resolution])
    attention_visualize_mode = None
    visualize_attention_now = True
    denoise_model = False
    callback_on_step_end = save_inter_latents_callback
    kwargs = {
        'index': 2
    }

    def filter(i, name):
        if i >= 0:
            return True
        else:
            return False
        
    result_image, source_image, controller = inference(
        image=image,
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        source_blended_words=source_blended_words,
        target_blended_words=target_blended_words,
        source_guidance_scale=source_guidance_scale,
        target_guidance_scale=target_guidance_scale,
        num_inference_steps=num_inference_steps,
        start_step=start_step,
        end_step=end_step,
        image_resolution=image_resolution,
        seed=seed,
        attention_visualize_mode=attention_visualize_mode,
        visualize_attention_now=visualize_attention_now,
        denoise_model=denoise_model,
        attention_processor_filter=filter,
        callback_on_step_end=callback_on_step_end,
        **kwargs
    )

    result_image.save("result.png")
    source_image.save("source.png")

