import torch
import torch.nn.functional as F
from typing import Callable, List, Optional, Tuple, Union
from diffusers.models.attention import Attention
from diffusers.models.attention_processor import JointAttnProcessor2_0
import math
import numpy as np
from PIL import Image
from diffusers import StableDiffusion3Pipeline

# Copied and revised from torch.nn.functional.scaled_dot_product_attention()
def scaled_dot_product_attention(
    query: torch.Tensor, 
    key: torch.Tensor, 
    value: torch.Tensor, 
    attn_mask: Optional[torch.Tensor] = None, 
    dropout_p: Optional[float] = 0.0, 
    is_causal: Optional[bool] = False,
    scale: Optional[float] = None,
    enable_gqa: Optional[bool] = False
) -> List[torch.Tensor]:
    """Computes scaled dot product attention on query, key and value tensors, using an optional attention mask if passed, 
        and applying dropout if a probability greater than 0.0 is specified. 
        The optional scale argument can only be specified as a keyword argument.
    
    Args:
        query: torch.Tensor of shape `[batch_size, num_heads, L, head_dim]`
        key: torch.Tensor of shape `[batch_size, num_heads, S, head_dim]`
        value: torch.Tensor of shape `[batch_size, num_heads, S, head_dim]`
        scale: Float, scale factor
    Return:
        attention_probs: torch.Tensor of shape `[batch_size, num_heads, L, S]`
        value: torch.Tensor of shape `[batch_size, num_heads, S, head_dim]`
    """
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight, value
    
# Copied and revised from diffusers.models.attention_processor.JointAttnProcessor2_0
class JointAttnStore:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(
        self, 
        name: Optional[str] = "",
        image_resolution: Optional[int] = 512,
        has_text_encoder_3: Optional[bool] = False,
        mode: Optional[str] = "all",
        visualize_now: Optional[bool] = False,
        **kwargs
    ):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.step_attn_store = []
        self.cur_step = 0
        self.name = name
        self.image_resolution = image_resolution
        self.has_text_encoder_3 = has_text_encoder_3
        self.mode = mode
        self.visualize_now = visualize_now
        self.extra_args = kwargs
        
    def reset(self):
        del self.step_attn_store
        self.step_attn_store = []
        self.cur_step = 0

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = encoder_hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        
        # attention
        query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)         # [B, HW, C]
        key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
        value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)
        
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)    # [B, h, HW, C / h]
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)        
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)    
        
        attn_weight, value = scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)        # [B, h, HW, C // h], [B, h, HW, HW]
        hidden_states = attn_weight @ value
            
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        
        if not self.visualize_now:
            self.step_attn_store.append(attn_weight) 
        else: 
            visualize_attention([attn_weight], [self.cur_step], self.image_resolution, self.has_text_encoder_3, self.mode, self.name, **self.extra_args)
            self.cur_step += 1

        hidden_states = hidden_states.view(batch_size, -1, inner_dim)       # [B, HW, C]
        hidden_states = hidden_states.to(query.dtype)
        
        # Split the attention outputs.
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : residual.shape[1]],
            hidden_states[:, residual.shape[1] :],
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        
        return hidden_states, encoder_hidden_states

def register_attention_processor(
    model, 
    controller,
    filter: Optional[Callable] = None
):
    class FlowEditAttentionProcessor:
        def __init__(
            self, 
            layer_id: Optional[int] = None,
            enable: Optional[bool] = False,
            **kwargs
        ):
            self.layer_id = layer_id
            self.enable = enable
            
        # Copied and revised from diffusers.models.attention_processor.JointAttnProcessor2_0.__call__()
        def __call__(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            *args,
            **kwargs
        ) -> torch.FloatTensor:
            residual = hidden_states

            input_ndim = hidden_states.ndim
            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
            context_input_ndim = encoder_hidden_states.ndim
            if context_input_ndim == 4:
                batch_size, channel, height, width = encoder_hidden_states.shape
                encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size = encoder_hidden_states.shape[0]

            # `sample` projections.
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
            
            # attention
            query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)         # [B, L, C]
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)
            
            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)    # [B, h, L, C // h]
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)        
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)    
            
            attn_weight, value = scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)        # [B, h, L, L], [B, h, L, C // h]
            
            if self.enable:
                attn_weight, value = controller(attn_weight, value)
                
            hidden_states = attn_weight @ value
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)                            # [B, L, C]
            hidden_states = hidden_states.to(query.dtype)
                
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
            if context_input_ndim == 4:
                encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
            
            return hidden_states, encoder_hidden_states
    
    # Set default filter
    if filter is None:
        filter = lambda attn_processor_i, attn_processor_name: True
    
    attn_processors = model.attn_processors
    attn_processor_names = list(attn_processors.keys())
    setted_attn_processors = {}
    registered_attn_processor_names = []
    registered_attn_processors_count = 0
    for attn_processor_i, attn_processor_name in enumerate(attn_processor_names):
        if filter(attn_processor_i, attn_processor_name):
            setted_attn_processors[attn_processor_name] = FlowEditAttentionProcessor(layer_id=attn_processor_i, enable=True)
            registered_attn_processors_count += 1
            registered_attn_processor_names.append(attn_processor_name)
        else:
            setted_attn_processors[attn_processor_name] = attn_processors[attn_processor_name]
    model.set_attn_processor(setted_attn_processors)
    controller.num_att_layers = registered_attn_processors_count
    return registered_attn_processor_names, registered_attn_processors_count
    

def register_attention_store(
    model, 
    filter: Optional[Callable] = None,
    image_resolution: Optional[int] = 512,
    has_text_encoder_3: Optional[bool] = True,
    mode: Optional[str] = None,
    visualize_now: Optional[bool] = True,
    **kwargs
):
    if filter is None:
        filter = lambda attn_processor_i, attn_processor_name : True

    attn_processors = model.attn_processors
    attn_processor_names = list(attn_processors.keys())
    setted_attn_processors = {}
    changed_attn_processor_names = []
    registered_attn_processors_count = 0
    for attn_processor_i, attn_processor_name in enumerate(attn_processor_names):
        if filter(attn_processor_i, attn_processor_name):
            setted_attn_processors[attn_processor_name] = JointAttnStore(
                name=f"layer{attn_processor_i}", 
                image_resolution=image_resolution,
                has_text_encoder_3=has_text_encoder_3,
                mode=mode,
                visualize_now=visualize_now,
                **kwargs
            )
            registered_attn_processors_count += 1
            changed_attn_processor_names.append(attn_processor_name)
        else:
            setted_attn_processors[attn_processor_name] = attn_processors[attn_processor_name]
    model.set_attn_processor(setted_attn_processors)
    return changed_attn_processor_names, registered_attn_processors_count

def visualize_attention(
    attention_store: List[torch.Tensor],
    steps: List[int],
    image_resolution: Optional[int] = 512,
    has_text_encoder_3: Optional[bool] = False,
    mode: Optional[str] = "all",
    save_prefix: Optional[str] = "",
    **kwargs
):
    """Visualizing attention scores of MM-DiT Attention layer.
    
    Args:
        attention_store: N steps attention store of the shape `[batch_size, num_heads, L, L]`, L is sequence length, 
                        equals the sum of resolution of `z_t`, max prompt length of `clip_prompt_embeds` and max prompt length of `t5_prompt_embeds`
        image_resolution: The resolution of the sampled image
        has_text_encoder_3: Whether the T5 text encoder is set
    """
    assert mode in [
        "all", "select",
        "latent2latent", "latent2clip", "latent2t5",
        "clip2latent", "clip2clip", "clip2t5",
        "t52latent", "t52clip", "t52t5"
    ]
    
    def save_attn_map_gray(step, attn_map, mode, save_prefix="", **kwargs):
        """Saving attention map with gray mode
        
        Args:
            step: int, denoising step
            attn_map: torch.Tensor of shape `[L, L]`
            mode: attention mode
        """ 
        if mode == "select":
            select_index = kwargs.pop("index")
            attn_map = attn_map[:, select_index]
            H = W = int(attn_map.shape[0] ** 0.5)
            attn_map = attn_map.view(H, W)
        attn_map = attn_map.cpu().numpy()
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())      # normalization
        attn_map_gray = (attn_map * 255).astype(np.uint8)                               # 0 for black, 255 for white
        attn_map_img = Image.fromarray(attn_map_gray, mode='L')  
        save_prefix = save_prefix + "_" if save_prefix != "" else ""
        if mode == 'select':
            attn_map_img.save(f"inters/attentions/{save_prefix}{mode}_{select_index}th_token_step_{step}.png")
        else:
            attn_map_img.save(f"inters/attentions/{save_prefix}{mode}_step_{step}.png")
            
    clip_l = 77
    t5_l = 256 if has_text_encoder_3 else 77
    latent_l = (image_resolution // 8 // 2) ** 2                # `// 8` for vae encode, `// 2` for patchify
    
    steps = np.arange(attention_store.shape[0]) if steps is None else steps
    for step_i, attn_map in zip(steps, attention_store):         # [batch_size, num_heads, L, L]
        attn_map = attn_map[1]                                  # [num_heads, L, L], conditional attention map with only 1 prompt
        attn_map = torch.mean(attn_map, dim=0)                  # [L, L]
        
        latent2latent = attn_map[:latent_l, :latent_l]
        latent2clip = attn_map[:latent_l, latent_l:latent_l+clip_l]
        latent2t5 = attn_map[:latent_l, latent_l+clip_l:]
        
        clip2latent = attn_map[latent_l:latent_l+clip_l, :latent_l]
        clip2clip = attn_map[latent_l:latent_l+clip_l, latent_l:latent_l+clip_l]
        clip2t5 = attn_map[latent_l:latent_l+clip_l, latent_l+clip_l:]
        
        t52latent = attn_map[latent_l+clip_l:, :latent_l]
        t52clip = attn_map[latent_l+clip_l:, latent_l:latent_l+clip_l]
        t52t5 = attn_map[latent_l+clip_l:, latent_l+clip_l:]
        
        if mode == 'all':
            save_attn_map_gray(step_i, attn_map, mode, save_prefix)
        elif mode == 'latent2latent':
            save_attn_map_gray(step_i, latent2latent, mode, save_prefix)
        elif mode == 'latent2clip':
            save_attn_map_gray(step_i, latent2clip, mode, save_prefix)
        elif mode == 'latent2t5':
            save_attn_map_gray(step_i, latent2t5, mode, save_prefix)
        elif mode == 'clip2latent':
            save_attn_map_gray(step_i, clip2latent, mode, save_prefix)
        elif mode == 'clip2clip':
            save_attn_map_gray(step_i, clip2clip, mode, save_prefix)
        elif mode == 'clip2t5':
            save_attn_map_gray(step_i, clip2t5, mode, save_prefix)
        elif mode == 't52latent':
            save_attn_map_gray(step_i, t52latent, mode, save_prefix)
        elif mode == 't52clip':
            save_attn_map_gray(step_i, t52clip, mode, save_prefix)
        elif mode == 't52t5':
            save_attn_map_gray(step_i, t52t5, mode, save_prefix)
        elif mode == 'select':                                      # for latent2clip
            save_attn_map_gray(step_i, latent2clip, mode, save_prefix, **kwargs)
        else:
            raise ValueError(f"Unsupport visualization mode of `{mode}`.")
        
        
        
if __name__ == '__main__':
    """DEMO
        python attention_processor.py \
            --image_resolution 512 \
            --use_t5 \
            --mode 'select' \
            --visualize_now \
            --prompt 'a photo of a cat and a dog' \
            --num_inference_steps 25 \
            --index   5
    """
    
    import logging
    import argparse
    
    model_id_or_path = "/data/pengzhengwei/checkpoints/stablediffusion/v3"
    torch_dtype = torch.float16
    device = "cuda:5"
    
    logging.basicConfig(
        level=logging.DEBUG, 
        datefmt='%Y/%m/%d %H:%M:%S', 
        format='%(asctime)s - [%(levelname)s] %(message)s', 
        filename="logs/attention_visualization.log", 
        filemode='w'
    )
    logger = logging.getLogger("AttentionStore")
    
    parser = argparse.ArgumentParser(description="Visualize Attention in MM-DiT")
    parser.add_argument('--image_resolution', type=int, default=512, help="Image resolution")
    parser.add_argument('--use_t5', action='store_true', default=False, help="Use T5 XXL text encoder")
    parser.add_argument('--mode', type=str, default='all', help="Attention saving mode")
    parser.add_argument('--visualize_now', action='store_true', default=False, help="Visualize attention map right now (may be slow)")
    parser.add_argument('--index', type=int, default=0, help="Token index for select mode")
    parser.add_argument('--prompt', type=str, default="", help="Source prompt")
    parser.add_argument('--num_inference_steps', type=int, default=25, help="Number of inference steps")
    parser.add_argument('--seed', type=int, default=319, help="Generator seed.")
    parser.add_argument('--filter_ids', type=int, nargs='*', default=list(np.arange(24)), help="Registered attention layer ids.")
    args = parser.parse_args()
    
    logger.info("============================ Visualize Attention in MM-DiT ============================")
    logger.info(f"The image resolution is `{args.image_resolution}`.")
    if args.use_t5:
        logger.info(f"Using T5 XXL text encoder.")
    else:
        logger.info(f"Not using T5 XXL text encoder.")
    logger.info(f"The selected after filter attention processor ids is `{args.filter_ids}`.")
    logger.info(f"Attention visualization mode is `{args.mode}`.")
    if args.visualize_now:
        logger.info("Visualizing attention maps in real time, saving at `inters/attentions/`")
    if args.mode == "select":
        logger.info(f"The selected token index is `{args.index}`")
    
    if args.use_t5:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id_or_path,
            torch_dtype=torch_dtype
        ).to(device)
    else:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id_or_path,
            text_encoder_3=None,
            tokenizer_3=None,
            torch_dtype=torch_dtype
        ).to(device)
    
    logger.info(f"Loaded model from `{model_id_or_path}` successfully, inferencing with `{torch_dtype}` on `{device}`.")
    
    kwargs = {
        "index": args.index
    }

    def filter(attn_processor_i, attn_process_name):
        if attn_processor_i in args.filter_ids:
            return True
        else:
            return False
        
    registered_attn_processor_names, registered_attn_processors_count = register_attention_store(
        pipe.transformer,
        filter=filter,
        image_resolution=args.image_resolution,
        has_text_encoder_3=args.use_t5,
        mode=args.mode,
        visualize_now=args.visualize_now,
        **kwargs
    )
    logger.info(f"Registered `{registered_attn_processors_count}` attention processors, namely `{registered_attn_processor_names}`.")
    
    generator = torch.manual_seed(args.seed)
    logger.info(f"Using seed of `{args.seed}`.")
    logger.info(f"The prompt is `{args.prompt}`, sampling `{args.num_inference_steps}` steps.")
    images = pipe(
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        generator=generator
    ).images
    images[0].save("sample.png")
    logger.info("Sampling done and saved at `sample.png`.")
        
    
    