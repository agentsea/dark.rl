from typing import Optional, Tuple, List, Dict, Any

import torch
from torch import nn
import torch.nn.functional as F

from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLTextConfig, Qwen2_5_VLVisionConfig

from dark.config import Config
from dark.layers.attention import flash_attention_forward
from dark.layers.layernorm import RMSNorm
from dark.layers.linear import ReplicatedLinear
from dark.layers.rotary_embedding import apply_rotary_pos_emb


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section):
    """
    Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors.
    q, k: [total_tokens, num_heads, head_dim]
    cos, sin: [3, total_tokens, head_dim]
    """
    mrope_section_total = mrope_section * 2
    
    cos_chunks = cos.split(mrope_section_total, dim=-1)
    sin_chunks = sin.split(mrope_section_total, dim=-1)

    cos_cat = torch.cat([m[i % 3] for i, m in enumerate(cos_chunks)], dim=-1)
    sin_cat = torch.cat([m[i % 3] for i, m in enumerate(sin_chunks)], dim=-1)
    
    cos_cat = cos_cat.unsqueeze(1)
    sin_cat = sin_cat.unsqueeze(1)

    q_embed = (q * cos_cat) + (rotate_half(q) * sin_cat)
    k_embed = (k * cos_cat) + (rotate_half(k) * sin_cat)
    return q_embed, k_embed


class Qwen2_5_VLMLP(nn.Module):
    def __init__(self, config: Qwen2_5_VLTextConfig, lora_rank=0, lora_alpha=1.0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = ReplicatedLinear(self.hidden_size, self.intermediate_size, bias=False, lora_rank=lora_rank, lora_alpha=lora_alpha)
        self.up_proj = ReplicatedLinear(self.hidden_size, self.intermediate_size, bias=False, lora_rank=lora_rank, lora_alpha=lora_alpha)
        self.down_proj = ReplicatedLinear(self.intermediate_size, self.hidden_size, bias=False, lora_rank=lora_rank, lora_alpha=lora_alpha)
        self.act_fn = F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen2_5_VLAttention(nn.Module):
    def __init__(self, config: Config, lora_rank=0, lora_alpha=1.0):
        super().__init__()
        hf_config: Qwen2_5_VLConfig = config.hf_config
        text_config = hf_config.text_config
        self.config = text_config
        self.hidden_size = text_config.hidden_size
        self.num_attention_heads = text_config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_heads = text_config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.rope_scaling = text_config.rope_scaling

        self.q_proj = ReplicatedLinear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=True, lora_rank=lora_rank, lora_alpha=lora_alpha)
        self.k_proj = ReplicatedLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True, lora_rank=lora_rank, lora_alpha=lora_alpha)
        self.v_proj = ReplicatedLinear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True, lora_rank=lora_rank, lora_alpha=lora_alpha)
        self.o_proj = ReplicatedLinear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=False, lora_rank=lora_rank, lora_alpha=lora_alpha)

    def forward(self, hidden_states: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor], cu_seqlens: torch.Tensor, max_seqlen: int, position_ids: torch.LongTensor) -> Tuple[torch.Tensor, None]:
        total_tokens, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(total_tokens, self.num_attention_heads, self.head_dim)
        key_states = key_states.view(total_tokens, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(total_tokens, self.num_key_value_heads, self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if self.num_key_value_heads != self.num_attention_heads:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        attn_output = flash_attention_forward(query_states, key_states, value_states, cu_seqlens, max_seqlen, softmax_scale=self.scaling, causal=True)
        attn_output = attn_output.view(total_tokens, self.num_attention_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None


class Qwen2_5_VLDecoderLayer(nn.Module):
    def __init__(self, config: Config, lora_rank=0, lora_alpha=1.0):
        super().__init__()
        text_config = config.hf_config.text_config
        self.self_attn = Qwen2_5_VLAttention(config, lora_rank=lora_rank, lora_alpha=lora_alpha)
        self.mlp = Qwen2_5_VLMLP(text_config, lora_rank=lora_rank, lora_alpha=lora_alpha)
        self.input_layernorm = RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor], cu_seqlens: torch.Tensor, max_seqlen: int, position_ids: torch.LongTensor) -> Tuple[torch.FloatTensor, None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, position_embeddings, cu_seqlens, max_seqlen, position_ids)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states, None)


class Qwen2_5_VLRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))

    def forward(self, position_ids: torch.LongTensor, dtype: torch.dtype, device: torch.device):
        # position_ids: [3, total_tokens]
        inv_freq = self.inv_freq.to(device).float()
        position_ids = position_ids.to(device).float()
        
        freqs = torch.einsum("i,tj->tij", inv_freq, position_ids) # [3, total_tokens, head_dim//2]
        emb = torch.cat((freqs, freqs), dim=-1) # [3, total_tokens, head_dim]
        
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        
        return cos, sin


class Qwen2_5_VLModel(nn.Module):
    def __init__(self, config: Config, lora_rank=0, lora_alpha=1.0):
        super().__init__()
        hf_config = config.hf_config
        
        # Patch the config to have head_dim, as the framework seems to expect it.
        if not hasattr(hf_config, 'head_dim'):
            text_config = hf_config.text_config
            hf_config.head_dim = text_config.hidden_size // text_config.num_attention_heads

        self.config = hf_config
        self.text_config = hf_config.text_config
        self.vision_config = hf_config.vision_config

        self.visual = Qwen2_5_VisionTransformer(self.vision_config)

        self.embed_tokens = nn.Embedding(self.text_config.vocab_size, self.text_config.hidden_size, self.text_config.pad_token_id)
        self.layers = nn.ModuleList([Qwen2_5_VLDecoderLayer(config, lora_rank=lora_rank, lora_alpha=lora_alpha) for _ in range(self.text_config.num_hidden_layers)])
        self.norm = RMSNorm(self.text_config.hidden_size, eps=self.text_config.rms_norm_eps)
        
        head_dim = self.text_config.hidden_size // self.text_config.num_attention_heads
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(
            head_dim, 
            max_position_embeddings=self.text_config.max_position_embeddings, 
            base=self.text_config.rope_theta
        )
        self.gradient_checkpointing = False

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
    ):
        # This is a simplified implementation for batched processing in dark.rl style
        # It needs to be further adapted for full correctness with videos and mixed batches.
        
        # For now, we only handle a simple case.
        # The logic from transformers `get_rope_index` is very complex and needs careful porting.
        position_ids_1d = torch.cat([torch.arange(0, ids.shape[0], device=ids.device) for ids in input_ids.split(1)])
        position_ids = position_ids_1d.unsqueeze(0).repeat(3, 1)
        return position_ids

    def forward(
        self, 
        input_ids: torch.LongTensor, 
        cu_seqlens: torch.Tensor, 
        max_seqlen: int, 
        position_ids: torch.LongTensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        
        inputs_embeds = self.embed_tokens(input_ids)

        if pixel_values is not None and image_grid_thw is not None:
            image_features = self.visual(pixel_values, image_grid_thw)
            
            # This is a simplified way to merge embeddings.
            # A more robust implementation would handle splitting image_features per sample
            # and inserting them at the correct positions.
            image_token_mask = (input_ids == self.text_config.image_token_id)
            inputs_embeds[image_token_mask] = image_features.to(inputs_embeds.dtype)

        # TODO: Correctly prepare position_ids for multimodal RoPE
        # The logic here is a placeholder and needs to be replaced with a ported `get_rope_index`
        position_ids_3d = position_ids.unsqueeze(0).repeat(3, 1) # [3, total_tokens]

        position_embeddings = self.rotary_emb(position_ids_3d, dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            if self.training and self.gradient_checkpointing:
                # TODO: Add gradient checkpointing support
                pass
            else:
                hidden_states, _ = decoder_layer(hidden_states, position_embeddings, cu_seqlens, max_seqlen, position_ids)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen2_5_VisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class Qwen2_5_VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Qwen2_5_VLVisionMLP(nn.Module):
    def __init__(self, config: Qwen2_5_VLVisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = F.gelu

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class Qwen2_5_VLPatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = RMSNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


class Qwen2_5_VLVisionAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.scaling = self.head_dim**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        total_tokens, _ = hidden_states.shape
        q, k, v = self.qkv(hidden_states).view(total_tokens, 3, self.num_heads, self.head_dim).unbind(1)
        
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # Vision attention is not causal
        attn_output = flash_attention_forward(q, k, v, cu_seqlens, max_seqlen, softmax_scale=self.scaling, causal=False)
        attn_output = attn_output.view(total_tokens, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class Qwen2_5_VLVisionBlock(nn.Module):
    def __init__(self, config: Qwen2_5_VLVisionConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = RMSNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen2_5_VLVisionAttention(config.hidden_size, num_heads=config.num_heads)
        self.mlp = Qwen2_5_VLVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen2_5_VisionTransformer(nn.Module):
    def __init__(self, config: Qwen2_5_VLVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.fullatt_block_indexes = config.fullatt_block_indexes
        self.window_size = config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [Qwen2_5_VLVisionBlock(config) for _ in range(config.depth)]
        )
        self.merger = Qwen2_5_VLPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
        )
        self.gradient_checkpointing = False

    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
                max_seqlen_now = max_seqlen
            else:
                cu_seqlens_now = cu_window_seqlens
                max_seqlen_now = (cu_window_seqlens[1:] - cu_window_seqlens[:-1]).max().item()

            if self.gradient_checkpointing and self.training:
                # TODO: add gradient checkpointing
                pass
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, max_seqlen=max_seqlen_now, position_embeddings=position_embeddings)

        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states


class Qwen2_5_VLForCausalLM(nn.Module):
    def __init__(self, config: Config, lora_rank=0, lora_alpha=1.0):
        super().__init__()
        self.config = config.hf_config
        self.model = Qwen2_5_VLModel(config, lora_rank=lora_rank, lora_alpha=lora_alpha)
        self.lm_head = nn.Linear(self.config.text_config.hidden_size, self.config.text_config.vocab_size, bias=False)

    @property
    def visual(self):
        return self.model.visual

    def forward(
        self, 
        input_ids: torch.LongTensor, 
        cu_seqlens: torch.Tensor, 
        max_seqlen: int, 
        position_ids: torch.LongTensor, 
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ):
        hidden_states = self.model(
            input_ids=input_ids, 
            cu_seqlens=cu_seqlens, 
            max_seqlen=max_seqlen, 
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            # The loss calculation in dark.rl seems to be handled outside the model.
            # Replicating a simplified version here.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))
        return logits, loss

    def freeze_base_model(self):
        for name, param in self.model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False 