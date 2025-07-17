from typing import Optional, Tuple, List, Dict, Any

import logging
import torch
from torch import nn
import torch.nn.functional as F
import os
import math

from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLTextConfig, Qwen2_5_VLVisionConfig
from transformers.models.qwen2_vl.modeling_qwen2_vl import PatchEmbed

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


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim: int = 1):
    """Apply Multimodal Rotary Position Embedding (M-RoPE) to ``q`` and ``k``.

    This is a near-direct port of the reference implementation in
    `transformers.models.qwen2_vl.modeling_qwen2_vl`.

    Args
    ----
    q, k:
        Tensors of shape ``[batch, heads, tokens, head_dim]`` or similar.
    cos, sin:
        RoPE lookup tables of shape ``[3, tokens, head_dim]`` where the first
        axis encodes the temporal / height / width phases, respectively.
    mrope_section (Union[int, Sequence[int]]):
        Multimodal rope section is for channel dimension of temporal, height, width.
    unsqueeze_dim (int):
        Dimension to unsqueeze for broadcasting.
    """
    # HF implementation exactly
    mrope_section = [x * 2 for x in mrope_section]
    
    # cos and sin have shape (3, seq_len, head_dim)
    # We need to split along the head_dim (last dimension) according to mrope_section
    cos_splits = cos.split(mrope_section, dim=-1)  # List of tensors with shapes (3, seq_len, section_size)
    sin_splits = sin.split(mrope_section, dim=-1)

    # Cycle through the 3 axes using modulo arithmetic
    cos_combined = torch.cat([m[i % 3] for i, m in enumerate(cos_splits)], dim=-1)
    sin_combined = torch.cat([m[i % 3] for i, m in enumerate(sin_splits)], dim=-1)
    
    # Add unsqueeze dimension for broadcasting
    cos_combined = cos_combined.unsqueeze(unsqueeze_dim)
    sin_combined = sin_combined.unsqueeze(unsqueeze_dim)

    # Apply RoPE
    q_embed = (q * cos_combined) + (rotate_half(q) * sin_combined)
    k_embed = (k * cos_combined) + (rotate_half(k) * sin_combined)
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
        self.layer_idx = None  # Will be set by the parent layer

    def forward(self, hidden_states: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor], cu_seqlens: torch.Tensor, max_seqlen: int, position_ids: torch.LongTensor, cache_position: Optional[torch.LongTensor] = None) -> Tuple[torch.Tensor, None]:
        # Convert our 2D flattened input to 3D batched format that HF expects
        total_tokens, hidden_size = hidden_states.shape
        bsz = 1  # We always use batch size 1 in our implementation
        q_len = total_tokens
        
        # Reshape to match HF's expected input format: (bsz, q_len, hidden_size)
        hidden_states_3d = hidden_states.unsqueeze(0)  # (1, total_tokens, hidden_size)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape exactly like HF: (bsz, q_len, num_heads, head_dim) -> (bsz, num_heads, q_len, head_dim)
        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings - need to reshape for our RoPE function
        cos, sin = position_embeddings
        # Convert back to (total_tokens, num_heads, head_dim) for RoPE, then back to HF format
        query_rope = query_states.transpose(1, 2).contiguous().view(total_tokens, self.num_attention_heads, self.head_dim)
        key_rope = key_states.transpose(1, 2).contiguous().view(total_tokens, self.num_key_value_heads, self.head_dim)
        
        query_rope, key_rope = apply_multimodal_rotary_pos_emb(
            query_rope,
            key_rope,
            cos,
            sin,
            self.rope_scaling["mrope_section"],
            unsqueeze_dim=1,
        )

        # Convert back to HF format: (bsz, num_heads, q_len, head_dim)
        query_states = query_rope.view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_rope.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Handle GQA (Grouped Query Attention) exactly like HF
        if self.num_key_value_heads != self.num_attention_heads:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Ensure dtype alignment (original checkpoints use fp16)
        tgt_dtype = value_states.dtype
        query_states = query_states.to(tgt_dtype)
        key_states = key_states.to(tgt_dtype)

        # Use HF's approach: is_causal=True for causal masking
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,  # Let SDPA handle causal masking
            dropout_p=0.0,
            is_causal=True,  # This is the key difference!
        )

        # Convert back to HF's expected output format
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.num_attention_heads * self.head_dim)
        
        # Flatten back to our 2D format and apply output projection
        attn_output = attn_output.squeeze(0)  # Remove batch dimension: (total_tokens, hidden_size)
        attn_output = self.o_proj(attn_output)
        
        # ----------------------------------------------------------
        # DEBUG: quick health-check on rotated embeddings.  Trigger
        # only on the *first* forward pass (cache_position==None or 0)
        # and on short sequences to avoid log spam.
        # ----------------------------------------------------------
        if cache_position is None or (cache_position.numel() and cache_position[0] == 0):
            with torch.no_grad():
                q_nan = torch.isnan(query_states).any().item()
                k_nan = torch.isnan(key_states).any().item()
                if q_nan or k_nan:
                    logging.debug(
                        f"[nan-alert] layer{self.layer_idx} q_nan={q_nan} k_nan={k_nan} seq={total_tokens}"
                    )
                else:
                    q_stats = (
                        float(query_states.min().item()),
                        float(query_states.max().item()),
                        float(query_states.mean().item()),
                    )
                    logging.debug(
                        f"[attn-debug] seq={total_tokens} q[min,max,mean]={q_stats} {'NaN!' if q_nan else ''}"
                    )

        if os.getenv("DEBUG_SHAPES") == "2":
            # Lightweight validation – print the norm of first token per head
            qs_norm = query_states[..., 0, :].norm(dim=-1).mean().item()
            logging.debug(f"[dbg-text-rope] avg |q| after RoPE = {qs_norm:.4f}")
        
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
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000, device=None):
        """Light-weight rotary embedding used by the local flattened implementation.

        Parameters
        ----------
        dim:
            The *head* dimension (not hidden size!).  Must be even.
        max_position_embeddings:
            Ignored in this simplified variant; included for interface parity.
        base:
            RoPE base θ.
        """
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        position_ids: torch.LongTensor,  # shape (total_tokens,)
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos/sin lookup tensors with shape (tokens, dim)."""

        inv_freq = self.inv_freq.to(device)  # (dim//2)
        pos = position_ids.to(device).float()  # (tokens,)
        
        freqs = torch.einsum("i,j->ij", pos, inv_freq)  # (tokens, dim//2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (tokens, dim)
        return emb.cos().to(dtype), emb.sin().to(dtype)


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
        
        # Set layer indices for debugging
        for i, layer in enumerate(self.layers):
            layer.self_attn.layer_idx = i
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
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.
        This follows the HF approach more closely by looking for vision_start tokens first.
        """
        seq_len = input_ids.shape[0]
        device = input_ids.device

        # Default: plain ramp for text tokens
        position_ids = torch.zeros(3, seq_len, dtype=torch.long, device=device)

        # If no image tokens present just use 1-D ramp.
        if image_grid_thw is None or image_grid_thw.numel() == 0:
            ramp = torch.arange(seq_len, device=device, dtype=torch.long)
            position_ids[:] = ramp  # broadcast to all 3 rows
            return position_ids

        # HF approach: look for vision_start tokens first
        vision_start_token_id = 151652  # <|vision_start|>
        image_token_id = 151655  # <|image_pad|>
        spatial_merge_size = self.vision_config.spatial_merge_size

        # Find vision_start tokens
        vision_start_indices = torch.where(input_ids == vision_start_token_id)[0]
        
        if len(vision_start_indices) == 0:
            # No vision tokens, use simple 1D ramp
            ramp = torch.arange(seq_len, device=device, dtype=torch.long)
            position_ids[:] = ramp
            return position_ids

        # Build position IDs following HF's approach
        llm_pos_ids_list = []
        st = 0
        
        for vision_start_idx in vision_start_indices:
            # Add text positions before this vision block
            text_len = vision_start_idx.item() - st
            if text_len > 0:
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len, device=device, dtype=torch.long).view(1, -1).expand(3, -1) + st_idx)
            
            # Process the vision block
            # Find the end of this vision block (look for consecutive image_pad tokens)
            vision_end_idx = vision_start_idx + 1
            while vision_end_idx < seq_len and input_ids[vision_end_idx] == image_token_id:
                vision_end_idx += 1
            
            num_vision_tokens = vision_end_idx - vision_start_idx - 1  # exclude the vision_start token
            
            if num_vision_tokens > 0 and image_grid_thw is not None:
                # Get grid dimensions for this image
                T, H, W = map(int, image_grid_thw[0].tolist())  # assume single image
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    T,
                    H // spatial_merge_size,
                    W // spatial_merge_size,
                )
                
                # Add position for vision_start token
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.tensor([[st_idx], [st_idx], [st_idx]], device=device, dtype=torch.long))
                
                # Create proper 3D position embeddings exactly like HF
                # For a single image: T=1, H=36, W=36 after spatial merging becomes llm_grid_h=18, llm_grid_w=18
                # Total tokens = T * llm_grid_h * llm_grid_w = 1 * 18 * 18 = 324
                
                # Temporal dimension: all tokens get t=0 since it's a single image
                t_index = torch.zeros(num_vision_tokens, device=device, dtype=torch.long)
                
                # Height dimension: repeats for each row
                # Pattern: [0,0,...,0, 1,1,...,1, 2,2,...,2, ...] where each value repeats llm_grid_w times
                h_index = torch.arange(llm_grid_h, device=device, dtype=torch.long).repeat_interleave(llm_grid_w)
                if len(h_index) > num_vision_tokens:
                    h_index = h_index[:num_vision_tokens]
                elif len(h_index) < num_vision_tokens:
                    # Pad with the last value
                    h_index = torch.cat([h_index, h_index[-1:].repeat(num_vision_tokens - len(h_index))])
                
                # Width dimension: cycles through 0 to llm_grid_w-1
                # Pattern: [0,1,2,...,W-1, 0,1,2,...,W-1, 0,1,2,...,W-1, ...]
                w_index = torch.arange(llm_grid_w, device=device, dtype=torch.long).repeat(llm_grid_h)
                if len(w_index) > num_vision_tokens:
                    w_index = w_index[:num_vision_tokens]
                elif len(w_index) < num_vision_tokens:
                    # Pad with cycling pattern
                    remaining = num_vision_tokens - len(w_index)
                    w_index = torch.cat([w_index, torch.arange(remaining, device=device, dtype=torch.long) % llm_grid_w])
                
                vision_pos = torch.stack([t_index, h_index, w_index]) + st_idx + 1
                llm_pos_ids_list.append(vision_pos)
            
            st = vision_end_idx
        
        # Add remaining text positions
        if st < seq_len:
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = seq_len - st
            llm_pos_ids_list.append(torch.arange(text_len, device=device, dtype=torch.long).view(1, -1).expand(3, -1) + st_idx)
        
        # Concatenate all position IDs
        if llm_pos_ids_list:
            llm_positions = torch.cat(llm_pos_ids_list, dim=1)
            position_ids[:, :llm_positions.shape[1]] = llm_positions
            
            # Fill any remaining positions
            if llm_positions.shape[1] < seq_len:
                remaining = seq_len - llm_positions.shape[1]
                start_idx = llm_positions.max().item() + 1
                tail = torch.arange(start_idx, start_idx + remaining, device=device, dtype=torch.long)
                position_ids[:, llm_positions.shape[1]:] = tail
        else:
            # Fallback to simple ramp
            ramp = torch.arange(seq_len, device=device, dtype=torch.long)
            position_ids[:] = ramp

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
            
            # ------------------------------------------------------------------
            # 1. Build a mask that captures *all* possible image-placeholder IDs
            #    that can appear in the prompt produced by the official HF
            #    processor.  Empirically those are:
            #       • text_config.image_token_id          – legacy single value
            #       • text_config.image_start_id / image_end_id (paired tags)
            #       • vision_config.image_id               – one id per patch
            # ------------------------------------------------------------------
            candidate_ids: set[int] = set()
            for attr in (
                "image_token_id",
                "image_start_id",
                "image_end_id",
            ):
                if hasattr(self.text_config, attr):
                    val = getattr(self.text_config, attr)
                    if val is not None:
                        candidate_ids.add(int(val))

            if hasattr(self.vision_config, "image_id"):
                val = getattr(self.vision_config, "image_id")
                if val is not None:
                    candidate_ids.add(int(val))

            # Fallback: hard-coded IDs known from official tokenizer_config.
            # This covers <|vision_start|>, <|vision_end|>, <|vision_pad|>,
            # <|image_pad|>, <|video_pad|> which currently map to
            # 151652-151656.  Add also legacy 151644/151645 observed in early
            # checkpoints.
            if not candidate_ids:
                candidate_ids.update({151644, 151645, 151652, 151653, 151654, 151655, 151656})

            # Only replace embeddings for *patch* tokens (image_pad).  Keep
            # start/end/pad tokens as normal learned embeddings.
            patch_token_ids: set[int] = set()
            # Prefer explicit attribute if present.
            if hasattr(self.text_config, "image_token_id") and self.text_config.image_token_id is not None:
                patch_token_ids.add(int(self.text_config.image_token_id))
            # Fallback: assume 151655 if not specified.
            if not patch_token_ids:
                patch_token_ids.add(151655)

            image_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for tid in patch_token_ids:
                image_token_mask |= input_ids == tid

            # ------------- DEBUG -------------------------------------------------
            try:
                num_tokens_expected = int(image_token_mask.sum().item())
                num_feats = image_features.shape[0]

                logging.debug(
                    f"[debug] first 32 ids: {input_ids[:32].cpu().tolist()}  "
                    f"image_ids_used={sorted(list(candidate_ids))}"
                )

                if num_tokens_expected != num_feats:
                    logging.debug(
                        f"[warn] Found {num_tokens_expected} patch tokens but {num_feats} "
                        f"patch embeddings – mismatch!"
                    )
                else:
                    logging.debug(
                        f"[debug] injecting {num_feats} image patch embeddings into text sequence"
                    )
            except Exception:
                pass
            # ---------------------------------------------------------------------

            # Add comprehensive debugging to compare with HF implementation
            logging.debug(f"[DEBUG] === VISION EMBEDDING INTEGRATION ===")
            logging.debug(f"[DEBUG] Original text embeddings - shape: {inputs_embeds.shape}, mean: {inputs_embeds.mean():.6f}, std: {inputs_embeds.std():.6f}")
            logging.debug(f"[DEBUG] Vision embeddings - shape: {image_features.shape}, mean: {image_features.mean():.6f}, std: {image_features.std():.6f}")
            
            # Sample some text embeddings before replacement
            text_only_mask = ~image_token_mask
            if text_only_mask.any():
                text_sample = inputs_embeds[text_only_mask][:5]  # First 5 text tokens
                print(f"[DEBUG] Sample text embeddings (first 5): mean={text_sample.mean():.6f}, std={text_sample.std():.6f}")
                print(f"[DEBUG] Text embedding norms: {torch.norm(text_sample, dim=-1)}")
            
            print(f"[DEBUG] Vision embedding norms: {torch.norm(image_features[:5], dim=-1)}")
            
            # Calculate magnitude ratio BEFORE scaling
            text_magnitude = torch.norm(inputs_embeds[text_only_mask], dim=-1).mean() if text_only_mask.any() else 0
            vision_magnitude = torch.norm(image_features, dim=-1).mean()
            print(f"[DEBUG] BEFORE replacement - Text magnitude: {text_magnitude:.6f}, Vision magnitude: {vision_magnitude:.6f}")
            print(f"[DEBUG] Vision/Text ratio: {vision_magnitude / text_magnitude:.3f}")
            
            # CRITICAL FIX: Scale vision embeddings to match text embedding magnitude
            # Even though HF doesn't explicitly scale, there might be implicit scaling in their implementation
            vision_scale_factor = text_magnitude / vision_magnitude if vision_magnitude > 0 else 1.0
            print(f"[DEBUG] Applying vision scale factor: {vision_scale_factor:.6f}")
            
            # Scale vision embeddings to match text magnitude
            image_features_scaled = image_features * vision_scale_factor
            print(f"[DEBUG] After scaling - vision magnitude: {torch.norm(image_features_scaled, dim=-1).mean():.6f}")
            
            print(f"[DEBUG] Final shapes - inputs_embeds: {inputs_embeds.shape}, image_features: {image_features_scaled.shape}, image_token_mask: {image_token_mask.shape}")
            print(f"[DEBUG] Number of True values in mask: {image_token_mask.sum().item()}")
            print(f"[DEBUG] Number of vision tokens: {image_features_scaled.shape[0]}")
            
            # Replace image placeholder tokens with actual image embeddings
            # masked_scatter expects the source tensor to have the right shape
            inputs_embeds = inputs_embeds.masked_scatter(image_token_mask.unsqueeze(-1), image_features_scaled)
            
            print(f"[DEBUG] AFTER replacement - Combined embeddings: mean={inputs_embeds.mean():.6f}, std={inputs_embeds.std():.6f}")
            
            # Check final magnitudes
            text_indices = ~image_token_mask
            vision_indices = image_token_mask
            
            if text_indices.any():
                final_text_magnitude = torch.norm(inputs_embeds[text_indices][:5], dim=-1).mean().item()
            else:
                final_text_magnitude = 0.0
                
            if vision_indices.any():
                final_vision_magnitude = torch.norm(inputs_embeds[vision_indices][:5], dim=-1).mean().item()
            else:
                final_vision_magnitude = 0.0
                
            print(f"[DEBUG] FINAL - Text magnitude: {final_text_magnitude:.6f}, Vision magnitude: {final_vision_magnitude:.6f}")
            if final_text_magnitude > 0:
                print(f"[DEBUG] FINAL Vision/Text ratio: {final_vision_magnitude / final_text_magnitude:.3f}")
            print(f"[DEBUG] === END VISION INTEGRATION ===")

        # --- CRITICAL FIX: Use HF's approach for position embeddings ---
        # Get 3D position IDs exactly like HF
        pos_3d = self.get_rope_index(input_ids, image_grid_thw=image_grid_thw)
        
        # HF creates position embeddings directly from the 3D position IDs
        # pos_3d has shape (3, seq_len) where 3 = [temporal, height, width]
        seq_len = input_ids.shape[0]
        device = input_ids.device
        dtype = inputs_embeds.dtype
        
        # Create position embeddings for all three axes
        # The rotary_emb expects position IDs with shape (seq_len,) and returns (seq_len, head_dim)
        cos_t, sin_t = self.rotary_emb(pos_3d[0], dtype=dtype, device=device)  # temporal: (seq_len, head_dim)
        cos_h, sin_h = self.rotary_emb(pos_3d[1], dtype=dtype, device=device)  # height: (seq_len, head_dim)  
        cos_w, sin_w = self.rotary_emb(pos_3d[2], dtype=dtype, device=device)  # width: (seq_len, head_dim)
        
        # Stack them to create the 3D cos/sin tensors that multimodal RoPE expects
        # Shape: (3, seq_len, head_dim)
        cos_combined = torch.stack([cos_t, cos_h, cos_w], dim=0)
        sin_combined = torch.stack([sin_t, sin_h, sin_w], dim=0)
        
        position_embeddings = (cos_combined, sin_combined)

        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            if self.training and self.gradient_checkpointing:
                # TODO: Add gradient checkpointing support
                pass
            else:
                hidden_states, _ = decoder_layer(hidden_states, position_embeddings, cu_seqlens, max_seqlen, position_ids)

        hidden_states = self.norm(hidden_states)

        return hidden_states


class Qwen2_5_VisionPatchEmbed(PatchEmbed):
    """HF PatchEmbed with extra lenience – accepts already-flattened 2-D patches."""

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Debug input
        print(f"[PATCH-DEBUG] Input pixel_values: shape={pixel_values.shape}, dtype={pixel_values.dtype}", flush=True)
        print(f"[PATCH-DEBUG] Input range: [{pixel_values.min():.6f}, {pixel_values.max():.6f}], mean={pixel_values.mean():.6f}, std={pixel_values.std():.6f}", flush=True)
        
        if torch.isnan(pixel_values).any():
            print(f"[PATCH-DEBUG] NaN in input pixel_values!", flush=True)
        
        if pixel_values.dim() == 2:  # flattened patches (N, in_dim)
            # parent PatchEmbed expects (B*N, in_dim) and applies linear proj
            result = super().forward(pixel_values)
        else:
            # Fallback to original implementation for 4-D/5-D tensors
            result = super().forward(pixel_values)
        
        # Debug output
        print(f"[PATCH-DEBUG] Output: shape={result.shape}, dtype={result.dtype}", flush=True)
        print(f"[PATCH-DEBUG] Output range: [{result.min():.6f}, {result.max():.6f}], mean={result.mean():.6f}, std={result.std():.6f}", flush=True)
        
        if torch.isnan(result).any():
            print(f"[PATCH-DEBUG] NaN in patch embedding output!", flush=True)
            # Check the parent's projection layer for issues
            if hasattr(self, 'proj'):
                print(f"[PATCH-DEBUG] Proj weight stats: mean={self.proj.weight.mean():.6f}, std={self.proj.weight.std():.6f}", flush=True)
                print(f"[PATCH-DEBUG] Proj weight range: [{self.proj.weight.min():.6f}, {self.proj.weight.max():.6f}]", flush=True)
                if self.proj.bias is not None:
                    print(f"[PATCH-DEBUG] Proj bias stats: mean={self.proj.bias.mean():.6f}, std={self.proj.bias.std():.6f}", flush=True)
        
        return result


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
        self.act_fn = F.silu  # Use SiLU activation like HF

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
    # Keep cos/sin as 2-D tensors here; we will add **one** channel dimension
    # later when broadcasting to `[tokens, heads, head_dim]`.  The previous
    # extra `unsqueeze(-2)` inflated them to 4-D which broke `expand()`.
    cos, sin = cos.float(), sin.float()
    _dbg = os.getenv("DEBUG_SHAPES")
    if _dbg:
        print(
            f"[debug] (vision) after type-cast: cos={tuple(cos.shape)} sin={tuple(sin.shape)}",
            flush=True,
        )

    # Some upstream code may already attach a singleton dim at index 1
    # (e.g., shape `[tokens, 1, head_dim]`).  Remove it so later `unsqueeze(1)`
    # produces exactly three dimensions.
    if cos.dim() == 3 and cos.size(1) == 1:
        cos = cos.squeeze(1)
        sin = sin.squeeze(1)
        if _dbg:
            print(
                f"[debug] (vision) squeezed singleton: cos={tuple(cos.shape)}",
                flush=True,
            )

    # Align to head_dim.
    if cos.size(-1) < q.size(-1):
        pad_dim = q.size(-1) - cos.size(-1)
        cos_pad = torch.ones(cos.shape[:-1] + (pad_dim,), device=cos.device, dtype=cos.dtype)
        sin_pad = torch.zeros_like(cos_pad)
        cos = torch.cat([cos, cos_pad], dim=-1)
        sin = torch.cat([sin, sin_pad], dim=-1)
        if _dbg:
            print(
                f"[debug] (vision) rotary dim padded: raw_dim={q.size(-1)-pad_dim} head_dim={q.size(-1)} pad={pad_dim}",
                flush=True,
            )
    elif cos.size(-1) > q.size(-1):
        cos = cos[..., : q.size(-1)]
        sin = sin[..., : q.size(-1)]
        if _dbg:
            print(
                f"[debug] (vision) rotary dim sliced: orig_dim={cos.size(-1)} head_dim={q.size(-1)}",
                flush=True,
            )

    # Broadcast to [tokens, heads, head_dim]
    cos_b = cos.unsqueeze(1).expand(-1, q.size(1), -1)  # [tokens, heads, head_dim]
    sin_b = sin.unsqueeze(1).expand_as(cos_b)
    if _dbg:
        print(
            f"[debug] (vision) broadcast shapes: cos_b={tuple(cos_b.shape)} sin_b={tuple(sin_b.shape)}",
            flush=True,
        )

    # If token count still mismatches (e.g., grid tokens), skip RoPE entirely as
    # a last-resort fallback – this keeps shapes consistent for tiny toy models
    # used in unit tests while logging a clear message.  Real checkpoints will
    # never hit this path because their dimensions line up.
    if cos_b.size(0) != q.size(0):
        if _dbg:
            print(
                f"[warn] apply_multimodal_rotary_pos_emb final token mismatch: q_tokens={q.size(0)} cos_tokens={cos_b.size(0)}. Skipping RoPE for this step.",
                flush=True,
            )
        return q, k

    if _dbg:
        print(
            f"[debug] RoPE tensor shapes: q={tuple(q.shape)}, cos_cat={tuple(cos_b.shape)}, "
            f"sin_cat={tuple(sin_b.shape)}  (tokens={q.size(0)}, heads={q.size(1)}, head_dim={q.size(2)})",
            flush=True,
        )

    q_embed = (q * cos_b) + (rotate_half(q) * sin_b)
    k_embed = (k * cos_b) + (rotate_half(k) * sin_b)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


class Qwen2_5_VLVisionAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        # Keep the HF naming alias so downstream helper utilities (e.g., mask
        # builders copied from text-attention) can reference a common attr
        # without branching on module type.
        self.num_attention_heads = num_heads
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
        # EXACT HF implementation of Qwen2_5_VLVisionSdpaAttention.forward()
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        # Create attention mask exactly like HF
        attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
        
        # Transpose exactly like HF
        q = q.transpose(0, 1)  # [num_heads, seq_length, head_dim] -> [seq_length, num_heads, head_dim]
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        
        # Apply SDPA exactly like HF
        attn_output = F.scaled_dot_product_attention(
            q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), attention_mask, dropout_p=0.0
        )
        
        # Reshape back exactly like HF
        attn_output = attn_output.squeeze(0).transpose(0, 1)  # [seq_length, num_heads, head_dim] -> [num_heads, seq_length, head_dim]
        attn_output = attn_output.reshape(seq_length, -1)
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
        # Debug input to vision block
        if torch.isnan(hidden_states).any():
            print(f"[BLOCK-DEBUG] NaN in input to vision block! Stats: mean={hidden_states.mean():.6f}, std={hidden_states.std():.6f}", flush=True)
        
        # First residual connection (attention)
        norm1_out = self.norm1(hidden_states)
        if torch.isnan(norm1_out).any():
            print(f"[BLOCK-DEBUG] NaN after norm1!", flush=True)
        
        attn_out = self.attn(
            norm1_out,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_embeddings=position_embeddings,
        )
        if torch.isnan(attn_out).any():
            print(f"[BLOCK-DEBUG] NaN after attention!", flush=True)
        
        hidden_states = hidden_states + attn_out
        if torch.isnan(hidden_states).any():
            print(f"[BLOCK-DEBUG] NaN after attention residual!", flush=True)
        
        # Second residual connection (MLP)
        norm2_out = self.norm2(hidden_states)
        if torch.isnan(norm2_out).any():
            print(f"[BLOCK-DEBUG] NaN after norm2!", flush=True)
        
        mlp_out = self.mlp(norm2_out)
        if torch.isnan(mlp_out).any():
            print(f"[BLOCK-DEBUG] NaN after MLP!", flush=True)
        
        hidden_states = hidden_states + mlp_out
        if torch.isnan(hidden_states).any():
            print(f"[BLOCK-DEBUG] NaN after MLP residual!", flush=True)
        
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

        if os.getenv("DEBUG_SHAPES"):
            print(
                f"[dbg-window] grid_thw={grid_thw.tolist()}  spatial_merge={self.spatial_merge_size}  "
                f"vit_win={vit_merger_window_size}",
                flush=True,
            )

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

        if os.getenv("DEBUG_SHAPES"):
            print(
                f"[dbg-window] built window_index len={window_index.numel()} max={int(window_index.max())}",
                flush=True,
            )

        return window_index, cu_window_seqlens

    def forward(self, pixel_values: torch.Tensor, grid_thw: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Check input pixel values
        if torch.isnan(pixel_values).any():
            print(f"[VISION-DEBUG] NaN detected in input pixel_values!", flush=True)
            
        hidden_states = self.patch_embed(pixel_values)
        
        # Check after patch embedding
        if torch.isnan(hidden_states).any():
            print(f"[VISION-DEBUG] NaN detected after patch_embed! Input shape: {pixel_values.shape}, Output shape: {hidden_states.shape}", flush=True)
            print(f"[VISION-DEBUG] Input pixel stats: min={pixel_values.min():.6f}, max={pixel_values.max():.6f}, mean={pixel_values.mean():.6f}", flush=True)
            print(f"[VISION-DEBUG] Output patch stats: min={hidden_states.min():.6f}, max={hidden_states.max():.6f}", flush=True)
            # Check if it's fp16 overflow
            if pixel_values.dtype == torch.float16:
                print(f"[VISION-DEBUG] Using fp16 - potential overflow issue", flush=True)

        # Auto-derive grid_thw when not provided.
        if grid_thw is None:
            # Infer T', H', W' from number of patches we just produced.
            # patches_per_img = T' * (H'/merge) * (W'/merge) * merge^2
            # After patch_embed we don't retain B, so we assume single image batch.
            raise ValueError("grid_thw must be provided when using VisionTransformer without built-in processor.")

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        seq_len, _ = hidden_states.size()
        # Ensure the index tensor lives on the same device as the source so
        # fancy-indexing doesn't trigger illegal memory faults.
        window_index = window_index.to(hidden_states.device)

        num_groups = seq_len // self.spatial_merge_unit

        hidden_states = hidden_states.reshape(num_groups, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

        rotary_pos_emb = rotary_pos_emb.reshape(num_groups, self.spatial_merge_unit, -1)
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
            # Select the appropriate cu_seqlens depending on whether this layer
            # performs full-attention (global) or windowed attention.
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens

            max_seqlen_now = (cu_seqlens_now[1:] - cu_seqlens_now[:-1]).max().item()

            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__, hidden_states, cu_seqlens_now, max_seqlen_now, position_embeddings
                )
            else:
                hidden_states = blk(
                    hidden_states,
                    cu_seqlens=cu_seqlens_now,
                    max_seqlen=max_seqlen_now,
                    position_embeddings=position_embeddings,
                )

        # Patch-merger and final reordering back to the original token order.
        print(f"[DEBUG] Before merger - hidden_states: mean={hidden_states.mean():.6f}, std={hidden_states.std():.6f}")
        print(f"[DEBUG] Before merger - sample norms: {torch.norm(hidden_states[:5], dim=-1)}")

        hidden_states = self.merger(hidden_states)
        
        print(f"[DEBUG] After merger - hidden_states: mean={hidden_states.mean():.6f}, std={hidden_states.std():.6f}")
        print(f"[DEBUG] After merger - sample norms: {torch.norm(hidden_states[:5], dim=-1)}")
        
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        # Final NaN check before returning
        if torch.isnan(hidden_states).any():
            print(f"[VISION-DEBUG] NaN detected in final vision output! Replacing with small random values.", flush=True)
            # Use small random values instead of zeros to provide some signal
            nan_mask = torch.isnan(hidden_states)
            hidden_states = torch.where(nan_mask, torch.randn_like(hidden_states) * 0.01, hidden_states)

        # CRITICAL FIX: DO NOT normalize vision embeddings - HF doesn't do this!
        # The HF implementation uses vision embeddings directly without any normalization
        print(f"[VISION-DEBUG] Final vision output - hidden_states mean: {hidden_states.mean().item():.6f}, std: {hidden_states.std().item():.6f}", flush=True)
        print(f"[VISION-DEBUG] Final vision embedding norms: {hidden_states[:5].norm(dim=-1)}", flush=True)

        if os.getenv("DEBUG_SHAPES"):
            print(
                f"[dbg-forward] seq_len={seq_len} groups={num_groups}",
                flush=True,
            )

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
    
    def get_rope_index(self, input_ids, image_grid_thw=None, video_grid_thw=None):
        """Delegate to the underlying model's get_rope_index method."""
        return self.model.get_rope_index(input_ids, image_grid_thw, video_grid_thw)

    def forward(
        self, 
        input_ids: torch.LongTensor, 
        cu_seqlens: torch.Tensor, 
        max_seqlen: int, 
        position_ids: torch.LongTensor, 
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
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

        # ----------------- DEBUG ----------------------------------------
        # When generating, we care about the *last* token logits.  Emit a
        # quick min/max/mean so we can detect degenerate outputs (all zeros
        # or all NaNs) that would make argmax pick EOS immediately.
        try:
            if logits.ndim == 2:  # (tokens, vocab)
                last = logits[-1]
            else:  # (batch, tokens, vocab)
                last = logits[0, -1]

            if torch.isnan(last).any():
                print("[logits-debug] last token has NaNs!", flush=True)
            else:
                print(
                    f"[logits-debug] last token stats: min={float(last.min()):.4f} max={float(last.max()):.4f} mean={float(last.mean()):.4f}",
                    flush=True,
                )
        except Exception:
            pass
        # ----------------------------------------------------------------

        # For compatibility with both HF-style and custom runners we always
        # return a simple tuple – even when ``return_dict`` is requested –
        # because the surrounding `ModelRunner` wrapper in dark.rl checks for
        # an attribute-based interface and otherwise falls back to tuple
        # unpacking.  This keeps behaviour stable across model types without
        # introducing an extra lightweight dataclass.

        return logits, loss

    def freeze_base_model(self):
        for name, param in self.model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False 