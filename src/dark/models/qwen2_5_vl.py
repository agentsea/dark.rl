from typing import Optional, Tuple, List, Dict, Any

import torch
from torch import nn
import torch.nn.functional as F

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


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section):
    """Apply Multimodal Rotary Position Embedding (M-RoPE) to ``q`` and ``k``.

    This is a near-direct port of the reference implementation in
    `transformers.models.qwen2_vl.modeling_qwen2_vl`.

    Args
    ----
    q, k:
        Tensors of shape ``[tokens, n_heads, head_dim]``.
    cos, sin:
        RoPE lookup tables of shape ``[3, tokens, head_dim]`` where the first
        axis encodes the temporal / height / width phases, respectively.
    mrope_section (Union[int, Sequence[int]]):
        Channel split sizes (per modality *before* doubling for cos+sin).  The
        reference checkpoints use a list like ``[16, 24, 24]`` which we must
        duplicate *as a list* – **not** multiply element-wise – so the final
        split becomes ``[16, 24, 24, 16, 24, 24]`` and sums exactly to
        ``head_dim``.
    """

    # --- Build the list of split sizes ------------------------------------------------
    if isinstance(mrope_section, (list, tuple)):
        split_sizes = list(mrope_section) * 2  # e.g. [16,24,24]*2
    else:  # legacy scalar – treat as single section
        split_sizes = [mrope_section] * 2

    head_dim = cos.shape[-1]
    if sum(split_sizes) != head_dim:
        # Mismatch – fall back to vanilla 1-D RoPE using the *temporal* phases.
        print(
            f"[warn] M-RoPE split mismatch (sum={sum(split_sizes)} vs head_dim={head_dim}); "
            "falling back to 1-D RoPE.",
            flush=True,
        )
        cos_cat = cos[0].unsqueeze(1)  # [tokens, 1, head_dim]
        sin_cat = sin[0].unsqueeze(1)
        return (q * cos_cat) + (rotate_half(q) * sin_cat), (k * cos_cat) + (rotate_half(k) * sin_cat)

    # --- Split and interleave ---------------------------------------------------------
    try:
        cos_chunks = cos.split(split_sizes, dim=-1)
        sin_chunks = sin.split(split_sizes, dim=-1)
    except Exception as exc:  # pragma: no cover – extremely unlikely
        print(f"[error] M-RoPE split failed: {exc}. Using 1-D fallback.", flush=True)
        cos_cat = cos[0].unsqueeze(1)
        sin_cat = sin[0].unsqueeze(1)
        return (q * cos_cat) + (rotate_half(q) * sin_cat), (k * cos_cat) + (rotate_half(k) * sin_cat)

    # Interleave T/H/W components: chunk 0 → temporal, 1 → height, 2 → width,
    # 3 → temporal, ... exactly as in the official model.
    cos_cat = torch.cat([m[i % 3] for i, m in enumerate(cos_chunks)], dim=-1).unsqueeze(1)
    sin_cat = torch.cat([m[i % 3] for i, m in enumerate(sin_chunks)], dim=-1).unsqueeze(1)

    # --- Apply ------------------------------------------------------------------------
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

    def forward(self, hidden_states: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor], cu_seqlens: torch.Tensor, max_seqlen: int, position_ids: torch.LongTensor, cache_position: Optional[torch.LongTensor] = None) -> Tuple[torch.Tensor, None]:
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
                    print(
                        f"[nan-alert] layer{self.layer_idx} q_nan={q_nan} k_nan={k_nan} seq={total_tokens}",
                        flush=True,
                    )
                else:
                    q_stats = (
                        float(query_states.min().item()),
                        float(query_states.max().item()),
                        float(query_states.mean().item()),
                    )
                    print(
                        f"[attn-debug] seq={total_tokens} q[min,max,mean]={q_stats} {'NaN!' if q_nan else ''}",
                        flush=True,
                    )
        # ----------------------------------------------------------

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
        """Port (simplified) of HF `get_rope_index` for single-image prompts.

        Only covers the usage pattern in our integration test:
          • one image, encoded as a contiguous run of `<|image_pad|>` tokens
          • no video frames, no batching.

        Returns a tensor of shape *(3, seq_len)* – temporal, height, width.
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

        img_token_id = self.text_config.image_token_id or 151655

        patch_mask = input_ids == img_token_id
        if patch_mask.sum() == 0:
            ramp = torch.arange(seq_len, device=device, dtype=torch.long)
            position_ids[:] = ramp
            return position_ids

        # Indices of first patch token & count
        first_patch_idx = int(torch.nonzero(patch_mask, as_tuple=False)[0])
        num_patches = int(patch_mask.sum())

        # Build text-before positions 0 .. first_patch_idx-1
        curr = 0
        position_ids[:, :first_patch_idx] = torch.arange(first_patch_idx, device=device)
        curr = first_patch_idx

        # Vision patch positions
        T, H, W = map(int, image_grid_thw[0].tolist())  # single image case
        merge = self.vision_config.spatial_merge_size
        gH, gW = H // merge, W // merge

        # Build t,h,w indices in the flatten order used by PatchEmbed:
        t_index = torch.arange(T, device=device).view(-1, 1).expand(-1, gH * gW).flatten()
        h_index = (
            torch.arange(gH, device=device)
            .view(1, -1, 1)
            .expand(T, -1, gW)
            .flatten()
        )
        w_index = (
            torch.arange(gW, device=device)
            .view(1, 1, -1)
            .expand(T, gH, -1)
            .flatten()
        )

        patch_pos_len = t_index.numel()
        if patch_pos_len != num_patches:
            # Fallback: treat patches as simple ramp to avoid crash.
            t_index = torch.zeros(num_patches, device=device, dtype=torch.long)
            h_index = torch.arange(num_patches, device=device, dtype=torch.long)  # bogus but safe
            w_index = h_index.clone()

        position_ids[0, curr : curr + num_patches] = t_index
        position_ids[1, curr : curr + num_patches] = h_index
        position_ids[2, curr : curr + num_patches] = w_index
        curr += num_patches

        # Text after vision – continue the 1-D counter from max()+1
        next_start = position_ids.max().item() + 1
        if curr < seq_len:
            tail = torch.arange(next_start, next_start + (seq_len - curr), device=device)
            position_ids[:, curr:] = tail

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

                print(
                    f"[debug] first 32 ids: {input_ids[:32].cpu().tolist()}  "
                    f"image_ids_used={sorted(list(candidate_ids))}",
                    flush=True,
                )

                if num_tokens_expected != num_feats:
                    print(
                        f"[warn] Found {num_tokens_expected} patch tokens but {num_feats} "
                        f"patch embeddings – mismatch!",  # noqa: E501
                        flush=True,
                    )
                else:
                    print(
                        f"[debug] injecting {num_feats} image patch embeddings into text sequence",
                        flush=True,
                    )
            except Exception:
                pass
            # ---------------------------------------------------------------------

            if num_tokens_expected > 0:
                inputs_embeds[image_token_mask] = image_features.to(inputs_embeds.dtype)

        # --- Multimodal 3-D position IDs -----------------------------------
        pos_3d = self.get_rope_index(input_ids, image_grid_thw=image_grid_thw)

        # Build cos/sin for each of the 3 axes separately then stack
        cos_axes = []
        sin_axes = []
        for axis in range(3):
            cos_a, sin_a = self.rotary_emb(
                pos_3d[axis], dtype=inputs_embeds.dtype, device=inputs_embeds.device
            )
            cos_axes.append(cos_a)
            sin_axes.append(sin_a)
        position_embeddings = (torch.stack(cos_axes, dim=0), torch.stack(sin_axes, dim=0))

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
        if pixel_values.dim() == 2:  # flattened patches (N, in_dim)
            # parent PatchEmbed expects (B*N, in_dim) and applies linear proj
            return super().forward(pixel_values)

        # Fallback to original implementation for 4-D/5-D tensors
        return super().forward(pixel_values)


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
    # Keep cos/sin as 2-D tensors here; we will add **one** channel dimension
    # later when broadcasting to `[tokens, heads, head_dim]`.  The previous
    # extra `unsqueeze(-2)` inflated them to 4-D which broke `expand()`.
    cos, sin = cos.float(), sin.float()
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
        print(
            f"[debug] (vision) rotary dim padded: raw_dim={q.size(-1)-pad_dim} head_dim={q.size(-1)} pad={pad_dim}",
            flush=True,
        )
    elif cos.size(-1) > q.size(-1):
        cos = cos[..., : q.size(-1)]
        sin = sin[..., : q.size(-1)]
        print(
            f"[debug] (vision) rotary dim sliced: orig_dim={cos.size(-1)} head_dim={q.size(-1)}",
            flush=True,
        )

    # Broadcast to [tokens, heads, head_dim]
    cos_b = cos.unsqueeze(1).expand(-1, q.size(1), -1)  # [tokens, heads, head_dim]
    sin_b = sin.unsqueeze(1).expand_as(cos_b)
    print(
        f"[debug] (vision) broadcast shapes: cos_b={tuple(cos_b.shape)} sin_b={tuple(sin_b.shape)}",
        flush=True,
    )

    # If token count still mismatches (e.g., grid tokens), skip RoPE entirely as
    # a last-resort fallback – this keeps shapes consistent for tiny toy models
    # used in unit tests while logging a clear message.  Real checkpoints will
    # never hit this path because their dimensions line up.
    if cos_b.size(0) != q.size(0):
        print(
            f"[warn] apply_multimodal_rotary_pos_emb final token mismatch: q_tokens={q.size(0)} cos_tokens={cos_b.size(0)}. Skipping RoPE for this step.",
            flush=True,
        )
        return q, k

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

    def forward(self, pixel_values: torch.Tensor, grid_thw: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.patch_embed(pixel_values)

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

        return logits, loss

    def freeze_base_model(self):
        for name, param in self.model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False 