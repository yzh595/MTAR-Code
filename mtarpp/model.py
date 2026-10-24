# Modified from:
#   VQGAN:    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/transformer/mingpt.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py
#   nanoGPT:  https://github.com/karpathy/nanoGPT/blob/master/model.py
#   llama:    https://github.com/facebookresearch/llama/blob/main/llama/model.py
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
#   PixArt:   https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py

from dataclasses import dataclass, field
from typing import Optional, List, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.drop_path import DropPath


def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)


def _parse_layer_list(x):
    if x is None:
        return []

    if isinstance(x, str):
        x = x.strip()
        if x == "":
            return []
        return [int(v.strip()) for v in x.split(",") if v.strip() != ""]

    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]

    return [int(x)]


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: Optional[int] = None
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None

    rope_base: float = 10000
    norm_eps: float = 1e-5
    initializer_range: float = 0.02

    token_dropout_p: float = 0.1
    attn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.1
    ffn_dropout_p: float = 0.1
    drop_path_rate: float = 0.0

    num_classes: int = 1000
    caption_dim: int = 2048
    class_dropout_prob: float = 0.1
    model_type: str = "c2i"

    vocab_size: int = 16384
    cls_token_num: int = 1
    block_size: int = 256
    max_batch_size: int = 32
    max_seq_len: int = 2048

    student_target_layers: Union[List[int], str] = field(default_factory=lambda: [3])
    teacher_target_layers: Union[List[int], str] = field(default_factory=list)

    teacher_dim: int = 1024
    projector_dim: int = 2048

    future_feature_pred_layer: int = 9
    future_feature_offsets: Union[List[int], str] = field(default_factory=lambda: [1, 16, 17])

    loss_weight_contrast: float = 0.0
    contrast_temp: float = 0.07
    contrast_dropout_p: float = 0.2
    contrast_num_samples: int = 2048

    input_token_drop_rate: float = 0.0
    max_physical_drop_rate: float = 0.0
    physical_drop_switch_ratio: float = 0.0
    semantic_drop_temperature: float = 0.06

    def __post_init__(self):
        self.student_target_layers = _parse_layer_list(self.student_target_layers)
        self.teacher_target_layers = _parse_layer_list(self.teacher_target_layers)
        self.future_feature_offsets = _parse_layer_list(self.future_feature_offsets)

        if len(self.student_target_layers) == 0:
            raise ValueError("student_target_layers should not be empty for KD model.")

        if self.teacher_target_layers and len(self.teacher_target_layers) != len(self.student_target_layers):
            raise ValueError(
                f"teacher_target_layers length {len(self.teacher_target_layers)} must match "
                f"student_target_layers length {len(self.student_target_layers)}. "
                f"Got student={self.student_target_layers}, teacher={self.teacher_target_layers}."
            )

        if not 1 <= self.future_feature_pred_layer <= self.n_layer:
            raise ValueError(f"future_feature_pred_layer must be in [1, {self.n_layer}].")
        if not self.future_feature_offsets or any(
            offset <= 0 for offset in self.future_feature_offsets
        ):
            raise ValueError("future_feature_offsets must contain positive offsets.")
        if any(offset >= self.block_size - 1 for offset in self.future_feature_offsets):
            raise ValueError(
                "future_feature_offsets must be smaller than block_size - 1."
            )


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()

        use_cfg_embedding = dropout_prob > 0

        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding,
            hidden_size,
        )

        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1

        labels = torch.where(drop_ids, self.num_classes, labels)

        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0

        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)

        embeddings = self.embedding_table(labels).unsqueeze(1)

        return embeddings


class CaptionEmbedder(nn.Module):
    def __init__(self, in_channels, hidden_size, uncond_prob, token_num=120):
        super().__init__()

        self.cap_proj = MLP(
            in_features=in_channels,
            hidden_features=hidden_size,
            out_features=hidden_size,
        )

        self.register_buffer(
            "uncond_embedding",
            nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5),
        )

        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1

        caption = torch.where(
            drop_ids[:, None, None],
            self.uncond_embedding,
            caption,
        )

        return caption

    def forward(self, caption, train, force_drop_ids=None):
        use_dropout = self.uncond_prob > 0

        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)

        embeddings = self.cap_proj(caption)

        return embeddings


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x


class ContrastiveHead(nn.Module):
    def __init__(self, dim: int, out_dim: int = 256, dropout_p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(
            torch.mean(x * x, dim=-1, keepdim=True) + self.eps
        )

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)

        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()

        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)

        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)

        hidden_dim = find_multiple(hidden_dim, config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)

        self.ffn_dropout = nn.Dropout(config.ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(
            self.w2(F.silu(self.w1(x)) * self.w3(x))
        )


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()

        cache_shape = (
            max_batch_size,
            n_head,
            max_seq_length,
            head_dim,
        )

        self.register_buffer(
            "k_cache",
            torch.zeros(cache_shape, dtype=dtype),
        )
        self.register_buffer(
            "v_cache",
            torch.zeros(cache_shape, dtype=dtype),
        )

    def update(self, input_pos, k_val, v_val):
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache

        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()

        assert config.dim % config.n_head == 0

        self.dim = config.dim
        self.head_dim = config.dim // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = (
            config.n_kv_head if config.n_kv_head is not None else config.n_head
        )

        total_kv_dim = (
            self.n_head + 2 * self.n_kv_head
        ) * self.head_dim

        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)

        self.kv_cache = None

        self.attn_dropout_p = config.attn_dropout_p
        self.resid_dropout = nn.Dropout(config.resid_dropout_p)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor = None,
        input_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        bsz, seqlen, _ = x.shape

        kv_size = self.n_kv_head * self.head_dim

        xq, xk, xv = self.wqkv(x).split(
            [self.dim, kv_size, kv_size],
            dim=-1,
        )

        xq = xq.view(
            bsz,
            seqlen,
            self.n_head,
            self.head_dim,
        )
        xk = xk.view(
            bsz,
            seqlen,
            self.n_kv_head,
            self.head_dim,
        )
        xv = xv.view(
            bsz,
            seqlen,
            self.n_kv_head,
            self.head_dim,
        )

        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        xq, xk, xv = map(
            lambda t: t.transpose(1, 2),
            (xq, xk, xv),
        )

        if self.kv_cache is not None and input_pos is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv

        keys = keys.repeat_interleave(
            self.n_head // self.n_kv_head,
            dim=1,
        )
        values = values.repeat_interleave(
            self.n_head // self.n_kv_head,
            dim=1,
        )

        if mask is not None and mask.dim() == 3:
            mask = mask[:, None, :, :]

        output = F.scaled_dot_product_attention(
            xq,
            keys,
            values,
            attn_mask=mask,
            is_causal=True if mask is None else False,
            dropout_p=self.attn_dropout_p if self.training else 0,
        )

        output = output.transpose(1, 2).contiguous().view(
            bsz,
            seqlen,
            self.dim,
        )

        output = self.resid_dropout(
            self.wo(output)
        )

        return output


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, drop_path: float):
        super().__init__()

        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)

        self.attention_norm = RMSNorm(
            config.dim,
            eps=config.norm_eps,
        )
        self.ffn_norm = RMSNorm(
            config.dim,
            eps=config.norm_eps,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        input_pos: Optional[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ):
        h = x + self.drop_path(
            self.attention(
                self.attention_norm(x),
                freqs_cis,
                input_pos,
                mask,
            )
        )

        out = h + self.drop_path(
            self.feed_forward(
                self.ffn_norm(h)
            )
        )

        return out


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()

        self.config = config

        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        self.block_size = config.block_size
        self.num_classes = config.num_classes
        self.model_type = config.model_type
        self.cls_token_num = config.cls_token_num

        self.input_token_drop_rate = config.input_token_drop_rate
        self.max_physical_drop_rate = config.max_physical_drop_rate
        self.physical_drop_switch_ratio = config.physical_drop_switch_ratio
        self.semantic_drop_temperature = config.semantic_drop_temperature

        if self.model_type == "c2i":
            self.cls_embedding = LabelEmbedder(
                config.num_classes,
                config.dim,
                config.class_dropout_prob,
            )
        elif self.model_type == "t2i":
            self.cls_embedding = CaptionEmbedder(
                config.caption_dim,
                config.dim,
                config.class_dropout_prob,
            )
        else:
            raise ValueError("please check model type")

        self.tok_embeddings = nn.Embedding(
            config.vocab_size,
            config.dim,
        )

        self.tok_dropout = nn.Dropout(
            config.token_dropout_p,
        )

        dpr = [
            x.item()
            for x in torch.linspace(
                0,
                config.drop_path_rate,
                config.n_layer,
            )
        ]

        self.layers = torch.nn.ModuleList()

        for layer_id in range(config.n_layer):
            self.layers.append(
                TransformerBlock(
                    config,
                    dpr[layer_id],
                )
            )

        self.student_target_layers = list(config.student_target_layers)
        self.teacher_target_layers = list(config.teacher_target_layers)

        for layer_idx in self.student_target_layers:
            assert 1 <= layer_idx <= config.n_layer, (
                f"student_target_layer {layer_idx} is out of range for "
                f"{config.n_layer}-layer model."
            )

        self.projectors = nn.ModuleList([
            MLP(
                in_features=config.dim,
                hidden_features=config.projector_dim,
                out_features=config.teacher_dim,
            )
            for _ in self.student_target_layers
        ])

        self.layer_to_projector_idx = {
            layer_id: projector_idx
            for projector_idx, layer_id in enumerate(self.student_target_layers)
        }

        self.norm = RMSNorm(
            config.dim,
            eps=config.norm_eps,
        )

        self.output = nn.Linear(
            config.dim,
            config.vocab_size,
            bias=False,
        )

        grid_size = int(self.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size, (
            f"block_size should be a square number for 2D RoPE, got {self.block_size}."
        )

        self.grid_size = grid_size

        self.freqs_cis = precompute_freqs_cis_2d(
            grid_size,
            self.config.dim // self.config.n_head,
            self.config.rope_base,
            self.cls_token_num,
        )

        self.max_batch_size = -1
        self.max_seq_length = -1

        self.initialize_weights()

        # Build and initialize the auxiliary head only after the complete old
        # model. Its constructor cannot advance the RNG used by shared weights.
        self.future_feature_pred_layer = config.future_feature_pred_layer
        self.future_feature_offsets = list(config.future_feature_offsets)
        self.future_feature_heads = nn.ModuleList([
            MLP(
                in_features=config.dim,
                hidden_features=config.projector_dim,
                out_features=config.teacher_dim,
            )
            for _ in self.future_feature_offsets
        ])
        self.future_feature_heads.apply(self._init_weights)

        self.loss_weight_contrast = float(config.loss_weight_contrast)
        self.contrast_temp = float(config.contrast_temp)
        self.contrast_num_samples = int(config.contrast_num_samples)
        if self.loss_weight_contrast > 0.0:
            self.contrast_head = ContrastiveHead(
                config.dim,
                out_dim=256,
                dropout_p=config.contrast_dropout_p,
            )
            self.contrast_head.apply(self._init_weights)
        else:
            self.contrast_head = None

    def initialize_weights(self):
        self.apply(self._init_weights)

        nn.init.constant_(
            self.output.weight,
            0,
        )

    def _init_weights(self, module):
        std = self.config.initializer_range

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0,
                std=std,
            )

            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0,
                std=std,
            )

    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        head_dim = self.config.dim // self.config.n_head

        max_seq_length = find_multiple(
            max_seq_length,
            8,
        )

        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size

        for b in self.layers:
            b.attention.kv_cache = KVCache(
                max_batch_size,
                max_seq_length,
                self.config.n_head,
                head_dim,
                dtype,
            )

        causal_mask = torch.tril(
            torch.ones(
                self.max_seq_length,
                self.max_seq_length,
                dtype=torch.bool,
            )
        )

        self.causal_mask = causal_mask.unsqueeze(0).repeat(
            self.max_batch_size,
            1,
            1,
        )

        grid_size = int(self.config.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size

        self.freqs_cis = precompute_freqs_cis_2d(
            grid_size,
            self.config.dim // self.config.n_head,
            self.config.rope_base,
            self.cls_token_num,
        )

    def _build_freqs_for_sparse_tokens(
        self,
        B: int,
        L_cond: int,
        ids_keep: torch.Tensor,
        device,
    ):
        """
        ids_keep:
            [B, K], image-token coordinate [0, N-1]

        return:
            [B, L_cond + K, 1, head_dim//2, 2]
        """
        ids_keep = ids_keep.to(device=device).long().contiguous()

        cond_pos = torch.arange(
            L_cond,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0).expand(B, -1)

        img_pos = ids_keep + self.cls_token_num

        full_pos = torch.cat(
            [cond_pos, img_pos],
            dim=1,
        )

        max_allowed = self.freqs_cis.shape[0] - 1

        if full_pos.min().item() < 0 or full_pos.max().item() > max_allowed:
            raise ValueError(
                f"Student sparse RoPE position out of range: "
                f"min={full_pos.min().item()}, max={full_pos.max().item()}, "
                f"allowed=[0,{max_allowed}], ids_keep.shape={ids_keep.shape}"
            )

        freqs = self.freqs_cis.to(device)[full_pos]

        return freqs.unsqueeze(2)

    def forward(
        self,
        idx: torch.Tensor,
        cond_idx: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        semantic_scores: Optional[torch.Tensor] = None,
        input_token_drop_rate: Optional[float] = None,
        ids_keep: Optional[torch.Tensor] = None,
        physical_sparse_input: bool = False,
        return_ids_keep: bool = True,
    ):
        """
        Two modes:

        1. Full-student mode, used by your current DINO sparse KD train script:
            physical_sparse_input = False
            idx       = z_indices[:, :-1]  [B, 255]
            targets   = z_indices          [B, 256]
            ids_keep  = DINO selected image token positions [B, K], range 0~254

            Student forward uses full AR sequence:
                h = [class, z0, z1, ..., z254]
                h length = 256

            CE:
                logits length = 256
                logits[:, i] predicts targets[:, i]
                full CE over z0~z255

            KD:
                ids_keep is image-token coordinate:
                    0 -> z0
                    1 -> z1
                    ...
                    254 -> z254

                student hidden coordinate:
                    h[:, 0] = class
                    h[:, 1] = z0
                    h[:, 2] = z1
                    ...
                    h[:, 255] = z254

                Therefore:
                    student KD position = ids_keep + cls_token_num

            This realizes:
                teacher selected z_i feature <-> student selected z_i feature

        2. Sparse-student compatibility mode:
            physical_sparse_input = True
            idx       = full z_indices [B, 256]
            ids_keep  = selected image token positions [B, K]

            Student physically keeps only class + selected image tokens.
            This mode is kept for compatibility, but your current train code does not use it.
        """

        # Official autoregressive prefill/decode path. During prefill only
        # cond_idx is present; during cached decoding only idx is present.
        if not self.training:
            if cond_idx is not None:
                h = self.cls_embedding(
                    cond_idx,
                    train=False,
                )[:, :self.cls_token_num]
            else:
                assert idx is not None
                h = self.tok_embeddings(idx)

            h = self.tok_dropout(h)
            self.freqs_cis = self.freqs_cis.to(h.device)

            if input_pos is None:
                freqs_cis = self.freqs_cis[:h.shape[1]]
                inference_mask = None
            else:
                freqs_cis = self.freqs_cis[input_pos]
                inference_mask = self.causal_mask[:, None, input_pos]

            for layer in self.layers:
                h = layer(
                    h,
                    freqs_cis,
                    input_pos,
                    inference_mask,
                )

            logits = self.output(self.norm(h)).float()
            return logits, None

        assert idx is not None and cond_idx is not None

        # =====================================================================================
        # Condition embedding
        # =====================================================================================

        cond_embeddings = self.cls_embedding(
            cond_idx,
            train=self.training,
        )[:, :self.cls_token_num]

        device = idx.device
        batch_size = idx.shape[0]

        self.freqs_cis = self.freqs_cis.to(device)

        # =====================================================================================
        # Prepare ids_keep
        # =====================================================================================

        if ids_keep is not None:
            ids_keep = ids_keep.to(device=device).long().contiguous()

            if ids_keep.dim() != 2:
                raise ValueError(f"ids_keep should be [B, K], got {ids_keep.shape}")

            if ids_keep.shape[0] != batch_size:
                raise ValueError(
                    f"ids_keep batch mismatch: ids_keep={ids_keep.shape}, batch={batch_size}"
                )

            if ids_keep.numel() == 0:
                raise ValueError("ids_keep is empty.")

        # =====================================================================================
        # Build token embeddings
        # =====================================================================================

        if physical_sparse_input and ids_keep is not None:
            # ---------------------------------------------------------------------------------
            # Sparse-student compatibility mode.
            #
            # idx should be full image tokens [B, N].
            # We physically gather selected image tokens.
            # ---------------------------------------------------------------------------------

            full_num_image_tokens = idx.shape[1]

            if ids_keep.min().item() < 0 or ids_keep.max().item() >= full_num_image_tokens:
                raise RuntimeError(
                    f"Student sparse input ids_keep out of range: "
                    f"min={ids_keep.min().item()}, max={ids_keep.max().item()}, "
                    f"allowed=[0,{full_num_image_tokens - 1}], "
                    f"ids_keep shape={tuple(ids_keep.shape)}, "
                    f"idx shape={tuple(idx.shape)}"
                )

            token_embeddings_full = self.tok_embeddings(idx)

            gather_index = ids_keep.unsqueeze(-1).expand(
                -1,
                -1,
                token_embeddings_full.shape[-1],
            )

            token_embeddings_sparse = torch.gather(
                token_embeddings_full,
                dim=1,
                index=gather_index,
            )

            token_embeddings = torch.cat(
                (
                    cond_embeddings,
                    token_embeddings_sparse,
                ),
                dim=1,
            )

            h = self.tok_dropout(token_embeddings)

            cond_pos = torch.arange(
                self.cls_token_num,
                device=device,
            ).unsqueeze(0).expand(batch_size, -1)

            img_pos = ids_keep + self.cls_token_num

            full_pos = torch.cat(
                [cond_pos, img_pos],
                dim=1,
            )

            freqs_cis = self.freqs_cis[full_pos].unsqueeze(2)

            mask = None

            kd_gather_mode = "sparse_input"

        else:
            # ---------------------------------------------------------------------------------
            # Full-student mode.
            #
            # idx should be AR input [B, 255].
            #
            # token_embeddings:
            #   [z0, z1, ..., z254]
            #
            # after concat:
            #   [class, z0, z1, ..., z254]
            #
            # h length:
            #   256
            # ---------------------------------------------------------------------------------

            token_embeddings = self.tok_embeddings(idx)

            token_embeddings = torch.cat(
                (
                    cond_embeddings,
                    token_embeddings,
                ),
                dim=1,
            )

            h = self.tok_dropout(token_embeddings)

            if input_pos is None:
                freqs_cis = self.freqs_cis[:token_embeddings.shape[1]]
            else:
                freqs_cis = self.freqs_cis[input_pos]

            kd_gather_mode = "full_ar"

            # 不再自动生成 ids_keep=0~255。
            # 新 train 脚本会始终传入 ids_keep：
            #   sparse 阶段: DINO top-k from 0~254
            #   full 阶段: 0~254 full range
            #
            # 这里保留 fallback，兼容旧训练脚本。
            if ids_keep is None and return_ids_keep:
                if targets is not None:
                    # token-level KD can only align input tokens:
                    #   [z0, ..., z254]
                    # not z255
                    num_kd_tokens = min(targets.shape[1] - 1, h.shape[1] - self.cls_token_num)
                else:
                    num_kd_tokens = h.shape[1] - self.cls_token_num

                ids_keep = torch.arange(
                    num_kd_tokens,
                    device=device,
                    dtype=torch.long,
                ).unsqueeze(0).repeat(batch_size, 1)

        # =====================================================================================
        # Transformer forward + collect student KD features
        # =====================================================================================

        zs_tilde = []
        future_feature_hidden = None

        for i, layer in enumerate(self.layers):
            h = layer(
                h,
                freqs_cis,
                input_pos,
                mask,
            )

            layer_id = i + 1

            if self.training and layer_id == self.future_feature_pred_layer:
                future_feature_hidden = h

            if self.training and layer_id in self.layer_to_projector_idx:
                projector_idx = self.layer_to_projector_idx[layer_id]

                if ids_keep is not None:
                    if kd_gather_mode == "full_ar":
                        # ---------------------------------------------------------------------
                        # Token-level KD under full AR student input.
                        #
                        # student input:
                        #   [class, z0, z1, ..., z254]
                        #
                        # h positions:
                        #   h[:, 0]   = class token feature
                        #   h[:, 1]   = z0 token feature
                        #   h[:, 2]   = z1 token feature
                        #   ...
                        #   h[:, 255] = z254 token feature
                        #
                        # ids_keep is image-token coordinate:
                        #   ids_keep = 0 -> z0
                        #   ids_keep = 1 -> z1
                        #   ...
                        #   ids_keep = 254 -> z254
                        #
                        # Therefore:
                        #   z_i feature in student hidden is h[:, i + cls_token_num]
                        #
                        # This realizes:
                        #   teacher selected z_i feature <-> student selected z_i feature
                        # ---------------------------------------------------------------------

                        ids_for_h = ids_keep + self.cls_token_num

                        max_pos = h.shape[1] - 1

                        if ids_for_h.min().item() < 0 or ids_for_h.max().item() > max_pos:
                            raise RuntimeError(
                                f"Student KD gather out of range: "
                                f"ids_keep min={ids_keep.min().item()}, "
                                f"ids_keep max={ids_keep.max().item()}, "
                                f"ids_for_h min={ids_for_h.min().item()}, "
                                f"ids_for_h max={ids_for_h.max().item()}, "
                                f"allowed=[0,{max_pos}], "
                                f"h shape={tuple(h.shape)}, "
                                f"idx shape={tuple(idx.shape)}, "
                                f"targets shape={None if targets is None else tuple(targets.shape)}"
                            )

                        gather_index_h = ids_for_h.unsqueeze(-1).expand(
                            -1,
                            -1,
                            h.shape[-1],
                        )

                        h_sparse = torch.gather(
                            h,
                            dim=1,
                            index=gather_index_h,
                        )

                        z_mapped_sparse = self.projectors[projector_idx](h_sparse)
                        zs_tilde.append(z_mapped_sparse)

                    else:
                        # ---------------------------------------------------------------------
                        # Sparse input student compatibility mode.
                        #
                        # h = [class, selected_z_i]
                        # remove class directly:
                        #   h[:, cls_token_num:, :]
                        # ---------------------------------------------------------------------

                        h_sparse = h[:, self.cls_token_num:, :]
                        z_mapped_sparse = self.projectors[projector_idx](h_sparse)
                        zs_tilde.append(z_mapped_sparse)

                else:
                    # -------------------------------------------------------------------------
                    # Fallback full KD.
                    #
                    # 如果没有传 ids_keep，就保持旧版 full-token KD 行为：
                    #   student h = [class, z0, ..., z254]
                    #
                    # 但在新的 DINO sparse KD train 中，ids_keep 应该永远不为 None。
                    # -------------------------------------------------------------------------

                    z_mapped = self.projectors[projector_idx](h)
                    zs_tilde.append(z_mapped)

        # =====================================================================================
        # Output + CE
        # =====================================================================================

        h = self.norm(h)

        logits = self.output(h)

        if self.training:
            if physical_sparse_input and ids_keep is not None:
                # Keep the official AR alignment under physical sparsity:
                # class predicts z0 and retained source z_i predicts z_{i+1}.
                logits_for_loss = logits.contiguous()

                if targets is not None:
                    condition_target = torch.zeros(
                        batch_size,
                        self.cls_token_num,
                        dtype=ids_keep.dtype,
                        device=device,
                    )
                    sparse_target_pos = torch.cat(
                        [condition_target, ids_keep + 1],
                        dim=1,
                    )
                    sparse_target_valid = sparse_target_pos < targets.shape[1]
                    safe_target_pos = sparse_target_pos.masked_fill(
                        ~sparse_target_valid,
                        0,
                    )
                    gather_targets = torch.gather(
                        targets,
                        dim=1,
                        index=safe_target_pos,
                    )
                else:
                    gather_targets = None
                    sparse_target_valid = None

            else:
                # Full AR mode:
                # h/logits length = 256.
                #
                # logits[:, 0] predicts z0
                # logits[:, 1] predicts z1
                # ...
                # logits[:, 255] predicts z255
                #
                # CE remains full 256-token CE.
                logits_for_loss = logits.contiguous()
                gather_targets = targets
        else:
            logits_for_loss = logits
            gather_targets = targets

        loss = None

        if valid is not None and gather_targets is not None:
            loss_all = F.cross_entropy(
                logits_for_loss.reshape(-1, logits_for_loss.size(-1)),
                gather_targets.reshape(-1),
                reduction="none",
            )

            valid_all = valid[:, None].repeat(
                1,
                gather_targets.shape[1],
            ).reshape(-1)

            loss = (loss_all * valid_all).sum() / max(
                valid_all.sum(),
                1,
            )

        elif gather_targets is not None:
            if physical_sparse_input and ids_keep is not None:
                token_loss = F.cross_entropy(
                    logits_for_loss.reshape(-1, logits_for_loss.size(-1)),
                    gather_targets.reshape(-1),
                    reduction="none",
                ).view_as(gather_targets)
                valid_weight = sparse_target_valid.to(token_loss.dtype)
                loss = (
                    (token_loss * valid_weight).sum()
                    / valid_weight.sum().clamp_min(1.0)
                )
            else:
                loss = F.cross_entropy(
                    logits_for_loss.reshape(-1, logits_for_loss.size(-1)),
                    gather_targets.reshape(-1),
                )

        # Match the two-item interface expected by the official autoregressive
        # generator. Training keeps the full KD/future-feature return contract.
        if not self.training:
            return logits_for_loss, None

        future_feature_predictions = []
        if self.training:
            if future_feature_hidden is None:
                raise RuntimeError(
                    "Future-feature hidden state for layer "
                    f"{self.future_feature_pred_layer} was not captured."
                )

            image_hidden = future_feature_hidden[:, self.cls_token_num:, :]
            # Return one prediction per retained Student source. The trainer
            # maps each source's original coordinate to its Teacher target.
            for head in self.future_feature_heads:
                future_feature_predictions.append(head(image_hidden))

        contrast_loss = h.new_tensor(0.0)
        if self.training and self.contrast_head is not None:
            image_tokens = h[:, self.cls_token_num:, :]
            flat_features = image_tokens.flatten(0, 1)
            if flat_features.shape[0] > self.contrast_num_samples:
                sample_ids = torch.randperm(
                    flat_features.shape[0],
                    device=flat_features.device,
                )[: self.contrast_num_samples]
                selected_features = flat_features[sample_ids]
            else:
                selected_features = flat_features

            z1 = F.normalize(self.contrast_head(selected_features), dim=-1)
            z2 = F.normalize(self.contrast_head(selected_features), dim=-1)
            similarities = torch.matmul(z1, z2.transpose(0, 1)) / self.contrast_temp
            contrast_labels = torch.arange(
                similarities.shape[0],
                device=similarities.device,
            )
            contrast_loss = F.cross_entropy(similarities, contrast_labels)

        return (
            logits_for_loss,
            zs_tilde,
            loss,
            ids_keep,
            future_feature_predictions,
            contrast_loss,
        )

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        modules = list(self.layers) + list(self.future_feature_heads)
        if self.contrast_head is not None:
            modules.append(self.contrast_head)
        return modules


def precompute_freqs_cis(
    seq_len: int,
    n_elem: int,
    base: int = 10000,
    cls_token_num=120,
):
    freqs = 1.0 / (
        base ** (
            torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem
        )
    )

    t = torch.arange(
        seq_len,
        device=freqs.device,
    )

    freqs = torch.outer(
        t,
        freqs,
    )

    freqs_cis = torch.polar(
        torch.ones_like(freqs),
        freqs,
    )

    cache = torch.stack(
        [
            freqs_cis.real,
            freqs_cis.imag,
        ],
        dim=-1,
    )

    cond_cache = torch.cat(
        [
            torch.zeros(cls_token_num, n_elem // 2, 2),
            cache,
        ]
    )

    return cond_cache


def precompute_freqs_cis_2d(
    grid_size: int,
    n_elem: int,
    base: int = 10000,
    cls_token_num=120,
):
    half_dim = n_elem // 2

    freqs = 1.0 / (
        base ** (
            torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim
        )
    )

    t = torch.arange(
        grid_size,
        device=freqs.device,
    )

    freqs = torch.outer(
        t,
        freqs,
    )

    freqs_grid = torch.concat(
        [
            freqs[:, None, :].expand(-1, grid_size, -1),
            freqs[None, :, :].expand(grid_size, -1, -1),
        ],
        dim=-1,
    )

    cache_grid = torch.stack(
        [
            torch.cos(freqs_grid),
            torch.sin(freqs_grid),
        ],
        dim=-1,
    )

    cache = cache_grid.flatten(0, 1)

    cond_cache = torch.cat(
        [
            torch.zeros(cls_token_num, n_elem // 2, 2),
            cache,
        ]
    )

    return cond_cache


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
):
    """
    x:
        [B, T, H, D]

    freqs_cis:
        full:
            [T, D//2, 2]

        sparse:
            [B, T, 1, D//2, 2]
    """
    xshaped = x.float().reshape(
        *x.shape[:-1],
        -1,
        2,
    )

    if freqs_cis.ndim == 3:
        freqs_cis = freqs_cis.view(
            1,
            xshaped.size(1),
            1,
            xshaped.size(3),
            2,
        )
    elif freqs_cis.ndim == 5:
        pass
    else:
        raise ValueError(
            f"Unsupported freqs_cis ndim={freqs_cis.ndim}, shape={freqs_cis.shape}"
        )

    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0]
            - xshaped[..., 1] * freqs_cis[..., 1],

            xshaped[..., 1] * freqs_cis[..., 0]
            + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        dim=-1,
    )

    x_out2 = x_out2.flatten(3)

    return x_out2.type_as(x)


# =========================================================================================
# Model Zoo
# =========================================================================================

def GPT_7B(**kwargs):
    return Transformer(
        ModelArgs(
            n_layer=32,
            n_head=32,
            dim=4096,
            **kwargs,
        )
    )


def GPT_3B(**kwargs):
    return Transformer(
        ModelArgs(
            n_layer=24,
            n_head=32,
            dim=3200,
            **kwargs,
        )
    )


def GPT_1B(**kwargs):
    return Transformer(
        ModelArgs(
            n_layer=22,
            n_head=32,
            dim=2048,
            **kwargs,
        )
    )


def GPT_XXXL(**kwargs):
    return Transformer(
        ModelArgs(
            n_layer=48,
            n_head=40,
            dim=2560,
            **kwargs,
        )
    )


def GPT_XXL(**kwargs):
    return Transformer(
        ModelArgs(
            n_layer=48,
            n_head=24,
            dim=1536,
            **kwargs,
        )
    )


def GPT_XL(**kwargs):
    return Transformer(
        ModelArgs(
            n_layer=36,
            n_head=20,
            dim=1280,
            **kwargs,
        )
    )


def GPT_L(**kwargs):
    return Transformer(
        ModelArgs(
            n_layer=24,
            n_head=16,
            dim=1024,
            **kwargs,
        )
    )


def GPT_M(**kwargs):
    return Transformer(
        ModelArgs(
            n_layer=18,
            n_head=16,
            dim=1024,
            **kwargs,
        )
    )


def GPT_B(**kwargs):
    return Transformer(
        ModelArgs(
            n_layer=12,
            n_head=12,
            dim=768,
            **kwargs,
        )
    )


GPT_models = {
    "GPT-B": GPT_B,
    "GPT-M": GPT_M,
    "GPT-L": GPT_L,
    "GPT-XL": GPT_XL,
    "GPT-XXL": GPT_XXL,
    "GPT-XXXL": GPT_XXXL,
    "GPT-1B": GPT_1B,
    "GPT-3B": GPT_3B,
    "GPT-7B": GPT_7B,
}
