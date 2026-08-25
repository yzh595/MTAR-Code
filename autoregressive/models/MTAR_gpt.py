from dataclasses import dataclass
from typing import Optional, List
import math
import os

import torch
import torch.nn as nn
from torch.nn import functional as F

# ==========================================
# 1. 基础组件
# ==========================================

try:
    from utils.drop_path import DropPath
except ImportError:
    class DropPath(nn.Module):
        def __init__(self, drop_prob=None):
            super().__init__()
            self.drop_prob = drop_prob
        def forward(self, x):
            return x

def find_multiple(n: int, k: int):
    if n % k == 0: return n
    return n + k - (n % k)

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
    model_type: str = 'c2i'
    vocab_size: int = 16384
    cls_token_num: int = 1
    block_size: int = 256
    max_batch_size: int = 32
    max_seq_len: int = 2048
    
    # --- Contrastive Learning Args ---
    loss_weight_contrast: float = 0.2
    contrast_temp: float = 0.07         
    contrast_dropout_p: float = 0.2
    contrast_num_samples: int = 2048

    # --- Semantic Drop Args ---
    input_token_drop_rate: float = 0.0
    semantic_drop_temperature: float = 0.0

class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None: drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else: drop_ids = force_drop_ids == 1
        return torch.where(drop_ids, self.num_classes, labels)
    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None): labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels).unsqueeze(1)

class CaptionEmbedder(nn.Module):
    def __init__(self, in_channels, hidden_size, uncond_prob, token_num=120):
        super().__init__()
        self.cap_proj = MLP(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size)
        self.register_buffer("uncond_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5))
        self.uncond_prob = uncond_prob
    def token_drop(self, caption, force_drop_ids=None):
        if force_drop_ids is None: drop_ids = torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
        else: drop_ids = force_drop_ids == 1
        return torch.where(drop_ids[:, None, None], self.uncond_embedding, caption)
    def forward(self, caption, train, force_drop_ids=None):
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None): caption = self.token_drop(caption, force_drop_ids)
        return self.cap_proj(caption)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
    def forward(self, x): return self.fc2(self.act(self.fc1(x)))

class ContrastiveHead(nn.Module):
    def __init__(self, dim, out_dim=256, dropout_p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x): return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
    def forward(self, x): return self._norm(x.float()).type_as(x) * self.weight

class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if config.ffn_dim_multiplier is not None: hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, config.multiple_of)
        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.ffn_dropout = nn.Dropout(config.ffn_dropout_p)
    def forward(self, x): return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))
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
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim
        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None
        self.attn_dropout_p = config.attn_dropout_p
        self.resid_dropout = nn.Dropout(config.resid_dropout_p)
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor = None, input_pos: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None):
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)
        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)
        
        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))
        if self.kv_cache is not None and input_pos is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        output = F.scaled_dot_product_attention(xq, keys, values, attn_mask=mask, is_causal=True if mask is None and input_pos is None else False, dropout_p=self.attn_dropout_p if self.training else 0)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        output = self.resid_dropout(self.wo(output))
        return output

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, drop_path: float):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None):
        h = x + self.drop_path(self.attention(self.attention_norm(x), freqs_cis, start_pos, mask))
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
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
        self.semantic_drop_temperature = getattr(config, 'semantic_drop_temperature', 0.0)
        
        self.contrast_num_samples = config.contrast_num_samples 
        
        if self.model_type == 'c2i':
            self.cls_embedding = LabelEmbedder(config.num_classes, config.dim, config.class_dropout_prob)
        elif self.model_type == 't2i':
            self.cls_embedding = CaptionEmbedder(config.caption_dim, config.dim, config.class_dropout_prob)
        else:
            raise Exception("please check model type")
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.tok_dropout = nn.Dropout(config.token_dropout_p)

        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]
        self.layers = torch.nn.ModuleList([TransformerBlock(config, dpr[i]) for i in range(config.n_layer)])

        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        
        self.output_t1 = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        self.head_td_layers = nn.ModuleList([
            TransformerBlock(config, drop_path=0.0) for _ in range(2)
        ])
        self.output_td = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.loss_weight_t1 = 1.0
        self.loss_weight_td = 0.1
        self.loss_weight_contrast = config.loss_weight_contrast

        if self.loss_weight_contrast > 0:
            self.contrast_head = ContrastiveHead(
                config.dim, 
                out_dim=256,
                dropout_p=config.contrast_dropout_p
            )
            self.contrast_temp = config.contrast_temp

        grid_size = int(self.block_size ** 0.5)
        self.grid_size = grid_size
        
        self.register_buffer("freqs_cis", precompute_freqs_cis_2d(self.grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num), persistent=False)
        
        self.max_batch_size = -1
        self.max_seq_length = -1
        self.causal_mask = None 
        self.register_buffer("causal_mask_template", torch.tril(torch.ones(2048, 2048, dtype=torch.bool)), persistent=False)

        self.initialize_weights()

    def initialize_weights(self):        
        self.apply(self._init_weights)
        nn.init.constant_(self.output_t1.weight, 0)
        nn.init.constant_(self.output_td.weight, 0)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
    
    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, b.attention.n_kv_head, head_dim, dtype)
        
        causal_mask_template = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
        self.register_buffer("causal_mask_template", causal_mask_template, persistent=False)

    def _build_freqs_for_train(self, B, L_cond, L_img, idx_device, ids_keep):
        cond_pos = torch.arange(L_cond, device=idx_device).unsqueeze(0).expand(B, -1)
        img_pos = ids_keep + self.cls_token_num
        full_pos = torch.cat([cond_pos, img_pos], dim=1) 
        freqs = self.freqs_cis[full_pos] 
        return freqs.unsqueeze(2)

    @torch._dynamo.disable
    def _mtp_ce_eager(self, logits: torch.Tensor, targets: torch.Tensor):
        """Avoid the unstable PyTorch 2.2.1 Inductor CE backward at 240 tokens."""
        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
        )

    def forward(
        self, 
        idx: torch.Tensor, 
        cond_idx: torch.Tensor,
        input_pos:  Optional[torch.Tensor] = None, 
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        semantic_scores: Optional[torch.Tensor] = None,
        ids_keep: Optional[torch.Tensor] = None,
        physical_sparse_input: bool = False,
    ):
        if self.training:
            assert idx is not None and cond_idx is not None and targets is not None
            cond_embeddings = self.cls_embedding(cond_idx, train=True)[:,:self.cls_token_num]
            token_embeddings = self.tok_embeddings(idx)
            target_tokens = targets
            L_cond = self.cls_token_num

            # =========================================================
            # 分支 1: Semantic Drop 模式 (Slow Path)
            # =========================================================
            if physical_sparse_input:
                B, L_img, D = token_embeddings.shape
                assert ids_keep is not None
                ids_keep = ids_keep.to(device=idx.device).long().contiguous()
                
                gather_index = ids_keep.unsqueeze(-1).expand(-1, -1, D)
                token_embeddings = torch.gather(token_embeddings, dim=1, index=gather_index)

                freqs_cis_to_use = self._build_freqs_for_train(B, L_cond, L_img, idx.device, ids_keep)
                h = torch.cat((cond_embeddings, token_embeddings), dim=1)
                h = self.tok_dropout(h)

                for layer in self.layers:
                    h = layer(h, freqs_cis_to_use)
                h_refined = self.norm(h)

                logits_t1 = self.output_t1(h_refined)
                # Preserve the original NTP selection over all 255 sources.
                # MTP attention remains causal with no explicit attention mask;
                # only selected tail positions lacking a +16 target are omitted
                # from the auxiliary loss below.
                h_for_td = h_refined
                for layer in self.head_td_layers:
                    h_for_td = layer(h_for_td, freqs_cis_to_use)
                logits_td = self.output_td(h_for_td)

                # Slow Path Loss Logic
                logits_t1_img = logits_t1[:, L_cond:, :]
                logits_td_img = logits_td[:, L_cond:, :]
                L_img_orig = target_tokens.shape[1]
                target_indices_t1 = ids_keep + 1
                mask_valid_t1 = target_indices_t1 < L_img_orig
                safe_indices_t1 = target_indices_t1.clone(); safe_indices_t1[~mask_valid_t1] = 0
                targets_kept_t1 = torch.gather(target_tokens, dim=1, index=safe_indices_t1)
                
                loss_t1_elem = F.cross_entropy(logits_t1_img.flatten(0, 1), targets_kept_t1.flatten(0, 1), reduction='none')
                loss_t1 = (loss_t1_elem * mask_valid_t1.flatten()).sum() / (mask_valid_t1.sum() + 1e-6)

                shift = self.grid_size
                target_indices_td = ids_keep + shift
                mask_valid_td = target_indices_td < L_img_orig
                safe_indices_td = target_indices_td.masked_fill(~mask_valid_td, 0)
                targets_td_kept = torch.gather(target_tokens, dim=1, index=safe_indices_td)
                loss_td_elem = F.cross_entropy(logits_td_img.flatten(0, 1), targets_td_kept.flatten(0, 1), reduction='none')
                valid_td_weight = mask_valid_td.flatten().to(loss_td_elem.dtype)
                loss_td = (loss_td_elem * valid_td_weight).sum() / valid_td_weight.sum().clamp_min(1.0)
                
                loss_contrast = torch.tensor(0.0, device=h.device)
                if self.loss_weight_contrast > 0:
                    image_tokens = h_refined[:, L_cond:, :]
                    B_curr, L_curr, _ = image_tokens.shape
                    flat_features = image_tokens.flatten(0, 1)
                    num_total_tokens = flat_features.shape[0]
                    
                    max_samples = self.contrast_num_samples 
                    
                    if num_total_tokens > max_samples:
                        perm = torch.randperm(num_total_tokens, device=idx.device)[:max_samples]
                        selected_features = flat_features[perm]
                    else:
                        selected_features = flat_features

                    z1 = self.contrast_head(selected_features)
                    z2 = self.contrast_head(selected_features)
                    z1 = F.normalize(z1, dim=-1); z2 = F.normalize(z2, dim=-1)
                    sim_matrix = torch.matmul(z1, z2.transpose(0, 1)) / self.contrast_temp
                    
                    # [已移除] 3x3 空间掩码逻辑
                    # 移除了: flat_spatial_ids, row_ids, col_ids, is_spatial_neighbor, mask_to_ignore, masked_fill
                    
                    labels_c = torch.arange(sim_matrix.shape[0], device=idx.device)
                    loss_contrast = F.cross_entropy(sim_matrix, labels_c)

                mtp_hidden_probe = h_for_td[:, L_cond::16, :].detach().float()
                mtp_logits_probe = logits_td_img[:, ::16, :512].detach().float()
                total_loss = self.loss_weight_t1 * loss_t1 + self.loss_weight_td * loss_td + self.loss_weight_contrast * loss_contrast
                all_losses = {
                    'total_loss': total_loss,
                    'loss_t1': loss_t1,
                    'loss_td': loss_td,
                    'loss_contrast': loss_contrast,
                    'mtp_hidden_rms': mtp_hidden_probe.square().mean().sqrt(),
                    'mtp_logits_rms': mtp_logits_probe.square().mean().sqrt(),
                    'mtp_logits_absmax': logits_td_img.detach().abs().amax().float(),
                }
                return logits_t1, all_losses

            # =========================================================
            # 分支 2: Fast Path (1.35 FPS 终极优化版)
            # =========================================================
            else:
                h = torch.cat((cond_embeddings, token_embeddings), dim=1)
                h = self.tok_dropout(h)
                
                freqs_cis_to_use = self.freqs_cis[:h.shape[1]].to(h.device)
                mask_to_use = None
                
                for layer in self.layers:
                    h = layer(h, freqs_cis_to_use, start_pos=input_pos, mask=mask_to_use)
                h_refined = self.norm(h)
                
                h_t1_input = h_refined[:, L_cond-1:, :] # class + z0...z254
                logits_t1_partial = self.output_t1(h_t1_input) # [B, 256, V]
                
                # Keep official backbone parity at class + 255 AR sources.
                # MTP only needs class + z0...z239 (241 states) for its 240
                # valid +16 targets; the causal tail cannot affect them.
                shift = self.grid_size
                mtp_loss_len = target_tokens.shape[1] - shift
                mtp_seq_len = L_cond + mtp_loss_len
                # The 240 supervised positions cannot attend to later states
                # under causal attention, so omit the unused z240...z254 tail.
                h_for_td = h_refined[:, :mtp_seq_len, :]
                freqs_td = freqs_cis_to_use[:mtp_seq_len]
                for layer in self.head_td_layers:
                    h_for_td = layer(
                        h_for_td,
                        freqs_td,
                        start_pos=input_pos,
                        mask=mask_to_use,
                    )
                h_td_input = h_for_td[:, L_cond:L_cond + mtp_loss_len, :] # [B, 240, D]
                logits_td_partial = self.output_td(h_td_input)

                vocab_size = self.vocab_size
                loss_t1 = F.cross_entropy(logits_t1_partial.reshape(-1, vocab_size), target_tokens.reshape(-1))
                loss_td = self._mtp_ce_eager(logits_td_partial, target_tokens[:, shift:])

                # Contrastive Loss (Fast Path)
                loss_contrast = torch.tensor(0.0, device=h_refined.device)
                if self.loss_weight_contrast > 0:
                    image_tokens = h_refined[:, self.cls_token_num:, :]
                    B, L, D = image_tokens.shape
                    
                    flat_tokens = image_tokens.reshape(-1, D)
                    
                    max_samples = self.contrast_num_samples 
                    
                    if B * L > max_samples:
                        perm = torch.randperm(B * L, device=h_refined.device)[:max_samples]
                        selected_tokens = flat_tokens[perm]
                    else:
                        selected_tokens = flat_tokens

                    z1 = self.contrast_head(selected_tokens)
                    z2 = self.contrast_head(selected_tokens)
                    z1 = F.normalize(z1, dim=-1); z2 = F.normalize(z2, dim=-1)
                    sim_matrix = torch.matmul(z1, z2.transpose(0, 1)) / self.contrast_temp
                    
                    # [已移除] 3x3 空间掩码逻辑
                    # 移除了: batch_ids, row_ids, is_spatial_neighbor, mask_to_ignore, masked_fill
                    
                    labels_c = torch.arange(sim_matrix.shape[0], device=sim_matrix.device)
                    loss_contrast = F.cross_entropy(sim_matrix, labels_c)
                
                mtp_hidden_probe = h_for_td[:, L_cond::16, :].detach().float()
                mtp_logits_probe = logits_td_partial[:, ::16, :512].detach().float()
                total_loss = self.loss_weight_t1 * loss_t1 + self.loss_weight_td * loss_td + self.loss_weight_contrast * loss_contrast
                all_losses = {
                    'total_loss': total_loss,
                    'loss_t1': loss_t1,
                    'loss_td': loss_td,
                    'loss_contrast': loss_contrast,
                    'mtp_hidden_rms': mtp_hidden_probe.square().mean().sqrt(),
                    'mtp_logits_rms': mtp_logits_probe.square().mean().sqrt(),
                    'mtp_logits_absmax': logits_td_partial.detach().abs().amax().float(),
                }
                    
                return logits_t1_partial.float(), all_losses

        else:
            # Inference Mode
            if cond_idx is not None: h = self.cls_embedding(cond_idx, train=False)[:,:self.cls_token_num]
            else: assert idx is not None; h = self.tok_embeddings(idx)
            mask_to_use = self.causal_mask_template[:self.max_seq_length, :self.max_seq_length].to(h.device)
            if input_pos is not None: mask_to_use = mask_to_use[None, None, input_pos]
            h = self.tok_dropout(h)
            if input_pos is not None: freqs_cis_to_use = self.freqs_cis.to(h.device)[input_pos]
            else: freqs_cis_to_use = self.freqs_cis[:h.shape[1]].to(h.device)

            for layer in self.layers:
                h = layer(h, freqs_cis_to_use, start_pos=input_pos, mask=mask_to_use)
            h = self.norm(h)
            logits = self.output_t1(h).float()
            return logits, None

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers) + list(self.head_td_layers)

def precompute_freqs_cis_2d(grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120):
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(grid_size, device=freqs.device); freqs = torch.outer(t, freqs)
    freqs_grid = torch.concat([freqs[:, None, :].expand(-1, grid_size, -1), freqs[None, :, :].expand(grid_size, -1, -1)], dim=-1)
    cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1); cache = cache_grid.flatten(0, 1)
    if cls_token_num > 0: return torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache])
    return cache

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    if freqs_cis.ndim == 3:
        freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack([xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1], xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],], dim=-1)
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)

def GPT_7B(**kwargs): return Transformer(ModelArgs(n_layer=32, n_head=32, dim=4096, **kwargs))
def GPT_3B(**kwargs): return Transformer(ModelArgs(n_layer=24, n_head=32, dim=3200, **kwargs))
def GPT_1B(**kwargs): return Transformer(ModelArgs(n_layer=22, n_head=32, dim=2048, **kwargs))
def GPT_XXXL(**kwargs): return Transformer(ModelArgs(n_layer=48, n_head=40, dim=2560, **kwargs))
def GPT_XXL(**kwargs): return Transformer(ModelArgs(n_layer=48, n_head=24, dim=1536, **kwargs))
def GPT_XL(**kwargs): return Transformer(ModelArgs(n_layer=36, n_head=20, dim=1280, **kwargs))
def GPT_L(**kwargs): return Transformer(ModelArgs(n_layer=24, n_head=16, dim=1024, **kwargs))
def GPT_B(**kwargs): return Transformer(ModelArgs(n_layer=12, n_head=12, dim=768, **kwargs))

GPT_models = {
    'GPT-B': GPT_B, 'GPT-L': GPT_L, 'GPT-XL': GPT_XL, 'GPT-XXL': GPT_XXL, 'GPT-XXXL': GPT_XXXL,
    'GPT-1B': GPT_1B, 'GPT-3B': GPT_3B, 'GPT-7B': GPT_7B, 
}
