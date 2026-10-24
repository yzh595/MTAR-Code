import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from glob import glob
from copy import deepcopy
import os
import time
import inspect
import importlib
import argparse
import random
import numpy as np
import torch.nn.functional as F

from utils.logger import create_logger
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from dataset.build import build_dataset

# Student: physically sparse AR input + grouped Teacher-feature heads
# importlib supports the public filename containing literal '+' characters.
GPT_models = importlib.import_module("autoregressive.models.MTAR++_gpt").GPT_models

# Teacher: physical sparse forward with ids_keep
from autoregressive.models.NAR_gpt import GPT_models as NAR_models


TEACHER_CONFIGS = {
    "NAR-L": {
        "model_name": "GPT-L",
        "dim": 1024,
        "n_layer": 24,
    },
    "NAR-XL": {
        "model_name": "GPT-XL",
        "dim": 1280,
        "n_layer": 36,
    },
    "NAR-XXL": {
        "model_name": "GPT-XXL",
        "dim": 1536,
        "n_layer": 48,
    },
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_layer_list(layer_str):
    if layer_str is None or str(layer_str).strip() == "":
        return []
    return [int(x.strip()) for x in str(layer_str).split(",") if x.strip() != ""]


def parse_float_list(value):
    if value is None or str(value).strip() == "":
        return []
    return [float(x.strip()) for x in str(value).split(",") if x.strip() != ""]


def creat_optimizer(model, weight_decay, learning_rate, betas, logger):
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": nodecay_params,
            "weight_decay": 0.0,
        },
    ]

    logger.info(
        f"num decayed parameter tensors: {len(decay_params)}, "
        f"with {sum(p.numel() for p in decay_params):,} parameters"
    )
    logger.info(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, "
        f"with {sum(p.numel() for p in nodecay_params):,} parameters"
    )

    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=betas,
        **extra_args,
    )

    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer


def get_epoch_drop_rate(
    current_epoch,
    total_epochs,
    max_drop_rate=0.5,
    switch_ratio=0.80,
):
    """
    前期 teacher sparse KD；
    后期 drop_rate = 0。

    注意：
        drop_rate=0 时，本代码不会 ids_keep=None，
        而是 ids_keep=0~254，使 teacher/student 继续走统一 selected-token KD 逻辑。
    """
    switch_epoch = int(total_epochs * switch_ratio)

    if current_epoch < switch_epoch:
        return max_drop_rate
    else:
        return 0.0


def build_dino_keep_indices(
    dino_scores,
    batch_size,
    num_image_tokens,
    drop_rate,
    device,
    temperature=0.0,
):
    """
    根据 DINO 分数决定保留哪些 image token。

    ids_keep 坐标:
        0 ~ num_image_tokens - 1

    在本实验中:
        num_image_tokens = 255
        对应 z0 ~ z254

    drop_rate:
        0.50 -> 保留约 128 / 255
        0.40 -> 保留约 153 / 255
    """
    drop_rate = float(drop_rate)
    drop_rate = max(0.0, min(drop_rate, 0.95))

    keep_num = int(round(num_image_tokens * (1.0 - drop_rate)))
    keep_num = max(1, min(keep_num, num_image_tokens))

    if keep_num == num_image_tokens:
        return (
            torch.arange(
                num_image_tokens,
                device=device,
                dtype=torch.long,
            )
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )

    if dino_scores is None:
        noise = torch.rand(
            batch_size,
            num_image_tokens,
            device=device,
        )
        ids_keep = torch.argsort(noise, dim=1)[:, :keep_num]
        ids_keep = torch.sort(ids_keep, dim=1).values
        return ids_keep.long().contiguous()

    if dino_scores.dim() > 2:
        dino_scores = dino_scores.reshape(dino_scores.shape[0], -1)

    dino_scores = dino_scores.to(device=device)

    if dino_scores.shape[0] != batch_size:
        dino_scores = dino_scores[:batch_size]

        if dino_scores.shape[0] < batch_size:
            pad_b = batch_size - dino_scores.shape[0]
            pad = torch.zeros(
                pad_b,
                dino_scores.shape[1],
                device=device,
                dtype=dino_scores.dtype,
            )
            dino_scores = torch.cat([dino_scores, pad], dim=0)

    if dino_scores.shape[1] > num_image_tokens:
        dino_scores = dino_scores[:, :num_image_tokens].contiguous()
    elif dino_scores.shape[1] < num_image_tokens:
        pad_len = num_image_tokens - dino_scores.shape[1]
        pad = torch.zeros(
            dino_scores.shape[0],
            pad_len,
            device=device,
            dtype=dino_scores.dtype,
        )
        dino_scores = torch.cat([dino_scores, pad], dim=1).contiguous()

    dino_scores = torch.nan_to_num(
        dino_scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if temperature is None or temperature <= 0:
        ids_keep = torch.topk(
            dino_scores,
            k=keep_num,
            dim=1,
            largest=True,
            sorted=False,
        ).indices
    else:
        # Power normalization requested for the public release. Evaluate in log
        # space for stability: p_i is proportional to max(score_i, 1e-9)^(1/T).
        scores = torch.clamp(dino_scores.float(), min=1e-9)
        probs = torch.softmax(torch.log(scores) / float(temperature), dim=1)

        ids_keep = torch.multinomial(
            probs,
            num_samples=keep_num,
            replacement=False,
        )

    # 保持 raster 顺序，保证 teacher selected token 顺序与 student gather 顺序一致。
    ids_keep = torch.sort(ids_keep, dim=1).values

    return ids_keep.long().contiguous()


def build_grouped_teacher_indices(
    dino_scores,
    batch_size,
    num_image_tokens,
    drop_rate,
    offsets,
    device,
):
    """Build deterministic Teacher groups under a physical token budget.

    A candidate anchor i owns {i, i+offset_1, ...}. Groups are ranked by
    their mean DINO score. A vectorized prefix search keeps the largest
    highest-scoring group prefix whose union fits the Teacher token budget.
    Remaining budget is filled with high-score context tokens; those context
    tokens are not promoted to distillation anchors.
    """
    if dino_scores is None:
        raise ValueError("Grouped Teacher drop requires DINO token scores.")

    drop_rate = max(0.0, min(float(drop_rate), 0.95))
    keep_num = int(round(num_image_tokens * (1.0 - drop_rate)))
    group_offsets = sorted(set([0] + [int(v) for v in offsets]))
    max_offset = max(group_offsets)
    num_candidates = num_image_tokens - max_offset
    if num_candidates <= 0:
        raise ValueError(f"No valid group anchors for N={num_image_tokens}, offsets={group_offsets}")
    if keep_num < len(group_offsets):
        raise ValueError(f"Teacher budget {keep_num} cannot hold one group of size " f"{len(group_offsets)}")

    scores = dino_scores.reshape(dino_scores.shape[0], -1).to(device=device)
    scores = scores[:batch_size, :num_image_tokens].float()
    scores = torch.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)

    anchor_pos = torch.arange(num_candidates, device=device, dtype=torch.long)
    offset_tensor = torch.tensor(group_offsets, device=device, dtype=torch.long)
    group_index = anchor_pos[:, None] + offset_tensor[None, :]
    group_scores = scores[:, group_index].mean(dim=-1)
    sorted_anchors = torch.argsort(group_scores, dim=1, descending=True)
    sorted_groups = sorted_anchors.unsqueeze(-1) + offset_tensor.view(1, 1, -1)
    rank = torch.arange(num_candidates, device=device).view(1, -1)

    def coverage_for_counts(counts):
        selected = rank < counts.unsqueeze(1)
        selected_groups = selected.unsqueeze(-1).expand_as(sorted_groups)
        coverage_count = torch.zeros(
            batch_size,
            num_image_tokens,
            device=device,
            dtype=torch.int16,
        )
        coverage_count.scatter_add_(
            1,
            sorted_groups.reshape(batch_size, -1),
            selected_groups.reshape(batch_size, -1).to(torch.int16),
        )
        coverage = coverage_count > 0
        return coverage, coverage.sum(dim=1)

    # Largest score-ranked prefix whose group union stays within keep_num.
    low = torch.zeros(batch_size, device=device, dtype=torch.long)
    high = torch.full_like(low, num_candidates + 1)
    for _ in range((num_candidates + 1).bit_length()):
        mid = (low + high) // 2
        _, token_count = coverage_for_counts(mid)
        fits = token_count <= keep_num
        low = torch.where(fits, mid, low)
        high = torch.where(fits, high, mid)

    group_keep_mask, group_token_count = coverage_for_counts(low)
    selected_rank = rank < low.unsqueeze(1)
    anchor_mask = torch.zeros(
        batch_size,
        num_image_tokens,
        device=device,
        dtype=torch.bool,
    )
    anchor_mask.scatter_(1, sorted_anchors, selected_rank)

    # Fill unused budget with deterministic high-score context tokens. They
    # improve sparse Teacher context but never become +0 KD anchors.
    remaining = keep_num - group_token_count
    fill_scores = scores.masked_fill(group_keep_mask, -torch.inf)
    fill_order = torch.argsort(fill_scores, dim=1, descending=True)
    fill_rank = torch.arange(num_image_tokens, device=device).view(1, -1)
    fill_selected = fill_rank < remaining.unsqueeze(1)
    fill_mask = torch.zeros_like(group_keep_mask)
    fill_mask.scatter_(1, fill_order, fill_selected)
    teacher_keep_mask = group_keep_mask | fill_mask

    final_count = teacher_keep_mask.sum(dim=1)
    if not torch.equal(final_count, torch.full_like(final_count, keep_num)):
        raise RuntimeError(f"Grouped Teacher budget mismatch: expected={keep_num}, " f"actual={final_count.tolist()}")

    teacher_ids_keep = torch.topk(
        teacher_keep_mask.float(),
        k=keep_num,
        dim=1,
        largest=True,
        sorted=False,
    ).indices
    teacher_ids_keep = torch.sort(teacher_ids_keep, dim=1).values.long().contiguous()

    return teacher_ids_keep, anchor_mask, low, group_token_count


def debug_tensor_stats(name, x, logger=None, rank=0, step=None, max_print=True):
    if not max_print or rank != 0:
        return

    prefix = f"[DEBUG step={step}] " if step is not None else "[DEBUG] "

    if x is None:
        msg = f"{prefix}{name}: None"
    elif not torch.is_tensor(x):
        msg = f"{prefix}{name}: type={type(x)}"
    else:
        with torch.no_grad():
            msg = f"{prefix}{name}: " f"shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}"

            if x.numel() > 0:
                try:
                    msg += f", min={x.min().item()}, max={x.max().item()}"
                except Exception as e:
                    msg += f", min/max failed: {e}"

    if logger is not None:
        logger.info(msg)
    else:
        print(msg, flush=True)


def sanitize_ids_keep(
    ids_keep,
    num_image_tokens,
    device,
    name="ids_keep",
    clamp=False,
    logger=None,
    rank=0,
    step=None,
):
    """
    ids_keep 永远表示 image token 坐标:
        0 ~ num_image_tokens - 1

    在本实验中:
        num_image_tokens = 255
        ids_keep range = 0 ~ 254
    """
    if ids_keep is None:
        return None

    ids_keep = ids_keep.to(device=device).long().contiguous()

    if ids_keep.dim() != 2:
        raise ValueError(f"{name} should be [B, K], got shape={ids_keep.shape}")

    if ids_keep.numel() == 0:
        raise ValueError(f"{name} is empty.")

    before_min = ids_keep.min().item()
    before_max = ids_keep.max().item()

    if clamp:
        ids_keep = ids_keep.clamp(
            min=0,
            max=num_image_tokens - 1,
        ).contiguous()

    after_min = ids_keep.min().item()
    after_max = ids_keep.max().item()

    if rank == 0 and step is not None:
        logger.info(
            f"[DEBUG step={step}] {name}: "
            f"shape={tuple(ids_keep.shape)}, "
            f"before_min={before_min}, before_max={before_max}, "
            f"after_min={after_min}, after_max={after_max}, "
            f"allowed=[0,{num_image_tokens - 1}]"
        )

    if after_min < 0 or after_max >= num_image_tokens:
        raise ValueError(
            f"{name} out of range: min={after_min}, max={after_max}, "
            f"allowed=[0, {num_image_tokens - 1}], shape={ids_keep.shape}"
        )

    return ids_keep


def build_sparse_proximity_mask(
    teacher_model,
    ids_keep,
    cls_token_num,
    full_seq_len,
    device,
):
    """
    从 teacher 原始 proximity_mask 中裁剪出 sparse teacher 使用的 mask。

    full teacher sequence:
        position 0                  -> class token
        position cls_token_num      -> z0
        position cls_token_num + 1  -> z1
        ...
        position cls_token_num + i  -> z_i

    ids_keep:
        [B, K], image-token coordinate
        0 -> z0
        1 -> z1
        ...

    sparse teacher sequence:
        [class, selected_z_i1, selected_z_i2, ...]

    return:
        sparse_mask: [B, cls_token_num + K, cls_token_num + K]

    关键修正：
        不要使用 full_mask.gather(1, row_idx).gather(2, col_idx)，
        因为第一次 gather 后最后一维已经变成 sparse_len，
        第二次再用 full-position index 会越界。

        正确做法是高级索引：
            sparse_mask[b, i, j] = full_mask[b, sparse_pos[b, i], sparse_pos[b, j]]
    """
    if ids_keep is None:
        return None

    ids_keep = ids_keep.to(device=device).long().contiguous()

    if ids_keep.dim() != 2:
        raise ValueError(f"ids_keep should be [B, K], got {ids_keep.shape}")

    B, K = ids_keep.shape

    full_mask = teacher_model.proximity_mask[
        :B,
        :full_seq_len,
        :full_seq_len,
    ].to(device)

    cls_pos = (
        torch.arange(
            cls_token_num,
            device=device,
            dtype=torch.long,
        )
        .unsqueeze(0)
        .expand(B, -1)
    )

    img_pos = ids_keep + cls_token_num

    sparse_pos = (
        torch.cat(
            [cls_pos, img_pos],
            dim=1,
        )
        .long()
        .contiguous()
    )

    sparse_len = sparse_pos.shape[1]

    min_pos = sparse_pos.min().item()
    max_pos = sparse_pos.max().item()

    if min_pos < 0 or max_pos >= full_seq_len:
        raise RuntimeError(
            f"sparse_pos out of range: "
            f"min={min_pos}, max={max_pos}, "
            f"allowed=[0,{full_seq_len - 1}], "
            f"sparse_pos shape={tuple(sparse_pos.shape)}, "
            f"ids_keep shape={tuple(ids_keep.shape)}"
        )

    batch_idx = (
        torch.arange(
            B,
            device=device,
            dtype=torch.long,
        )
        .view(B, 1, 1)
        .expand(B, sparse_len, sparse_len)
    )

    row_idx = sparse_pos.unsqueeze(2).expand(
        B,
        sparse_len,
        sparse_len,
    )

    col_idx = sparse_pos.unsqueeze(1).expand(
        B,
        sparse_len,
        sparse_len,
    )

    sparse_mask = full_mask[batch_idx, row_idx, col_idx]

    return sparse_mask.contiguous()


def check_vq_and_label_ids(
    z_indices,
    c_indices,
    vocab_size,
    num_classes,
    clamp=True,
    logger=None,
    rank=0,
    step=None,
):
    if z_indices.numel() == 0:
        raise ValueError("z_indices is empty.")
    if c_indices.numel() == 0:
        raise ValueError("c_indices is empty.")

    z_before_min = z_indices.min().item()
    z_before_max = z_indices.max().item()
    c_before_min = c_indices.min().item()
    c_before_max = c_indices.max().item()

    if clamp:
        z_indices = z_indices.clamp(0, vocab_size - 1).contiguous()
        c_indices = c_indices.clamp(0, num_classes - 1).contiguous()

    z_after_min = z_indices.min().item()
    z_after_max = z_indices.max().item()
    c_after_min = c_indices.min().item()
    c_after_max = c_indices.max().item()

    if rank == 0 and step is not None:
        logger.info(
            f"[DEBUG step={step}] z_indices: shape={tuple(z_indices.shape)}, "
            f"before=[{z_before_min},{z_before_max}], "
            f"after=[{z_after_min},{z_after_max}], "
            f"allowed=[0,{vocab_size - 1}]"
        )
        logger.info(
            f"[DEBUG step={step}] c_indices: shape={tuple(c_indices.shape)}, "
            f"before=[{c_before_min},{c_before_max}], "
            f"after=[{c_after_min},{c_after_max}], "
            f"allowed=[0,{num_classes - 1}]"
        )

    if z_after_min < 0 or z_after_max >= vocab_size:
        raise ValueError(
            f"Invalid VQ token ids: min={z_after_min}, max={z_after_max}, "
            f"allowed=[0, {vocab_size - 1}], shape={z_indices.shape}"
        )

    if c_after_min < 0 or c_after_max >= num_classes:
        raise ValueError(
            f"Invalid class ids: min={c_after_min}, max={c_after_max}, "
            f"allowed=[0, {num_classes - 1}], shape={c_indices.shape}"
        )

    return z_indices, c_indices


def check_feature_shapes(s_feats, t_feats, ids_keep, logger=None, rank=0, step=None):
    if len(s_feats) != len(t_feats):
        raise RuntimeError(f"Student features and teacher features mismatch: " f"{len(s_feats)} vs {len(t_feats)}")

    for i, (s_feat, t_feat) in enumerate(zip(s_feats, t_feats)):
        if rank == 0 and step is not None:
            logger.info(
                f"[DEBUG step={step}] feature[{i}]: "
                f"s_feat={tuple(s_feat.shape)}, "
                f"t_feat={tuple(t_feat.shape)}, "
                f"ids_keep={None if ids_keep is None else tuple(ids_keep.shape)}"
            )

        if s_feat.shape != t_feat.shape:
            raise RuntimeError(
                f"Feature shape mismatch at pair {i}: "
                f"student={s_feat.shape}, teacher={t_feat.shape}, "
                f"ids_keep={None if ids_keep is None else ids_keep.shape}"
            )


def unpack_batch(batch):
    if len(batch) == 3:
        x, y, dino_scores = batch
    elif len(batch) == 2:
        x, y = batch
        dino_scores = None
    else:
        raise RuntimeError(f"Unexpected batch length: {len(batch)}")
    return x, y, dino_scores


def get_raw_model(model):
    raw_model = model.module
    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod
    return raw_model


def should_debug(args, rank, train_steps):
    return rank == 0 and args.debug_first_steps > 0 and train_steps < args.debug_first_steps


def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    checkpoint_steps = {int(step.strip()) for step in args.checkpoint_steps.split(",") if step.strip()}

    init_distributed_mode(args)

    assert args.global_batch_size % dist.get_world_size() == 0, "Batch size must be divisible by world size."

    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()

    seed = args.global_seed * dist.get_world_size() + rank
    set_seed(seed)
    torch.cuda.set_device(device)

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)

        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.gpt_model.replace("/", "-")

        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        time_record = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

        cloud_results_dir = f"{args.cloud_save_path}/{time_record}"
        cloud_checkpoint_dir = f"{cloud_results_dir}/{experiment_index:03d}-{model_string_name}/checkpoints"
        os.makedirs(cloud_checkpoint_dir, exist_ok=True)

        logger.info(f"Experiment directory created in cloud at {cloud_checkpoint_dir}")
    else:
        logger = create_logger(None)
        checkpoint_dir = None
        cloud_checkpoint_dir = None

    logger.info(f"{args}")
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    logger.info("Scheduled checkpoint steps: " + (", ".join(str(step) for step in sorted(checkpoint_steps)) or "none"))

    if args.drop_path_rate > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = args.dropout_p

    latent_size = args.image_size // args.downsample_size
    num_image_tokens = latent_size**2

    t_config = TEACHER_CONFIGS[args.teacher_model]

    student_target_layers = parse_layer_list(args.student_target_layers)
    teacher_target_layers = parse_layer_list(args.teacher_target_layers)
    future_feature_offsets = parse_layer_list(args.future_feature_offsets)
    future_feature_loss_weights = parse_float_list(args.future_feature_loss_weights)

    assert len(student_target_layers) > 0, "student_target_layers cannot be empty."
    assert len(teacher_target_layers) > 0, "teacher_target_layers cannot be empty."
    assert len(student_target_layers) == len(
        teacher_target_layers
    ), f"student_target_layers and teacher_target_layers must have same length."

    assert min(student_target_layers) >= 1
    assert min(teacher_target_layers) >= 1

    student_depth_by_model = {
        "GPT-B": 12,
        "GPT-M": 18,
        "GPT-L": 24,
        "GPT-XL": 36,
        "GPT-XXL": 48,
        "GPT-XXXL": 48,
        "GPT-1B": 22,
        "GPT-3B": 24,
        "GPT-7B": 32,
    }
    assert args.gpt_model in student_depth_by_model, f"Unknown student model depth for {args.gpt_model}"
    student_depth = student_depth_by_model[args.gpt_model]
    assert (
        max(student_target_layers) <= student_depth
    ), f"{args.gpt_model} student only has {student_depth} layers, got {max(student_target_layers)}"
    assert (
        1 <= args.future_feature_pred_layer <= student_depth
    ), f"future_feature_pred_layer must be within [1, {student_depth}] for {args.gpt_model}"

    assert (
        max(teacher_target_layers) <= t_config["n_layer"]
    ), f"{args.teacher_model} only has {t_config['n_layer']} layers"
    assert len(future_feature_offsets) == len(
        future_feature_loss_weights
    ), "future_feature_offsets and future_feature_loss_weights must have the same length"
    assert args.max_student_drop_rate == 0.5, "This controlled experiment requires Student SD drop rate 0.50."
    assert args.max_physical_drop_rate in (
        0.4,
        0.5,
        0.7,
    ), "Teacher GroupDrop sweep only supports controlled rates 0.40, 0.50 and 0.70."

    logger.info("Layer-wise distillation mapping:")
    for s_l, t_l in zip(student_target_layers, teacher_target_layers):
        logger.info(f"  Student layer {s_l}  <--  Teacher layer {t_l}")

    init_student_drop_rate = get_epoch_drop_rate(
        0,
        args.epochs,
        max_drop_rate=args.max_student_drop_rate,
        switch_ratio=args.student_drop_switch_ratio,
    )

    logger.info(
        f"DINO sparse token-level KD config: distill_loss=normalized_mse_unit_l2, "
        f"student_input=SD50_physical_sparse_then_full, "
        f"teacher_input=group_topk_physical_sparse, "
        f"teacher_mask=sparse_proximity_mask, "
        f"student_ce=original_next_token_targets, "
        f"kd_image_range=z0~z254, "
        f"num_image_tokens={num_image_tokens}, "
        f"student_drop={args.max_student_drop_rate}, "
        f"student_switch_ratio={args.student_drop_switch_ratio}, "
        f"init_student_drop={init_student_drop_rate}, "
        f"student_temperature={args.semantic_drop_temperature}, "
        f"student_sampling=power_normalized, "
        f"teacher_drop={args.max_physical_drop_rate}, "
        f"teacher_temperature=0.0, "
        f"teacher_group_offsets={[0] + future_feature_offsets}, "
        f"kd_alignment=student_teacher_anchor_intersection"
    )
    logger.info(
        f"Final-layer teacher-feature prediction config: "
        f"student_layer={args.future_feature_pred_layer}, "
        f"teacher_layer={teacher_target_layers[0]}, "
        f"offsets={future_feature_offsets}, "
        f"weights={future_feature_loss_weights}, "
        f"loss=normalized_mse_unit_l2"
    )

    model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=num_image_tokens,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        resid_dropout_p=dropout_p,
        ffn_dropout_p=dropout_p,
        drop_path_rate=args.drop_path_rate,
        token_dropout_p=args.token_dropout_p,
        student_target_layers=student_target_layers,
        teacher_dim=t_config["dim"],
        projector_dim=args.projector_dim,
        future_feature_pred_layer=args.future_feature_pred_layer,
        future_feature_offsets=future_feature_offsets,
        loss_weight_contrast=args.loss_weight_contrast,
        contrast_temp=args.contrast_temp,
        contrast_dropout_p=args.contrast_dropout_p,
        contrast_num_samples=args.contrast_num_samples,
        input_token_drop_rate=0.0,
        semantic_drop_temperature=args.semantic_drop_temperature,
    ).to(device)

    logger.info(f"GPT Student Parameters: {sum(p.numel() for p in model.parameters()):,}")
    if args.loss_weight_contrast <= 0.0 or model.contrast_head is None:
        raise RuntimeError("This training entry point requires an active TCR head.")
    tcr_params = sum(p.numel() for p in model.contrast_head.parameters())
    logger.info(
        f"TCR config: weight={args.loss_weight_contrast}, "
        f"temperature={args.contrast_temp}, dropout={args.contrast_dropout_p}, "
        f"samples={args.contrast_num_samples}, params={tcr_params:,}"
    )
    logger.info(f"Student target layers: {student_target_layers}")
    logger.info(f"Teacher target layers: {teacher_target_layers}")
    logger.info(f"Teacher feature dim: {t_config['dim']}")

    teacher_model = None
    teacher_features = {}

    if args.teacher_ckpt is not None:
        logger.info(f"Loading Teacher model {args.teacher_model} from {args.teacher_ckpt}")

        teacher_model = NAR_models[t_config["model_name"]](
            vocab_size=args.vocab_size,
            block_size=num_image_tokens,
            num_classes=args.num_classes,
            cls_token_num=args.cls_token_num,
            model_type=args.gpt_type,
            resid_dropout_p=0.0,
            ffn_dropout_p=0.0,
            drop_path_rate=0.0,
            token_dropout_p=0.0,
        ).to(device)

        checkpoint = torch.load(args.teacher_ckpt, map_location="cpu")
        teacher_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

        missing, unexpected = teacher_model.load_state_dict(
            teacher_state_dict,
            strict=True,
        )

        logger.info("Teacher checkpoint loaded.")
        logger.info(f"Teacher missing keys: {len(missing)}")
        logger.info(f"Teacher unexpected keys: {len(unexpected)}")

        del checkpoint

        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

        ptdtype = {
            "none": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[args.mixed_precision]

        if args.mixed_precision == "bf16":
            teacher_model.bfloat16()
        elif args.mixed_precision == "fp16":
            teacher_model.half()

        # setup_caches 主要用于创建 teacher_model.proximity_mask。
        teacher_model.setup_caches(
            max_batch_size=int(args.global_batch_size // dist.get_world_size()),
            max_seq_length=args.max_seq_len,
            dtype=ptdtype,
        )

        teacher_model.to(device)

        # 关闭 KV cache，避免 full/sparse sequence forward 被 cache 干扰。
        for block in teacher_model.layers:
            block.attention.kv_cache = None

        def get_activation(name):
            def hook(module, input, output):
                teacher_features[name] = output.detach()

            return hook

        for layer_idx in teacher_target_layers:
            teacher_model.layers[layer_idx - 1].register_forward_hook(get_activation(f"layer_{layer_idx}"))

        logger.info(f"Teacher loaded and frozen. Hooking teacher layers: {teacher_target_layers}")

    if args.ema:
        ema = deepcopy(model).to(device)
        requires_grad(ema, False)
    else:
        ema = None

    optimizer = creat_optimizer(
        model,
        args.weight_decay,
        args.lr,
        (args.beta1, args.beta2),
        logger,
    )

    args.strict_semantic_data = True
    dataset = build_dataset(args)

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    if args.gpt_ckpt:
        checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")

        model.load_state_dict(checkpoint["model"], strict=False)

        if args.ema:
            ema.load_state_dict(
                checkpoint["ema"] if "ema" in checkpoint else checkpoint["model"],
                strict=False,
            )

        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

        train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.gpt_ckpt.split("/")[-1].split(".")[0])

        start_epoch = int(train_steps / int(len(dataset) / args.global_batch_size))
        train_steps = int(start_epoch * int(len(dataset) / args.global_batch_size))

        del checkpoint

        logger.info(f"Resume training from checkpoint: {args.gpt_ckpt}")
        logger.info(f"Resume from epoch={start_epoch}, steps={train_steps}")

    else:
        train_steps = 0
        start_epoch = 0

        if args.ema:
            update_ema(ema, model, decay=0)

    if not args.no_compile:
        # PyTorch 2.2's DDP graph partitioner cannot sort symbolic dimensions
        # when the compiled model returns multiple offset-dependent tensors.
        # Disable only DDP graph splitting; Inductor compilation remains enabled.
        torch._dynamo.config.optimize_ddp = False
        logger.info("torch.compile enabled with optimize_ddp=False")
        logger.info("compiling the model...")
        model = torch.compile(model)

    model = DDP(model.to(device), device_ids=[args.gpu])
    model.train()

    if args.ema:
        ema.eval()

    ptdtype = {
        "none": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[args.mixed_precision]

    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == "fp16"))

    running_loss = 0.0
    running_ce_loss = 0.0
    running_distill_loss = 0.0
    running_future_feature_loss = 0.0
    running_contrast_loss = 0.0
    running_student_drop_rate = 0.0
    running_teacher_drop_rate = 0.0
    running_student_tokens = 0.0
    running_teacher_tokens = 0.0
    running_teacher_groups = 0.0
    running_anchor_tokens = 0.0

    log_steps = 0
    start_time = time.time()

    logger.info(f"Training for {args.epochs} epochs...")

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)

        current_student_drop_rate = get_epoch_drop_rate(
            epoch,
            args.epochs,
            max_drop_rate=args.max_student_drop_rate,
            switch_ratio=args.student_drop_switch_ratio,
        )
        current_teacher_drop_rate = args.max_physical_drop_rate

        if rank == 0:
            logger.info(
                f"Beginning epoch {epoch} | "
                f"Student Drop Rate: {current_student_drop_rate:.4f} | "
                f"Teacher Group Drop Rate: {current_teacher_drop_rate:.4f} | "
                f"Student Semantic Temp: {args.semantic_drop_temperature:.4f} | "
                f"Teacher Temp: 0.0000"
            )

        for batch in loader:
            debug_step = train_steps if should_debug(args, rank, train_steps) else None

            x, y, dino_scores = unpack_batch(batch)

            x = x.to(device, non_blocking=True).long()
            y = y.to(device, non_blocking=True).long()

            z_indices = x.squeeze(1) if x.dim() == 3 else x
            z_indices = z_indices.reshape(z_indices.shape[0], -1).long().contiguous()
            c_indices = y.reshape(-1).long().contiguous()

            if z_indices.shape[1] != num_image_tokens:
                raise ValueError(
                    f"z_indices length mismatch: got {z_indices.shape}, " f"expected image tokens={num_image_tokens}"
                )

            z_indices, c_indices = check_vq_and_label_ids(
                z_indices=z_indices,
                c_indices=c_indices,
                vocab_size=args.vocab_size,
                num_classes=args.num_classes,
                clamp=True,
                logger=logger,
                rank=rank,
                step=debug_step,
            )

            if should_debug(args, rank, train_steps):
                debug_tensor_stats("raw dino_scores", dino_scores, logger, rank, train_steps)

            if dino_scores is not None:
                dino_scores = dino_scores.to(device, non_blocking=True)

                if dino_scores.dim() == 3 and dino_scores.shape[1] == 1:
                    dino_scores = dino_scores.squeeze(1)

                dino_scores = dino_scores.reshape(dino_scores.shape[0], -1).contiguous()

                dino_scores = torch.nan_to_num(
                    dino_scores,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

                if dino_scores.shape[1] > num_image_tokens:
                    dino_scores = dino_scores[:, :num_image_tokens].contiguous()
                elif dino_scores.shape[1] < num_image_tokens:
                    pad_len = num_image_tokens - dino_scores.shape[1]
                    pad = torch.zeros(
                        dino_scores.shape[0],
                        pad_len,
                        device=dino_scores.device,
                        dtype=dino_scores.dtype,
                    )
                    dino_scores = torch.cat([dino_scores, pad], dim=1).contiguous()

                if should_debug(args, rank, train_steps):
                    debug_tensor_stats("processed dino_scores", dino_scores, logger, rank, train_steps)
            else:
                dino_scores = None

            # Build independent Student SD and grouped Teacher coordinates in
            # the same original z0~z254 coordinate system.

            student_sparse_mode = current_student_drop_rate > 0.0

            num_kd_image_tokens = z_indices.shape[1] - 1  # 255, z0~z254

            if dino_scores is not None:
                kd_dino_scores = dino_scores[:, :num_kd_image_tokens].contiguous()
            else:
                kd_dino_scores = None

            if student_sparse_mode:
                student_ids_keep = build_dino_keep_indices(
                    dino_scores=kd_dino_scores,
                    batch_size=z_indices.shape[0],
                    num_image_tokens=num_kd_image_tokens,
                    drop_rate=current_student_drop_rate,
                    device=device,
                    temperature=args.semantic_drop_temperature,
                )
            else:
                student_ids_keep = (
                    torch.arange(
                        num_kd_image_tokens,
                        device=device,
                        dtype=torch.long,
                    )
                    .unsqueeze(0)
                    .repeat(z_indices.shape[0], 1)
                )

            student_ids_keep = sanitize_ids_keep(
                ids_keep=student_ids_keep,
                num_image_tokens=num_kd_image_tokens,
                device=device,
                name="student ids_keep",
                clamp=False,
                logger=logger,
                rank=rank,
                step=debug_step,
            )

            (
                teacher_base_ids_keep,
                teacher_anchor_mask,
                teacher_group_count,
                teacher_group_token_count,
            ) = build_grouped_teacher_indices(
                dino_scores=kd_dino_scores,
                batch_size=z_indices.shape[0],
                num_image_tokens=num_kd_image_tokens,
                drop_rate=current_teacher_drop_rate,
                offsets=future_feature_offsets,
                device=device,
            )
            teacher_base_ids_keep = sanitize_ids_keep(
                ids_keep=teacher_base_ids_keep,
                num_image_tokens=num_kd_image_tokens,
                device=device,
                name="grouped teacher ids_keep",
                clamp=False,
                logger=logger,
                rank=rank,
                step=debug_step,
            )
            anchor_at_student = torch.gather(
                teacher_anchor_mask,
                dim=1,
                index=student_ids_keep,
            )
            if not anchor_at_student.any(dim=1).all():
                raise RuntimeError("At least one sample has no Student/Teacher group-anchor intersection.")
            anchor_tokens_per_sample = anchor_at_student.sum(dim=1).float().mean()

            with torch.cuda.amp.autocast(dtype=ptdtype):
                # ============================================================
                # 1. Student forward
                #
                # Student:
                #   full AR input = z_indices[:, :-1]
                #   full CE target = z_indices
                #
                # Student KD gather 在 Know_17.py 内部:
                #   ids_for_h = ids_keep + cls_token_num
                # ============================================================

                student_idx = z_indices if student_sparse_mode else z_indices[:, :-1]
                (
                    _,
                    s_feats,
                    ce_loss,
                    returned_ids_keep,
                    future_feature_predictions,
                    contrast_loss,
                ) = model(
                    cond_idx=c_indices,
                    idx=student_idx,
                    targets=z_indices,
                    semantic_scores=None,
                    ids_keep=student_ids_keep,
                    physical_sparse_input=student_sparse_mode,
                    return_ids_keep=True,
                )

                returned_ids_keep = sanitize_ids_keep(
                    ids_keep=returned_ids_keep,
                    num_image_tokens=num_kd_image_tokens,
                    device=device,
                    name="student returned ids_keep",
                    clamp=False,
                    logger=logger,
                    rank=rank,
                    step=debug_step,
                )

                if returned_ids_keep.shape != student_ids_keep.shape:
                    raise RuntimeError(
                        f"returned ids_keep shape mismatch: "
                        f"returned={returned_ids_keep.shape}, student={student_ids_keep.shape}"
                    )

                if not torch.equal(returned_ids_keep, student_ids_keep):
                    raise RuntimeError("Student returned ids_keep differs from student_ids_keep.")

                # ============================================================
                # 2. Teacher forward
                #
                # Teacher physical sparse input:
                #   [class] + selected z_i
                #
                # Teacher mask:
                #   sparse_mask cropped from original NAR proximity_mask.
                #
                # 注意:
                #   不要 mask=None。
                #   mask=None 会触发 causal attention。
                # ============================================================

                t_feats = None

                if teacher_model is not None:
                    with torch.no_grad():
                        teacher_features.clear()

                        bs = z_indices.shape[0]
                        full_seq_len = args.cls_token_num + z_indices.shape[1]

                        terminal_teacher_id = torch.full(
                            (teacher_base_ids_keep.shape[0], 1),
                            z_indices.shape[1] - 1,
                            device=device,
                            dtype=torch.long,
                        )
                        teacher_ids_keep = torch.cat([teacher_base_ids_keep, terminal_teacher_id], dim=1).contiguous()

                        sparse_mask = build_sparse_proximity_mask(
                            teacher_model=teacher_model,
                            ids_keep=teacher_ids_keep,
                            cls_token_num=args.cls_token_num,
                            full_seq_len=full_seq_len,
                            device=device,
                        )

                        if should_debug(args, rank, train_steps):
                            logger.info(
                                f"[DEBUG step={train_steps}] DINO sparse token-level KD with sparse proximity mask: "
                                f"z_indices={tuple(z_indices.shape)}, "
                                f"student_idx={tuple(student_idx.shape)}, "
                                f"c_indices={tuple(c_indices.shape)}, "
                                f"student_K={student_ids_keep.shape[1]}, "
                                f"teacher_K={teacher_base_ids_keep.shape[1]}, "
                                f"teacher_groups_mean={teacher_group_count.float().mean().item():.2f}, "
                                f"teacher_group_tokens_mean={teacher_group_token_count.float().mean().item():.2f}, "
                                f"anchor_intersection_mean={anchor_tokens_per_sample.item():.2f}, "
                                f"sparse_mask={tuple(sparse_mask.shape)}"
                            )

                            debug_tensor_stats(
                                "sparse_mask",
                                sparse_mask,
                                logger,
                                rank,
                                train_steps,
                            )

                            # 关键：同步一下，方便定位 CUDA 错误。
                            torch.cuda.synchronize()

                        _ = teacher_model(
                            cond_idx=c_indices,
                            idx=z_indices,
                            mask=sparse_mask,
                            input_pos=None,
                            targets=None,
                            ids_keep=teacher_ids_keep,
                        )

                        if should_debug(args, rank, train_steps):
                            for l in teacher_target_layers:
                                debug_tensor_stats(
                                    f"teacher_features layer_{l}",
                                    teacher_features.get(f"layer_{l}", None),
                                    logger,
                                    rank,
                                    train_steps,
                                )

                        teacher_full_features = []
                        t_feats = []
                        for layer_id in teacher_target_layers:
                            teacher_sparse = teacher_features[f"layer_{layer_id}"][:, args.cls_token_num : -1, :]
                            teacher_full = teacher_sparse.new_zeros(
                                teacher_sparse.shape[0],
                                num_kd_image_tokens,
                                teacher_sparse.shape[-1],
                            )
                            scatter_index = teacher_base_ids_keep.unsqueeze(-1).expand(-1, -1, teacher_sparse.shape[-1])
                            teacher_full.scatter_(1, scatter_index, teacher_sparse)
                            teacher_full_features.append(teacher_full)
                            student_index = student_ids_keep.unsqueeze(-1).expand(-1, -1, teacher_sparse.shape[-1])
                            t_feats.append(torch.gather(teacher_full, 1, student_index))

                # ============================================================
                # 3. Distillation Loss
                # ============================================================

                loss = ce_loss
                distill_loss_val = 0.0
                future_feature_loss = ce_loss.new_tensor(0.0)

                if t_feats is not None:
                    check_feature_shapes(
                        s_feats=s_feats,
                        t_feats=t_feats,
                        ids_keep=student_ids_keep,
                        logger=logger,
                        rank=rank,
                        step=debug_step,
                    )

                    distill_loss = 0.0

                    for s_feat, t_feat in zip(s_feats, t_feats):
                        # Normalized MSE / squared L2 on unit token features.
                        # This aligns representation direction and removes hidden-norm scale mismatch
                        # between NAR-L/XL/XXL teachers. We sum over hidden dim so the loss stays
                        # on a cosine-like scale: ||s_hat - t_hat||^2 = 2 - 2*cos(s, t).
                        s_norm = F.normalize(s_feat.float(), dim=-1)
                        t_norm = F.normalize(t_feat.float(), dim=-1)

                        token_distill_loss = (s_norm - t_norm).pow(2).sum(dim=-1)
                        valid_weight = anchor_at_student.to(token_distill_loss.dtype)
                        distill_loss = distill_loss + (
                            (token_distill_loss * valid_weight).sum() / valid_weight.sum().clamp_min(1.0)
                        )

                    distill_loss = distill_loss / len(s_feats)
                    distill_loss_val = distill_loss.item()

                    loss = ce_loss + args.distill_weight * distill_loss

                    teacher_future_targets = teacher_full_features[0]
                    for prediction, offset, weight in zip(
                        future_feature_predictions,
                        future_feature_offsets,
                        future_feature_loss_weights,
                    ):
                        target_positions = student_ids_keep + int(offset)
                        target_index = (
                            target_positions.clamp(0, num_kd_image_tokens - 1)
                            .unsqueeze(-1)
                            .expand(-1, -1, teacher_future_targets.shape[-1])
                        )
                        target = torch.gather(
                            teacher_future_targets,
                            dim=1,
                            index=target_index,
                        )
                        if prediction.shape != target.shape:
                            raise RuntimeError(
                                "Future teacher-feature shape mismatch: "
                                f"prediction={tuple(prediction.shape)}, "
                                f"target={tuple(target.shape)}, offset={offset}"
                            )

                        prediction_norm = F.normalize(prediction.float(), dim=-1)
                        target_norm = F.normalize(target.float(), dim=-1)
                        token_future_loss = (prediction_norm - target_norm).pow(2).sum(dim=-1)
                        valid_weight = anchor_at_student.to(token_future_loss.dtype)
                        future_feature_loss = future_feature_loss + float(weight) * (
                            (token_future_loss * valid_weight).sum() / valid_weight.sum().clamp_min(1.0)
                        )

                    loss = loss + future_feature_loss
                elif future_feature_predictions:
                    raise RuntimeError("Future teacher-feature prediction was enabled without teacher features.")

                loss = loss + args.loss_weight_contrast * contrast_loss

            # ================================================================
            # 4. Backward
            # ================================================================

            scaler.scale(loss).backward()

            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if args.ema:
                update_ema(
                    ema,
                    model.module._orig_mod if not args.no_compile else model.module,
                )

            running_loss += loss.item()
            running_ce_loss += ce_loss.item()
            running_distill_loss += distill_loss_val
            running_future_feature_loss += future_feature_loss.item()
            running_contrast_loss += contrast_loss.item()
            running_student_drop_rate += current_student_drop_rate
            running_teacher_drop_rate += current_teacher_drop_rate
            running_student_tokens += student_ids_keep.shape[1]
            running_teacher_tokens += teacher_base_ids_keep.shape[1]
            running_teacher_groups += teacher_group_count.float().mean().item()
            running_anchor_tokens += anchor_tokens_per_sample.item()

            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()

                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)

                avg_loss = torch.tensor(
                    [
                        running_loss / log_steps,
                        running_ce_loss / log_steps,
                        running_distill_loss / log_steps,
                        running_future_feature_loss / log_steps,
                        running_contrast_loss / log_steps,
                        running_student_drop_rate / log_steps,
                        running_teacher_drop_rate / log_steps,
                        running_student_tokens / log_steps,
                        running_teacher_tokens / log_steps,
                        running_teacher_groups / log_steps,
                        running_anchor_tokens / log_steps,
                    ],
                    device=device,
                )

                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss / dist.get_world_size()

                logger.info(
                    f"(step={train_steps:07d}) "
                    f"Total: {avg_loss[0].item():.4f} "
                    f"[CE: {avg_loss[1].item():.4f}, "
                    f"Distill: {avg_loss[2].item():.4f}, "
                    f"FutureFeature: {avg_loss[3].item():.4f}, "
                    f"TCR: {avg_loss[4].item():.4f}, "
                    f"wTCR: {args.loss_weight_contrast * avg_loss[4].item():.4f}], "
                    f"StudentDrop: {avg_loss[5].item():.4f}, "
                    f"TeacherDrop: {avg_loss[6].item():.4f}, "
                    f"StudentTokens: {avg_loss[7].item():.1f}, "
                    f"TeacherTokens: {avg_loss[8].item():.1f}, "
                    f"TeacherGroups: {avg_loss[9].item():.1f}, "
                    f"AnchorTokens: {avg_loss[10].item():.1f}, "
                    f"Steps/Sec: {steps_per_sec:.2f}"
                )

                running_loss = 0.0
                running_ce_loss = 0.0
                running_distill_loss = 0.0
                running_future_feature_loss = 0.0
                running_contrast_loss = 0.0
                running_student_drop_rate = 0.0
                running_teacher_drop_rate = 0.0
                running_student_tokens = 0.0
                running_teacher_tokens = 0.0
                running_teacher_groups = 0.0
                running_anchor_tokens = 0.0
                log_steps = 0
                start_time = time.time()

            should_save_checkpoint = (
                train_steps in checkpoint_steps
                if checkpoint_steps
                else train_steps % args.ckpt_every == 0 and train_steps > 0
            )
            if should_save_checkpoint:
                if rank == 0:
                    model_weight = (
                        model.module._orig_mod.state_dict() if not args.no_compile else model.module.state_dict()
                    )

                    checkpoint = {
                        "model": model_weight,
                        "optimizer": optimizer.state_dict(),
                        "steps": train_steps,
                        "args": args,
                    }

                    if args.ema:
                        checkpoint["ema"] = ema.state_dict()

                    if not args.no_local_save:
                        torch.save(
                            checkpoint,
                            f"{checkpoint_dir}/{train_steps:07d}.pt",
                        )

                    torch.save(
                        checkpoint,
                        f"{cloud_checkpoint_dir}/{train_steps:07d}.pt",
                    )

                dist.barrier()

    model.eval()
    logger.info("Done!")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument("--code-path", type=str, required=True)
    parser.add_argument("--cloud-save-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--no-local-save", action="store_true")

    # Student
    parser.add_argument(
        "--gpt-model",
        type=str,
        choices=list(GPT_models.keys()),
        default="GPT-B",
    )
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=["c2i", "t2i"], default="c2i")

    # Teacher / Distillation
    parser.add_argument(
        "--teacher-model",
        type=str,
        choices=list(TEACHER_CONFIGS.keys()),
        default="NAR-L",
    )
    parser.add_argument("--teacher-ckpt", type=str, required=True)
    parser.add_argument("--distill-weight", type=float, default=1.0)

    parser.add_argument("--student-target-layers", type=str, default="11")
    parser.add_argument("--teacher-target-layers", type=str, default="22")
    parser.add_argument("--projector-dim", type=int, default=2048)
    parser.add_argument("--future-feature-pred-layer", type=int, default=9)
    parser.add_argument("--future-feature-offsets", type=str, default="1,16,17")
    parser.add_argument(
        "--future-feature-loss-weights",
        type=str,
        default="0.1,0.1,0.1",
    )

    parser.add_argument("--loss-weight-contrast", type=float, default=0.2)
    parser.add_argument("--contrast-temp", type=float, default=0.07)
    parser.add_argument("--contrast-dropout-p", type=float, default=0.2)
    parser.add_argument("--contrast-num-samples", type=int, default=2048)

    # Student SD and grouped Teacher token drop
    parser.add_argument(
        "--max-student-drop-rate",
        type=float,
        default=0.5,
        help="Student physical SD rate during the early training stage.",
    )
    parser.add_argument(
        "--student-drop-switch-ratio",
        type=float,
        default=0.80,
        help="After this epoch ratio, Student SD becomes zero.",
    )
    parser.add_argument(
        "--max-physical-drop-rate",
        type=float,
        default=0.6,
        help="Fixed grouped Teacher physical drop rate.",
    )
    parser.add_argument(
        "--physical-drop-switch-ratio",
        type=float,
        default=1.0,
        help="Reserved for compatibility; grouped Teacher drop stays fixed.",
    )
    parser.add_argument(
        "--semantic-drop-temperature",
        type=float,
        default=0.35,
        help="Student SD sampling temperature; Teacher group selection is always deterministic.",
    )

    # Model/data
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--cls-token-num", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--dataset", type=str, default="imagenet_code")
    parser.add_argument(
        "--image-size",
        type=int,
        choices=[256, 384, 448, 512],
        default=256,
    )
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)

    # Regularization
    parser.add_argument("--dropout-p", type=float, default=0.1)
    parser.add_argument("--token-dropout-p", type=float, default=0.1)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)

    # Training
    parser.add_argument("--epochs", type=int, default=53)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", default=1.0, type=float)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument(
        "--checkpoint-steps",
        type=str,
        default="",
        help="Comma-separated exact optimizer steps to save; overrides ckpt-every when set.",
    )

    # Debug
    parser.add_argument(
        "--debug-first-steps",
        type=int,
        default=0,
        help="Print detailed debug info for the first N steps. 0 disables debug logs.",
    )

    # AMP / compile / EMA
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="bf16",
        choices=["none", "fp16", "bf16"],
    )
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--no-compile", action="store_true")

    args = parser.parse_args()
    main(args)
