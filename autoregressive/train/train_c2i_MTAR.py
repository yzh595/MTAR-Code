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
import argparse
import random
import numpy as np

from utils.logger import create_logger
from utils.distributed import init_distributed_mode
from utils.ema import requires_grad
from dataset.build import build_dataset

from autoregressive.models.MTAR_gpt import GPT_models

LEGACY_TCR_KEY_MAP = {
    "contrast_head.net.0.weight": "contrast_head.net.1.weight",
    "contrast_head.net.0.bias": "contrast_head.net.1.bias",
    "contrast_head.net.2.weight": "contrast_head.net.3.weight",
    "contrast_head.net.2.bias": "contrast_head.net.3.bias",
}


def remap_legacy_tcr_state_dict(state_dict):
    """Map Linear-GELU-Linear-Dropout checkpoints to GPU0 TCR layout."""
    remapped = state_dict.copy()
    if hasattr(state_dict, "_metadata"):
        remapped._metadata = state_dict._metadata

    applied = []
    for old_key, new_key in LEGACY_TCR_KEY_MAP.items():
        if old_key in remapped and new_key in remapped:
            raise RuntimeError(
                f"Checkpoint contains both legacy and GPU0 TCR keys: {old_key}, {new_key}"
            )
        if old_key in remapped:
            remapped[new_key] = remapped.pop(old_key)
            applied.append((old_key, new_key))
    return remapped, applied

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = dict(ema_model.named_parameters())
    for name, param in model.named_parameters():
        if name in ema_params:
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

    ema_buffers = dict(ema_model.named_buffers())
    for name, buf in model.named_buffers():
        if name in ema_buffers:
            ema_buffers[name].copy_(buf)

def get_epoch_drop_rate(current_epoch, total_epochs, max_rate=0.5):
    switch_epoch = int(total_epochs * 0.80)
    
    if current_epoch < switch_epoch:
        return max_rate
    else:
        return 0.0


def build_semantic_keep_indices(
    semantic_scores,
    batch_size,
    num_image_tokens,
    drop_rate,
    device,
    temperature,
):
    """Build sorted sparse coordinates outside the compiled model."""
    keep_num = int(round(num_image_tokens * (1.0 - float(drop_rate))))
    keep_num = max(1, min(keep_num, num_image_tokens))

    if semantic_scores is None:
        noise = torch.rand(batch_size, num_image_tokens, device=device)
        ids_keep = torch.argsort(noise, dim=1)[:, :keep_num]
    else:
        scores = semantic_scores.to(device=device, non_blocking=True)
        if scores.dim() > 2:
            scores = scores.reshape(scores.shape[0], -1)
        scores = scores[:, :num_image_tokens].contiguous()
        scores = torch.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)

        if temperature > 1e-6:
            sampling_logits = torch.log(scores.float().clamp_min(1e-9)) / float(temperature)
            probs = torch.softmax(sampling_logits, dim=-1)
            ids_keep = torch.multinomial(
                probs,
                num_samples=keep_num,
                replacement=False,
            )
        else:
            ids_keep = torch.argsort(scores, dim=1, descending=True)[:, :keep_num]

    return torch.sort(ids_keep, dim=1).values.long().contiguous()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def creat_optimizer(model, weight_decay, learning_rate, betas, logger):
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    if dist.get_rank() == 0:
        logger.info(f"Optimizer: LR={learning_rate}, WD={weight_decay}, Trainable Params={len(param_dict)}")
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    return optimizer


def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    init_distributed_mode(args)
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    
    seed = args.global_seed * dist.get_world_size() + rank
    set_seed(seed)
    torch.cuda.set_device(device)

    checkpoint_dir = ""
    cloud_checkpoint_dir = None
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.gpt_model.replace("/", "-") + f"-seed{args.global_seed}"
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        
        if args.cloud_save_path:
            cloud_experiment_dir = f"{args.cloud_save_path}/{experiment_index:03d}-{model_string_name}"
            cloud_checkpoint_dir = f"{cloud_experiment_dir}/checkpoints"
            os.makedirs(cloud_checkpoint_dir, exist_ok=True)
            logger.info(f"Cloud checkpoint dir: {cloud_checkpoint_dir}")
    else:
        logger = create_logger(None)
    
    latent_size = args.image_size // args.downsample_size
    
    init_drop_rate = get_epoch_drop_rate(0, args.epochs, args.max_drop_rate)

    model_kwargs = dict(
        vocab_size=args.vocab_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        resid_dropout_p=args.dropout_p,
        ffn_dropout_p=args.dropout_p,
        drop_path_rate=args.drop_path_rate,
        token_dropout_p=args.token_dropout_p,
        input_token_drop_rate=init_drop_rate,
        loss_weight_contrast=args.loss_weight_contrast,
        contrast_temp=args.contrast_temp,
        contrast_dropout_p=args.contrast_dropout_p,
        contrast_num_samples=args.contrast_num_samples,
        semantic_drop_temperature=args.semantic_drop_temperature
    )
    
    if rank == 0:
        logger.info(f"Creating model: {args.gpt_model}")
        logger.info(f"Init Drop Rate: {init_drop_rate:.4f} (Max: {args.max_drop_rate})")
        logger.info(f"Semantic Drop Temperature: {args.semantic_drop_temperature} (0.0=TopK, >0.0=Sampling)")
        if args.loss_weight_contrast > 0:
            logger.info(f"Contrastive Loss Enabled: W={args.loss_weight_contrast}, T={args.contrast_temp}, Drop={args.contrast_dropout_p}, Samples={args.contrast_num_samples}")

    model = GPT_models[args.gpt_model](**model_kwargs).to(device)
    
    ema = None
    if args.ema:
        ema_kwargs = deepcopy(model_kwargs)
        ema_kwargs['input_token_drop_rate'] = 0.0 
        if rank == 0: logger.info("Initializing EMA model...")  
        ema = GPT_models[args.gpt_model](**ema_kwargs).to(device)
        ema.load_state_dict(model.state_dict(), strict=False)
        requires_grad(ema, False)

    optimizer = creat_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)

    dataset = build_dataset(args)
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=True, seed=args.global_seed)
    loader = DataLoader(dataset, batch_size=int(args.global_batch_size // dist.get_world_size()), shuffle=False, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    train_steps, start_epoch = 0, 0
    if args.gpt_ckpt:
        logger.info(f"Loading checkpoint from: {args.gpt_ckpt}")
        checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
        model_state_dict = checkpoint.get("model", checkpoint)
        model_state_dict, model_tcr_remap = remap_legacy_tcr_state_dict(model_state_dict)
        model_incompatible = model.load_state_dict(model_state_dict, strict=False)
        if model_tcr_remap:
            logger.info(f"Remapped {len(model_tcr_remap)} legacy TCR keys for model.")
        if model_incompatible.missing_keys or model_incompatible.unexpected_keys:
            logger.info(
                f"Checkpoint model incompatibilities: missing={model_incompatible.missing_keys}, "
                f"unexpected={model_incompatible.unexpected_keys}"
            )
        if "optimizer" in checkpoint and "train_steps" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
                train_steps = checkpoint["train_steps"]
                start_epoch = checkpoint.get("epoch", 0)
            except Exception as e:
                logger.info(f"Failed to load optimizer state: {e}. Starting from fresh optimizer.")
        if args.ema and "ema" in checkpoint:
            ema_state_dict, ema_tcr_remap = remap_legacy_tcr_state_dict(checkpoint["ema"])
            ema_incompatible = ema.load_state_dict(ema_state_dict, strict=False)
            if ema_tcr_remap:
                logger.info(f"Remapped {len(ema_tcr_remap)} legacy TCR keys for EMA.")
            if ema_incompatible.missing_keys or ema_incompatible.unexpected_keys:
                logger.info(
                    f"Checkpoint EMA incompatibilities: missing={ema_incompatible.missing_keys}, "
                    f"unexpected={ema_incompatible.unexpected_keys}"
                )
        del checkpoint

        if args.resume_start_epoch >= 0:
            logger.info(
                f"Overriding checkpoint start_epoch {start_epoch} -> {args.resume_start_epoch}"
            )
            start_epoch = args.resume_start_epoch

        if args.reset_mtp_optimizer_state:
            reset_names = []
            for name, parameter in model.named_parameters():
                if name.startswith("head_td_layers.") or name.startswith("output_td."):
                    if parameter in optimizer.state:
                        del optimizer.state[parameter]
                    reset_names.append(name)
            logger.info(
                "Reset Adam state for MTP parameters only: "
                f"count={len(reset_names)}, names={reset_names}"
            )

        if args.disable_mtp_dropout:
            changed = []
            for layer_index, layer in enumerate(model.head_td_layers):
                layer.attention.attn_dropout_p = 0.0
                for module_name, module in layer.named_modules():
                    if isinstance(module, torch.nn.Dropout):
                        module.p = 0.0
                        changed.append(f"head_td_layers.{layer_index}.{module_name}")
            logger.info(
                "Disabled dropout inside MTP blocks only: "
                f"count={len(changed)}, modules={changed}"
            )
    elif args.ema:
        update_ema(ema, model, decay=0)

    if not args.no_compile:
        torch._dynamo.config.optimize_ddp = False
        if args.inductor_fallback_random:
            torch._inductor.config.fallback_random = True
            import torch._inductor.fx_passes.fuse_attention as fuse_attention
            fuse_attention._sfdp_init = lambda: None
            logger.info(
                "Inductor random ops fallback enabled; SDPA pattern fusion disabled "
                "for PyTorch 2.2.1 compatibility."
            )
        logger.info("torch.compile enabled with optimize_ddp=False")
        logger.info("Compiling model...")
        model = torch.compile(model)
    
    model = DDP(model, device_ids=[device], find_unused_parameters=False)
    
    model.train()
    if args.ema: ema.eval()

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == 'fp16'))
    
    log_steps = 0
    running_losses = {'total': 0.0, 't1': 0.0, 'td': 0.0, 'cont': 0.0}
    running_mtp_debug = {'hidden_rms': 0.0, 'logits_rms': 0.0, 'logits_absmax': 0.0}
    last_mtp_grad_norm = float('nan')
    last_total_grad_norm = float('nan')
    last_output_td_weight_norm = float('nan')
    last_output_td_exp_avg_rms = float('nan')
    last_output_td_exp_avg_sq_rms = float('nan')
    start_time = time.time()

    if rank == 0: logger.info(f"Training for {args.epochs} epochs...")
    
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        
        current_drop_rate = get_epoch_drop_rate(epoch, args.epochs, max_rate=args.max_drop_rate)
            
        raw_model = model.module
        if hasattr(raw_model, '_orig_mod'):
            raw_model = raw_model._orig_mod

        if hasattr(loader.dataset, 'load_semantic_scores'):
            should_load = (current_drop_rate > 0.0)
            if loader.dataset.load_semantic_scores != should_load:
                if rank == 0: 
                    status = "ENABLED" if should_load else "DISABLED"
                    logger.info(f"Epoch {epoch}: Switching Semantic Score Loading -> {status}")
                loader.dataset.load_semantic_scores = should_load
        
        if rank == 0: 
            logger.info(f"Beginning epoch {epoch} | Drop Rate: {current_drop_rate:.4f}")
            
        for batch in loader:
            s_cpu = None
            if len(batch) == 3:
                x, y, s_cpu = batch 
            else:
                if rank == 0 and train_steps == 0 and current_drop_rate > 0:
                    logger.warning("Warning: Dataset did not return semantic scores! Token dropping disabled.")
                x, y = batch
                s_cpu = None

            x = x.to(device, non_blocking=True).long()
            y = y.to(device, non_blocking=True).long()
            
            z_indices = (x.squeeze(1) if x.dim() == 3 else x).long().contiguous()
            c_indices = y.reshape(-1).long().contiguous()
            source_indices = z_indices[:, :-1].contiguous()

            s_device = None
            if current_drop_rate > 0.0 and s_cpu is not None:
                s_device = s_cpu.to(device, non_blocking=True)
                if s_device.dim() == 3 and s_device.shape[1] == 1:
                    s_device = s_device.squeeze(1)

            sparse_mode = current_drop_rate > 0.0
            if sparse_mode:
                ids_keep = build_semantic_keep_indices(
                    semantic_scores=s_device,
                    batch_size=source_indices.shape[0],
                    num_image_tokens=source_indices.shape[1],
                    drop_rate=current_drop_rate,
                    device=device,
                    temperature=args.semantic_drop_temperature,
                )
            else:
                ids_keep = torch.arange(
                    source_indices.shape[1],
                    device=device,
                    dtype=torch.long,
                ).unsqueeze(0).repeat(source_indices.shape[0], 1)

            with torch.cuda.amp.autocast(dtype=ptdtype):  
                logits, loss_dict = model(
                    idx=source_indices,
                    cond_idx=c_indices,
                    targets=z_indices,
                    ids_keep=ids_keep,
                    physical_sparse_input=sparse_mode,
                )
                loss = loss_dict['total_loss']
            
            scaler.scale(loss).backward()
            should_probe_grad = ((train_steps + 1) % args.log_every == 0) and rank == 0
            if args.max_grad_norm > 0.0 or should_probe_grad:
                scaler.unscale_(optimizer)

            if should_probe_grad:
                mtp_grad_sq = []
                for name, param in raw_model.named_parameters():
                    if param.grad is None:
                        continue
                    if name.startswith('head_td_layers.') or name.startswith('output_td.'):
                        mtp_grad_sq.append(param.grad.detach().float().square().sum())
                if mtp_grad_sq:
                    last_mtp_grad_norm = torch.stack(mtp_grad_sq).sum().sqrt().item()

            if args.max_grad_norm > 0.0:
                total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if should_probe_grad:
                    last_total_grad_norm = total_grad_norm.detach().float().item()
            scaler.step(optimizer)
            scaler.update()

            if should_probe_grad:
                output_td_weight = raw_model.output_td.weight
                last_output_td_weight_norm = output_td_weight.detach().float().norm().item()
                output_td_state = optimizer.state.get(output_td_weight, {})
                exp_avg = output_td_state.get('exp_avg')
                exp_avg_sq = output_td_state.get('exp_avg_sq')
                if exp_avg is not None:
                    last_output_td_exp_avg_rms = exp_avg.detach().float().square().mean().sqrt().item()
                if exp_avg_sq is not None:
                    last_output_td_exp_avg_sq_rms = exp_avg_sq.detach().float().mean().sqrt().item()
            optimizer.zero_grad(set_to_none=True)
            
            if args.ema:
                update_ema(ema, raw_model)

            if rank == 0:
                running_losses['total'] += loss.item()
                running_losses['t1'] += loss_dict['loss_t1'].item()
                running_losses['td'] += loss_dict['loss_td'].item()
                running_losses['cont'] += loss_dict['loss_contrast'].item()
                running_mtp_debug['hidden_rms'] += loss_dict['mtp_hidden_rms'].item()
                running_mtp_debug['logits_rms'] += loss_dict['mtp_logits_rms'].item()
                running_mtp_debug['logits_absmax'] = max(
                    running_mtp_debug['logits_absmax'],
                    loss_dict['mtp_logits_absmax'].item(),
                )
                log_steps += 1
            
            train_steps += 1
            
            if train_steps > 0 and train_steps % args.log_every == 0 and rank == 0:
                steps_per_sec = log_steps / (time.time() - start_time)
                avg_loss = {k: v / log_steps for k, v in running_losses.items()}
                avg_mtp_debug = {
                    'hidden_rms': running_mtp_debug['hidden_rms'] / log_steps,
                    'logits_rms': running_mtp_debug['logits_rms'] / log_steps,
                    'logits_absmax': running_mtp_debug['logits_absmax'],
                }
                
                logger.info(
                    f"(step={train_steps:07d}) | "
                    f"Total: {avg_loss['total']:.4f} | "
                    f"Main: {avg_loss['t1']:.4f} | "
                    f"Aux: {avg_loss['td']:.4f} | "
                    f"Cont: {avg_loss['cont']:.4f} | "
                    f"Drop: {current_drop_rate:.4f} | "
                    f"FPS: {steps_per_sec:.2f}"
                )
                logger.info(
                    f"[MTPDBG step={train_steps:07d}] "
                    f"HiddenRMS={avg_mtp_debug['hidden_rms']:.6f} | "
                    f"LogitsRMS={avg_mtp_debug['logits_rms']:.6f} | "
                    f"LogitsAbsMax={avg_mtp_debug['logits_absmax']:.6f} | "
                    f"MTPGradPreClip={last_mtp_grad_norm:.6f} | "
                    f"TotalGradPreClip={last_total_grad_norm:.6f} | "
                    f"OutputTDWeightNorm={last_output_td_weight_norm:.6f} | "
                    f"AdamExpAvgRMS={last_output_td_exp_avg_rms:.8f} | "
                    f"AdamExpAvgSqRoot={last_output_td_exp_avg_sq_rms:.8f}"
                )
                
                running_losses = {k: 0.0 for k in running_losses}
                running_mtp_debug = {'hidden_rms': 0.0, 'logits_rms': 0.0, 'logits_absmax': 0.0}
                log_steps = 0
                start_time = time.time()

            is_regular_checkpoint = train_steps > 0 and train_steps % args.ckpt_every == 0
            is_extra_checkpoint = args.extra_ckpt_step > 0 and train_steps == args.extra_ckpt_step
            if (is_regular_checkpoint or is_extra_checkpoint) and rank == 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "ema": ema.state_dict() if ema is not None else None,
                    "args": args,
                    "epoch": epoch,
                    "train_steps": train_steps,
                }
                if not args.no_local_save:
                    torch.save(checkpoint, f"{checkpoint_dir}/{train_steps:07d}.pt")
                if cloud_checkpoint_dir:
                    torch.save(checkpoint, f"{cloud_checkpoint_dir}/{train_steps:07d}.pt")

        next_epoch = epoch + 1
        if args.save_transition_checkpoint and next_epoch < args.epochs:
            next_drop_rate = get_epoch_drop_rate(next_epoch, args.epochs, max_rate=args.max_drop_rate)
            is_transition = current_drop_rate > 0.0 and next_drop_rate == 0.0
            if is_transition and rank == 0:
                transition_checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "ema": ema.state_dict() if ema is not None else None,
                    "args": args,
                    "epoch": next_epoch,
                    "train_steps": train_steps,
                }
                transition_name = f"{train_steps:07d}-pre-nodrop.pt"
                if not args.no_local_save:
                    torch.save(transition_checkpoint, f"{checkpoint_dir}/{transition_name}")
                if cloud_checkpoint_dir:
                    torch.save(transition_checkpoint, f"{cloud_checkpoint_dir}/{transition_name}")
                logger.info(f"Saved pre-NoDrop transition checkpoint: {transition_name}")
            if dist.is_initialized():
                dist.barrier()

    if rank == 0: logger.info("Training finished.")
    if dist.is_initialized(): dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-path", type=str, default=None)
    parser.add_argument("--cloud-save-path", type=str, default=None)
    parser.add_argument("--no-local-save", action='store_true')
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--gpt-model", type=str, default="GPT-B")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--resume-start-epoch", type=int, default=-1)
    parser.add_argument("--reset-mtp-optimizer-state", action='store_true')
    parser.add_argument("--inductor-fallback-random", action='store_true')
    parser.add_argument("--disable-mtp-dropout", action='store_true')
    parser.add_argument("--gpt-type", type=str, default="c2i")
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--ema", action='store_true')
    parser.add_argument("--cls-token-num", type=int, default=1)
    parser.add_argument("--dropout-p", type=float, default=0.1)
    parser.add_argument("--token-dropout-p", type=float, default=0.1)
    parser.add_argument("--drop-path-rate", type=float, default=0.0)
    parser.add_argument("--no-compile", action='store_true')
    parser.add_argument("--dataset", type=str, default='imagenet_code')
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--downsample-size", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--extra-ckpt-step", type=int, default=-1)
    parser.add_argument("--save-transition-checkpoint", action='store_true')
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument('--gpu', default=None, type=int, help=argparse.SUPPRESS) 
    
    parser.add_argument("--max-drop-rate", type=float, default=0.5, help="Maximum physical token drop rate")
    
    parser.add_argument("--loss-weight-contrast", type=float, default=0.2, help="Weight for contrastive loss")
    parser.add_argument("--contrast-temp", type=float, default=0.07, help="SimCSE temperature")
    parser.add_argument("--contrast-dropout-p", type=float, default=0.2, help="SimCSE dropout probability")
    parser.add_argument("--contrast-num-samples", type=int, default=2048, help="Max number of samples for contrastive loss calculation (limit memory usage)")
    parser.add_argument("--semantic-drop-temperature", type=float, default=0.0, 
                        help="Temperature for semantic token dropping (0.0 = Top-K, >0.0 = Sampling)")

    args = parser.parse_args()
    main(args)
