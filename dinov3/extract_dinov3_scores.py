# Modified for DINOv3 (Auto-detect Registers)
import sys
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import numpy as np
import argparse
from transformers import AutoModel

# =======================================================
# =======================================================
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from dataset.augmentation import center_crop_arr
    from dataset.build import build_dataset
    from tokenizer.tokenizer_image.vq_model import VQ_models
except ImportError as e:
    print(f"Error importing project modules: {e}")
    sys.exit(1)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# =======================================================
# =======================================================
class DinoInputProcessor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = x * 0.5 + 0.5
        x = (x - self.mean) / self.std
        return x

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
        args.distributed = True
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
        args.distributed = True
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        return

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    dist.init_process_group(
        backend=args.dist_backend, 
        init_method=args.dist_url,
        world_size=args.world_size, 
        rank=args.rank
    )
    dist.barrier()

def main(args):
    init_distributed_mode(args)
    device = torch.device(f"cuda:{args.gpu}") if args.distributed else torch.device("cuda")
    
    seed = args.global_seed * dist.get_world_size() + args.rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if args.rank == 0:
        print(f"Running with {args.world_size} GPUs.")
        os.makedirs(args.code_path, exist_ok=True)
        os.makedirs(os.path.join(args.code_path, f'{args.dataset}{args.image_size}_codes'), exist_ok=True)
        os.makedirs(os.path.join(args.code_path, f'{args.dataset}{args.image_size}_labels'), exist_ok=True)
        os.makedirs(os.path.join(args.code_path, f'{args.dataset}{args.image_size}_scores'), exist_ok=True)
    
    if args.distributed:
        dist.barrier()

    # 1. Load VQ
    if args.rank == 0:
        print(f"Loading VQ Model: {args.vq_model}...")
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim
    )
    vq_model.to(device)
    vq_model.eval()
    try:
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        vq_model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading VQ Checkpoint: {e}")
        exit(1)

    # 2. Load DINOv3
    if args.rank == 0:
        print(f"Loading DINOv3 Model from: {args.dino_model_name}...")
    
    try:
        dino_model = AutoModel.from_pretrained(
            args.dino_model_name, 
            output_attentions=True,
            trust_remote_code=True 
        ).to(device)
    except OSError as e:
        if args.rank == 0:
            print(f"Error: Could not load model {args.dino_model_name}: {e}")
        sys.exit(1)
        
    dino_model.eval()
    dino_processor = DinoInputProcessor().to(device)

    # 3. Build Dataset
    if args.ten_crop:
        crop_size = int(args.image_size * args.crop_range) 
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
            transforms.TenCrop(args.image_size),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    else:
        crop_size = args.image_size 
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        
    dataset = build_dataset(args, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, sampler=sampler, num_workers=args.num_workers, pin_memory=True)

    # 4. Processing Loop
    total = 0 
    if args.image_size % 16 != 0:
        raise ValueError("image-size must be divisible by 16 for VQ-16/DINOv3 alignment")
    target_grid_size = args.image_size // 16
    target_patches = target_grid_size * target_grid_size

    if args.rank == 0:
        print(f"Start processing (Auto-detecting registers)...")

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        
        if args.ten_crop:
            x_all = x.flatten(0, 1)
            num_aug = 10
        else:
            x_flip = torch.flip(x, dims=[-1])
            x_all = torch.cat([x, x_flip])
            num_aug = 2
            
        with torch.no_grad():
            # VQ
            _, _, [_, _, indices] = vq_model.encode(x_all)
            codes = indices.reshape(x.shape[0], num_aug, -1)

            # DINO
            dino_input = dino_processor(x_all)
            outputs = dino_model(dino_input)
            
            # [Fix]: Handle Registers
            # Get attention from CLS token (index 0) to others
            last_layer_attn = outputs.attentions[-1] # [Batch, Heads, SeqLen, SeqLen]
            
            # SeqLen includes CLS, optional registers, and spatial patches.
            seq_len = last_layer_attn.shape[-1]
            
            # Spatial patches are at the end, after CLS/register tokens.
            if seq_len < target_patches:
                 print(f"Error: Sequence length {seq_len} is smaller than target {target_patches}")
                 break
            
            start_index = seq_len - target_patches
            
            # CLS (index 0) attention to all spatial patch tokens.
            cls_attn = last_layer_attn[:, :, 0, start_index:]
            
            patch_scores = cls_attn.mean(dim=1)
            
            # Reshape
            try:
                scores_map = patch_scores.view(num_aug, 1, target_grid_size, target_grid_size)
            except RuntimeError as e:
                print(f"Reshape Error: Expected {target_patches} patches, got {patch_scores.shape[-1]}")
                raise e
            
            scores_out = scores_map.flatten(1).unsqueeze(0).cpu().numpy().astype(np.float16)

        # Save
        train_steps = args.rank + total * args.world_size
        np.save(f'{args.code_path}/{args.dataset}{args.image_size}_scores/{train_steps}.npy', scores_out)
        np.save(f'{args.code_path}/{args.dataset}{args.image_size}_codes/{train_steps}.npy', codes.detach().cpu().numpy().astype(np.int16))
        np.save(f'{args.code_path}/{args.dataset}{args.image_size}_labels/{train_steps}.npy', y.detach().cpu().numpy())

        total += 1
        if total % 1000 == 0:
            print(f"[Rank {args.rank}] Processed {total} images...")

    if args.distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="ImageNet train folder")
    parser.add_argument("--code-path", type=str, required=True, help="Output folder")
    
    parser.add_argument("--dino-model-name", type=str, 
                        default="facebook/dinov3-vitl16-pretrain-lvd1689m")
                        
    parser.add_argument("--vq-model", type=str, default="VQ-16", choices=list(VQ_models.keys()))
    parser.add_argument("--vq-ckpt", type=str, required=True)
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--image-size", type=int, default=256) 
    parser.add_argument("--ten-crop", action='store_true')
    parser.add_argument("--crop-range", type=float, default=1.1)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--dist-url', default='env://')

    args = parser.parse_args()
    main(args)
