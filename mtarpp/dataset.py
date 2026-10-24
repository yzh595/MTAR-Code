import torch
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import glob


def resize_flat_scores_to_match_tokens(scores, target_len):
    """Resize flat 2D-grid semantic scores, e.g. 16x16 -> 24x24 for 384 DS16."""
    if scores is None:
        return None

    scores_arr = np.asarray(scores)
    orig_shape = scores_arr.shape
    flat = scores_arr.reshape(-1, scores_arr.shape[-1])
    source_len = flat.shape[-1]
    target_len = int(target_len)

    if source_len == target_len:
        return scores

    source_hw = int(round(source_len ** 0.5))
    target_hw = int(round(target_len ** 0.5))
    if source_hw * source_hw != source_len or target_hw * target_hw != target_len:
        raise ValueError(
            f"Cannot resize semantic scores from length {source_len} to {target_len}; "
            "both lengths must be square grids."
        )

    tensor = torch.from_numpy(flat.astype(np.float32)).view(-1, 1, source_hw, source_hw)
    tensor = F.interpolate(tensor, size=(target_hw, target_hw), mode="bilinear", align_corners=False)
    resized = tensor.view(*orig_shape[:-1], target_len).numpy().astype(scores_arr.dtype, copy=False)
    return resized

class CustomDataset(Dataset):
    def __init__(self, feature_dir, label_dir):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.score_dir = feature_dir.replace('_codes', '_scores')

        self.flip = 'flip' in self.feature_dir

        aug_feature_dir = feature_dir.replace('ten_crop/', 'ten_crop_105/')
        aug_label_dir = label_dir.replace('ten_crop/', 'ten_crop_105/')
        aug_score_dir = self.score_dir.replace('ten_crop/', 'ten_crop_105/')

        if os.path.exists(aug_feature_dir) and os.path.exists(aug_label_dir):
            self.aug_feature_dir = aug_feature_dir
            self.aug_label_dir = aug_label_dir
            self.aug_score_dir = aug_score_dir if os.path.exists(aug_score_dir) else None
        else:
            self.aug_feature_dir = None
            self.aug_label_dir = None
            self.aug_score_dir = None

        print(f"正在扫描目录中的 .npy 文件: {feature_dir}")
        self.feature_files = sorted(glob.glob(os.path.join(feature_dir, '*.npy')))
        self.label_files = sorted(glob.glob(os.path.join(label_dir, '*.npy')))

        if os.path.exists(self.score_dir):
            self.score_files = sorted(glob.glob(os.path.join(self.score_dir, '*.npy')))
            self.use_scores = True
            print(f"检测到分数目录: {self.score_dir}, 将加载语义分数。")
        else:
            self.score_files = []
            self.use_scores = False
            print(f"未检测到分数目录 {self.score_dir}，将跳过分数加载。")

        if not self.feature_files:
            raise FileNotFoundError(f"错误：在目录 {feature_dir} 中没有找到任何 .npy 文件。")

        # A partial/misaligned extraction must never silently train on dummy data.
        stems = lambda paths: [os.path.basename(path) for path in paths]
        if not self.use_scores or stems(self.feature_files) != stems(self.label_files) or stems(self.feature_files) != stems(self.score_files):
            raise ValueError("MTAR++ requires matching code, label and score .npy filenames.")

        print(f"成功找到 {len(self.feature_files)} 个数据样本。")

        # 性能优化开关
        self.load_semantic_scores = True

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx):
        use_aug_folder = (self.aug_feature_dir is not None) and (torch.rand(1).item() < 0.5)

        if use_aug_folder:
            base_name = os.path.basename(self.feature_files[idx])
            feature_path = os.path.join(self.aug_feature_dir, base_name)
            label_path = os.path.join(self.aug_label_dir, base_name)
            score_path = os.path.join(self.aug_score_dir, base_name) if self.aug_score_dir else None

            if not os.path.exists(feature_path):
                feature_path = self.feature_files[idx]
                label_path = self.label_files[idx]
                score_path = self.score_files[idx] if self.use_scores else None
        else:
            feature_path = self.feature_files[idx]
            label_path = self.label_files[idx]
            score_path = self.score_files[idx] if self.use_scores else None

        try:
            features = np.load(feature_path)
            labels = np.load(label_path)

            # [FPS 关键优化] 动态截断
            scores = None
            if self.use_scores and self.load_semantic_scores and score_path and os.path.exists(score_path):
                scores = np.load(score_path)

        except Exception as e:
            raise RuntimeError(f"Cannot load aligned training sample: {feature_path}") from e

        selected_feat = features
        selected_score = scores

        num_augs = 1
        if features.ndim == 3:
            num_augs = features.shape[1]
            aug_idx = torch.randint(0, num_augs, (1,)).item()
            selected_feat = features[:, aug_idx, :]

            if scores is not None and scores.ndim == 3:
                selected_score = scores[:, aug_idx, :]

        elif features.ndim == 2:
            if features.shape[0] <= 20:
                num_augs = features.shape[0]
                aug_idx = torch.randint(0, num_augs, (1,)).item()
                selected_feat = features[aug_idx]
                selected_feat = selected_feat[None, :]

                if scores is not None and scores.ndim == 2:
                    selected_score = scores[aug_idx][None, :]

        features_t = torch.from_numpy(selected_feat).long()
        labels_t = torch.from_numpy(labels).long()

        if features_t.ndim == 1: features_t = features_t.unsqueeze(0)

        # [FPS 关键优化] 彻底不返回第三个参数，不分配 Zero Tensor
        if selected_score is not None:
            selected_score = resize_flat_scores_to_match_tokens(selected_score, features_t.shape[-1])
            scores_t = torch.from_numpy(selected_score).float()
            if scores_t.ndim == 1: scores_t = scores_t.unsqueeze(0)
            return features_t, labels_t, scores_t
        else:
            # Code B 模式：只返回两个，0 CPU 开销，0 内存拷贝
            return features_t, labels_t

def build_imagenet(args, transform):
    return ImageFolder(args.data_path, transform=transform)

def build_imagenet_code(args):
    feature_dir = f"{args.code_path}/imagenet{args.image_size}_codes"
    label_dir = f"{args.code_path}/imagenet{args.image_size}_labels"

    if not os.path.exists(feature_dir):
         feature_dir = f"{args.code_path}/{args.dataset}{args.image_size}_codes"
         label_dir = f"{args.code_path}/{args.dataset}{args.image_size}_labels"

    assert os.path.exists(feature_dir), f"数据目录不存在: {feature_dir}"
    assert os.path.exists(label_dir), f"标签目录不存在: {label_dir}"

    return CustomDataset(feature_dir, label_dir)

def build_dataset(args, transform=None):
    if args.dataset == 'imagenet':
        return build_imagenet(args, transform)
    elif args.dataset == 'imagenet_code':
        return build_imagenet_code(args)
    else:
        return build_imagenet_code(args)
