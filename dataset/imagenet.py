# import torch
# import numpy as np
# import os
# from torch.utils.data import Dataset
# from torchvision.datasets import ImageFolder
# import glob 

# class CustomDataset(Dataset):
#     def __init__(self, feature_dir, label_dir):
#         """
#         初始化数据集。
#         适配离线 SigLIP 分数加载。
#         """
#         self.feature_dir = feature_dir
#         self.label_dir = label_dir
        
#         # [逻辑] 自动推断分数目录 (xxx_codes -> xxx_scores)
#         self.score_dir = feature_dir.replace('_codes', '_scores')
        
#         # 检查是否包含 TenCrop 增强
#         self.flip = 'flip' in self.feature_dir # 这是一个简单的标记检查，不一定完全准确，但够用

#         # --- 处理 TenCrop 等增强目录逻辑 ---
#         # 假设增强后的目录名会有变化，例如 ten_crop -> ten_crop_105
#         # 注意：这里路径替换比较硬编码，请确保你的目录结构确实如此
#         aug_feature_dir = feature_dir.replace('ten_crop/', 'ten_crop_105/')
#         aug_label_dir = label_dir.replace('ten_crop/', 'ten_crop_105/')
#         aug_score_dir = self.score_dir.replace('ten_crop/', 'ten_crop_105/')

#         # 检查增强目录是否存在
#         if os.path.exists(aug_feature_dir) and os.path.exists(aug_label_dir):
#             self.aug_feature_dir = aug_feature_dir
#             self.aug_label_dir = aug_label_dir
#             self.aug_score_dir = aug_score_dir if os.path.exists(aug_score_dir) else None
#         else:
#             self.aug_feature_dir = None
#             self.aug_label_dir = None
#             self.aug_score_dir = None

#         # --- 扫描文件 ---
#         print(f"正在扫描目录中的 .npy 文件: {feature_dir}")
#         self.feature_files = sorted(glob.glob(os.path.join(feature_dir, '*.npy')))
#         self.label_files = sorted(glob.glob(os.path.join(label_dir, '*.npy')))
        
#         # [逻辑] 扫描分数文件
#         if os.path.exists(self.score_dir):
#             self.score_files = sorted(glob.glob(os.path.join(self.score_dir, '*.npy')))
#             self.use_scores = True
#             print(f"检测到分数目录: {self.score_dir}, 将加载语义分数。")
#         else:
#             self.score_files = []
#             self.use_scores = False
#             print(f"未检测到分数目录 {self.score_dir}，将跳过分数加载。")
        
#         if not self.feature_files:
#             raise FileNotFoundError(f"错误：在目录 {feature_dir} 中没有找到任何 .npy 文件。")
            
#         print(f"成功找到 {len(self.feature_files)} 个数据样本。")

#     def __len__(self):
#         # 严格校验数量一致性
#         assert len(self.feature_files) == len(self.label_files), \
#             f"特征数 ({len(self.feature_files)}) 与 标签数 ({len(self.label_files)}) 不匹配!"
        
#         if self.use_scores:
#             assert len(self.feature_files) == len(self.score_files), \
#                 f"特征数 ({len(self.feature_files)}) 与 分数数 ({len(self.score_files)}) 不匹配!"
                
#         return len(self.feature_files)

#     def __getitem__(self, idx):
#         """
#         获取单个数据样本。
#         返回: (features, labels, scores)
#         """
#         # --- 1. 确定文件路径 ---
#         # 50% 概率使用增强目录的数据（如果存在）
#         use_aug_folder = (self.aug_feature_dir is not None) and (torch.rand(1).item() < 0.5)

#         if use_aug_folder:
#             base_name = os.path.basename(self.feature_files[idx])
#             feature_path = os.path.join(self.aug_feature_dir, base_name)
#             label_path = os.path.join(self.aug_label_dir, base_name) 
#             score_path = os.path.join(self.aug_score_dir, base_name) if self.aug_score_dir else None
            
#             # 安全检查：防止增强目录缺文件
#             if not os.path.exists(feature_path):
#                 # 回退到主目录
#                 feature_path = self.feature_files[idx]
#                 label_path = self.label_files[idx]
#                 score_path = self.score_files[idx] if self.use_scores else None
#         else:
#             feature_path = self.feature_files[idx]
#             label_path = self.label_files[idx]
#             score_path = self.score_files[idx] if self.use_scores else None
        
#         # --- 2. 加载数据 ---
#         try:
#             features = np.load(feature_path) # Expected: (1, num_augs, L)
#             labels = np.load(label_path)
#             scores = np.load(score_path) if (score_path and os.path.exists(score_path)) else None
#         except Exception as e:
#             print(f"Error loading {feature_path}: {e}")
#             # 如果出错，返回全 0（或者在这里 raise error）
#             return torch.zeros(1, 256).long(), torch.tensor([0]).long(), torch.zeros(1, 256).float()

#         # --- 3. 同步随机增强选择 (Critical Step) ---
#         # 这里的核心假设：数据格式为 [Batch=1, Num_Augs, Seq_Len]
#         # 如果格式不同 (例如没有 batch 维度)，需要调整 dim 索引
        
#         selected_feat = features
#         selected_score = scores

#         # 检查是否包含 Augmentation 维度
#         # Case 1: [1, Num_Aug, Seq] -> ndim=3
#         # Case 2: [Num_Aug, Seq] -> ndim=2 (假设 Num_Aug < Seq_Len，避免混淆)
        
#         num_augs = 1
#         if features.ndim == 3:
#             num_augs = features.shape[1]
#             # 随机选一个增强
#             aug_idx = torch.randint(0, num_augs, (1,)).item()
#             selected_feat = features[:, aug_idx, :] # [1, Seq]
            
#             if scores is not None and scores.ndim == 3:
#                 selected_score = scores[:, aug_idx, :] # [1, Seq]
                
#         elif features.ndim == 2:
#             # 这里的判断有点风险，假设第一维如果是小的数(<=10)就是 aug，如果是大的(256)就是 seq
#             if features.shape[0] <= 20: 
#                 num_augs = features.shape[0]
#                 aug_idx = torch.randint(0, num_augs, (1,)).item()
#                 selected_feat = features[aug_idx] # [Seq]
#                 # 统一加回 batch 维度 -> [1, Seq]
#                 selected_feat = selected_feat[None, :] 
                
#                 if scores is not None and scores.ndim == 2:
#                     selected_score = scores[aug_idx][None, :]

#         # --- 4. 转换为 Tensor ---
#         features_t = torch.from_numpy(selected_feat).long() # Int16 -> Long
#         labels_t = torch.from_numpy(labels).long()
        
#         if selected_score is not None:
#             scores_t = torch.from_numpy(selected_score).float() # Float16 -> Float32
#         else:
#             # [修正] 如果没有分数，返回与 feature 形状相同的全零 tensor
#             # 这样 DataLoader stack 的时候才不会报错
#             scores_t = torch.zeros_like(features_t).float()

#         # 确保返回维度是 [1, Seq] 而不是 [Seq]
#         # 如果上面处理完已经是 [1, Seq] 则保持，否则 unsqueeze
#         if features_t.ndim == 1: features_t = features_t.unsqueeze(0)
#         if scores_t.ndim == 1: scores_t = scores_t.unsqueeze(0)

#         return features_t, labels_t, scores_t

# def build_imagenet(args, transform):
#     return ImageFolder(args.data_path, transform=transform)

# def build_imagenet_code(args):
#     feature_dir = f"{args.code_path}/imagenet{args.image_size}_codes"
#     label_dir = f"{args.code_path}/imagenet{args.image_size}_labels"
    
#     # 兼容性检查：有些脚本可能叫 imagenet256_codes，有些叫 train256_codes
#     if not os.path.exists(feature_dir):
#          # 尝试用 dataset 名字拼接
#          feature_dir = f"{args.code_path}/{args.dataset}{args.image_size}_codes"
#          label_dir = f"{args.code_path}/{args.dataset}{args.image_size}_labels"

#     # 最终断言
#     assert os.path.exists(feature_dir), f"数据目录不存在: {feature_dir}"
#     assert os.path.exists(label_dir), f"标签目录不存在: {label_dir}"
        
#     return CustomDataset(feature_dir, label_dir)

# def build_dataset(args, transform=None):
#     if args.dataset == 'imagenet':
#         return build_imagenet(args, transform)
#     elif args.dataset == 'imagenet_code':
#         return build_imagenet_code(args)
#     else:
#         return build_imagenet_code(args)


# import torch
# import numpy as np
# import os
# from torch.utils.data import Dataset
# from torchvision.datasets import ImageFolder
# import glob # 确保导入 glob 模块

# class CustomDataset(Dataset):
#     def __init__(self, feature_dir, label_dir):
#         """
#         初始化数据集。
#         这个版本结合了官方代码的优点和动态文件扫描的灵活性。
#         """
#         # --- [保留的优点 1] ---
#         # 来自官方代码：通过检查目录名来智能设置 flip 属性，解决 AttributeError
#         self.feature_dir = feature_dir
#         self.label_dir = label_dir
#         self.flip = 'flip' in self.feature_dir

#         # --- [保留的不变逻辑] ---
#         # 处理 ten_crop 增强目录的逻辑保持不变
#         aug_feature_dir = feature_dir.replace('ten_crop/', 'ten_crop_105/')
#         aug_label_dir = label_dir.replace('ten_crop/', 'ten_crop_105/')
#         if os.path.exists(aug_feature_dir) and os.path.exists(aug_label_dir):
#             self.aug_feature_dir = aug_feature_dir
#             self.aug_label_dir = aug_label_dir
#         else:
#             self.aug_feature_dir = None
#             self.aug_label_dir = None

#         # --- [核心修改] ---
#         # 替换硬编码的文件列表，使用 glob 动态扫描您目录下的真实文件
#         # 这解决了 FileNotFoundError 的问题
#         print(f"正在扫描目录中的 .npy 文件: {feature_dir}")
#         self.feature_files = sorted(glob.glob(os.path.join(feature_dir, '*.npy')))
#         self.label_files = sorted(glob.glob(os.path.join(label_dir, '*.npy')))
        
#         # 增加一个检查，确保找到了文件
#         if not self.feature_files:
#             raise FileNotFoundError(f"错误：在目录 {feature_dir} 中没有找到任何 .npy 文件。请检查您的数据路径。")
            
#         print(f"成功找到 {len(self.feature_files)} 个数据样本。")

#     def __len__(self):
#         """
#         返回数据集的真实大小。
#         """
#         assert len(self.feature_files) == len(self.label_files), \
#             f"特征文件和标签文件的数量不匹配! 特征文件数: {len(self.feature_files)}, 标签文件数: {len(self.label_files)}"
#         return len(self.feature_files)

#     def __getitem__(self, idx):
#         """
#         获取单个数据样本。
#         修改了文件路径处理，以适应 glob 返回的完整路径。
#         """
#         # --- 路径处理 ---
#         # 决定是从主目录还是增强目录加载
#         if self.aug_feature_dir is not None and torch.rand(1) < 0.5:
#             # 如果使用增强目录，需要构造正确的路径
#             # os.path.basename() 获取文件名 (例如 '123.npy')
#             # 然后与增强目录的路径拼接
#             base_feature_file = os.path.basename(self.feature_files[idx])
#             base_label_file = os.path.basename(self.label_files[idx])
#             feature_path = os.path.join(self.aug_feature_dir, base_feature_file)
#             label_path = os.path.join(self.aug_label_dir, base_label_file)
#         else:
#             # 如果不使用增强目录，直接使用 glob 扫描到的完整路径
#             feature_path = self.feature_files[idx]
#             label_path = self.label_files[idx]
        
#         # --- 数据加载与处理 ---
#         features = np.load(feature_path)
        
#         # 从加载的 .npy 文件中随机选择一个数据增强版本
#         # 假设 features 的形状是 (1, num_augs, seq_len) 或 (num_augs, seq_len)
#         # 这个逻辑比官方代码的 `if self.flip:` 更健壮
#         if features.ndim > 1 and features.shape[1] > 1:
#             aug_idx = torch.randint(low=0, high=features.shape[1], size=(1,)).item()
#             # 保持维度一致性
#             if features.ndim == 3: # (1, num_augs, seq_len)
#                 features = features[:, aug_idx, :]
#             else: # (num_augs, seq_len)
#                 features = features[aug_idx]

#         labels = np.load(label_path)
        
#         return torch.from_numpy(features), torch.from_numpy(labels)


# def build_imagenet(args, transform):
#     return ImageFolder(args.data_path, transform=transform)

# def build_imagenet_code(args):
#     # 这部分函数保持不变，因为它正确地构造了目录路径
#     feature_dir = f"{args.code_path}/imagenet{args.image_size}_codes"
#     label_dir = f"{args.code_path}/imagenet{args.image_size}_labels"
#     assert os.path.exists(feature_dir) and os.path.exists(label_dir), \
#         f"找不到词元目录。请确认 '{feature_dir}' 和 '{label_dir}' 存在。\n" \
#         f"请先运行: bash scripts/autoregressive/extract_codes_c2i.sh ..."
#     return CustomDataset(feature_dir, label_dir)



################fps最佳
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
            print(f"Error loading {feature_path}: {e}")
            # Fallback 也要匹配返回长度
            if self.load_semantic_scores:
                return torch.zeros(1, 256).long(), torch.tensor([0]).long(), torch.zeros(1, 256).float()
            else:
                return torch.zeros(1, 256).long(), torch.tensor([0]).long()

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
