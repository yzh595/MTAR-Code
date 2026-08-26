# DINOv3 semantic-score extraction

`extract_dinov3_scores.py` is the ImageNet DINOv3 extraction entry identified
in the original MTAR assets directory. It jointly writes aligned VQ codes,
class labels, and semantic scores.

For each image augmentation, the extractor:

1. encodes the image with the LlamaGen VQ-16 tokenizer;
2. runs DINOv3 with attention outputs enabled;
3. takes last-layer attention from the CLS token to spatial patch tokens;
4. averages the attention over heads;
5. saves the scores as float16 under `imagenet{SIZE}_scores`.

Run from the MTAR repository root:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
bash assets/extract_dinov3_scores.sh \
    /path/to/imagenet/train \
    /path/to/features_dinov3_large_flip_all \
    /path/to/vq_ds16_c2i.pt \
    facebook/dinov3-vitl16-pretrain-lvd1689m
```

The default `IMAGE_SIZE=256` path produces:

```text
features_dinov3_large_flip_all/
├── imagenet256_codes/*.npy
├── imagenet256_labels/*.npy
└── imagenet256_scores/*.npy
```

By default the extractor saves the original image and a horizontal flip. Set
`TEN_CROP=1` to use ten-crop augmentation. Relevant dependencies include
PyTorch, torchvision, NumPy, Pillow, Transformers with DINOv3 support, the
LlamaGen VQ implementation, and a VQ-16 checkpoint.

The public Hugging Face model ID is the default. For an offline server, pass a
local DINOv3 model directory as the fourth argument.
