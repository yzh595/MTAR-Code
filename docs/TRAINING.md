# Usage Guide

[Back to the project overview](../README.md)

Run all commands below from the repository root. The repository name and clone URL remain unchanged.

## 1. Environment

Use a [LlamaGen](https://github.com/FoundationVision/LlamaGen)-compatible PyTorch environment. The shared requirements target Python 3.10, PyTorch 2.2.1 with CUDA 12.1, and torchvision 0.17.1.

```bash
git clone https://github.com/yzh595/MTAR-Code.git
cd MTAR-Code
pip install -r requirements.txt
```

See the [DINOv3 extraction instructions](../dinov3/README.md) for semantic-score dependencies and model access.

## 2. Data preparation

Extract aligned VQ tokens, class labels, and DINOv3 semantic scores:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/extract_dinov3_scores.sh \
    /path/to/imagenet/train \
    /path/to/imagenet256_features \
    /path/to/vq_ds16_c2i.pt
```

Semantic scores are computed offline once and reused during training. The output directory contains:

```text
imagenet256_features/
├── imagenet256_codes/
├── imagenet256_labels/
└── imagenet256_scores/
```

## 3. Checkpoints

Prepare the following local checkpoints before training or sampling:

| Checkpoint | Used for |
|:--|:--|
| LlamaGen VQ-16 tokenizer | Discrete token extraction and image decoding |
| Pretrained NAR-XL generative teacher | MTAR++ training only; passed as `TEACHER_CKPT` |
| Trained MTAR / MTAR++ student | Sampling and evaluation |

The training launcher does not download the teacher automatically. Student sampling does not require the teacher checkpoint.

## 4. Train MTAR++

The following launcher uses a LlamaGen-L student, a frozen NAR-XL teacher, MFP(C,R,B), TCR, and dual-path semantic dropping. Student dropping is 50% for the first 80% of training; teacher dropping is 50% throughout training.

```bash
METHOD=MTAR++ \
TEACHER_CKPT=/path/to/NAR-XL.pt \
CUDA_DEVICES=0,1 \
bash scripts/train_c2i.sh \
    /path/to/imagenet256_features \
    /path/to/mtarpp_results
```

The default MTAR++ launcher targets the large model. Its student feature layer is set to 24; reproducing a base-model configuration also requires matching the student layer and other scale-specific settings, not just changing `GPT_MODEL`.

<details>
<summary><strong>Train the conference MTAR model</strong></summary>

The default MTAR launcher trains LlamaGen-B for 300 epochs with MTP, TCR, and 50% student semantic dropping during the first 80% of training, followed by full-sequence training.

```bash
METHOD=MTAR CUDA_DEVICES=0 bash scripts/train_c2i.sh \
    /path/to/imagenet256_features \
    /path/to/mtar_results
```

</details>

Shared defaults include batch size 256 and learning rate `1e-4`. The launcher enables student EMA. Additional settings are available in [scripts/train_c2i.sh](../scripts/train_c2i.sh).

## 5. Sampling and evaluation

Sample 50,000 images from an MTAR++ large checkpoint using the unchanged autoregressive inference path:

```bash
GPT_MODEL=GPT-L CUDA_VISIBLE_DEVICES=0 \
bash scripts/sample_c2i.sh \
    /path/to/mtarpp_large_checkpoint.pt \
    /path/to/vq_ds16_c2i.pt \
    1.80 \
    ./fid_samples \
    50000
```

Use `GPT_MODEL=GPT-B` and guidance 1.85 for MTAR++ base. The reported MTAR base and large results use guidance 2.00.

Evaluate the generated NPZ file:

```bash
EVAL_CONDA_ENV=lla bash scripts/evaluate_c2i.sh \
    /path/to/VIRTUAL_imagenet256_labeled.npz \
    /path/to/generated_samples.npz
```

See [evaluations/README.md](../evaluations/README.md) for the ADM TensorFlow evaluation environment.
