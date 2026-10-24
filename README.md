# Efficient Training with Foresight: Multi-Token Auxiliary Supervision for Autoregressive Image Generation [arXiv](https://arxiv.org/abs/2608.25386) 

Official PyTorch implementation of **MTAR**, with the **MTAR++** full-ImageNet training code.

## MTAR++ code release

MTAR++ replaces discrete future-token auxiliary supervision with **Teacher-Guided
Multi-Feature Prediction (TG-MFP)**, and combines it with TCR and dual-path semantic
dropping. The released Large-model recipe uses a GPT-L student (layer 24), a frozen
NAR-XL teacher (layer 36), and current/future feature targets at offsets **0, 1, 16**.
The original MTAR entry points below are preserved.

**[MTAR++ setup, data, training, sampling and evaluation →](docs/MTARPP.md)**

```bash
# Full ImageNet, 300 epochs, two GPUs; paths are supplied by you.
CUDA_VISIBLE_DEVICES=0,1 bash train_c2i_mtarpp.sh \
    /path/to/imagenet256_features /path/to/NAR-imagenet-256-XL.pt ./results/mtarpp_l
```

The public MTAR++ recipe uses **power-normalized Student SD** with temperature
0.35, as an intentional update to the historical 300-epoch run. Historical
metrics are not presented as measurements of this updated release.

| Framework | Auxiliary supervision | Efficient training | Entry point |
|---|---|---|---|
| MTAR | Discrete future-token prediction (MTP) | Student SD | `train_c2i_mtar.sh` |
| MTAR++ | Teacher features at offsets 0/1/16 (TG-MFP) | Student SD + target-aware Teacher group drop | `train_c2i_mtarpp.sh` |

![Overview of the MTP and token-level contrastive regularization components in MTAR](assets/mtar_overview.png)

MTAR improves autoregressive image generation with three training-time components:

- **Multi-Token Prediction (MTP):** adds future-token supervision beyond next-token prediction.
- **Token-level Contrastive Regularization (TCR):** regularizes token representations with two dropout views.
- **Semantic Dropping (SD):** uses offline DINOv3 scores to retain informative visual tokens during training.

All auxiliary components are removed at inference time, so sampling follows the standard LlamaGen autoregressive path.

## Results on ImageNet 256x256

| Model | Parameters | FID | IS | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| LlamaGen-B | 111M | 5.46 | 193.61 | 0.84 | 0.46 |
| **MTAR-B** | 138M | **4.50** | **213.96** | **0.84** | **0.48** |
| LlamaGen-L | 343M | 3.80 | 248.28 | 0.83 | 0.52 |
| **MTAR-L** | 387M | **2.85** | **271.41** | **0.83** | **0.54** |

At 300 epochs, MTAR-B and MTAR-L provide 1.27x and 1.39x training speedups, respectively.

## Setup

Use a [LlamaGen](https://github.com/FoundationVision/LlamaGen)-compatible PyTorch environment. 

## Data preparation

Extract aligned VQ tokens, labels, and DINOv3 semantic scores:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash dinov3/extract_dinov3_scores.sh \
    /path/to/imagenet/train \
    /path/to/imagenet256_features \
    /path/to/vq_ds16_c2i.pt
```

The output directory contains:

```text
imagenet256_features/
├── imagenet256_codes/
├── imagenet256_labels/
└── imagenet256_scores/
```

## Training

The default launcher trains MTAR-B for 300 epochs with 50% semantic dropping during the first 80% of training, followed by full-token training during the last 20%.

```bash
CUDA_DEVICES=0 bash train_c2i_mtar.sh \
    /path/to/imagenet256_features \
    /path/to/results
```

The launcher enables EMA and `torch.compile` and uses the main paper settings. Other settings can be changed directly in `train_c2i_mtar.sh`.

## Sampling and evaluation

Generate 50,000 samples:

```bash
CUDA_VISIBLE_DEVICES=0 bash sample_c2i_mtar.sh \
    /path/to/mtar_checkpoint.pt \
    /path/to/vq_ds16_c2i.pt \
    2.0 \
    ./fid_samples \
    50000
```

Evaluate the generated NPZ file:

```bash
EVAL_CONDA_ENV=lla bash evaluate_c2i_mtar.sh \
    /path/to/VIRTUAL_imagenet256_labeled.npz \
    /path/to/generated_samples.npz
```

## Citation

```bibtex
@article{niu2026mtar,
  title         = {Efficient Training with Foresight: Multi-Token Auxiliary Supervision for Autoregressive Image Generation},
  author        = {Niu, Guo and Yao, Xiongfei and Wang, Teng and Zhu, Nannan},
  journal       = {arXiv preprint arXiv:2608.25386},
  year          = {2026},
  eprint        = {2608.25386},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2608.25386}
}
```

## Acknowledgements

This repository is built on [LlamaGen](https://github.com/FoundationVision/LlamaGen). We thank the authors for releasing their code and models.
