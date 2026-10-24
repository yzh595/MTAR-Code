# MTAR++ on ImageNet 256×256

This release packages the full-ImageNet **GPT-L, 300-epoch** training implementation.
It is separate from MTAR's MTP implementation and from the Mirai-I / DINOv2 experiments.
No dataset, pretrained teacher, tokenizer checkpoint, training checkpoint or raw
server log is included.

## Method and release scope

- **Student:** GPT-L, hidden width 1024, 24 layers; auxiliary heads read layer 24.
- **Teacher:** separately pretrained and frozen NAR-XL, width 1280, 36 layers.
  Teacher features are taken from transformer block 36, before the final model norm.
- **TG-MFP:** three MLPs predict teacher features at offsets 0, 1 and 16 from
  the current Student representation. Each head has hidden width 2048. The
  feature losses are squared L2 distances between unit-normalized vectors,
  summed over channels and averaged over valid anchors, each with weight 1.0.
- **TCR:** weight 0.2, temperature 0.07, dropout views 0.2, at most 2048 token
  features per local batch. It operates on the Student's final normalized features.
- **Student SD:** first 240 epochs drop 50%; final 60 epochs retain all tokens.
  Selection uses `p_i ∝ max(score_i, 1e-9)^(1/0.35)`, sampled **without
  replacement**, with retained indices sorted back into raster order.
- **Teacher group drop:** fixed 50% for all 300 epochs. Rank candidate groups
  `{i, i+1, i+16}` by mean semantic score, retain a score-ranked group prefix
  whose union fits the budget, then fill unused budget with high-score context.
  Only retained Student positions that are selected Teacher group anchors
  receive feature supervision. Context-fill tokens do not become anchors.
- **Inference:** standard NTP-only LlamaGen decoding; no teacher, TG-MFP head,
  TCR head or semantic scores are needed.

### Important differences from historical runs

The original 300-epoch run used `softmax(score / 0.35)` for Student SD. **This
public release deliberately uses power normalization**, per the author's release
choice. All remaining launch hyperparameters follow that full-data recipe. Do
not attribute archived generation metrics to this updated recipe without rerunning it.

This implementation uses the original 255 source positions `z_0 … z_254` for
Student feature anchors: Student SD50 keeps 128 image tokens plus the class
token. Teacher group selection has a 128-token budget on the same source range;
the terminal image token `z_255` is additionally retained as context, along with
the class token. Thus a sparse Teacher forward has 130 tokens, not 128 total.
Original spatial positions and the NAR proximity mask are gathered at retained
indices. CE still predicts the **original** next token (`z_i → z_(i+1)`), not the
next retained token.

## 1. Environment

Linux and CUDA GPUs are required for the training/sampling entry points. The
checked runtime is Python 3.10, PyTorch 2.2.1+cu121, torchvision 0.17.1+cu121,
NumPy 1.26.4 and Pillow 10.4.0. The historical recipe used two A100 80GB GPUs.

```bash
conda create -n mtarpp python=3.10 -y
conda activate mtarpp
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-mtarpp.txt
```

Keep TensorFlow evaluation dependencies in a separate environment:

```bash
conda create -n mtarpp-eval python=3.10 -y
conda activate mtarpp-eval
pip install -r requirements-eval.txt
conda activate mtarpp
```

For offline DINOv3 extraction, additionally install a Transformers release that
supports DINOv3 and obtain access to the official DINOv3 checkpoint. DINOv3 is
used **only for offline semantic scores**, not as the online generative teacher.
Follow its upstream access/license requirements. The training requirements do
not download any model or require a Hugging Face token.

## 2. Prepare data and checkpoints

You need:

1. A legally obtained ImageNet training split (1,281,167 images for the original
   full-data run), in class-subdirectory form, if extracting features yourself.
2. LlamaGen's VQ-16 tokenizer checkpoint (`vq_ds16_c2i.pt`, 16,384 codebook entries).
3. A **NAR-XL ImageNet-256** pretrained checkpoint matching `mtarpp/teacher.py`.
   Supply this externally; DINOv2, a causal GPT-XL or a DINOv3 vision encoder
   is not an interchangeable teacher checkpoint. The loader requires all keys
   and tensor shapes to match. This release does not host teacher weights.
   Load only trusted `.pt` files: this historical PyTorch runtime uses pickle-based loading.
4. Aligned VQ codes, labels and offline semantic scores:

```text
imagenet256_features/
├── imagenet256_codes/    # <sample_id>.npy, [1, A, 256]
├── imagenet256_labels/   # <sample_id>.npy, one integer in [0, 999]
└── imagenet256_scores/   # <sample_id>.npy, [1, A, 256]
```

`A` is the number of stored views (for example 2 for original+horizontal flip,
or 10 for TenCrop). The loader samples the same view for codes and scores.
Every image must have exactly matching filenames across all three directories.
Keep the original feature-extraction/view ordering when using existing features.
Raw RGB images are not loaded during MTAR++ training.

The repository's extraction entry point can produce aligned features:

```bash
CUDA_VISIBLE_DEVICES=0,1 TEN_CROP=1 CROP_RANGE=1.1 \
    bash dinov3/extract_dinov3_scores.sh \
    /path/to/imagenet/train /path/to/imagenet256_features \
    /path/to/vq_ds16_c2i.pt facebook/dinov3-vitl16-pretrain-lvd1689m
```

Choose the view/crop settings appropriate for your features; changing extraction
settings is an experimental change, not a byte-identical reproduction. ImageNet,
pretrained VQ/NAR/DINO weights and their licenses remain the user's responsibility.

## 3. Train the Large model for 300 epochs

```bash
CUDA_VISIBLE_DEVICES=0,1 bash train_c2i_mtarpp.sh \
    /path/to/imagenet256_features \
    /path/to/NAR-imagenet-256-XL.pt \
    ./results/mtarpp_l_300ep
```

The launcher is the executable recipe; the readable reference is
[`configs/mtarpp/imagenet256_gptl_300ep.json`](../configs/mtarpp/imagenet256_gptl_300ep.json).
The JSON is documentation, not a second configuration loader.

| Setting | Value |
|---|---|
| Global batch / GPUs | 256 / 2 (128 samples per rank) |
| Learning rate | constant 1e-4; no LR scheduler |
| Optimizer | AdamW, betas (0.9, 0.95), weight decay 0.05 |
| Gradient clipping | 1.0 |
| Seed | 0; per-rank seed follows the original LlamaGen convention |
| Precision / compile / EMA | bf16 / enabled / enabled |
| DataLoader | 24 workers per rank, ordinary multiprocessing defaults |
| Dropouts | token, residual and FFN 0.1; DropPath 0 |
| Checkpoint steps | 1.20M, 1.25M, 1.30M, 1.35M, 1.40M, 1.45M, 1.50M |

`torch.compile` is enabled; only the PyTorch DDP graph-partitioning optimization
is disabled (`optimize_ddp=False`), matching the tested multi-output model recipe.
The teacher runs frozen and is not compiled. EMA updates and optimizer settings
are unchanged from the original implementation.

Checkpoint retention follows the historical schedule. The original full split
produces 5,004 steps per epoch, or **1,501,200 steps** for 300 epochs. The last
scheduled checkpoint, `1500000.pt`, is therefore near the end, not the very last
optimizer step. No extra unscheduled final checkpoint is claimed or created.

Artifacts are written under your output directory:

```text
results/mtarpp_l_300ep/
├── training.log
├── logs/000-GPT-L/log.txt
└── checkpoints/<timestamp>/000-GPT-L/checkpoints/*.pt
```

The launcher saves only one checkpoint copy (not duplicate local/cloud copies).
Allow roughly 5.7 GB per full training checkpoint for this Large model; the
exact size depends on serialization/runtime. Seven retained checkpoints require
roughly 40 GB, in addition to data, compile caches and sampling artifacts.

You can inspect the launch without using a GPU:

```bash
DRY_RUN=1 bash train_c2i_mtarpp.sh FEATURE_DIR TEACHER_CKPT OUTPUT_DIR
```

Additional trainer flags may be appended, but any override changes the published
recipe. Resume support is inherited from the historical trainer and restarts at
an epoch boundary; it does not guarantee exact mid-epoch/RNG-state continuation.

## 4. Sample and evaluate

Use the **model** state, not EMA, to match the historical full-data sampling
protocol. The sampler strictly loads every standard NTP parameter, ignores only
the known training-only auxiliary heads, and does not instantiate the teacher.

```bash
CUDA_VISIBLE_DEVICES=1 bash sample_c2i_mtarpp.sh \
    /path/to/1500000.pt /path/to/vq_ds16_c2i.pt \
    1.80 ./fid_samples/mtarpp_l_cfg180 50000
```

Defaults: per-GPU batch 32, seed 0, bf16, compile, temperature 1.0, top-k 0,
top-p 1.0, CFG interval -1. Labels are sampled with the original uniform-random
LlamaGen protocol. This is **not** the later balanced-per-class 10k protocol.
The sampler may generate a few padding images to fill batches and packs exactly
the requested number into the NPZ. Set `SAMPLE_BATCH_SIZE` to change the batch,
but changing batch size/GPU count can change the generated sequence.

Use the original full ImageNet-256 evaluation reference, not a 10k-image reference:

```bash
EVAL_CONDA_ENV=mtarpp-eval bash evaluate_c2i_mtar.sh \
    /path/to/VIRTUAL_imagenet256_labeled.npz \
    /path/to/generated_samples.npz
```

The bundled OpenAI-style evaluator reports FID, sFID, IS, Precision and Recall.
It downloads/caches its Inception graph on first use. Sampling/evaluation scripts
**do not delete** any generated files automatically in this public release.

## 5. Checks, source provenance and limitations

```bash
python -m unittest discover -s tests -p 'test_mtarpp*.py' -v
python -m mtarpp.train --help
python -m mtarpp.sample --help
```

Checks use small synthetic inputs, not ImageNet or pretrained checkpoints. They
cover power sampling, drop scheduling, Teacher group coverage, sparse spatial
mask alignment, NTP-only checkpoint loading and dataset view alignment.
They do not establish full 300-epoch convergence or the updated recipe's FID.

The Student and Teacher network definitions are copied from the archived run;
their parameter names are retained for historical-checkpoint compatibility.
Trainer changes are import/module naming, required/strict teacher loading and
the explicitly requested power-normalized SD update. Dataset errors now fail
instead of substituting zero-valued samples. The sampler now rejects incomplete
backbones instead of silently ignoring missing parameters. See
[`SOURCE_PROVENANCE.json`](../mtarpp/SOURCE_PROVENANCE.json) for archive hashes.

## Acknowledgements

This code builds on [LlamaGen](https://github.com/FoundationVision/LlamaGen),
the NAR teacher implementation used in the experiments, and the model sources
credited in the retained file headers. Semantic scores use the official DINOv3
model, and evaluation follows the bundled OpenAI-style evaluator. Upstream
licenses and pretrained-weight/data terms continue to apply. No additional
MTAR++ paper citation, model download link or unmeasured result is fabricated.
