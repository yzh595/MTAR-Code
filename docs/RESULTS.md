# Additional Results

[Back to the project overview](../README.md)

Detailed training-budget comparisons, student data-scale results, and ablations from the MTAR++ manuscript.

## ImageNet 256 × 256

All entries below use 300 training epochs. Parameter counts include student-side training auxiliaries for MTAR and MTAR++, excluding the frozen teacher and tokenizer. Auxiliary heads are not required at inference.

| Model | Training parameters | FID ↓ | IS ↑ |
|:--|--:|--:|--:|
| LlamaGen-B | 111M | 5.46 | 193.61 |
| MTAR(B) | 138M | 4.50 | 213.96 |
| MTAR++(B) | 124M | 4.22 | 230.27 |
| LlamaGen-L | 343M | 3.80 | 248.28 |
| MTAR(L) | 387M | 2.85 | 271.41 |
| MTAR++(L) | 358M | 3.09 | 267.18 |

## Training quality and efficiency

MTAR(B) and MTAR(L) provide **1.27× and 1.39× training-time speedups** at the same 300-epoch budget. After only 50 epochs, MTAR++(B) and MTAR++(L) reach FIDs of **5.35 and 3.53**, beating their corresponding 300-epoch LlamaGen baselines with one-sixth as many iterations.

<p align="center">
  <img src="../assets/mtarpp_training_budgets.png" width="960" alt="FID versus epochs and total training-time speedup for MTAR and MTAR++ base and large models, compared with 300-epoch LlamaGen baselines.">
  <br><sub>Speedup = total training time of the matching 300-epoch LlamaGen baseline / total training time of the evaluated configuration.</sub>
</p>

## Learning with limited student data

MTAR++(B) achieves lower FID than MTAR(B) and LlamaGen-B across the evaluated student data scales. With **300k student images**, it reaches **FID 6.53**, compared with **7.29** for LlamaGen-B using all **1.281M images**.

<p align="center">
  <img src="../assets/mtarpp_data_efficiency.png" width="960" alt="ImageNet FID and absolute FID reductions across 30k, 50k, 100k, 300k, and 1.281M student training images.">
  <br><sub>All models train for 50 epochs at each data scale. Evaluation uses 50,000 generated images and the same ImageNet validation reference. Generative teachers are existing checkpoints pretrained on the full ImageNet training set.</sub>
</p>


### Dual-path dropping

MFP(C,R,B), LlamaGen-B student, NAR-XL teacher; TCR disabled. Student SD drops 50% of input patches during the first 80% of training and uses full sequences thereafter. Speedups are relative to MFP without dropping in either pathway. These are 50-epoch ImageNet-100k ablations, evaluated with 10k generated images.

| Student SD | Teacher drop | FID ↓ | Training-time speedup ↑ |
|:--:|--:|--:|--:|
| No | 0% | 15.40 | 1.00× |
| Yes | 0% | 15.69 | 1.09× |
| Yes | 40% | 15.60 | 1.61× |
| Yes | 50% | 15.38 | 1.82× |
| Yes | 60% | 16.14 | 2.15× |
| Yes | 70% | 17.16 | 2.53× |

### Transfer under limited target-domain data

The LlamaGen-B baseline and MFP(C,R,B) student are trained from scratch for 1,500 optimization steps with batch size 256. Evaluation uses 200 generated images per class against each dataset's held-out validation set.

| Dataset | Method | FID ↓ | Precision ↑ | Recall ↑ |
|:--|:--|--:|--:|--:|
| Oxford16 | LlamaGen-B | 70.62 | 0.44 | 0.39 |
| Oxford16 | MFP(C,R,B) | 51.21 | 0.68 | 0.37 |
| Five-class | LlamaGen-B | 96.81 | 0.40 | 0.71 |
| Five-class | MFP(C,R,B) | 79.21 | 0.56 | 0.68 |

Oxford16 excludes Daisy from Oxford 17 Flowers and uses 960 training / 320 validation images. The five-class dataset uses Obama, Grumpy Cat, Panda, Cat, and Dog, with 400 training / 100 validation images. Both use within-class splits with seed 42 and 256 × 256 resolution.

### Qualitative comparison

<p align="center">
  <img src="../assets/mtarpp_qualitative.png" width="960" alt="Qualitative comparison of LlamaGen-L, MTAR large, and MTAR++ large on birds and animals; red boxes highlight head structures and facial details.">
  <br><sub>Left: LlamaGen-L. Middle: MTAR(L). Right: MTAR++(L).</sub>
</p>
