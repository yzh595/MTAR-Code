# Third-party notices

The MTAR/MTAR++ code builds on LlamaGen by FoundationVision, copyright 2024,
under the MIT license. Its license text is retained in
[`licenses/LlamaGen-LICENSE`](licenses/LlamaGen-LICENSE). Model files retain
their original source-credit headers (including references to VQGAN, DiT,
nanoGPT, Llama, gpt-fast and PixArt).

`dataset/augmentation.py` credits the center/random crop implementations from
OpenAI guided-diffusion at commit `8fb3ad9197f16bbc40620447b2742e13458d2831`.
The bundled evaluation code follows the OpenAI-style image-generation evaluator.
The NAR teacher is the pretrained generative model used in the original
experiments; its definition is distributed without pretrained weights.

No ImageNet data or VQ, NAR, DINOv2 or DINOv3 weights are distributed. These
assets retain their respective licenses and access requirements. In particular,
the source-code license does not grant permission to redistribute ImageNet or
third-party pretrained checkpoints. Users must obtain them separately from their
authorized sources. DINOv2 and Mirai experiment code are not part of this release.
