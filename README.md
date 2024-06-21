# Boosting Image Restoration by Reducing Model Redundancy

<hr />

>**Abstract:** *Image restoration is a crucial task in computational photography, aiming to restore high-quality images from the degraded inputs. While recent advancements primarily focus on customizing complex neural network models, we have identified significant model redundancy within these methods. The model redundancy manifests as parameter harmfulness and parameter uselessness, limiting the image restoration performance. However, the current research lacks a comprehensive exploration of model redundancy in image restoration, and existing techniques in other tasks often sacrifice performance to accelerate computation. In this paper, we propose a novel method to reduce model redundancy while boosting image restoration. Our method consists of two innovative techniques: attention dynamic reallocation (ADR) and parameter orthogonal generation (POG). For one thing, ADR dynamically reallocates appropriate attention based on original attention, thereby alleviating parameter harmfulness. For another, POG learns orthogonal basis embeddings of parameters and prevents degradation to static parameters, thereby alleviating parameter uselessness. Our method achieves state-of-the-art performance across real-world image denoising and low-light image enhancement on publicly available datasets. We will release the code to the public, presently at the anonymized URL: https://anonymous.4open.science/r/RRR-CCED.*
<hr />

## 🚀 News

- 2024.5 The code and pre-trained model for evaluation are available.


## ⏳ Todo lists

- [ ] We will release the training code after our paper is accepted.


## Quick Start

To quickly reproduce the NAFNet+ours results of SIDD:

- Down load SIDD ValidationNoisyBlocksSrgb.mat and ValidationGtBlocksSrgb.mat from their [website](https://abdokamel.github.io/sidd/).

- All neccessary code is in NAFNet-fast-test.py, just run:
```
python NAFNet-fast-test.py
```

