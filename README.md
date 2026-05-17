# Learning from Semantic Dictionaries (LEASE)

![](LEASE.png)

Codebase for **LEASE — Learning From Semantic Dictionaries**, a generative pre-training method for Vision Transformers based on joint Codebook learning (forthcoming in CVPR 2026)! Based on MAGE codebase, this repository includes all required code to pretrain and evaluate either Sorcen (Echo Contrast based model) and LEASE. Feel free to fork!


## Models provided

| Model | Description | Checkpoint |
| :---: | :---: | :---: |
| **MAGE** | Token reconstruction based unified architecture. Refer to the original repository! | [:link:](https://github.com/LTH14/mage/tree/main) |
| **Sorcen** | Unified architecture which creates its own positive contrastive pairs during training | [:link:](https://huggingface.co/St3p/Sorcen_ViTB) |
| **LEASE** | Joint codebook training for efficient representation learning and image synthesis | [:link:](https://huggingface.co/St3p/Lease_ViTB) |

All backbones are ViT-Base transformer encoders.



## Results

| Model | IN-1K LP | FID (uncond.) | IS |
| :---: | :---: | :---: | :---: |
| Sorcen | 75.1% | 9.61 | 90.96 |
| MAGE | 74.7% | 11.1 | 81.17 |
| MAGE† | 75.0% | 10.88 | 81.59 |
| LEASE | 76.7% | 9.62 | 91.78 |

† MAGE results from reproduced from its original checkpoint.


## Setup

Dependencies in `requirements.txt`. You also need two files at the repository root:

| File | Description |
|---|---|
| `vqgan_jax_strongaug.ckpt` | VQGAN tokenizer weights (from [MAGE](https://github.com/LTH14/mage/tree/main))|
| `km_16k.npy` | 16k-entry semantic codebook (from [DiGIT](https://github.com/DAMO-NLP-SG/DiGIT)) |



## Data

Both methods work with **precomputed image tokens**. Training expects a pre-tokenized `.pt` file under `token_datasets/`. The file must contain:

| Key | Shape | Description |
|:---:|:---:|:---:|
| `tokens_vqgan` | `(N, 256)` | VQ-GAN (or a generative tokenizer) patch token indices |
| `labels` | `(N,)` | Class labels |
| `tokens_dino` | `(N, 256)` | Semantic dictionary tokens (required for LEASE, optional for Sorcen) |

You can download precomputed IN-1k training set [here](https://drive.google.com/drive/folders/1Qrkt2f2K0pFrhjOEUcy52UOQw6XU0PE4?usp=sharing) :)

## Training

### LEASE — full run (ImageNet-1K, 1600 epochs)

```bash
bash launch_scripts/launch_pretrain_lease.sh
```

Key hyperparameters:

| Argument | Value |
|:---:|:---:|
| `--model` | `lease_vit_base_patch16_single` |
| `--method` | `lease` |
| `--epochs` | `1600` |
| `--warmup_epochs` | `40` |
| `--blr` | `1.5e-4` |
| `--weight_decay` | `0.05` |
| `--batch_size` | `64` (per GPU, 64 GPUs → effective 4096) |
| `--mask_ratio_min/max` | `0.5 / 1.0` |
| `--mask_ratio_mu/std` | `0.55 / 0.25` |

### Sorcen — full run (ImageNet-1K, 1600 epochs)

```bash
bash launch_scripts/launch_pretrain_sorcen.sh
```

Key hyperparameters:

| Argument | Value |
|:---:|:---:|
| `--model` | `sorcen_vit_base_patch16_single` |
| `--method` | `sorcen` |
| `--epochs` | `1600` |
| `--warmup_epochs` | `40` |
| `--blr` | `1.5e-4` |
| `--weight_decay` | `0.05` |
| `--batch_size` | `128` (per GPU, 32 GPUs → effective 4096) |
| `--mask_ratio_min/max` | `0.5 / 1.0` |
| `--mask_ratio_mu/std` | `0.55 / 0.25` |



## Evaluation

### Linear probing

```bash
bash launch_scripts/launch_linprobe_lease.sh   
bash launch_scripts/launch_linprobe_sorcen.sh  
```

### Image generation (unconditional)

```bash
bash launch_scripts/launch_gen_uncond_lease.sh   
bash launch_scripts/launch_gen_uncond_sorcen.sh  
```

Both scripts copy the checkpoint, verify it with an md5 checksum, record provenance, and skip generation if output already exists. Generated images are saved to `generation_results/<experiment_name>/`.


## Citation

> *To appear in CVPR'26 proceedings...*
