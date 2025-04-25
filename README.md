# Mask-Aware Texture Rectification: Semantic Control for Non-Stationary Texture Synthesis

> This repository extends the CVPR 2024 paper:  
> **Generating Non-Stationary Textures using Self-Rectification**  
> Yujun Zhou, Hongzhi Zhang, Jiangtao Fu, Baoquan Chen  
> [Original Paper](https://arxiv.org/abs/2401.02847)

## ✨ Overview

We introduce **Mask-Aware Texture Rectification**, an extension of the self-rectification framework for non-stationary texture synthesis.  
Our method adds semantic controllability using spatial binary masks, enabling region-specific texture transfer during both **inversion** and **sampling** phases of DDIM.

This approach preserves the original framework's strengths — high-fidelity synthesis and structure preservation — while adding powerful user control through masks.

---

## 🔨 Features

- Plug-and-play extension of Stable Diffusion v1.4 (no retraining needed)
- Supports both **manual masks** and **auto-generated semantic masks**
- Coarse-to-fine self-rectification using masked KV injection
- Built-in scripts for ablation studies, figure export, and result analysis

---

## 🛠️ Setup

We recommend Python ≥ 3.10 and conda environment:

```bash
git clone https://github.com/theKwabena/mask-aware-self-rectification.git
cd mask-aware-self-rectification
pip install -r requirements.txt
```

We rely on Hugging Face's [diffusers](https://github.com/huggingface/diffusers), `torch`, and optionally `torchvision` for masks.

---

## 🚀 Quick Start

### Generate Results from Main Pipeline

```bash
python main.py
```

Edit the top of `main.py` to configure:

```python
P1 = 20
P2 = 5
S1 = 20
S2 = 5

ref_images = torch.cat([
    load_image('./images/aug/203.jpg', 512, device),
    load_image('./images/aug/203-1.jpg', 512, device),
    ...
])
target_image = load_image('./images/tgts/203-1.jpg', 512, device)
target_mask = load_mask('./images/masks/203-1-mask.png', 512, device)
```

---

## 🎛️ Mask Control

We support:

- **Manual masks** (grayscale PNG masks in `images/masks/`)
- **Auto-masks** with DeepLabV3:

```bash
python utils/make_mask_auto.py --input ./images/tgts/203-1.jpg --output ./images/masks/203-1-mask.png
```

---

## 🔍 Ablation: With vs Without Mask

To run both variants and save results side-by-side:

```bash
python ablate.py --id 203
```

This will save results to:

```
ablation_results/
  └── 203/
      ├── no_mask/
      └── global_mask/
```

---

## 📂 Directory Structure

```
assets/
  └── <sample_id>/
      ├── ref.jpg
      ├── target.jpg
      └── result.jpg

images/
  ├── refs/
  ├── tgts/
  ├── masks/      # Optional mask images
  └── aug/        # Reference augmentations

tools/
  ├── gen_bg.py          # for background init
  ├── gen_mask.py        # DeepLabV3-based mask generator
  └── gen_comparison.py  # generate result strips
  
utils/
  ├── make_mask.py          # Manual mask generator  
  └── make_mask_auto.py    # Automated mask generated

masactrl/
  └── ... (attention controller code)

main.py              # standard run
ablate.py            # ablation with/without mask
requirements.txt
```

---

## 📊 Examples

| Reference | Target | No Mask Output | Mask-Aware Output |
|-----------|--------|----------------|--------------------|
| ![](./assets/203/ref.jpg) | ![](./assets/203/target.jpg) | ![](./assets/203/no_mask.jpg) | ![](./assets/203/result.jpg) |

---

## 📜 Citation

If you use this code, please cite both the original paper and our extension:

```bibtex
@inproceedings{zhou2024generating,
  title={Generating Non-Stationary Textures using Self-Rectification},
  author={Zhou, Yujun and Zhang, Hongzhi and Fu, Jiangtao and Chen, Baoquan},
  booktitle={CVPR},
  year={2024}
}
```

---

## 🙏 Acknowledgements

This work extends [MasaCtrl](https://github.com/TencentARC/MasaCtrl) and the original [Self-Rectification](https://github.com/xiaorongjun000/Self-Rectification). We thank the authors for releasing their models and code.
