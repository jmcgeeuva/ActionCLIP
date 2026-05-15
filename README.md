# SemanticBox: Bounding Box-Guided Caption-Enhanced Action Recognition for Instructional Videos

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.12-blue.svg" /></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.7+-ee4c2c.svg" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" /></a>
</p>

<p align="center">
  <img src="SemanticBox.PNG" width="85%" alt="SemanticBox Architecture" />
</p>

SemanticBox combines CLIP visual-language alignment with Florence-2 region captioning to recognize instructional activities in classroom video. By grounding predictions in both full-frame context and semantically cropped bounding-box regions, the model achieves strong performance on the Education6 benchmark without requiring dense manual annotations at test time.

---

## Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Pretrained Models](#pretrained-models)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

## Installation

```bash
conda create -n semanticbox python=3.12
conda activate semanticbox

# PyTorch (CUDA 12.8)
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# Core dependencies
pip install numpy randaugment ffmpeg tqdm pyyaml pprintpp dotmap pathlib \
            opencv-python ftfy regex omegaconf scikit-learn

# Florence-2
pip install transformers==4.49.0 einops timm peft
```

See [INSTALL.md](INSTALL.md) for additional details.

---

## Data Preparation

SemanticBox is evaluated on the **Education6** dataset — six classroom activity classes recorded from a fixed overhead camera.

| Class | Label |
|-------|-------|
| 0 | Using a Book |
| 1 | Teacher Sitting |
| 2 | Teacher Standing |
| 3 | Teacher Writing |
| 4 | Using Technology Device |
| 5 | Using a Worksheet |

**Frame extraction.** Extract each video into JPEG frames named `%04d.jpg` using [ffmpeg](https://ffmpeg.org/):

```bash
ffmpeg -i video.mp4 -q:v 1 frames/%04d.jpg
```

**Split files.** Pre-built train/val splits are located in `splits/split/`. Each line has the format:

```
/path/to/video/frames NUM_FRAMES CLASS_ID
```

Update `train_list` and `val_list` in your config to point to the correct split files.

---

## Configuration

All hyperparameters live in YAML files under `configs/`. The main ablation configs are in `configs/education/ablation/`:

| Config | Description |
|--------|-------------|
| `all.yaml` | Full model: FF image + BB image + BB caption |
| `ffimg_bbimg_bbtxt.yaml` | FF image + BB image + BB text |
| `ffimg_bbimg_no_flo.yaml` | FF image + BB image (no Florence) |
| `ffimg_fftxt.yaml` | FF image + FF text only |
| `bbimg_bbtxt.yaml` | BB image + BB text only |
| `no_flo.yaml` | CLIP baseline, no Florence |

Key fields:

```yaml
network:
  arch: ViT-B/16          # CLIP backbone
  sim_header: "Transf"    # temporal fusion head

data:
  num_segments: 8         # frames sampled per clip
  batch_size: 16

  florence:
    activate: True
    use_bounded_text: True
    num_beams: 1           # greedy decode (faster, less memory)

solver:
  epochs: 50
  lr: 5.0e-6
  optim: adamw
```

---

## Training

Training uses `torchrun` for multi-GPU distributed training (DDP):

```bash
bash scripts/run_train.sh configs/education/ablation/ffimg_bbimg_bbtxt.yaml
```

This automatically detects the number of available GPUs and sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce memory fragmentation. Checkpoints and logs are saved to `exp/<type>/<arch>/<dataset>/<timestamp>/`.

To train on a specific number of GPUs:

```bash
torchrun --standalone --nproc_per_node=2 train.py \
    --config configs/education/ablation/ffimg_bbimg_bbtxt.yaml \
    --log_time $(date +"%Y%m%d_%H%M%S")
```

---

## Evaluation

```bash
bash scripts/run_test.sh configs/education/ablation/ffimg_bbimg_bbtxt.yaml
```

Pass a trained checkpoint via the `pretrain` field in the config:

```yaml
pretrain: ./exp/clip_k400/ViT-B16/education6/20250101_120000/model_best.pt
```

---

## Results

Results on the Education6 test split (single-crop, Top-1 / Top-2 accuracy):

| Method | Backbone | Florence | Top-1 (%) | Top-2 (%) |
|--------|----------|----------|-----------|-----------|
| ActionCLIP (baseline) | ViT-B/16 | — | — | — |
| SemanticBox (FF only) | ViT-B/16 | — | — | — |
| SemanticBox (BB only) | ViT-B/16 | ✓ | — | — |
| **SemanticBox (Full)** | **ViT-B/16** | **✓** | **—** | **—** |

*Fill in numbers after training.*

---

## Pretrained Models

| Config | Top-1 | Download |
|--------|-------|----------|
| `ffimg_bbimg_bbtxt` | — | Coming soon |
| `all` | — | Coming soon |

---

## Citation

If you use SemanticBox in your research, please cite:

```bibtex
@article{semanticbox2025,
  title   = {SemanticBox: Bounding Box-Guided Caption-Enhanced Action Recognition for Instructional Videos},
  author  = {},
  year    = {2025}
}
```

---

## Acknowledgments

This work is funded in part by NSF under awards 2322993 and 2000487, and by the Gates Foundation. Views expressed here are those of the authors and do not necessarily reflect positions or policies of the Gates Foundation.

<img width="3145" height="207" alt="Gates Foundation" src="https://github.com/user-attachments/assets/9b0fd657-1b51-4be9-b4b7-7c871e1fb6f5" />

Built on top of [ActionCLIP](https://github.com/sallymmx/ActionCLIP) (Wang et al.) and [Florence-2](https://huggingface.co/microsoft/Florence-2-base-ft) (Microsoft).
