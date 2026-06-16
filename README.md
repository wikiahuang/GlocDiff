# GlocDiff

This repository is the official implementation of paper "Robust Floor Plan-Guided Visual Navigation Incorporating Depth and Directional Cues".

## Overview

GlocDiff predicts a robot's future trajectory with a diffusion policy conditioned on two parts:
- **Shortest path**: the shortest-path waypoints to the goal are fed directly into the 1D conditional U-Net as the local condition.
- **Depth + floor plan + pose + goal**: a short history of RGB observations is converted to depth latents with a pretrained [Marigold](https://huggingface.co/prs-eth/marigold-depth-lcm-v1-0) depth model and encoded by `depth_latent_processor` (MLP + Transformer encoder); this is fused with the scene's floor plan image, current position/heading, and goal position by `deep_floor_net` (EfficientNet-b0 + MLP fusion) into a single global condition embedding.

The U-Net (`ConditionalUnet1D`, from [diffusion_policy](https://github.com/real-stanford/diffusion_policy)) is trained with a DDPM noise scheduler to denoise future waypoints given these two conditions.

```
model/
  glocdiff.py               # top-level module wrapping the three submodules below
  depth_latent_processor.py # depth latents -> condition embedding
  deepfloor_net.py          # floor plan + pose + goal -> condition embedding
  rgb_processer.py          # RGB-only condition embedding (for ablation, currently unused)
utils/
  train.py                  # training entry point
  train_utils.py            # training loop, loss computation, visualization
  glocdiff_dataset.py        # dataset class
  data_utils.py              # coordinate/image transform helpers
```

## Training

### Pre-requisites
- Linux, NVIDIA GPU(s) with a CUDA 12.4-compatible driver
- conda (or miniconda)

### Setup
1. Create the conda environment:
   ```bash
   conda env create -f config/environment.yaml
   conda activate GlocDiff
   ```
2. `diffusion_policy` is not published on PyPI and must be cloned into the project root and installed in editable mode:
   ```bash
   cd GlocDiff
   git clone https://github.com/real-stanford/diffusion_policy.git
   pip install -e diffusion_policy/
   ```
   It must live at `GlocDiff/diffusion_policy/` for the import path in `utils/train.py` to resolve correctly.

### Data Preparation
1. Download Dataset

   [TODO: dataset download link]

### Let's train GlocDiff!

1. Fill in `config/glocdiff.yaml`: at minimum set `datasets.data_folder`, `datasets.traversable_map_folder`, and `datasets.scene_names` to your own data. Set `logger` to `tensorboard`, `wandb`, or `none`, and `load_run` to `null` for a fresh run (or to a checkpoint directory / a standalone `.pth` weights file to resume/warm-start from).

2. Launch training with `torchrun` (or `python -m torch.distributed.run` if `torchrun`'s own shebang doesn't match your conda environment's Python):
   ```bash
   cd utils
   torchrun --nproc_per_node=<num_gpus> --master_port=29500 train.py --config ../config/glocdiff.yaml
   ```

Checkpoints are saved every epoch under `utils/logs/<project_name>/<run_name>/`.

## Test in Simulator
