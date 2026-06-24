# 2D Image Segmentation

This directory contains the 2D medical image segmentation pipeline.

The first target dataset is Fundus-AVSeg, a public retinal artery-vein vessel
segmentation dataset. The initial implementation will use a hand-written PyTorch
U-Net instead of `segmentation_models_pytorch` or similar model libraries.

## Planned Layout

```text
seg2d/
  configs/
    fundus_avseg_unet.yaml
  data/
    README.md
  scripts/
    inspect_fundus_avseg.py
    train.py
    evaluate.py
    predict.py
  seg2d/
    datasets/
    losses/
    metrics/
    models/
    utils/
```

## Dataset

Expected local dataset layout:

```text
seg2d/data/Fundus-AVSeg/
  images/
  annotation/
  metadata.xlsx
  training.txt
  testing.txt
```

Annotation colors:

- `0`: background, black
- `1`: artery, red
- `2`: vein, blue
- `3`: artery-vein crossing, green
- `4`: uncertain vessel, white

The initial training plan is to use the official training/testing split and
derive a validation split from the official training list.

Implemented dataset utilities:

- `seg2d.utils.mask`: RGB annotation to class-id mask conversion.
- `seg2d.datasets.fundus_avseg`: official split loading and PyTorch dataset adapter.

With `val_fraction: 0.1`, the official 80 training images are split into 72
training images and 8 validation images using the configured seed.
