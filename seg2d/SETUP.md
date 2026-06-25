# Seg2D Server Setup

This guide prepares a server environment for the Fundus-AVSeg 2D segmentation
pipeline.

## Clone

```bash
git lfs install
git clone https://github.com/ZhenchangLiu/bme.git
cd bme
git lfs pull
```

Confirm the dataset files are real images rather than Git LFS pointers:

```bash
file seg2d/data/Fundus-AVSeg/images/001_G.png
file seg2d/data/Fundus-AVSeg/metadata.xlsx
du -sh seg2d/data/Fundus-AVSeg
```

Expected size is about `205M`.

## Conda

```bash
conda create -n bme-seg python=3.11 -y
conda activate bme-seg
```

Install PyTorch using the CUDA wheel that matches the server. For example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

For CUDA 12.4:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Then install the non-PyTorch dependencies:

```bash
pip install -r seg2d/requirements.txt
```

Set the package path:

```bash
export PYTHONPATH=$PWD/seg2d:$PYTHONPATH
```

## Smoke Tests

Inspect data:

```bash
python seg2d/scripts/inspect_fundus_avseg.py
```

Check dataset/model forward and backward:

```bash
PYTHONPATH=seg2d python seg2d/scripts/smoke_forward.py
```

Check loss and metrics:

```bash
PYTHONPATH=seg2d python seg2d/scripts/smoke_loss_metrics.py
```

Check a tiny train/validation step:

```bash
PYTHONPATH=seg2d python seg2d/scripts/smoke_train_step.py
```

After a checkpoint exists, check evaluation and prediction:

```bash
PYTHONPATH=seg2d python seg2d/scripts/smoke_eval_predict.py
```

## Train

Short run:

```bash
PYTHONPATH=seg2d python seg2d/scripts/train.py --epochs 2 --batch-size 1 --num-workers 0
```

Disable AMP for debugging:

```bash
PYTHONPATH=seg2d python seg2d/scripts/train.py --epochs 2 --batch-size 1 --num-workers 0 --no-amp
```

Default config run:

```bash
PYTHONPATH=seg2d python seg2d/scripts/train.py --config seg2d/configs/fundus_avseg_unet.yaml
```

8-GPU distributed smoke run:

```bash
PYTHONPATH=seg2d torchrun --standalone --nproc_per_node=8 \
  seg2d/scripts/train.py \
  --epochs 1 \
  --batch-size 1 \
  --num-workers 0 \
  --limit-train-batches 1 \
  --limit-val-batches 1
```

8-GPU distributed full run:

```bash
PYTHONPATH=seg2d torchrun --standalone --nproc_per_node=8 \
  seg2d/scripts/train.py \
  --config seg2d/configs/fundus_avseg_unet.yaml \
  --batch-size 2 \
  --num-workers 4
```

`--batch-size` is per GPU, so the effective global batch size is
`batch_size * nproc_per_node`.

The default config enables train-time augmentation, CUDA AMP, cosine learning
rate scheduling, per-class history metrics, and validation prediction overlays.

Evaluate:

```bash
PYTHONPATH=seg2d python seg2d/scripts/evaluate.py --split val --batch-size 1 --num-workers 0
PYTHONPATH=seg2d python seg2d/scripts/evaluate.py --split test --batch-size 1 --num-workers 0
```

Predict:

```bash
PYTHONPATH=seg2d python seg2d/scripts/predict.py \
  --input seg2d/data/Fundus-AVSeg/images/001_G.png
```
