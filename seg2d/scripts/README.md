# Scripts

Planned command-line entry points:

- `smoke_*.py`: smoke tests that validate one narrow pipeline boundary.
- `inspect_fundus_avseg.py`: inspect image, annotation, and metadata layout.
- `smoke_forward.py`: run a one-batch dataset/model/loss/backward check.
- `smoke_loss_metrics.py`: run a one-batch combined-loss and metrics check.
- `smoke_train_step.py`: run one train batch and one validation batch.
- `train.py`: train the hand-written U-Net baseline and save checkpoints.
- `evaluate.py`: evaluate checkpoints on official dataset splits.
- `predict.py`: run single-image or folder inference with visualization output.

Run the dataset inspector after placing the dataset under `seg2d/data`:

```bash
python seg2d/scripts/inspect_fundus_avseg.py
```

Use `--root /path/to/Fundus-AVSeg` if the dataset is stored elsewhere.

Run the U-Net smoke test after installing PyTorch:

```bash
PYTHONPATH=seg2d python seg2d/scripts/smoke_forward.py
```

Run the loss/metrics smoke test:

```bash
PYTHONPATH=seg2d python seg2d/scripts/smoke_loss_metrics.py
```

Run the train-step smoke test:

```bash
PYTHONPATH=seg2d python seg2d/scripts/smoke_train_step.py
```

Run baseline training:

```bash
PYTHONPATH=seg2d python seg2d/scripts/train.py --config seg2d/configs/fundus_avseg_unet.yaml
```

Evaluate a checkpoint:

```bash
PYTHONPATH=seg2d python seg2d/scripts/evaluate.py --split test
```

Predict one image:

```bash
PYTHONPATH=seg2d python seg2d/scripts/predict.py \
  --input seg2d/data/Fundus-AVSeg/images/001_G.png
```
