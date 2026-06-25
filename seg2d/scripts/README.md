# Scripts

Planned command-line entry points:

- `smoke_*.py`: smoke tests that validate one narrow pipeline boundary.
- `inspect_fundus_avseg.py`: inspect image, annotation, and metadata layout.
- `smoke_forward.py`: run a one-batch dataset/model/loss/backward check.
- `smoke_loss_metrics.py`: run a one-batch combined-loss and metrics check.
- `train.py`: train the hand-written U-Net baseline.
- `evaluate.py`: evaluate checkpoints on the official test split.
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
