# Scripts

Planned command-line entry points:

- `inspect_fundus_avseg.py`: inspect image, annotation, and metadata layout.
- `train.py`: train the hand-written U-Net baseline.
- `evaluate.py`: evaluate checkpoints on the official test split.
- `predict.py`: run single-image or folder inference with visualization output.

Run the dataset inspector after placing the dataset under `seg2d/data`:

```bash
python seg2d/scripts/inspect_fundus_avseg.py
```

Use `--root /path/to/Fundus-AVSeg` if the dataset is stored elsewhere.
