# Data

The fastMRI dataset is not included in this repository.

Place the official fastMRI brain `.h5` files in `data/raw/`, then run:

```bash
python prepare_dataset.py --input-dir data/raw --output-dir data/npz
python create_splits.py --src data/npz --out data/splits
```

The training scripts read `data/splits/train`, `data/splits/val`, and `data/splits/test`.
Data files are excluded by `.gitignore`.
