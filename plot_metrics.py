import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

p = argparse.ArgumentParser(description="Plot training/validation metrics from experiment folders")
p.add_argument("--experiments-dir", default="./experiments", help="Directory containing experiment subfolders")
p.add_argument("--output-dir", default=None, help="Output directory (default: <experiments-dir>/plots)")
p.add_argument("--exclude", default="plots", help="Comma-separated folder names to skip")
args = p.parse_args()

EXPERIMENTS_DIR = args.experiments_dir
EXCLUDE = [x.strip() for x in args.exclude.split(",") if x.strip()]
OUTPUT_DIR = args.output_dir or os.path.join(EXPERIMENTS_DIR, "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)

subdirs = sorted([
    d for d in os.listdir(EXPERIMENTS_DIR)
    if os.path.isdir(os.path.join(EXPERIMENTS_DIR, d)) and d not in EXCLUDE
])

# Group by prefix (noise_*, ratio_*)
groups = {}
for d in subdirs:
    prefix = d.rsplit('_', 1)[0]
    groups.setdefault(prefix, []).append(d)

for prefix, dirs in groups.items():
    # --- Train Loss ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for d in dirs:
        path = os.path.join(EXPERIMENTS_DIR, d, 'metrics_train.csv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        ax.plot(df['epoch'], df['train_loss'], label=d)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train Loss')
    ax.set_title(f'Training Loss — {prefix}')
    ax.legend()
    ax.grid(True)
    out = os.path.join(OUTPUT_DIR, f'{prefix}_train_loss.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

    # --- Val PSNR ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for d in dirs:
        path = os.path.join(EXPERIMENTS_DIR, d, 'metrics_val.csv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        ax.plot(df['epoch'], df['val_psnr'], label=d)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val PSNR')
    ax.set_title(f'Validation PSNR — {prefix}')
    ax.legend()
    ax.grid(True)
    out = os.path.join(OUTPUT_DIR, f'{prefix}_val_psnr.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

    # --- Val SSIM ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for d in dirs:
        path = os.path.join(EXPERIMENTS_DIR, d, 'metrics_val.csv')
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        ax.plot(df['epoch'], df['val_ssim'], label=d)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val SSIM')
    ax.set_title(f'Validation SSIM — {prefix}')
    ax.legend()
    ax.grid(True)
    out = os.path.join(OUTPUT_DIR, f'{prefix}_val_ssim.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

# Also create combined overview: all experiments on one figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for d in subdirs:
    trn_path = os.path.join(EXPERIMENTS_DIR, d, 'metrics_train.csv')
    val_path = os.path.join(EXPERIMENTS_DIR, d, 'metrics_val.csv')
    if os.path.exists(trn_path):
        df = pd.read_csv(trn_path)
        axes[0].plot(df['epoch'], df['train_loss'], label=d)
    if os.path.exists(val_path):
        df = pd.read_csv(val_path)
        axes[1].plot(df['epoch'], df['val_psnr'], label=d)
        axes[2].plot(df['epoch'], df['val_ssim'], label=d)

axes[0].set_title('Train Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend(fontsize=6)
axes[0].grid(True)

axes[1].set_title('Val PSNR')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('PSNR')
axes[1].legend(fontsize=6)
axes[1].grid(True)

axes[2].set_title('Val SSIM')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('SSIM')
axes[2].legend(fontsize=6)
axes[2].grid(True)

fig.tight_layout()
out = os.path.join(OUTPUT_DIR, 'all_experiments_overview.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out}')
print('Done!')
