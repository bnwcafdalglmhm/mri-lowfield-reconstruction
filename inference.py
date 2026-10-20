import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from fastmri.models.unet import Unet
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from dataset import MRIDataset


def parse_args():
    p = argparse.ArgumentParser(description="Run one low-field MRI reconstruction sample")
    p.add_argument("--data-path", default="./data/splits/test", help="Directory containing test .npz files")
    p.add_argument("--ckpt", default="./checkpoints/exp_v3/best_unet_lowfield.pth", help="Path to model weights")
    p.add_argument("--index", type=int, default=0, help="Sample index to visualize")
    p.add_argument("--output", default="./test_results/inference.png", help="Output image path")
    p.add_argument("--noise", type=float, default=0.05, help="Rician noise level")
    p.add_argument("--res-ratio", type=float, default=0.7, help="K-space resolution ratio")
    p.add_argument("--no-bias", action="store_true", help="Disable bias field simulation")
    p.add_argument("--residual", action="store_true", help="Use residual connection")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Unet(in_chans=1, out_chans=1, chans=128, num_pool_layers=4)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device)
    model.eval()

    dataset = MRIDataset(
        root_dir=args.data_path,
        noise_std=args.noise,
        res_ratio=args.res_ratio,
        bias_field=not args.no_bias,
        noise_seed=0,
    )
    x, y = dataset[args.index]
    x = x.unsqueeze(0).to(device)
    y = y.unsqueeze(0)

    with torch.no_grad():
        pred = model(x)
        if args.residual:
            pred = x + pred

    x_np = x.cpu().squeeze().numpy()
    y_np = y.squeeze().numpy()
    pred_np = np.clip(pred.cpu().squeeze().numpy(), 0, 1)
    y_np = np.clip(y_np, 0, 1)

    psnr_value = peak_signal_noise_ratio(y_np, pred_np, data_range=1.0)
    ssim_value = structural_similarity(y_np, pred_np, data_range=1.0)
    print(f"PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, image, title in zip(
        axes,
        [x_np, pred_np, y_np],
        ["Input (degraded)", f"Prediction\nPSNR {psnr_value:.2f} / SSIM {ssim_value:.3f}", "Ground truth"],
    ):
        ax.imshow(image, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
