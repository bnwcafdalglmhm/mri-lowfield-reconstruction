#!/usr/bin/env python3
import os
import argparse
import csv
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from dataset import MRIDataset
from fastmri.models.unet import Unet


def read_detailed_metrics(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                idx = int(r.get('index', -1))
            except:
                idx = -1
            file = r.get('file', '')
            try:
                psnr = float(r.get('psnr', 0.0))
            except:
                psnr = 0.0
            try:
                ssim = float(r.get('ssim', 0.0))
            except:
                ssim = 0.0
            rows.append({'index': idx, 'file': file, 'psnr': psnr, 'ssim': ssim})
    return rows


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def find_file_in_dataset(ds, basename):
    for i, f in enumerate(ds.files):
        if os.path.basename(f) == basename:
            return i, f
    return None, None


def save_npz(outpath, inp, pred, gt):
    np.savez_compressed(outpath, input=inp.astype(np.float32), pred=pred.astype(np.float32), gt=gt.astype(np.float32))


def plot_residual_heatmap(res, outpath, vmax=None):
    plt.figure(figsize=(6,6))
    if vmax is None:
        vmax = np.max(np.abs(res)) if res.size>0 else 1.0
    plt.imshow(res, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar()
    plt.title('Residual (pred - gt)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_residual_hist(res, outpath):
    plt.figure(figsize=(6,4))
    plt.hist(res.ravel(), bins=100, color='C3', alpha=0.8)
    plt.axvline(np.mean(res), color='r', linestyle='--', label=f"mean {np.mean(res):.4e}")
    plt.axvline(np.median(res), color='g', linestyle='--', label=f"median {np.median(res):.4e}")
    plt.legend()
    plt.xlabel('Residual value')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def analyze(metrics_csv, topk, outdir, data_root, ckpt, test_images_dir, noise, res_ratio, no_bias, device):
    rows = read_detailed_metrics(metrics_csv)
    if len(rows) == 0:
        raise RuntimeError('no rows in metrics csv')

    rows_sorted = sorted(rows, key=lambda x: x['psnr'])
    worst = rows_sorted[:topk]

    # prepare folders
    raw_dir = os.path.join(outdir, 'raw_npz')
    imgs_dir = os.path.join(outdir, 'result_images')
    preds_dir = os.path.join(outdir, 'pred_npz')
    res_dir = os.path.join(outdir, 'residuals')
    hist_dir = os.path.join(outdir, 'histograms')
    ensure_dir(raw_dir)
    ensure_dir(imgs_dir)
    ensure_dir(preds_dir)
    ensure_dir(res_dir)
    ensure_dir(hist_dir)

    # load dataset and model
    ds = MRIDataset(root_dir=data_root, noise_std=noise, res_ratio=res_ratio, bias_field=(not no_bias), noise_seed=0)
    model = Unet(in_chans=1, out_chans=1, chans=128, num_pool_layers=4)
    model.to(device)
    model.eval()
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)

    out_rows = []

    for r in worst:
        idx = int(r['index'])
        fname = r['file']
        print(f"Processing idx={idx}, file={fname}")

        # copy raw npz
        ds_idx, fullpath = find_file_in_dataset(ds, fname)
        if fullpath is None:
            # fallback: look under data_root
            candidate = os.path.join(data_root, fname)
            if os.path.exists(candidate):
                fullpath = candidate
                ds_idx = idx if idx >=0 else None
        if fullpath is None:
            print(f"Warning: cannot find raw file for {fname}")
        else:
            try:
                shutil.copy2(fullpath, os.path.join(raw_dir, os.path.basename(fullpath)))
            except Exception as e:
                print(f"Failed to copy raw: {e}")

        # copy result image if exists
        src_img = os.path.join(test_images_dir, f"result_idx{idx}.png")
        if os.path.exists(src_img):
            shutil.copy2(src_img, os.path.join(imgs_dir, os.path.basename(src_img)))

        # get sample using dataset (prefer ds_idx)
        if ds_idx is None:
            # try to find by idx provided
            try:
                inp_t, gt_t = ds[idx]
            except Exception as e:
                print(f"Cannot load from dataset by idx {idx}: {e}")
                continue
        else:
            inp_t, gt_t = ds[ds_idx]

        inp = inp_t.numpy().squeeze()
        gt = gt_t.numpy().squeeze()

        # inference
        with torch.no_grad():
            x = inp_t.unsqueeze(0).to(device)
            pred_t = model(x)
            pred = pred_t.cpu().squeeze().numpy()

        # save preds/arrays
        npz_out = os.path.join(preds_dir, f"idx{idx}_{fname}.npz")
        save_npz(npz_out, inp, pred, gt)

        # residual and plots
        res = pred - gt
        mean_res = float(np.mean(res))
        std_res = float(np.std(res))
        mae = float(np.mean(np.abs(res)))
        max_abs = float(np.max(np.abs(res)))

        # residual heatmap
        hm_path = os.path.join(res_dir, f"res_idx{idx}.png")
        plot_residual_heatmap(res, hm_path, vmax=max_abs)

        # residual histogram
        hist_path = os.path.join(hist_dir, f"hist_idx{idx}.png")
        plot_residual_hist(res, hist_path)

        # additional stats: gt/input dynamic range
        gt_max, gt_min, gt_mean, gt_std = float(gt.max()), float(gt.min()), float(gt.mean()), float(gt.std())
        inp_max, inp_min, inp_mean, inp_std = float(inp.max()), float(inp.min()), float(inp.mean()), float(inp.std())
        pred_max, pred_min, pred_mean, pred_std = float(pred.max()), float(pred.min()), float(pred.mean()), float(pred.std())

        out_rows.append({
            'index': idx,
            'file': fname,
            'psnr': r['psnr'],
            'ssim': r['ssim'],
            'gt_max': gt_max,
            'gt_min': gt_min,
            'gt_mean': gt_mean,
            'gt_std': gt_std,
            'inp_max': inp_max,
            'inp_min': inp_min,
            'inp_mean': inp_mean,
            'inp_std': inp_std,
            'pred_max': pred_max,
            'pred_min': pred_min,
            'pred_mean': pred_mean,
            'pred_std': pred_std,
            'res_mean': mean_res,
            'res_std': std_res,
            'res_mae': mae,
            'res_maxabs': max_abs,
            'res_heatmap': hm_path,
            'res_hist': hist_path,
            'result_image': os.path.join(imgs_dir, os.path.basename(src_img)) if os.path.exists(src_img) else ''
        })

    # write analysis csv
    out_csv = os.path.join(outdir, 'worst_samples_analysis.csv')
    keys = list(out_rows[0].keys()) if len(out_rows)>0 else []
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for rr in out_rows:
            writer.writerow(rr)

    print(f"Analysis finished. Files written to: {outdir}")
    print(f"- Raw npz copied: {raw_dir}")
    print(f"- Result images copied: {imgs_dir}")
    print(f"- Pred arrays: {preds_dir}")
    print(f"- Residual heatmaps: {res_dir}")
    print(f"- Residual histograms: {hist_dir}")
    print(f"- Summary CSV: {out_csv}")
    return out_csv


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--metrics', required=True, help='metrics.csv path')
    p.add_argument('--topk', type=int, default=5)
    p.add_argument('--outdir', default='./test_results/worst_analysis')
    p.add_argument('--data-root', default='./data/splits/test')
    p.add_argument('--ckpt', default='./checkpoints/exp_v3/best_unet_lowfield.pth')
    p.add_argument('--test-images-dir', default='./test_results/exp_v3')
    p.add_argument('--noise', type=float, default=0.05)
    p.add_argument('--res-ratio', type=float, default=0.7)
    p.add_argument('--no-bias', action='store_true')
    p.add_argument('--device', default=None)
    args = p.parse_args()

    dev = args.device if args.device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
    analyze(args.metrics, args.topk, args.outdir, args.data_root, args.ckpt, args.test_images_dir, args.noise, args.res_ratio, args.no_bias, dev)
