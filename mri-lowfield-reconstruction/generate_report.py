#!/usr/bin/env python3
import os
import argparse
import csv
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_metrics(csv_path):
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                idx = int(r.get('index', r.get('idx', '-1')))
            except Exception:
                idx = -1
            file_name = r.get('file', '')
            try:
                psnr_v = float(r.get('psnr', 0.0))
            except Exception:
                psnr_v = 0.0
            try:
                ssim_v = float(r.get('ssim', 0.0))
            except Exception:
                ssim_v = 0.0
            rows.append({'index': idx, 'file': file_name, 'psnr': psnr_v, 'ssim': ssim_v})
    return rows


def stats_array(arr):
    a = np.array(arr, dtype=float)
    return {
        'count': int(a.size),
        'mean': float(a.mean()) if a.size>0 else 0.0,
        'median': float(np.median(a)) if a.size>0 else 0.0,
        'std': float(a.std(ddof=0)) if a.size>0 else 0.0,
        'min': float(a.min()) if a.size>0 else 0.0,
        'max': float(a.max()) if a.size>0 else 0.0,
        'p5': float(np.percentile(a, 5)) if a.size>0 else 0.0,
        'p25': float(np.percentile(a, 25)) if a.size>0 else 0.0,
        'p75': float(np.percentile(a, 75)) if a.size>0 else 0.0,
        'p95': float(np.percentile(a, 95)) if a.size>0 else 0.0,
    }


def make_plots(vals_psnr, vals_ssim, outdir):
    os.makedirs(outdir, exist_ok=True)
    # PSNR histogram
    mean = float(np.mean(vals_psnr))
    med = float(np.median(vals_psnr))
    plt.figure(figsize=(6,4))
    plt.hist(vals_psnr, bins=30, color='C0', alpha=0.8)
    plt.axvline(mean, color='r', linestyle='--', label=f"mean {mean:.2f}")
    plt.axvline(med, color='g', linestyle='--', label=f"median {med:.2f}")
    plt.legend()
    plt.xlabel('PSNR')
    plt.ylabel('Count')
    plt.title('PSNR Distribution')
    plt.tight_layout()
    p_psnr = os.path.join(outdir, 'psnr_hist.png')
    plt.savefig(p_psnr, dpi=200)
    plt.close()

    # SSIM histogram
    mean_s = float(np.mean(vals_ssim))
    med_s = float(np.median(vals_ssim))
    plt.figure(figsize=(6,4))
    plt.hist(vals_ssim, bins=30, color='C1', alpha=0.8)
    plt.axvline(mean_s, color='r', linestyle='--', label=f"mean {mean_s:.4f}")
    plt.axvline(med_s, color='g', linestyle='--', label=f"median {med_s:.4f}")
    plt.legend()
    plt.xlabel('SSIM')
    plt.ylabel('Count')
    plt.title('SSIM Distribution')
    plt.tight_layout()
    p_ssim = os.path.join(outdir, 'ssim_hist.png')
    plt.savefig(p_ssim, dpi=200)
    plt.close()

    # Boxplots: 分别绘制 PSNR 与 SSIM（使用独立刻度避免尺度混淆）
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # PSNR 单独 boxplot
    axes[0].boxplot(vals_psnr)
    axes[0].set_title('PSNR Boxplot')
    axes[0].set_ylabel('PSNR')
    axes[0].grid(True, linestyle='--', alpha=0.4)

    # SSIM 单独 boxplot
    axes[1].boxplot(vals_ssim)
    axes[1].set_title('SSIM Boxplot')
    axes[1].set_ylabel('SSIM')
    axes[1].grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    p_box = os.path.join(outdir, 'boxplot.png')
    plt.savefig(p_box, dpi=200)
    plt.close()

    # Scatter PSNR vs SSIM
    plt.figure(figsize=(6,6))
    plt.scatter(vals_psnr, vals_ssim, alpha=0.7)
    plt.xlabel('PSNR')
    plt.ylabel('SSIM')
    plt.title('PSNR vs SSIM')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    p_scatter = os.path.join(outdir, 'psnr_ssim_scatter.png')
    plt.savefig(p_scatter, dpi=200)
    plt.close()

    return {'psnr_hist': p_psnr, 'ssim_hist': p_ssim, 'boxplot': p_box, 'scatter': p_scatter}


def write_augmented_csv(rows, outpath, stats_psnr, stats_ssim):
    with open(outpath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['index','file','psnr','ssim','psnr_z','ssim_z','psnr_outlier','ssim_outlier'])
        mean_p = stats_psnr['mean']
        std_p = stats_psnr['std'] if stats_psnr['std']>0 else 1e-8
        mean_s = stats_ssim['mean']
        std_s = stats_ssim['std'] if stats_ssim['std']>0 else 1e-8
        for r in rows:
            pz = (r['psnr'] - mean_p) / std_p
            sz = (r['ssim'] - mean_s) / std_s
            pout = 1 if r['psnr'] < (mean_p - 2*std_p) else 0
            sout = 1 if r['ssim'] < (mean_s - 2*std_s) else 0
            writer.writerow([r['index'], r['file'], f"{r['psnr']:.4f}", f"{r['ssim']:.6f}", f"{pz:.4f}", f"{sz:.4f}", pout, sout])


def generate_report(metrics_csv, outdir, topk=5):
    if not os.path.exists(metrics_csv):
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv}")
    rows = read_metrics(metrics_csv)
    if len(rows) == 0:
        raise RuntimeError("No metric rows found in CSV")

    vals_psnr = [r['psnr'] for r in rows]
    vals_ssim = [r['ssim'] for r in rows]

    stats_p = stats_array(vals_psnr)
    stats_s = stats_array(vals_ssim)

    report_dir = os.path.abspath(outdir)
    os.makedirs(report_dir, exist_ok=True)
    plots = make_plots(vals_psnr, vals_ssim, report_dir)

    # augmented csv
    aug_csv = os.path.join(report_dir, 'metrics_detailed.csv')
    write_augmented_csv(rows, aug_csv, stats_p, stats_s)

    # top/bottom lists
    sorted_by_psnr = sorted(rows, key=lambda x: x['psnr'])
    worst = sorted_by_psnr[:topk]
    best = sorted_by_psnr[-topk:][::-1]

    # outliers
    mean_p = stats_p['mean']
    std_p = stats_p['std'] if stats_p['std']>0 else 1e-8
    outliers = [r for r in rows if r['psnr'] < (mean_p - 2*std_p)]

    # write markdown report
    md_path = os.path.join(report_dir, 'report.md')
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# Test Report\n\n")
        f.write(f"Generated: {now}\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- Samples: **{stats_p['count']}**\n")
        f.write(f"\n### PSNR\n")
        f.write(f"- Mean: {stats_p['mean']:.4f}\n")
        f.write(f"- Median: {stats_p['median']:.4f}\n")
        f.write(f"- Std: {stats_p['std']:.4f}\n")
        f.write(f"- Min: {stats_p['min']:.4f}, Max: {stats_p['max']:.4f}\n")
        f.write(f"- 5/25/75/95 percentiles: {stats_p['p5']:.4f}/{stats_p['p25']:.4f}/{stats_p['p75']:.4f}/{stats_p['p95']:.4f}\n\n")
        f.write(f"### SSIM\n")
        f.write(f"- Mean: {stats_s['mean']:.6f}\n")
        f.write(f"- Median: {stats_s['median']:.6f}\n")
        f.write(f"- Std: {stats_s['std']:.6f}\n")
        f.write(f"- Min: {stats_s['min']:.6f}, Max: {stats_s['max']:.6f}\n")
        f.write(f"- 5/25/75/95 percentiles: {stats_s['p5']:.6f}/{stats_s['p25']:.6f}/{stats_s['p75']:.6f}/{stats_s['p95']:.6f}\n\n")

        f.write(f"## Plots\n\n")
        f.write(f"![PSNR Histogram](./{os.path.basename(plots['psnr_hist'])})\n\n")
        f.write(f"![SSIM Histogram](./{os.path.basename(plots['ssim_hist'])})\n\n")
        f.write(f"![Boxplot](./{os.path.basename(plots['boxplot'])})\n\n")
        f.write(f"![PSNR vs SSIM](./{os.path.basename(plots['scatter'])})\n\n")

        f.write(f"## Top {topk} Best PSNR\n\n")
        for r in best:
            f.write(f"- `{r['file']}` — PSNR: {r['psnr']:.4f}, SSIM: {r['ssim']:.6f}\n")
        f.write('\n')

        f.write(f"## Top {topk} Worst PSNR\n\n")
        for r in worst:
            f.write(f"- `{r['file']}` — PSNR: {r['psnr']:.4f}, SSIM: {r['ssim']:.6f}\n")
        f.write('\n')

        f.write(f"## Outliers (PSNR < mean - 2*std) — {len(outliers)} found\n\n")
        for r in outliers:
            f.write(f"- `{r['file']}` — PSNR: {r['psnr']:.4f}, SSIM: {r['ssim']:.6f}\n")
        f.write('\n')

        f.write(f"## Files\n\n")
        f.write(f"- Augmented metrics CSV: `metrics_detailed.csv`\n")
        f.write(f"- Markdown report: `report.md`\n")
        f.write('\n')

    print(f"Report generated in: {report_dir}")
    print(f"- Augmented CSV: {aug_csv}")
    print(f"- Markdown: {md_path}")
    print(f"- Plots: {', '.join(plots.values())}")
    return {'report_dir': report_dir, 'aug_csv': aug_csv, 'md': md_path, 'plots': plots}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Generate detailed report from metrics.csv')
    p.add_argument('--metrics', default='./test_results/metrics.csv', help='Path to metrics.csv')
    p.add_argument('--outdir', default='./test_results/report', help='Output report directory')
    p.add_argument('--topk', type=int, default=5, help='Top K entries to list')
    args = p.parse_args()

    res = generate_report(args.metrics, args.outdir, topk=args.topk)
