import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import argparse
import csv
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from fastmri.models.unet import Unet
from dataset import MRIDataset

def parse_args():
    p = argparse.ArgumentParser(description="Test U-Net for Low-field MRI Enhancement")
    p.add_argument("--data-path", default="./data/splits/test", help="Path to test npz files")
    p.add_argument("--ckpt", default="./checkpoints/exp_v3/best_unet_lowfield.pth", help="Path to model weights")
    p.add_argument("--noise", type=float, default=0.05, help="Rician noise level used in test")
    p.add_argument("--res-ratio", type=float, default=0.7, help="K-space resolution ratio used in test")
    p.add_argument("--no-bias", action="store_true", help="Disable bias field in test")
    p.add_argument("--num-show", type=int, default=10, help="Number of samples to visualize")
    p.add_argument("--output-dir", default="./test_results/exp_v3", help="Directory for visualizations and metrics")
    p.add_argument("--all", action="store_true", help="Evaluate all samples instead of random subset")
    p.add_argument("--residual", action="store_true", help="Use residual connection: output = input + model(input)")
    p.add_argument("--seed", type=int, default=0, help="Random seed for sample selection")
    return p.parse_args()

args = parse_args()
random.seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Unet(1, 1, 128, 4).to(device)
weight_file = args.ckpt

if not os.path.exists(weight_file):
    raise FileNotFoundError(f"Checkpoint not found: {weight_file}")

print(f"Loading weights from: {weight_file}")
model.load_state_dict(torch.load(weight_file, map_location=device))
model.eval()

# =====================
# 📦 数据 (确保标准与训练一致)
# =====================
# 这里传入训练时相同的参数
ds = MRIDataset(
    root_dir=args.data_path, 
    noise_std=args.noise, 
    res_ratio=args.res_ratio, 
    bias_field=not args.no_bias,
    noise_seed=0  # 固定随机种子，保证结果可重复
)

os.makedirs(args.output_dir, exist_ok=True)

# 选择评估全部样本或随机子集
if args.all:
    indices = list(range(len(ds)))
    num_show = len(indices)
else:
    num_show = min(args.num_show, len(ds))
    indices = random.sample(range(len(ds)), num_show)

psnr_list = []
ssim_list = []
metrics_rows = []

print(f"📊 Testing on {num_show} samples with Noise:{args.noise}, ResRatio:{args.res_ratio}")

# =====================
# 🖼️ 可视化与评估
# =====================

for i, idx in enumerate(indices):
    x, y = ds[idx] # 此时获取的 x 已经是经过物理模拟退化后的“低场图”

    with torch.no_grad():
        # 推理
        input_tensor = x.unsqueeze(0).to(device)
        pred_tensor = model(input_tensor)
        if args.residual:
            pred_tensor = input_tensor + pred_tensor
        pred = pred_tensor.cpu().squeeze().numpy()

    gt = y.squeeze().numpy()
    inp_img = x.squeeze().numpy()

    # 后处理归一化
    pred = np.clip(pred, 0, 1)
    gt = np.clip(gt, 0, 1)

    # 计算指标
    psnr_val = psnr(gt, pred, data_range=1.0)
    # 兼容性：skimage.structural_similarity 的参数在不同版本中可能不同
    try:
        ssim_val = ssim(gt, pred, data_range=1.0)
    except TypeError:
        # 备用调用
        ssim_val = ssim(gt, pred)

    psnr_list.append(psnr_val)
    ssim_list.append(ssim_val)

    # 记录每个样本的指标（索引、文件名、PSNR、SSIM）
    if hasattr(ds, 'files'):
        file_name = os.path.basename(ds.files[idx])
    else:
        file_name = f"idx_{idx}"
    metrics_rows.append((idx, file_name, psnr_val, ssim_val))

    print(f"Sample [{idx}] -> PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")

    # =========================
    # 🖼️ 绘图
    # =========================
    plt.figure(figsize=(12, 4))

    # Input (模拟的低场图像)
    plt.subplot(1, 3, 1)
    plt.imshow(inp_img, cmap="gray")
    plt.title(f"Input (Degraded)\nNoise:{args.noise} Res:{args.res_ratio}")
    plt.axis("off")

    # Prediction (模型增强后的图像)
    plt.subplot(1, 3, 2)
    plt.imshow(pred, cmap="gray")
    plt.title(f"Model Prediction\nPSNR:{psnr_val:.2f} SSIM:{ssim_val:.3f}")
    plt.axis("off")

    # GT (原始高场图像)
    plt.subplot(1, 3, 3)
    plt.imshow(gt, cmap="gray")
    plt.title("Ground Truth (High-field)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"result_idx{idx}.png"), dpi=200)
    plt.close()

# 打印平均指标
if len(psnr_list) > 0:
    print("\n" + "="*30)
    print(f"平均测试指标 (n={num_show}):")
    print(f"Average PSNR: {np.mean(psnr_list):.2f}")
    print(f"Average SSIM: {np.mean(ssim_list):.4f}")
    print("="*30)

    # 保存每个样本的指标到 CSV
    csv_path = os.path.join(args.output_dir, "metrics.csv")
    try:
        with open(csv_path, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["index", "file", "psnr", "ssim"])
            for r in metrics_rows:
                writer.writerow([r[0], r[1], f"{r[2]:.4f}", f"{r[3]:.6f}"])
        print(f"✅ Metrics saved to: {csv_path}")
    except Exception as e:
        print(f"⚠️ Failed to save CSV: {e}")