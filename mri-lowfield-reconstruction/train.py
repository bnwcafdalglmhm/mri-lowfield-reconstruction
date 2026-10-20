import os
import argparse
import random
import numpy as np
import torch
import csv
from torch.utils.data import DataLoader
from fastmri.models.unet import Unet
from dataset import MRIDataset
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def parse_args():
    p = argparse.ArgumentParser(description="Train U-Net for Low-field MRI Enhancement")
    p.add_argument("--data-path", default="./data/splits/train", help="Path to training npz files")
    p.add_argument("--val-path", default="./data/splits/val", help="Path to validation npz files")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save-dir", default="./checkpoints/baseline")
    
    # 低场模拟控制参数
    p.add_argument("--noise", type=float, default=0.05, help="Rician noise level")
    p.add_argument("--res-ratio", type=float, default=0.7, help="K-space resolution ratio (0.1 to 1.0)")
    p.add_argument("--no-bias", action="store_true", help="Disable bias field simulation")
    
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience on val PSNR")
    p.add_argument("--init-ckpt", default=None, help="Path to initial weights to load (optional)")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # --------------------
    # Reproducibility
    # --------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 1. 加载数据集
    train_dataset = MRIDataset(
        root_dir=args.data_path, 
        noise_std=args.noise, 
        res_ratio=args.res_ratio,
        bias_field=not args.no_bias
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 可选的验证集
    val_loader = None
    if args.val_path is not None:
        val_dataset = MRIDataset(
            root_dir=args.val_path,
            noise_std=args.noise,
            res_ratio=args.res_ratio,
            bias_field=not args.no_bias,
            noise_seed=0
        )
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 2. 定义模型 (fastMRI Unet)
    model = Unet(
        in_chans=1,
        out_chans=1,
        chans=128,
        num_pool_layers=4
    ).to(device)

    # 可选：加载初始权重（用于微调或从预训练开始）
    if args.init_ckpt is not None:
        if os.path.exists(args.init_ckpt):
            model.load_state_dict(torch.load(args.init_ckpt, map_location=device))
            print(f"🔁 已加载初始权重: {args.init_ckpt}")
        else:
            print(f"⚠️ 指定的初始权重不存在: {args.init_ckpt}")

    # 3. 损失函数与优化器
    mse_loss = torch.nn.MSELoss()
    l1_loss = torch.nn.L1Loss()

    def criterion(pred, target):
        # 混合损失函数，平衡平滑度与细节
        return 0.2 * mse_loss(pred, target) + 0.8 * l1_loss(pred, target)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    # 4. 训练循环
    best_loss = float("inf")
    best_val_psnr = -1.0
    no_improve_epochs = 0
    print(f"🚀 开始在 {device} 上训练...")

    # metrics 文件
    train_metrics_path = os.path.join(args.save_dir, "metrics_train.csv")
    val_metrics_path = os.path.join(args.save_dir, "metrics_val.csv")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} 结束, 平均 Loss: {avg_loss:.6f}, 当前学习率: {optimizer.param_groups[0]['lr']}")

        # 保存训练损失到 CSV（追加模式）
        write_header = not os.path.exists(train_metrics_path)
        try:
            with open(train_metrics_path, "a" , newline='') as tf:
                writer = csv.writer(tf)
                if write_header:
                    writer.writerow(["epoch", "train_loss"])
                writer.writerow([epoch+1, f"{avg_loss:.6f}"])
        except Exception as e:
            print(f"⚠️ 无法写入训练指标CSV: {e}")

        # 如果有验证集，则在验证集上评估并以 val PSNR 为准保存最优模型
        if val_loader is not None:
            model.eval()
            val_psnr_list = []
            val_ssim_list = []
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    vp = model(vx)
                    vp_np = vp.cpu().squeeze().numpy()
                    vy_np = vy.cpu().squeeze().numpy()
                    vp_np = np.clip(vp_np, 0, 1)
                    vy_np = np.clip(vy_np, 0, 1)
                    val_psnr_list.append(psnr(vy_np, vp_np, data_range=1.0))
                    try:
                        val_ssim_list.append(ssim(vy_np, vp_np, data_range=1.0))
                    except Exception:
                        val_ssim_list.append(0.0)

            mean_val_psnr = float(np.mean(val_psnr_list)) if len(val_psnr_list) > 0 else -1.0
            mean_val_ssim = float(np.mean(val_ssim_list)) if len(val_ssim_list) > 0 else 0.0
            print(f"验证集指标 -> PSNR: {mean_val_psnr:.3f}, SSIM: {mean_val_ssim:.4f}")

            # 保存验证指标到 CSV
            write_header = not os.path.exists(val_metrics_path)
            try:
                with open(val_metrics_path, "a", newline='') as vf:
                    writer = csv.writer(vf)
                    if write_header:
                        writer.writerow(["epoch", "val_psnr", "val_ssim"])
                    writer.writerow([epoch+1, f"{mean_val_psnr:.4f}", f"{mean_val_ssim:.6f}"])
            except Exception as e:
                print(f"⚠️ 无法写入验证指标CSV: {e}")

            # 保存基于验证 PSNR 的最优权重
            if mean_val_psnr > best_val_psnr:
                best_val_psnr = mean_val_psnr
                save_path = os.path.join(args.save_dir, "best_unet_lowfield.pth")
                torch.save(model.state_dict(), save_path)
                print(f"⭐ 已更新最优权重（基于验证PSNR）: {save_path}")
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            # 早停检查
            if no_improve_epochs >= args.patience:
                print(f"⏸ 早停触发（{args.patience} 轮未改进验证PSNR），停止训练。")
                break

        else:
            # 保存最优模型（无验证集时基于训练损失）
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_path = os.path.join(args.save_dir, "best_unet_lowfield.pth")
                torch.save(model.state_dict(), save_path)
                print(f"⭐ 已更新最优权重: {save_path}")

        # 每 5 轮保存一个备份
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"unet_epoch_{epoch+1}.pth"))

        scheduler.step()

    print("✅ 训练全部完成！")

if __name__ == "__main__":
    main()