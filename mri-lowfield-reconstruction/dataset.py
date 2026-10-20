import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.fft as fft

class MRIDataset(Dataset):
    def __init__(self, root_dir, noise_std=0.05, res_ratio=0.7, bias_field=True, noise_seed=None):
        """
        Args:
            root_dir: 数据路径 (npz文件)
            noise_std: 噪声标准差 (模拟低信噪比)
            res_ratio: K空间保留比例 (0.5-0.8 模拟低场低分辨率模糊感)
            bias_field: 是否添加亮度偏移场 (模拟磁场不均匀)
            noise_seed: 随机种子
        """
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(".npz")]
        self.files.sort()
        
        self.noise_std = float(noise_std)
        self.res_ratio = float(res_ratio)
        self.bias_field = bias_field
        self.rng = np.random.RandomState(noise_seed) if noise_seed is not None else np.random.RandomState()

        print(f"✅ 数据加载完成，共 {len(self.files)} 个样本")
        print(f"🛠  低场模拟参数: 噪声={self.noise_std}, 分辨率比例={self.res_ratio}, 偏置场={self.bias_field}")

    def __len__(self):
        return len(self.files)

    def _simulate_low_res(self, x):
        """通过K空间截断模拟低分辨率"""
        h, w = x.shape
        # 转到 K 空间
        x_complex = torch.from_numpy(x).to(torch.complex64)
        k_space = fft.fftshift(fft.fft2(x_complex))
        
        # 创建中心掩码
        mask = torch.zeros((h, w))
        rh, rw = int(h * self.res_ratio), int(w * self.res_ratio)
        mask[h//2-rh//2 : h//2+rh//2, w//2-rw//2 : w//2+rw//2] = 1
        
        # 应用掩码并转回图像域
        k_space_low = k_space * mask
        img_low = fft.ifft2(fft.ifftshift(k_space_low))
        return np.abs(img_low.numpy())

    def _apply_bias_field(self, x):
        """模拟亮度不均匀"""
        h, w = x.shape
        y_coord = np.linspace(-1, 1, h)
        x_coord = np.linspace(-1, 1, w)
        xx, yy = np.meshgrid(x_coord, y_coord)
        
        # 随机中心点和范围的Gaussian Field
        center_x = self.rng.uniform(-0.5, 0.5)
        center_y = self.rng.uniform(-0.5, 0.5)
        sigma = self.rng.uniform(0.5, 1.5)
        bias = np.exp(-((xx - center_x)**2 + (yy - center_y)**2) / (2.0 * sigma**2))
        
        return x * (0.6 + 0.6 * bias) # 缩放亮度范围

    def _add_rician_noise(self, x):
        """添加莱斯噪声 (MRI幅度图的真实噪声)"""
        n1 = self.rng.normal(0, self.noise_std, x.shape)
        n2 = self.rng.normal(0, self.noise_std, x.shape)
        return np.sqrt((x + n1)**2 + n2**2)

    def __getitem__(self, idx):
        try:
            data = np.load(self.files[idx])
            gt = data["target"].astype(np.float32)
        except Exception as e:
            print(f"Error loading {self.files[idx]}: {e}")
            return torch.zeros((1, 256, 256)), torch.zeros((1, 256, 256))

        # 1. 基础归一化
        gt = gt / (gt.max() + 1e-8)
        
        # 2. 模拟退化过程
        # 第一步：低分辨率模糊
        inp = self._simulate_low_res(gt)
        # 第二步：亮度偏移
        if self.bias_field:
            inp = self._apply_bias_field(inp)
        # 第三步：加入莱斯噪声
        if self.noise_std > 0:
            inp = self._add_rician_noise(inp)

        # 3. 再次归一化输入图像并防止溢出
        inp = inp / (inp.max() + 1e-8)
        inp = np.clip(inp, 0, 1)

        # 转 tensor
        inp_tensor = torch.from_numpy(inp).float().unsqueeze(0)
        gt_tensor = torch.from_numpy(gt).float().unsqueeze(0)

        return inp_tensor, gt_tensor

if __name__ == "__main__":
    # 简单的本地测试
    ds = MRIDataset(".", noise_std=0.05)
    print("Dataset Test OK")