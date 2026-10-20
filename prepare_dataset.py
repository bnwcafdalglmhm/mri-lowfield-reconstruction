import argparse
import os

import h5py
import numpy as np
from tqdm import tqdm


def ifft2c(kspace):
    return np.fft.ifftshift(
        np.fft.ifft2(np.fft.fftshift(kspace, axes=(-2, -1))),
        axes=(-2, -1),
    )


def rss_combine(img):
    return np.sqrt((np.abs(img) ** 2).sum(axis=0))


def center_crop(img, size=320):
    h, w = img.shape
    if h < size or w < size:
        raise ValueError(f"Image {img.shape} is smaller than crop size {size}")
    startx = w // 2 - size // 2
    starty = h // 2 - size // 2
    return img[starty:starty + size, startx:startx + size]


def create_mask(shape, accel=4, center_fraction=0.08, rng=None):
    rng = rng or np.random
    mask = np.zeros(shape)
    num_low_freqs = int(shape[-2] * center_fraction)
    center = shape[-2] // 2
    mask[:, center - num_low_freqs // 2:center + num_low_freqs // 2, :] = 1
    prob = (shape[-2] / accel - num_low_freqs) / (shape[-2] - num_low_freqs)
    random_mask = rng.rand(*shape) < prob
    return np.logical_or(mask, random_mask)


def parse_args():
    p = argparse.ArgumentParser(description="Convert fastMRI H5 files to 320x320 NPZ slices")
    p.add_argument("--input-dir", default="./data/raw", help="Directory containing fastMRI .h5 files")
    p.add_argument("--output-dir", default="./data/npz", help="Directory for converted .npz slices")
    p.add_argument("--size", type=int, default=320, help="Center crop size")
    p.add_argument("--accel", type=int, default=4, help="Acceleration factor for the undersampling mask")
    p.add_argument("--seed", type=int, default=0, help="Random seed for undersampling masks")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {args.input_dir}")

    for fname in tqdm(files):
        with h5py.File(os.path.join(args.input_dir, fname), "r") as f:
            kspace = f["kspace"][()]
            target = f["reconstruction_rss"][()]

        stem = os.path.splitext(fname)[0]
        for i in range(kspace.shape[0]):
            ks = kspace[i]
            mask = create_mask(ks.shape, accel=args.accel, rng=rng)
            img = rss_combine(ifft2c(ks * mask))
            img = center_crop(img, args.size)
            tgt = center_crop(target[i], args.size)
            save_path = os.path.join(args.output_dir, f"{stem}_slice{i}.npz")
            np.savez_compressed(save_path, input=img.astype(np.float32), target=tgt.astype(np.float32))

    print(f"Converted {len(files)} H5 files to {args.output_dir}")


if __name__ == "__main__":
    main()
