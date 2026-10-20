import os
import argparse
import shutil
import random


def parse_args():
    p = argparse.ArgumentParser(description="Split NPZ dataset into train/val/test folders")
    p.add_argument("--src", required=True, help="Source folder containing .npz files")
    p.add_argument("--out", default="./data_split", help="Output base folder (will contain train/val/test)")
    p.add_argument("--train", type=float, default=0.8, help="Train ratio")
    p.add_argument("--val", type=float, default=0.1, help="Validation ratio")
    p.add_argument("--test", type=float, default=0.1, help="Test ratio")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--move", action="store_true", help="Move files instead of copying")
    return p.parse_args()


def main():
    args = parse_args()

    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train+val+test ratios must sum to 1.0")

    os.makedirs(args.out, exist_ok=True)
    train_dir = os.path.join(args.out, "train")
    val_dir = os.path.join(args.out, "val")
    test_dir = os.path.join(args.out, "test")

    for d in (train_dir, val_dir, test_dir):
        os.makedirs(d, exist_ok=True)

    files = [os.path.join(args.src, f) for f in os.listdir(args.src) if f.endswith('.npz')]
    files.sort()

    rng = random.Random(args.seed)
    rng.shuffle(files)

    n = len(files)
    n_train = int(n * args.train)
    n_val = int(n * args.val)
    n_test = n - n_train - n_val

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    def _copy_or_move(lst, dst):
        for src in lst:
            dst_path = os.path.join(dst, os.path.basename(src))
            if args.move:
                shutil.move(src, dst_path)
            else:
                shutil.copy2(src, dst_path)

    _copy_or_move(train_files, train_dir)
    _copy_or_move(val_files, val_dir)
    _copy_or_move(test_files, test_dir)

    print(f"Total files: {n}")
    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    print(f"Files written to: {os.path.abspath(args.out)}")


if __name__ == '__main__':
    main()
