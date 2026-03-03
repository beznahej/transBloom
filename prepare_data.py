import argparse
import pickle
import random
import shutil
from pathlib import Path

import scipy.io
from PIL import Image


CIFAR_FLOWER_FINE_LABELS = {"orchid", "poppy", "rose", "sunflower", "tulip"}


def ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_split_ids(setid_path: Path):
    setid = scipy.io.loadmat(setid_path)
    train_ids = [int(x) for x in setid["trnid"].reshape(-1)]
    valid_ids = [int(x) for x in setid["valid"].reshape(-1)]
    return train_ids, valid_ids


def copy_oxford_flowers(flowers_root: Path, output_root: Path):
    setid_path = flowers_root / "setid.mat"
    jpg_dir = flowers_root / "jpg"
    ensure_exists(setid_path, "Oxford split file")
    ensure_exists(jpg_dir, "Oxford image directory")

    train_ids, valid_ids = load_split_ids(setid_path)
    train_dst = output_root / "train" / "flower"
    val_dst = output_root / "val" / "flower"
    clean_dir(train_dst)
    clean_dir(val_dst)

    for split_name, ids, dst in [("train", train_ids, train_dst), ("val", valid_ids, val_dst)]:
        missing = 0
        for image_id in ids:
            src = jpg_dir / f"image_{image_id:05d}.jpg"
            if not src.exists():
                missing += 1
                continue
            out_name = f"oxford_{image_id:05d}.jpg"
            shutil.copy2(src, dst / out_name)
        print(f"{split_name}: flower={len(ids) - missing} (missing={missing})")

    return len(train_ids), len(valid_ids)


def load_cifar100_split(cifar_root: Path, split_file: str):
    obj = pickle.load(open(cifar_root / split_file, "rb"), encoding="bytes")
    images = obj[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = obj[b"fine_labels"]
    return images, labels


def export_cifar_non_flowers(
    cifar_root: Path,
    output_root: Path,
    train_target: int,
    val_target: int,
    balanced: bool,
    seed: int,
):
    meta = pickle.load(open(cifar_root / "meta", "rb"), encoding="bytes")
    fine_names = [name.decode() for name in meta[b"fine_label_names"]]
    flower_ids = {fine_names.index(name) for name in CIFAR_FLOWER_FINE_LABELS}

    train_dst = output_root / "train" / "non_flower"
    val_dst = output_root / "val" / "non_flower"
    clean_dir(train_dst)
    clean_dir(val_dst)

    split_specs = [
        ("train", "train", train_dst, train_target),
        ("val", "test", val_dst, val_target),
    ]
    rng = random.Random(seed)

    for split_name, split_file, dst, target in split_specs:
        images, labels = load_cifar100_split(cifar_root, split_file)
        keep_indices = [i for i, label in enumerate(labels) if label not in flower_ids]

        if balanced:
            sample_size = min(target, len(keep_indices))
            selected_indices = rng.sample(keep_indices, k=sample_size)
        else:
            selected_indices = keep_indices

        selected_indices.sort()
        for idx in selected_indices:
            out_name = f"cifar_{split_file}_{idx:05d}.png"
            Image.fromarray(images[idx]).save(dst / out_name)

        print(
            f"{split_name}: non_flower={len(selected_indices)} "
            f"(available={len(keep_indices)}, balanced={balanced})"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Prepare binary flower/non_flower dataset from Oxford Flowers102 + CIFAR-100."
    )
    parser.add_argument("--flowers-root", type=str, default="data/flowers102")
    parser.add_argument("--cifar-root", type=str, default="data/cifar100")
    parser.add_argument("--output-root", type=str, default="data")
    parser.add_argument(
        "--mode",
        choices=["balanced", "all_non_flower"],
        default="balanced",
        help=(
            "balanced: sample non_flower count to match flower count per split; "
            "all_non_flower: export all CIFAR-100 non-flower images."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    flowers_root = Path(args.flowers_root)
    cifar_root = Path(args.cifar_root)
    output_root = Path(args.output_root)

    ensure_exists(flowers_root / "setid.mat", "Oxford setid.mat")
    ensure_exists(flowers_root / "jpg", "Oxford jpg directory")
    ensure_exists(cifar_root / "meta", "CIFAR meta file")
    ensure_exists(cifar_root / "train", "CIFAR train file")
    ensure_exists(cifar_root / "test", "CIFAR test file")

    print("Preparing flower class from Oxford Flowers102...")
    train_flowers, val_flowers = copy_oxford_flowers(flowers_root, output_root)

    print("Preparing non_flower class from CIFAR-100...")
    balanced = args.mode == "balanced"
    export_cifar_non_flowers(
        cifar_root=cifar_root,
        output_root=output_root,
        train_target=train_flowers,
        val_target=val_flowers,
        balanced=balanced,
        seed=args.seed,
    )

    print("Done.")
    print(f"Output ready at: {output_root.resolve()}")
    print("Expected layout:")
    print("  data/train/flower")
    print("  data/train/non_flower")
    print("  data/val/flower")
    print("  data/val/non_flower")


if __name__ == "__main__":
    main()
