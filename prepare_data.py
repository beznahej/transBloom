import argparse
import pickle
import random
import shutil
from pathlib import Path
from typing import List

import scipy.io
from PIL import Image


CIFAR_FLOWER_FINE_LABELS = {"orchid", "poppy", "rose", "sunflower", "tulip"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


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
    with (cifar_root / split_file).open("rb") as f:
        obj = pickle.load(f, encoding="bytes")
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
    with (cifar_root / "meta").open("rb") as f:
        meta = pickle.load(f, encoding="bytes")
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


def list_image_files(root: Path) -> List[Path]:
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def sample_paths(paths: List[Path], target: int, balanced: bool, rng: random.Random) -> List[Path]:
    if not balanced:
        return paths

    sample_size = min(target, len(paths))
    if sample_size <= 0:
        return []
    selected = rng.sample(paths, k=sample_size)
    return sorted(selected)


def copy_images(paths: List[Path], dst: Path, prefix: str) -> int:
    copied = 0
    for idx, src in enumerate(paths):
        ext = src.suffix.lower() if src.suffix else ".jpg"
        out_name = f"{prefix}_{idx:06d}{ext}"
        shutil.copy2(src, dst / out_name)
        copied += 1
    return copied


def export_folder_non_flowers(
    non_flower_root: Path,
    output_root: Path,
    train_target: int,
    val_target: int,
    balanced: bool,
    seed: int,
    val_ratio: float,
) -> None:
    train_dst = output_root / "train" / "non_flower"
    val_dst = output_root / "val" / "non_flower"
    clean_dir(train_dst)
    clean_dir(val_dst)

    rng = random.Random(seed)
    split_train_dir = non_flower_root / "train"
    split_val_dir = non_flower_root / "val"

    if split_train_dir.exists() and split_val_dir.exists():
        train_candidates = list_image_files(split_train_dir)
        val_candidates = list_image_files(split_val_dir)
        split_mode = "presplit(train/val)"
    else:
        all_candidates = list_image_files(non_flower_root)
        if not all_candidates:
            raise ValueError(f"No image files found under --non-flower-root: {non_flower_root}")
        if not (0.0 < val_ratio < 1.0):
            raise ValueError(f"--non-flower-val-ratio must be between 0 and 1, got {val_ratio}")

        shuffled = list(all_candidates)
        rng.shuffle(shuffled)
        if len(shuffled) == 1:
            train_candidates = shuffled
            val_candidates = []
        else:
            val_count = int(round(len(shuffled) * val_ratio))
            val_count = min(max(1, val_count), len(shuffled) - 1)
            val_candidates = shuffled[:val_count]
            train_candidates = shuffled[val_count:]
        split_mode = f"single_dir(val_ratio={val_ratio})"

    selected_train = sample_paths(train_candidates, train_target, balanced, rng)
    selected_val = sample_paths(val_candidates, val_target, balanced, rng)

    copied_train = copy_images(selected_train, train_dst, prefix="hr_train")
    copied_val = copy_images(selected_val, val_dst, prefix="hr_val")

    print(f"non_flower source={non_flower_root} mode={split_mode}")
    print(
        f"train: non_flower={copied_train} "
        f"(available={len(train_candidates)}, balanced={balanced})"
    )
    print(
        f"val: non_flower={copied_val} "
        f"(available={len(val_candidates)}, balanced={balanced})"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare binary flower/non_flower dataset from Oxford Flowers102 and "
            "a configurable non_flower source (CIFAR-100 or high-res folder)."
        )
    )
    parser.add_argument("--flowers-root", type=str, default="data/flowers102")
    parser.add_argument("--cifar-root", type=str, default="data/cifar100")
    parser.add_argument(
        "--non-flower-source",
        choices=["cifar100", "folder"],
        default="cifar100",
        help=(
            "cifar100: existing low-res CIFAR pipeline. "
            "folder: ingest high-res non_flower images from --non-flower-root."
        ),
    )
    parser.add_argument(
        "--non-flower-root",
        type=str,
        default="",
        help=(
            "Root folder for high-res non_flower images. "
            "If it contains train/ and val/ subfolders, those are used; "
            "otherwise files are split by --non-flower-val-ratio."
        ),
    )
    parser.add_argument(
        "--non-flower-val-ratio",
        type=float,
        default=0.2,
        help="Validation ratio used when --non-flower-source folder has no train/val split.",
    )
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
    non_flower_root = Path(args.non_flower_root) if args.non_flower_root else Path()

    ensure_exists(flowers_root / "setid.mat", "Oxford setid.mat")
    ensure_exists(flowers_root / "jpg", "Oxford jpg directory")

    print("Preparing flower class from Oxford Flowers102...")
    train_flowers, val_flowers = copy_oxford_flowers(flowers_root, output_root)

    balanced = args.mode == "balanced"
    if args.non_flower_source == "cifar100":
        ensure_exists(cifar_root / "meta", "CIFAR meta file")
        ensure_exists(cifar_root / "train", "CIFAR train file")
        ensure_exists(cifar_root / "test", "CIFAR test file")
        print("Preparing non_flower class from CIFAR-100...")
        export_cifar_non_flowers(
            cifar_root=cifar_root,
            output_root=output_root,
            train_target=train_flowers,
            val_target=val_flowers,
            balanced=balanced,
            seed=args.seed,
        )
    else:
        if not args.non_flower_root:
            raise ValueError(
                "--non-flower-root is required when --non-flower-source folder is selected."
            )
        ensure_exists(non_flower_root, "non_flower folder root")
        print("Preparing non_flower class from high-res folder...")
        export_folder_non_flowers(
            non_flower_root=non_flower_root,
            output_root=output_root,
            train_target=train_flowers,
            val_target=val_flowers,
            balanced=balanced,
            seed=args.seed,
            val_ratio=args.non_flower_val_ratio,
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
