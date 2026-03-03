from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class Plan:
    image_dir: Path
    total_images: int
    delete_count: int
    keep_count: int
    selected_images: list[Path]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Randomly delete images and matching annotation files by basename. "
            "Dry-run by default; use --execute to apply."
        )
    )
    parser.add_argument(
        "--image-dirs",
        nargs="+",
        required=True,
        help="One or more image directories to prune (e.g. data/train/non_flower data/val/non_flower).",
    )
    parser.add_argument(
        "--annotation-dirs",
        nargs="*",
        default=[],
        help="Optional annotation directories. If omitted, only image-side annotations are checked.",
    )
    parser.add_argument(
        "--image-exts",
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".bmp", ".webp"],
        help="Image extensions to consider.",
    )
    parser.add_argument(
        "--annotation-exts",
        nargs="+",
        default=[".txt", ".xml", ".json"],
        help="Annotation file extensions to remove when image is removed.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        help="Keep exactly this many images in each image directory.",
    )
    parser.add_argument(
        "--delete-count",
        type=int,
        help="Delete exactly this many images in each image directory.",
    )
    parser.add_argument(
        "--delete-fraction",
        type=float,
        help="Delete this fraction of images in each image directory (0.0 to 1.0).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--marker-name",
        type=str,
        default=".random_destroyer_done",
        help="Marker filename written in each image directory after execute.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="random_destroyer_manifest.json",
        help="Manifest file path (JSON).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow execute even if marker exists.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete files. Without this flag it only prints a dry-run plan.",
    )
    args = parser.parse_args()

    active_modes = [
        args.target_count is not None,
        args.delete_count is not None,
        args.delete_fraction is not None,
    ]
    if sum(active_modes) != 1:
        raise ValueError("Provide exactly one of --target-count, --delete-count, or --delete-fraction.")

    if args.target_count is not None and args.target_count < 0:
        raise ValueError("--target-count must be >= 0.")
    if args.delete_count is not None and args.delete_count < 0:
        raise ValueError("--delete-count must be >= 0.")
    if args.delete_fraction is not None and not (0.0 <= args.delete_fraction <= 1.0):
        raise ValueError("--delete-fraction must be between 0.0 and 1.0.")
    return args


def find_images(image_dir: Path, image_exts: set[str]) -> list[Path]:
    files = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts]
    files.sort()
    return files


def compute_delete_count(total: int, target_count: int | None, delete_count: int | None, delete_fraction: float | None) -> int:
    if target_count is not None:
        return max(0, total - min(total, target_count))
    if delete_count is not None:
        return min(total, delete_count)
    return min(total, int(round(total * delete_fraction)))


def build_plan(
    image_dirs: list[Path],
    image_exts: set[str],
    target_count: int | None,
    delete_count: int | None,
    delete_fraction: float | None,
    seed: int,
) -> list[Plan]:
    rng = random.Random(seed)
    plans: list[Plan] = []
    for image_dir in image_dirs:
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        images = find_images(image_dir, image_exts)
        total = len(images)
        remove_n = compute_delete_count(total, target_count, delete_count, delete_fraction)
        selected = rng.sample(images, k=remove_n) if remove_n > 0 else []
        selected.sort()
        plans.append(
            Plan(
                image_dir=image_dir,
                total_images=total,
                delete_count=remove_n,
                keep_count=total - remove_n,
                selected_images=selected,
            )
        )
    return plans


def candidate_annotation_paths(
    image_path: Path,
    image_dir: Path,
    annotation_dirs: list[Path],
    annotation_exts: list[str],
) -> list[Path]:
    rel = image_path.relative_to(image_dir)
    rel_no_ext = rel.with_suffix("")
    candidates: list[Path] = []

    # Same folder as image (flat datasets / sidecar labels).
    for ext in annotation_exts:
        candidates.append(image_path.with_suffix(ext))

    # Explicit annotation directories, with mirrored relative layout and flat fallback.
    for ann_dir in annotation_dirs:
        for ext in annotation_exts:
            candidates.append((ann_dir / rel_no_ext).with_suffix(ext))
            candidates.append(ann_dir / f"{image_path.stem}{ext}")
    return candidates


def apply_plan(
    plans: list[Plan],
    annotation_dirs: list[Path],
    annotation_exts: list[str],
    marker_name: str,
    force: bool,
) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "created_at_utc": now,
        "summary": {
            "image_files_deleted": 0,
            "annotation_files_deleted": 0,
        },
        "runs": [],
    }

    for plan in plans:
        marker_path = plan.image_dir / marker_name
        if marker_path.exists() and not force:
            raise RuntimeError(
                f"Marker exists for {plan.image_dir}: {marker_path}. "
                "Use --force to run again."
            )

        run_entry = {
            "image_dir": str(plan.image_dir),
            "total_before": plan.total_images,
            "deleted_count": 0,
            "kept_after": plan.keep_count,
            "deleted_images": [],
            "deleted_annotations": [],
            "marker": str(marker_path),
        }

        deleted_annotations_seen = set()
        for img in plan.selected_images:
            if img.exists():
                img.unlink()
                run_entry["deleted_images"].append(str(img))
                run_entry["deleted_count"] += 1
                manifest["summary"]["image_files_deleted"] += 1

            for candidate in candidate_annotation_paths(
                image_path=img,
                image_dir=plan.image_dir,
                annotation_dirs=annotation_dirs,
                annotation_exts=annotation_exts,
            ):
                if candidate.exists() and str(candidate) not in deleted_annotations_seen:
                    candidate.unlink()
                    deleted_annotations_seen.add(str(candidate))
                    run_entry["deleted_annotations"].append(str(candidate))
                    manifest["summary"]["annotation_files_deleted"] += 1

        marker_payload = {
            "completed_at_utc": now,
            "deleted_count": run_entry["deleted_count"],
            "kept_after": plan.keep_count,
        }
        marker_path.write_text(json.dumps(marker_payload, indent=2) + "\n")
        manifest["runs"].append(run_entry)

    return manifest


def build_manifest_preview(plans: list[Plan]) -> dict:
    return {
        "mode": "dry_run",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "image_files_would_delete": sum(p.delete_count for p in plans),
            "directories": len(plans),
        },
        "runs": [
            {
                "image_dir": str(p.image_dir),
                "total_before": p.total_images,
                "would_delete": p.delete_count,
                "would_keep_after": p.keep_count,
            }
            for p in plans
        ],
    }


def main():
    args = parse_args()
    image_dirs = [Path(p) for p in args.image_dirs]
    annotation_dirs = [Path(p) for p in args.annotation_dirs]
    image_exts = {ext if ext.startswith(".") else f".{ext}" for ext in args.image_exts}
    annotation_exts = [ext if ext.startswith(".") else f".{ext}" for ext in args.annotation_exts]

    plans = build_plan(
        image_dirs=image_dirs,
        image_exts=image_exts,
        target_count=args.target_count,
        delete_count=args.delete_count,
        delete_fraction=args.delete_fraction,
        seed=args.seed,
    )

    for plan in plans:
        print(
            f"{plan.image_dir}: total={plan.total_images} "
            f"delete={plan.delete_count} keep={plan.keep_count}"
        )

    manifest_path = Path(args.manifest)
    if args.execute:
        manifest = apply_plan(
            plans=plans,
            annotation_dirs=annotation_dirs,
            annotation_exts=annotation_exts,
            marker_name=args.marker_name,
            force=args.force,
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print("Execute complete.")
        print(f"Manifest: {manifest_path.resolve()}")
        print(
            f"Deleted images={manifest['summary']['image_files_deleted']} "
            f"annotations={manifest['summary']['annotation_files_deleted']}"
        )
    else:
        preview = build_manifest_preview(plans)
        manifest_path.write_text(json.dumps(preview, indent=2) + "\n")
        print("Dry-run only. No files deleted.")
        print(f"Preview manifest: {manifest_path.resolve()}")
        print("Re-run with --execute to apply.")


if __name__ == "__main__":
    main()
