#!/usr/bin/env python3
"""Generate an Open Images image_list.txt for non-flower downloads.

Output format matches downloader.py input:
  train/<image_id>
  validation/<image_id>
"""

import argparse
import csv
import io
import random
import re
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


URL_CLASS_DESCRIPTIONS = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv"
URL_TRAIN_IMAGE_IDS = (
    "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv"
)
URL_VALIDATION_IMAGE_IDS = (
    "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv"
)
URL_TRAIN_IMAGE_LABELS = (
    "https://storage.googleapis.com/openimages/v5/train-annotations-human-imagelabels-boxable.csv"
)
URL_VALIDATION_IMAGE_LABELS = (
    "https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels-boxable.csv"
)


DEFAULT_EXCLUDE_REGEX = (
    r"flower|rose|tulip|orchid|sunflower|blossom|bouquet|lily|daisy|lavender|"
    r"hibiscus|lotus|marigold|violet|poppy"
)


def stream_csv_rows(url: str) -> Iterable[List[str]]:
    with urllib.request.urlopen(url, timeout=60) as response:
        wrapper = io.TextIOWrapper(response, encoding="utf-8")
        reader = csv.reader(wrapper)
        for row in reader:
            yield row


def load_excluded_label_mids(exclude_regex: str) -> Set[str]:
    pattern = re.compile(exclude_regex, flags=re.IGNORECASE)
    excluded: Set[str] = set()
    total = 0
    for row in stream_csv_rows(URL_CLASS_DESCRIPTIONS):
        if len(row) < 2:
            continue
        total += 1
        mid, name = row[0], row[1]
        if pattern.search(name):
            excluded.add(mid)
    print(
        f"Scanned {total} class descriptions; excluding {len(excluded)} class labels by regex.",
        file=sys.stderr,
    )
    return excluded


def load_flower_tagged_image_ids(url: str, excluded_mids: Set[str], split: str) -> Set[str]:
    excluded_image_ids: Set[str] = set()
    rows = 0
    for idx, row in enumerate(stream_csv_rows(url)):
        if idx == 0:
            # header
            continue
        rows += 1
        if len(row) < 4:
            continue
        image_id, _, label_mid, confidence = row[0], row[1], row[2], row[3]
        if confidence == "1" and label_mid in excluded_mids:
            excluded_image_ids.add(image_id)
    print(
        f"{split}: scanned {rows} image-level label rows; "
        f"excluded {len(excluded_image_ids)} image IDs with flower labels.",
        file=sys.stderr,
    )
    return excluded_image_ids


def load_allowed_image_ids(url: str, blocked_ids: Set[str], split: str) -> List[str]:
    allowed: List[str] = []
    rows = 0
    for idx, row in enumerate(stream_csv_rows(url)):
        if idx == 0:
            # header
            continue
        if not row:
            continue
        image_id = row[0]
        rows += 1
        if image_id not in blocked_ids:
            allowed.append(image_id)
    print(
        f"{split}: scanned {rows} image IDs; {len(allowed)} remain after exclusion.",
        file=sys.stderr,
    )
    return allowed


def sample_ids(ids: List[str], limit: int, rng: random.Random) -> List[str]:
    if limit <= 0 or len(ids) <= limit:
        return sorted(ids)
    return sorted(rng.sample(ids, k=limit))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate split/image_id list for non-flower Open Images downloads."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/openimages_nonflower_image_list.txt",
        help="Output path for downloader.py image list file.",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=5000,
        help="Max number of train image IDs to include (<=0 means all).",
    )
    parser.add_argument(
        "--validation-limit",
        type=int,
        default=2000,
        help="Max number of validation image IDs to include (<=0 means all).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-regex",
        type=str,
        default=DEFAULT_EXCLUDE_REGEX,
        help="Regex over class description names used to exclude flower-related labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    excluded_mids = load_excluded_label_mids(args.exclude_regex)

    blocked_train = load_flower_tagged_image_ids(
        URL_TRAIN_IMAGE_LABELS, excluded_mids, split="train"
    )
    blocked_validation = load_flower_tagged_image_ids(
        URL_VALIDATION_IMAGE_LABELS, excluded_mids, split="validation"
    )

    train_ids = load_allowed_image_ids(
        URL_TRAIN_IMAGE_IDS, blocked_train, split="train"
    )
    validation_ids = load_allowed_image_ids(
        URL_VALIDATION_IMAGE_IDS, blocked_validation, split="validation"
    )

    train_selected = sample_ids(train_ids, args.train_limit, rng)
    validation_selected = sample_ids(validation_ids, args.validation_limit, rng)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for image_id in train_selected:
            f.write(f"train/{image_id}\n")
        for image_id in validation_selected:
            f.write(f"validation/{image_id}\n")

    print(
        f"Wrote {len(train_selected) + len(validation_selected)} IDs to {output_path.resolve()}",
        file=sys.stderr,
    )
    print(
        f"  train={len(train_selected)} validation={len(validation_selected)}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

