import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import FlowerNet


def pick_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested cuda, but CUDA is not available.")
        return torch.device("cuda")

    if requested == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            raise RuntimeError("Requested mps, but MPS is not available.")
        return torch.device("mps")

    return torch.device("cpu")


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, float, torch.Tensor]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item() * images.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

            flat_idx = labels * num_classes + preds
            confusion += torch.bincount(flat_idx, minlength=num_classes * num_classes).reshape(
                num_classes, num_classes
            ).cpu()

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)
    return avg_loss, avg_acc, confusion


def per_class_metrics(confusion: torch.Tensor, class_names: List[str]) -> List[Dict[str, float]]:
    metrics: List[Dict[str, float]] = []
    for idx, class_name in enumerate(class_names):
        tp = confusion[idx, idx].item()
        fp = confusion[:, idx].sum().item() - tp
        fn = confusion[idx, :].sum().item() - tp
        support = confusion[idx, :].sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics.append(
            {
                "class": class_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    return metrics


def format_confusion_matrix(confusion: torch.Tensor, class_names: List[str]) -> str:
    header = "true\\pred".ljust(15) + " ".join(name.rjust(12) for name in class_names)
    lines = [header]
    for i, name in enumerate(class_names):
        row = " ".join(str(confusion[i, j].item()).rjust(12) for j in range(len(class_names)))
        lines.append(name.ljust(15) + row)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate FlowerNet checkpoint.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--report-json", type=str, default="")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    split_dir = data_dir / args.split
    if not split_dir.exists():
        raise FileNotFoundError(f"Expected split directory: {split_dir}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    eval_tfms = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.ImageFolder(split_dir, transform=eval_tfms)
    if len(dataset.classes) != 2:
        raise ValueError(f"FlowerNet outputs 2 classes, but found {len(dataset.classes)} classes in {split_dir}.")

    pin_memory = torch.cuda.is_available()
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    device = pick_device(args.device)
    model = FlowerNet().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    criterion = nn.CrossEntropyLoss()
    loss, acc, confusion = evaluate(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
        num_classes=len(dataset.classes),
    )

    class_metrics = per_class_metrics(confusion, dataset.classes)
    macro_precision = sum(item["precision"] for item in class_metrics) / len(class_metrics)
    macro_recall = sum(item["recall"] for item in class_metrics) / len(class_metrics)
    macro_f1 = sum(item["f1"] for item in class_metrics) / len(class_metrics)

    print(f"Using device: {device}")
    print(f"Split: {args.split}")
    print(f"Checkpoint: {checkpoint_path.resolve()}")
    print(f"Samples: {len(dataset)}")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(format_confusion_matrix(confusion, dataset.classes))
    print("Per-class metrics:")
    for item in class_metrics:
        print(
            f"  {item['class']}: precision={item['precision']:.4f} "
            f"recall={item['recall']:.4f} f1={item['f1']:.4f} support={int(item['support'])}"
        )
    print(
        f"Macro avg: precision={macro_precision:.4f} "
        f"recall={macro_recall:.4f} f1={macro_f1:.4f}"
    )

    if args.report_json:
        report = {
            "device": str(device),
            "split": args.split,
            "checkpoint": str(checkpoint_path.resolve()),
            "samples": len(dataset),
            "loss": loss,
            "accuracy": acc,
            "classes": dataset.classes,
            "confusion_matrix": confusion.tolist(),
            "per_class": class_metrics,
            "macro_avg": {
                "precision": macro_precision,
                "recall": macro_recall,
                "f1": macro_f1,
            },
        }
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
