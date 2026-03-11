import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SUPPORTED_MODEL_TYPES, FlowerNet, make_checkpoint_payload


# PSEUDOCODE (high level):
# - Parse training/config args.
# - Build datasets/dataloaders and model.
# - Run epoch loop: train step -> validation step.
# - Save best checkpoint and final checkpoint.

def evaluate(model, dataloader, criterion, device):
    # PSEUDOCODE:
    # switch to eval mode
    # for each batch:
    #   forward pass only (no grads)
    #   accumulate weighted loss + correct predictions
    # return average loss/accuracy
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def main():
    # PSEUDOCODE:
    # declare CLI with architecture + optimization controls
    parser = argparse.ArgumentParser(description="Train FlowerNet")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--optimizer",
        choices=["auto", "adam", "adamw"],
        default="auto",
        help="Optimizer choice. 'auto' uses Adam for cnn and AdamW for transformer.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay. If omitted, transformer+AdamW defaults to 0.05 and others to 0.0.",
    )
    parser.add_argument(
        "--scheduler",
        choices=["none", "cosine"],
        default="none",
        help="Learning-rate scheduler.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="Number of linear warmup epochs (used with --scheduler cosine).",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing for cross-entropy loss.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=0.0,
        help="Clip gradient norm to this value (<=0 disables clipping).",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--model-type", choices=SUPPORTED_MODEL_TYPES, default="cnn")
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            "Expected dataset directories at "
            f"{train_dir} and {val_dir}. "
            "Use ImageFolder layout: data/train/<class_name> and data/val/<class_name>."
        )

    # PSEUDOCODE:
    # - Transformer gets stronger augmentation.
    # - CNN keeps lightweight augmentation.
    # - Validation remains deterministic.
    if args.model_type == "transformer":
        train_tfms = transforms.Compose(
            [
                transforms.RandomResizedCrop(args.img_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)],
                    p=0.5,
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    else:
        train_tfms = transforms.Compose(
            [
                transforms.Resize((args.img_size, args.img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tfms)

    if len(train_ds.classes) != 2:
        raise ValueError(
            f"FlowerNet outputs 2 classes, but found {len(train_ds.classes)} classes in {train_dir}."
        )

    # PSEUDOCODE:
    # build train/val dataloaders
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # PSEUDOCODE:
    # instantiate model with the selected backbone + hyperparameters
    transformer_config = {
        "patch_size": args.patch_size,
        "embed_dim": args.embed_dim,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "mlp_ratio": args.mlp_ratio,
        "dropout": args.dropout,
    }
    model = FlowerNet(
        model_type=args.model_type,
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    ).to(device)

    # PSEUDOCODE:
    # configure objective + optimizer:
    # - auto => Adam for CNN, AdamW for Transformer
    # - optional weight decay override
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    resolved_optimizer = args.optimizer
    if resolved_optimizer == "auto":
        resolved_optimizer = "adamw" if args.model_type == "transformer" else "adam"

    if args.weight_decay is None:
        resolved_weight_decay = 0.05 if (args.model_type == "transformer" and resolved_optimizer == "adamw") else 0.0
    else:
        resolved_weight_decay = args.weight_decay

    if resolved_optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=resolved_weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=resolved_weight_decay)

    # PSEUDOCODE:
    # optional cosine schedule with linear warmup.
    # this keeps early training stable and decays LR later.
    scheduler = None
    if args.scheduler == "cosine":
        warmup_epochs = max(0, min(args.warmup_epochs, max(0, args.epochs - 1)))

        def lr_lambda(epoch_idx: int) -> float:
            if warmup_epochs > 0 and epoch_idx < warmup_epochs:
                return float(epoch_idx + 1) / float(warmup_epochs)
            denom = max(1, args.epochs - warmup_epochs)
            progress = float(epoch_idx - warmup_epochs) / float(denom)
            progress = max(0.0, min(progress, 1.0))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "class_to_idx.json").open("w") as f:
        json.dump(train_ds.class_to_idx, f, indent=2)

    best_val_acc = 0.0
    print(f"Using device: {device}")
    print(f"Model type: {args.model_type} | img_size={args.img_size}")
    print(
        f"Optimizer: {resolved_optimizer} | lr={args.lr} | weight_decay={resolved_weight_decay} | "
        f"scheduler={args.scheduler} | warmup_epochs={args.warmup_epochs}"
    )
    print(
        f"Loss: cross_entropy(label_smoothing={args.label_smoothing}) | "
        f"grad_clip_norm={args.grad_clip_norm if args.grad_clip_norm > 0 else 'disabled'}"
    )
    if args.model_type == "transformer":
        print(
            "Transformer config: "
            f"patch_size={args.patch_size} embed_dim={args.embed_dim} "
            f"depth={args.depth} num_heads={args.num_heads} "
            f"mlp_ratio={args.mlp_ratio} dropout={args.dropout}"
        )
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    print(f"Classes: {train_ds.classes}")

    # PSEUDOCODE:
    # for each epoch:
    #   train over mini-batches
    #   evaluate on validation set
    #   track and save best checkpoint
    #   step scheduler
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_samples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # PSEUDOCODE:
            # zero grads -> forward -> loss -> backward -> optional grad clip -> optimizer step
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            running_correct += (preds == labels).sum().item()
            running_samples += images.size(0)

        train_loss = running_loss / running_samples
        train_acc = running_correct / running_samples
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={current_lr:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save the strongest validation model so far.
            best_payload = make_checkpoint_payload(
                model=model,
                model_type=args.model_type,
                img_size=args.img_size,
                transformer_config=transformer_config if args.model_type == "transformer" else None,
                extra={
                    "class_to_idx": train_ds.class_to_idx,
                    "best_val_acc": best_val_acc,
                    "best_epoch": epoch,
                },
            )
            torch.save(best_payload, output_dir / "best_model.pt")

        if scheduler is not None:
            scheduler.step()

    # Save final weights, even if they are not the best checkpoint.
    last_payload = make_checkpoint_payload(
        model=model,
        model_type=args.model_type,
        img_size=args.img_size,
        transformer_config=transformer_config if args.model_type == "transformer" else None,
        extra={
            "class_to_idx": train_ds.class_to_idx,
            "best_val_acc": best_val_acc,
            "epochs": args.epochs,
        },
    )
    torch.save(last_payload, output_dir / "last_model.pt")
    print(f"Best val acc: {best_val_acc:.4f}")
    print(f"Saved checkpoints to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
