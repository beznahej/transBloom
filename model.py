from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


SUPPORTED_MODEL_TYPES = ("cnn", "transformer")


def normalize_model_type(model_type: str) -> str:
    value = model_type.strip().lower()
    if value not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model_type={model_type!r}. "
            f"Expected one of: {', '.join(SUPPORTED_MODEL_TYPES)}"
        )
    return value


class FlowerNetCnn(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class FlowerNetTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        num_classes: int = 2,
        patch_size: int = 4,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size={img_size} must be divisible by patch_size={patch_size}")

        grid = img_size // patch_size
        num_patches = grid * grid
        ff_dim = int(embed_dim * mlp_ratio)

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.embed_dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.embed_dropout(x + self.pos_embed)
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)


class FlowerNet(nn.Module):
    def __init__(self, model_type: str = "cnn", num_classes: int = 2, img_size: int = 32):
        super().__init__()
        self.model_type = normalize_model_type(model_type)
        self.img_size = img_size

        if self.model_type == "cnn":
            self.model = FlowerNetCnn(num_classes=num_classes)
        else:
            self.model = FlowerNetTransformer(
                img_size=img_size,
                num_classes=num_classes,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def make_checkpoint_payload(
    model: nn.Module,
    model_type: str,
    img_size: int,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "state_dict": model.state_dict(),
        "model_type": normalize_model_type(model_type),
        "img_size": int(img_size),
    }
    if extra:
        payload.update(extra)
    return payload


def unpack_checkpoint(checkpoint_obj: Any) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj:
        state_dict = checkpoint_obj["state_dict"]
        metadata = {k: v for k, v in checkpoint_obj.items() if k != "state_dict"}
        return state_dict, metadata

    # Backward compatibility with old checkpoints that stored only the state_dict.
    return checkpoint_obj, {}


def load_model_state_compat(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        pass

    model_keys = list(model.state_dict().keys())
    state_keys = list(state_dict.keys())

    model_uses_prefix = any(key.startswith("model.") for key in model_keys)
    state_uses_prefix = any(key.startswith("model.") for key in state_keys)

    if model_uses_prefix and not state_uses_prefix:
        remapped = {f"model.{k}": v for k, v in state_dict.items()}
        model.load_state_dict(remapped)
        return

    if state_uses_prefix and not model_uses_prefix:
        remapped = {
            (k[len("model.") :] if k.startswith("model.") else k): v for k, v in state_dict.items()
        }
        model.load_state_dict(remapped)
        return

    # Raise original shape/key mismatch details if remapping did not apply.
    model.load_state_dict(state_dict)
