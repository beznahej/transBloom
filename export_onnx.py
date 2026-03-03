import argparse
import importlib.util
import inspect
import json
from pathlib import Path

import torch

from model import FlowerNet


def parse_args():
    parser = argparse.ArgumentParser(description="Export FlowerNet checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--output", type=str, default="checkpoints/flowernet.onnx")
    parser.add_argument("--class-map", type=str, default="checkpoints/class_to_idx.json")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Allow dynamic batch dimension in ONNX input/output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if importlib.util.find_spec("onnx") is None:
        raise ImportError("onnx is required for export_onnx.py. Install it with: pip install onnx")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = FlowerNet()
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 3, args.img_size, args.img_size, dtype=torch.float32)

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}}

    with torch.no_grad():
        export_kwargs = {
            "export_params": True,
            "opset_version": args.opset,
            "do_constant_folding": True,
            "input_names": ["input"],
            "output_names": ["logits"],
            "dynamic_axes": dynamic_axes,
        }

        # Prefer legacy exporter path to avoid requiring onnxscript in some environments.
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            export_kwargs["dynamo"] = False

        torch.onnx.export(model, dummy_input, str(output_path), **export_kwargs)

    print(f"Exported ONNX model: {output_path.resolve()}")

    class_map_path = Path(args.class_map)
    if class_map_path.exists():
        class_map = json.loads(class_map_path.read_text())
        idx_to_class = {idx: name for name, idx in class_map.items()}
        print(f"Class mapping ({class_map_path}):")
        for idx in sorted(idx_to_class):
            print(f"  {idx}: {idx_to_class[idx]}")
    else:
        print(f"Class map not found at: {class_map_path}")

    print("Model input contract:")
    print(f"  shape: [N, 3, {args.img_size}, {args.img_size}]")
    print("  dtype: float32")
    print("  normalization: x = (x/255 - 0.5) / 0.5")
    print("  output: logits [N, 2]")


if __name__ == "__main__":
    main()
