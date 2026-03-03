import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def preprocess_image(image_path: Path, img_size: int) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr.astype(np.float32)


def load_class_names(class_map_path: Path) -> List[str]:
    if not class_map_path.exists():
        return []
    class_to_idx: Dict[str, int] = json.loads(class_map_path.read_text())
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    return [idx_to_class[i] for i in sorted(idx_to_class)]


def run_onnx_inference(
    model_path: Path,
    input_tensor: np.ndarray,
    providers: List[str],
) -> np.ndarray:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required for local_predict.py. Install it with: pip install onnxruntime"
        ) from exc

    session = ort.InferenceSession(str(model_path), providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    logits = session.run([output_name], {input_name: input_tensor})[0]
    return logits[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Run ONNX inference for one image.")
    parser.add_argument("--model", type=str, default="checkpoints/flowernet.onnx")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--class-map", type=str, default="checkpoints/class_to_idx.json")
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument(
        "--provider",
        choices=["auto", "cpu"],
        default="auto",
        help="ONNX Runtime provider preference for local inference.",
    )
    parser.add_argument("--json-out", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model)
    image_path = Path(args.image)
    class_map_path = Path(args.class_map)

    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    providers = ["CPUExecutionProvider"]
    if args.provider == "auto":
        # Keep local script predictable; CPU works everywhere.
        providers = ["CPUExecutionProvider"]

    input_tensor = preprocess_image(image_path, args.img_size)
    logits = run_onnx_inference(model_path, input_tensor, providers=providers)
    probs = softmax(logits)
    pred_idx = int(np.argmax(probs))

    class_names = load_class_names(class_map_path)
    if class_names and pred_idx < len(class_names):
        pred_label = class_names[pred_idx]
    else:
        pred_label = str(pred_idx)

    print(f"Image: {image_path.resolve()}")
    print(f"Model: {model_path.resolve()}")
    print(f"Predicted class: {pred_label} (idx={pred_idx})")
    print("Probabilities:")
    for idx, prob in enumerate(probs.tolist()):
        label = class_names[idx] if idx < len(class_names) else str(idx)
        print(f"  {idx} ({label}): {prob:.6f}")

    if args.json_out:
        out = {
            "image": str(image_path.resolve()),
            "model": str(model_path.resolve()),
            "pred_index": pred_idx,
            "pred_label": pred_label,
            "probabilities": probs.tolist(),
            "class_names": class_names,
        }
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2) + "\n")
        print(f"Wrote prediction JSON: {out_path.resolve()}")


if __name__ == "__main__":
    main()
