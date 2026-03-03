import pickle
from pathlib import Path
from PIL import Image

root = Path("data/cifar100")
out = Path("data")
flower_names = {"orchid", "poppy", "rose", "sunflower", "tulip"}

meta = pickle.load(open(root / "meta", "rb"), encoding="bytes")
fine_names = [x.decode() for x in meta[b"fine_label_names"]]
flower_ids = {fine_names.index(n) for n in flower_names}

for split_name, src_file in [("train", "train"), ("val", "test")]:
    obj = pickle.load(open(root / src_file, "rb"), encoding="bytes")
    images = obj[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = obj[b"fine_labels"]

    dst = out / split_name / "non_flower"
    dst.mkdir(parents=True, exist_ok=True)

    kept = 0
    for i, (img, label) in enumerate(zip(images, labels)):
        if label in flower_ids:
            continue
        Image.fromarray(img).save(dst / f"{src_file}_{i:05d}.png")
        kept += 1

    print(f"{split_name}: kept {kept} non-flower images")