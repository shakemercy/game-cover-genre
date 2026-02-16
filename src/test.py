import json
from pathlib import Path

import torch

from dataset import get_loaders
from models.baseline_cnn import BaselineCNN

DATA_DIR = "data/balanced_raw"
CKPT = Path("checkpoints/baseline/best.pt")
OUT_DIR = Path("outputs/baseline")

SEED = 42
BATCH = 32
IMG = 224


def save_confusion_matrix_csv(cm, classes, path):
    lines = []
    header = ["true/pred"] + list(classes)
    lines.append(",".join(header))

    for i, cls in enumerate(classes):
        row = [cls] + [str(int(x)) for x in cm[i]]
        lines.append(",".join(row))

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, _, test_loader, num_classes, classes = get_loaders(
        DATA_DIR, batch_size=BATCH, img_size=IMG, seed=SEED
    )

    if not CKPT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT}")

    ckpt = torch.load(CKPT, map_location=device)

    if "classes" in ckpt:
        classes = ckpt["classes"]
        num_classes = len(classes)

    model = BaselineCNN(num_classes, in_channels=9).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            pred = torch.argmax(logits, dim=1)

            correct += (pred == y).sum().item()
            total += y.numel()

            for t, p in zip(y.tolist(), pred.tolist()):
                cm[t][p] += 1

    acc = correct / total if total > 0 else 0.0

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    result = {
        "test_acc": acc,
        "test_correct": correct,
        "test_total": total,
        "classes": classes,
        "checkpoint": str(CKPT),
        "val_acc_in_ckpt": float(ckpt.get("val_acc", -1.0)),
    }

    (OUT_DIR / "test.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    save_confusion_matrix_csv(cm, classes, OUT_DIR / "confusion_matrix.csv")

    print(f"TEST acc: {acc:.4f} ({correct}/{total})")
    print("Saved:", OUT_DIR / "test.json")
    print("Saved:", OUT_DIR / "confusion_matrix.csv")


if __name__ == "__main__":
    main()
