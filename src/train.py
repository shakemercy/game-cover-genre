import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import get_loaders
from models.baseline_cnn import BaselineCNN

DATA_DIR = "data/balanced_raw"
OUT_DIR = Path("outputs/baseline")
CKPT = Path("checkpoints/baseline/best.pt")

SEED = 42
EPOCHS = 10
BATCH = 32
LR = 1e-3
IMG = 224


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, test_loader, num_classes, classes = get_loaders(
        DATA_DIR, batch_size=BATCH, img_size=IMG, seed=SEED
    )

    x, y = next(iter(train_loader))
    print(x.shape, y.shape)

    model = BaselineCNN(num_classes, in_channels=9).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=LR)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT.parent.mkdir(parents=True, exist_ok=True)

    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best = -1.0

    for epoch in range(EPOCHS):
        model.train()
        tl = 0.0
        tc = 0
        tn = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            tl += float(loss.item())
            pred = torch.argmax(logits, dim=1)
            tc += (pred == y).sum().item()
            tn += y.numel()

        model.eval()
        vl = 0.0
        vc = 0
        vn = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                loss = loss_fn(logits, y)

                vl += float(loss.item())
                pred = torch.argmax(logits, dim=1)
                vc += (pred == y).sum().item()
                vn += y.numel()

        train_loss = tl / max(1, len(train_loader))
        val_loss = vl / max(1, len(val_loader))
        train_acc = tc / max(1, tn)
        val_acc = vc / max(1, vn)

        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)
        hist["train_acc"].append(train_acc)
        hist["val_acc"].append(val_acc)

        print(epoch + 1, EPOCHS, train_loss, train_acc, val_loss, val_acc)

        if val_acc > best:
            best = val_acc
            torch.save({"model": model.state_dict(), "classes": classes, "val_acc": best}, CKPT)

    plt.figure()
    plt.plot(hist["train_loss"], label="train")
    plt.plot(hist["val_loss"], label="val")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "loss.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(hist["train_acc"], label="train")
    plt.plot(hist["val_acc"], label="val")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "acc.png", dpi=160)
    plt.close()

    (OUT_DIR / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
