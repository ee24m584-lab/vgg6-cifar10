"""Evaluation: loss/accuracy over a loader, or a saved checkpoint from the CLI."""
import argparse

import torch
import torch.nn as nn

from data import get_cifar10
from models import build_vgg6


@torch.no_grad()
def evaluate(model, loader, device):
    """Returns (mean loss, top-1 accuracy %)."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss += criterion(out, y).item()
        correct += out.argmax(1).eq(y).sum().item()
        total += y.size(0)
    return loss / total, 100.0 * correct / total


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate a saved VGG6 checkpoint on the CIFAR-10 test set")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", default="./data")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    model = build_vgg6(cfg["activation"], bool(cfg["batch_norm"])).to(device)
    model.load_state_dict(ckpt["model"])
    _, test_loader = get_cifar10(256, args.data_dir)
    loss, acc = evaluate(model, test_loader, device)
    print(f"test loss {loss:.4f} | test top-1 acc {acc:.2f}%")
