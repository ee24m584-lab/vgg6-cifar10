"""Train VGG6 on CIFAR-10 with a configurable setup; logs to Weights & Biases.

Works standalone and as the target of a W&B sweep (sweep args arrive as CLI flags).
"""
import argparse
import os

import torch
import torch.nn as nn
import wandb

from data import get_cifar10
from evaluate import evaluate
from models import ACTIVATIONS, build_vgg6
from utils import OPTIMIZERS, build_optimizer, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="VGG6 / CIFAR-10 training")
    p.add_argument("--activation", default="relu", choices=list(ACTIVATIONS))
    p.add_argument("--optimizer", default="adam", choices=OPTIMIZERS)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_norm", type=int, default=1, choices=[0, 1])
    p.add_argument("--cutout", type=int, default=1, choices=[0, 1])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--save_dir", default="./checkpoints")
    p.add_argument("--project", default="cs6886w-a1-vgg6")
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()  # re-enter train mode every epoch (evaluate() switches to eval)
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * y.size(0)
        correct += out.argmax(1).eq(y).sum().item()
        total += y.size(0)
    return loss_sum / total, 100.0 * correct / total


def run_name(c):
    return (f"act_{c['activation']}_bs_{c['batch_size']}_opt_{c['optimizer']}_lr_{c['lr']}"
            f"_ep_{c['epochs']}_wd_{c['weight_decay']}_bn_{bool(c['batch_norm'])}")


def main():
    args = parse_args()
    wandb.init(project=args.project, config=vars(args), name=run_name(vars(args)),
               mode="disabled" if args.no_wandb else "online")
    cfg = dict(wandb.config)  # in a sweep this holds the sweep-chosen values

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_cifar10(cfg["batch_size"], cfg["data_dir"],
                                            cfg["num_workers"], bool(cfg["cutout"]), cfg["seed"])
    model = build_vgg6(cfg["activation"], bool(cfg["batch_norm"])).to(device)
    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), cfg["lr"],
                                cfg["momentum"], cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    os.makedirs(cfg["save_dir"], exist_ok=True)
    ckpt_path = os.path.join(cfg["save_dir"], run_name(cfg) + ".pt")
    for epoch in range(cfg["epochs"]):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, test_loader, device)
        wandb.log({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                   "val_loss": val_loss, "val_acc": val_acc})
        print(f"epoch {epoch:3d} | train {tr_loss:.4f}/{tr_acc:.2f} | val {val_loss:.4f}/{val_acc:.2f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "config": cfg, "val_acc": val_acc}, ckpt_path)

    wandb.summary["best_val_acc"] = best_acc
    print(f"best val acc: {best_acc:.2f}%  (checkpoint: {ckpt_path})")
    wandb.finish()


if __name__ == "__main__":
    main()
