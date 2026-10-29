"""Seeding and optimizer factory."""
import random

import numpy as np
import torch
import torch.optim as optim

OPTIMIZERS = ["sgd", "nesterov", "adam", "adagrad", "rmsprop", "nadam"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(name: str, params, lr: float, momentum: float = 0.9,
                    weight_decay: float = 0.0):
    """`momentum` applies to nesterov and rmsprop; plain 'sgd' uses none by definition."""
    if name == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "nesterov":
        return optim.SGD(params, lr=lr, momentum=momentum, nesterov=True,
                         weight_decay=weight_decay)
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adagrad":
        return optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "nadam":
        return optim.NAdam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer '{name}', choose from {OPTIMIZERS}")
