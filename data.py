"""CIFAR-10 data loading, normalisation and augmentation."""
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


class Cutout:
    """Zero out `n_holes` random square patches of side `length` (on a CHW tensor)."""

    def __init__(self, n_holes: int = 1, length: int = 16):
        self.n_holes, self.length = n_holes, length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape
        mask = np.ones((h, w), np.float32)
        for _ in range(self.n_holes):
            y, x = np.random.randint(h), np.random.randint(w)
            y1, y2 = np.clip([y - self.length // 2, y + self.length // 2], 0, h)
            x1, x2 = np.clip([x - self.length // 2, x + self.length // 2], 0, w)
            mask[y1:y2, x1:x2] = 0.0
        return img * torch.from_numpy(mask).expand_as(img)


def build_transforms(cutout: bool = True):
    train_tf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]
    if cutout:
        train_tf.append(Cutout(n_holes=1, length=16))
    test_tf = [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
    return transforms.Compose(train_tf), transforms.Compose(test_tf)


def _seed_worker(worker_id: int):
    # numpy RNG (used by Cutout) must be re-seeded per worker
    np.random.seed(torch.initial_seed() % 2**32)


def get_cifar10(batch_size: int, root: str = "./data", num_workers: int = 4,
                cutout: bool = True, seed: int = 42):
    train_tf, test_tf = build_transforms(cutout)
    train_set = datasets.CIFAR10(root, train=True, transform=train_tf, download=True)
    test_set = datasets.CIFAR10(root, train=False, transform=test_tf, download=True)
    g = torch.Generator().manual_seed(seed)  # reproducible shuffling order
    persistent = num_workers > 0
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, generator=g,
                              worker_init_fn=_seed_worker, persistent_workers=persistent)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, persistent_workers=persistent)
    return train_loader, test_loader
