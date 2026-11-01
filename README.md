# VGG6 on CIFAR-10 (CS6886W, Assignment 1)

Trains a small VGG-style network (VGG6) on CIFAR-10 while varying the activation function,
optimizer, batch size, epochs and learning rate. Experiments are tracked and swept with
[Weights & Biases](https://wandb.ai).

> Status: work in progress. Results, plots and the best configuration are filled in after the sweeps.

## Repository layout

```
models/vgg6.py     VGG6 definition (conv 64-64-M-128-128-M, global avg-pool, linear), configurable activation / BatchNorm
data.py            CIFAR-10 loaders, normalisation, random-crop / flip / Cutout augmentation
utils.py           seeding and optimizer factory (sgd, nesterov, adam, adagrad, rmsprop, nadam)
train.py           training entry point; logs to W&B, usable standalone or as a sweep target
evaluate.py        evaluation loop (also used for the final model)
configs/sweep.yaml W&B sweep configuration(s)
requirements.txt   pinned dependencies
checkpoints/       best checkpoint per run (git-ignored except the final model)
```

## Environment

| Item | Version |
|---|---|
| Python | 3.14 |
| torch / torchvision | 2.14.1+cu126 / 0.29.1+cu126 |
| wandb | 0.31.0 |
| numpy / pillow | 2.5.4 / 13.0.0 |
| GPU | NVIDIA RTX 2000 Ada (laptop), CUDA 12.6 |

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
pip install wandb==0.31.0 numpy==2.5.4 pillow==13.0.0
wandb login
```

## Reproducibility

- **Seed:** 42 by default (`--seed`). It seeds `random`, `numpy` and `torch` (CPU and CUDA), shuffling uses a seeded `torch.Generator`, and DataLoader workers re-seed numpy for Cutout.
- cuDNN is set to deterministic mode with benchmark off. Small run-to-run differences can still occur across different GPUs or library versions.
- Data: CIFAR-10 train (50,000) for training; the 10,000-image test set is used as the validation set (as in the assignment template). Normalisation uses the standard CIFAR-10 mean/std.
- Augmentation (train only): random crop (pad 4), horizontal flip, Cutout (1 hole, 16 px).
- `model.train()` is re-entered at the start of every epoch so BatchNorm and any train-time behaviour stay active after the first evaluation.
- Reported train accuracy is the running accuracy over the augmented training batches.
- Loader speed: `num_workers=4` was fastest on the development machine (3 epochs, relu/adam/bs64: 211 s with 0 workers, 66 s with 4, 94 s with 8, including startup), so it is the default.

## Usage

Single run:

```bash
python train.py --activation relu --optimizer adam --lr 1e-3 --batch_size 64 --epochs 30
```

Useful flags: `--activation {relu,sigmoid,tanh,silu,gelu}`, `--optimizer {sgd,nesterov,adam,adagrad,rmsprop,nadam}`,
`--lr`, `--batch_size`, `--epochs`, `--weight_decay`, `--momentum`, `--batch_norm {0,1}`, `--cutout {0,1}`,
`--seed`, `--num_workers`, `--no_wandb` (disable W&B logging).

W&B sweep:

```bash
wandb sweep configs/sweep.yaml
wandb agent <entity>/cs6886w-a1-vgg6/<sweep_id>
```

Evaluate a saved checkpoint:

```bash
python evaluate.py --checkpoint checkpoints/<run>.pt
```

## Results
 
<img width="475" height="250" alt="image" src="https://github.com/user-attachments/assets/47f7b5df-aac1-43b4-92e8-8ae98dd5d437" />

Figure 1. W&B parallel-coordinates plot: which configuration achieves which accuracy.<img width="468" height="36" alt="image" src="https://github.com/user-attachments/assets/c3934217-9d90-4ff8-a31f-e2ea08f584c5" />





Figure 2. W&B scatter plot of validation accuracy across runs.
<img width="476" height="288" alt="image" src="https://github.com/user-attachments/assets/a0385a6d-0a98-4e0b-bc00-011c33aff4f2" />

 
Figure 3. Training loss (W&B).
<img width="476" height="288" alt="image" src="https://github.com/user-attachments/assets/33edcf33-7a6a-4b05-b403-1bd3275fedf0" />

 
Figure 4. Training accuracy (W&B).
<img width="476" height="288" alt="image" src="https://github.com/user-attachments/assets/13c4fe2c-d984-4e48-ada3-9fe4e89bbb27" />

 
Figure 5. Validation loss (W&B).
<img width="476" height="288" alt="image" src="https://github.com/user-attachments/assets/5a9173e2-82b0-4c03-b743-1050c0f4accb" />

 
Figure 6. Validation accuracy (W&B).
<img width="476" height="288" alt="image" src="https://github.com/user-attachments/assets/334c6739-242e-4616-92cb-134250ab86e9" />




### Best configuration

_To be filled in: configuration, validation accuracy, and the command that reproduces it._

### Trained model

_To be filled in: path to the final checkpoint._

## Links

- W&B project: _to be added_
- Report PDF: _to be added_
