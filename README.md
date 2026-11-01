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
 
<img width="5056" height="2656" alt="parallel coordinates" src="https://github.com/user-attachments/assets/37eb4349-298f-43be-8e32-2b6eefed3e95" />

Figure 1:W&B Parallel-coordinates Plot






<img width="5056" height="2656" alt="W B Chart 01_11_2026, 18_30_10" src="https://github.com/user-attachments/assets/2ba1ca45-029f-4770-aafc-08cbd11ff645" />

Figure 2: Validation accuracy vs. step plot.


 
<img width="5056" height="2656" alt="W B Chart 01_11_2026, 18_33_42" src="https://github.com/user-attachments/assets/80c21d83-7c26-4856-a768-90494d168699" />

Figure 3: Training accuracy plot.


<img width="5056" height="2656" alt="W B Chart 01_11_2026, 18_33_16" src="https://github.com/user-attachments/assets/8941e534-1703-484d-b15b-c80cb1ce1537" />

Figure 4: Training loss plot.


<img width="5056" height="2656" alt="W B Chart 01_11_2026, 18_34_27" src="https://github.com/user-attachments/assets/28369064-06c5-4fa9-9f06-c74e816ce75f" />

Figure 5: Validation accuracy plot.
 
<img width="5056" height="2656" alt="W B Chart 01_11_2026, 18_34_05" src="https://github.com/user-attachments/assets/8731e7f5-5573-4029-ad37-6953c7c110e7" />

Figure 5: Validation accuracy plot.



### Best configuration

_To be filled in: configuration, validation accuracy, and the command that reproduces it._

### Trained model

_To be filled in: path to the final checkpoint._

## Links

- W&B project: _to be added_
- Report PDF: _to be added_
