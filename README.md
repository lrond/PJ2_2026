# Project 2: CIFAR-10 Classification and Batch Normalization

This repository contains the code and lightweight experiment evidence for Project 2 of
Neural Network and Deep Learning. The project has two parts:

1. Train and analyze CIFAR-10 classifiers with different architectures and training choices.
2. Compare VGG-A with and without Batch Normalization, then analyze why BN helps optimization.

## Project Links

- Code: <https://github.com/lrond/PJ2_2026>
- CIFAR-10 dataset: <https://drive.google.com/file/d/1BSH50BA7XpiBP2791YqKjNeY4RMbkrGx/view?usp=drive_link>
- Trained model weights: <https://drive.google.com/file/d/1jEUilYckLD_TS6n8jpKKymsLjpulvtAT/view?usp=drive_link>

## Main Results

### Task 1: CIFAR-10 classification

Best observed custom classifier:

- Model: custom ResCNN
- Widths: `48,96,192`
- Blocks per stage: `3`
- Optimizer: AdamW
- Loss: cross entropy with label smoothing `0.1`
- Weight decay: `5e-4`
- Parameters: `2,436,346`
- Best single-seed test accuracy: `92.84%`
- Best single-seed test error: `7.16%`
- Three-seed mean test accuracy: `92.78%`

Three-seed matched family comparison:

All rows use 45 epochs, AdamW, learning rate `1e-3`, weight decay `1e-4`,
cross entropy, ReLU, and the same CIFAR-10 split protocol.

| Model family | Params | Seeds | Mean test accuracy | Val-test gap | Time/run |
| --- | ---: | ---: | ---: | ---: | ---: |
| VGG-A-Light | `0.29M` | 3 | `75.58%` | `-0.14 pp` | `1.8 min` |
| VGG-A-Dropout | `9.75M` | 3 | `84.21%` | `+0.21 pp` | `2.4 min` |
| PlainCNN | `0.65M` | 3 | `89.65%` | `+0.08 pp` | `1.8 min` |
| Conv-Stem Mixer | `0.48M` | 3 | `85.25%` | `+0.27 pp` | `2.0 min` |
| Conv-Token Transformer | `0.78M` | 3 | `85.49%` | `+0.74 pp` | `2.3 min` |
| ResCNN | `1.56M` | 3 | `91.19%` | `+0.71 pp` | `2.8 min` |

The report interprets this table as an accuracy/capacity/compute/generalization
trade-off, not an accuracy-only ranking. PlainCNN is the strongest compact
baseline, while ResCNN gives the best final accuracy for a moderate increase in
parameters and training time. VGG-A-Dropout has the largest parameter count in
this matched comparison but does not outperform the smaller convolutional
baselines, so raw capacity alone is not treated as the explanation.

Seed policy:

- Main model-family comparison: 3 seeds (`2020`, `2021`, `2022`)
- Final selected ResCNN setting: 3 seeds (`2020`, `2021`, `2022`)
- Screening ablations and hybrid regularization controls: seed `2020`

The single-seed rows are used to explain design choices, not to make stability
claims.

Additional Task 1 comparisons cover:

- filter widths and depths
- activations: ReLU, LeakyReLU, ELU
- losses: cross entropy, label smoothing, focal loss
- weight decay controls for the hybrid families
- optimizers: AdamW, SGD with momentum, RMSprop
- parameter efficiency, training time, repeated-seed stability, and validation-test gaps

### Task 2: Batch Normalization

VGG-A and VGG-A-BN were compared at learning rates `1e-3`, `2e-3`, `5e-4`,
and `1e-4`.

- Best VGG-A test accuracy: `83.80%`
- Best VGG-A-BN test accuracy: `86.96%`
- BN improved matched-learning-rate test accuracy at all four tested rates.
- The BN runs also show lower post-warm-up loss bands and higher gradient cosine
  values in the optimization mechanism analysis.

## Repository Layout

```text
.
├── data/
│   └── loaders.py                 # CIFAR-10 loading and train/validation split
├── models/
│   ├── rescnn.py                  # custom residual CNN
│   ├── plaincnn.py                # non-residual baseline
│   ├── modern.py                  # Conv-Stem Mixer and Conv-Token Transformer
│   └── vgg.py                     # VGG-A, VGG-A-BN, and VGG variants
├── reports/
│   ├── figures/                   # generated report figures
│   ├── logs/                      # lightweight run logs
│   └── results/                   # CSV/JSON summaries and loss traces
├── tests/
│   └── test_cpu_foundation.py     # CPU smoke/regression tests
├── train_cifar.py                 # Task 1 training entrypoint
├── VGG_Loss_Landscape.py          # Task 2 VGG-A/BN and loss landscape workflow
├── analyze_results.py             # summary and figure generation
├── run_full_experiments.sh        # full GPU experiment launcher
└── requirements.txt
```

## Included Artifacts

The repository provides the training/evaluation code, model definitions, CPU
regression tests, full GPU experiment launcher, experiment summaries, and report
figures used for the Project 2 analysis.

## Setup

Python 3.10+ with PyTorch and torchvision is recommended. On the GPU machine used
for these runs, PyTorch `2.8.0+cu128` and torchvision `0.23.0+cu128` were used.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run CPU-only checks:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider tests/test_cpu_foundation.py -q
```

## Data

The code uses the official CIFAR-10 dataset through `torchvision.datasets.CIFAR10`
with `download=True`. By default, data is stored under `./data`.

The training split is:

- `45,000` training images
- `5,000` validation images
- official `10,000` test images

## Reproducing Experiments

A quick synthetic smoke test that does not download CIFAR-10:

```bash
python train_cifar.py --synthetic --epochs 1 --batch-size 8 --model rescnn
python VGG_Loss_Landscape.py --synthetic --epochs 1 --batch-size 4 --learning-rates 1e-3 --mechanism-metrics
```

Full GPU experiment batch:

```bash
SWEEP_EPOCHS=45 FINAL_EPOCHS=120 VGG_EPOCHS=20 ./run_full_experiments.sh
```

The script skips completed runs when their JSON summaries already exist, so it
can be used both as the full reproduction entrypoint and as a safe resume
entrypoint.

The script runs:

- Task 1 structure, activation, loss/regularization, optimizer, and model-family comparisons
- three-seed matched family comparison for VGG-A-Light, VGG-A-Dropout, PlainCNN, Conv-Stem Mixer, Conv-Token Transformer, and ResCNN
- single-seed hybrid-family controls separating label smoothing from weight decay
- three final seeds for the selected ResCNN configuration
- Task 2 VGG-A/VGG-A-BN learning-rate sweep
- loss landscape and gradient mechanism analysis
- summary and figure generation

## Rebuilding Summaries

After experiments finish:

```bash
python analyze_results.py --reports-dir reports --device cuda:0
```
