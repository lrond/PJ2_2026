# Project 2: CIFAR-10 Classification and Batch Normalization

This repository contains the code and lightweight experiment evidence for Project 2 of
Neural Network and Deep Learning. The project has two parts:

1. Train and analyze CIFAR-10 classifiers with different architectures and training choices.
2. Compare VGG-A with and without Batch Normalization, then analyze why BN helps optimization.

## Main Results

### Task 1: CIFAR-10 classification

Best custom classifier:

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

Model families compared:

- VGG-A-Light
- VGG-A-Dropout
- PlainCNN
- Conv-Stem Mixer
- Conv-Token Transformer
- ResCNN

The experiments also cover different filter widths/depths, activations, loss functions,
regularization settings, and optimizers.

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

The script runs:

- Task 1 structure, activation, loss/regularization, optimizer, and model-family comparisons
- three final seeds for the selected ResCNN configuration
- Task 2 VGG-A/VGG-A-BN learning-rate sweep
- loss landscape and gradient mechanism analysis
- summary and figure generation

## Rebuilding Summaries

After experiments finish:

```bash
python analyze_results.py --reports-dir reports --device cuda:0
```
