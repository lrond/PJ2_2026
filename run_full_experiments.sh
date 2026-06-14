#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

REPORTS_DIR="${REPORTS_DIR:-reports}"
DEVICE="${DEVICE:-cuda:0}"
SWEEP_EPOCHS="${SWEEP_EPOCHS:-45}"
FINAL_EPOCHS="${FINAL_EPOCHS:-120}"
VGG_EPOCHS="${VGG_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
VGG_BATCH_SIZE="${VGG_BATCH_SIZE:-128}"
SEED="${SEED:-2020}"

mkdir -p "$REPORTS_DIR/logs"

run() {
  echo "[$(date '+%F %T')] $*" | tee -a "$REPORTS_DIR/logs/full_experiments_trace.log"
  "$@"
}

run python - <<'PY'
import torch, torchvision
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("cuda_available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY

# Task 1: structure sweep.
run python train_cifar.py --model rescnn --widths 24,48,96 --blocks-per-stage 2 \
  --epochs "$SWEEP_EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
  --optimizer adamw --lr 1e-3 --weight-decay 1e-4 --loss cross_entropy \
  --activation relu --seed "$SEED" --output-dir "$REPORTS_DIR"

run python train_cifar.py --model rescnn --widths 32,64,128 --blocks-per-stage 2 \
  --epochs "$SWEEP_EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
  --optimizer adamw --lr 1e-3 --weight-decay 1e-4 --loss cross_entropy \
  --activation relu --seed "$SEED" --output-dir "$REPORTS_DIR"

run python train_cifar.py --model rescnn --widths 48,96,192 --blocks-per-stage 2 \
  --epochs "$SWEEP_EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
  --optimizer adamw --lr 1e-3 --weight-decay 1e-4 --loss cross_entropy \
  --activation relu --seed "$SEED" --output-dir "$REPORTS_DIR"

run python train_cifar.py --model rescnn --widths 48,96,192 --blocks-per-stage 3 \
  --epochs "$SWEEP_EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
  --optimizer adamw --lr 1e-3 --weight-decay 1e-4 --loss cross_entropy \
  --activation relu --seed "$SEED" --output-dir "$REPORTS_DIR"

# Task 1: activation sweep.
for activation in relu leaky_relu elu; do
  run python train_cifar.py --model rescnn --widths 48,96,192 --blocks-per-stage 2 \
    --epochs "$SWEEP_EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
    --optimizer adamw --lr 1e-3 --weight-decay 1e-4 --loss cross_entropy \
    --activation "$activation" --seed "$SEED" --output-dir "$REPORTS_DIR"
done

# Task 1: loss and regularization sweep.
run python train_cifar.py --model rescnn --widths 48,96,192 --blocks-per-stage 2 \
  --epochs "$SWEEP_EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
  --optimizer adamw --lr 1e-3 --weight-decay 1e-4 --loss label_smoothing \
  --label-smoothing 0.1 --activation relu --seed "$SEED" --output-dir "$REPORTS_DIR"

run python train_cifar.py --model rescnn --widths 48,96,192 --blocks-per-stage 2 \
  --epochs "$SWEEP_EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
  --optimizer adamw --lr 1e-3 --weight-decay 1e-4 --loss focal \
  --focal-gamma 2.0 --activation relu --seed "$SEED" --output-dir "$REPORTS_DIR"

run python train_cifar.py --model rescnn --widths 48,96,192 --blocks-per-stage 2 \
  --epochs "$SWEEP_EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
  --optimizer adamw --lr 1e-3 --weight-decay 5e-4 --loss cross_entropy \
  --activation relu --seed "$SEED" --output-dir "$REPORTS_DIR"

# Task 1: optimizer sweep.
run python train_cifar.py --model rescnn --widths 48,96,192 --blocks-per-stage 2 \
  --epochs "$SWEEP_EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
  --optimizer sgd --lr 0.1 --weight-decay 5e-4 --loss cross_entropy \
  --activation relu --seed "$SEED" --output-dir "$REPORTS_DIR"

run python train_cifar.py --model rescnn --widths 48,96,192 --blocks-per-stage 2 \
  --epochs "$SWEEP_EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
  --optimizer rmsprop --lr 1e-3 --weight-decay 1e-4 --loss cross_entropy \
  --activation relu --seed "$SEED" --output-dir "$REPORTS_DIR"

# Task 1: model-family comparison.
run python train_cifar.py --model vgg_light \
  --epochs "$SWEEP_EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
  --optimizer adamw --lr 1e-3 --weight-decay 1e-4 --loss cross_entropy \
  --activation relu --seed "$SEED" --output-dir "$REPORTS_DIR"

run python train_cifar.py --model vgg_dropout \
  --epochs "$SWEEP_EPOCHS" --batch-size "$VGG_BATCH_SIZE" --device "$DEVICE" \
  --optimizer adamw --lr 1e-3 --weight-decay 1e-4 --loss cross_entropy \
  --activation relu --seed "$SEED" --output-dir "$REPORTS_DIR"

run python train_cifar.py --model plaincnn --widths 48,96,192 --blocks-per-stage 2 \
  --epochs "$SWEEP_EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
  --optimizer adamw --lr 1e-3 --weight-decay 1e-4 --loss cross_entropy \
  --activation relu --seed "$SEED" --output-dir "$REPORTS_DIR"

run python train_cifar.py --model conv_stem_mixer --widths 48,96,192 --blocks-per-stage 2 \
  --epochs "$SWEEP_EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
  --optimizer adamw --lr 1e-3 --weight-decay 1e-4 --loss cross_entropy \
  --activation relu --seed "$SEED" --output-dir "$REPORTS_DIR"

run python train_cifar.py --model conv_token_transformer --widths 48,96,192 --blocks-per-stage 2 \
  --epochs "$SWEEP_EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
  --optimizer adamw --lr 1e-3 --weight-decay 1e-4 --loss cross_entropy \
  --activation relu --seed "$SEED" --output-dir "$REPORTS_DIR"

run python train_cifar.py --model conv_stem_mixer --widths 48,96,192 --blocks-per-stage 2 \
  --epochs "$SWEEP_EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
  --optimizer adamw --lr 1e-3 --weight-decay 5e-4 --loss label_smoothing \
  --label-smoothing 0.1 --activation relu --seed "$SEED" --output-dir "$REPORTS_DIR"

run python train_cifar.py --model conv_token_transformer --widths 48,96,192 --blocks-per-stage 2 \
  --epochs "$SWEEP_EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
  --optimizer adamw --lr 1e-3 --weight-decay 5e-4 --loss label_smoothing \
  --label-smoothing 0.1 --activation relu --seed "$SEED" --output-dir "$REPORTS_DIR"

# Strong final configuration, repeated for three seeds.
for seed in 2020 2021 2022; do
  run python train_cifar.py --model rescnn --widths 48,96,192 --blocks-per-stage 3 \
    --epochs "$FINAL_EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
    --optimizer adamw --lr 1e-3 --weight-decay 5e-4 --loss label_smoothing \
    --label-smoothing 0.1 --activation relu --seed "$seed" --output-dir "$REPORTS_DIR"
done

# Task 2: VGG-A vs VGG-A-BN and BN optimization metrics.
run python VGG_Loss_Landscape.py --epochs "$VGG_EPOCHS" --batch-size "$VGG_BATCH_SIZE" \
  --learning-rates 1e-3,2e-3,5e-4,1e-4 --seed "$SEED" --device "$DEVICE" \
  --save-models --mechanism-metrics --output-dir "$REPORTS_DIR"

run python analyze_results.py --reports-dir "$REPORTS_DIR" --device "$DEVICE" \
  --batch-size "$BATCH_SIZE"
