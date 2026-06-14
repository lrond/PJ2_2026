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

run_train() {
  local run_name="$1"
  shift
  local summary_path="$REPORTS_DIR/results/${run_name}.json"
  if [[ -f "$summary_path" ]]; then
    echo "[$(date '+%F %T')] skip existing $run_name" | tee -a "$REPORTS_DIR/logs/full_experiments_trace.log"
    return
  fi
  run python train_cifar.py "$@"
}

vgg_outputs_exist() {
  local lr model summary_path
  for lr in 0.001 0.002 0.0005 0.0001; do
    for model in vgg_a vgg_a_bn; do
      summary_path="$REPORTS_DIR/results/${model}_lr-${lr}_seed-${SEED}_summary.json"
      [[ -f "$summary_path" ]] || return 1
    done
  done
  [[ -f "$REPORTS_DIR/results/vgg_gradient_mechanism_metrics.csv" ]]
}

run_vgg_landscape() {
  if vgg_outputs_exist; then
    echo "[$(date '+%F %T')] skip existing VGG-A/BN landscape sweep" | tee -a "$REPORTS_DIR/logs/full_experiments_trace.log"
    return
  fi
  run python VGG_Loss_Landscape.py --epochs "$VGG_EPOCHS" --batch-size "$VGG_BATCH_SIZE" \
    --learning-rates 1e-3,2e-3,5e-4,1e-4 --seed "$SEED" --device "$DEVICE" \
    --save-models --mechanism-metrics --output-dir "$REPORTS_DIR"
}

run python - <<'PY'
import torch, torchvision
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("cuda_available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY

# Task 1: structure sweep.
run_train "rescnn_w-24-48-96_b-2_act-relu_opt-adamw_lr-0.001_wd-0.0001_loss-cross_entropy_seed-${SEED}" \
  --model rescnn --widths 24,48,96 --blocks-per-stage 2 \
  --epochs "$SWEEP_EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
  --optimizer adamw --lr 1e-3 --weight-decay 1e-4 --loss cross_entropy \
  --activation relu --seed "$SEED" --output-dir "$REPORTS_DIR"

run_train "rescnn_w-32-64-128_b-2_act-relu_opt-adamw_lr-0.001_wd-0.0001_loss-cross_entropy_seed-${SEED}" \
  --model rescnn --widths 32,64,128 --blocks-per-stage 2 \
  --epochs "$SWEEP_EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
  --optimizer adamw --lr 1e-3 --weight-decay 1e-4 --loss cross_entropy \
  --activation relu --seed "$SEED" --output-dir "$REPORTS_DIR"

run_train "rescnn_w-48-96-192_b-2_act-relu_opt-adamw_lr-0.001_wd-0.0001_loss-cross_entropy_seed-${SEED}" \
  --model rescnn --widths 48,96,192 --blocks-per-stage 2 \
  --epochs "$SWEEP_EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
  --optimizer adamw --lr 1e-3 --weight-decay 1e-4 --loss cross_entropy \
  --activation relu --seed "$SEED" --output-dir "$REPORTS_DIR"

run_train "rescnn_w-48-96-192_b-3_act-relu_opt-adamw_lr-0.001_wd-0.0001_loss-cross_entropy_seed-${SEED}" \
  --model rescnn --widths 48,96,192 --blocks-per-stage 3 \
  --epochs "$SWEEP_EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
  --optimizer adamw --lr 1e-3 --weight-decay 1e-4 --loss cross_entropy \
  --activation relu --seed "$SEED" --output-dir "$REPORTS_DIR"

# Task 1: activation sweep.
for activation in relu leaky_relu elu; do
  run_train "rescnn_w-48-96-192_b-2_act-${activation}_opt-adamw_lr-0.001_wd-0.0001_loss-cross_entropy_seed-${SEED}" \
    --model rescnn --widths 48,96,192 --blocks-per-stage 2 \
    --epochs "$SWEEP_EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
    --optimizer adamw --lr 1e-3 --weight-decay 1e-4 --loss cross_entropy \
    --activation "$activation" --seed "$SEED" --output-dir "$REPORTS_DIR"
done

# Task 1: loss and regularization sweep.
run_train "rescnn_w-48-96-192_b-2_act-relu_opt-adamw_lr-0.001_wd-0.0001_loss-label_smoothing_seed-${SEED}" \
  --model rescnn --widths 48,96,192 --blocks-per-stage 2 \
  --epochs "$SWEEP_EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
  --optimizer adamw --lr 1e-3 --weight-decay 1e-4 --loss label_smoothing \
  --label-smoothing 0.1 --activation relu --seed "$SEED" --output-dir "$REPORTS_DIR"

run_train "rescnn_w-48-96-192_b-2_act-relu_opt-adamw_lr-0.001_wd-0.0001_loss-focal_seed-${SEED}" \
  --model rescnn --widths 48,96,192 --blocks-per-stage 2 \
  --epochs "$SWEEP_EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
  --optimizer adamw --lr 1e-3 --weight-decay 1e-4 --loss focal \
  --focal-gamma 2.0 --activation relu --seed "$SEED" --output-dir "$REPORTS_DIR"

run_train "rescnn_w-48-96-192_b-2_act-relu_opt-adamw_lr-0.001_wd-0.0005_loss-cross_entropy_seed-${SEED}" \
  --model rescnn --widths 48,96,192 --blocks-per-stage 2 \
  --epochs "$SWEEP_EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
  --optimizer adamw --lr 1e-3 --weight-decay 5e-4 --loss cross_entropy \
  --activation relu --seed "$SEED" --output-dir "$REPORTS_DIR"

# Task 1: optimizer sweep.
run_train "rescnn_w-48-96-192_b-2_act-relu_opt-sgd_lr-0.1_wd-0.0005_loss-cross_entropy_seed-${SEED}" \
  --model rescnn --widths 48,96,192 --blocks-per-stage 2 \
  --epochs "$SWEEP_EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
  --optimizer sgd --lr 0.1 --weight-decay 5e-4 --loss cross_entropy \
  --activation relu --seed "$SEED" --output-dir "$REPORTS_DIR"

run_train "rescnn_w-48-96-192_b-2_act-relu_opt-rmsprop_lr-0.001_wd-0.0001_loss-cross_entropy_seed-${SEED}" \
  --model rescnn --widths 48,96,192 --blocks-per-stage 2 \
  --epochs "$SWEEP_EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
  --optimizer rmsprop --lr 1e-3 --weight-decay 1e-4 --loss cross_entropy \
  --activation relu --seed "$SEED" --output-dir "$REPORTS_DIR"

# Task 1: three-seed matched model-family comparison.
for seed in 2020 2021 2022; do
  run_train "plaincnn_w-48-96-192_b-2_act-relu_opt-adamw_lr-0.001_wd-0.0001_loss-cross_entropy_seed-${seed}" \
    --model plaincnn --widths 48,96,192 --blocks-per-stage 2 \
    --epochs "$SWEEP_EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
    --optimizer adamw --lr 1e-3 --weight-decay 1e-4 --loss cross_entropy \
    --activation relu --seed "$seed" --output-dir "$REPORTS_DIR"

  run_train "conv_stem_mixer_w-48-96-192_b-2_act-relu_opt-adamw_lr-0.001_wd-0.0001_loss-cross_entropy_seed-${seed}" \
    --model conv_stem_mixer --widths 48,96,192 --blocks-per-stage 2 \
    --epochs "$SWEEP_EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
    --optimizer adamw --lr 1e-3 --weight-decay 1e-4 --loss cross_entropy \
    --activation relu --seed "$seed" --output-dir "$REPORTS_DIR"

  run_train "conv_token_transformer_w-48-96-192_b-2_act-relu_opt-adamw_lr-0.001_wd-0.0001_loss-cross_entropy_seed-${seed}" \
    --model conv_token_transformer --widths 48,96,192 --blocks-per-stage 2 \
    --epochs "$SWEEP_EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
    --optimizer adamw --lr 1e-3 --weight-decay 1e-4 --loss cross_entropy \
    --activation relu --seed "$seed" --output-dir "$REPORTS_DIR"

  run_train "rescnn_w-48-96-192_b-2_act-relu_opt-adamw_lr-0.001_wd-0.0001_loss-cross_entropy_seed-${seed}" \
    --model rescnn --widths 48,96,192 --blocks-per-stage 2 \
    --epochs "$SWEEP_EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
    --optimizer adamw --lr 1e-3 --weight-decay 1e-4 --loss cross_entropy \
    --activation relu --seed "$seed" --output-dir "$REPORTS_DIR"

  run_train "vgg_light_w-32-64-128_b-2_act-relu_opt-adamw_lr-0.001_wd-0.0001_loss-cross_entropy_seed-${seed}" \
    --model vgg_light \
    --epochs "$SWEEP_EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
    --optimizer adamw --lr 1e-3 --weight-decay 1e-4 --loss cross_entropy \
    --activation relu --seed "$seed" --output-dir "$REPORTS_DIR"

  run_train "vgg_dropout_w-32-64-128_b-2_act-relu_opt-adamw_lr-0.001_wd-0.0001_loss-cross_entropy_seed-${seed}" \
    --model vgg_dropout \
    --epochs "$SWEEP_EPOCHS" --batch-size "$VGG_BATCH_SIZE" --device "$DEVICE" \
    --optimizer adamw --lr 1e-3 --weight-decay 1e-4 --loss cross_entropy \
    --activation relu --seed "$seed" --output-dir "$REPORTS_DIR"
done

# Task 1: hybrid regularization controls.
for model in conv_stem_mixer conv_token_transformer; do
  run_train "${model}_w-48-96-192_b-2_act-relu_opt-adamw_lr-0.001_wd-0.0005_loss-cross_entropy_seed-${SEED}" \
    --model "$model" --widths 48,96,192 --blocks-per-stage 2 \
    --epochs "$SWEEP_EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
    --optimizer adamw --lr 1e-3 --weight-decay 5e-4 --loss cross_entropy \
    --activation relu --seed "$SEED" --output-dir "$REPORTS_DIR"

  run_train "${model}_w-48-96-192_b-2_act-relu_opt-adamw_lr-0.001_wd-0.0001_loss-label_smoothing_seed-${SEED}" \
    --model "$model" --widths 48,96,192 --blocks-per-stage 2 \
    --epochs "$SWEEP_EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
    --optimizer adamw --lr 1e-3 --weight-decay 1e-4 --loss label_smoothing \
    --label-smoothing 0.1 --activation relu --seed "$SEED" --output-dir "$REPORTS_DIR"

  run_train "${model}_w-48-96-192_b-2_act-relu_opt-adamw_lr-0.001_wd-0.0005_loss-label_smoothing_seed-${SEED}" \
    --model "$model" --widths 48,96,192 --blocks-per-stage 2 \
    --epochs "$SWEEP_EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
    --optimizer adamw --lr 1e-3 --weight-decay 5e-4 --loss label_smoothing \
    --label-smoothing 0.1 --activation relu --seed "$SEED" --output-dir "$REPORTS_DIR"
done

# Strong final configuration, repeated for three seeds.
for seed in 2020 2021 2022; do
  run_train "rescnn_w-48-96-192_b-3_act-relu_opt-adamw_lr-0.001_wd-0.0005_loss-label_smoothing_seed-${seed}" \
    --model rescnn --widths 48,96,192 --blocks-per-stage 3 \
    --epochs "$FINAL_EPOCHS" --batch-size "$BATCH_SIZE" --device "$DEVICE" \
    --optimizer adamw --lr 1e-3 --weight-decay 5e-4 --loss label_smoothing \
    --label-smoothing 0.1 --activation relu --seed "$seed" --output-dir "$REPORTS_DIR"
done

# Task 2: VGG-A vs VGG-A-BN and BN optimization metrics.
run_vgg_landscape

run python analyze_results.py --reports-dir "$REPORTS_DIR" --device "$DEVICE" \
  --batch-size "$BATCH_SIZE"
