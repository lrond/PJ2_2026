"""
VGG-A / VGG-A-BN comparison and loss landscape utilities.
"""
import argparse
import csv
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from data.loaders import get_cifar_loaders
from models.vgg import VGG_A, VGG_A_BatchNorm
from training import (
    evaluate,
    fit,
    get_device,
    make_synthetic_loaders,
    save_history_csv,
    save_json,
    set_random_seeds,
)


def parse_learning_rates(value):
    return [float(item) for item in value.split(',')]


def flatten_losses(losses_by_epoch):
    return [loss for epoch_losses in losses_by_epoch for loss in epoch_losses]


def flatten_loss_run(run):
    array = np.asarray(run, dtype=float)
    if array.ndim == 1:
        return array
    return np.asarray(flatten_losses(run), dtype=float)


def compute_loss_stats(loss_runs):
    flattened_runs = [flatten_loss_run(run) for run in loss_runs]
    if not flattened_runs:
        return np.array([]), np.array([]), np.array([])

    min_length = min(len(run) for run in flattened_runs)
    if min_length == 0:
        return np.array([]), np.array([]), np.array([])

    aligned = np.array([run[:min_length] for run in flattened_runs])
    return aligned.min(axis=0), aligned.mean(axis=0), aligned.max(axis=0)


def compute_loss_band(loss_runs):
    lower, _, upper = compute_loss_stats(loss_runs)
    return lower, upper


def smooth(values, window):
    if window <= 1 or len(values) <= 2:
        return values
    window = min(window, len(values))
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode='edge')
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode='valid')


def plot_loss_panel(ax, stats, label, color, start, density, smooth_window):
    lower, center, upper = stats
    if not len(center):
        return
    start = min(start, len(center) - 1)
    x = np.arange(len(center))[start::density]
    lower = smooth(lower[start::density], smooth_window)
    center = smooth(center[start::density], smooth_window)
    upper = smooth(upper[start::density], smooth_window)
    ax.fill_between(x, lower, upper, color=color, alpha=0.16, label=f'{label} min-max band')
    ax.plot(x, center, color=color, linewidth=1.9, label=f'{label} mean')
    ax.plot(x, lower, color=color, linewidth=0.8, alpha=0.55)
    ax.plot(x, upper, color=color, linewidth=0.8, alpha=0.55)


def plot_loss_landscape(
    no_bn_runs,
    bn_runs,
    output_path,
    skip_steps=0,
    plot_density=1,
    zoom_start=500,
    smooth_window=50,
):
    no_bn_stats = compute_loss_stats(no_bn_runs)
    bn_stats = compute_loss_stats(bn_runs)
    plot_density = max(1, plot_density)
    zoom_start = max(skip_steps, zoom_start)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9))
    panels = [
        (axes[0], skip_steps, 'Full trajectory, log scale'),
        (axes[1], zoom_start, f'Zoom after first {zoom_start} steps'),
    ]
    for ax, start, title in panels:
        plot_loss_panel(ax, no_bn_stats, 'VGG-A', 'tab:blue', start, plot_density, smooth_window)
        plot_loss_panel(ax, bn_stats, 'VGG-A-BN', 'tab:orange', start, plot_density, smooth_window)
        ax.set_xlabel('Training step')
        ax.set_title(title)
        ax.grid(True, alpha=0.22)
    axes[0].set_ylabel('Cross entropy loss')
    axes[0].set_yscale('log')
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def load_saved_loss_runs(output_dir, model_name):
    results_dir = Path(output_dir) / 'results'
    runs = []
    for path in sorted(results_dir.glob(f'{model_name}_lr-*_losses.txt')):
        runs.append(np.loadtxt(path))
    return runs


def _loss_and_gradient(model, inputs, labels, criterion):
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    gradients = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad])
    grad_vector = torch.cat([grad.detach().reshape(-1) for grad in gradients])
    return loss.detach(), grad_vector


def collect_gradient_metrics(model, data_loader, criterion, device, step_sizes):
    model.to(device)
    model.eval()
    inputs, labels = next(iter(data_loader))
    inputs = inputs.to(device)
    labels = labels.to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    base_vector = parameters_to_vector(parameters).detach()

    base_loss, base_grad = _loss_and_gradient(model, inputs, labels, criterion)
    grad_norm = base_grad.norm().clamp_min(1e-12)
    direction = base_grad / grad_norm

    rows = []
    for step_size in step_sizes:
        vector_to_parameters(base_vector - step_size * direction, parameters)
        loss, grad = _loss_and_gradient(model, inputs, labels, criterion)
        cosine = torch.nn.functional.cosine_similarity(base_grad, grad, dim=0).item()
        rows.append({
            'step_size': float(step_size),
            'loss': float(loss.item()),
            'loss_delta': float(loss.item() - base_loss.item()),
            'gradient_cosine': cosine,
            'max_gradient_difference': float((grad - base_grad).abs().max().item()),
        })

    vector_to_parameters(base_vector, parameters)
    return rows


def save_gradient_metrics(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                'model',
                'learning_rate',
                'step_size',
                'loss',
                'loss_delta',
                'gradient_cosine',
                'max_gradient_difference',
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_losses(losses_by_epoch, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.array(flatten_losses(losses_by_epoch)), fmt='%.8f')


def configure_arguments():
    parser = argparse.ArgumentParser(description='Train VGG-A and VGG-A-BN loss landscape runs')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning-rates', type=parse_learning_rates,
                        default=parse_learning_rates('1e-3,2e-3,5e-4,1e-4'))
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--root', default='./data')
    parser.add_argument('--output-dir', default='reports')
    parser.add_argument('--n-items', type=int, default=-1)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--save-models', action='store_true')
    parser.add_argument('--skip-steps', type=int, default=0)
    parser.add_argument('--plot-density', type=int, default=1)
    parser.add_argument('--zoom-start', type=int, default=500)
    parser.add_argument('--smooth-window', type=int, default=50)
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--mechanism-metrics', action='store_true')
    return parser.parse_args()


def run_model_family(model_name, model_factory, args, train_loader, val_loader, test_loader, device):
    losses = []
    gradient_rows = []
    criterion = nn.CrossEntropyLoss()
    for lr in args.learning_rates:
        set_random_seeds(args.seed, device)
        model = model_factory()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        run_name = f'{model_name}_lr-{lr:g}_seed-{args.seed}'
        model_path = (
            Path(args.output_dir) / 'models' / f'{run_name}.pth'
            if args.save_models else None
        )

        result = fit(
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            device,
            epochs=args.epochs,
            best_model_path=model_path,
        )
        if model_path is not None and model_path.exists():
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        losses.append(result['step_losses'])
        save_losses(result['step_losses'], Path(args.output_dir) / 'results' / f'{run_name}_losses.txt')
        save_history_csv(result['history'], Path(args.output_dir) / 'results' / f'{run_name}_history.csv')

        test_metrics = evaluate(model, test_loader, criterion, device)
        save_json({
            'run_name': run_name,
            'model': model_name,
            'learning_rate': lr,
            'seed': args.seed,
            'epochs': args.epochs,
            'device': str(device),
            'total_train_seconds': sum(row['epoch_seconds'] for row in result['history']),
            'best_val_accuracy': max(row['val_accuracy'] for row in result['history']),
            'best_val_error': min(row['val_error'] for row in result['history']),
            'test_metrics': test_metrics,
            'model_path': str(model_path) if model_path is not None else None,
        }, Path(args.output_dir) / 'results' / f'{run_name}_summary.json')
        print(
            f'{run_name}: test_accuracy={test_metrics["accuracy"]:.4f} '
            f'test_error={test_metrics["error"]:.4f}'
        )

        if args.mechanism_metrics:
            rows = collect_gradient_metrics(
                model,
                val_loader,
                criterion,
                device,
                step_sizes=[0.0, 1e-3, 2e-3, 5e-3, 1e-2],
            )
            for row in rows:
                row['model'] = model_name
                row['learning_rate'] = lr
            gradient_rows.extend(rows)

    return losses, gradient_rows


def main():
    args = configure_arguments()
    if args.plot_only:
        figure_path = Path(args.output_dir) / 'figures' / 'vgg_loss_landscape.pdf'
        plot_loss_landscape(
            load_saved_loss_runs(args.output_dir, 'vgg_a'),
            load_saved_loss_runs(args.output_dir, 'vgg_a_bn'),
            figure_path,
            skip_steps=args.skip_steps,
            plot_density=args.plot_density,
            zoom_start=args.zoom_start,
            smooth_window=args.smooth_window,
        )
        print(f'loss_landscape={figure_path}')
        return

    device = get_device(args.device)
    set_random_seeds(args.seed, device)

    if args.synthetic:
        train_loader, val_loader, test_loader = make_synthetic_loaders(batch_size=args.batch_size)
    else:
        train_loader, val_loader, test_loader = get_cifar_loaders(
            root=args.root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            n_items=args.n_items,
        )

    no_bn_losses, no_bn_gradient_rows = run_model_family(
        'vgg_a',
        VGG_A,
        args,
        train_loader,
        val_loader,
        test_loader,
        device,
    )
    bn_losses, bn_gradient_rows = run_model_family(
        'vgg_a_bn',
        VGG_A_BatchNorm,
        args,
        train_loader,
        val_loader,
        test_loader,
        device,
    )

    figure_path = Path(args.output_dir) / 'figures' / 'vgg_loss_landscape.pdf'
    plot_loss_landscape(
        no_bn_losses,
        bn_losses,
        figure_path,
        skip_steps=args.skip_steps,
        plot_density=args.plot_density,
        zoom_start=args.zoom_start,
        smooth_window=args.smooth_window,
    )
    print(f'loss_landscape={figure_path}')

    if args.mechanism_metrics:
        save_gradient_metrics(
            no_bn_gradient_rows + bn_gradient_rows,
            Path(args.output_dir) / 'results' / 'vgg_gradient_mechanism_metrics.csv',
        )


if __name__ == '__main__':
    main()
