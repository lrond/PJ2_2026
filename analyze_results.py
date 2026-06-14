"""
Summarize experiment outputs and generate report figures.
"""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.loaders import get_cifar_loaders
from train_cifar import build_criterion, build_model
from training import ensure_dir, evaluate, get_device


CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]


def load_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def collect_task1_summaries(results_dir):
    rows = []
    for path in sorted(Path(results_dir).glob('*.json')):
        if path.name.endswith('_summary.json'):
            continue
        data = load_json(path)
        args = data.get('args', {})
        test = data.get('test_metrics', {})
        rows.append({
            'run_name': data.get('run_name', path.stem),
            'model': args.get('model', ''),
            'widths': ','.join(str(item) for item in args.get('widths', [])),
            'blocks_per_stage': args.get('blocks_per_stage', ''),
            'activation': args.get('activation', ''),
            'optimizer': args.get('optimizer', ''),
            'lr': args.get('lr', ''),
            'weight_decay': args.get('weight_decay', ''),
            'loss': args.get('loss', ''),
            'seed': args.get('seed', ''),
            'parameters': data.get('parameters', ''),
            'total_train_seconds': data.get('total_train_seconds', ''),
            'best_val_accuracy': data.get('best_val_accuracy', ''),
            'test_accuracy': test.get('accuracy', ''),
            'test_error': test.get('error', ''),
            'summary_path': str(path),
            'model_path': data.get('best_model_path', ''),
            'history_path': data.get('history_path', ''),
        })
    return rows


def collect_vgg_summaries(results_dir):
    rows = []
    for path in sorted(Path(results_dir).glob('*_summary.json')):
        data = load_json(path)
        test = data.get('test_metrics', {})
        rows.append({
            'run_name': data.get('run_name', path.stem),
            'model': data.get('model', ''),
            'learning_rate': data.get('learning_rate', ''),
            'seed': data.get('seed', ''),
            'epochs': data.get('epochs', ''),
            'total_train_seconds': data.get('total_train_seconds', ''),
            'best_val_accuracy': data.get('best_val_accuracy', ''),
            'test_accuracy': test.get('accuracy', ''),
            'test_error': test.get('error', ''),
            'summary_path': str(path),
            'model_path': data.get('model_path', ''),
        })
    return rows


def write_csv(rows, path):
    ensure_dir(Path(path).parent)
    if not rows:
        return
    with open(path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_history(path):
    with open(path, newline='') as csv_file:
        return list(csv.DictReader(csv_file))


def _display_loss(value):
    return {
        'cross_entropy': 'CE',
        'label_smoothing': 'LS',
        'focal': 'Focal',
    }.get(value, value)


def _display_optimizer(value):
    return {
        'adamw': 'AdamW',
        'sgd': 'SGD',
        'rmsprop': 'RMSprop',
    }.get(value, value)


def _display_model(value):
    return {
        'plaincnn': 'Plain',
        'rescnn': 'Res',
        'conv_stem_mixer': 'Mixer',
        'conv_token_transformer': 'Trans',
        'vgg_light': 'VGG-L',
        'vgg_dropout': 'VGG-D',
        'vgg_a': 'VGG-A',
        'vgg_a_bn': 'VGG-BN',
    }.get(value, value)


def _compact_widths(value):
    widths = str(value).split(',')
    if len(widths) >= 2:
        return f'w{widths[0]}-{widths[-1]}'
    return f'w{value}' if value else ''


def _compact_float(value):
    try:
        return f'{float(value):.0e}'.replace('e-0', 'e-').replace('e+0', 'e+')
    except (TypeError, ValueError):
        return str(value)


def history_label(row):
    return (
        f"{_display_model(row.get('model'))} "
        f"{_compact_widths(row.get('widths'))} "
        f"b{row.get('blocks_per_stage')} "
        f"{_display_optimizer(row.get('optimizer'))} "
        f"{_display_loss(row.get('loss'))} "
        f"wd{_compact_float(row.get('weight_decay'))} "
        f"s{row.get('seed')}"
    )


def _as_float(row, key, default=0.0):
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _std(values):
    if len(values) < 2:
        return 0.0
    center = _mean(values)
    return float(np.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1)))


def matched_family_groups(rows):
    groups = {}
    for row in rows:
        model = row.get('model', '')
        if row.get('optimizer') != 'adamw':
            continue
        if str(row.get('lr')) != '0.001':
            continue
        if str(row.get('weight_decay')) != '0.0001':
            continue
        if row.get('loss') != 'cross_entropy':
            continue
        if row.get('activation') != 'relu':
            continue
        if model in {'plaincnn', 'rescnn', 'conv_stem_mixer', 'conv_token_transformer'}:
            if row.get('widths') != '48,96,192' or str(row.get('blocks_per_stage')) != '2':
                continue
        elif model in {'vgg_light', 'vgg_dropout'}:
            if row.get('widths') != '32,64,128' or str(row.get('blocks_per_stage')) != '2':
                continue
        else:
            continue
        groups.setdefault(model, []).append(row)
    for model in groups:
        groups[model] = sorted(groups[model], key=lambda item: int(item.get('seed') or 0))
    return groups


def plot_histories(summary_rows, output_path, limit=8):
    rows = sorted(
        [row for row in summary_rows if row.get('history_path')],
        key=lambda row: _as_float(row, 'best_val_accuracy'),
        reverse=True,
    )[:limit]
    if not rows:
        return

    plt.figure(figsize=(9, 5))
    for row in rows:
        history = read_history(row['history_path'])
        epochs = [int(item['epoch']) for item in history]
        vals = [float(item['val_accuracy']) for item in history]
        label = history_label(row)
        plt.plot(epochs, vals, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Validation accuracy')
    plt.title('Task 1 validation accuracy curves')
    plt.legend(fontsize=8, loc='lower right')
    plt.tight_layout()
    ensure_dir(Path(output_path).parent)
    plt.savefig(output_path)
    plt.close()


def plot_matched_family_comparison(summary_rows, output_path):
    groups = matched_family_groups(summary_rows)
    order = [
        'vgg_light', 'vgg_dropout', 'plaincnn', 'conv_stem_mixer',
        'conv_token_transformer', 'rescnn',
    ]
    rows = []
    for model in order:
        model_rows = groups.get(model, [])
        if not model_rows:
            continue
        scores = [_as_float(row, 'test_accuracy') for row in model_rows]
        rows.append((model, _mean(scores), _std(scores), len(scores)))
    if len(rows) < 2:
        return

    labels = [_display_model(model) for model, _, _, _ in rows]
    means = [100 * score for _, score, _, _ in rows]
    errors = [100 * spread for _, _, spread, _ in rows]
    colors = ['#7f7f7f', '#4c78a8', '#72b7b2', '#f58518', '#54a24b', '#e45756']
    x_positions = np.arange(len(rows))

    plt.figure(figsize=(9.5, 4.8))
    bars = plt.bar(
        x_positions,
        means,
        yerr=errors,
        capsize=4,
        color=colors[:len(rows)],
        edgecolor='#222222',
        linewidth=0.7,
    )
    plt.ylabel('Test accuracy (%)')
    plt.title('Three-seed matched model-family comparison')
    plt.xticks(x_positions, labels, rotation=18, ha='right')
    lower = max(70.0, min(means) - max(errors or [0]) - 2.0)
    upper = min(95.0, max(means) + max(errors or [0]) + 2.0)
    plt.ylim(lower, upper)
    plt.grid(axis='y', linestyle='--', alpha=0.35)
    for bar, (_, mean_score, _, n_seeds) in zip(bars, rows):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.25,
            f'{100 * mean_score:.2f}%\nn={n_seeds}',
            ha='center',
            va='bottom',
            fontsize=8,
        )
    plt.tight_layout()
    ensure_dir(Path(output_path).parent)
    plt.savefig(output_path)
    plt.close()


def plot_efficiency_tradeoff(summary_rows, output_path):
    groups = matched_family_groups(summary_rows)
    order = [
        'vgg_light', 'vgg_dropout', 'plaincnn', 'conv_stem_mixer',
        'conv_token_transformer', 'rescnn',
    ]
    rows = []
    for model in order:
        model_rows = groups.get(model, [])
        if not model_rows:
            continue
        test_scores = [_as_float(row, 'test_accuracy') for row in model_rows]
        val_scores = [_as_float(row, 'best_val_accuracy') for row in model_rows]
        times = [_as_float(row, 'total_train_seconds') / 60 for row in model_rows]
        params = _as_float(model_rows[0], 'parameters') / 1_000_000
        rows.append({
            'model': model,
            'params_m': params,
            'mean_test': 100 * _mean(test_scores),
            'gap_pp': 100 * (_mean(val_scores) - _mean(test_scores)),
            'time_min': _mean(times),
        })
    if len(rows) < 2:
        return

    x_values = [row['params_m'] for row in rows]
    y_values = [row['mean_test'] for row in rows]
    gaps = [row['gap_pp'] for row in rows]
    sizes = [120 + 90 * row['time_min'] for row in rows]

    plt.figure(figsize=(8.8, 4.9))
    scatter = plt.scatter(
        x_values,
        y_values,
        s=sizes,
        c=gaps,
        cmap='coolwarm',
        edgecolor='#222222',
        linewidth=0.7,
        alpha=0.9,
    )
    for row in rows:
        plt.annotate(
            _display_model(row['model']),
            (row['params_m'], row['mean_test']),
            xytext=(6, 4),
            textcoords='offset points',
            fontsize=8,
        )
    plt.xscale('log')
    plt.xlabel('Parameters (millions, log scale)')
    plt.ylabel('Mean test accuracy (%)')
    plt.title('Task 1 efficiency and generalization trade-off')
    plt.grid(True, which='both', linestyle='--', alpha=0.30)
    colorbar = plt.colorbar(scatter)
    colorbar.set_label('Validation - test gap (pp)')
    for time_min in (1.8, 2.4, 2.8):
        plt.scatter([], [], s=120 + 90 * time_min, color='#bbbbbb',
                    edgecolor='#222222', label=f'{time_min:.1f} min/run')
    plt.legend(
        title='Point size',
        fontsize=8,
        title_fontsize=8,
        loc='lower right',
        frameon=True,
    )
    plt.tight_layout()
    ensure_dir(Path(output_path).parent)
    plt.savefig(output_path)
    plt.close()


def plot_hybrid_controls(summary_rows, output_path):
    models = ['conv_stem_mixer', 'conv_token_transformer']
    variants = [
        ('cross_entropy', '0.0001', 'CE wd1e-4'),
        ('label_smoothing', '0.0001', 'LS wd1e-4'),
        ('cross_entropy', '0.0005', 'CE wd5e-4'),
        ('label_smoothing', '0.0005', 'LS wd5e-4'),
    ]
    scores = []
    for model in models:
        model_scores = []
        for loss, weight_decay, _ in variants:
            matches = [
                row for row in summary_rows
                if row.get('model') == model
                and row.get('widths') == '48,96,192'
                and str(row.get('blocks_per_stage')) == '2'
                and row.get('activation') == 'relu'
                and row.get('optimizer') == 'adamw'
                and str(row.get('lr')) == '0.001'
                and str(row.get('seed')) == '2020'
                and row.get('loss') == loss
                and str(row.get('weight_decay')) == weight_decay
            ]
            model_scores.append(100 * _as_float(matches[0], 'test_accuracy') if matches else np.nan)
        scores.append(model_scores)
    if all(np.isnan(value) for model_scores in scores for value in model_scores):
        return

    x_positions = np.arange(len(models))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(variants))
    colors = ['#4c78a8', '#72b7b2', '#f58518', '#e45756']

    plt.figure(figsize=(8.8, 4.5))
    for idx, (_, _, label) in enumerate(variants):
        values = [model_scores[idx] for model_scores in scores]
        bars = plt.bar(
            x_positions + offsets[idx],
            values,
            width=width,
            label=label,
            color=colors[idx],
            edgecolor='#222222',
            linewidth=0.6,
        )
        for bar, value in zip(bars, values):
            if np.isnan(value):
                continue
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.12,
                f'{value:.1f}',
                ha='center',
                va='bottom',
                fontsize=8,
            )
    plt.ylabel('Test accuracy (%)')
    plt.title('Hybrid-family regularization controls, seed 2020')
    plt.xticks(x_positions, [_display_model(model) for model in models])
    finite_scores = [value for model_scores in scores for value in model_scores if not np.isnan(value)]
    plt.ylim(max(80.0, min(finite_scores) - 1.5), min(90.0, max(finite_scores) + 1.5))
    plt.grid(axis='y', linestyle='--', alpha=0.35)
    plt.legend(fontsize=8, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.12))
    plt.tight_layout()
    ensure_dir(Path(output_path).parent)
    plt.savefig(output_path)
    plt.close()


def find_first_conv(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d) and module.weight.shape[1] == 3:
            return module
    raise ValueError('No first RGB convolution found')


def plot_first_layer_filters(model, output_path, max_filters=32):
    conv = find_first_conv(model)
    weights = conv.weight.detach().cpu()
    num_filters = min(max_filters, weights.shape[0])
    cols = min(8, num_filters)
    rows = int(np.ceil(num_filters / cols))
    plt.figure(figsize=(cols * 1.2, rows * 1.2))
    for idx in range(num_filters):
        filt = weights[idx]
        filt = filt - filt.min()
        filt = filt / filt.max().clamp_min(1e-12)
        image = np.transpose(filt.numpy(), (1, 2, 0))
        ax = plt.subplot(rows, cols, idx + 1)
        ax.imshow(image)
        ax.axis('off')
    plt.suptitle('First-layer convolution filters')
    plt.tight_layout()
    ensure_dir(Path(output_path).parent)
    plt.savefig(output_path)
    plt.close()


def confusion_matrix(model, data_loader, device):
    matrix = torch.zeros(10, 10, dtype=torch.int64)
    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs.to(device))
            preds = outputs.argmax(dim=1).cpu()
            for label, pred in zip(labels.cpu(), preds):
                matrix[label, pred] += 1
    return matrix.numpy()


def plot_confusion_matrix(matrix, output_path):
    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, interpolation='nearest', cmap='Blues')
    plt.title('Confusion matrix on CIFAR-10 test set')
    plt.colorbar(fraction=0.046, pad=0.04)
    ticks = np.arange(len(CLASS_NAMES))
    plt.xticks(ticks, CLASS_NAMES, rotation=45, ha='right')
    plt.yticks(ticks, CLASS_NAMES)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    ensure_dir(Path(output_path).parent)
    plt.savefig(output_path)
    plt.close()


def load_task1_model(summary_row, device):
    data = load_json(summary_row['summary_path'])
    args = SimpleNamespace(**data['args'])
    model = build_model(args).to(device)
    checkpoint = torch.load(summary_row['model_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, args


def best_task1_row(rows):
    candidates = [row for row in rows if row.get('model_path') and Path(row['model_path']).exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row.get('test_accuracy') or 0.0))


def configure_arguments():
    parser = argparse.ArgumentParser(description='Analyze PJ2 experiment outputs')
    parser.add_argument('--reports-dir', default='reports')
    parser.add_argument('--root', default='./data')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--skip-model-figures', action='store_true')
    return parser.parse_args()


def main():
    args = configure_arguments()
    reports_dir = Path(args.reports_dir)
    results_dir = reports_dir / 'results'
    figures_dir = reports_dir / 'figures'

    task1_rows = collect_task1_summaries(results_dir)
    vgg_rows = collect_vgg_summaries(results_dir)
    write_csv(task1_rows, results_dir / 'task1_summary.csv')
    write_csv(vgg_rows, results_dir / 'vgg_bn_summary.csv')
    plot_histories(task1_rows, figures_dir / 'task1_validation_curves.pdf')
    plot_matched_family_comparison(task1_rows, figures_dir / 'task1_matched_family_comparison.pdf')
    plot_efficiency_tradeoff(task1_rows, figures_dir / 'task1_efficiency_tradeoff.pdf')
    plot_hybrid_controls(task1_rows, figures_dir / 'task1_hybrid_controls.pdf')

    best_row = best_task1_row(task1_rows)
    if best_row is not None and not args.skip_model_figures:
        device = get_device(args.device)
        model, model_args = load_task1_model(best_row, device)
        plot_first_layer_filters(model, figures_dir / 'task1_first_layer_filters.pdf')
        _, _, test_loader = get_cifar_loaders(
            root=args.root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=model_args.seed,
        )
        criterion = build_criterion(model_args)
        test_metrics = evaluate(model, test_loader, criterion, device)
        matrix = confusion_matrix(model, test_loader, device)
        plot_confusion_matrix(matrix, figures_dir / 'task1_confusion_matrix.pdf')
        print(f'best_task1={best_row["run_name"]}')
        print(f'best_task1_test_accuracy={test_metrics["accuracy"]:.4f}')

    print(f'task1_summary={results_dir / "task1_summary.csv"}')
    print(f'vgg_summary={results_dir / "vgg_bn_summary.csv"}')


if __name__ == '__main__':
    main()
