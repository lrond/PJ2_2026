"""
Train CIFAR-10 models for Project 2 task 1.
"""
import argparse
from pathlib import Path

import torch
from torch import nn

from data.loaders import get_cifar_loaders
from models.modern import ConvStemMixer, ConvTokenTransformer
from models.plaincnn import PlainCNN
from models.rescnn import CIFARResCNN
from models.vgg import VGG_A, VGG_A_BatchNorm, VGG_A_Dropout, VGG_A_Light, get_number_of_parameters
from training import (
    FocalLoss,
    count_parameters,
    evaluate,
    fit,
    get_device,
    make_synthetic_loaders,
    save_history_csv,
    save_json,
    set_random_seeds,
)


def parse_widths(value):
    widths = tuple(int(item) for item in value.split(','))
    if len(widths) != 3:
        raise argparse.ArgumentTypeError('--widths must contain exactly 3 integers')
    return widths


def build_model(args):
    if args.model == 'plaincnn':
        return PlainCNN(
            widths=args.widths,
            blocks_per_stage=args.blocks_per_stage,
            activation=args.activation,
            dropout=args.dropout,
        )
    if args.model == 'rescnn':
        return CIFARResCNN(
            widths=args.widths,
            blocks_per_stage=args.blocks_per_stage,
            activation=args.activation,
            dropout=args.dropout,
        )
    if args.model == 'conv_token_transformer':
        return ConvTokenTransformer(
            widths=args.widths,
            blocks_per_stage=args.blocks_per_stage,
            activation=args.activation,
            dropout=args.dropout,
        )
    if args.model == 'conv_stem_mixer':
        return ConvStemMixer(
            widths=args.widths,
            blocks_per_stage=args.blocks_per_stage,
            activation=args.activation,
            dropout=args.dropout,
        )
    if args.model == 'vgg_a':
        return VGG_A()
    if args.model == 'vgg_a_bn':
        return VGG_A_BatchNorm()
    if args.model == 'vgg_light':
        return VGG_A_Light()
    if args.model == 'vgg_dropout':
        return VGG_A_Dropout()
    raise ValueError(f'Unknown model: {args.model}')


def build_optimizer(args, model):
    if args.optimizer == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    if args.optimizer == 'rmsprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    raise ValueError(f'Unknown optimizer: {args.optimizer}')


def build_criterion(args):
    if args.loss == 'cross_entropy':
        return nn.CrossEntropyLoss()
    if args.loss == 'label_smoothing':
        return nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if args.loss == 'focal':
        return FocalLoss(gamma=args.focal_gamma)
    raise ValueError(f'Unknown loss: {args.loss}')


def build_scheduler(args, optimizer):
    if args.scheduler == 'none':
        return None
    if args.scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr,
        )
    raise ValueError(f'Unknown scheduler: {args.scheduler}')


def make_run_name(args):
    width_part = '-'.join(str(width) for width in args.widths)
    return (
        f'{args.model}_w-{width_part}_b-{args.blocks_per_stage}_'
        f'act-{args.activation}_opt-{args.optimizer}_lr-{args.lr:g}_'
        f'wd-{args.weight_decay:g}_loss-{args.loss}_seed-{args.seed}'
    )


def configure_arguments():
    parser = argparse.ArgumentParser(description='CIFAR-10 training entrypoint')
    parser.add_argument('--model', default='rescnn',
                        choices=[
                            'plaincnn', 'rescnn', 'conv_token_transformer',
                            'conv_stem_mixer', 'vgg_a', 'vgg_a_bn',
                            'vgg_light', 'vgg_dropout',
                        ])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--root', default='./data')
    parser.add_argument('--output-dir', default='reports')
    parser.add_argument('--n-items', type=int, default=-1)
    parser.add_argument('--val-size', type=int, default=5000)
    parser.add_argument('--synthetic', action='store_true')

    parser.add_argument('--widths', type=parse_widths, default=(32, 64, 128))
    parser.add_argument('--blocks-per-stage', type=int, default=2)
    parser.add_argument('--activation', default='relu',
                        choices=['relu', 'leaky_relu', 'elu', 'tanh'])
    parser.add_argument('--dropout', type=float, default=0.2)

    parser.add_argument('--optimizer', default='adamw',
                        choices=['adamw', 'adam', 'sgd', 'rmsprop'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min-lr', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--scheduler', default='cosine', choices=['cosine', 'none'])
    parser.add_argument('--loss', default='cross_entropy',
                        choices=['cross_entropy', 'label_smoothing', 'focal'])
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    return parser.parse_args()


def main():
    args = configure_arguments()
    device = get_device(args.device)
    set_random_seeds(args.seed, device)

    if args.synthetic:
        train_loader, val_loader, test_loader = make_synthetic_loaders(batch_size=args.batch_size)
    else:
        train_loader, val_loader, test_loader = get_cifar_loaders(
            root=args.root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_size=args.val_size,
            seed=args.seed,
            n_items=args.n_items,
        )

    model = build_model(args)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)
    criterion = build_criterion(args)

    output_dir = Path(args.output_dir)
    run_name = make_run_name(args)
    model_path = output_dir / 'models' / f'{run_name}.pth'
    history_path = output_dir / 'results' / f'{run_name}.csv'
    summary_path = output_dir / 'results' / f'{run_name}.json'

    result = fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        epochs=args.epochs,
        scheduler=scheduler,
        best_model_path=model_path,
    )
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = evaluate(model, test_loader, criterion, device)
    parameter_count = (
        get_number_of_parameters(model)
        if args.model.startswith('vgg')
        else count_parameters(model)
    )

    save_history_csv(result['history'], history_path)
    save_json({
        'run_name': run_name,
        'args': vars(args),
        'device': str(device),
        'parameters': parameter_count,
        'total_train_seconds': sum(row['epoch_seconds'] for row in result['history']),
        'best_val_accuracy': max(row['val_accuracy'] for row in result['history']),
        'best_val_error': min(row['val_error'] for row in result['history']),
        'test_metrics': test_metrics,
        'best_model_path': str(model_path),
        'history_path': str(history_path),
    }, summary_path)

    print(f'run_name={run_name}')
    print(f'device={device}')
    print(f'parameters={parameter_count}')
    print(f'test_accuracy={test_metrics["accuracy"]:.4f}')
    print(f'test_error={test_metrics["error"]:.4f}')
    print(f'history={history_path}')
    print(f'model={model_path}')


if __name__ == '__main__':
    main()
