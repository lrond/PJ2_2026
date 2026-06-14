"""
Training and evaluation helpers shared by CIFAR-10 experiments.
"""
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_device(device_name='auto'):
    if device_name != 'auto':
        return torch.device(device_name)
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if str(device) != 'cpu' and torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def accuracy_from_logits(logits, labels):
    predictions = logits.argmax(dim=1)
    return (predictions == labels).float().mean().item()


def make_synthetic_loaders(batch_size=8, train_size=32, val_size=16, image_size=32,
                           num_classes=10):
    train_x = torch.randn(train_size, 3, image_size, image_size)
    train_y = torch.arange(train_size) % num_classes
    val_x = torch.randn(val_size, 3, image_size, image_size)
    val_y = torch.arange(val_size) % num_classes
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, val_loader


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    return {
        'loss': total_loss / total_samples,
        'accuracy': accuracy,
        'error': 1.0 - accuracy,
    }


def train_one_epoch(model, data_loader, optimizer, criterion, device, grad_parameter=None):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    step_losses = []
    grad_norms = []

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        if grad_parameter is not None and grad_parameter.grad is not None:
            grad_norms.append(grad_parameter.grad.detach().norm().item())

        optimizer.step()

        step_losses.append(loss.item())
        total_loss += loss.item() * labels.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    return {
        'loss': total_loss / total_samples,
        'accuracy': accuracy,
        'error': 1.0 - accuracy,
        'step_losses': step_losses,
        'grad_norms': grad_norms,
    }


def fit(model, train_loader, val_loader, optimizer, criterion, device, epochs,
        scheduler=None, best_model_path=None, grad_parameter=None):
    model.to(device)
    best_accuracy = -1.0
    history = []
    all_step_losses = []
    all_grad_norms = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            grad_parameter=grad_parameter,
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        epoch_seconds = time.perf_counter() - epoch_start
        lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step()

        row = {
            'epoch': epoch,
            'lr': lr,
            'train_loss': train_metrics['loss'],
            'train_accuracy': train_metrics['accuracy'],
            'train_error': train_metrics['error'],
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'val_error': val_metrics['error'],
            'epoch_seconds': epoch_seconds,
        }
        history.append(row)
        all_step_losses.append(train_metrics['step_losses'])
        all_grad_norms.append(train_metrics['grad_norms'])

        if best_model_path is not None and val_metrics['accuracy'] > best_accuracy:
            best_accuracy = val_metrics['accuracy']
            ensure_dir(Path(best_model_path).parent)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
            }, best_model_path)

    return {
        'history': history,
        'step_losses': all_step_losses,
        'grad_norms': all_grad_norms,
    }


def save_history_csv(history, path):
    ensure_dir(Path(path).parent)
    if not history:
        return
    with open(path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_json(data, path):
    ensure_dir(Path(path).parent)
    with open(path, 'w') as json_file:
        json.dump(data, json_file, indent=2)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, labels):
        ce_loss = self.cross_entropy(logits, labels)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()
