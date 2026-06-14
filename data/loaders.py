"""
Data loaders
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset



class PartialDataset(Dataset):
    def __init__(self, dataset, n_items=10):
        self.dataset = dataset
        self.n_items = n_items

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return min(self.n_items, len(self.dataset))


def _require_torchvision():
    try:
        from torchvision import transforms
        import torchvision.datasets as datasets
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            'torchvision is required for CIFAR-10 loading. Install it in the GPU '
            'environment before running full experiments.'
        ) from exc
    return transforms, datasets


def _cifar_transforms(augment=False):
    transforms, _ = _require_torchvision()
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

    ops = []
    if augment:
        ops.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    ops.extend([transforms.ToTensor(), normalize])
    return transforms.Compose(ops)


def _split_indices(total_items, val_size, seed):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_items, generator=generator).tolist()
    val_size = min(val_size, max(1, total_items // 5))
    return indices[val_size:], indices[:val_size]


def get_cifar_loader(root='../data/', batch_size=128, train=True, shuffle=True, num_workers=4, n_items=-1):
    _, datasets = _require_torchvision()
    dataset = datasets.CIFAR10(
        root=root,
        train=train,
        download=True,
        transform=_cifar_transforms(augment=train),
    )
    if n_items > 0:
        dataset = PartialDataset(dataset, n_items)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return loader


def get_cifar_loaders(root='../data/', batch_size=128, num_workers=4, val_size=5000,
                      seed=2020, n_items=-1):
    _, datasets = _require_torchvision()
    train_full = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=_cifar_transforms(augment=True),
    )
    val_full = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=_cifar_transforms(augment=False),
    )
    test_dataset = datasets.CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=_cifar_transforms(augment=False),
    )

    total_items = len(train_full) if n_items <= 0 else min(n_items, len(train_full))
    train_idx, val_idx = _split_indices(total_items, val_size, seed)
    train_dataset = Subset(train_full, train_idx)
    val_dataset = Subset(val_full, val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    train_loader = get_cifar_loader()
    for X, y in train_loader:
        print(X[0])
        print(y[0])
        print(X[0].shape)
        img = np.transpose(X[0], [1,2,0])
        plt.imshow(img*0.5 + 0.5)
        plt.savefig('sample.png')
        print(X[0].max())
        print(X[0].min())
        break
