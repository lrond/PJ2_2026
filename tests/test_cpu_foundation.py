import importlib
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


class ToyDataset(Dataset):
    def __init__(self):
        self.items = [("zero", 0), ("one", 1), ("two", 2)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 4 * 4, 10),
        )

    def forward(self, x):
        return self.net(x)


def test_partial_dataset_forwards_requested_index():
    from data.loaders import PartialDataset

    dataset = PartialDataset(ToyDataset(), n_items=2)

    assert len(dataset) == 2
    assert dataset[1] == ("one", 1)


def test_vgg_batchnorm_forward_contract():
    from models.vgg import VGG_A_BatchNorm, get_number_of_parameters

    model = VGG_A_BatchNorm()
    output = model(torch.randn(2, 3, 32, 32))

    assert output.shape == (2, 10)
    assert get_number_of_parameters(model) > 0
    assert any(isinstance(layer, nn.BatchNorm2d) for layer in model.modules())


def test_plaincnn_forward_contract():
    from models.plaincnn import PlainCNN

    model = PlainCNN(widths=(8, 16, 32), blocks_per_stage=1)
    output = model(torch.randn(2, 3, 32, 32))

    assert output.shape == (2, 10)
    assert sum(parameter.numel() for parameter in model.parameters()) > 0
    assert any(isinstance(layer, nn.MaxPool2d) for layer in model.modules())


def test_hybrid_models_forward_contract():
    from models.modern import ConvStemMixer, ConvTokenTransformer

    for model_class in (ConvStemMixer, ConvTokenTransformer):
        model = model_class(widths=(8, 16, 32), blocks_per_stage=1)
        output = model(torch.randn(2, 3, 32, 32))

        assert output.shape == (2, 10)
        assert sum(parameter.numel() for parameter in model.parameters()) > 0
        assert any(isinstance(layer, nn.Conv2d) for layer in model.modules())
        assert any(isinstance(layer, nn.MaxPool2d) for layer in model.modules())


def test_training_helpers_run_on_cpu_synthetic_data():
    from training import evaluate, train_one_epoch

    loader = DataLoader(
        TensorDataset(
            torch.randn(8, 3, 4, 4),
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
        ),
        batch_size=4,
    )
    model = TinyClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")

    train_metrics = train_one_epoch(model, loader, optimizer, criterion, device)
    eval_metrics = evaluate(model, loader, criterion, device)

    assert train_metrics["loss"] > 0
    assert len(train_metrics["step_losses"]) == 2
    assert "accuracy" in eval_metrics
    assert "error" in eval_metrics


def test_loss_landscape_module_import_has_no_training_side_effects():
    module = importlib.import_module("VGG_Loss_Landscape")

    assert hasattr(module, "compute_loss_band")
