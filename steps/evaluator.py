import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: int,
) -> float:
    accuracy = 0.0
    model.eval()

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            accuracy += (outputs.argmax(1) == labels).sum().item()

    accuracy /= len(test_loader.dataset)

    return accuracy
