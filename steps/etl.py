import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def load_data(batch_size: int) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            ),  # Normalize the RGB channels
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )  # TODO: add argument to choose different dataset
    test_set = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    print(f"Number of training examples: {len(train_set)}")
    print(f"Number of test examples: {len(test_set)}")

    training_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    validation_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return training_loader, validation_loader
