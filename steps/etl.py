import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


def filter_classes(dataset: DataLoader, class_labels: list[str] | tuple[str]) -> Subset:
    indices = [i for i, label in enumerate(dataset.targets) if label in class_labels]
    return Subset(dataset, indices)


def load_data(batch_size: int) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(
                # (0.485, 0.456, 0.406),
                # (0.229, 0.224, 0.225),
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.201),  # from the source
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

    # class_labels = (
    #     CIFAR10Classes.AIRPLANE,
    #     CIFAR10Classes.AUTOMOBILE,
    # )  # TODO: get only two classes; might be changed later for experimentation
    # train_set = filter_classes(train_set, class_labels)
    # test_set = filter_classes(test_set, class_labels)

    print(f"Number of training examples: {len(train_set)}")
    print(f"Number of test examples: {len(test_set)}")

    training_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    validation_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return training_loader, validation_loader
