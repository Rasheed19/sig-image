import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from shared.definition import ModelMode, SignatureMode
from shared.helper import dump_data
from shared.plotter import plot_training_pipeline_history
from shared.signature import Diehl2DSignature, Zhang2DSignature


def create_benchmark_model(benchmark_model_name: str) -> nn.Module:
    model = torch.hub.load(
        repo_or_dir="chenyaofo/pytorch-cifar-models",
        model=benchmark_model_name,
        trust_repo=True,
        pretrained=True,
    )

    return model


class Signature2DPoolingLayer(nn.Module):
    def __init__(self, mode: str, channels: int, depth: int):
        super().__init__()

        self.mode = mode
        self.channels = channels
        self.depth = depth

        if self.mode == SignatureMode.ZHANG:
            self.sig = Zhang2DSignature()
        elif self.mode == SignatureMode.DIEHL:
            self.sig = Diehl2DSignature()
        else:
            raise ValueError(
                f"Signature mode must be an element of {[m for m in SignatureMode]}"
            )

    def calculate_feature_dim(self):
        return self.sig.calculate_feature_dim(channels=self.channels, depth=self.depth)

    def forward(self, x):
        return self.sig.calculate_batch_sig(x=x, depth=self.depth)


def create_signature_informed_model(
    num_classes: int,
    mode: str,
    depth: int,
    channels: int | None = None,
    replace_avgpool: bool = False,
) -> nn.Module:
    model = create_benchmark_model(
        benchmark_model_name="cifar10_resnet20",  # TODO: make this an StrEnum
    )

    if replace_avgpool:
        pen_channel = model.layer3[2].bn2.num_features

        sig_layer = Signature2DPoolingLayer(
            mode=mode,
            channels=pen_channel,
            depth=depth,
        )

        model.avgpool = sig_layer

        model.fc = nn.Linear(
            in_features=sig_layer.calculate_feature_dim(), out_features=num_classes
        )

        return model

    pen_channel = (
        model.layer2[2].bn2.num_features
    )  # get the channel just before the last layer; specific to resnet18!!

    sig_layer = Signature2DPoolingLayer(
        mode=mode,
        channels=pen_channel,
        depth=depth,
    )

    # # stack non-fully-connected layers for dim reduction
    # model.layer4 = nn.Sequential(
    #     nn.Conv2d(
    #         in_channels=pen_channel,
    #         out_channels=128,
    #         kernel_size=3,
    #         stride=1,
    #         padding=1,
    #     ),
    #     nn.BatchNorm2d(128),
    #     nn.ReLU(inplace=True),
    #     nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
    #     nn.BatchNorm2d(64),
    #     nn.ReLU(inplace=True),
    #     nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
    #     nn.BatchNorm2d(32),
    #     nn.ReLU(inplace=True),
    #     nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
    #     nn.BatchNorm2d(16),
    #     nn.ReLU(inplace=True),
    #     nn.Conv2d(
    #         in_channels=16, out_channels=channels, kernel_size=3, stride=1, padding=1
    #     ),
    #     nn.BatchNorm2d(channels),
    #     nn.ReLU(inplace=True),
    #     sig_layer,
    # )

    model.layer3 = sig_layer

    model.avgpool = nn.Identity()  # remove the avg pool just before final fc layer

    model.fc = nn.Linear(
        in_features=sig_layer.calculate_feature_dim(), out_features=num_classes
    )

    return model


def train_model(
    model_mode: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    batch_size: int,
    epoch: int,
    device: str,
) -> tuple[nn.Module, dict]:
    NUM_CLASSES = 10

    if model_mode == ModelMode.BENCHMARK:
        model = create_benchmark_model(
            benchmark_model_name="cifar10_resnet20",  # TODO: make this an StrEnum
        )

    elif model_mode == ModelMode.SIGNATURE:
        model = create_signature_informed_model(
            num_classes=NUM_CLASSES,
            mode="zhang",
            depth=1,
            replace_avgpool=True,
        )

    else:
        raise ValueError(
            f"'model_mode' can only take element in {[m for m in ModelMode]}"
        )

    print(model)
    model.to(device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    # Loop through the number of epochs
    for i in range(epoch):
        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0

        # set model to train mode
        model.train()
        # iterate over the training data
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            # compute the loss
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            # increment the running loss and accuracy
            train_loss += loss.item()
            train_acc += (outputs.argmax(1) == labels).sum().item()

        # calculate the average training loss and accuracy
        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)

        # set the model to evaluation mode
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_acc += (outputs.argmax(1) == labels).sum().item()

        # calculate the average validation loss and accuracy
        val_loss /= len(test_loader)
        val_acc /= len(test_loader.dataset)

        print(
            f"Epoch {i+1}/{epoch}, train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, val loss: {val_loss:.4f}, val acc: {val_acc:.4f}"
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["test_loss"].append(val_loss)
        history["test_accuracy"].append(val_acc)

    torch.save(
        obj=model.state_dict(),
        f=f"./models/{model_mode}_batch_size={batch_size}_epoch={epoch}.pt",
    )

    dump_data(
        data=history,
        fname=f"history_model={model_mode}_batch_size={batch_size}_epoch={epoch}.pkl",
        path="./data",
    )

    plot_training_pipeline_history(
        history=history, epoch=epoch, model_mode=model_mode, batch_size=batch_size
    )

    return None
