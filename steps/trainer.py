import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from shared.definition import ModelMode
from shared.helper import dump_data
from shared.models import (
    SignatureAsBlockModel,
    SignaturedAsPoolingModel,
    load_pretrained_model,
)
from shared.plotter import plot_training_pipeline_history


def train_model(
    model_mode: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    sig_mode: str,
    sig_depth: int,
    batch_size: int,
    epoch: int,
    device: str,
) -> tuple[nn.Module, nn.Module]:
    NUM_CLASSES = 10

    pretrained_model = load_pretrained_model()
    pretrained_clone = copy.deepcopy(pretrained_model)

    # print(pretrained_model)

    if model_mode == ModelMode.POOL:
        model = SignaturedAsPoolingModel(
            pretrained_model=pretrained_clone,
            sig_mode=sig_mode,
            sig_depth=sig_depth,
            num_classes=NUM_CLASSES,
            device=device,
        )

    elif model_mode == ModelMode.BLOCK:
        model = SignatureAsBlockModel(
            pretrained_model=pretrained_clone,
            sig_mode=sig_mode,
            sig_depth=sig_depth,
            num_classes=NUM_CLASSES,
            device=device,
        )

    else:
        raise ValueError(
            f"'model_mode' can only take element in {[m.value for m in ModelMode]}"
        )

    print(model)

    # print(model)
    model.to(device=device)

    optimizer = optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.001
    )
    # optimizer = optimizer = torch.optim.SGD(
    #     filter(lambda p: p.requires_grad, model.parameters()),
    #     lr=0.1,
    #     momentum=0.9,
    #     dampening=0,
    #     weight_decay=0.0005,
    #     nesterov=True,
    # )
    # scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=200, eta_min=0)

    criterion = nn.CrossEntropyLoss()

    history = {
        "epochs": [],
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

            # scheduler.step()

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
        history["epochs"].append(i + 1)

    artifact_tag = f"model_mode={model_mode}+sig_mode={sig_mode}+sig_depth={sig_depth}+batch_size={batch_size}+epoch={epoch}"
    torch.save(
        obj=model.state_dict(),
        f=f"./models/{artifact_tag}.pt",
    )

    dump_data(
        data=history,
        fname=f"{artifact_tag}.pkl",
        path="./data",
    )

    plot_training_pipeline_history(history=history, artifact_tag=artifact_tag)

    return pretrained_model, model
