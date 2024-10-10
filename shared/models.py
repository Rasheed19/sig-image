from collections import OrderedDict

import torch
import torch.nn as nn

from shared.definition import SignatureMode
from shared.signature import Diehl2DSignature, Zhang2DSignature


def load_pretrained_model() -> nn.Module:
    model = torch.hub.load(
        repo_or_dir="chenyaofo/pytorch-cifar-models",
        model="cifar10_resnet20",
        trust_repo=True,
        pretrained=True,
    )

    return model


class Signature2DPoolingLayer(nn.Module):
    def __init__(self, sig_mode: str, channels: int, sig_depth: int):
        super().__init__()

        self.sig_mode = sig_mode
        self.channels = channels
        self.sig_depth = sig_depth

        if self.sig_mode == SignatureMode.ZHANG:
            self.sig = Zhang2DSignature()
        elif self.sig_mode == SignatureMode.DIEHL:
            self.sig = Diehl2DSignature()
        else:
            raise ValueError(
                f"Signature mode must be an element of {[m for m in SignatureMode]}"
            )

    def calculate_feature_dim(self):
        return self.sig.calculate_feature_dim(
            channels=self.channels, depth=self.sig_depth
        )

    def forward(self, x):
        return self.sig.calculate_batch_sig(x=x, depth=self.sig_depth)


class SignaturedAsPoolingModel(nn.Module):
    def __init__(
        self,
        pretrained_model: nn.Module,
        sig_mode: str,
        sig_depth: int,
        num_classes: int,
    ):
        super().__init__()

        # fix all layers fc
        for name, param in pretrained_model.named_parameters():
            param.requires_grad = True if name.startswith("fc") else False

        sig_layer = Signature2DPoolingLayer(
            sig_mode=sig_mode,
            channels=pretrained_model.layer3[2].bn2.num_features,
            sig_depth=sig_depth,
        )
        sig_feature_dim = sig_layer.calculate_feature_dim()

        pretrained_model.avgpool = sig_layer
        pretrained_model.fc = nn.Identity()

        self.model = nn.Sequential(
            OrderedDict(
                [
                    ("pretrained", pretrained_model),
                    ("batchnorm1d", nn.BatchNorm1d(num_features=sig_feature_dim)),
                    (
                        "sig_fc",
                        nn.Linear(
                            in_features=sig_feature_dim, out_features=sig_feature_dim
                        ),
                    ),
                    (
                        "fc",
                        nn.Linear(
                            in_features=sig_feature_dim,
                            out_features=num_classes,
                        ),
                    ),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SignatureAsBlockModel(nn.Module):
    def __init__(
        self,
        pretrained_model: nn.Module,
        sig_mode: str,
        sig_depth: int,
        num_classes: int,
    ):
        super().__init__()

        # fix all layers except layer 3 and fc
        for name, param in pretrained_model.named_parameters():
            if name.startswith("layer3") or name.startswith("fc"):
                param.requires_grad = True
            else:
                param.requires_grad = False

        sig_layer = Signature2DPoolingLayer(
            sig_mode=sig_mode,
            channels=pretrained_model.layer2[2].bn2.num_features,
            sig_depth=sig_depth,
        )
        sig_feature_dim = sig_layer.calculate_feature_dim()

        pretrained_model.layer3 = sig_layer
        pretrained_model.avgpool = (
            nn.Identity()
        )  # remove the avg pool just before final fc layer
        pretrained_model.fc = nn.Identity()  # remove the fc layer

        self.model = nn.Sequential(
            OrderedDict(
                [
                    ("pretrained", pretrained_model),
                    ("batchnorm1d", nn.BatchNorm1d(num_features=sig_feature_dim)),
                    (
                        "sig_linear",
                        nn.Linear(
                            in_features=sig_feature_dim,
                            out_features=sig_feature_dim,
                        ),
                    ),
                    (
                        "fc",
                        nn.Linear(
                            in_features=sig_feature_dim,
                            out_features=num_classes,
                        ),
                    ),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
