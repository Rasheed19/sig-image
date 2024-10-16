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
    def __init__(self, sig_mode: str, channels: int, sig_depth: int, device: str):
        super().__init__()

        self.sig_mode = sig_mode
        self.channels = channels
        self.sig_depth = sig_depth
        self.device = device

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
        return self.sig.calculate_batch_sig(x=x, depth=self.sig_depth).to(
            device=self.device
        )


class SignaturedAsPoolingModel(nn.Module):
    def __init__(
        self,
        pretrained_model: nn.Module,
        sig_mode: str,
        sig_depth: int,
        num_classes: int,
        device: str,
    ):
        super().__init__()

        # fix all layers of the pretrained model
        for _, param in pretrained_model.named_parameters():
            param.requires_grad = False

        pretrained_model.avgpool = nn.Identity()
        pretrained_model.fc = nn.Identity()
        self.pretrained = pretrained_model

        in_conv_channels = pretrained_model.layer3[2].bn2.num_features
        in_sig_channels = 32  # TODO: play around with this??

        sig_layer = Signature2DPoolingLayer(
            sig_mode=sig_mode,
            channels=in_sig_channels,
            sig_depth=sig_depth,
            device=device,
        )
        sig_feature_dim = sig_layer.calculate_feature_dim()
        self.sig = nn.Sequential(
            nn.Conv2d(
                in_conv_channels,
                in_sig_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            sig_layer,
            nn.BatchNorm1d(num_features=sig_feature_dim),
            nn.Linear(
                in_features=sig_feature_dim, out_features=sig_feature_dim
            ),  # TODO: maybe increase the out_features
        )

        self.fc = nn.Linear(
            in_features=sig_feature_dim,
            out_features=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pretrained(x)
        x = self.sig(x.view(-1, 64, 8, 8))
        x = self.fc(x)

        return x


class SignatureAsBlockModel(nn.Module):
    def __init__(
        self,
        pretrained_model: nn.Module,
        sig_mode: str,
        sig_depth: int,
        num_classes: int,
        device: str,
    ):
        super().__init__()

        # fix all layers of the pretrained model
        for _, param in pretrained_model.named_parameters():
            param.requires_grad = False

        pretrained_model.layer3 = nn.Identity()
        pretrained_model.avgpool = nn.Identity()
        pretrained_model.fc = nn.Identity()
        self.pretrained = pretrained_model

        in_conv_channels = pretrained_model.layer2[2].bn2.num_features
        #  in_sig_channels = 32  # TODO: play around with this??

        sig_layer = Signature2DPoolingLayer(
            sig_mode=sig_mode,
            channels=in_conv_channels,  # in_sig_channels,
            sig_depth=sig_depth,
            device=device,
        )
        sig_feature_dim = sig_layer.calculate_feature_dim()

        self.sig = nn.Sequential(
            # nn.Conv2d(
            #     in_conv_channels,
            #     in_sig_channels,
            #     kernel_size=(3, 3),
            #     stride=(1, 1),
            #     padding=(1, 1),
            #     bias=False,
            # ),
            sig_layer,
            nn.BatchNorm1d(num_features=sig_feature_dim),
            nn.Linear(
                in_features=sig_feature_dim, out_features=sig_feature_dim
            ),  # TODO: maybe increase the out_features
        )

        self.fc = nn.Linear(
            in_features=sig_feature_dim,
            out_features=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pretrained(x)
        x = self.sig(x.view(-1, 32, 16, 16))
        x = self.fc(x)

        return x
