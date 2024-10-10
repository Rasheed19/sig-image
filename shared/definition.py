from enum import IntEnum, StrEnum


class ModelMode(StrEnum):
    POOL = "pool"
    BLOCK = "block"


class PipelineMode(StrEnum):
    EDA = "eda"
    DOWNLOAD = "download"
    TRAIN = "training"


class SignatureMode(StrEnum):
    ZHANG = "zhang"
    DIEHL = "diehl"


class CIFAR10Classes(IntEnum):
    AIRPLANE = 0
    AUTOMOBILE = 1
    BIRD = 2
    CAT = 3
    DEER = 4
    DOG = 5
    FROG = 6
    HORSE = 7
    SHIP = 8
    TRUCK = 9


if __name__ == "__main__":
    print(CIFAR10Classes.DOG)
