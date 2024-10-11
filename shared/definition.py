from enum import Enum, EnumMeta


class MetaEnum(EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True


class BaseEnum(Enum, metaclass=MetaEnum):
    pass


class ModelMode(str, BaseEnum):
    POOL = "pool"
    BLOCK = "block"


class PipelineMode(str, BaseEnum):
    EDA = "eda"
    DOWNLOAD = "download"
    TRAIN = "training"


class SignatureMode(str, BaseEnum):
    ZHANG = "zhang"
    DIEHL = "diehl"


# class CIFAR10Classes(IntEnum):
#     AIRPLANE = 0
#     AUTOMOBILE = 1
#     BIRD = 2
#     CAT = 3
#     DEER = 4
#     DOG = 5
#     FROG = 6
#     HORSE = 7
#     SHIP = 8
#     TRUCK = 9


# if __name__ == "__main__":
#     print(CIFAR10Classes.DOG)
