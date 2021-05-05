# Copyright (c) 2021 Massachusetts Institute of Technology
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


#################
# CIFAR10 Dataset
#################

# transforms.Compose takes a list of transforms
# - Each transform can be configured and appended to the list
@dataclass
class RandomCropConf:
    _target_: str = "torchvision.transforms.RandomCrop"
    size: int = 32
    padding: int = 4


@dataclass
class RandomHorizontalFlipConf:
    _target_: str = "torchvision.transforms.transforms.RandomHorizontalFlip"


@dataclass
class ColorJitterConf:
    _target_: str = "torchvision.transforms.transforms.ColorJitter"
    brightness: float = 0.25
    contrast: float = 0.25
    saturation: float = 0.25


@dataclass
class RandomRotationConf:
    _target_: str = "torchvision.transforms.transforms.RandomRotation"
    degrees: float = 2


@dataclass
class ToTensorConf:
    _target_: str = "torchvision.transforms.transforms.ToTensor"


@dataclass
class NormalizeConf:
    _target_: str = "torchvision.transforms.transforms.Normalize"
    mean: List[float] = field(default_factory=lambda: [0.4914, 0.4822, 0.4465])
    std: List[float] = field(default_factory=lambda: [0.2023, 0.1994, 0.2010])


@dataclass
class TrainTransformsConf:
    _target_: str = "torchvision.transforms.Compose"
    transforms: List[Any] = field(
        default_factory=lambda: [
            RandomCropConf,
            RandomHorizontalFlipConf,
            ColorJitterConf,
            RandomRotationConf,
            ToTensorConf,
            NormalizeConf,
        ]
    )


@dataclass
class TestTransformsConf:
    _target_: str = "torchvision.transforms.Compose"
    transforms: List[Any] = field(default_factory=lambda: [ToTensorConf, NormalizeConf])


# The base configuration for torchvision.dataset.CIFAR10
# - `transform` is left as None and defined later
@dataclass
class CIFAR10DatasetConf:
    _target_: str = "torchvision.datasets.CIFAR10"
    root: str = str(Path().home() / ".raiden")
    train: bool = True
    transform: Optional[Any] = None
    download: bool = True


# Uses the classmethod `LightningDataModule.from_datasets`
# - Each dataset is a dataclass with training or testing transforms
@dataclass
class CIFAR10ModuleConf:
    _target_: str = "pytorch_lightning.LightningDataModule.from_datasets"
    _convert_: str = "all"
    num_workers: int = 4
    batch_size: int = 256
    train_dataset: CIFAR10DatasetConf = CIFAR10DatasetConf(
        transform=TrainTransformsConf
    )
    val_dataset: CIFAR10DatasetConf = CIFAR10DatasetConf(
        transform=TestTransformsConf, train=False
    )
    test_dataset: CIFAR10DatasetConf = CIFAR10DatasetConf(
        transform=TestTransformsConf, train=False
    )


##################
# PyTorch Model
##################

# Build a base config for torchvision.models configurations to implement
# - This allows Hydra to recognize the base type for each model during type validation
# - `builds_base` ensures the configuration inherits the base configuration
@dataclass
class TorchModuleConf:
    ...


@dataclass
class ResNet18Conf(TorchModuleConf):
    _target_: str = "image_classifier.resnet.resnet18"


@dataclass
class ResNet50Conf(TorchModuleConf):
    _target_: str = "image_classifier.resnet.resnet50"


# ####################################
# # PyTorch Optimizer and LR Scheduler
# ####################################

# Build a base config for torch.optim configurations to implement
# - This allows Hydra to recognize the base type for each model during type validation
# - `builds_base` ensures the configuration inherits the base configuration
@dataclass
class OptimizerConf:
    name: str = "None"


@dataclass
class SGDConf(OptimizerConf):
    name: str = "sgd"
    params: Dict = field(default_factory=lambda: dict(lr=0.1, momentum=0.9))


@dataclass
class AdamConf(OptimizerConf):
    name: str = "adam"
    params: Dict = field(
        default_factory=lambda: dict(
            lr=0.1,
        )
    )


@dataclass
class StepLRConf:
    step_size: float = 50
    gamma: float = 0.1


##########################
# PyTorch Lightning Module
##########################
@dataclass
class SoftmaxConf:
    _target_: str = "torch.nn.Softmax"
    dim: int = 1


@dataclass
class CrossEntropyLossConf:
    _target_: str = "torch.nn.CrossEntropyLoss"


@dataclass
class AccuracyConf:
    _target_: str = "torchmetrics.Accuracy"


@dataclass
class ImageClassificationConf:
    _target_: str = "image_classifier.model.ImageClassification"
    _convert_: str = "all"
    model: Any = MISSING
    optim: Optional[Any] = None
    predict: SoftmaxConf = SoftmaxConf
    criterion: CrossEntropyLossConf = CrossEntropyLossConf
    lr_scheduler: StepLRConf = StepLRConf
    metrics: Dict = field(default_factory=lambda: dict(Accuracy=AccuracyConf))


###################
# Lightning Trainer
###################
@dataclass
class ModelCheckpointConf:
    _target_: str = "pytorch_lightning.callbacks.ModelCheckpoint"
    mode: str = "min"


@dataclass
class TrainerConf:
    _target_: str = "pytorch_lightning.Trainer"
    callbacks: List[Any] = field(default_factory=lambda: [ModelCheckpointConf])
    accelerator: str = "ddp"
    num_nodes: int = 1
    gpus: int = 1
    max_epochs: int = 150
    default_root_dir: str = "."
    check_val_every_n_epoch: int = 1


def register_configs():
    """
    Register Configs in Hydra's Config Store

    This allows user to override configs with "GROUP=NAME" using Hydra's Command Line Interface
    or by using hydra-zen's `hydra_run` or `hydra_multirun` commands.

    For example using Hydra's CLI:
    $ python run_file.py optim=sgd

    or the equivalent command using `hydra_run`:
    >> hydra_run(config, task_function, overrides=["optim=sgd"])
    """
    cs = ConfigStore.instance()
    cs.store(
        group="experiment/lightning_data_module", name="cifar10", node=CIFAR10ModuleConf
    )

    cs.store(
        group="experiment/lightning_module",
        name="image_classification",
        node=ImageClassificationConf,
    )

    cs.store(group="experiment/lightning_trainer", name="trainer", node=TrainerConf)

    cs.store(group="model", name="resnet18", node=ResNet18Conf)

    cs.store(group="model", name="resnet50", node=ResNet50Conf)

    cs.store(group="optim", name="sgd", node=SGDConf)

    cs.store(group="optim", name="adam", node=AdamConf)
