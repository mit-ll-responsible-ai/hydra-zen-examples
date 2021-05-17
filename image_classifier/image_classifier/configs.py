# Copyright (c) 2021 Massachusetts Institute of Technology
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

import torch
from hydra.core.config_store import ConfigStore
from hydra_zen import builds
from omegaconf import MISSING
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import Module
from torch.optim import Optimizer
from torchmetrics import Accuracy
from torchvision import datasets, transforms

from .model import ImageClassification
from .resnet import resnet18, resnet50

#################
# CIFAR10 Dataset
#################

# transforms.Compose takes a list of transforms
# - Each transform can be configured and appended to the list
TrainTransformsConf = builds(
    transforms.Compose,
    transforms=[
        builds(transforms.RandomCrop, size=32, padding=4),
        builds(transforms.RandomHorizontalFlip),
        builds(transforms.ColorJitter, brightness=0.25, contrast=0.25, saturation=0.25),
        builds(transforms.RandomRotation, degrees=2),
        builds(transforms.ToTensor),
        builds(
            transforms.Normalize,
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        ),
    ],
)

TestTransformsConf = builds(
    transforms.Compose,
    transforms=[
        builds(transforms.ToTensor),
        builds(
            transforms.Normalize,
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        ),
    ],
)

# The base configuration for torchvision.dataset.CIFAR10
# - `transform` is left as None and defined later
CIFAR10DatasetConf = builds(
    datasets.CIFAR10,
    root=str(Path().home() / ".raiden"),
    train=True,
    transform=None,
    download=True,
)

LightningDataModuleConf = builds(LightningDataModule)

# Uses the classmethod `LightningDataModule.from_datasets`
# - Each dataset is a dataclass with training or testing transforms
CIFAR10ModuleConf = builds(
    LightningDataModule.from_datasets,
    num_workers=4,
    batch_size=256,
    train_dataset=CIFAR10DatasetConf(transform=TrainTransformsConf),
    val_dataset=CIFAR10DatasetConf(transform=TestTransformsConf, train=False),
    test_dataset=CIFAR10DatasetConf(transform=TestTransformsConf, train=False),
    builds_bases=(LightningDataModuleConf,),
)


##################
# PyTorch Model
##################

# Build a base config for torchvision.models configurations to implement
# - This allows Hydra to recognize the base type for each model during type validation
# - `builds_base` ensures the configuration inherits the base configuration
TorchModuleConf = builds(Module)
ResNet18Conf = builds(resnet18, builds_bases=(TorchModuleConf,))
ResNet50Conf = builds(resnet50, builds_bases=(TorchModuleConf,))

####################################
# PyTorch Optimizer and LR Scheduler
####################################

# Build a base config for torch.optim configurations to implement
# - This allows Hydra to recognize the base type for each model during type validation
# - `builds_base` ensures the configuration inherits the base configuration
OptimizerConf = builds(Optimizer)
SGDConf = builds(
    torch.optim.SGD,
    lr=0.1,
    momentum=0.9,
    builds_bases=(OptimizerConf,),
    hydra_partial=True,
)
AdamConf = builds(
    torch.optim.Adam, lr=0.1, builds_bases=(OptimizerConf,), hydra_partial=True
)

StepLRConf = builds(
    torch.optim.lr_scheduler.StepLR, step_size=50, gamma=0.1, hydra_partial=True
)

##########################
# PyTorch Lightning Module
##########################
LightningModuleConf = builds(LightningModule)

ImageClassificationConf = builds(
    ImageClassification,
    model=ResNet18Conf,
    optim=SGDConf,
    predict=builds(torch.nn.Softmax, dim=1),
    criterion=builds(torch.nn.CrossEntropyLoss),
    lr_scheduler=StepLRConf,
    metrics=dict(Accuracy=builds(Accuracy)),
    builds_bases=(LightningModuleConf,),
)


###################
# Lightning Trainer
###################
TrainerConf = builds(
    Trainer,
    callbacks=[builds(ModelCheckpoint, mode="min")],  # easily build a list of callbacks
    accelerator="ddp",
    num_nodes=1,
    gpus=1,
    max_epochs=150,
    default_root_dir=".",
    check_val_every_n_epoch=1,
)


##############################
# Experiment Configs
# - Replaces config.yaml
##############################
@dataclass
class ExperimentConf:
    lightning_data_module: LightningDataModuleConf = MISSING
    lightning_module: LightningModuleConf = ImageClassificationConf(
        model="${model}", optim="${optim}"
    )
    lightning_trainer: TrainerConf = TrainerConf


@dataclass
class Config:
    defaults: List[Any] = field(
        default_factory=lambda: [
            {"experiment/lightning_data_module": "cifar10"},
            {"model": "resnet18"},
            {"optim": "sgd"},
        ]
    )
    experiment: ExperimentConf = ExperimentConf()
    model: TorchModuleConf = MISSING
    optim: OptimizerConf = MISSING


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

cs.store(name="config", node=Config)

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
