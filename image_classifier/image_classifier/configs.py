# Copyright (c) 2021 Massachusetts Institute of Technology
from typing import Any

import torch
from hydra.core.config_store import ConfigStore
from hydra_zen import builds
from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import Accuracy
from torchvision import transforms, datasets

from .model import ImageClassification
from .resnet import resnet50, resnet18


#################
# CIFAR10 Dataset
#################
TrainTransformsConf = builds(
    transforms.Compose,
    transforms=[
        builds(transforms.RandomCrop, size=32, padding=4),
        builds(transforms.RandomHorizontalFlip),
        builds(transforms.ColorJitter, brightness=0.25, contrast=0.25, saturation=0.25),
        builds(transforms.RandomRotation, degrees=2),
        builds(transforms.ToTensor),
    ],
)

TestTransformsConf = builds(transforms.ToTensor)

NormalizerConf = builds(
    transforms.Normalize, mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
)

# The TorchVision Dataset
CIFAR10DatasetConf = builds(
    datasets.CIFAR10,
    root="${oc.env:HOME}/.raiden",
    train=True,
    transform=None,
    download=True,
)


CIFAR10ModuleConf = builds(
    LightningDataModule.from_datasets,
    num_workers=4,
    batch_size=256,
    train_dataset=CIFAR10DatasetConf(transform=TrainTransformsConf),
    val_dataset=CIFAR10DatasetConf(transform=TestTransformsConf),
    test_dataset=CIFAR10DatasetConf(transform=TestTransformsConf),
)


##################
# Image Classifier
##################
ResNet18Conf = builds(resnet18)
ResNet50Conf = builds(resnet50)
SGDConf = builds(torch.optim.SGD, lr=0.1, momentum=0.9, hydra_partial=True)
AdamConf = builds(torch.optim.Adam, lr=0.1, hydra_partial=True)
StepLRConf = builds(
    torch.optim.lr_scheduler.StepLR, step_size=50, gamma=0.1, hydra_partial=True
)

ImageClassificationConf = builds(
    ImageClassification,
    model=None,
    optim=None,
    normalizer=NormalizerConf,
    predict=builds(torch.nn.Softmax, dim=1),
    criterion=builds(torch.nn.CrossEntropyLoss),
    lr_scheduler=StepLRConf,
    metrics=dict(Accuracy=builds(Accuracy)),
)


###################
# Lightning Trainer
###################
TrainerConf = builds(
    Trainer,
    callbacks=[builds(ModelCheckpoint, mode="min")],
    accelerator="ddp",
    num_nodes=1,
    gpus=1,
    max_epochs=150,
    default_root_dir=".",
    check_val_every_n_epoch=1,
)

##########################################
# Register Configs in Hydra's Config Store
##########################################
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


__all__ = [k for k in globals().keys() if "Conf" in k]
