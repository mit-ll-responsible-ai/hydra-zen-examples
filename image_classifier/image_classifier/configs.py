# Copyright (c) 2021 Massachusetts Institute of Technology
import torch
from hydra.core.config_store import ConfigStore
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.datamodule import LightningDataModule
from torchmetrics import Accuracy, MetricCollection
from torchvision import datasets, transforms

from hydra_zen import MISSING, builds, make_custom_builds_fn

from .model import ImageClassification
from .resnet import resnet18, resnet50
from .utils import random_split

###############
# Custom Builds
###############
pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

#################
# CIFAR10 Dataset
#################
CIFARNormalize = builds(
    transforms.Normalize,
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010],
)

# transforms.Compose takes a list of transforms
# - Each transform can be configured and appended to the list
TrainTransforms = builds(
    transforms.Compose,
    transforms=[
        builds(transforms.RandomCrop, size=32, padding=4),
        builds(transforms.RandomHorizontalFlip),
        builds(transforms.ColorJitter, brightness=0.25, contrast=0.25, saturation=0.25),
        builds(transforms.RandomRotation, degrees=2),
        builds(transforms.ToTensor),
        CIFARNormalize,
    ],
)

TestTransforms = builds(
    transforms.Compose,
    transforms=[builds(transforms.ToTensor), CIFARNormalize],
)

# Define a function to split the dataset into train and validation sets
SplitDataset = builds(random_split, dataset=MISSING, populate_full_signature=True)

# The base configuration for torchvision.dataset.CIFAR10
# - `transform` is left as None and defined later
CIFAR10 = builds(
    datasets.CIFAR10,
    root=MISSING,
    train=True,
    transform=None,
    download=True,
)

# Uses the classmethod `LightningDataModule.from_datasets`
# - Each dataset is a dataclass with training or testing transforms
CIFAR10DataModule = builds(
    LightningDataModule.from_datasets,
    num_workers=4,
    batch_size=256,
    train_dataset=SplitDataset(
        dataset=CIFAR10(root="${...root}", transform=TrainTransforms), train=True
    ),
    val_dataset=SplitDataset(
        dataset=CIFAR10(root="${...root}", transform=TestTransforms, train=True),
        train=False,
    ),
    test_dataset=CIFAR10(root="${..root}", transform=TestTransforms, train=False),
    zen_meta=dict(root="${data_dir}"),
)

##################
# PyTorch Model
##################
ResNet18 = builds(resnet18)
ResNet50 = builds(resnet50)

####################################
# PyTorch Optimizer and LR Scheduler
####################################
SGD = pbuilds(torch.optim.SGD, lr=0.1, momentum=0.9)
Adam = pbuilds(torch.optim.Adam, lr=0.1)
StepLR = pbuilds(torch.optim.lr_scheduler.StepLR, step_size=50, gamma=0.1)

##########################
# PyTorch Lightning Module
##########################
ImageClassificationConf = builds(
    ImageClassification,
    model=MISSING,
    optim=None,
    predict=builds(torch.nn.Softmax, dim=1),
    criterion=builds(torch.nn.CrossEntropyLoss),
    lr_scheduler=StepLR,
    metrics=builds(
        MetricCollection,
        builds(dict, Accuracy=builds(Accuracy)),
        hydra_convert="all",
    ),
)

###################
# Lightning Trainer
###################
TrainerConf = builds(
    Trainer,
    callbacks=[builds(ModelCheckpoint, mode="min")],  # easily build a list of callbacks
    accelerator="ddp",
    num_nodes=1,
    gpus=builds(torch.cuda.device_count),
    max_epochs=150,
    populate_full_signature=True,
)


"""
Register Configs in Hydra's Config Store

This allows user to override configs with "GROUP=NAME" using Hydra's Command Line Interface
or by using hydra-zen's `launch`.

For example using Hydra's CLI:
$ python run_file.py optim=sgd

or the equivalent command using `hydra_run`:
>> launch(config, task_function, overrides=["optim=sgd"])
"""
cs = ConfigStore.instance()

cs.store(group="data", name="cifar10", node=CIFAR10DataModule)
cs.store(group="lightning", name="image_classification", node=ImageClassificationConf)
cs.store(group="trainer", name="trainer", node=TrainerConf)
cs.store(group="model", name="resnet18", node=ResNet18)
cs.store(group="model", name="resnet50", node=ResNet50)
cs.store(group="optim", name="sgd", node=SGD)
cs.store(group="optim", name="adam", node=Adam)
