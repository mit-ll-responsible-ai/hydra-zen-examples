# Copyright (c) 2021 Massachusetts Institute of Technology
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import MetricCollection
from typing_extensions import Literal

from .resnet import resnet18, resnet50

# Types for documentation
PartialOptimizer = Callable[[Iterable], Optimizer]
PartialLRScheduler = Callable[[Optimizer], _LRScheduler]
Criterion = Callable[[Tensor, Tensor], Tensor]
Perturbation = Callable[[Module, Tensor, Tensor], Tuple[Tensor, Tensor]]
Predictor = Callable[[Tensor], Tensor]


class BaseImageClassification(LightningModule):
    model: Module

    def __init__(
        self,
        predict: Predictor,
        criterion: Criterion,
        optim: Optional[PartialOptimizer] = None,
        lr_scheduler: Optional[PartialLRScheduler] = None,
        metrics: Optional[MetricCollection] = None,
    ) -> None:
        """Initialization of Module

        Parameters
        ----------
        predict: Predictor
            The function to map the output of the model to predictions (e.g., `torch.softmax`)

        criterion: Criterion
            Criterion for calculating the loss. If `criterion` is a string the loss function
            is assumed to be an attribute of the model.

        optim: PartialOptimizer | None (default: None)
            Parameters for the optimizer. Current default optimizer is `SGD`.

        lr_scheduler: PartialLRScheduler | None (default: None)
            Parameters for StepLR. Current default scheduler is `StepLR`.

        metrics: MetricCollection | None (default: None)
            Define PyTorch Lightning `Metric`s.  This module utilizes `MetricCollection`.
        """
        super().__init__()
        self.predictor = predict
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion

        if metrics is not None:
            assert isinstance(metrics, MetricCollection)

        self.metrics = metrics

    def forward(self, x: Tensor) -> Tensor:
        """Forward method for Module."""
        return self.model(x)

    def predict(self, batch: Any, *args, **kwargs):
        return self.predictor(self(batch[0]))

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[List[Optimizer], List[Any]], None]:
        """Sets up optimizer and learning-rate scheduler."""
        if self.optim is not None:
            optim = self.optim(self.parameters())

            if self.lr_scheduler is not None:
                sched = [self.lr_scheduler(optim)]
                return [optim], sched

            return optim
        return None

    def step(
        self, batch: Tuple[Tensor, Tensor], stage: str = "Train"
    ) -> Dict[str, Tensor]:
        """Executes common step across training, validation and test."""
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.log(f"{stage}/Loss", loss)
        #
        # Normally metrics would be updated here (See below when needing to support DP)
        #
        return dict(loss=loss, pred=self.predictor(logits), target=y)

    def training_step(
        self, batch: Tuple[Tensor, Tensor], *args, **kwargs
    ) -> Dict[str, Tensor]:
        return self.step(batch, stage="Train")

    def validation_step(
        self, batch: Tuple[Tensor, Tensor], *args, **kwargs
    ) -> Dict[str, Tensor]:
        return self.step(batch, stage="Val")

    def test_step(
        self, batch: Tuple[Tensor, Tensor], *args, **kwargs
    ) -> Dict[str, Tensor]:
        return self.step(batch, stage="Test")

    # To support both DDP and DP we need to use `<stage>_step_end` to update metrics
    # See: https://github.com/PyTorchLightning/pytorch-lightning/pull/4494

    def training_step_end(self, outputs: Union[Tensor, Dict]) -> torch.Tensor:
        if isinstance(outputs, dict):
            loss = outputs["loss"].mean()
            self.update_metrics(outputs, stage="Train")
            return loss
        return outputs

    def validation_step_end(self, outputs: Union[Tensor, Dict]) -> Tensor:
        if isinstance(outputs, dict):
            loss = outputs["loss"].mean()
            self.update_metrics(outputs, stage="Val")
            return loss
        return outputs

    def test_step_end(self, outputs: Union[Tensor, Dict]) -> Tensor:
        if isinstance(outputs, dict):
            loss = outputs["loss"].mean()
            self.update_metrics(outputs, stage="Test")
            return loss
        return outputs

    def update_metrics(
        self, outputs: Dict[Literal["pred", "target"], Tensor], stage: str = "Train"
    ):
        if "pred" not in outputs or "target" not in outputs:
            raise TypeError(
                "ouputs dictionary expected contain 'pred' and 'target' keys."
            )

        pred_y: Tensor = outputs["pred"]
        true_y: Tensor = outputs["target"]
        if self.metrics is not None:
            assert isinstance(self.metrics, MetricCollection)
            for key, metric in self.metrics.items():
                val = metric(pred_y, true_y)
                if isinstance(val, Tensor) and val.ndim == 0:
                    self.log(f"{stage}/{key}", val)


class ResNet18Classifier(BaseImageClassification):
    def __init__(
        self,
        *,
        predict: Predictor,
        criterion: Criterion,
        optim: Optional[PartialOptimizer] = None,
        lr_scheduler: Optional[PartialLRScheduler] = None,
        metrics: Optional[MetricCollection] = None,
    ) -> None:
        super().__init__(
            predict, criterion, optim=optim, lr_scheduler=lr_scheduler, metrics=metrics
        )

        self.model = resnet18()


class ResNet50Classifier(BaseImageClassification):
    def __init__(
        self,
        *,
        predict: Predictor,
        criterion: Criterion,
        optim: Optional[PartialOptimizer] = None,
        lr_scheduler: Optional[PartialLRScheduler] = None,
        metrics: Optional[MetricCollection] = None,
    ) -> None:
        super().__init__(
            predict, criterion, optim=optim, lr_scheduler=lr_scheduler, metrics=metrics
        )

        self.model = resnet50()
