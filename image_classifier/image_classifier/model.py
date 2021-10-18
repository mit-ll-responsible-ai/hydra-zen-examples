# Copyright (c) 2021 Massachusetts Institute of Technology
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import Metric, MetricCollection
from typing_extensions import Literal

PartialOptimizer = Callable[[Iterable], Optimizer]
PartialLRScheduler = Callable[[Optimizer], _LRScheduler]
Criterion = Callable[[Tensor, Tensor], Tensor]
Perturbation = Callable[[Module, Tensor, Tensor], Tuple[Tensor, Tensor]]
Predictor = Callable[[Optional[Tensor]], Tensor]


class ImageClassification(LightningModule):
    def __init__(
        self,
        *,
        model: Module,
        predict: Predictor,
        criterion: Criterion,
        optim: Optional[PartialOptimizer] = None,
        lr_scheduler: Optional[PartialLRScheduler] = None,
        metrics: Optional[MetricCollection] = None,
        **kwargs: Any,
    ) -> None:
        """Initialization of Module

        Parameters
        ----------
        model: Module
            A PyTorch Module (e.g., Resnet)

        predict: Optional[Union[Module, Callable]] (default: None)
            The function to map the output of the model to predictions (e.g., `torch.softmax`)

        criterion: Optional[Union[Module, Callable, str]] (default: None)
            Criterion for calculating the loss. If `criterion` is a string the loss function
            is assumed to be an attribute of the model.

        optim: Optional[Dict] (default: None)
            Parameters for the optimizer. Current default optimizer is `SGD`.

        lr_scheduler: Optional[Dict] (default: None)
            Parameters for StepLR. Current default scheduler is `StepLR`.

        metrics: Optional[Union[MetricCollection, Mapping, List[Metric]]] (default: None)
            Define PyTorch Lightning `Metric`s.  This module utilizes `MetricCollection`.
        """
        super().__init__(**kwargs)

        # Load model
        self.model = model
        self.predictor = predict
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.metrics = metrics

    def forward(self, x: Tensor) -> Tensor:
        """Forward method for Module."""
        return self.model(x)

    def predict(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None):
        x, y = batch
        return self.predictor(self(x))

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[List[Optimizer], List[Any]], None]:
        """Sets up optimizer and learning-rate scheduler.

        Optimizer: `torch.optim.SGD`

        Scheduler: `torch.optim.lr_scheduler.StepLR`
        """
        if self.optim is not None:
            optim = self.optim(self.parameters())

            if self.lr_scheduler is not None:
                sched = [self.lr_scheduler(optim)]
                return [optim], sched

            return optim
        return None

    def step(
        self, batch: Tuple[Tensor, Tensor], batch_id: int, stage: str = "Train"
    ) -> Dict[str, Tensor]:
        """Executes common step across training, validation and test."""
        x, y = batch

        # Pass through the model and get the loss
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.log(f"{stage}/Loss", loss)
        return dict(loss=loss, pred=self.predictor(logits), target=y)

    def update_metrics(
        self, outputs: Dict[Literal["pred", "target"], Tensor], stage: str = "Train"
    ):
        # TODO: Metrics are updated here because it won't work to update in "step" in "dp" mode
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

    def training_step(
        self, batch: Tuple[Tensor, Tensor], batch_id: int
    ) -> Dict[str, Tensor]:
        return self.step(batch, batch_id, stage="Train")

    def training_step_end(self, outputs: Union[Tensor, Dict]) -> torch.Tensor:
        if isinstance(outputs, dict):
            loss = outputs["loss"].mean()
            self.update_metrics(outputs, stage="Train")
            return loss
        return outputs

    def validation_step(
        self, batch: Tuple[Tensor, Tensor], batch_id: int
    ) -> Dict[str, Tensor]:
        return self.step(batch, batch_id, stage="Val")

    def validation_step_end(self, outputs: Union[Tensor, Dict]) -> Tensor:
        if isinstance(outputs, dict):
            loss = outputs["loss"].mean()
            self.update_metrics(outputs, stage="Val")
            return loss
        return outputs

    def test_step(
        self, batch: Tuple[Tensor, Tensor], batch_id: int
    ) -> Dict[str, Tensor]:
        return self.step(batch, batch_id, stage="Test")

    def test_step_end(self, outputs: Union[Tensor, Dict]) -> Tensor:
        if isinstance(outputs, dict):
            loss = outputs["loss"].mean()
            self.update_metrics(outputs, stage="Test")
            return loss
        return outputs
