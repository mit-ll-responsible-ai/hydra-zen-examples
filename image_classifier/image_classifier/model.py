# Copyright (c) 2021 Massachusetts Institute of Technology
from types import FunctionType
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torchmetrics import Metric, MetricCollection
from typing_extensions import Literal


class ImageClassification(LightningModule):
    def __init__(
        self,
        *,
        model: Module,
        predict: Optional[Union[Module, Callable]] = None,
        optim: Optional[FunctionType] = None,
        lr_scheduler: Optional[FunctionType] = None,
        criterion: Union[Module, Callable, None] = None,
        metrics: Union[MetricCollection, Mapping, Sequence[Metric], None] = None,
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

        if (
            criterion is not None
            and isinstance(criterion, str)
            and hasattr(model, criterion)
        ):
            criterion = getattr(model, criterion)

        self.criterion = criterion

        # Metrics
        if metrics is not None:
            if isinstance(metrics, Mapping):
                metrics = MetricCollection(dict(metrics))
            elif isinstance(metrics, Metric):
                metrics = MetricCollection([metrics])
            elif isinstance(metrics, Sequence):
                metrics = MetricCollection(metrics)

        self.metrics = metrics

    def forward(self, x: Tensor) -> Tensor:
        """Forward method for Module."""
        return self.model(x)

    def predict(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None):
        x, y = batch
        return self.predictor(self(x))

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[Any]]:
        """Sets up optimizer and learning-rate scheduler.

        Optimizer: `torch.optim.SGD`

        Scheduler: `torch.optim.lr_scheduler.StepLR`
        """
        from hydra.utils import instantiate
        if self.optim.optimizer == "sgd":
            optim = torch.optim.SGD(self.parameters(), **self.optim.params)
        elif self.optim.optimizer == "adam":
            optim = torch.optim.Adam(self.parameters(), **self.optim.params)

        if self.lr_scheduler is not None:
            lr_scheduler = instantiate(self.lr_scheduler, optimizer=optim)
            return [optim], [lr_scheduler]

        return optim

    def step(
        self, batch: Tuple[Tensor, Tensor], batch_id: int, stage: str = "Train"
    ) -> Tensor:
        """Executes common step across training, validation and test."""
        x, y = batch

        # Pass through the model and get the loss
        logits = self(x)
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
        if self.metrics is not None and isinstance(self.metrics, MetricCollection):
            for key, metric in self.metrics.items():
                val = metric(pred_y, true_y)
                if isinstance(val, Tensor) and val.ndim == 0:
                    self.log(f"{stage}/{key}", val)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_id: int) -> Tensor:
        return self.step(batch, batch_id, stage="Train")

    def training_step_end(self, outputs: Union[Tensor, Dict]) -> torch.Tensor:
        if isinstance(outputs, dict):
            loss = outputs["loss"].mean()
            self.update_metrics(outputs, stage="Train")
            return loss
        return outputs

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_id: int) -> Tensor:
        return self.step(batch, batch_id, stage="Val")

    def validation_step_end(self, outputs: Union[Tensor, Dict]) -> Tensor:
        if isinstance(outputs, dict):
            loss = outputs["loss"].mean()
            self.update_metrics(outputs, stage="Val")
            return loss
        return outputs

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_id: int) -> Tensor:
        return self.step(batch, batch_id, stage="Test")

    def test_step_end(self, outputs: Union[Tensor, Dict]) -> Tensor:
        if isinstance(outputs, dict):
            loss = outputs["loss"].mean()
            self.update_metrics(outputs, stage="Test")
            return loss
        return outputs
