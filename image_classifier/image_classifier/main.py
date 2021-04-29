# Copyright (c) 2021 Massachusetts Institute of Technology
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    data = instantiate(cfg.experiment.lightning_data_module)
    model = instantiate(cfg.experiment.lightning_module)
    trainer = instantiate(cfg.experiment.lightning_trainer)
    trainer.fit(model, datamodule=data)
    return model

if __name__ == "__main__":
    main()
