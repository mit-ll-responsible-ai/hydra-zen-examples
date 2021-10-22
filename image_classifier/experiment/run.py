# Copyright (c) 2021 Massachusetts Institute of Technology
from pathlib import Path

import hydra
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.nn import Module

from hydra_zen import MISSING, make_config

# If the repo isn't in the PYTHONPATH let's load it
try:
    import hydra_zen_example.image_classifier
except ImportError:
    import sys

    path = (Path.cwd() / "..").absolute()
    sys.path.insert(0, str(path))
finally:
    from hydra_zen_example.image_classifier.configs import TrainerConf
    from hydra_zen_example.image_classifier.utils import set_seed


# Experiment Configs
# - Replaces config.yaml
Config = make_config(
    #
    # Experiment Defaults: See https://hydra.cc/docs/next/advanced/defaults_list
    defaults=[
        "_self_",  # See https://hydra.cc/docs/upgrades/1.0_to_1.1/default_composition_order
        {"data": "cifar10"},
        {"model": "resnet18"},
        {"model/optim": "sgd"},
    ],
    #
    # Experiment Modules
    data=MISSING,
    model=MISSING,
    trainer=TrainerConf,
    #
    # Experiment Constants
    data_dir=str(Path().home() / ".data"),
    random_seed=928,
    testing=False,
    ckpt_path=None,
)


cs = ConfigStore.instance()

cs.store(name="config", node=Config)

# Experiment Task Function
def task_fn(cfg: DictConfig) -> Module:
    # Set seed BEFORE instantiating anything
    set_seed(cfg.random_seed)

    # Data and Lightning Modules
    data = instantiate(cfg.data)
    pl_module = instantiate(cfg.model)

    # Load a checkpoint if defined
    if cfg.ckpt_path is not None:
        ckpt_data = torch.load(cfg.ckpt_path)
        assert "state_dict" in ckpt_data
        pl_module.load_state_dict(ckpt_data["state_dict"])

    # The PL Trainer
    trainer = instantiate(cfg.trainer)

    # Set training or testing mode
    if cfg.testing:
        trainer.test(pl_module, datamodule=data)
    else:
        trainer.fit(pl_module, datamodule=data)

    return pl_module


@hydra.main(config_path=None, config_name="config")
def main(cfg: DictConfig):
    return task_fn(cfg)


if __name__ == "__main__":
    main()
