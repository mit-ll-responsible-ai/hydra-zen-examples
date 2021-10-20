# Copyright (c) 2021 Massachusetts Institute of Technology
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import DictConfig

from hydra_zen import MISSING, make_config


try:
    import image_classifier
except ImportError:
    import sys
    path = (Path.cwd() / "..").absolute()
    sys.path.insert(0, str(path))
finally:
    from image_classifier.configs import ImageClassificationConf, TrainerConf
    from image_classifier.utils import set_seed


# Experiment Configs
# - Replaces config.yaml
Config = make_config(
    #
    # Experiment Defaults: See https://hydra.cc/docs/next/advanced/defaults_list
    defaults=[
        "_self_",  # See https://hydra.cc/docs/upgrades/1.0_to_1.1/default_composition_order
        {"data": "cifar10"},
        {"model": "resnet18"},
        {"optim": "sgd"},
    ],
    #
    # Experiment Modules
    data=MISSING,
    model=MISSING,
    optim=MISSING,
    lightning=ImageClassificationConf,
    trainer=TrainerConf,
    #
    # Experiment Constants
    data_dir=str(Path().home() / ".data"),
    random_seed=928,
    testing=False,
)


cs = ConfigStore.instance()

cs.store(name="config", node=Config)


def task_fn(cfg: DictConfig):
    set_seed(cfg.random_seed)
    data = instantiate(cfg.data)
    model = instantiate(cfg.model)
    optim = instantiate(cfg.optim)
    pl_module = instantiate(cfg.lightning, model=model, optim=optim)
    trainer = instantiate(cfg.trainer)
    if cfg.testing:
        trainer.test(pl_module, datamodule=data)
    else:
        trainer.fit(pl_module, datamodule=data)
    return model


@hydra.main(config_path=None, config_name="config")
def main(cfg: DictConfig):
    return task_fn(cfg)


if __name__ == "__main__":
    main()
