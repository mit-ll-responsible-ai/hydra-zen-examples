# Image Classification Example

This example demonstrates configuration of an image classification model using PyTorch Lightning.

## Configs

See `hydra_zen_examples/image_classification/configs.py` to see experiment configurations.

## Experimentation

In the `experiment` directory run the following to train or test a ResNet-50 model on CIFAR-10:

```python
python run.py
```

Within `run.py` the main experiment configuration is defined by `Config`,

```python
# Experiment Configs
# - Replaces Hydra's config.yaml
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
```

and the task function is defined by `task_fn`,

```python
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
```

See `Experiment.ipynb` for an example of running experiments in the notebook.
