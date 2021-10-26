# Image Classification Example

This example demonstrates configuration of an image classification model using PyTorch Lightning.

## Requirements

This example was tested using the following package versions:

  - hydra-zen==0.3.0
  - numpy==1.18.5
  - pytorch==1.9.1
  - torchvision==0.8.1
  - pytorch-lightning==1.4.9
  - torchmetrics==0.5.1

## Configs

See `hydra_zen_examples/image_classification/configs.py` to see experiment configurations.

## Experimentation

In the `experiment` directory run the following to train a ResNet-50 model on CIFAR-10:

```bash
$ python run.py
```

The task function and experiment configuration can be found in `run.py`,

```python
>>> from run import task_fn, Config
```

See `Experiment.ipynb` for an example of running experiments in the notebook.
