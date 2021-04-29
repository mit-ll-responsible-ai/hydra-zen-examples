# Image Classification Example

Check configuration:

```bash
python -m image_classifier.main --info
```

Run a regular Hydra Run:

```bash
PYTHONPATH=$PWD python -m image_classifier.main model=resnet18 optim=sgd optim.params.lr=0.05 experiment.lightning_data_module.batch_size=512 experiment.lightning_trainer.max_epochs=1
```

Run a Hydra Multirun:

```bash
PYTHONPATH=$PWD python -m image_classifier.main model=resnet18,resnet50 optim=sgd,adam experiment.lightning_data_module.batch_size=256,512 experiment.lightning_trainer.max_epochs=1 --multirun
```