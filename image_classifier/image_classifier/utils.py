import random

import numpy as np
import torch
from torch.utils.data import random_split as _random_split
from torch.utils.data.dataset import Dataset


def random_split(
    dataset: Dataset,
    val_split: float = 0.1,
    train: bool = True,
    random_seed: int = 32,
) -> Dataset:
    g = torch.Generator().manual_seed(random_seed)
    nval = int(len(dataset) * val_split)
    ntrain = len(dataset) - nval
    train_data, val_data = _random_split(dataset, [ntrain, nval], g)
    if train:
        return train_data
    return val_data


def set_seed(random_seed) -> None:
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
