import torch
from torch.utils.data import Subset
from .shape_mol_dataset import ShapeMolDataset
from .shape_data import ShapeDataset
import pdb
import numpy as np

def get_dataset(config, *args, **kwargs):
    name = config.name
    root = config.path
    if name == 'shapemol':
        dataset = ShapeMolDataset(config, *args, **kwargs)
    elif name == 'shape':
        dataset = ShapeDataset(config, *args, **kwargs)
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)

    if 'split' in config and config.dataset == 'moses2':
        subsets = {}
        dataset._connect_db()
        np.random.seed(2023)
        random_valid_indices = np.random.choice(dataset.size, 1000).tolist()
        random_train_indices = [idx for idx in range(dataset.size) if idx not in random_valid_indices]
        subsets['valid'] = Subset(dataset, indices=random_valid_indices)
        subsets['train'] = Subset(dataset, indices=random_train_indices)
        return dataset, subsets
    else:
        return dataset
