from phobos.dataset import getDistLoaders

import os
import random
import glob

import numpy as np

import torch.utils.data as data




def get_train_val_metadata(args):
    """Get training and validation samples.
    Parameters
    ----------
    args : basecamp.grain.grain.Grain
        Argument to create training and validation samples.
    Returns
    -------
    tuple
        Tuple of train and validation samples.
    """
    train_metadata = list(range(100))
    val_metadata = list(range(101,150))

    return train_metadata, val_metadata


class DummyPreloader(data.Dataset):
    """Dummy dataset preloader which generates one sample from the dataset.

    Your problem might require more logic in preloader and some other functions
    to load, preprocess and handle image, mask/vector/geojson, etc.
    Parameters
    ----------
    metadata : list
        Samples.
    args : basecamp.grain.grain.Grain
        Argument to create preloader.
    Attributes
    ----------
    samples : list
        Samples for this preloader.
    """
    def __init__(self, metadata, args=None):
        random.shuffle(metadata)
        self.samples = metadata
        self.args = args

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index
                   of the target class.
        """
        return np.zeros((3,32,32),dtype=np.float32), np.zeros((1,32,32),dtype=np.float32)

    def __len__(self):
        return len(self.samples)


def get_dataloaders(args):
    """Get train and val dataloaders.
    Given user arguments, loads dataset metadata
    defines a preloader and returns train and val dataloaders.
    Parameters
    ----------
    args : basecamp.grain.grain.Grain
        Dictionary of argsions/flags
    Returns
    -------
    (DataLoader, DataLoader)
        returns train and val dataloaders
    """
    train_samples, val_samples = get_train_val_metadata(args)
    print('train samples : ', len(train_samples))
    print('val samples : ', len(val_samples))

    train_dataset = DummyPreloader(train_samples, args)
    val_dataset = DummyPreloader(val_samples, args)

    train_loader, val_loader = getDistLoaders(
        train_dataset,
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed=args.distributed,
        distributed_val=args.distributed_val
        )
    
    return train_loader, val_loader
