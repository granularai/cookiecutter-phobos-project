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
    # geojson and image tuple 
    train_metadata = glob.glob(os.path.join(args.dataset_dir, "train/*.geojson"))
    val_metadata = glob.glob(os.path.join(args.dataset_dir, "test/*.geojson")) 

    return train_metadata, val_metadata


def loader(geojson_path):
    """Load a tile.
    Parameters
    ----------
    geojson_path : str
        GeoJSON path.
    Returns
    -------
    tuple
        (rgb, label_mask).
    """
    r = rio.open(geojson_path.replace('geojson', 'png'))
    gdf = gpd.read_file(geojson_path)
    
    mask = np.zeros(r.shape)
    for i in range(gdf.shape[0]):
        xs, ys = gdf.iloc[i].geometry.exterior.coords.xy
        rc = rio.transform.rowcol(r.transform, xs, ys)
        poly = np.asarray(list(zip(rc[0], rc[1])))
        rr, cc = polygon(poly[:,0], poly[:,1], mask.shape)
        mask[rr,cc] = 1

    return np.asarray([r.read()]), mask


class RarePlanesPreloader(data.Dataset):
    """RarePlanes Preloader.
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
        geojson_path = self.samples[index]
        tile, label = loader(geojson_path)
        return tile, label

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

    train_dataset = RarePlanesPreloader(train_samples, args)
    val_dataset = RarePlanesPreloader(val_samples, args)

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   pin_memory=True)
    val_loader = data.DataLoader(val_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 pin_memory=True)
    return train_loader, val_loader
