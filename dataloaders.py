import os
import time
import random
from typing import Dict 

import pandas as pd
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize

from utils.file_io import load_image, h5_multi_load
from imageops.preprocess import rescale_01, clip_min_max


def seed_everything(seed=0) -> None:
    """seed_everything

    Args:
        seed (int, optional): random seed. Defaults to 0.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)

    return


class ResizeImage(object):
    def __init__(self, output_size, rescale=False):

        self.output_size = output_size
        self.rescale = rescale

    def __call__(self, img):

        resized = resize(img, self.output_size)

        if self.rescale:
            resized = clip_min_max(resized)
            rescaled = (resized - np.min(resized)) / \
                (np.max(resized) - np.min(resized))
            return rescaled

        return resized


class SliceDataset(Dataset):
    def __init__(self, data_dir, annotations_file, multi_channel=None, transforms=None):

        # where the CT scan images are
        self.data_dir = data_dir 

        # set transforms
        self.transforms = transforms

        labels_ = np.load(annotations_file)
        
        # save the number of channels for the output 
        self.multi_channel = multi_channel

        self.images = labels_['img_id']
        self.labels = labels_['label']

    def __len__(self):  
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir,  "%s.hdf5" % (self.images[idx]))

        # load the image
        pair = h5_multi_load(img_path)
        image = pair['img']
        label = pair['label']

        if self.transforms:
            if callable(self.transforms): 
                image = self.transforms(image)
            else: 
                for t in self.transforms: 
                    image = t(image)

        if self.multi_channel is not None: 
            stacked = np.stack([image for _ in range(self.multi_channel)], axis=0)
            return stacked, label

        return image, label


class SliceResamplingDataset(Dataset):
    def __init__(self, data_dir, annotations_file, transforms=None, multi_channel=None):

        # where the CT scan images are
        self.data_dir = data_dir 

        # set transforms
        self.transforms = transforms

        # save the number of channels for the output
        self.multi_channel = multi_channel

        # read in the file locations + labels
        csv = pd.read_csv(annotations_file, header=None, names=['img_id', 'label'])

        self.images = list(csv['img_id'])
        self.labels = list(csv['label'])

    def __len__(self):  
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir,  "%d.mha" % (self.images[idx]))

        # load the image
        image, _ = load_image(img_path, return_spacing=True)

        # obtain label
        label = self.labels[idx]

        # random index -> random axial slice
        lower_lim = min(int(image.shape[0] * 2. / 5), 110)
        upper_lim = max(int(image.shape[0] * 4. / 5), image.shape[0] - 100)
        ax_slice = self._get_random_idx(lower_lim, upper_lim)
        image = image[ax_slice, :,:]

        if self.transforms:
            if callable(self.transforms): 
                image = self.transforms(image)
            else: 
                for t in self.transforms: 
                    image = t(image)

        if self.multi_channel is not None:
            stacked = np.stack(
                [image for _ in range(self.multi_channel)], axis=0)
            return stacked, label

        return image, label
    
    def _get_random_idx(self, lower_lim, upper_lim): 

        ax_idx = torch.randint(lower_lim, upper_lim, (1,))

        return ax_idx


def get_resampling_dataloader(data_dir, annotations: Dict, batch_sz, num_workers=0, **kwargs):

    # add random seed
    seed = kwargs.get('random_seed', 682)
    seed_everything(seed)

    # resizing to protopnet input size
    resizer = ResizeImage((224, 224), rescale=True)
    transforms = [resizer]

    # get datasets
    train_ds = SliceResamplingDataset(data_dir, annotations['train'], transforms=transforms)
    val_ds = SliceResamplingDataset(data_dir, annotations['val'], transforms=transforms)
    test_ds = SliceResamplingDataset(data_dir, annotations['test'], transforms=transforms)

    # insert into dataloaders
    trainloader = DataLoader(
        train_ds, batch_size=batch_sz, num_workers=num_workers, shuffle=True
    )

    validloader = DataLoader(
        val_ds, batch_size=batch_sz, num_workers=num_workers, shuffle=True
    )

    testloader = DataLoader(
        test_ds, batch_size=batch_sz, num_workers=num_workers, shuffle=True
    )
    
    print('done getting dataloaders')
    return trainloader, validloader, testloader


def get_slice_dataloaders(data_dir, annotations: Dict, batch_sz, num_workers=0, **kwargs):

    # add random seed
    seed = kwargs.get('random_seed', 682)
    seed_everything(seed)

    # resizing to protopnet input size
    resizer = ResizeImage((224, 224), rescale=True)
    transforms = [resizer]

    # get datasets
    train_ds = SliceDataset(data_dir, annotations['train'], transforms=transforms)
    val_ds = SliceDataset(data_dir, annotations['val'], transforms=transforms)
    test_ds = SliceDataset(data_dir, annotations['test'], transforms=transforms)

    # insert into dataloaders
    trainloader = DataLoader(
        train_ds, batch_size=batch_sz, num_workers=num_workers, shuffle=True
    )

    validloader = DataLoader(
        val_ds, batch_size=batch_sz, num_workers=num_workers, shuffle=True
    )

    testloader = DataLoader(
        test_ds, batch_size=batch_sz, num_workers=num_workers, shuffle=True
    )
    
    print('done getting dataloaders')
    return trainloader, validloader, testloader


def get_slice_dataloaders_proto(data_dir, annotations: Dict, batch_sz, num_workers=0, **kwargs):

    # add random seed
    seed = kwargs.get('random_seed', 682)
    seed_everything(seed)

    # resizing to protopnet input size
    resizer = ResizeImage((224, 224), rescale=True)
    transforms = [resizer]

    # get datasets
    train_ds = SliceDataset(data_dir, annotations['train'], transforms=transforms, multi_channel=3)
    val_ds = SliceDataset(data_dir, annotations['val'], transforms=transforms, multi_channel=3)
    test_ds = SliceDataset(data_dir, annotations['test'], transforms=transforms, multi_channel=3)

    # insert into dataloaders
    trainloader = DataLoader(
        train_ds, batch_size=batch_sz, num_workers=num_workers, shuffle=True
    )

    validloader = DataLoader(
        val_ds, batch_size=batch_sz, num_workers=num_workers, shuffle=True
    )

    testloader = DataLoader(
        test_ds, batch_size=batch_sz, num_workers=num_workers, shuffle=True
    )
    
    print('done getting dataloaders')
    return trainloader, validloader, testloader


def get_resampling_dataloaders_proto(data_dir, annotations: Dict, batch_sz, num_workers=0, **kwargs):

    # add random seed
    seed = kwargs.get('random_seed', 682)
    seed_everything(seed)

    # resizing to protopnet input size
    resizer = ResizeImage((224, 224), rescale=True)
    transforms = [resizer]

    # get datasets
    train_ds = SliceResamplingDataset(
        data_dir, annotations['train'], transforms=transforms, multi_channel=3)
    val_ds = SliceResamplingDataset(
        data_dir, annotations['val'], transforms=transforms, multi_channel=3)
    test_ds = SliceResamplingDataset(
        data_dir, annotations['test'], transforms=transforms, multi_channel=3)

    # insert into dataloaders
    trainloader = DataLoader(
        train_ds, batch_size=batch_sz, num_workers=num_workers, shuffle=True
    )

    validloader = DataLoader(
        val_ds, batch_size=batch_sz, num_workers=num_workers, shuffle=True
    )

    testloader = DataLoader(
        test_ds, batch_size=batch_sz, num_workers=num_workers, shuffle=True
    )

    print('done getting dataloaders')
    return trainloader, validloader, testloader


def test_dataloaders():
    raise NotImplementedError('testing for this method not complete')
    # data_dir = os.path.abspath("trimmed")
    # train_file = os.path.abspath("./metadata/train.csv")
    # val_file = os.path.abspath("./metadata/val.csv")
    # test_file = os.path.abspath("./metadata/test.csv")
    # return 


if __name__ == "__main__": 

    test_dataloaders()
