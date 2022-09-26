"""
Create train, valid, test iterators for CIFAR-10.
Train set size: 45000
Val set size: 5000
Test set size: 10000
If there is a separate set for meta-learning,
it is created from train set and has 5000 examples by default.
"""

import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
# from corruptions import transforms_dict


class CorruptedCIFAR10(datasets.CIFAR10):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        corruption_type_list = ['Gaussian Noise', 'Shot Noise',  'Defocus Blur',
                                'Glass Blur', 'Zoom Blur', 'Snow', 'Frost',
                                'Brightness', 'Contrast', 'Pixelate', 'JPEG',
                                'Speckle Noise', 'Gaussian Blur', 'Spatter', 'Saturate']
        corruption_type_idx = np.random.randint(0, len(corruption_type_list))

        # following the ARM paper, we will set aside the following as the test set:
        # ['Impulse Noise', 'Motion Blur', 'Fog', 'Elastic'] - all severity levels
        # ['Spatter', 'JPEG'] - severity 5
        corruption_type = corruption_type_list[corruption_type_idx]
        if corruption_type == 'Spatter' or corruption_type == 'JPEG':
            severity = np.random.randint(1, 5)
        else:
            severity = np.random.randint(1, 6)

        raw_img_c = np.uint8(transforms_dict[corruption_type](img, severity))

        img = Image.fromarray(raw_img_c)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def get_train_valid_loader(batch_size,
                           augment,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=False,
                           get_val_temp=0,
                           meta_val=False,
                           meta_val_size=1.0,
                           multi_domain=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. 
    Params:
    ------
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    - get_val_temp: set to 1 if temperature is to be set on a separate
                    val set other than normal val set.
    - meta_val: if to use a separate validation set for meta-learning.
    - meta_val_size: what multiple of the validation set size 
                     meta-validation set size should be.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    # load the dataset
    # data_dir = './data'
    data_dir = '/home/rotm/causal_inference/cifar10data'
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=train_transform)

    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=False, transform=valid_transform)

    if meta_val:
        if multi_domain:
            meta_dataset = CorruptedCIFAR10(
                root=data_dir, train=True,
                download=False, transform=valid_transform)
        else:
            meta_dataset = datasets.CIFAR10(
                root=data_dir, train=True,
                download=False, transform=valid_transform)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if meta_val:
        # our meta validation set will be the same size as the validation set
        split_meta = int(
            np.floor((1 + meta_val_size) * valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    if meta_val:
        train_idx, valid_idx, meta_idx = indices[split_meta:], indices[:split], indices[split:split_meta]
    else:
        train_idx, valid_idx = indices[split:], indices[:split]
    if get_val_temp > 0:
        valid_temp_dataset = datasets.CIFAR10(
            root=data_dir, train=True,
            download=False, transform=valid_transform)
        split = int(np.floor(get_val_temp * split))
        valid_idx, valid_temp_idx = valid_idx[split:], valid_idx[:split]
        valid_temp_sampler = SubsetRandomSampler(valid_temp_idx)
        valid_temp_loader = torch.utils.data.DataLoader(
            valid_temp_dataset, batch_size=batch_size, sampler=valid_temp_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory)

    if meta_val:
        meta_sampler = SubsetRandomSampler(meta_idx)
        meta_loader = torch.utils.data.DataLoader(
            meta_dataset, batch_size=batch_size, sampler=meta_sampler,
            num_workers=num_workers, pin_memory=pin_memory)

    if get_val_temp > 0:
        return (train_loader, valid_loader, valid_temp_loader)
    elif meta_val:
        return (train_loader, valid_loader, meta_loader)
    else:
        return (train_loader, valid_loader)


def get_test_loader(batch_size,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=False):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    data_dir = './data'
    dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=True, transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader