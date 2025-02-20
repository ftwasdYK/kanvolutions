import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from torchvision import datasets, transforms
from PIL import Image
import os
import torch


def torch_train_val_split(
    dataset, batch_train, batch_eval, val_size=0.2, shuffle=True, seed=420, test=False
):
    """
    Split a dataset into training, validation, and optionally test sets with PyTorch DataLoader.

    Args:
        dataset: PyTorch Dataset object
        batch_train: Batch size for training
        batch_eval: Batch size for validation/test
        val_size: Size of validation (and test if test=True) set as fraction of total dataset
        shuffle: Whether to shuffle the indices before splitting
        seed: Random seed for reproducibility
        test: If True, creates three splits (train/val/test) instead of two (train/val)

    Returns:
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        test_loader: DataLoader for test set if test=True, else None
    """
    # Calculate dataset size and create indices
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    if test:
        # For test mode, we want three equal splits
        # If val_size is 0.2, we want:
        # - test_size = 0.1 (half of val_size)
        # - val_size = 0.1 (half of val_size)
        # - train_size = 0.8 (remaining portion)
        split_size = int(np.floor(val_size * dataset_size / 2))

        # Create the splits
        test_indices = indices[:split_size]  # First portion for test
        val_indices = indices[split_size:split_size * 2]  # Second portion for validation
        train_indices = indices[split_size * 2:]  # Remainder for training
    else:
        # For validation-only mode, we want two splits
        val_split = int(np.floor(val_size * dataset_size))
        val_indices = indices[:val_split]
        train_indices = indices[val_split:]
        test_indices = None

    # Create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices) if test else None

    # Create data loaders
    train_loader = DataLoader(dataset, batch_size=batch_train, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_eval, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_eval, sampler=test_sampler) if test else None

    return train_loader, val_loader, test_loader