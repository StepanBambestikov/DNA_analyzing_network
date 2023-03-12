import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler


def get_train_and_val_loaders(train_dataset, batch_size):
    data_size = train_dataset.tensors[0].data.shape[0]

    validation_split = .2
    validation_elements_count = int(np.floor(validation_split * data_size))
    data_indices = list(range(data_size))
    np.random.shuffle(data_indices)

    train_indices, val_indices = data_indices[validation_elements_count:], data_indices[:validation_elements_count]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                             sampler=val_sampler)
    return train_loader, val_loader
