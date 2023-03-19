import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler
import pandas as pd
from torch.utils.data import TensorDataset

dna_bases_count = 4
dna_base_dict = {'a': 2, 't': 3, 'c': 4, 'g': 5}
dna_binary_view = {'a': 0, 't': 1, 'c': 2, 'g': 3}


def make_1d_from_text_dna(text_dna_array):
    max_string_size = len(max(text_dna_array, key=len)) + 1
    data_features = np.zeros((text_dna_array.shape[0], max_string_size))
    for data_row_index, current_str in enumerate(text_dna_array):
        for data_col_index, current_symbol in enumerate(current_str[0]):
            data_features[data_row_index, data_col_index + 1] = dna_base_dict[current_symbol]
    return data_features


def make_2d_data_from_text_dna(text_dna_array):
    max_string_size = len(max(text_dna_array, key=len)) + 1
    first_axis_padding = 1
    second_axis_padding = 1
    summary_padding = 2

    data_features = np.zeros((text_dna_array.shape[0], dna_bases_count + summary_padding, max_string_size))
    for data_row_index, current_str in enumerate(text_dna_array):
        for data_col_index, current_symbol in enumerate(current_str[0]):
            data_features[data_row_index, dna_binary_view[current_symbol] + first_axis_padding,
                                          data_col_index + second_axis_padding] = 1
    return data_features


def get_dataset_from_excel_file(file_name, label_column_number, begin_feature_column, end_feature_column=None,
                                dna_to_numeric_strategy=None, first_row=0):
    raw_data_array = pd.read_excel(file_name, header=None).to_numpy()
    if dna_to_numeric_strategy is not None:
        raw_features = dna_to_numeric_strategy(raw_data_array[first_row:, begin_feature_column: end_feature_column])
        # translating dna text into numeric matrix or numeric string
    else:
        raw_features = raw_data_array[first_row:, begin_feature_column: end_feature_column].astype('float32')
        # just take certain columns

    raw_y = raw_data_array[first_row:, label_column_number:label_column_number + 1].astype('float32')
    dataset = TensorDataset(torch.Tensor(raw_features), torch.Tensor(raw_y))
    return dataset


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
