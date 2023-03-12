import pandas as pd
import torch
from torch.utils.data import TensorDataset
import numpy as np

dna_bases_count = 4
dna_enum = {'a': 2, 't': 3, 'c': 4, 'g': 5}
dna_array_row = {'a': 0, 't': 1, 'c': 2, 'g': 3}


def make_numeric_from_text_dna(text_dna_array):
    max_string_size = len(max(text_dna_array, key=len)) + 1

    data_features = np.zeros((text_dna_array.shape[0], max_string_size))
    for data_row_index, current_str in enumerate(text_dna_array):
        for data_col_index, current_symbol in enumerate(current_str[0]):
            data_features[data_row_index, data_col_index + 1] = dna_enum[current_symbol]
    return data_features


def make_2d_data_from_text_dna(text_dna_array):
    max_string_size = len(max(text_dna_array, key=len)) + 1
    first_axis_padding = 1
    second_axis_padding = 1
    summary_padding = 2

    data_features = np.zeros((text_dna_array.shape[0], dna_bases_count + summary_padding, max_string_size))
    for data_row_index, current_str in enumerate(text_dna_array):
        for data_col_index, current_symbol in enumerate(current_str[0]):
            data_features[data_row_index, dna_array_row[current_symbol] + first_axis_padding, data_col_index + second_axis_padding] = 1
    return data_features


def get_dataset_from_excel_file(file_name, label_column_number, begin_feature_column, end_feature_column=None, dna_to_numeric_strategy=None, first_row=0):
    raw_data_darray = pd.read_excel(file_name, header=None).to_numpy()
    if dna_to_numeric_strategy is not None:
        raw_X = dna_to_numeric_strategy(raw_data_darray[first_row:, begin_feature_column:end_feature_column]) #translating dna text into numeric matrix or numeric string
    else:
        raw_X = raw_data_darray[first_row:, begin_feature_column: end_feature_column].astype('float32') #just take certain columns

    raw_y = raw_data_darray[first_row:, label_column_number:label_column_number + 1].astype('float32')
    dataset = TensorDataset(torch.Tensor(raw_X), torch.Tensor(raw_y))
    return dataset
