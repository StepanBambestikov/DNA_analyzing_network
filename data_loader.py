import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler
import pandas as pd
from torch.utils.data import TensorDataset
from dna_enumeration import BASE_1D, BASE_2D

dna_bases_count = 4
dna_base_dict = {'a': BASE_1D.A, 't': BASE_1D.T, 'c': BASE_1D.C, 'g': BASE_1D.G} # todo add padding
dna_binary_view = {'a': BASE_2D.A, 't': BASE_2D.T, 'c': BASE_2D.C, 'g': BASE_2D.G, 'A': BASE_2D.A, 'T': BASE_2D.T,
                   'C': BASE_2D.C, 'G': BASE_2D.G}
dG_analytic_coefficients = [1.03, 0.98, -1.00, -0.88, -0.58, -1.45, -1.44, -1.28, -1.30, -2.17, -2.24, -1.84]


def make_coefficients_multiplier(coefficients):
    def coefficients_multiplier(raw_nn_array):
        for row_index in range(raw_nn_array.shape[0]):
            raw_nn_array[row_index] = raw_nn_array[row_index] * coefficients
        return raw_nn_array

    return coefficients_multiplier


def pair_finder(pairs, text_sequence):
    pair_count = 0
    for current_pair in pairs:
        pair_count += text_sequence.count(current_pair)
    return pair_count


def end_finder(ends, text_sequence):
    end_count = 0
    if type(text_sequence) is float:
        i = 0

    if text_sequence[0] in ends:
        end_count += 1
    if text_sequence[-1] in ends:
        end_count += 1
    return end_count


nearest_neighbors = [["aa", "tt", "AA", "TT"], ["at", "AT"], ["ta", "TA"], ["ca", "tg", "CA", "TG"],
                     ["gt", "ac", "GT", "AC"], ["ct", "ag", "CT", "AG"], ["ga", "tc", "GA", "TC"],
                     ["cg", "CG"], ["gc", "GC"], ["gg", "cc", "GG", "CC"]]

ends = [['a', 't', 'A', 'T'], ['g', 'c', 'G', 'C']]


def make_nn_data_from_text_dna(text_dna_array):
    nn_vector_len = 12
    data_features = np.zeros((text_dna_array.shape[0], nn_vector_len))
    for data_row_index, current_str in enumerate(text_dna_array):
        for vector_index, current_end in enumerate(ends):
            data_features[data_row_index, vector_index] = end_finder(current_end, current_str)

        for vector_index, current_nn in enumerate(nearest_neighbors):
            data_features[data_row_index, vector_index + 2] = pair_finder(current_nn, current_str)
    return data_features


def _get_sequences_lengths(text_dna_sequences):
    vector_len_function = np.vectorize(len)
    return vector_len_function(text_dna_sequences)


def _get_bases_count(text_dna_sequences, *searching_bases):
    result = torch.zeros(text_dna_sequences.shape[0])
    for current_index, current_string in enumerate(text_dna_sequences):
        for searching_base in searching_bases:
            result[current_index] += current_string.count(searching_base)
    return result


def _get_gc_concentration(text_dna_sequences):
    sequences_lengths = _get_sequences_lengths(text_dna_sequences)
    gc_count = _get_bases_count(text_dna_sequences, "g", "c", "G", "C")
    return gc_count / sequences_lengths


def _array_pow_expantion(input_column, pows):
    columns = [input_column]
    for current_pow in pows:
        columns.append(np.power(input_column, current_pow))
    return torch.from_numpy(np.column_stack((columns)))


# gc_pows = [2, 3]
# length_pows = [2, 3]
gc_pows = [0, 2, 3]
length_pows = [2, 3]
dna_column = 0


def Na_data_function_maker(origin_preparator):
    def function(dna_and_na_array):
        origin_data = torch.from_numpy(origin_preparator(dna_and_na_array[:, 0]))
        gc_concetrations = _get_gc_concentration(dna_and_na_array[:, 0])
        sequences_lengths = _get_sequences_lengths(dna_and_na_array[:, 0])
        gc_concetrations = _array_pow_expantion(gc_concetrations, gc_pows)
        sequences_lengths = _array_pow_expantion(sequences_lengths, length_pows)
        Na_features = torch.from_numpy(dna_and_na_array[:, 1].astype(np.float64))[:, None]
        Mg_features = torch.from_numpy(dna_and_na_array[:, 2].astype(np.float64))[:, None]
        return origin_data, torch.cat((gc_concetrations, sequences_lengths, Na_features, Mg_features), dim=1)
    return function


def make_Na_data_from_text_dna(dna_and_na_array, origin_preparator):
    origin_data = torch.from_numpy(origin_preparator(dna_and_na_array[:, 0]))
    gc_concetrations = _get_gc_concentration(dna_and_na_array[:, 0])
    sequences_lengths = _get_sequences_lengths(dna_and_na_array[:, 0])
    gc_concetrations = _array_pow_expantion(gc_concetrations, gc_pows)
    sequences_lengths = _array_pow_expantion(sequences_lengths, length_pows)
    Na_features = torch.from_numpy(dna_and_na_array[:, 1].astype(np.float64))[:, None]
    Mg_features = torch.from_numpy(dna_and_na_array[:, 2].astype(np.float64))[:, None]
    return origin_data, torch.cat((gc_concetrations, sequences_lengths, Na_features, Mg_features), dim=1)


def make_1d_data_from_text_dna(text_dna_array):
    #todo make constant related to max length of test dna!!!!
    max_string_size = len(max(text_dna_array, key=len)) + 1
    data_features = np.zeros((text_dna_array.shape[0], max_string_size))
    try:
        for data_row_index, current_str in enumerate(text_dna_array):
            for data_col_index, current_symbol in enumerate(current_str):
                data_features[data_row_index, data_col_index + 1] = dna_base_dict[current_symbol]
    except KeyError:
        raise ValueError("Unexpected letter in dna sequence")
    return data_features


def make_2d_data_from_text_dna(text_dna_array):
    #todo make constant related to max length of test dna!!!!
    # max_string_size = len(max(text_dna_array, key=len)) + 1
    max_string_size = 31
    # first_axis_padding = 1
    # second_axis_padding = 1
    first_axis_padding = 0
    second_axis_padding = 0
    summary_padding = 0

    data_features = np.zeros((text_dna_array.shape[0], dna_bases_count + summary_padding, max_string_size))
    for data_row_index, current_str in enumerate(text_dna_array):
        for data_col_index, current_symbol in enumerate(current_str):
            data_features[data_row_index, dna_binary_view[current_symbol] + first_axis_padding,
                                          data_col_index + second_axis_padding] = 1
    return data_features.astype('float32')


def get_dataset_from_excel_file(file_name, label_column_number, begin_feature_column, end_feature_column=None,
                                is_dna_analysis=True, data_handler=None, first_row=0):
    raw_data_array = pd.read_excel(file_name, header=None).to_numpy()

    if is_dna_analysis is True:
        raw_features = raw_data_array[first_row:, begin_feature_column]
        # take dna
    else:
        raw_features = raw_data_array[first_row:, begin_feature_column: end_feature_column].astype('float32')
        # take nearest neighbour

    if data_handler is not None:
        raw_features = data_handler(raw_features)

    raw_y = raw_data_array[first_row:, label_column_number:label_column_number + 1].astype('float32')
    dataset = TensorDataset(torch.Tensor(raw_features), torch.Tensor(raw_y))
    return dataset


def get_jumbled_indices_for_train_and_val(validation_split, data_size):
    validation_elements_count = int(np.floor(validation_split * data_size))
    data_indices = list(range(data_size))
    np.random.shuffle(data_indices)
    train_indices, val_indices = data_indices[validation_elements_count:], data_indices[:validation_elements_count]
    return train_indices, val_indices


def get_train_and_val_loaders_from_indices(train_indices, val_indices, dataset, batch_size=None):
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    if batch_size is None:
        train_batch_size = len(train_indices)
        val_batch_size = len(val_indices)
    else:
        train_batch_size = batch_size
        val_batch_size = batch_size
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=train_batch_size,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=val_batch_size,
                                             sampler=val_sampler)
    return train_loader, val_loader


def get_train_and_val_loaders(train_dataset, batch_size=None):
    data_size = train_dataset.tensors[0].data.shape[0]
    validation_split = .2
    train_indices, val_indices = get_jumbled_indices_for_train_and_val(validation_split, data_size)
    train_loader, val_loader = get_train_and_val_loaders_from_indices(train_indices, val_indices, train_dataset,
                                                                      batch_size)
    return train_loader, val_loader

