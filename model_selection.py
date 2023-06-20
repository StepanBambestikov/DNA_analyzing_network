import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset
import math
import data_loader as loader
import network_service as nn_service
import prediction_analyzer as pl

import network_classes as networks
import excel_parameters as params
import pandas as pd
import loss
import butch_handlers
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from enum import IntEnum


def _load_dna_dataset(column_number=params.column_numbers[params.Parameters.dG],
                      data_handler=loader.make_2d_data_from_text_dna):
    dataset = loader.get_dataset_from_excel_file("ML_Stepan.xlsx", label_column_number=column_number,
                                                 begin_feature_column=params.begin_feature_column,
                                                 end_feature_column=params.end_feature_column,
                                                 is_dna_analysis=True,
                                                 data_handler=data_handler,
                                                 first_row=params.first_row)
    return dataset


def _load_nn_dataset(column_number=params.column_numbers[params.Parameters.dG], data_handler=None):
    dataset = loader.get_dataset_from_excel_file("ML_Stepan.xlsx", label_column_number=column_number,
                                                 begin_feature_column=params.begin_feature_column,
                                                 end_feature_column=params.end_feature_column,
                                                 is_dna_analysis=False,
                                                 data_handler=data_handler,
                                                 first_row=params.first_row)
    return dataset


def _train_network(current_model, train_loader, val_loader, loss, current_model_is_conv):
    # training of one network for one parameter
    optimizer = optim.Adam(current_model.parameters(), lr=0.006, weight_decay=1e-4)
    epoch_number = 20000
    history_manager = nn_service.train_history_manager(epoch_number)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.6)
    is_convolution_training = current_model_is_conv
    train_manager = nn_service.network_train_service(current_model, loss, optimizer,
                                                     scheduler,
                                                     butch_handler=None,
                                                     is_convolution_training=is_convolution_training,
                                                     is_relative_error=True)
    painter = nn_service.epoch_painter(print, 1)
    # painter = None
    nn_service.train_model(train_manager, train_loader, val_loader, epoch_number,
                           epoch_painter=painter,
                           history_manager=history_manager)
    print("network trained!")
    print(history_manager.train_val_min_error_Tm())
    return history_manager.train_val_min_error_Tm()


class Data_types(IntEnum):
    NN = 0,
    D1 = 1,
    D2 = 2,
    Na_NN = 3,
    Na_D2 = 4


data_handlers = {
    Data_types.NN: loader.make_nn_data_from_text_dna,
    Data_types.D1: loader.make_1d_data_from_text_dna,
    Data_types.D2: loader.make_2d_data_from_text_dna,
    Data_types.Na_NN: loader.Na_data_function_maker(loader.make_nn_data_from_text_dna),
    Data_types.Na_D2: loader.Na_data_function_maker(loader.make_2d_data_from_text_dna)
}

# main_file_name = "ML_Stepan_25_04_2023.xlsx"
# main_file_name = "ML_Stepan_25_04_2023_add.xlsx"
main_file_name = "ML_Stepan_25_05_2023_Mg.xlsx"
# first_row = 3
# end_row = 986 #ML_Stepan_25_04_2023.xlsx end row
first_row = 1
end_row = 196 #ML_Stepan_25_04_2023_Mg.xlsx end row
# end_row = 864 #ML_Stepan_25_04_2023.xlsx end row
# end_row = 305 #ML_Stepan_25_04_2023.xlsx end row


def save_model(model, model_name):
    torch.save(model.state_dict(), model_name)


def _get_raw_features(added_input_columns=None):
    raw_data_array = pd.read_excel(main_file_name, header=None).to_numpy()
    raw_dna_features = raw_data_array[first_row: end_row, params.dna_feature_column_number]
    if added_input_columns is not None:
        raw_dna_features = np.column_stack((raw_dna_features, raw_data_array[first_row: end_row, added_input_columns]))
    return raw_dna_features


complement_base = {
    'a': 't',
    'c': 'g',
    't': 'a',
    'g': 'c',
    'A': 'T',
    'C': 'G',
    'T': 'A',
    'G': 'C',
}

all_column_numbers = [params.column_numbers[parameter] for parameter in params.Parameters]


def make_self_complement(dna_sequence_text):
    complement_bases_list = []
    for current_base in dna_sequence_text:
        complement_bases_list.insert(0, complement_base[current_base])
    return ''.join(complement_bases_list)


def no_self_complement(dna_sequence_text):
    complement_pair = make_self_complement(dna_sequence_text)
    return complement_pair != dna_sequence_text


def get_no_self_complement_indexes(dna_data):
    no_self_complement_indexes = []
    for current_index in range(len(dna_data)):
        if no_self_complement(dna_data[current_index]):
            no_self_complement_indexes.append(current_index)
    return no_self_complement_indexes


def _get_features_and_labels(origin_data_type, input_Na_column_number, input_Mg_column_number,
                             without_complement, *labels_column_numbers):
    raw_dna_data = _get_raw_features(added_input_columns=[input_Na_column_number, input_Mg_column_number])
    labels = _get_labels(*labels_column_numbers)
    if without_complement:
        no_self_complement_indexes = get_no_self_complement_indexes(raw_dna_data[:, 0])
        labels = labels[no_self_complement_indexes, :]
        raw_dna_data = raw_dna_data[no_self_complement_indexes, :]
    features = [*data_handlers[origin_data_type](raw_dna_data)]
    return features, labels


def _get_features(origin_data_type, added_input_columns=None):
    raw_dna_features = _get_raw_features(added_input_columns)
    raw_features_list = [*data_handlers[origin_data_type](raw_dna_features)]
    return raw_features_list


def _get_labels(*labels_column_numbers):
    raw_data_array = pd.read_excel(main_file_name, header=None).to_numpy()
    if len(labels_column_numbers) == 1:
        labels = raw_data_array[first_row: end_row, labels_column_numbers[0]: labels_column_numbers[0] + 1].astype('float32')
        return labels

    label_list = []
    for label_column_number in labels_column_numbers:
        label_list.append(raw_data_array[first_row: end_row, label_column_number: label_column_number + 1].astype('float32'))

    return torch.from_numpy(np.stack(label_list, axis=1)[:, :, 0])


split_coefficient = 0.2


def _split_data_on_train_val(features, labels, train_indices, val_indices, butch_size=None):
    data = TensorDataset(torch.Tensor(np.array(features)), torch.Tensor(labels))
    train_loader, val_loader = loader.get_train_and_val_loaders_from_indices(train_indices, val_indices, data,
                                                                             butch_size)
    return train_loader, val_loader


def Tm_train_val_calculation(train_prediction, val_prediction):
    dH_train_prediction = train_prediction[:, 0]
    dS_train_prediction = train_prediction[:, 2]
    Tm_train_prediction = nn_service.Tm_calculation(dH_train_prediction, dS_train_prediction)
    dH_val_prediction = val_prediction[:, 0]
    dS_val_prediction = val_prediction[:, 2]
    Tm_val_prediction = nn_service.Tm_calculation(dH_val_prediction, dS_val_prediction)
    return Tm_train_prediction, Tm_val_prediction


def train_and_evaluate_models_for_all_parameters(train_indices, val_indices, features, current_model_type,
                                                 current_model_is_conv, loss):
    evaluations = {}
    for current_parameter in params.Parameters:
        if current_parameter == params.Parameters.Tm:
            continue

        current_label_column_number = params.column_numbers[current_parameter]
        current_model = current_model_type()
        current_labels = _get_labels(current_label_column_number)
        train_loader, val_loader = _split_data_on_train_val(features, current_labels, train_indices, val_indices)
        _train_network(current_model, train_loader, val_loader, loss, current_model_is_conv)

        train_features, train_labels = torch.Tensor(features[train_indices]), torch.Tensor(
            current_labels[train_indices])
        val_features, val_labels = torch.Tensor(features[val_indices]), torch.Tensor(current_labels[val_indices])

        train_prediction = nn_service.forward_propagation(current_model, train_features, current_model_is_conv)
        val_prediction = nn_service.forward_propagation(current_model, val_features, current_model_is_conv)
        evaluations[current_parameter] = [train_prediction, train_labels, val_prediction, val_labels]

    # adding Tm!
    Tm_labels = _get_labels(params.column_numbers[params.Parameters.Tm])
    dH_train_prediction, dH_train_ground_truth, dH_val_prediction, dH_val_ground_truth = evaluations[
        params.Parameters.dH]
    dS_train_prediction, dS_train_ground_truth, dS_val_prediction, dS_val_ground_truth = evaluations[
        params.Parameters.dS]
    Tm_train_prediction = nn_service.Tm_calculation(dH_train_prediction, dS_train_prediction)
    Tm_val_prediction = nn_service.Tm_calculation(dH_val_prediction, dS_val_prediction)
    Tm_train_ground_truth = torch.Tensor(Tm_labels[train_indices])
    # diff = abs((Tm_train_prediction - Tm_train_ground_truth) / Tm_train_ground_truth) * 100
    # i = torch.mean(diff)

    # Tm_absolute_error = nn_service.get_prediction_error(Tm_train_prediction, Tm_train_ground_truth, is_relative_error=False,
    #                                                  is_absolute_value=False)
    # pl.draw_error_histograms("dsd", pl.Graph("val relative error", relative_error))
    # i = torch.cat((Tm_train_prediction, Tm_train_ground_truth), dim=1).numpy()
    Tm_val_ground_truth = torch.Tensor(Tm_labels[val_indices])
    # pl.draw_correlation("s",
    #                     pl.CorrelationPair(dH_train_prediction, dS_train_prediction,
    #                                        "dHdS prediction"),
    #                     pl.CorrelationPair(dH_train_ground_truth, dS_train_ground_truth, "dHdS ground truth"))
    evaluations[params.Parameters.Tm] = [Tm_train_prediction, Tm_train_ground_truth, Tm_val_prediction,
                                         Tm_val_ground_truth]
    return evaluations


# losses = {"normalized_l1_loss": loss.normalized_l1_loss, "normalized_l2_loss": loss.normalized_l2_loss}
# losses = {"n_l1_loss": loss.normalized_l1_loss}



def train_and_evaluate_multi_model(train_indices, val_indices, features, model_type,
                                   current_model_is_conv, loss, train_val_splitter=_split_data_on_train_val,
                                   labels=None):
    evaluations = {}
    model = model_type("conv2d_net_06_05normalized_l1_multi_loss_absolute_Tm")
    # model.load_state_dict(torch.load("conv2d_net_06_05normalized_l1_multi_loss_absolute_Tm"))
    # create and pass train_loader, val_loader
    if labels is None:
        labels = _get_labels(*all_column_numbers)
    train_loader, val_loader = train_val_splitter(features, labels, train_indices, val_indices, butch_size=64)
    _train_network(model, train_loader, val_loader, loss, current_model_is_conv) # todo return!

    train_loader, val_loader = train_val_splitter(features, labels, train_indices, val_indices, butch_size=None)
    train_features, train_labels = next(next(iter(train_loader)))
    val_features, val_labels = next(next(iter(val_loader)))

    # 4 labels and predictions is 4-dimentional vector (dH, dG, dS, Tm)
    train_prediction = nn_service.forward_propagation(model, train_features, current_model_is_conv)
    val_prediction = nn_service.forward_propagation(model, val_features, current_model_is_conv)
    Tm_train_prediction, Tm_val_prediction = Tm_train_val_calculation(train_prediction, val_prediction)
    Tm_train_labels, Tm_val_labels = train_labels[:, 3], val_labels[:, 3]

    for current_index, current_parameter in enumerate(params.Parameters):
        if current_parameter == params.Parameters.Tm:
            evaluations[current_parameter] = [Tm_train_prediction, Tm_train_labels, Tm_val_prediction, Tm_val_labels]
            continue
        evaluations[current_parameter] = [train_prediction[:, current_index], train_labels[:, current_index],
                                          val_prediction[:, current_index], val_labels[:, current_index]]
    return evaluations, model


# input_Na_column_number = 11  # activity, not Na_concentration
input_Na_column_number = 6  # activity, not Na_concentration
input_Mg_column_number = 7


def models_analysis(model_types, model_names, conv_factors, origin_data_type, evaluation_manager, loss_types,
                    train_val_splitter=_split_data_on_train_val):
    model_types_evaluations = {}
    models = []
    # data_features = _get_features(origin_data_type, input_Na_column_number)
    data_features, labels = _get_features_and_labels(origin_data_type, input_Na_column_number, input_Mg_column_number,
                                                     True, *all_column_numbers)
    validation_split = .2
    train_indices, val_indices = loader.get_jumbled_indices_for_train_and_val(validation_split, len(data_features[0]))
    for current_model_type, current_model_name, current_model_is_conv in zip(model_types, model_names, conv_factors):
        print(current_model_name)
        current_loss_evaluations = {}
        for current_loss_name, current_loss in loss_types.items():
            print(current_loss_name)
            # current_evaluations = train_and_eval_models_for_all_parameters(train_indices, val_indices,
            #     data_features, current_model_type, current_model_is_conv, current_loss)
            current_evaluations, current_model = evaluation_manager(train_indices, val_indices,
                                                     data_features, current_model_type, current_model_is_conv,
                                                     current_loss, train_val_splitter, labels)
            models.append(current_model)
            current_loss_evaluations[current_loss_name] = current_evaluations
        model_types_evaluations[current_model_name] = current_loss_evaluations
    raw_features = _get_raw_features(added_input_columns=input_Na_column_number)
    return model_types_evaluations, (raw_features[train_indices], raw_features[val_indices]), models


parameters_to_str = {
    params.Parameters.dH: "dH",
    params.Parameters.dG: "dG",
    params.Parameters.dS: "dS",
    params.Parameters.Tm: "Tm"
}


def predictions_plotting(model_types_evaluations):
    for current_model_name, current_model_values in model_types_evaluations.items():
        for current_loss_name, current_loss_values in current_model_values.items():
            for current_parameter_name, current_parameter_value in current_loss_values.items():
                current_graph_name = current_model_name + " " + current_loss_name + " " + parameters_to_str[
                    current_parameter_name]
                is_relative_error = True
                if current_parameter_name == params.Parameters.Tm:
                    is_relative_error = False
                prediction_draw(current_graph_name, *current_parameter_value, is_relative_error)


saving_columns_names = ['DNA', 'Activity', 'dH_error %', 'dH predicted', 'dH truth', 'dS_error %', 'dS predicted',
                        'dS truth', 'dG_error %', 'dG predicted', 'dG truth', 'Tm_error absolute',
                        'Tm predicted', 'Tm truth']
params_to_column_name = {params.Parameters.dH: saving_columns_names[2],
                         params.Parameters.dG: saving_columns_names[5],
                         params.Parameters.dS: saving_columns_names[8],
                         params.Parameters.Tm: saving_columns_names[11]}


def save_predictions(model_types_evaluations, file_name, pandas_load_method, train_additive_columns, val_additive_columns):
    for current_model_name, current_model_values in model_types_evaluations.items():
        for current_loss_name, current_loss_values in current_model_values.items():
            parameters_errors = {}
            absolute_values = {}
            for current_parameter_name, current_parameter_value in current_loss_values.items():
                is_relative_error = True
                if current_parameter_name == params.Parameters.Tm:
                    is_relative_error = False
                train_predicted, train_labels, val_predicted, val_labels = current_parameter_value
                train_error = nn_service.get_prediction_error(train_predicted, train_labels,
                                                              is_relative_error=is_relative_error)
                val_error = nn_service.get_prediction_error(val_predicted, val_labels,
                                                            is_relative_error=is_relative_error)
                parameters_errors[current_parameter_name] = (train_error, val_error)
                absolute_values[current_parameter_name] = (train_predicted, train_labels, val_predicted, val_labels)
            _, train_sorted_indexes = torch.sort(parameters_errors[params.Parameters.Tm][0])
            _, val_sorted_indexes = torch.sort(parameters_errors[params.Parameters.Tm][1])
            train_errors = torch.column_stack((
                                      parameters_errors[params.Parameters.dH][0][train_sorted_indexes],
                                      absolute_values[params.Parameters.dH][0][train_sorted_indexes],
                                      absolute_values[params.Parameters.dH][1][train_sorted_indexes],

                                      parameters_errors[params.Parameters.dS][0][train_sorted_indexes],
                                      absolute_values[params.Parameters.dS][0][train_sorted_indexes],
                                      absolute_values[params.Parameters.dS][1][train_sorted_indexes],

                                      parameters_errors[params.Parameters.dG][0][train_sorted_indexes],
                                      absolute_values[params.Parameters.dG][0][train_sorted_indexes],
                                      absolute_values[params.Parameters.dG][1][train_sorted_indexes],

                                      parameters_errors[params.Parameters.Tm][0][train_sorted_indexes],
                                      absolute_values[params.Parameters.Tm][0][train_sorted_indexes],
                                      absolute_values[params.Parameters.Tm][1][train_sorted_indexes]
            ))
            val_errors = torch.column_stack((
                                    parameters_errors[params.Parameters.dH][1][val_sorted_indexes],
                                    absolute_values[params.Parameters.dH][2][val_sorted_indexes],
                                    absolute_values[params.Parameters.dH][3][val_sorted_indexes],

                                    parameters_errors[params.Parameters.dS][1][val_sorted_indexes],
                                    absolute_values[params.Parameters.dS][2][val_sorted_indexes],
                                    absolute_values[params.Parameters.dS][3][val_sorted_indexes],

                                    parameters_errors[params.Parameters.dG][1][val_sorted_indexes],
                                    absolute_values[params.Parameters.dG][2][val_sorted_indexes],
                                    absolute_values[params.Parameters.dG][3][val_sorted_indexes],

                                    parameters_errors[params.Parameters.Tm][1][val_sorted_indexes],
                                    absolute_values[params.Parameters.Tm][2][val_sorted_indexes],
                                    absolute_values[params.Parameters.Tm][3][val_sorted_indexes]
            ))

            train_dna_col = train_additive_columns[:, 0][train_sorted_indexes]
            train_activity_col = train_additive_columns[:, 1][train_sorted_indexes]
            val_dna_col = val_additive_columns[:, 0][val_sorted_indexes]
            val_activity_col = val_additive_columns[:, 1][val_sorted_indexes]
            data = pd.DataFrame(
                np.concatenate((train_dna_col[:, None], train_activity_col[:, None], train_errors.detach().numpy()),
                                               axis=1), columns=saving_columns_names)
            pandas_load_method(data, "train_output_" + current_model_name + file_name, index=False,
                               columns=saving_columns_names)
            data = pd.DataFrame(
                np.concatenate((val_dna_col[:, None], val_activity_col[:, None], val_errors.detach().numpy()),
                               axis=1), columns=saving_columns_names)
            pandas_load_method(data, "val_output_" + current_model_name + file_name, index=False,
                               columns=saving_columns_names)


def prediction_draw(graph_name, train_prediction, train_ground_truth, val_prediction, val_ground_truth,
                    relative_error_in_correlation):
    val_relative_error = nn_service.get_prediction_error(val_prediction, val_ground_truth, is_relative_error=True,
                                                         is_absolute_value=False)
    train_relative_error = nn_service.get_prediction_error(train_prediction, train_ground_truth, is_relative_error=True,
                                                           is_absolute_value=False)
    pl.draw_error_histograms(graph_name, pl.Graph("val relative error", val_relative_error),
                             pl.Graph("train relative error", train_relative_error))
    pl.draw_correlation(graph_name, [relative_error_in_correlation, relative_error_in_correlation],
                        pl.CorrelationPair(train_ground_truth, train_prediction, "train"),
                        pl.CorrelationPair(val_ground_truth, val_prediction, "val"))
