import torch
from view_enum import *
import network_classes as networks
import pathlib
import numpy as np


import pandas_adapter as pd
import predictor_adapter as pa


def _get_file_type(file_name):
    return pd.file_extensions[pathlib.Path(file_name).suffix]


def _load_data_from_file(input_file_name):
    read_function = pd.pandas_readers[_get_file_type(input_file_name)]
    data = read_function(input_file_name, header=None)
    # todo data
    first_row_index = 1
    return data.to_numpy()[first_row_index:, :]


def load_data_for_predictor(enum_input_data):
    """
    download data from input_file or pass input dna if input exists
    :param enum_input_data:
    :return: return table with columns (dna, *sol_conditions)
    """
    input_file_name = enum_input_data[INPUT_INFO.INPUT_FILE_NAME]
    if not input_file_name:
        return _make_entered_data(enum_input_data) # get all inputs
        #  todo get only one string
    return _load_data_from_file(input_file_name)


def make_predictor(enum_input_data):
    """
    return predictor object using enum_type from input info
    :param enum_input_data:
    :return:
    """
    enum_predictor_type = enum_input_data[INPUT_INFO.PREDICTOR_TYPE]
    return pa.predictor_to_object[enum_predictor_type]


class Output_stream:
    def __init__(self, output_function=None, save_file_name=None, save_file_type=None):
        self.save_file_name = save_file_name,
        self.save_file_type = save_file_type
        self.output_function = output_function

    def __lshift__(self, saving_dataframe):
        if self.output_function is not None:
            self.output_function(saving_dataframe)
        else:
            writer = pd.pandas_writers[self.save_file_type]
            writer(saving_dataframe, self.save_file_name[0])


def make_output_streams(enum_input_data):
    """
    make output streams from output file and view_function
    :param enum_input_data:
    :return:
    """
    save_file_name = enum_input_data[INPUT_INFO.OUTPUT_FILE_NAME]
    save_file_type = enum_input_data[INPUT_INFO.OUTPUT_FILE_TYPE]
    output_streams = []
    if save_file_name:
        output_streams.append(Output_stream(save_file_name=save_file_name, save_file_type=save_file_type))
    view_output_function = enum_input_data[INPUT_INFO.OUTPUT_FUNCTION]
    output_streams.append(Output_stream(output_function=view_output_function))
    return output_streams


def calculate_and_pass_predictions(predictor, input_data, output_streams):
    predictions = predictor(input_data)
    predictions = pd.make_DataFrame_from_tensor(predictions)
    for current_output_stream in output_streams:
        current_output_stream << predictions


def parse_info_and_calculate_parameters(enum_input_data):
    data = load_data_for_predictor(enum_input_data)
    predictor = make_predictor(enum_input_data)
    output_streams = make_output_streams(enum_input_data)
    calculate_and_pass_predictions(predictor, data, output_streams)

# todo add error manager when error occurs