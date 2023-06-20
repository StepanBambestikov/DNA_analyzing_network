import torch
import data_loader
from view_enum import *
import network_classes as networks
import pandas as pd
import pathlib
import numpy as np


def parse_input_to_enum(user_text_input):
    enum_input = {}

    for current_glyph_enum in main_glyphs_to_text.keys():
        try:
            value = user_text_input[main_glyphs_to_text[current_glyph_enum]]
            enum_input[current_glyph_enum] = value
        except:
            break

    return enum_input


# loadable_network_classes = {
#     Processing.D1: networks.dna_1d_linear_classifier,
#     Processing.D2: networks.dna_2d_linear_classifier,
#     Processing.NN: networks.linear_classifier,
# }


# def _load_network_from_file(network_path, processing_type):
#     network_type = loadable_network_classes[processing_type]
#     network = network_type()
#     try:
#         network.load_state_dict(torch.load(network_path))
#     except FileNotFoundError:
#         raise ValueError("Where is no such file for network loading with name " + network_path)
#     return network


def make_predictors_from_network_paths(enum_input_data):
    processing_type = enum_input_data[Glyphs.DATA_TYPE]
    predictors_dict = {}
    parameters_for_calculations = []
    for (param, is_needed) in enum_input_data[Glyphs.CALCULATE].items():
        if is_needed:
            parameters_for_calculations.append(param)
    network_paths = dict(filter(lambda param_pair: param_pair[1],
                                enum_input_data[Glyphs.NETWORKS_PATHS].items()))
    if not parameters_for_calculations:
        raise ValueError("There is no parameters for calculation!")
    for current_parameter in parameters_for_calculations:
        if current_parameter not in network_paths.keys():
            raise ValueError("No path to network for parameter " + current_parameter + "!")
        else:
            loaded_network = _load_network_from_file(network_paths[current_parameter], processing_type)
            predictors_dict[current_parameter] = loaded_network
    return predictors_dict


class File_types(IntEnum):
    CSV = 0,
    EXCEL = 1


pandas_writers = {
    File_types.CSV: pd.DataFrame.to_csv,
    File_types.EXCEL: pd.DataFrame.to_excel,
}


class Output_stream:

    def __init__(self, save_file_name=None, save_file_type=None, output_function=None):
        self.save_file_name = save_file_name,
        self.save_file_type = save_file_type
        self.output_function = output_function

    def __lshift__(self, other):
        if self.output_function is not None:
            self.output_function(other)
        else:
            writer = pandas_writers[self.save_file_type]
            writer(self.save_file_name)


def create_output_stream(enum_input_data):
    save_file_name = enum_input_data[Glyphs.OUTPUT_FILE_NAME]
    save_file_type = enum_input_data[Glyphs.SAVE_FILE_TYPE]
    output_function = None  # todo как это добавить и надо ли
    if not save_file_name:
        output_stream = Output_stream(output_function=output_function)  # todo who will pass output_function to us?
    else:
        output_stream = Output_stream(save_file_name=save_file_name, save_file_type=save_file_type)
    return output_stream





pandas_readers = {
    File_types.CSV: pd.read_csv,
    File_types.EXCEL: pd.read_excel
}


file_extensions = {
    '.xlsx': File_types.EXCEL

}


def _get_file_type(file_name):
    return file_extensions[pathlib.Path(file_name).suffix]


def load_dna_text_from_file(input_file_name):
    read_function = pandas_readers[_get_file_type(input_file_name)]
    text_dna = read_function(input_file_name)
    text_dna = text_dna.to_numpy()[:, -1]
    return text_dna


def load_and_prepare_dna_data(enum_input_data):
    input_file_name = enum_input_data[Glyphs.INPUT_FILE_NAME]
    if not input_file_name:
        dna_text = np.array([enum_input_data[Glyphs.DNA_DATA]])
        if not dna_text:
            raise ValueError("There is no input file with data and also not text input")
    else:
        dna_text = load_dna_text_from_file(input_file_name)
    dna_handler_type = enum_input_data[Glyphs.DATA_TYPE]
    dna_handler = dna_handlers[dna_handler_type]
    dna_data = dna_handler(dna_text)

    return dna_data


def parse_input_information(user_text_input):
    user_enum_input = parse_input_to_enum(user_text_input)
    # predictors_dict = make_predictors_from_network_paths(user_enum_input)
    dna_data = load_and_prepare_dna_data(user_enum_input)
    output_stream = create_output_stream(user_enum_input)
    # return predictors_dict, dna_data, output_stream  # todo должен быть набор из output stream бро!
