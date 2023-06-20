from enum import IntEnum
import torch

import network_service as nn_service
import network_classes as nn_classes
import data_loader


class Processing(IntEnum):
    NN = 0
    D1 = 1
    D2 = 2


dna_handlers = {
    Processing.D1: data_loader.make_1d_data_from_text_dna,
    Processing.D2: data_loader.make_2d_data_from_text_dna,
    Processing.NN: data_loader.make_nn_data_from_text_dna,
}


class prediction_columns(IntEnum):
    dH_INDEX = 0
    dG_INDEX = 1
    dS_INDEX = 2
    Tm_INDEX = 3


def make_test_network():
    def test_network(processed_dna, sol_data):
        prediction = torch.ones([4, 3])
        return prediction
    return test_network


model_types_to_saved_files = {
    nn_classes.two_layer_network: "two_layer_network"

}


def _load_network(model_type):
    model = model_type()
    # todo need to take back!!!!!!!!!!!!!!
    saved_model_file_name = model_types_to_saved_files[model_type]
    model.load_state_dict(torch.load(saved_model_file_name))
    return model


class Predictor:
    def __init__(self, model_type, conv_factor, dna_process_manager):
        self.model_type = model_type
        self.conv_factor = conv_factor
        self.dna_process_manager = dna_process_manager

    def __call__(self, input_data):
        processed_dna = self.dna_process_manager(input_data[:, 0])
        model = _load_network(self.model_type)
        predictions = model(processed_dna, input_data[:, 1:])
        Tm_prediction = nn_service.Tm_calculation(predictions[:, prediction_columns.dH_INDEX],
                                                  predictions[:, prediction_columns.dS_INDEX])
        return torch.cat((predictions, Tm_prediction[:, None]), dim=1)


