from enum import IntEnum
import predictor
import network_classes as nn_classes
import data_loader

#for testing
import torch


linear_predictor = predictor.Predictor(nn_classes.two_layer_network, conv_factor=False,
                                       dna_process_manager=data_loader.make_2d_data_from_text_dna)


test_predictor = predictor.Predictor(predictor.make_test_network, conv_factor=False,
                                       dna_process_manager=data_loader.make_2d_data_from_text_dna)


predictor_to_object = {
    Predictor_types.LINEAR_PREDICTOR: linear_predictor,
    Predictor_types.TEST_PREDICTOR: test_predictor
    # Predictor_types.CONV_PREDICTOR: conv_predictor,
}