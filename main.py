import numpy as np
import torch.optim as optim
import torch.nn as nn
import data_loader as loader
import network_service as nn_service
import prediction_analyzer as pl

import network_classes as networks
import excel_parameters as constant
import pandas as pd
import loss
import butch_handlers
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

if __name__ == '__main__':
    butch_size = 64 + 32
    # dataset = loader.get_dataset_from_excel_file("ML_Stepan.xlsx", label_column_number=constant.dH_column_number,
    #                                              begin_feature_column=constant.begin_feature_column,
    #                                              end_feature_column=constant.end_feature_column,
    #                                              dna_to_numeric_strategy=None, first_row=constant.first_row)
    # train_loader, val_loader = loader.get_train_and_val_loaders(dataset, butch_size)
    #dna data
    dataset = loader.get_dataset_from_excel_file("ML_Stepan.xlsx", label_column_number=constant.dH_column_number,
                                             begin_feature_column=constant.dna_feature_column_number,
                                             end_feature_column=None,
                                             dna_to_numeric_strategy=loader.make_2d_data_from_text_dna,
                                             first_row=constant.first_row)
    # train service
    train_loader, val_loader = loader.get_train_and_val_loaders(dataset, butch_size)
    epoch_number = 10000
    nn_model = networks.conv_2d_1fc()
    loss_function = loss.complementary_normalized_l1_loss(complement_coefficient=0.3)
    butch_handler = butch_handlers.complementarity_butch_adder(
        butch_size=butch_size, complementarity_additive_size=32,
        complementary_pair_maker=butch_handlers.complementary_pair_maker_from_2d)
    optimizer = optim.Adam(nn_model.parameters(), lr=0.001)
    history_manager = nn_service.train_history_manager(epoch_number)
    epoch_painter = nn_service.train_event_painter(print, epoch_number_per_drawing=10)
    # scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # reduce lr every n epochs
    scheduler = ReduceLROnPlateau(optimizer, 'min')  # reduce lr on a plateu

    train_manager = nn_service.network_train_service(nn_model, loss_function, optimizer, scheduler,
                                                     butch_handler=butch_handler,
                                                     is_convolution_training=True,
                                                     is_relative_error=True)
    # training
    nn_service.train_model(train_manager, train_loader, val_loader, epoch_number, epoch_painter, history_manager)
    # print(np.min(val_error_history[val_error_history > 0]))
    print(history_manager.get_min_val_error())
    nn_service.plot_train_history(history_manager.get_train_history())

    # testing
    test_features, test_labels = dataset.tensors
    prediction = nn_model(test_features[:, None, :])
    absolute_error = nn_service.get_prediction_error(prediction, test_labels, is_relative_error=False,
                                                     is_absolute_value=False)
    relative_error = nn_service.get_prediction_error(prediction, test_labels, is_relative_error=True,
                                                     is_absolute_value=False)
    pl.draw_error_histograms(pl.Graph("absolute error", absolute_error), pl.Graph("relative error", relative_error))
    pl.draw_correlation(pl.CorrelationPair(prediction, test_labels, "test"))
