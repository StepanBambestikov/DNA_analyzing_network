import numpy as np
import torch.optim as optim
import torch
import data_loader as loader
import network_service as nn_service
import prediction_analyzer as pl

import network_classes as networks
import excel_parameters as constant
import pandas as pd
import loss


if __name__ == '__main__':
    butch_size = 64
    epoch_number = 500

    dataset = loader.get_dataset_from_excel_file("ML_Stepan.xlsx", label_column_number=constant.dH_column_number,
                                                 begin_feature_column=constant.begin_feature_column,
                                                 end_feature_column=constant.end_feature_column,
                                                 dna_to_numeric_strategy=None, first_row=constant.first_row)
    #dna data
    # dataset = ds.get_dataset_from_excel_file("ML_Stepan.xlsx", label_column_number=constant.dG_column_number,
    #                                          begin_feature_column=constant.dna_feature_column_number,
    #                                          end_feature_column=None,
    #                                          dna_to_numeric_strategy=ds.make_2d_data_from_text_dna,
    #                                          first_row=constant.first_row)



    train_loader, val_loader = loader.get_train_and_val_loaders(dataset, butch_size)

    nn_model = networks.SimpleNetwork()

    # loss = torch.nn.MSELoss().type(torch.FloatTensor)
    optimizer = optim.Adam(nn_model.parameters(), lr=0.001)
    train_loss_history, val_loss_history, train_error_history, val_error_history = nn_service.train_model(nn_model,
                                                                                                          train_loader, val_loader, loss.normalized_l2_loss, optimizer, epoch_number, is_convolution_training=True)
    print(np.min(val_error_history[val_error_history > 0]))
    nn_service.plot_train_history(train_loss_history, val_loss_history, train_error_history, val_error_history)

    nn_model = networks.SimpleNetwork()
    nn_model.load_state_dict(torch.load("dH_model"))
