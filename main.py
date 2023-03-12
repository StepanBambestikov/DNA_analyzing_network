import torch
import torch.optim as optim
import torch.nn as nn

import dataset_maker as ds
import data_loader as loaders_handler
import network_model_service as nn_service
import plot_manager as pl

import network_classes as networks
import parameters as constant

if __name__ == '__main__':
    butch_size = 64
    epoch_number = 500

    dataset = ds.get_dataset_from_excel_file("ML_Stepan.xlsx", label_column_number=constant.dG_column_number,
                                             begin_feature_column=constant.begin_feature_column,
                                             end_feature_column=constant.end_feature_column,
                                             dna_to_numeric_strategy=None, first_row=constant.first_row)
    #dna data
    # dataset = ds.get_dataset_from_excel_file("ML_Stepan.xlsx", label_column_number=constant.dH_column_number,
    #                                          begin_feature_column=constant.dna_feature_column_number,
    #                                          end_feature_column=None,
    #                                          dna_to_numeric_strategy=ds.make_2d_data_from_text_dna,
    #                                          first_row=constant.first_row)



    train_loader, val_loader = loaders_handler.get_train_and_val_loaders(dataset, butch_size)

    nn_model = networks.SimpleNetwork()
    loss = torch.nn.MSELoss().type(torch.FloatTensor)
    optimizer = optim.Adam(nn_model.parameters(), lr=0.001)
    train_loss_history, val_loss_history, train_error_history, val_error_history = nn_service.train_model(nn_model,
                                                                train_loader, val_loader, loss, optimizer, epoch_number)
    pl.plot_train_history(train_loss_history, val_loss_history, train_error_history, val_error_history)