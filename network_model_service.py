import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import numpy as np


def get_model_prediction(model, input_data):
    model.eval()
    model_prediction = model(input_data)
    return model_prediction


def compute_error(model, data_loader, loss_function, convolution_training=False):
    error_accumulation = 0
    loss_accumulation = 0
    batch_count = 0
    #todo this ^(
    for _, (current_butch, ground_truth_labels) in enumerate(data_loader):
        if convolution_training:
            model_prediction = model(current_butch[:, None, :])  # TODO clone?
        else:
            model_prediction = model(current_butch)
        mean_model_error = torch.mean(torch.abs(model_prediction - ground_truth_labels))
        error_accumulation += mean_model_error
        batch_count += 1
        loss_accumulation += loss_function(model_prediction, ground_truth_labels)
    average_error = (float)(error_accumulation / batch_count)
    average_loss = (float)(loss_accumulation / batch_count)
    return average_error, average_loss


def train_model(model, train_loader, val_loader, loss_function, optimizer, num_epochs, convolution_training=False):
    train_loss_history = np.zeros(num_epochs)
    val_loss_history = np.zeros(num_epochs)
    train_error_history = np.zeros(num_epochs)
    val_error_history = np.zeros(num_epochs)
    scheduler = ReduceLROnPlateau(optimizer, 'min')  # reduce lr on a plateu
    #scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # reduce lr every n epochs

    min_lr_epoch_count = 0
    for current_epoch in range(num_epochs):
        model.train()
        loss_accumulation = 0
        error_accumulation = 0
        batch_count = 0
        for _, (current_butch, ground_truth_labels) in enumerate(train_loader):
            if convolution_training:
                model_prediction = model(current_butch[:, None, :])  # TODO clone?
            else:
                model_prediction = model(current_butch)
            loss_value = loss_function(model_prediction, ground_truth_labels)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            mean_model_error = torch.mean(torch.abs(model_prediction - ground_truth_labels))
            error_accumulation += mean_model_error
            loss_accumulation += loss_value
            batch_count += 1

        average_loss = (float)(loss_accumulation / batch_count)
        epoch_train_error = (float)(error_accumulation / batch_count)
        epoch_val_error, average_val_loss = compute_error(model, val_loader, loss_function, convolution_training)

        if scheduler.optimizer.param_groups[0]['lr'] > 0.000002:
            scheduler.step(average_val_loss)  #reduce lr
        else:
            min_lr_epoch_count += 1
            if min_lr_epoch_count == 10:
                break

        train_loss_history[current_epoch] = average_loss
        val_loss_history[current_epoch] = average_val_loss
        train_error_history[current_epoch] = epoch_train_error
        val_error_history[current_epoch] = epoch_val_error

        print("Average test loss: %f, Average val loss: %f train error in one epoch: %f, validation error in one epoch: %f, current lr: %f" %
              (average_val_loss, average_val_loss, epoch_train_error, epoch_val_error, scheduler.optimizer.param_groups[0]['lr']))
    return train_loss_history, val_loss_history, train_error_history, val_error_history
