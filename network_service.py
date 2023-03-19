import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt


def get_prediction_error(predictions, ground_truth, is_relative_error=False):
    error = predictions - ground_truth
    if is_relative_error:
        error /= ground_truth
    return error


def _forward_propagation(model, butch, is_convolution_training=False):
    if is_convolution_training:
        model_prediction = model(butch[:, None, :])
    else:
        model_prediction = model(butch)
    return model_prediction


def _back_propagation(optimizer, loss_value):
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()


def compute_error(model, data_loader, loss_function, is_convolution_training=False, is_relative_error=False):
    error_accumulation = 0
    loss_accumulation = 0
    batch_count = 0
    for current_butch, ground_truth_labels in data_loader:
        model_prediction = _forward_propagation(model, current_butch, is_convolution_training)
        mean_model_error = torch.mean(get_prediction_error(model_prediction, ground_truth_labels, is_relative_error))
        error_accumulation += mean_model_error
        batch_count += 1
        loss_accumulation += loss_function(model_prediction, ground_truth_labels)
    average_error = (float)(error_accumulation / batch_count)
    average_loss = (float)(loss_accumulation / batch_count)
    return average_error, average_loss


class NetworkTrainService:
    def __init__(self, model, loss_function, optimizer, is_convolution_training, scheduler):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.is_convolution_training = is_convolution_training
        self.scheduler = scheduler
        self.min_lr_epoch_count = 0

    def one_butch_pass(self, butch, ground_truth_labels):
        model_prediction = _forward_propagation(self.model, butch, self.is_convolution_training)
        loss_value = self.loss_function(model_prediction, ground_truth_labels)
        _back_propagation(self.optimizer, loss_value)
        mean_model_error = torch.mean(get_prediction_error(model_prediction, ground_truth_labels))
        return loss_value, mean_model_error

    def scheduler_step(self, average_val_loss):
        if self.scheduler.optimizer.param_groups[0]['lr'] > 0.000002:
            self.scheduler.step(average_val_loss)  #reduce lr
            return True
        else:
            self.min_lr_epoch_count += 1
            if self.min_lr_epoch_count == 10:
                return False


def train_model(model, train_loader, val_loader, loss_function, optimizer, epoch_number, is_convolution_training=False,
                is_relative_error=False):
    train_loss_history = np.zeros(epoch_number)
    val_loss_history = np.zeros(epoch_number)
    train_error_history = np.zeros(epoch_number)
    val_error_history = np.zeros(epoch_number)
    scheduler = ReduceLROnPlateau(optimizer, 'min')  # reduce lr on a plateu
    #scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # reduce lr every n epochs

    min_lr_epoch_count = 0
    train_service = NetworkTrainService(model, loss_function, optimizer, is_convolution_training, scheduler)
    for current_epoch in range(epoch_number):
        model.train()
        loss_accumulation = 0
        error_accumulation = 0
        batch_count = 0
        for (current_butch, ground_truth_labels) in train_loader:
            loss_value, mean_model_error = train_service.one_butch_pass(current_butch, ground_truth_labels)
            error_accumulation += mean_model_error
            loss_accumulation += loss_value
            batch_count += 1

        average_loss = (float)(loss_accumulation / batch_count)
        epoch_train_error = (float)(error_accumulation / batch_count)
        epoch_val_error, average_val_loss = compute_error(model, val_loader, loss_function, is_convolution_training,
                                                          is_relative_error=is_relative_error)

        if not train_service.scheduler_step(average_val_loss):
            break

        train_loss_history[current_epoch] = average_loss
        val_loss_history[current_epoch] = average_val_loss
        train_error_history[current_epoch] = epoch_train_error
        val_error_history[current_epoch] = epoch_val_error

        print("Average test loss: %f, Average val loss: %f train error in one epoch: %f, validation error in one epoch: %f, current lr: %f" %
              (average_val_loss, average_val_loss, epoch_train_error, epoch_val_error, scheduler.optimizer.param_groups[0]['lr']))
    return train_loss_history, val_loss_history, train_error_history, val_error_history


def plot_train_history(train_loss_history, val_loss_history, train_error_history, val_error_history):
    # Initialise the subplot function using number of rows and columns
    _, (ax1, ax2) = plt.subplots(1, 2)

    x_line = np.linspace(0, len(train_loss_history), len(train_loss_history))
    # For Loss Function
    ax1.plot(x_line, train_loss_history, label='train loss')
    ax1.plot(x_line, val_loss_history, label='validation loss')
    ax1.set_title("Loss Function")

    # For Error Function
    ax2.plot(x_line, train_error_history, label='train error')
    ax2.plot(x_line, val_error_history, label='validation error')
    ax2.set_title("Error Function")

    ax1.legend(fontsize=14)
    ax2.legend(fontsize=14)
    ax1.minorticks_on()
    ax1.grid(which='major')
    ax1.grid(which='minor', linestyle=':')
    ax2.minorticks_on()
    ax2.grid(which='major')
    ax2.grid(which='minor', linestyle=':')
    plt.show()
