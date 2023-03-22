import torch
import numpy as np
import matplotlib.pyplot as plt


def _forward_propagation(model, butch, is_convolution_training=False, just_predict=False):
    if just_predict is True:
        model.eval()
    if is_convolution_training:
        model_prediction = model(butch[:, None, :])
    else:
        model_prediction = model(butch)
    return model_prediction


def _back_propagation(optimizer, loss_value):
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()


def get_prediction_error(prediction_tensor, ground_truth_tensor, is_relative_error=False, is_absolute_value=True):
    error = prediction_tensor - ground_truth_tensor
    if is_absolute_value:
        error = torch.abs(error)
    if is_relative_error:
        error /= torch.abs(ground_truth_tensor)
        error *= 100
    return error


class network_train_service:
    def __init__(self, model, loss_function, optimizer, scheduler, butch_handler=None, is_convolution_training=False,
                 is_relative_error=False):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.is_convolution_training = is_convolution_training
        self.scheduler = scheduler
        self.min_lr_epoch_count = 0
        self.is_relative_error = is_relative_error
        if butch_handler is not None:
            self.butch_handler = butch_handler

    def one_butch_pass(self, butch, ground_truth_labels):
        if self.butch_handler is not None:  # complex processing of butch before pass it to network
            butch_info = self.butch_handler(butch, ground_truth_labels)
            model_prediction = _forward_propagation(self.model, butch, self.is_convolution_training)
            loss_value = self.loss_function(model_prediction, ground_truth_labels, butch_info)
        else:  # pass butch without preprocessing
            model_prediction = _forward_propagation(self.model, butch, self.is_convolution_training)
            loss_value = self.loss_function(model_prediction, ground_truth_labels)

        _back_propagation(self.optimizer, loss_value)
        mean_model_error = torch.mean(get_prediction_error(model_prediction, ground_truth_labels, self.is_relative_error))
        return loss_value, mean_model_error

    def scheduler_step(self, average_val_loss):
        if self.scheduler.optimizer.param_groups[0]['lr'] > 0.0000002:
            self.scheduler.step(average_val_loss)  # reduce lr
            self.min_lr_epoch_count = 0
        else:
            self.min_lr_epoch_count += 1
            if self.min_lr_epoch_count == 1000:
                return False
        return True

    def compute_error(self, data_loader, is_relative_error=False):
        error_accumulation = 0
        loss_accumulation = 0
        batch_count = 0
        for current_butch, ground_truth_labels in data_loader:
            if self.butch_handler is not None:  # complex processing of butch before pass it to network
                butch_info = self.butch_handler(current_butch, ground_truth_labels)
                model_prediction = _forward_propagation(self.model, current_butch, self.is_convolution_training,
                                                        just_predict=True)
                loss_accumulation += self.loss_function(model_prediction, ground_truth_labels, butch_info)
            else:
                model_prediction = _forward_propagation(self.model, current_butch, self.is_convolution_training,
                                                        just_predict=True)
                loss_accumulation += self.loss_function(model_prediction, ground_truth_labels)
            mean_model_error = torch.mean(
                get_prediction_error(model_prediction, ground_truth_labels, is_relative_error))
            error_accumulation += mean_model_error
            batch_count += 1
        average_error = (float)(error_accumulation / batch_count)
        average_loss = (float)(loss_accumulation / batch_count)
        return average_error, average_loss


class train_event_painter:
    def __init__(self, drawing_function, epoch_number_per_drawing):
        self.drawing_function = drawing_function
        self.epoch_number_per_drawing = epoch_number_per_drawing
        self.current_epoch = 0

    def step(self, *event_features_dict):
        if self.current_epoch % self.epoch_number_per_drawing == self.epoch_number_per_drawing - 1:
            self.drawing_function(event_features_dict)
        self.current_epoch += 1


class train_history_manager:
    def __init__(self, epoch_number):
        self.train_loss_history = np.zeros(epoch_number)
        self.val_loss_history = np.zeros(epoch_number)
        self.train_error_history = np.zeros(epoch_number)
        self.val_error_history = np.zeros(epoch_number)
        self.current_epoch = 0

    def step(self, average_loss, average_val_loss, epoch_train_error, epoch_val_error):
        self.train_loss_history[self.current_epoch] = average_loss
        self.val_loss_history[self.current_epoch] = average_val_loss
        self.train_error_history[self.current_epoch] = epoch_train_error
        self.val_error_history[self.current_epoch] = epoch_val_error
        self.current_epoch += 1

    def get_train_history(self):
        return self.train_loss_history[:self.current_epoch], self.val_loss_history[:self.current_epoch], \
            self.train_error_history[:self.current_epoch], self.val_error_history[:self.current_epoch]

    def get_min_val_error(self):
        return np.min(self.val_error_history)


def train_model(train_manager, train_loader, val_loader, epoch_number, epoch_painter=None, history_manager=None):
    for current_epoch in range(epoch_number):
        train_manager.model.train()
        loss_accumulation = 0
        error_accumulation = 0
        batch_count = 0
        for (current_butch, ground_truth_labels) in train_loader:
            loss_value, mean_model_error = train_manager.one_butch_pass(current_butch, ground_truth_labels)
            error_accumulation += mean_model_error
            loss_accumulation += loss_value
            batch_count += 1

        average_loss = (float)(loss_accumulation / batch_count)
        epoch_train_error = (float)(error_accumulation / batch_count)
        epoch_val_error, average_val_loss = train_manager.compute_error(val_loader)

        if not train_manager.scheduler_step(average_val_loss):
            break

        if history_manager is not None:
            history_manager.step(average_loss, average_val_loss, epoch_train_error, epoch_val_error)
        if epoch_painter is not None:
            epoch_painter.step({"epoch:": current_epoch, "t_loss:": average_loss, "v_loss:": average_val_loss,
                            "t_error:": epoch_train_error, "v_error:": epoch_val_error,
                            "lr": train_manager.scheduler.optimizer.param_groups[0]['lr']})


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
