import torch
import numpy as np
import matplotlib.pyplot as plt
import math


def forward_propagation(model, butch, is_convolution_training=False, just_predict=False):
    if just_predict is True:
        model.eval()
    model_prediction = model(butch)
    return model_prediction


def _back_propagation(optimizer, loss_value):
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()


def get_prediction_error(prediction_tensor, ground_truth_tensor, is_relative_error=False, is_absolute_value=True):
    if prediction_tensor.dim() > 1:  # todo хуёвый повтор кода
        error = prediction_tensor - ground_truth_tensor[:, :3]
        if is_absolute_value:
            error = torch.abs(error)
        if is_relative_error:
            error /= torch.abs(ground_truth_tensor[:, :3])
            error *= 100

    error = prediction_tensor - ground_truth_tensor
    if is_absolute_value:
        error = torch.abs(error)
    if is_relative_error:
        error /= torch.abs(ground_truth_tensor)
        error *= 100
    return error


DEFAULT_Ct = 1e-5


def Tm_calculation(dH_values, dS_values, is_celsius=True, Ct=DEFAULT_Ct):
    Tm_additive = 1.987 * math.log(Ct / 4)
    if is_celsius:
        celsius_additive = 273
    else:
        celsius_additive = 0
    Tm_values = ((dH_values * 1000) / (dS_values + Tm_additive)) - celsius_additive
    return Tm_values


def calculate_mean_error(model_prediction, ground_truth_labels, is_relative_error):
    if model_prediction.dim() > 1 and model_prediction.shape[1] > 1:
        Tm_column_prediction = Tm_calculation(model_prediction[:, 0], model_prediction[:, 2])
        mean_model_error = torch.mean(
            get_prediction_error(model_prediction, ground_truth_labels[:, :3], is_relative_error), dim=0)

        Tm_error = torch.mean(get_prediction_error(Tm_column_prediction, ground_truth_labels[:, 3], False),
                              dim=0)
        return torch.tensor([mean_model_error[0], mean_model_error[1], mean_model_error[2], Tm_error])  # todo мне похуй
    else:
        mean_model_error = torch.mean(get_prediction_error(model_prediction, ground_truth_labels, is_relative_error))
    return mean_model_error


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
        else:
            self.butch_handler = None

    def one_butch_pass(self, butch, ground_truth_labels):
        if self.butch_handler is not None:  # complex processing of butch before pass it to network
            butch_info = self.butch_handler(butch, ground_truth_labels)
            model_prediction = forward_propagation(self.model, butch, self.is_convolution_training)
            if self.butch_handler.has_butch_info is True:
                loss_value = self.loss_function(model_prediction, ground_truth_labels, butch_info)
            else:
                loss_value = self.loss_function(model_prediction, ground_truth_labels)
        else:  # pass butch without preprocessing
            model_prediction = forward_propagation(self.model, butch, self.is_convolution_training)
            loss_value = self.loss_function(model_prediction, ground_truth_labels)

        _back_propagation(self.optimizer, loss_value)
        mean_model_error = calculate_mean_error(model_prediction, ground_truth_labels, self.is_relative_error)
        return loss_value, mean_model_error

    def scheduler_step(self, average_val_loss):
        if self.scheduler.optimizer.param_groups[0]['lr'] > 0.0000002:
            self.scheduler.step(average_val_loss)  # reduce lr
            self.min_lr_epoch_count = 0
        else:
            self.min_lr_epoch_count += 1
            if self.min_lr_epoch_count == 100:
                return False
        return True

    def compute_error(self, data_loader):
        error_accumulation = 0
        loss_accumulation = 0
        batch_count = 0
        for current_butch, ground_truth_labels in next(data_loader):
            if self.butch_handler is not None:  # complex processing of butch before pass it to network
                butch_info = self.butch_handler(current_butch, ground_truth_labels)
                model_prediction = forward_propagation(self.model, current_butch, self.is_convolution_training,
                                                       just_predict=True)
                if self.butch_handler.has_butch_info is True:
                    loss_accumulation += self.loss_function(model_prediction, ground_truth_labels, butch_info)
                else:
                    loss_accumulation += self.loss_function(model_prediction, ground_truth_labels)
            else:
                model_prediction = forward_propagation(self.model, current_butch, self.is_convolution_training,
                                                       just_predict=True)
                loss_accumulation += self.loss_function(model_prediction, ground_truth_labels)

            mean_model_error = calculate_mean_error(model_prediction, ground_truth_labels, self.is_relative_error)

            error_accumulation += mean_model_error
            batch_count += 1
        average_error = error_accumulation / batch_count
        average_loss = (float)(loss_accumulation / batch_count)
        return average_error, average_loss


class epoch_painter:
    def __init__(self, drawing_function, epoch_number_per_drawing):
        self.drawing_function = drawing_function
        self.epoch_number_per_drawing = epoch_number_per_drawing
        self.current_epoch = 0

    def paint_epoch(self, *epoch_features):
        if self.current_epoch % self.epoch_number_per_drawing == self.epoch_number_per_drawing - 1:
            self.drawing_function(epoch_features)
        self.current_epoch += 1


class train_history_manager:
    def __init__(self, epoch_number):
        self.train_loss_history = np.zeros(epoch_number)
        self.val_loss_history = np.zeros(epoch_number)
        # self.train_error_history = [np.zeros(epoch_number)]
        # self.val_error_history = np.zeros(epoch_number)
        self.train_error_history = []
        self.val_error_history = []
        self.current_epoch = 0

    def add_epoch(self, average_loss, average_val_loss, epoch_train_error, epoch_val_error):
        self.train_loss_history[self.current_epoch] = average_loss
        self.val_loss_history[self.current_epoch] = average_val_loss
        # self.train_error_history[self.current_epoch] = epoch_train_error
        # self.val_error_history[self.current_epoch] = epoch_val_error
        self.train_error_history.append(epoch_train_error)
        self.val_error_history.append(epoch_val_error)
        self.current_epoch += 1

    def get_train_history(self):
        return self.train_loss_history[:self.current_epoch], self.val_loss_history[:self.current_epoch], \
            self.train_error_history[:self.current_epoch], self.val_error_history[:self.current_epoch]

    def get_min_val_error(self):
        return np.min(self.val_error_history[self.val_error_history > 0])

    def train_val_min_error_Tm(self):
        train_error, val_error = torch.stack(self.train_error_history[:self.current_epoch]), \
            torch.stack(self.val_error_history[:self.current_epoch])
        Tm_min_index = np.argmin(val_error[:, -1])
        train_Tm_min, val_Tm_min = train_error[Tm_min_index, :], val_error[Tm_min_index, :]
        return train_Tm_min, val_Tm_min


def train_model(train_manager, train_loader, val_loader, epoch_number, epoch_painter=None, history_manager=None):
    for current_epoch in range(epoch_number):
        train_manager.model.train()
        loss_accumulation = 0
        error_accumulation = 0
        batch_count = 0
        try:
            for current_butch, ground_truth_labels in next(train_loader):
                loss_value, mean_model_error = train_manager.one_butch_pass(current_butch, ground_truth_labels)
                error_accumulation += mean_model_error
                loss_accumulation += loss_value
                batch_count += 1

            average_loss = (float)(loss_accumulation / batch_count)
            epoch_train_error = error_accumulation / batch_count
            epoch_val_error, average_val_loss = train_manager.compute_error(val_loader)

            if not train_manager.scheduler_step(average_loss):
                break

            if history_manager is not None:
                history_manager.add_epoch(average_loss, average_val_loss, epoch_train_error, epoch_val_error)
            if epoch_val_error[3] < 1.40:
                raise KeyboardInterrupt
            if epoch_painter is not None:
                epoch_painter.paint_epoch({"epoch:": current_epoch, "t_loss:": average_loss, "v_loss:": average_val_loss,
                            "t_error:": epoch_train_error, "v_error:": epoch_val_error,
                            "lr": train_manager.scheduler.optimizer.param_groups[0]['lr']})
        except KeyboardInterrupt:
            break
    plot_train_history(*history_manager.get_train_history())

def plot_train_history(train_loss_history, val_loss_history, train_error_history, val_error_history):
    # Initialise the subplot function using number of rows and columns

    x_line = np.linspace(0, len(train_loss_history), len(train_loss_history))
    # For Loss Function
    plt.plot(x_line, train_loss_history, label='train loss')
    plt.plot(x_line, val_loss_history, label='validation loss')
    plt.title("Loss Function")


    plt.legend(fontsize=14)
    # ax2.legend(fontsize=14)
    plt.minorticks_on()
    plt.grid(which='major')
    plt.grid(which='minor', linestyle=':')
    plt.show()
