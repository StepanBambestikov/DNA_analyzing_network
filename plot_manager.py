import numpy as np
import matplotlib.pyplot as plt


def plot_train_history(train_loss_history, val_loss_history, train_error_history, val_error_history):
    # Initialise the subplot function using number of rows and columns
    f, (ax1, ax2) = plt.subplots(1, 2)

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
    # включаем основную сетку
    ax1.grid(which='major')
    # включаем дополнительную сетку
    ax1.grid(which='minor', linestyle=':')


    # включаем основную сетку
    ax2.grid(which='major')
    # включаем дополнительную сетку
    ax2.grid(which='minor', linestyle=':')

    plt.tight_layout()
    plt.show()