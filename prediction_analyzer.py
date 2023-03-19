import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import norm
import pandas as pd


class Graph:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def get_numpy_array(self):
        return self.value.detach().numpy()


def add_grid_to_plot(axis=None):
    if axis is None:
        plt.minorticks_on()
        plt.grid(which='major')
        plt.grid(which='minor', linestyle=':')
    else:
        axis.minorticks_on()
        axis.grid(which='major')
        axis.grid(which='minor', linestyle=':')


def get_normal_distribution(axis, mean_value, standard_deviation, point_number=100):
    begin_x, end_x = axis.get_xlim()
    x_line = np.linspace(begin_x, end_x, point_number)
    normal_distribution = norm.pdf(x_line, mean_value, standard_deviation)
    return x_line, normal_distribution


def _draw_one_histogram(axis, error_graph_name, error_graph_value):
    add_grid_to_plot(axis)
    mean_error, standard_deviation = norm.fit(error_graph_value)
    axis.axvline(mean_error, color='k', linestyle='dashed')
    axis.axvline(mean_error + standard_deviation, color='y', linestyle='dashed')
    axis.axvline(mean_error - standard_deviation, color='y', linestyle='dashed')

    axis.hist(error_graph_value, density=True, bins=70, edgecolor='k', color='c',
              label=f'$\sigma = {standard_deviation:.2f}$')

    point_number = 100
    x_line, normal_distribution = get_normal_distribution(axis, mean_error, standard_deviation, point_number)
    axis.plot(x_line, normal_distribution, 'k', linewidth=2)
    axis.set_ylabel('Probability')
    axis.set_xlabel(error_graph_name)
    axis.legend()


def draw_error_histograms(*error_graphs):
    graph_number = len(error_graphs)
    _, axis_list = plt.subplots(graph_number, 1)
    if graph_number == 1:
        _draw_one_histogram(axis_list, error_graphs[0].name, error_graphs[0].get_numpy_array())
    else:
        axis_list = axis_list.reshape(-1)
        for current_index, current_error_graph in enumerate(error_graphs):
            _draw_one_histogram(axis_list[current_index], current_error_graph.name,
                                current_error_graph.get_numpy_array())
    plt.show()


class CorrelationPair:
    def __init__(self, first_tensor, second_tensor, name):
        self.first_tensor = first_tensor
        self.second_tensor = second_tensor
        self.name = name

    def get_numpy_data(self):
        return self.first_tensor.detach().numpy(), self.second_tensor.detach().numpy()


def _draw_pair_correlation(graph_pair):
    graph_color = (np.random.random(), np.random.random(), np.random.random())
    first_graph, second_graph = graph_pair.get_numpy_data()
    plt.scatter(first_graph, second_graph, marker='o', s=4, color=graph_color)
    m, b = np.polyfit(first_graph[:, 0], second_graph[:, 0], deg=1)
    plt.axline(xy1=(0, b), slope=m, color=graph_color, label=f'{graph_pair.name}: $y = {m:.2f}x {b:+.2f}$')


def draw_correlation(*correlation_pairs):
    plt.xlabel('Truth values')
    plt.ylabel('Predictions')
    add_grid_to_plot()
    pairs_number = len(correlation_pairs)
    if pairs_number == 1:
        _draw_pair_correlation(correlation_pairs[0])
    else:
        for current_pair in correlation_pairs:
            _draw_pair_correlation(current_pair)
    plt.legend()
    plt.show()


def save_data_in_file(predictions, ground_truth, file_name, pandas_load_method):
    data = pd.DataFrame(torch.cat((predictions, ground_truth), dim=1).detach().numpy(),
                        columns=['Prediction', 'Ground truth'])
    pandas_load_method(data, file_name, index=False, columns=['Prediction', 'Ground truth'])
