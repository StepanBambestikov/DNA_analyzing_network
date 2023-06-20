import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy import stats
import pandas as pd
import network_service as nn_service


def _add_grid_to_plot(axis=None):
    if axis is None:
        plt.minorticks_on()
        plt.grid(which='major')
        plt.grid(which='minor', linestyle=':')
    else:
        axis.minorticks_on()
        axis.grid(which='major')
        axis.grid(which='minor', linestyle=':')


def _get_normal_distribution(begin_x, end_x, mean_value, standard_deviation, point_number=100):
    x_line = np.linspace(begin_x, end_x, point_number)
    normal_distribution = stats.norm.pdf(x_line, mean_value, standard_deviation)
    return x_line, normal_distribution


def _draw_one_histogram(axis, error_graph_name, error_graph_value, add_normal_distribution=True):
    _add_grid_to_plot(axis)
    mean_error, standard_deviation = stats.norm.fit(error_graph_value)
    axis.axvline(mean_error, color='k', linestyle='dashed')
    axis.axvline(mean_error + standard_deviation, color='y', linestyle='dashed')
    axis.axvline(mean_error - standard_deviation, color='y', linestyle='dashed')
    axis.hist(error_graph_value, density=True, bins=70, edgecolor='k', color='c',
              label=f'$\sigma = {standard_deviation:.2f}$')

    if add_normal_distribution is True:
        point_number = 100
        begin_x, end_x = axis.get_xlim()
        x_line, normal_distribution = _get_normal_distribution(begin_x, end_x, mean_error, standard_deviation, point_number)
        axis.plot(x_line, normal_distribution, 'k', linewidth=2)

    axis.set_ylabel('Probability')
    axis.set_xlabel(error_graph_name)
    axis.legend()


def _draw_pair_correlation(graph_pair, is_relative_error=True):
    graph_color = (np.random.random(), np.random.random(), np.random.random())
    first_graph, second_graph = graph_pair.get_numpy_data()
    plt.scatter(first_graph, second_graph, marker='o', s=4, color=graph_color)
    if first_graph.ndim > 1:
        first_graph, second_graph = first_graph[:, 0], second_graph[:, 0]
    m, b = np.polyfit(first_graph, second_graph, deg=1)
    first_tensor, second_tensor = graph_pair.get_tensors()
    mean_error = nn_service.calculate_mean_error(first_tensor, second_tensor, is_relative_error=is_relative_error)
    slope, intercept, r_value, p_value, std_err = stats.linregress(first_graph, second_graph)
    plt.axline(xy1=(0, b), slope=m, color=graph_color, label=f'{graph_pair.name}: $y = {m:.2f}x {b:+.2f},'
                                                             f'R2 = {r_value**2:.6f},$'
                                                             f'mean_error = {mean_error:.2f}')


class Graph:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def get_numpy_array(self):
        return self.value.detach().numpy()


def draw_error_histograms(figure_name, *error_graphs):
    graph_number = len(error_graphs)
    _, axis_list = plt.subplots(graph_number, 1)
    if graph_number == 1:
        axis_list.set_title(figure_name)
        _draw_one_histogram(axis_list, error_graphs[0].name, error_graphs[0].get_numpy_array())
    else:
        axis_list = axis_list.reshape(-1)
        axis_list[0].set_title(figure_name)
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

    def get_tensors(self):
        return self.first_tensor.detach(), self.second_tensor.detach()


def draw_correlation(figure_name, error_type_list, *correlation_pairs):
    plt.xlabel('Truth values')
    plt.ylabel('Predictions')
    _add_grid_to_plot()
    pairs_number = len(correlation_pairs)
    if pairs_number == 1:
        _draw_pair_correlation(correlation_pairs[0], error_type_list[0])
    else:
        for current_pair, current_error_is_relative in zip(correlation_pairs, error_type_list):
            _draw_pair_correlation(current_pair, current_error_is_relative)

    plt.title(figure_name)
    plt.legend()
    plt.show()


def save_data_in_file(predictions, ground_truth, file_name, pandas_load_method):
    data = pd.DataFrame(torch.cat((predictions, ground_truth), dim=1).detach().numpy(),
                        columns=['Prediction', 'Ground truth'])
    pandas_load_method(data, file_name, index=False, columns=['Prediction', 'Ground truth'])


def save_all_data_in_file(predictions, ground_truth, file_name, pandas_load_method):
    data = pd.DataFrame(torch.cat((predictions, ground_truth), dim=1).detach().numpy(),
                        columns=['Prediction', 'Ground truth'])
    pandas_load_method(data, file_name, index=False, columns=['Prediction', 'Ground truth'])