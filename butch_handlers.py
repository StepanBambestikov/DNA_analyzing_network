import numpy as np
from dna_enumeration import BASE_1D, BASE_2D
import torch


def complementary_pair_maker_from_2d(dna_row):
    padding = 1
    dna_row_copy = dna_row.detach().clone()

    dna_row_copy[BASE_2D.A + padding] = dna_row[BASE_2D.T + padding]
    dna_row_copy[BASE_2D.C + padding] = dna_row[BASE_2D.G + padding]
    dna_row_copy[BASE_2D.T + padding] = dna_row[BASE_2D.A + padding]
    dna_row_copy[BASE_2D.G + padding] = dna_row[BASE_2D.C + padding]
    return torch.flip(dna_row_copy, dims=(1,))


class complementarity_butch_manager:
    def __init__(self, butch_size, complementarity_additive_size, complementary_pair_maker):
        self.butch_size = butch_size
        self.complementarity_additive_size = complementarity_additive_size
        self.complementary_pair_maker = complementary_pair_maker
        self.has_butch_info = False

    def __call__(self, butch, ground_truth):
        butch_size = len(butch)
        butch_indices = list(range(butch_size))
        np.random.shuffle(butch_indices)
        additive_size = butch_size // 2 if butch_size < self.complementarity_additive_size * 2 \
            else self.complementarity_additive_size

        origin_rows_indices = butch_indices[0: additive_size]
        complementary_rows_indices = butch_indices[additive_size:
                                                   additive_size * 2]

        for complement_row_index, origin_row_index in zip(complementary_rows_indices, origin_rows_indices):
            butch[complement_row_index] = self.complementary_pair_maker(butch[origin_row_index])
            ground_truth[complement_row_index] = ground_truth[origin_row_index]

        complementary_bonds_indexes = np.column_stack((origin_rows_indices, complementary_rows_indices))

        self.has_butch_info = True
        return complementary_bonds_indexes


class complementarity_butch_filler:
    def __init__(self, butch_size, complementarity_additive_size, complementary_pair_maker):
        self.butch_manager = complementarity_butch_manager(butch_size, complementarity_additive_size,
                                                           complementary_pair_maker)
        self.has_butch_info = False

    def __call__(self, butch, ground_truth):
        self.butch_manager(butch, ground_truth)


class butch_noise_adder:
    def __init__(self, mean, sigma, noise_ground_truth=False, noise_butch=False):
        self.mean = mean
        self.sigma = sigma
        self.has_butch_info = False
        self.noise_ground_truth = noise_ground_truth
        self.noise_butch = noise_butch

    def _noise_array(self, array):
        numpy_array = array.detach().numpy()
        butch_noise_additive = np.random.normal(self.mean, self.sigma, size=numpy_array.shape)
        # for butch_row in range(butch.shape[0]):
        #     butch_noise_additive = np.random.normal(self.mean, self.sigma, size=butch_numpy.shape[1])
        #     butch[butch_row] += butch_noise_additive
        numpy_array += butch_noise_additive

    def __call__(self, butch, ground_truth):
        array_list = []
        if self.noise_ground_truth is True:
            array_list.append(ground_truth)
        if self.noise_butch is True:
            array_list.append(butch)

        for current_array in array_list:
            self._noise_array(current_array)
