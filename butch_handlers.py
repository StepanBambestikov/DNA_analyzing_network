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


class complementarity_butch_adder:
    def __init__(self, butch_size, complementarity_additive_size, complementary_pair_maker):
        self.butch_size = butch_size
        self.complementarity_additive_size = complementarity_additive_size
        self.complementary_pair_maker = complementary_pair_maker

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
        return complementary_bonds_indexes
