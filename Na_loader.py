import random


def add_axis_to_one_dimentional_arrays(*array_list):
    for current_index in range(len(array_list)):
        if len(array_list[current_index].shape) == 1:
            array_list[current_index] = array_list[current_index][:, None]


class Na_loader:
    def __init__(self, features_list, labels, indices, butch_size=None):
        self.features_list = features_list
        self.labels = labels
        self.indices = indices
        if butch_size is not None:
            self.butch_size = butch_size
        else:
            self.butch_size = len(self.indices)

    def __iter__(self):
        return self

    def __next__(self):
        random.shuffle(self.indices)
        for current_begin_butch_index in range(0, len(self.indices), self.butch_size):
            begin_index = current_begin_butch_index
            if len(self.indices) - current_begin_butch_index < self.butch_size:
                end_index = current_begin_butch_index + len(self.indices) - current_begin_butch_index
            else:
                end_index = current_begin_butch_index + self.butch_size
            butch_feature_list = []
            for current_features in self.features_list:
                butch_feature_list.append(current_features[self.indices[begin_index: end_index], :])
            butch_labels = self.labels[self.indices[begin_index: end_index], :]
            yield butch_feature_list, butch_labels


def multi_splitter(features_list, labels, train_indices, val_indices, butch_size=None):
    train_loader = Na_loader(features_list, labels, train_indices, butch_size)
    val_loader = Na_loader(features_list, labels, val_indices, butch_size)

    return train_loader, val_loader
