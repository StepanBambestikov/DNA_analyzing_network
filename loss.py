import torch


def normalized_l2_loss(prediction, target):
    loss_value = torch.mean(((prediction - target) / target) ** 2)
    return loss_value


def normalized_l1_loss(prediction, target):
    loss_value = torch.mean((abs((prediction - target) / target)))
    return loss_value


class complementary_normalized_l1_loss:
    def __init__(self, complement_coefficient):
        self.complement_coefficient = complement_coefficient

    def __call__(self, prediction, target, complementary_bonds):
        origin_rows_indices, complement_rows_indices = complementary_bonds[:, 0], complementary_bonds[:, 1]
        old_l1_loss = normalized_l1_loss(prediction, target)
        complement_loss_additive = self.complement_coefficient * torch.mean(abs(prediction[origin_rows_indices] -
                                                                                prediction[complement_rows_indices]))
        return old_l1_loss + complement_loss_additive
