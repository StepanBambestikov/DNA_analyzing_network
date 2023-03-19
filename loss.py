import torch


def normalized_l2_loss(output, target):
    loss_value = torch.mean(((output - target) / target)**2)
    return loss_value


def normalized_l1_loss(output, target):
    loss_value = torch.mean((abs((output - target) / target)))
    return loss_value
