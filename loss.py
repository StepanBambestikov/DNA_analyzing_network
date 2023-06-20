import torch
import network_service as nn_service


def normalized_l2_loss(prediction, target):
    loss_value = torch.mean(((prediction - target) / target) ** 2)
    return loss_value


def normalized_l1_loss(prediction, target, get_relative_error=True):
    if get_relative_error:
        loss_value = torch.mean((abs((prediction - target) / target)))
    else:
        loss_value = torch.mean(abs((prediction - target)))
    return loss_value


class complementary_normalized_l1_loss:
    def __init__(self, complement_coefficient):
        self.complement_coefficient = complement_coefficient

    def __call__(self, prediction, target, complementary_bonds):
        origin_rows_indices, complement_rows_indices = complementary_bonds[:, 0], complementary_bonds[:, 1]
        old_l1_loss = normalized_l1_loss(prediction, target)
        target_max_difference = torch.max(target) - torch.min(target)
        prediction_max_difference = torch.max(prediction) - torch.min(prediction)

        complement_loss_additive = self.complement_coefficient * torch.mean(abs(prediction[origin_rows_indices] -
                                                                                prediction[complement_rows_indices]))
        complement_loss_additive *= target_max_difference / prediction_max_difference
        return old_l1_loss + complement_loss_additive


def normalized_l1_multi_loss(prediction, target):
    dH_loss = normalized_l1_loss(prediction[:, 0], target[:, 0])
    dG_loss = normalized_l1_loss(prediction[:, 1], target[:, 1])
    dS_loss = normalized_l1_loss(prediction[:, 2], target[:, 2])

    Tm_column_prediction = nn_service.Tm_calculation(prediction[:, 0], prediction[:, 2])
    Tm_loss = normalized_l1_loss(Tm_column_prediction, target[:, 3])
    # return dH_loss * 1.2 + dG_loss + dS_loss * 1.3 + Tm_loss * 0.7
    # return dH_loss * 1.0 + dG_loss + dS_loss * 1.0 + Tm_loss * 0.3 # used for 1layer, 2layer net
    return dH_loss * 2 + dG_loss * 1.0 + dS_loss * 0.5 + Tm_loss * 1.2 #used for 3layer net


def normalized_l1_multi_loss_06_05_absolute_Tm(prediction, target):
    dH_loss = normalized_l1_loss(prediction[:, 0], target[:, 0])
    dG_loss = normalized_l1_loss(prediction[:, 1], target[:, 1])
    dS_loss = normalized_l1_loss(prediction[:, 2], target[:, 2])

    Tm_column_prediction = nn_service.Tm_calculation(prediction[:, 0], prediction[:, 2], is_celsius=True)
    Tm_loss = normalized_l1_loss(Tm_column_prediction, target[:, 3], get_relative_error=False)
    # return dH_loss * 1.2 + dG_loss + dS_loss * 1.3 + Tm_loss * 0.7
    # return dH_loss * 1.0 + dG_loss + dS_loss * 1.0 + Tm_loss * 0.3 # used for 1layer, 2layer net
    return dH_loss * 2 + dG_loss * 1.0 + dS_loss * 0.5 + Tm_loss * 0.05  #used for 3layer net


def normalized_l1_multi_loss_30_04_relative_Tm(prediction, target):
    dH_loss = normalized_l1_loss(prediction[:, 0], target[:, 0])
    dG_loss = normalized_l1_loss(prediction[:, 1], target[:, 1])
    dS_loss = normalized_l1_loss(prediction[:, 2], target[:, 2])

    Tm_column_prediction = nn_service.Tm_calculation(prediction[:, 0], prediction[:, 2], is_celsius=False)
    Tm_loss = normalized_l1_loss(Tm_column_prediction, target[:, 3] + 273, get_relative_error=True)

    if dH_loss < 0.06:
        # adding similarity of error dH,dS with Tm loss
        Tm_dS_dH_error_similarity = Tm_loss - (dH_loss + dS_loss)
        return dH_loss * 2 + dG_loss * 1.0 + dS_loss * 0.5 + Tm_loss * 1.2 + 10 * Tm_dS_dH_error_similarity # used for 3layer net
    # return dH_loss * 2 + dG_loss * 1.0 + dS_loss * 0.5 + Tm_loss * 1.2  #used for 3layer net
    return dH_loss * 2 + dG_loss * 1.0 + dS_loss * 0.5 + Tm_loss * 1.2  #used for 3layer net
