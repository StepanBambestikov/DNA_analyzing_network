import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import data_loader as loader
import network_classes as classes1


class Ne_net(nn.Sequential):
    """
    mini net for Ne adding to main net
    """
    def __init__(self, input_vector_size=4):
        middle_vector_size = int(input_vector_size * 4 // 7)
        super().__init__(
            nn.Linear(input_vector_size, 2)
            # nn.ReLU(inplace=True),
            # nn.Linear(middle_vector_size, 1)
        )
        super().type(torch.float64)


class Ne_conv_net(nn.Module):
    def __init__(self, input_vector_size=4):
        super().__init__()
        self.fc = nn.Linear(input_vector_size, 2).type(torch.float64)

    def forward(self, nearest_neibours, Na_data):
        result = self.fc(torch.cat((nearest_neibours, Na_data), dim=1))
        return result


begin_nn_column = 0
end_nn_column = 11
begin_Na_column = 12
end_Na_column = 18
Activity_column = 7 # activity number
Na_factor = 0.006


class Ne_multi_nn_2layer_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.origin_net = classes1.multi_nn_2layer_net()
        self.Ne_net = Ne_conv_net()

    def forward(self, input):
        nearest_neibours, Na_data = input
        #Na args preparation
        Na_coeffitient = self.Ne_model(nearest_neibours, Na_data) # todo исправить потом
        dS_activity_additive = Na_coeffitient[:, 0] * torch.log(Na_data[:, Activity_column]) * Na_factor
        dG_activity_additive = Na_coeffitient[:, 1] * torch.log(Na_data[:, Activity_column]) * Na_factor
        #Main nn preparation
        result = self.origin_net(nearest_neibours)
        result[:, 2] += dS_activity_additive
        result[:, 1] += dG_activity_additive
        return result


class Ne_multi_conv2d_paral_layers_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.origin_net = classes1.multi_conv2d_paral_layers_net()
        self.Ne_model = Ne_net(input_vector_size=7)

    def forward(self, input):
        dna_data, Na_data = input
        #Main nn preparation
        origin_out = self.origin_net(dna_data[:, None, :])
        #Na args preparation
        Na_coeffitient = self.Ne_model(origin_out, Na_data[:, :Activity_column])
        dS_activity_additive = Na_coeffitient[:, 0] * torch.log(Na_data[:, Activity_column]) * Na_factor
        dG_activity_additive = Na_coeffitient[:, 1] * torch.log(Na_data[:, Activity_column]) * Na_factor
        #summary
        origin_out[:, 2] += dS_activity_additive
        origin_out[:, 1] += dG_activity_additive
        return origin_out


class Ne_multi_conv2d_net_28_04(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(4, 1), padding=(0, 0))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=(4, 2), padding=(0, 0))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(4, 3), padding=(0, 0))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(4, 4), padding=(0, 1))
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 2))
        self.conv_relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        linear_input = 300
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(linear_input, 3)
        )
        self.Na_net = Ne_conv_net(input_vector_size=linear_input + 8)

    def forward(self, input):
        dna_data, Na_data = input
        conv0_out = self.conv0(dna_data[:, None, :])
        conv1_out = self.conv1(dna_data[:, None, :])
        conv2_out = self.conv2(dna_data[:, None, :])
        conv3_out = self.conv3(dna_data[:, None, :])

        conv1_out = F.pad(conv1_out, (1, 0, 0, 0), mode='replicate') # todo add ker of size 3
        conv2_out = F.pad(conv2_out, (2, 0, 0, 0), mode='replicate')
        conv3_out = F.pad(conv3_out, (1, 0, 0, 0), mode='replicate')
        full_conv_out = torch.cat([conv3_out], dim=1)
        dna_data = self.max_pool(full_conv_out)
        dna_data = self.flatten(dna_data)
        Na_out = self.Na_net(dna_data.type(torch.float64), Na_data)

        dna_data = self.conv_relu(dna_data)
        result = self.linear_relu_stack(dna_data)

        dS_activity_additive = Na_out[:, 0] * torch.log(Na_data[:, Activity_column]) * Na_factor
        dG_activity_additive = Na_out[:, 1] * torch.log(Na_data[:, Activity_column]) * Na_factor
        # summary
        result[:, 2] += dS_activity_additive
        result[:, 1] += dG_activity_additive
        return result


# class Ne_multi_conv2d_net_28_04(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv0 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(4, 1), padding=(0, 0))
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=(4, 2), padding=(0, 0))
#         self.conv2 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(4, 3), padding=(0, 0))
#         self.conv3 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(4, 4), padding=(0, 1))
#         self.max_pool = nn.MaxPool2d(kernel_size=(1, 2))
#         self.conv_relu = nn.ReLU(inplace=True)
#         self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
#         linear_input = 480
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(linear_input, 3),
#         )
#         self.Na_net = Ne_conv_net(input_vector_size=linear_input + 8)
#
#     def forward(self, input):
#         dna_data, Na_data = input
#         conv0_out = self.conv0(dna_data[:, None, :])
#         conv1_out = self.conv1(dna_data[:, None, :])
#         conv2_out = self.conv2(dna_data[:, None, :])
#         conv3_out = self.conv3(dna_data[:, None, :])
#
#         conv1_out = F.pad(conv1_out, (1, 0, 0, 0), mode='replicate') # todo add ker of size 3
#         conv2_out = F.pad(conv2_out, (2, 0, 0, 0), mode='replicate')
#         conv3_out = F.pad(conv3_out, (1, 0, 0, 0), mode='replicate')
#         full_conv_out = torch.cat([conv1_out, conv3_out], dim=1)
#         dna_data = self.max_pool(full_conv_out)
#         dna_data = self.flatten(dna_data)
#         Na_out = self.Na_net(dna_data.type(torch.float64), Na_data)
#
#         dna_data = self.conv_relu(dna_data)
#         result = self.linear_relu_stack(dna_data)
#
#         dS_activity_additive = Na_out[:, 0] * torch.log(Na_data[:, Activity_column]) * Na_factor
#         dG_activity_additive = Na_out[:, 1] * torch.log(Na_data[:, Activity_column]) * Na_factor
#         # summary
#         result[:, 2] += dS_activity_additive
#         result[:, 1] += dG_activity_additive
#         return result




class Ne_multi_nn_2layer_net_28_04(nn.Module):
    def __init__(self):
        super().__init__()
        self.origin_net = classes1.multi_nn_2layer_net()
        self.Ne_net = Ne_conv_net(input_vector_size=12 + 8)

    def forward(self, input):
        nearest_neibours, Na_data = input
        #Na args preparation
        Na_coeffitient = self.Ne_net(nearest_neibours, Na_data)
        dS_activity_additive = Na_coeffitient[:, 0] * torch.log(Na_data[:, Activity_column]) * Na_factor
        dG_activity_additive = Na_coeffitient[:, 1] * torch.log(Na_data[:, Activity_column]) * Na_factor
        #Main nn preparation
        result = self.origin_net(nearest_neibours)
        result[:, 2] += dS_activity_additive
        result[:, 1] += dG_activity_additive
        return result

#ниже для курсача
class _multi_nn_0layer_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.origin_net = classes1.multi_nn_1layer_net()

    def forward(self, input):
        nearest_neibours, Na_data = input
        #Na args preparation
        result = self.origin_net(nearest_neibours)
        return result


class _multi_nn_1layer_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.origin_net = classes1.multi_nn_2layer_net()

    def forward(self, input):
        nearest_neibours, Na_data = input
        #Na args preparation
        result = self.origin_net(nearest_neibours)
        return result


class _multi_nn_2layer_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.origin_net = classes1.multi_nn_3layer_net()

    def forward(self, input):
        nearest_neibours, Na_data = input
        #Na args preparation
        result = self.origin_net(nearest_neibours)
        return result


class _multi_conv2d_net_12_20(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(4, 1), padding=(0, 0))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=(4, 2), padding=(0, 0))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(4, 3), padding=(0, 0))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(4, 4), padding=(0, 1))
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 4))
        self.conv_relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        linear_input = 182
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(linear_input, 3)
        )

    def forward(self, input):
        dna_data, Na_data = input
        conv0_out = self.conv0(dna_data[:, None, :])
        conv1_out = self.conv1(dna_data[:, None, :])
        conv2_out = self.conv2(dna_data[:, None, :])
        conv3_out = self.conv3(dna_data[:, None, :])

        conv1_out = F.pad(conv1_out, (1, 0, 0, 0), mode='replicate') # todo add ker of size 3
        conv2_out = F.pad(conv2_out, (2, 0, 0, 0), mode='replicate')
        conv3_out = F.pad(conv3_out, (1, 0, 0, 0), mode='replicate')
        full_conv_out = torch.cat([conv1_out, conv3_out], dim=1)
        dna_data = self.max_pool(full_conv_out)
        dna_data = self.flatten(dna_data)
        dna_data = self.conv_relu(dna_data)
        result = self.linear_relu_stack(dna_data)
        return result
