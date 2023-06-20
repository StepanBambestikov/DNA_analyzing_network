import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import data_loader as loader
import network_classes as classes1
import classes2

begin_nn_column = 0
end_nn_column = 11
begin_Na_column = 12
end_Na_column = 18
Activity_column = 7 # activity number
Mg_column = 8 # activity number
Na_factor = 0.006


class conv2d_net_06_05(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(4, 4), padding=(0, 1))
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 2))
        self.conv_relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        linear_input = 300
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(linear_input, 3)
        )
        self.Na_net = classes2.Ne_conv_net(input_vector_size=linear_input + 8)

    def forward(self, input):
        dna_data, Na_data = input
        conv_out = self.conv(dna_data[:, None, :])
        dna_data = self.max_pool(conv_out)
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


# class Ne_Mg_conv_net(nn.Module):
#     def __init__(self, input_vector_size=4):
#         super().__init__()
#         self.fc = nn.Linear(input_vector_size, temp_layer_size).type(torch.float64)
#
#     def forward(self, Na_Mg_data):
#         result = self.fc1(Na_Mg_data)
#         result = self.fc2(result)
#         return result

# class Ne_Mg_conv_net(nn.Module):
#     def __init__(self, input_vector_size=4):
#         super().__init__()
#         temp_layer_size = int((input_vector_size / 4) * 3)
#         self.fc1 = nn.Linear(input_vector_size, temp_layer_size).type(torch.float64)
#         self.fc2 = nn.Linear(temp_layer_size, 2).type(torch.float64)
#
#     def forward(self, Na_Mg_data):
#         result = self.fc1(Na_Mg_data)
#         result = self.fc2(result)
#         return result

class _Ne_conv_net(nn.Module):
    def __init__(self, input_vector_size=4):
        super().__init__()
        self.fc = nn.Linear(input_vector_size, 8).type(torch.float64)

    def forward(self, nearest_neibours, Na_data):
        result = self.fc(torch.cat((nearest_neibours, Na_data), dim=1))
        return result



class conv2d_Mg_net_24_05(nn.Module):
    def __init__(self, conv2d_net_file_name):
        super().__init__()
        # load and froze conv2d_net
        self.conv2d_net = conv2d_net_06_05()
        self.conv2d_net.load_state_dict(torch.load(conv2d_net_file_name))
        self.salt_max_pool = nn.MaxPool3d((2, 1, 2))
        for param in self.conv2d_net.parameters():
            param.requires_grad = False
        self.Na_Mg_net = _Ne_conv_net(input_vector_size=70 + 9)

    def forward(self, input, prepare_salt=True):
        dna_data, Na_Mg_data = input
        if prepare_salt:
            Na = Na_Mg_data[:, -2] / 1000
            Mg = Na_Mg_data[:, -1] / 1000
            I = Na + 3 * Mg
            I_sqrt = np.sqrt(I)
            Na_Mg_data[:, -2] = Na * (10 ** (-0.509 * ((I_sqrt / (1 + I_sqrt)) - 0.2 * I)))
            Na_Mg_data[:, -1] = Mg * (10 ** (-0.509 * 4 * ((I_sqrt / (1 + I_sqrt)) - 0.2 * I)))
        else:
            Na_activity = Na_Mg_data[:, -2]
            Mg_activity = Na_Mg_data[:, -1]
        conv_out = self.conv2d_net.conv(dna_data[:, None, :])
        dna_data = self.conv2d_net.max_pool(conv_out)
        salt_dna_data = dna_data.clone()
        dna_data = self.conv2d_net.flatten(dna_data)
        dna_data = self.conv2d_net.conv_relu(dna_data)
        result = self.conv2d_net.linear_relu_stack(dna_data)

        salt_dna_data = self.salt_max_pool(salt_dna_data)
        salt_dna_data = self.conv2d_net.flatten(salt_dna_data)
        Na_out = self.Na_Mg_net(salt_dna_data.type(torch.float64), Na_Mg_data)



        # Na_dS_additive = Na_out[:, 0] * torch.log(Na_Mg_data[:, Activity_column]) + Na_out[:, 2]
        # Na_dG_additive = Na_out[:, 1] * torch.log(Na_Mg_data[:, Activity_column]) + Na_out[:, 3]
        # Mg_dS_additive = Na_out[:, 0] * torch.log(Na_Mg_data[:, Mg_column]) + Na_out[:, 2]
        # Mg_dG_additive = Na_out[:, 1] * torch.log(Na_Mg_data[:, Mg_column]) + Na_out[:, 3]
        Na_dS_additive = Na_out[:, 0] * torch.log(Na_Mg_data[:, -2]) + Na_out[:, 1]
        Na_dG_additive = Na_out[:, 2] * torch.log(Na_Mg_data[:, -2]) + Na_out[:, 3]
        Mg_dS_additive = Na_out[:, 4] * torch.log(Na_Mg_data[:, -1]) + Na_out[:, 5]
        Mg_dG_additive = Na_out[:, 6] * torch.log(Na_Mg_data[:, -1]) + Na_out[:, 7]

        dS_activity_additive = (Na_dS_additive + Mg_dS_additive) * Na_factor
        dG_activity_additive = (Na_dG_additive + Mg_dG_additive) * Na_factor
        # summary
        result[:, 2] += dS_activity_additive
        result[:, 1] += dG_activity_additive
        return result









class linear_net_06_05(nn.Module):
    def __init__(self):
        super().__init__()
        self.origin_net = classes1.multi_nn_2layer_net()
        self.Ne_net = classes2.Ne_conv_net(12 + 8)

    def forward(self, input):
        nearest_neibours, Na_data = input
        #Na args preparation
        Na_coeffitient = self.Ne_net(nearest_neibours, Na_data) # todo исправить потом
        dS_activity_additive = Na_coeffitient[:, 0] * torch.log(Na_data[:, Activity_column]) * Na_factor
        dG_activity_additive = Na_coeffitient[:, 1] * torch.log(Na_data[:, Activity_column]) * Na_factor
        #Main nn preparation
        result = self.origin_net(nearest_neibours)
        result[:, 2] += dS_activity_additive
        result[:, 1] += dG_activity_additive
        return result
