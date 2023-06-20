import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------------------------------------------
# multi full layer network classes:
class multi_nn_1layer_net(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(12, 3),
        )
        super().type(torch.float64)


class multi_nn_2layer_net(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(12, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 3)
        )
        super().type(torch.float64)


class multi_nn_3layer_net(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(12, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 3)
        )
        super().type(torch.float64)

# multi convolution network classes:

#multi convolution network with 1 conv layer
class multi_conv2d_1layer_ker2_net(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(6, 2)),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(400, 3)
        )


class multi_conv2d_1layer_ker3_net(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(6, 3)),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(380, 3)
        )


class multi_conv2d_1layer_ker4_net(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(4, 4)),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(1260, 3)
        )

# multi convolution network with 2 conv layer


class multi_conv2d_paral_layers_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=(4, 2), padding=(0, 0))
        # self.conv2 = nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(4, 3), padding=(0, 0))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(4, 4), padding=(0, 1))
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 2))
        self.conv_relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(420, 3)
        )

    def forward(self, input):
        conv1_out = self.conv1(input)
        # conv2_out = self.conv2(input)
        # conv2_out = F.pad(conv2_out, (0, 0, 0, 1), mode='replicate') # todo add ker of size 3
        conv3_out = self.conv3(input)
        full_conv_out = torch.cat([conv1_out, conv3_out], dim=1)
        input = self.max_pool(full_conv_out)
        input = self.conv_relu(input)
        input = self.flatten(input)
        result = self.linear_relu_stack(input)
        return result


# --------------------------------------------------------------------------------------------------------------------
class dna_2d_linear_classifier(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Flatten(start_dim=1),
            nn.Linear(126, 1),
        )
        super().type(torch.FloatTensor)


class dna_1d_linear_classifier(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(21, 1),
        )
        super().type(torch.FloatTensor)



class dna_1d_two_layer_network(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(21, 15),
            nn.ReLU(inplace=True),
            nn.Linear(15, 1),
        )
        super().type(torch.FloatTensor)


class multi_2d_two_layer_network(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Flatten(start_dim=1),
            nn.Linear(126, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 3),
        )
        super().type(torch.FloatTensor)

class dna_2d_two_layer_network(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Flatten(start_dim=1),
            nn.Linear(126, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 1),
        )
        super().type(torch.FloatTensor)



class two_layer_network(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(12, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
        )
        super().type(torch.FloatTensor)


class linear_classifier(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(12, 1)
        )
        super().type(torch.FloatTensor)


class SimpleNetwork(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(12, 8),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(8),
            nn.Linear(8, 8),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(8),
            nn.Linear(8, 1)
        )
        super().type(torch.FloatTensor)


class ConvNetwork1d(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv1d(in_channels=1, out_channels=30, kernel_size=4),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(30),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=30, out_channels=50, kernel_size=6),
            nn.ReLU(inplace=True),
            #nn.BatchNorm1d(50),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(start_dim=1, end_dim=2),
            nn.Linear(100, 50),
            nn.ReLU(inplace=True),
            #nn.BatchNorm1d(128),
            nn.Linear(50, 1)
        )
        super().type(torch.FloatTensor)


class ConvNetwork2d(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(4, 2)), #(4, 2)
            nn.ReLU(inplace=True),
            #nn.BatchNorm2d(20),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(in_channels=16, out_channels=50, kernel_size=(1, 2)), #16, 50, kernel_size=(2, 2)
            nn.ReLU(inplace=True),
            #nn.BatchNorm2d(100),
            #nn.MaxPool1d(kernel_size=2),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(450, 256),
            nn.ReLU(inplace=True),
            #nn.BatchNorm1d(256),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            #nn.BatchNorm1d(128),
            nn.Linear(128, 1)
        )
        super().type(torch.FloatTensor)


# class multi_conv_2d_1fc(nn.Sequential):
#     def __init__(self):
#         super().__init__(
#             nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(6, 3)),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(20),
#             nn.MaxPool2d(kernel_size=(1, 2)),
#             nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(1, 4)),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(50),
#             nn.Flatten(start_dim=1, end_dim=-1),
#             nn.Linear(300, 150),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm1d(150),
#             nn.Dropout(p=0.3),
#             nn.Linear(150, 50),
#             nn.BatchNorm1d(50),
#             nn.ReLU(inplace=True),
#             nn.Linear(50, 3)
#         )

class multi_conv_2d_1fc(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(6, 2)),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(20),
            # nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(400, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 3)
            # nn.ReLU(inplace=True),
            # # nn.Dropout(p=0.3),
            # nn.Linear(50, 3),
        )


class conv_2d_1fc(nn.Sequential):
    # def __init__(self):
    #     super().__init__(
    #         nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(6, 2)),
    #         nn.ReLU(inplace=True),
    #         nn.BatchNorm2d(50),
    #         nn.MaxPool2d(kernel_size=(1, 2)),
    #         nn.Flatten(start_dim=1, end_dim=-1),
    #         nn.Linear(500, 256),
    #         nn.ReLU(inplace=True),
    #         nn.BatchNorm1d(256),
    #         nn.Dropout(p=0.3),
    #         nn.Linear(256, 128),
    #         nn.BatchNorm1d(128),
    #         nn.ReLU(inplace=True),
    #         nn.Linear(128, 1)
    #     )
    def __init__(self):
        super().__init__(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(6, 4)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(20),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(1, 4)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(50),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(300, 150),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(150),
            nn.Dropout(p=0.3),
            nn.Linear(150, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 1)
        )
    # def __init__(self):
    #     super().__init__(
    #         nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(4, 4)),
    #         nn.ReLU(inplace=True),
    #         nn.MaxPool2d(kernel_size=(2, 2)),
    #         nn.Conv2d(in_channels=20, out_channels=40, kernel_size=(1, 4)),
    #         nn.ReLU(inplace=True),
    #         nn.MaxPool2d(kernel_size=(1, 2)),
    #         nn.Flatten(start_dim=1, end_dim=-1),
    #         nn.Linear(120, 70),
    #         nn.ReLU(inplace=True),
    #         nn.BatchNorm1d(70),
    #         # nn.Dropout(p=0.3),
    #         nn.ReLU(inplace=True),
    #         nn.Linear(70, 1)
    #     )
        super().type(torch.FloatTensor)