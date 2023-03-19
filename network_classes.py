import torch
import torch.nn as nn


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
            #nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=30, out_channels=50, kernel_size=6),
            nn.ReLU(inplace=True),
            #nn.BatchNorm1d(50),
            #nn.MaxPool1d(kernel_size=2),
            nn.Flatten(start_dim=1, end_dim=2),
            nn.Linear(650, 256),
            nn.ReLU(inplace=True),
            #nn.BatchNorm1d(256),
            #nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            #nn.BatchNorm1d(128),
            nn.Linear(128, 1)
        )
        super().type(torch.FloatTensor)


class ConvNetwork2d(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(4, 2)), #(4, 2)
            nn.ReLU(inplace=True),
            #nn.BatchNorm2d(20),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(in_channels=16, out_channels=50, kernel_size=(2, 2)), #16, 50, kernel_size=(2, 2)
            nn.ReLU(inplace=True),
            #nn.BatchNorm2d(100),
            #nn.MaxPool1d(kernel_size=2),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(450, 256),
            nn.ReLU(inplace=True),
            #nn.BatchNorm1d(256),
            #nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            #nn.BatchNorm1d(128),
            nn.Linear(128, 1)
        )
        super().type(torch.FloatTensor)