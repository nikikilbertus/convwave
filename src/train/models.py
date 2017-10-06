# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as func


# -----------------------------------------------------------------------------
# STANDARD FCN FOR SPECTROGRAMS
# -----------------------------------------------------------------------------

class SpectrogramFCN(nn.Module):

    # -------------------------------------------------------------------------
    # Initialize the net and define functions for the layers
    # -------------------------------------------------------------------------

    def __init__(self):

        # Inherit from the PyTorch neural net module
        super(SpectrogramFCN, self).__init__()

        # Convolutional layers: (in_channels, out_channels, kernel_size)
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=64,
                               kernel_size=(3, 7), padding=(1, 3), stride=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=(3, 7), padding=(1, 6),
                               stride=1, dilation=(1, 2))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=(3, 7), padding=(1, 6),
                               stride=1, dilation=(1, 2))
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=(3, 7), padding=(1, 6),
                               stride=1, dilation=(1, 2))
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=(3, 7), padding=(1, 6),
                               stride=1, dilation=(1, 2))
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=(3, 7), padding=(1, 6),
                               stride=1, dilation=(1, 2))
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=(3, 7), padding=(1, 9),
                               stride=1, dilation=(1, 3))
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=1,
                               kernel_size=(1, 1), padding=(0, 0), stride=1)

        # Batch norm layers
        self.batchnorm1 = nn.BatchNorm2d(num_features=128)
        self.batchnorm2 = nn.BatchNorm2d(num_features=128)
        self.batchnorm3 = nn.BatchNorm2d(num_features=128)
        self.batchnorm4 = nn.BatchNorm2d(num_features=128)
        self.batchnorm5 = nn.BatchNorm2d(num_features=128)
        self.batchnorm6 = nn.BatchNorm2d(num_features=128)

        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

    # -------------------------------------------------------------------------
    # Define a forward pass through the network (apply the layers)
    # -------------------------------------------------------------------------

    def forward(self, x):

        # Layer 1
        # ---------------------------------------------------------------------
        x = self.conv1(x)
        x = func.elu(x)

        # Layers 2 to 3
        # ---------------------------------------------------------------------
        convolutions = [self.conv2, self.conv3, self.conv4, self.conv5,
                        self.conv6, self.conv7]
        batchnorms = [self.batchnorm1, self.batchnorm2, self.batchnorm3,
                      self.batchnorm4, self.batchnorm5, self.batchnorm6]

        for conv, batchnorm in zip(convolutions, batchnorms):
            x = conv(x)
            x = batchnorm(x)
            x = func.elu(x)
            x = self.pool(x)

        # Layer 8
        # ---------------------------------------------------------------------
        x = self.conv8(x)
        x = func.sigmoid(x)

        return x

    # -------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# STANDARD FCN FOR TIME SERIES
# -----------------------------------------------------------------------------

class TimeSeriesFCN(nn.Module):

    # -------------------------------------------------------------------------
    # Initialize the net and define functions for the layers
    # -------------------------------------------------------------------------

    def __init__(self):

        # Inherit from the PyTorch neural net module
        super(TimeSeriesFCN, self).__init__()

        # Convolutional layers: (in_channels, out_channels, kernel_size)
        self.conv0 = nn.Conv1d(in_channels=2, out_channels=128,
                               kernel_size=3, padding=1, dilation=1)
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=128,
                               kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128,
                               kernel_size=2, padding=1, dilation=2)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128,
                               kernel_size=2, padding=2, dilation=4)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128,
                               kernel_size=2, padding=4, dilation=8)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=128,
                               kernel_size=2, padding=8, dilation=16)
        self.conv6 = nn.Conv1d(in_channels=128, out_channels=128,
                               kernel_size=2, padding=16, dilation=32)
        self.conv7 = nn.Conv1d(in_channels=128, out_channels=128,
                               kernel_size=2, padding=32, dilation=64)
        self.conv8 = nn.Conv1d(in_channels=128, out_channels=128,
                               kernel_size=2, padding=64, dilation=128)
        self.conv9 = nn.Conv1d(in_channels=128, out_channels=128,
                               kernel_size=2, padding=128, dilation=256)
        self.conv10 = nn.Conv1d(in_channels=128, out_channels=128,
                                kernel_size=2, padding=256, dilation=512)
        self.conv11 = nn.Conv1d(in_channels=128, out_channels=128,
                                kernel_size=2, padding=512, dilation=1024)
        self.conv12 = nn.Conv1d(in_channels=128, out_channels=128,
                                kernel_size=2, padding=1024, dilation=2048)
        self.conv13 = nn.Conv1d(in_channels=128, out_channels=1,
                                kernel_size=1, padding=0, dilation=1)
        # This should give a receptive field of size 4096?

        # Batch norm layers
        self.batchnorm1 = nn.BatchNorm2d(num_features=128)
        self.batchnorm2 = nn.BatchNorm2d(num_features=128)
        self.batchnorm3 = nn.BatchNorm2d(num_features=128)
        self.batchnorm4 = nn.BatchNorm2d(num_features=128)
        self.batchnorm5 = nn.BatchNorm2d(num_features=128)
        self.batchnorm6 = nn.BatchNorm2d(num_features=128)
        self.batchnorm7 = nn.BatchNorm2d(num_features=128)
        self.batchnorm8 = nn.BatchNorm2d(num_features=128)
        self.batchnorm9 = nn.BatchNorm2d(num_features=128)
        self.batchnorm10 = nn.BatchNorm2d(num_features=128)
        self.batchnorm11 = nn.BatchNorm2d(num_features=128)
        self.batchnorm12 = nn.BatchNorm2d(num_features=128)

    # -------------------------------------------------------------------------
    # Define a forward pass through the network (apply the layers)
    # -------------------------------------------------------------------------

    def forward(self, x):

        # Layer 0
        # ---------------------------------------------------------------------
        x = self.conv0(x)
        x = func.elu(x)

        # Layers 1 to 12
        # ---------------------------------------------------------------------
        convolutions = [self.conv1, self.conv2, self.conv3, self.conv4,
                        self.conv5, self.conv6, self.conv7, self.conv8,
                        self.conv9, self.conv10, self.conv11, self.conv12]
        batchnorms = [self.batchnorm1, self.batchnorm2, self.batchnorm3,
                      self.batchnorm4, self.batchnorm5, self.batchnorm6,
                      self.batchnorm7, self.batchnorm8, self.batchnorm9,
                      self.batchnorm10, self.batchnorm11, self.batchnorm12]

        for conv, batchnorm in zip(convolutions, batchnorms):
            x = conv(x)
            x = batchnorm(x)
            x = func.elu(x)

        # Layer 13
        # ---------------------------------------------------------------------
        x = self.conv13(x)
        x = func.sigmoid(x)

        return x

    # -------------------------------------------------------------------------
