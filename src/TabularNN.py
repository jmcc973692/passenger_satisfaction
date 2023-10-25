import torch.nn as nn


class TabularNN(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.3, activation_func=nn.ReLU()):
        super(TabularNN, self).__init__()

        # Define layers
        self.layer1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.layer2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.layer3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.layer4 = nn.Linear(256, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.layer5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.layer6 = nn.Linear(128, 128)
        self.bn6 = nn.BatchNorm1d(128)
        self.layer7 = nn.Linear(128, 64)
        self.bn7 = nn.BatchNorm1d(64)
        self.layer8 = nn.Linear(64, 64)
        self.bn8 = nn.BatchNorm1d(64)
        self.layer9 = nn.Linear(64, 32)
        self.bn9 = nn.BatchNorm1d(32)
        self.layer10 = nn.Linear(32, 16)
        self.bn10 = nn.BatchNorm1d(16)
        self.output_layer = nn.Linear(16, 1)

        # Activation and dropout
        self.activation = activation_func  # Set activation function
        self.dropout = nn.Dropout(dropout_rate)  # Set dropout rate
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        layers = [
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5,
            self.layer6,
            self.layer7,
            self.layer8,
            self.layer9,
            self.layer10,
        ]

        bns = [
            self.bn1,
            self.bn2,
            self.bn3,
            self.bn4,
            self.bn5,
            self.bn6,
            self.bn7,
            self.bn8,
            self.bn9,
            self.bn10,
        ]

        for i in range(10):
            x = layers[i](x)
            x = bns[i](x)
            x = self.activation(x)  # Use the provided activation function
            x = self.dropout(x)

        x_out = self.sigmoid(self.output_layer(x))
        return x_out
