import torch.nn as nn


class TabularNN(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.3, activation_func=nn.ReLU(), use_batch_norm=True):
        super(TabularNN, self).__init__()

        # Define layers
        self.layer1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256) if use_batch_norm else None
        self.layer2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128) if use_batch_norm else None
        self.layer3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64) if use_batch_norm else None
        self.layer4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32) if use_batch_norm else None
        self.layer5 = nn.Linear(32, 16)
        self.bn5 = nn.BatchNorm1d(16) if use_batch_norm else None
        self.layer6 = nn.Linear(16, 8)
        self.bn6 = nn.BatchNorm1d(8) if use_batch_norm else None
        self.output_layer = nn.Linear(8, 1)

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
        ]

        bns = [
            self.bn1,
            self.bn2,
            self.bn3,
            self.bn4,
            self.bn5,
            self.bn6,
        ]

        for i in range(6):
            x = layers[i](x)
            if bns[i]:
                x = bns[i](x)
            x = self.activation(x)  # Use the provided activation function
            x = self.dropout(x)

        x_out = self.sigmoid(self.output_layer(x))
        return x_out
