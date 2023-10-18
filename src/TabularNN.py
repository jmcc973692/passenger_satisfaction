import torch.nn as nn


class TabularNN(nn.Module):
    def __init__(self, input_dim):
        super(TabularNN, self).__init__()

        # Define layers
        self.layer1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.layer2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.layer3 = nn.Linear(128, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.layer4 = nn.Linear(128, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.downsample = nn.Linear(128, 64)  # Downsample layer
        self.layer5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.layer6 = nn.Linear(64, 64)
        self.bn6 = nn.BatchNorm1d(64)
        self.layer7 = nn.Linear(64, 32)
        self.bn7 = nn.BatchNorm1d(32)
        self.output_layer = nn.Linear(32, 1)

        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.layer1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)

        x2 = self.layer2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)

        x3 = self.layer3(x2)
        x3 = self.bn3(x3)
        x3 = self.relu(x3)
        x3 = self.dropout(x3)

        x4 = self.layer4(x3) + x2  # residual connection
        x4 = self.bn4(x4)
        x4 = self.relu(x4)
        x4 = self.dropout(x4)

        x5 = self.layer5(x4)
        x5 = self.bn5(x5)
        x5 = self.relu(x5)
        x5 = self.dropout(x5)

        x_downsampled = self.downsample(x4)
        x6 = self.layer6(x5) + x_downsampled  # residual connection
        x6 = self.bn6(x6)
        x6 = self.relu(x6)
        x6 = self.dropout(x6)

        x7 = self.layer7(x6)
        x7 = self.bn7(x7)
        x7 = self.relu(x7)
        x7 = self.dropout(x7)

        x_out = self.sigmoid(self.output_layer(x7))
        return x_out
