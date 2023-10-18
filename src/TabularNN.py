import torch.nn as nn
import torch.nn.functional as F


class TabularNN(nn.Module):
    def __init__(self, input_dim):
        super(TabularNN, self).__init__()

        # Define layers
        self.layer1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.layer2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.res1 = nn.Linear(512, 256)  # Residual connection for layer2

        self.layer3 = nn.Linear(256, 256)
        self.bn3 = nn.BatchNorm1d(256)

        self.layer4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.res2 = nn.Linear(256, 128)  # Residual connection for layer4

        self.layer5 = nn.Linear(128, 128)
        self.bn5 = nn.BatchNorm1d(128)

        self.layer6 = nn.Linear(128, 64)
        self.bn6 = nn.BatchNorm1d(64)
        self.res3 = nn.Linear(128, 64)  # Residual connection for layer6

        self.output_layer = nn.Linear(64, 1)

        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Store the output for residual connection
        res1_output = x
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu(x + self.res1(res1_output))
        x = self.dropout(x)

        x = self.layer3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Store the output for residual connection
        res2_output = x
        x = self.layer4(x)
        x = self.bn4(x)
        x = self.relu(x + self.res2(res2_output))
        x = self.dropout(x)

        x = self.layer5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Store the output for residual connection
        res3_output = x
        x = self.layer6(x)
        x = self.bn6(x)
        x = self.relu(x + self.res3(res3_output))
        x = self.dropout(x)

        x = self.sigmoid(self.output_layer(x))
        return x
