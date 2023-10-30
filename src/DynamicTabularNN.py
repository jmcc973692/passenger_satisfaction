import torch.nn as nn


class DynamicTabularNN(nn.Module):
    def __init__(self, input_dim, layers, dropout_rate=0.3, activation_func=nn.ReLU(), use_batch_norm=True):
        super(DynamicTabularNN, self).__init__()

        # List to hold layers and batch norms
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList() if use_batch_norm else None
        self.dropout = nn.Dropout(dropout_rate)

        # Create layers dynamically based on the "layers" list
        for i in range(len(layers)):
            # Input size for the current layer
            in_size = input_dim if i == 0 else layers[i - 1]
            # Output size for the current layer
            out_size = layers[i]

            # Append the linear layer
            self.layers.append(nn.Linear(in_size, out_size))

            # Append batch normalization layer if use_batch_norm is True
            if use_batch_norm:
                self.bns.append(nn.BatchNorm1d(out_size))

        # Output layer (Binary classification)
        self.output_layer = nn.Linear(layers[-1], 1)

        # Activation and dropout
        self.activation = activation_func
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.bns:
                x = self.bns[i](x)
            x = self.activation(x)
            x = self.dropout(x)  # Apply Dropout

        x_out = self.sigmoid(self.output_layer(x))
        return x_out
