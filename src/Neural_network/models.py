import torch
import torch.nn as nn


class MLP_H2(nn.Module):
    def __init__(self, input_size, hidden_layer1=256, hidden_layer2=128, dropout_rate=0.5):
        super(MLP_H2, self).__init__()

        # Layers
        self.input_layer = nn.Linear(input_size, hidden_layer1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.hidden_layer = nn.Linear(hidden_layer1, hidden_layer2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(hidden_layer2, 1)  # Dernière couche de sortie

    def forward(self, x_input):
        x_input = self.input_layer(x_input)
        x_input = self.relu1(x_input)
        x_input = self.dropout1(x_input)

        x_input = self.hidden_layer(x_input)
        x_input = self.relu2(x_input)
        x_input = self.dropout2(x_input)

        y_output = self.output_layer(x_input)
        return y_output


class MLP_H1(nn.Module):
    def __init__(self, input_size, hidden_layer=256, dropout_rate=0.5):
        super(MLP_H1, self).__init__()

        # Layers
        self.input_layer = nn.Linear(input_size, hidden_layer)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(hidden_layer, 1)

    def forward(self, x_input):
        x_input = self.input_layer(x_input)
        x_input = self.relu(x_input)
        x_input = self.dropout(x_input)
        y_output = self.output_layer(x_input)
        return y_output


class Perceptron(nn.Module):

    def __init__(self, input_size: int):
        super().__init__()

        # Layers
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x_input) -> torch.Tensor:
        y_output = self.linear(x_input)
        return y_output
