import torch
import torch.nn as nn
from classes_functions.gelu import GeLu


class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GeLu()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GeLu()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GeLu()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GeLu()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GeLu())
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)

            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x


def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.]])

    loss = nn.MSELoss()
    loss = loss(output, target)
    loss.backward()

    for name, param in model.named_parameters():
        if "weight" in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")
