import torch


class LinReLU(torch.nn.Module):
    """
    A linear layer followed by a ReLU activation layer.
    """

    def __init__(self, in_size, out_size):
        super(LinReLU, self).__init__()

        linear = torch.nn.Linear(in_size, out_size)
        ReLU = torch.nn.ReLU()
        # self.Dropout = nn.Dropout(0.25)
        self.layers = torch.nn.Sequential(linear, ReLU)

    def reset_parameters(self):
        self.layers[0].reset_parameters()
        return self

    def forward(self, x):
        x = self.layers(x)
        return x


class FullyConnected(torch.nn.Module):
    """
    A simple fully connected neural network with ReLU activations.
    """

    def __init__(self, input_size, layout):

        super(FullyConnected, self).__init__()
        layers = [torch.nn.Flatten()]  # does not play any role, but makes the code neater
        prev_fc_size = input_size
        for i, fc_size in enumerate(layout):
            if i + 1 < len(layout):
                layers += [LinReLU(prev_fc_size, fc_size)]
            else:
                layers += [torch.nn.Linear(prev_fc_size, fc_size)]
            prev_fc_size = fc_size
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x
        