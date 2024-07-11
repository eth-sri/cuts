import torch.nn as nn
import torch
from utils import straight_through_softmax


class LinReLU(nn.Module):
    """
    A linear layer followed by a ReLU activation layer.
    """

    def __init__(self, in_size, out_size):
        super(LinReLU, self).__init__()

        linear = nn.Linear(in_size, out_size)
        ReLU = nn.ReLU()
        # self.Dropout = nn.Dropout(0.25)
        self.layers = nn.Sequential(linear, ReLU)

    def reset_parameters(self):
        self.layers[0].reset_parameters()
        return self

    def forward(self, x):
        x = self.layers(x)
        return x


class FeatureSoftmax(nn.Module):
    """
    A layer that applies softmax feature-wise.
    """

    def __init__(self, one_hot_index_map):
        super(FeatureSoftmax, self).__init__()
        self.one_hot_index_map = one_hot_index_map

    def forward(self, x, tau=1.0):
        for feature, feature_index in self.one_hot_index_map.items():
            x[:, feature_index] = nn.functional.softmax(x[:, feature_index]/tau, dim=-1)
        return x


class FeatureGumbelSoftmax(nn.Module):
    """
    A layer that applies gumbel-softmax feature-wise.
    """
    def __init__(self, one_hot_index_map):
        super(FeatureGumbelSoftmax, self).__init__()
        self.one_hot_index_map = one_hot_index_map

    def forward(self, x, tau=1.0):
        for feature, feature_index in self.one_hot_index_map.items():
            x[:, feature_index] = nn.functional.gumbel_softmax(x[:, feature_index]/tau, hard=True, dim=-1)
        return x


class FeatureHardSoftmax(nn.Module):
    """
    A layer that applies hard-softmax feature-wise.
    """
    def __init__(self, one_hot_index_map):
        super(FeatureHardSoftmax, self).__init__()
        self.one_hot_index_map = one_hot_index_map

    def forward(self, x, tau=1.0):
        for feature, feature_index in self.one_hot_index_map.items():
            x[:, feature_index] = straight_through_softmax(x[:, feature_index]/tau, dim=-1)
        return x


class FullyConnectedDenoiser(nn.Module):

    def __init__(self, input_size, layout, one_hot_index_map, head='gumbel'):
        super(FullyConnectedDenoiser, self).__init__()
        layers = [nn.Flatten()]  # does not play any role, but makes the code neater
        prev_fc_size = input_size
        for i, fc_size in enumerate(layout):
            if i + 1 < len(layout):
                layers += [LinReLU(prev_fc_size, fc_size)]
            else:
                layers += [nn.Linear(prev_fc_size, fc_size)]
            prev_fc_size = fc_size
        if head == 'gumbel':
            layers += [FeatureGumbelSoftmax(one_hot_index_map)]
        elif head == 'hard_softmax':
            layers += [FeatureHardSoftmax(one_hot_index_map)]
        else:
            layers += [FeatureSoftmax(one_hot_index_map)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

"""
The following code is adapted with some changes from the CTGAN library: Xu et al., Modeling Tabular Data using 
Conditional GAN, 2019, https://github.com/sdv-dev/CTGAN/blob/master/ctgan/synthesizers/ctgan.py

License:
MIT License

Copyright (c) 2019, MIT Data To AI Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

class Residual(nn.Module):
    """Residual layer for the CTGAN."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = nn.Linear(i, o)
        self.bn = nn.BatchNorm1d(o)
        self.relu = nn.ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class ResidualDenoiser(nn.Module):

    def __init__(self, input_size, layout, one_hot_index_map, head='gumbel'):
        super(ResidualDenoiser, self).__init__()
        layers = [nn.Flatten()]  # does not play any role, but makes the code neater
        prev_fc_size = input_size
        for i, fc_size in enumerate(layout):
            if i + 1 < len(layout):
                layers += [Residual(prev_fc_size, fc_size)]
            else:
                layers += [nn.Linear(prev_fc_size, fc_size)]
            prev_fc_size = fc_size + prev_fc_size
        if head == 'gumbel':
            layers += [FeatureGumbelSoftmax(one_hot_index_map)]
        elif head == 'hard_softmax':
            layers += [FeatureHardSoftmax(one_hot_index_map)]
        else:
            layers += [FeatureSoftmax(one_hot_index_map)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x
