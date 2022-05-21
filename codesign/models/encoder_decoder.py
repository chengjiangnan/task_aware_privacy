import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


class SimpleEncoder(nn.Module):
    def __init__(self, input_dim, z_dim, use_bias = True):
        super(SimpleEncoder, self).__init__()

        self.input_dim = input_dim
        self.z_dim = z_dim

        self.encoder = nn.Linear(input_dim, z_dim, bias = use_bias)

        self.name = "SimpleEncoder"

    def forward(self, s):
        # s: (batch_size, input_dim)
        # z: (batch_size, z_dim)

        z = self.encoder(s)
        return z


class SimpleDecoder(nn.Module):
    def __init__(self, z_dim, output_dim, use_bias = True):
        super(SimpleDecoder, self).__init__()

        self.z_dim = z_dim
        self.output_dim = output_dim

        self.decoder = nn.Linear(z_dim, output_dim, bias = use_bias)

        self.name = "SimpleDecoder"

    def forward(self, z):
        # z: (batch_size, z_dim)
        # s_hat: (batch_size, output_dim)

        s_hat = self.decoder(z)
        return s_hat


class DNNEncoder(nn.Module):
    def __init__(self, input_dim, z_dim, use_bias = True):
        super(DNNEncoder, self).__init__()

        self.input_dim = input_dim
        self.z_dim = z_dim

        self.hidden_1 = nn.Linear(input_dim, input_dim, bias = use_bias)
        self.output = nn.Linear(input_dim, z_dim, bias = use_bias)

        self.name = "DNNEncoder"

    def forward(self, s):
        # s: (batch_size, input_dim)
        # z: (batch_size, z_dim)

        # z = F.relu(self.hidden_1(s))
        z = torch.sigmoid(self.hidden_1(s))
        z = self.output(z)

        return z


class DNNDecoder(nn.Module):
    def __init__(self, z_dim, output_dim, use_bias = True):
        super(DNNDecoder, self).__init__()

        self.z_dim = z_dim
        self.output_dim = output_dim

        self.hidden_1 = nn.Linear(z_dim, output_dim, bias = use_bias)
        self.output = nn.Linear(output_dim, output_dim, bias = use_bias)

        self.name = "DNNDecoder"

    def forward(self, z):
        # z: (batch_size, z_dim)
        # s_hat: (batch_size, output_dim)

        # s_hat = F.relu(self.hidden_1(z))
        s_hat = torch.sigmoid(self.hidden_1(z))
        s_hat = self.output(s_hat)
        return s_hat
