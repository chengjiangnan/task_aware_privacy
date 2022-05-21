import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


class LinearRegressor(nn.Module):
    def __init__(self, input_dim, use_bias = True):
        super(LinearRegressor, self).__init__()

        self.linear = nn.Linear(input_dim, 1, bias = use_bias)

        self.name = "LinearRegressor"

    def forward(self, s):
        # s: (batch_size, input_dim)
        
        # res = F.softmax(self.linear(s), dim=1)
        res = self.linear(s)
        return res


class DNNRegressor(nn.Module):
    def __init__(self, input_dim, use_bias = True):
        super(DNNRegressor, self).__init__()

        self.hidden_1 = nn.Linear(input_dim, int(input_dim*1.5), bias = use_bias)
        # self.hidden_2 = nn.Linear(int(input_dim*1.5), int(input_dim*1.5), bias = use_bias)
        self.output = nn.Linear(int(input_dim*1.5), 1, bias = use_bias)

        self.name = "DNNRegressor"

    def forward(self, s):
        # s: (batch_size, input_dim)
        
        res = F.relu(self.hidden_1(s))
        # res = F.relu(self.hidden_2(res))
        res = F.relu(self.output(res))

        return res


if __name__=="__main__":

    regressor = LinearRegressor(5)

    data = torch.randn(8, 5)

    output = regressor(data)

    print(output)
    print(output.sum())
