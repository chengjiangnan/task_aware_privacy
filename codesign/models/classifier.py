import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, n_classes, use_bias = True):
        super(LinearClassifier, self).__init__()

        self.linear = nn.Linear(input_dim, n_classes, bias = use_bias)

        self.name = "LinearClassifier"

    def forward(self, s):
        # s: (batch_size, input_dim)
        
        # res = F.softmax(self.linear(s), dim=1)
        res = self.linear(s)
        return res


class DNNClassifier(nn.Module):
    def __init__(self, input_dim, n_classes, use_bias = True):
        super(DNNClassifier, self).__init__()

        self.hidden_1 = nn.Linear(input_dim, int(input_dim*1.5), bias = use_bias)
        # self.hidden_2 = nn.Linear(int(input_dim*1.5), int(input_dim*1.5), bias = use_bias)
        self.output = nn.Linear(int(input_dim*1.5), n_classes, bias = use_bias)

        self.name = "DNNClassifier"

    def forward(self, s):
        # s: (batch_size, input_dim)
        
        res = F.relu(self.hidden_1(s))
        # res = F.relu(self.hidden_2(res))
        res = F.relu(self.output(res))
        # res = self.output(res)

        return res

class CNNClassifier(nn.Module):
    def __init__(self, input_dim, n_classes, use_bias = True):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )

        self.out = nn.Linear(32 * 7 * 7, n_classes)

    def forward(self, x):
        x = x.view(x.size(0), 28, 28)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output


if __name__=="__main__":

    clssifier = LinearClassifier(5, 3)

    data = torch.randn(8, 5)

    output = clssifier(data)

    print(output)
    print(output.sum())
