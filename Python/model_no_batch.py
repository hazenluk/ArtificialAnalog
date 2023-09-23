import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        state_skip = x #residuals
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = torch.cat((x[0].unsqueeze(0), x[1:] + state_skip[1:]), dim=0)
        return x