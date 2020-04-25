import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.dropout1(F.relu(self.fc1(x)))
        out = torch.sigmoid(self.fc2(out))
        return out

def do_train():

