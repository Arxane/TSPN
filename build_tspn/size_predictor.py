import torch
import torch.nn as nn

class SizePredictor(nn.Module):
    def __init__(self,in_features, hidden_size, max_units):
        super(SizePredictor, self).__init__()
        self.h1 = nn.Linear(in_features, hidden_size)
        self.h2 = nn.Linear(hidden_size, max_units)

    def forward(self, x):
        out1 = nn.ReLU(self.h1(x))
        out2 = self.h2(out1)
        return out2