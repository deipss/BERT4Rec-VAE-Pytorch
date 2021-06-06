import torch
import torch.nn as nn
import numpy as np

if __name__ == '__main__':
    loss = nn.BCEWithLogitsLoss()
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    output = loss(input, target)
    output.backward()