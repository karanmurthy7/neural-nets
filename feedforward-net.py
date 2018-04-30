import torch
from torch import autograd, nn

batch_size = 5
input_size = 3

torch.manual_seed(123)
input = torch.rand(batch_size, input_size)
print('input', input)