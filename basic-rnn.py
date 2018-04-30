import torch
from torch import autograd

batch_size = 5
seq_len = 7
input_size = 2
hidden_size = 32

input = autograd.Variable(torch.rand(seq_len, batch_size, input_size))
print('input size : ', input.size())
state = autograd.Variable(torch.zeros(1, batch_size, hidden_size))
rnn = torch.nn.RNN(
        input_size = input_size,
        hidden_size = hidden_size,
        num_layers = 1,
        nonlinearity = 'tanh'
    )

print('rnn', rnn)
out, state = rnn(input, state)
print('out size ', out.size())
print('state size', state.size())
