import torch
from torch import autograd, nn, optim
import torch.nn.functional as F
from torch.backends.cudnn.rnn import _hidden_size


batch_size = 5
input_size = 3
hidden_size = 8
num_classes = 4
learning_rate = 0.001

torch.manual_seed(123)
input = autograd.Variable(torch.rand(batch_size, input_size))
target = autograd.Variable((torch.rand(batch_size) * num_classes).long())


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.h1 = nn.Linear(input_size, hidden_size)
        self.h2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = self.h1(x)
        x = F.tanh(x)
        x = self.h2(x)
        x = F.softmax(x)
        return x
    
    
    
model = Net(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
opt = optim.Adam(params= model.parameters(), lr = learning_rate)

for epoch in range(1000):
    output = model(input)
    _, pred = output.max(1)
    
    print('target : ', target.view(1, -1))
    print('pred : ', pred.view(1, -1))
    
    loss = F.nll_loss(output, target=target)
    print('loss : ', loss)
    
    model.zero_grad()
    loss.backward()
    opt.step()