# endcoding: utf-8

'''
Created by
@author: Dianyi Hu
@date: 2024/3/15 
@time: 01:13
'''
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import  nn
from d2l import torch as d2l
from torch.utils import data

def synthetic_data(w, b, num_examples):
    X = torch.normal(0,1,(num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0,0.01,y.shape)
    return X, y.reshape((-1,1))

true_w = torch.tensor([2,-3.4])
true_b = 4.2

features, labels = synthetic_data(true_w, true_b, 1000)

# plt.scatter(features[:,(1)].detach().numpy(), labels.detach().numpy(), s=5, color='r')
# plt.show()


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))

    random.shuffle(indices)

    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i+batch_size, num_examples)]
        )
        yield features[batch_indices], labels[batch_indices]


batch_size = 10

# i = 0
# for X, y in data_iter(batch_size, features, labels):
#     print(X, '\n', y)
#     break
    # i += 1
    # print(i)


w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


def linreg(X, w, b):
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    return (y_hat-y.reshape(y_hat.shape))**2/2


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss


# for epoch in range(num_epochs):
#     for X, y in data_iter(batch_size, features, labels):
#         l = loss(net(X, w, b), y)
#         l.sum().backward()
#         sgd([w,b], lr, batch_size)
#
#     with torch.no_grad():
#         train_l = loss(net(features, w, b), labels)
#         print(f'epoch {epoch+1}, loss {float(train_l.mean()):f}')


true_w = torch.tensor([2, -3.4])
true_b = 4.2

features, labels = d2l.synthetic_data(true_w, true_b, 1000)


def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)

print(next(iter(data_iter)))


net = nn.Sequential(nn.Linear(2,1))

net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()

trainer = torch.optim.SGD(net.parameters(), lr=0.03)


num_epochs = 3

for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch+1}, loss {l:f}')



