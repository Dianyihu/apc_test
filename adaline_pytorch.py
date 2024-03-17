# endcoding: utf-8

'''
Created by
@author: Dianyi Hu
@date: 2024/3/16 
@time: 17:34
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn
from d2l import torch as d2l
from torch.utils import data


# ***************************************************

def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y

# ***************************************************

class FP_Machine:
    def __init__(self, model_path, name_dict):
        self.in_features = name_dict['in_features']
        self.profile_features = name_dict['profile_features']

        self.out_features = [f'Delta_Thk{loc}' for loc in name_dict['profile_features']]
        self.pre_features = [f'Thk{loc}_5150' for loc in name_dict['profile_features']]
        self.post_features = [f'Thk{loc}_6200' for loc in name_dict['profile_features']]

        self.models = self.load_model_by_head(model_path)
        self.records = None

    def load_model_by_head(self, fname):
        return {head: torch.load(fname) for head in [1,2,3,4]}

    def fetch_records(self):
        pass

    def update_model_by_head(self, data, lr=0.01, num_epochs=1):
        if len(data):
            head = data['head'].iloc[0]
            loss = nn.MSELoss()
            trainer = torch.optim.SGD(self.models[head].parameters(), lr)

            X_tensor = torch.from_numpy(data[self.in_features].values)
            y_tensor = torch.from_numpy(data[self.out_features].values)

            for epoch in range(num_epochs):
                for X, y in zip(X_tensor, y_tensor):
                    l = loss(self.models[head](X), y)
                    trainer.zero_grad()
                    l.backward()
                    trainer.step()

    def predict_tar_by_head(self, data):
        if len(data):
            head = data['head'].iloc[0]
            data[self.out_features] = self.model_predict(self.models[head], data[self.in_features].values)

        return data

    def loss_fun(self, ser):
        if len(self.out_features)==3:
            out_features = self.model_predict(self.models[ser['head']], ser[self.in_features].values).reshape(-1)
            post_features = out_features + ser[self.pre_features].values.reshape(-1)
            ero120_6200 = -post_features[1]-(120-self.profile_features[1])*(post_features[1]-post_features[0])/(self.profile_features[1]-self.profile_features[0])
            ero148_6200 = post_features[2]-post_features[1]-(148-self.profile_features[1])*(post_features[1]-post_features[0])/(self.profile_features[1]-self.profile_features[0])

            return self.elp_distance(np.array([ero120_6200, ero148_6200]))[0]

    def elp_distance(self, X, center=(0,0), theta=0, w1=1, w2=1):
        theta = theta*np.pi/180
        scale = np.array([[w1/(w1+w2), 0], [0, w2/(w1+w2)]])
        rotation = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])

        X_prime = rotation@scale@(X-center).T

        return np.sum(np.square(X_prime.T).reshape(-1, 2), axis=1)

    def model_predict(self, model, X):
        with torch.no_grad():
            y = model(torch.from_numpy(X))

        return y.numpy()

    def plot_dist(self, x, y):
        xx = np.linspace(x.min(),x.max(),50)
        yy = np.linspace(y.min(),y.max(),50)
        X,Y = np.meshgrid(xx,yy)

        dist = self.elp_distance(np.vstack([X.flat, Y.flat]).T).reshape(50,50)

        plt.contour(X,Y,dist)


# ***************************************************

class Ada_Regressor:
    def __init__(self, in_features=None, out_features=None):
        self.net = self.load_net() if in_features is None else self.create_net(in_features, out_features)

    def create_net(self, in_features, out_features):
        net = nn.Sequential(nn.Linear(in_features, out_features))
        net[0].weight.data.normal_(0, 0.01)
        net[0].bias.data.fill_(0)

        return net

    def update_net(self, data_iter, lr=0.01, num_epochs=1):
        trainer = torch.optim.SGD(self.net.parameters(), lr)
        loss = nn.MSELoss()

        for epoch in range(num_epochs):
            for X, y in data_iter:
                l = loss(self.net(X), y)
                trainer.zero_grad()
                l.backward()
                trainer.step()
            print(f'epoch {epoch+1}, loss {l:f}')

    def net_predict(self, X):
        with torch.no_grad():
            y = self.net(X)

        return y.numpy()

    def save_net(self):
        torch.save(self.net, './adaline_net.pkl')

    def load_net(self):
        return torch.load('./adaline_net.pkl')

# ***************************************************

def load_array(data_arrays, batch_size=1, is_train=False):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


# true_w = torch.tensor([[2,-3.4,3],[1.5,-1,3]])
# true_b = 4.2
# features, labels = synthetic_data(true_w, true_b, 1000)
#
# ada_regressor = Ada_Regressor()
# ada_regressor.update_net(zip(features[:10,:], labels[:10,:]), 0.1)
#
#
# for param in ada_regressor.net.parameters():
#   print(param.data)
#
#
#
# testDistance = np.array([1,2])
#
# fpMachine = FP_Machine()
#
# test = fpMachine.elp_distance(testDistance)
# print(test)
#
#
# x = np.linspace(-10,10,100)
# y = np.linspace(-10,10,100)
#
# X,Y = np.meshgrid(x,y)
#
# test2 = np.vstack([X.flat, Y.flat]).T
#
# test = fpMachine.elp_distance(test2)
# print(test.reshape(100,100))
#
# plt.contour(X,Y,test.reshape(100,100))
# plt.show()
#
