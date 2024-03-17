# endcoding: utf-8

'''
Created by
@author: Dianyi Hu
@date: 2024/3/16 
@time: 17:34
'''

import numpy as np
import pandas as pd
import torch
from pyswarm import pso
from torch import nn
from torch.utils import data


# ***************************************************

def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y


def load_array(data_arrays, batch_size=1, is_train=False):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


# ***************************************************

class Metro_Worker:
    def __init__(self):
        pass

    def find_eros_by_wafer(self, metro_operation=5150):
        pass

    def find_rec_from_time(self, t_point=None):
        if t_point is None:
            pass
        else:
            pass


# ***************************************************

class FP_Machine:
    def __init__(self, model_path, feature_dict, bounds_dict):
        self.in_features = feature_dict['in_features']
        self.out_features = feature_dict['out_features']
        self.pre_features = feature_dict['pre_features']
        self.post_features = feature_dict['post_features']
        self.upper_bounds = bounds_dict['upper_bounds']
        self.lower_bounds = bounds_dict['lower_bounds']

        self.models = self.load_models(model_path)
        self.rec_time = None

    def load_models(self, file_name):
        return {head: torch.load(file_name) for head in [1, 2, 3, 4]}

    def update_models(self, train_data):
        if len(train_data):
            self.rec_time = train_data['proc_time'].max()
            train_data.groupby('head').apply(self.update_model_by_head)
        else:
            print('New records cannot be found!')

    def update_model_by_head(self, batch, lr=0.001, num_epochs=3):
        if len(batch):
            head = batch['head'].iloc[0]
            print(f'Head{head} model updating!')

            loss = nn.MSELoss()
            trainer = torch.optim.SGD(self.models[head].parameters(), lr)

            X_tensor = torch.from_numpy(batch[self.in_features].values).float()
            y_tensor = torch.from_numpy(batch[self.out_features].values).float()
            for epoch in range(num_epochs):
                l = loss(self.models[head](X_tensor), y_tensor)
                trainer.zero_grad()
                l.backward()
                trainer.step()
                print(f'epoch {epoch + 1}, loss: {l:.2f}')
                print(self.models[head].state_dict()['0.weight'].numpy().reshape(-1))

    def loss_fun(self, z_list, head, pre_values):
        return self.elp_distance(np.array(pre_values) + self.model_predict(head, np.array(z_list)))

    def recom_pressure(self, head, pre_values):
        p_opt, _ = pso(self.loss_fun, lb=self.lower_bounds, ub=self.upper_bounds,
                       args=(head, pre_values), minfunc=0.1, maxiter=10)

        return p_opt

    def elp_distance(self, X, center=(0, 0), theta=0, w1=1, w2=1):
        theta = np.deg2rad(theta)
        scale = np.array([[w1 / (w1 + w2), 0],
                          [0, w2 / (w1 + w2)]])
        rotation = np.array([[np.cos(theta), np.sin(theta)],
                             [-np.sin(theta), np.cos(theta)]])

        return np.sum(np.square(scale @ rotation @ (X - center).T), axis=0)

    def model_predict(self, head, X):
        with torch.no_grad():
            y = self.models[head](torch.from_numpy(X).float())

        return y.numpy()


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
            print(f'epoch {epoch + 1}, loss {l:f}')
            print(self.net.state_dict()['0.weight'].numpy().reshape(-1))

    def net_predict(self, X):
        with torch.no_grad():
            y = self.net(X)

        return y.numpy()

    def save_net(self):
        torch.save(self.net, './adaline_net.pkl')

    def load_net(self):
        return torch.load('./adaline_net.pkl')


# ***************************************************

true_w = torch.tensor([[2, -3.4], [1.5, -1]])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# ada_regressor = Ada_Regressor(in_features=2, out_features=2)
# ada_regressor.update_net(zip(features[:10,:], labels[:10,:]), 0.1)
# ada_regressor.save_net()


testDf = pd.DataFrame(columns=['proc_time', 'head', 'z1Pressure', 'z2Pressure', 'delta_ERO120', 'delta_ERO148',
                               'ERO120_5150', 'ERO148_5150', 'ERO120_6200', 'ERO148_6200'])

testDf['head'] = [1, 2, 3, 4] * 250
testDf['proc_time'] = pd.date_range(end='1/1/2024', periods=len(features))
testDf[['z1Pressure', 'z2Pressure']] = features.numpy()
testDf[['delta_ERO120', 'delta_ERO148']] = labels.numpy()
testDf[['ERO120_5150', 'ERO148_5150']] = np.random.normal(size=(len(features), 2))
testDf[['ERO120_6200', 'ERO148_6200']] = np.nan

feature_dict = {
    'in_features': ['z1Pressure', 'z2Pressure'],
    'out_features': ['delta_ERO120', 'delta_ERO148'],
    'pre_features': ['ERO120_5150', 'ERO148_5150'],
    'post_features': ['ERO120_6200', 'ERO148_6200'],
}

bounds_dict = {
    'upper_bounds': [7.5, 3.72],
    'lower_bounds': [6.2, 3.62],
}

fpMachine = FP_Machine('adaline_net.pkl', feature_dict, bounds_dict)
fpMachine.update_models(testDf)

# for _, wafer in testDf.iterrows():
#     p = fpMachine.recom_pressure(wafer['head'], wafer[['ERO120_5150', 'ERO148_5150']])
#     print(f'Recommeding z1z2 pressure: {p}')
