# endcoding: utf-8

'''
Created by
@author: Dianyi Hu
@date: 2024/1/10 
@time: 00:30
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ***************************************************

class lr_model:
    def __init__(self, coeff_, intercept):
        self.k, self.b = np.array(coeff_).reshape(-1,1), intercept

    def predict(self, X:np.ndarray):
        return np.dot(X, self.k) + np.array([self.b]*X.shape[0]).reshape(-1,1)

    def resolve_pressure(self, X, y, bias):
        return (y-bias-self.b-X[1]*self.k[1]-X[2]*self.k[2])/self.k[0]


class cmp_controller:
    def __init__(self, lr_model):
        self.lr_model = lr_model
        self.ewma = {1:0,2:0,3:0,4:0}
        self.selected_vars = ['Press_Z1_rough','head_usage','platen1_usage']
        self.target = 'delta_ero2'
        self.recommend = 'Press_Z1_rough'
        self.setpoint = -23

    def update_ewma(self, data, head):
        df_with_metro = data.query(f'head=={head}').dropna(subset=[self.target])
        df_with_metro = df_with_metro.sort_values(by=['proc_time'])
        df_with_metro['lm_predict'] = self.lr_model.predict(df_with_metro[self.selected_vars].values)
        df_with_metro['lm_error'] = df_with_metro[self.target]-df_with_metro['lm_predict']
        df_with_metro['ewma'] = df_with_metro['lm_error'].ewm(alpha=0.2).mean()
        self.ewma[head] = df_with_metro['ewma'].iloc[-1]

        return df_with_metro

    def recommend_pressure(self, wafer):
        if np.isnan(wafer[self.recommend]):
            return lr_model.resolve_pressure(wafer[self.selected_vars].values, self.setpoint, self.ewma[wafer['head']])[0]
        else:
            return wafer[self.recommend]

# ***************************************************

data = pd.read_csv('test_data.csv', header=0)
data['proc_time'] = pd.to_datetime(data['proc_time'])

lr_model = lr_model([-25,-0.001,-0.001],150)
controller = cmp_controller(lr_model)
for head in [1,2,3,4]:
    controller.update_ewma(data, head)

data['Press_Z1_rough'] = data.apply(controller.recommend_pressure, axis=1)
data['lm_predicted'] = lr_model.predict(data[['Press_Z1_rough','head_usage','platen1_usage']])

# ***************************************************














