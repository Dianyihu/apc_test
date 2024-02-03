# endcoding: utf-8

'''
Created by
@author: Dianyi Hu
@date: 2024/2/3 
@time: 22:28
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)

X,Y = np.meshgrid(x, y)


def ecDistance(X, Y, center=(0,5), beta=np.pi/3, a=1, b=3):
    X2 = np.cos(beta)*(X-center[0]) + np.sin(beta)*(Y-center[1])
    Y2 = -np.sin(beta)*(X-center[0]) + np.cos(beta)*(Y-center[1])

    return np.sqrt(np.power(X2/a,2)+np.power(Y2/b,2))

R = ecDistance(X,Y)


fig,ax = plt.subplots()
CS = ax.contour(X,Y,R)
ax.axis('equal')
ax.clabel(CS, inline=True, fontsize=9)
plt.tight_layout()
plt.show()
