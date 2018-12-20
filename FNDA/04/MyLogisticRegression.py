import pandas as pd
import numpy as np
import statsmodels.api as sm

def BSD(X,y, w_initial, h=0.01, maxSteps = 100000):   #批量梯度下降法求逻辑回归参数
    '''
    :param X: 输入特征矩阵，第一列为向量1
    :param y: 标签，用1、0表示
    :param w_initial: 初始化权重
    :param h: 固定步长
    :param maxSteps: 最大迭代步数
    :return: 权重的估计值
    '''
    w0 = w_initial
    for i in range(maxSteps):
        s = np.exp(X*w0)/(1+np.exp(X*w0))
        descent = X.T*y-X.T*s  #梯度
        w1 = w0 + descent*h    #梯度上升
        w0 = w1
        if max(abs(descent*h)) < 0.00001:    #若当前的权重的更新很小时，认为迭代已经收敛，可以提前退出迭代
            break
    return w1

#读取数据，并转换成矩阵
data = pd.read_csv('/Users/Code/lianshu/Lecture 4/data.csv',header = 0)
data2 = np.matrix(data)
X,y = data2[:,:2],data2[:,-1]
ones = np.mat(np.ones((X.shape[0],1)))
X = np.hstack((ones, X))

#设置初始值，带入批量梯度下降法中
w_initial = np.matrix(np.array([1,1,1])).T
w2 = BSD(X,y, w_initial, 0.01, maxSteps = 100000)
print(w2)
'''
[[14.73245533]
 [ 1.25224897]
 [-2.00004285]]
'''

#同时，用python内置逻辑回归算法求解参数，作为对比
logit = sm.Logit(y, X)
result = logit.fit()
result.summary2()
'''
---------------------------------------------------------------
            Coef.   Std.Err.     z     P>|z|    [0.025   0.975]
---------------------------------------------------------------
const      14.7521    4.3948   3.3567  0.0008   6.1385  23.3658
x1          1.2536    0.5770   2.1726  0.0298   0.1227   2.3845
x2         -2.0027    0.5924  -3.3805  0.0007  -3.1638  -0.8416
==============================================================
'''