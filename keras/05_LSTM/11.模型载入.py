
# coding: utf-8

# In[1]:

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.models import load_model


# In[2]:

# 载入数据
(x_train,y_train),(x_test,y_test) = mnist.load_data()
# (60000,28,28)
print('x_shape:',x_train.shape)
# (60000)
print('y_shape:',y_train.shape)
# (60000,28,28)->(60000,784)
x_train = x_train.reshape(x_train.shape[0],-1)/255.0
x_test = x_test.reshape(x_test.shape[0],-1)/255.0
# 换one hot格式
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)

# 载入模型
model = load_model('model.h5')

# 评估模型
loss,accuracy = model.evaluate(x_test,y_test)

print('\ntest loss',loss)
print('accuracy',accuracy)


# In[3]:

# 训练模型
model.fit(x_train,y_train,batch_size=64,epochs=2)

# 评估模型
loss,accuracy = model.evaluate(x_test,y_test)

print('\ntest loss',loss)
print('accuracy',accuracy)


# In[4]:

# 保存参数，载入参数
model.save_weights('my_model_weights.h5')
model.load_weights('my_model_weights.h5')
# 保存网络结构，载入网络结构
from keras.models import model_from_json
json_string = model.to_json()
model = model_from_json(json_string)


# In[5]:

print(json_string)


# In[ ]:



