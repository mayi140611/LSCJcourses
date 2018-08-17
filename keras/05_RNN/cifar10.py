
# coding: utf-8

# In[11]:

import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Convolution2D,MaxPooling2D,Flatten
from keras.optimizers import Adam
import time


# In[12]:

# 载入数据
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
print(x_train.shape)
print(y_train.shape)


# In[13]:

# 数据归一化
x_train = x_train/255.0
x_test = x_test/255.0
# 换one hot格式
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)


# In[14]:

# 定义模型
model = Sequential()
model.add(Convolution2D(input_shape=(32,32,3), filters=64, kernel_size=3, strides=1, padding='same', activation = 'relu'))
model.add(Convolution2D(filters=64, kernel_size=3, strides=1, padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))

model.add(Convolution2D(filters=128, kernel_size=3, strides=1, padding='same', activation = 'relu'))
model.add(Convolution2D(filters=128, kernel_size=3, strides=1, padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))

model.add(Convolution2D(filters=256, kernel_size=3, strides=1, padding='same', activation = 'relu'))
model.add(Convolution2D(filters=256, kernel_size=3, strides=1, padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))

model.add(Flatten())
model.add(Dense(128,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation = 'softmax'))

# 定义优化器
adam = Adam(lr=1e-4)

# 定义优化器，loss function，训练过程中计算准确率
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()


# In[15]:

start = time.time()
model.fit(x_train, y_train, batch_size=256, epochs=50, validation_data=(x_test, y_test), shuffle=True)
print('@ Total Time Spent: %.2f seconds' % (time.time() - start))


# In[ ]:



