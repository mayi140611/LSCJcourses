
# coding: utf-8

# In[ ]:

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import SGD
import os


# In[2]:

# 载入预训练的VGG16模型，不包括全连接层
vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))


# In[3]:

# 搭建全连接层
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(2, activation='softmax'))

# 载入训练过的权值
top_model.load_weights('bottleneck_fc_model.h5')

model = Sequential()
model.add(vgg16_model)
model.add(top_model)


# In[4]:

# 训练集数据生成
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# 测试集数据处理
test_datagen = ImageDataGenerator(rescale=1./255)


# In[5]:

batch_size = 32
# 生成训练数据
train_generator = train_datagen.flow_from_directory(
        'image/train',  # 训练数据路径
        target_size=(150, 150),  # 设置图片大小
        batch_size=batch_size # 批次大小
        ) 

# 测试数据
test_generator = test_datagen.flow_from_directory(
        'image/test',  # 训练数据路径
        target_size=(150, 150),  # 设置图片大小
        batch_size=batch_size # 批次大小
        )


# In[6]:

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])


# 统计文件个数
totalFileCount = sum([len(files) for root, dirs, files in os.walk('image/train')])

model.fit_generator(
        train_generator,
        steps_per_epoch=totalFileCount/batch_size,
        epochs=10,
        validation_data=test_generator,
        validation_steps=1000/batch_size,
        )


# In[ ]:



