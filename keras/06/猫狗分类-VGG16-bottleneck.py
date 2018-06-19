
# coding: utf-8

# In[ ]:

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import Adam


# In[59]:

# 载入预训练的VGG16模型，不包括全连接层
model = VGG16(weights='imagenet', include_top=False)


# In[60]:

model.summary()


# In[47]:

datagen = ImageDataGenerator(
        rotation_range = 40,      # 随机旋转角度
        width_shift_range = 0.2,  # 随机水平平移
        height_shift_range = 0.2, # 随机竖直平移
        rescale = 1./255,         # 数值归一化
        shear_range = 0.2,        # 随机裁剪
        zoom_range  =0.2,         # 随机放大
        horizontal_flip = True,   # 水平翻转
        fill_mode='nearest')      # 填充方式


# In[48]:

batch_size = 32
# 
train_steps = int((2000 +  batch_size - 1)/batch_size)*10
test_steps = int((1000 +  batch_size - 1)/batch_size)*10
generator = datagen.flow_from_directory(
        'image/train',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode=None,  # 不生成标签
        shuffle=False)    # 不随机打乱

# 得到训练集数据
bottleneck_features_train = model.predict_generator(generator, train_steps)
print(bottleneck_features_train.shape)
# 保存训练集bottleneck结果
np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)

generator = datagen.flow_from_directory(
        'image/test',
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode=None, # 不生成标签
        shuffle=False)  # 不随机打乱
# 得到预测集数据
bottleneck_features_test = model.predict_generator(generator, test_steps)
print(bottleneck_features_test.shape)
# 保存测试集bottleneck结果
np.save(open('bottleneck_features_test.npy', 'wb'), bottleneck_features_test)


# In[50]:

train_data = np.load(open('bottleneck_features_train.npy','rb'))
# the features were saved in order, so recreating the labels is easy
labels = np.array([0] * 1000 + [1] * 1000)
train_labels = np.array([])
for _ in range(10):
    train_labels=np.concatenate((train_labels,labels))

test_data = np.load(open('bottleneck_features_test.npy','rb'))
labels = np.array([0] * 500 + [1] * 500)
test_labels = np.array([])
for _ in range(10):
    test_labels=np.concatenate((test_labels,labels))

train_labels = np_utils.to_categorical(train_labels,num_classes=2)
test_labels = np_utils.to_categorical(test_labels,num_classes=2)


# In[56]:

model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# 定义优化器
adam = Adam(lr=1e-4)

# 定义优化器，loss function，训练过程中计算准确率
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(train_data, train_labels,
          epochs=20, batch_size=batch_size,
          validation_data=(test_data, test_labels))

model.save_weights('bottleneck_fc_model.h5')


# In[62]:

len(model.layers)


# In[ ]:



