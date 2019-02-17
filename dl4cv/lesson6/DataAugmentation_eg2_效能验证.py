from keras.datasets import cifar10
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import generic_utils

(x_train, y_train),(x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

def preprocess_data(x):
    x /= 255
    x -= 0.5
    x *= 2
    return x

# 预处理
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

x_train = preprocess_data(x_train)
x_test = preprocess_data(x_test)

# one-hot encoding
n_classes = 10
y_train = np_utils.to_categorical(y_train, n_classes)
y_test = np_utils.to_categorical(y_test, n_classes)

# 取 20% 的训练数据
x_train_part = x_train[:10000]
y_train_part = y_train[:10000]

print(x_train_part.shape, y_train_part.shape)

# 建立一个简单的卷积神经网络,序贯结构
def build_model():
    model = Sequential()

    model.add(Conv2D(64, (3,3), input_shape=(32,32,3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization(scale=False, center=False))

    model.add(Conv2D(32, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization(scale=False, center=False))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    return model

# 训练参数
batch_size = 128
#epochs = 20
epochs = 2
#cifar-10 20%数据,训练结果,绘图
model = build_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train_part, y_train_part, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.1)

loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print('Loss: ', loss)
print('Accuracy: ', acc)

#cifar-10 20%数据 + Data Augmentation.训练结果
# 设置生成参数
img_generator = ImageDataGenerator(
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    zoom_range = 0.2
    )
model_2 = build_model()
model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Data Augmentation
for e in range(epochs):
    print('Epoch', e)
    print('Training...')
    progbar = generic_utils.Progbar(x_train_part.shape[0])
    batches = 0

    for x_batch, y_batch in img_generator.flow(x_train_part, y_train_part, batch_size=batch_size, shuffle=True):
        loss,train_acc = model_2.train_on_batch(x_batch, y_batch)
        batches += x_batch.shape[0]
        if batches > x_train_part.shape[0]:
            break
        progbar.add(x_batch.shape[0], values=[('train loss', loss),('train acc', train_acc)])
loss, acc = model_2.evaluate(x_test, y_test, batch_size=32)
print('Loss: ', loss)
print('Accuracy: ', acc)

###最后，我尝试采用文档中提示方法
img_generator.fit(x_train_part)

# fits the model_2 on batches with real-time data augmentation:
model_2.fit_generator(img_generator.flow(x_train_part, y_train_part, batch_size=batch_size),
                    steps_per_epoch=len(x_train_part), epochs=epochs)