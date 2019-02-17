from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

# 指定参数
# rotation_range 旋转
# width_shift_range 左右平移
# height_shift_range 上下平移
# zoom_range 随机放大或缩小
img_generator = ImageDataGenerator(
    rotation_range = 90,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    zoom_range = 0.3
    )
# 导入并显示图片
img_path = 'e:/template/lena.jpg'
img = image.load_img(img_path)
plt.imshow(img)
plt.show()

# 将图片转为数组
x = image.img_to_array(img)
# 扩充一个维度
x = np.expand_dims(x, axis=0)
# 生成图片
gen = img_generator.flow(x, batch_size=1)

# 显示生成的图片
plt.figure()
for i in range(3):
    for j in range(3):
        x_batch = next(gen)
        idx = (3*i) + j
        plt.subplot(3, 3, idx+1)
        plt.imshow(x_batch[0]/256)
x_batch.shape
plt.show()