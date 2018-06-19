
# coding: utf-8

# In[1]:

from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg19 import VGG19


# In[ ]:

ResNet50_model = ResNet50(weights='imagenet', include_top=False, input_shape=(150,150,3))
InceptionV3_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(150,150,3))
VGG19_model = VGG19(weights='imagenet', include_top=False, input_shape=(150,150,3))

