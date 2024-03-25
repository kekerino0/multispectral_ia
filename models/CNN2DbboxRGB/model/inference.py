
import tensorflow as tf
import keras
from keras import layers
from keras import ops
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import shutil
from PIL import Image
class_names = ['PET', 'PLA']
model = keras.models.load_model(r'CNN2DBboxRGB\model\CNN2DBboxRGB.keras')

a = cv2.imread(r'CNN2DBboxRGB\data\MeldrickData\RGB_real\PLA\DSC_2790.JPG')

b = cv2.resize(a,(500,500))
b = tf.expand_dims(b, axis = 0)

#print(a.shape)
#plt.imshow(b)
#plt.show()

result = model.predict(b)
result = np.argmax(result[0])
result = class_names[result]
print('result is : ',result)
