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

model = keras.models.load_model('CNN2D.keras')

a = cv2.imread(r'C:\Users\STAGIAIRE\Documents\model\CNN2D\data\inference\chien\chien2.jpg')

b = cv2.resize(a,(224,224))
b = tf.expand_dims(b, axis = 0)

#print(a.shape)
#plt.imshow(b)
#plt.show()

result = model.predict(b)
print(result)

