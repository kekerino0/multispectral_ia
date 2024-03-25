import tensorflow as tf
import keras
from keras import layers
INPUT_WIDTH = 500
INPUT_HEIGHT = 500
N_CHANNELS = 9
N_CLASSES = 2

input_shape = [INPUT_WIDTH, INPUT_HEIGHT, N_CHANNELS]

input_shape =layers.Input((500, 500, 9))

conv1 = layers.Conv2D(3, (1,1))(input_shape)


# 1. Import the empty architecture
model_arch = tf.keras.applications.VGG16(
    
    # Removing the fully-connected layer at the top of the network.
    #  Unless you have the same number of labels as the original architecture, 
    #  you should remove it.
    include_top=False,  
    
    # Using no pretrained weights (random initialization)
    weights='imagenet')(conv1)

#x = model_arch.output
x = keras.layers.Flatten()(model_arch)
x = keras.layers.Dense(N_CLASSES) (x)
model = keras.Model(inputs=input_shape, outputs=x)