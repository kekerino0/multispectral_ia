import tensorflow as tf
from keras import backend as K

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

