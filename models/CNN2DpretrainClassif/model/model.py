import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import tensorflow_datasets as tfds
from keras.utils import Sequence, img_to_array
import keras
import tensorflow as tf
from keras import layers, models
import dataGenerator

batch_size = 10
image_size = (500,500)
input_shape = (500,500,9)
num_classes = 10
epochs = 20

train_dir = r'CNN2DpretrainClassif\data\multiSpectralImagenette2\train'
validation_dir = r'CNN2DpretrainClassif\data\multiSpectralImagenette2\val'

train_data = dataGenerator.DataGenerator(train_dir,num_classes=num_classes, batch_size=batch_size)
val_data = dataGenerator.DataGenerator(validation_dir, num_classes=num_classes, batch_size=batch_size)

#### model
def pretrainModel(input_shape = input_shape, num_classes = num_classes):
    
    backbone = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', name = '1', input_shape=input_shape),
            layers.Dropout(0.5),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu', name = '2'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu', name = '3'),
            layers.Dropout(0.5),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten()
    ])

    classif = models.Sequential([
        backbone,
        layers.Dense(256, activation = 'relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation = 'softmax')
    ])

    return backbone, classif

backbone, classif = pretrainModel()
classif.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
history = classif.fit(train_data, validation_data = val_data,  epochs=epochs, batch_size=batch_size)
backbone.trainable = False
backbone.save_weights('pretrained_weights.h5')

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()