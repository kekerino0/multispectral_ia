
from keras import layers, models
import os 
import cv2
import matplotlib.pyplot as plt
# import dataGenerator
from keras.optimizers import Adam
import numpy as np
from sklearn.model_selection import KFold
import keras
# import meanAveragePrecision

directory = r'C:\Users\STAGIAIRE\Documents\model\CNN2DBboxRGB\data\MeldrickData\RGB_real'


# print(dict_images, dict_labels)

input_shape = (500,500,3)
img_size = (500,500)
batch_size = 1
epochs = 40
max_num_box = 2
num_classes = 2
lr = 0.001 # ça fait rien
#len_dataset = 20
#print('input_img_paths : ',input_img_paths)

def model_bb(input_shape=input_shape, num_classes = num_classes):
    
    input = layers.Input(shape=input_shape)

    data_augmentation = models.Sequential([
        layers.RandomFlip(mode = 'HORIZONTAL_AND_VERTICAL'),
        layers.RandomRotation(0.5),
        layers.GaussianNoise(0.3)
    ])
    
    input = data_augmentation(input)

    backbone = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', name = '1'),
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

    features = backbone(input)
    '''
    Bbox = models.Sequential([
        layers.Dense(256, activation='relu', name = '4'),
        layers.Dense(128, activation='relu', name = '5'),
        layers.Dense(4 * max_num_box, activation='linear', name = 'Bbox_out')
    ])

    classif1 = models.Sequential([
        layers.Dense(256, activation='relu',name = '7' ),
        layers.Dense(128, activation='relu',name = '8'),
        layers.Dense(num_classes, activation='softmax', name = 'Classif1_out') 
    ])'''

    classif = models.Sequential([
        layers.Dense(256, activation='relu',name = '9'),
        layers.Dense(128, activation='relu',name = '10'),
        layers.Dense(num_classes, activation='softmax', name = 'Classif_out') 
    ])

    classif_out = classif(features)

    model = models.Model(inputs = input, outputs = classif_out)
    
    return model

class_names = ['PET', 'PLA']
train_data = keras.utils.image_dataset_from_directory(directory = directory,
                                                      labels = 'inferred',
                                                      label_mode = 'categorical',
                                                      class_names = class_names,
                                                      batch_size = batch_size,
                                                      image_size = img_size)

model = model_bb(input_shape, num_classes)
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(train_data, epochs=epochs, batch_size=batch_size)
# model.evaluate(validation_data)

model.save(r'CNN2DBboxRGB\model\CNN2DBboxRGB.keras')



