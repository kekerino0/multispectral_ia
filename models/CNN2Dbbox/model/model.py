
from keras import layers, models
import os 
import cv2
import matplotlib.pyplot as plt
import dataGenerator
from keras.optimizers import Adam
import numpy as np
from sklearn.model_selection import KFold
import keras
import meanAveragePrecision

#import tensorflow_ranking as tfr 


'''
train_input_dir = r'data\MeldrickData\train\images'
train_label_dir = r'data\MeldrickData\train\labels'

val_input_dir = r'data\MeldrickData\validation\images'
val_label_dir = r'data\MeldrickData\validation\labels'

test_input_dir = r'data\MeldrickData\test\images'
test_label_dir = r'data\MeldrickData\test\labels'
'''

image_dir = r'CNN2DBbox\data\MeldrickData\images'
label_dir = r'CNN2DBbox\data\MeldrickData\labels'

list_image_path = os.listdir(image_dir)
list_label_path = os.listdir(label_dir)

# print(list_image_path)
# print(list_label_path)



# print(dict_images, dict_labels)

input_shape = (500,500,9)
img_size = (500,500)
batch_size = 1
epochs = 40
max_num_box = 2
num_classes = 2
lr = 0.001 # ça fait rien
len_dataset = 20
#print('input_img_paths : ',input_img_paths)

def model_bb(input_shape=input_shape, num_classes = num_classes):
    
    input = layers.Input(shape=input_shape)

    data_augmentation = models.Sequential([
        layers.RandomFlip(mode = 'HORIZONTAL_AND_VERTICAL'),
        layers.RandomRotation(0.5),
        layers.GaussianNoise(0.3)
    ])

    '''base_model = keras.applications.MobileNetV2(input_shape=input_shape,
                                               include_top=False,
                                               weights='imagenet')'''
    
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
    
    Bbox = models.Sequential([
        layers.Dense(256, activation='relu', name = '4'),
        layers.Dense(128, activation='relu', name = '5'),
        layers.Dense(4 * max_num_box, activation='linear', name = 'Bbox_out')
    ])

    classif1 = models.Sequential([
        layers.Dense(256, activation='relu',name = '7' ),
        layers.Dense(128, activation='relu',name = '8'),
        layers.Dense(num_classes, activation='softmax', name = 'Classif1_out') 
    ])

    classif2 = models.Sequential([
        layers.Dense(256, activation='relu',name = '9'),
        layers.Dense(128, activation='relu',name = '10'),
        layers.Dense(num_classes, activation='softmax', name = 'Classif2_out') 
    ])

    Bbox_out = Bbox(features)
    classif1_out = classif1(features)
    classif2_out = classif2(features)

    concatenated_out = layers.Concatenate(axis=1)([classif1_out, classif2_out, Bbox_out])
    #print('bbox out : ' ,Bbox_out)
    #print('classif out : ', classif_out)
    #print('out : ', [classif_out, Bbox_out])
    model = models.Model(inputs = input, outputs = concatenated_out)
    
    return model

dict_images = {}
for index, value in enumerate(list_image_path):
    dict_images[index] = value

dict_labels = {}
for index, value in enumerate(list_label_path):
    dict_labels[index] = value

samples_ids = np.arange(1, len_dataset + 1) + 1
kfold = KFold(4)
folds = kfold.split(samples_ids)

for k, fold in enumerate(folds):
    print('------------------------------------------------------------------------')
    print(f'Training for fold {k} ...')

    train_ids = fold[0]
    test_ids = fold[1]
    train_img_paths = []
    train_label_paths = []
    test_img_paths = []
    test_label_paths = []
    for i in train_ids:
        train_img_paths.append(dict_images[i])
        train_label_paths.append(dict_labels[i])
    for i in test_ids:
        test_img_paths.append(dict_images[i])
        test_label_paths.append(dict_labels[i])
    print(train_img_paths, train_label_paths,test_img_paths,test_label_paths)
    train_data = dataGenerator.DataGenerator(label_dir = label_dir, image_dir = image_dir,max_num_box = max_num_box, num_classes = num_classes, list_of_image_paths = train_img_paths, list_of_label_paths = train_label_paths, batch_size = batch_size, image_size = img_size)
    test_data = dataGenerator.DataGenerator(label_dir = label_dir, image_dir = image_dir,max_num_box = max_num_box, num_classes = num_classes, list_of_image_paths = test_img_paths, list_of_label_paths = test_label_paths, batch_size = batch_size, image_size = img_size)
    model = model_bb()
    
    model.compile(
        optimizer='adam',
        loss= 'mean_squared_error',
        # metrics= [tfr.keras.metrics.MeanAveragePrecisionMetric()]
        metrics = ['accuracy']
    )
    #history = model.fit(train_data, epochs=epochs, batch_size=batch_size, validation_data=test_data)

#train_data = dataGenerator.augment_data(train_data, True,True)


    # model.summary()
'''    plt.plot(history.history['accuracy'], label=f'Training Accuracy {k}')
    plt.plot(history.history['val_accuracy'], label=f'Validation Accuracy {k}')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
'''
# model.save(r'CNN2DBbox\model\BboxModel.keras')

# model = keras.models.load_model(r'CNN2DBbox\model\BboxModel.keras')

# mAP = meanAveragePrecision.MeanAveragePrecision(test_data, 0.2)

model.fit(train_data, epochs = epochs, batch_size= batch_size)




