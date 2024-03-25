from keras import layers, models
import keras
import dataGenerator
import os

num_classes = 2
epochs = 10
batch_size = 1
img_size = (500,500)

image_dir = r'C:\Users\STAGIAIRE\Documents\model\CNN2DpretrainClassif\data\gears\images'
label_dir = r'C:\Users\STAGIAIRE\Documents\model\CNN2DpretrainClassif\data\gears\labels'
list_of_img_paths = os.listdir(image_dir)
list_of_label_paths = os.listdir(label_dir)

train_data = dataGenerator.DataGenerator(label_dir=label_dir, image_dir=image_dir, max_num_box=0,num_classes=num_classes, list_of_image_paths=list_of_img_paths, list_of_label_paths=list_of_label_paths, batch_size=batch_size, image_size=img_size)

def model(num_classes = num_classes):
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

    classif = models.Sequential([
        backbone,
        layers.Dense(256, activation = 'relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation = 'softmax')
    ])

    return classif, backbone

classif, backbone = model()
backbone.load_weights('pretrained_weights.h5')
backbone.trainable = False
classif.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
classif.fit(train_data, epochs=epochs, batch_size=batch_size)
classif.save('CNN2DpretrainClassif\model\pretrained_classif.keras')
