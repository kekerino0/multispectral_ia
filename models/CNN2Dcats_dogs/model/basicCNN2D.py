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

num_classes = 2
batch_size = 30
epochs = 10
count = 0

chat = r'C:\Users\STAGIAIRE\Documents\model\objectDetection\data\kagglecatsanddogs_5340 (1)\PetImages\Cat'
chien = r'C:\Users\STAGIAIRE\Documents\model\objectDetection\data\kagglecatsanddogs_5340 (1)\PetImages\Dog'

def delNonJpg(cheminDossier):
    for filename in os.listdir(cheminDossier):
        if filename.endswith('.jpg') == False :
            print("supprimé : ",cheminDossier + '\\' + filename)
            os.remove(cheminDossier + '\\' + filename)

#delNonJpg(chat)
#delNonJpg(chien)
            
def remove_corrupted_files(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'rb') as file:
                # Try to read the file to check for corruption
                file_content = file.read()
        except Exception as e:
            # If an exception occurs during reading, assume the file is corrupted
            print(f"Corrupted file found: {file_path}")
            os.remove(file_path)
            print(f"Removed: {file_path}")

# Call the function to remove corrupted files
#remove_corrupted_files(chat)
#remove_corrupted_files(chien)
            
def checkDossier(dossier):
    for element in os.listdir(dossier):
        count = 0
        chemin = dossier +'\\'+ element
        #print(chemin)
        #print(cv2.imread(chemin).shape)
        if (type(cv2.imread(chemin)) == type(None)):
            
            os.remove(chemin)
            count = count + 1
            print('supprimé : ', chemin)
    print(count, ' elements deleted')

def checkCanaux(dossier):
    for element in os.listdir(dossier):
        count = 0
        chemin = dossier +'\\'+ element
        #print(chemin)
        try:
            cv2.imread(chemin)
        except Exception as e:
            os.remove(chemin)
            count = count + 1
            print('supprimé : ', chemin)        
    print(count, ' elements deleted')
    

#checkCanaux(chat)
#checkCanaux(chien)
    
def sortClasses(dossier):
    for element in os.listdir(dossier):
        if element.startswith('dog'):
            shutil.move(dossier + '\\' + element, r'C:\Users\STAGIAIRE\Documents\model\objectDetection\data\dog')
        if element.startswith('cat'):
            shutil.move(dossier + '\\' + element, r'C:\Users\STAGIAIRE\Documents\model\objectDetection\data\cat')
    print('sorting done')

#sortClasses(r'C:\Users\STAGIAIRE\Documents\model\objectDetection\data\archive\train_transformed')

ds = keras.utils.image_dataset_from_directory(
    directory = r'C:\Users\STAGIAIRE\Documents\model\CNN2D\data\training',
    labels = 'inferred',
    label_mode = 'categorical',
    batch_size = batch_size,
    image_size = (224,224)
)

print("cardinality : ",tf.data.experimental.cardinality(ds))

#print("train_ds : ",ds)

def addNoise(image, labels):
    noise = np.random.normal(loc = 0, scale = 0.1,)

model = keras.Sequential(
    [
        keras.layers.GaussianNoise(0.5, input_shape = (224,224,3)),
        keras.layers.Conv2D(32,(3,3),activation = 'relu'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(64,(3,3),activation = 'relu'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation = 'relu'),
        keras.layers.Dense(num_classes, activation = 'softmax')
    ]
)

optimizer = keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

"""
train_size = int(0.8 * len(ds))
val_size = len(ds) - train_size
train_ds = ds.take(train_size)
val_ds = ds.skip(train_size)

print("train_size : ",train_size)
print("train_ds : ",train_ds)
"""
for images, labels in ds:
    #for element in images:
        #print(element.shape[2])
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())


# Boucle d'entraînement personnalisée

#for epoch in range(epochs):
#    print(f"\nEpoch {epoch + 1}/{epochs}")
    
#    for step, (x_batch, y_batch) in enumerate(ds):
        # Entraînement sur le batch
#        loss, accuracy = model.train_on_batch(x_batch, y_batch)
        
#        print(f"Step {step + 1} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")

#model.save(r'C:\Users\STAGIAIRE\Documents\model\CNN2D\CNN2D.keras')




