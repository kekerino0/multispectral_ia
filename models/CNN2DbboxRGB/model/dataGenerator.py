import keras
import os
import numpy as np
from keras.utils import load_img, img_to_array
import csv
from PIL import UnidentifiedImageError
import cv2
import numpy as np
from keras.utils import Sequence
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import KFold

AUTOTUNE = tf.data.AUTOTUNE

class DataGenerator(Sequence):
    def __init__(self, label_dir, image_dir, max_num_box, num_classes, list_of_image_paths, list_of_label_paths, batch_size=32, image_size=(256, 256)):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.max_num_box = max_num_box
        self.num_classes = num_classes
        self.list_of_image_paths = list_of_image_paths
        self.list_of_label_paths = list_of_label_paths
        self.batch_size = batch_size
        self.image_size = image_size

    def __len__(self):
        return int((len(self.list_of_image_paths) / self.batch_size))

    def __getitem__(self, index):

        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        batch_image_paths = self.list_of_image_paths[start_idx:end_idx]
        batch_label_paths = self.list_of_label_paths[start_idx:end_idx]

        batch_images = [self.load_and_preprocess_image(image_path) for image_path in batch_image_paths]
        batch_labels = [self.load_and_preprocess_label(label_path) for label_path in batch_label_paths]
        #print('batch_label : ' ,batch_labels)
        #batch_label = np.concatenate(batch_label)
        batch_images = np.array(batch_images)
        batch_labels = np.array(batch_labels)
        print(batch_labels)
        #print('image batch shape : ',batch_images.shape)
        #print('label label shape : ',batch_labels.shape)
        
        return batch_images, batch_labels 
    
    def to_cat_every_class(self, label_content):
        cat_label = []
        classes = []
        boxes = []
        for i in range(0, len(label_content), 5):
            bbox = label_content[i:i+5]
            cat = tf.keras.utils.to_categorical(bbox[0], self.num_classes)
            box = label_content[i+1:i+5]
            boxes.append(box)
            classes.append(cat)
        #classes = np.concatenate(classes,axis=0)
        #flattened_list = [item for sublist in boxes for item in sublist]
        classes = np.concatenate(classes)
        boxes = np.concatenate(boxes)
        cat_label.append(classes)
        cat_label.append(boxes)
        #print('cat label : ',cat_label)
        cat_label = np.concatenate(cat_label)
        #print('classified box : ',cat_label)    
        return cat_label
        #return classes, boxes

    def pad(self, cat_label):
        while (len(cat_label) != (self.num_classes + 4)*self.max_num_box):
            cat_label = np.concatenate([cat_label, [0]])
        return cat_label

    def pad_class(self, class_label):
        print('class_label : ',class_label)
        while(len(class_label) != self.num_classes*self.max_num_box):
            class_label = np.concatenate([class_label, [0]])
        return class_label
        
    def pad_box(self, box_label):
        while(len(box_label) != 4*self.max_num_box):
            box_label = np.concatenate([box_label,[0]])
        return box_label
    
    def load_and_preprocess_label(self, label_path):
        with open(self.label_dir + '\\' +label_path ,'r') as label_file:
            content = label_file.read()
        lignes = content.strip().split('\n')
        label_content = [int(valeur) for ligne in lignes for valeur in ligne.split(',')]
        label_content = self.to_cat_every_class(label_content)
        #classes_content = self.pad_class(classes_content)
        #boxes_content = self.pad_box(boxes_content)
        label_content = self.pad(label_content)
        #print('class content : ',classes_content)
        #print('boxes content : ', boxes_content)
        return label_content

    def load_and_preprocess_image(self, image_path):
        # returns one image cube
        grayScaleImagesPath = self.image_dir + '\\' + image_path + '\\Rendered_Image\\Bands' 
        listGrayImages = os.listdir(grayScaleImagesPath)
        image = []
        for grayImage in listGrayImages:
            grayScaleImagePath = os.path.join(grayScaleImagesPath, grayImage)
            try:
                img = load_img(grayScaleImagePath, target_size=self.image_size)
                img_array = img_to_array(img)
                img_array = cv2.cvtColor(img_array.astype('uint8'), cv2.COLOR_RGB2GRAY)
                #print(img_array.shape)
                #print(f"Image shape before concatenation: {img_array.shape}")
                image.append(img_array)
            except UnidentifiedImageError as e:
                print(f"Error loading image {grayScaleImagePath}: {str(e)}")

        # passer en niveau de gris les images grises

        if image:
            # Ensure each image has 31 channels
            final_image = np.stack(image, axis=2)
            #final_image = np.expand_dims(final_image, axis = 0)
            print('Final concatenated shape:', final_image.shape)
            return final_image
        else:
            return None
