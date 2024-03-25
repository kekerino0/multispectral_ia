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
import random

class DataGenerator(Sequence):
    def __init__(self, folder, num_classes, batch_size, image_size = (500,500)):
        self.folder = folder
        self.class_folders = [class_folder for class_folder in os.listdir(folder)]
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()
        # dict is a dictionnary which contains 'class_name' : [list of all the objects]
        self.dict = {}
        self.list_object_paths = []
        self.class_index = {}
        for i, class_folder in enumerate(self.class_folders):
            objects = os.listdir(os.path.join(self.folder,class_folder))
            self.dict[class_folder] = objects
            self.list_object_paths = self.list_object_paths + objects
            self.class_index[class_folder] = i
        random.shuffle(self.list_object_paths)

    def __len__(self):
        return len(self.list_object_paths) // self.batch_size

    def find_class(self, object_path):
        for class_folder in self.class_folders:
            current_arr = self.dict[class_folder]
            if (object_path in current_arr):
                return class_folder

    # def on_epoch_end(self):
    #     random.shuffle(self.list_object_paths)

    # returns batch
    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size

        # dict = { class1 : [obj1, obj2 ...], class2 = [obj1, obj2 ....], ... }
        # list_object_path = [obj1, obj2, obj3 ... ]

        batch_object_paths = self.list_object_paths[start_idx:end_idx]

        batch_objects = [self.load_and_preprocess_image(object_path) for object_path in batch_object_paths]
        batch_labels = [self.load_and_preprocess_label(object_path) for object_path in batch_object_paths]
        
        batch_objects = np.array(batch_objects)
        batch_labels = np.array(batch_labels)
                
        # print('batch objects : ', batch_objects.shape)
        # print('batch labels : ', batch_labels.shape)

        return batch_objects, batch_labels 
    
    # returns one label
    def load_and_preprocess_label(self, object_path):
        str_label = self.find_class(object_path)
        int_label = self.class_index[str_label]
        cat_label = keras.utils.to_categorical(int_label, num_classes = self.num_classes)
        # print('cat label : ', cat_label)
        return cat_label

    # returns one image cube
    def load_and_preprocess_image(self, object_path):
        # print('object path : ', object_path)
        object_class = self.find_class(object_path)
        grayScaleImagesPath = os.path.join(self.folder, object_class, object_path)
        listGrayImages = os.listdir(grayScaleImagesPath)
        image = []
        for grayImage in listGrayImages:
            grayScaleImagePath = os.path.join(grayScaleImagesPath, grayImage)
            grayImg = cv2.imread(grayScaleImagePath)
            grayImg = cv2.cvtColor(grayImg, cv2.COLOR_BGR2GRAY)
            grayImg = cv2.resize(grayImg, self.image_size)
            image.append(grayImg)
        image = np.stack(image, axis = 2)
        # print('object path : ', object_path)
        # print('image : ', image)
        return image


