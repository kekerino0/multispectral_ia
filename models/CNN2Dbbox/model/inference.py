from keras.utils import load_img, img_to_array
import cv2
import os
from PIL import UnidentifiedImageError
import numpy as np
import keras
import keras.metrics

image_size = (500,500)
image_dir = r'data\MeldrickData\validation\images'
image_path = r'ScenePET1'

def load_and_preprocess_image(image_path):
        # returns one image cube
        grayScaleImagesPath = image_dir + '\\' + image_path + '\\Rendered_Image\\Bands' 
        listGrayImages = os.listdir(grayScaleImagesPath)
        image = []
        for grayImage in listGrayImages:
            grayScaleImagePath = os.path.join(grayScaleImagesPath, grayImage)
            try:
                img = load_img(grayScaleImagePath, target_size=image_size)
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
            #print('Final concatenated shape:', final_image.shape)
            return final_image
        else:
            return None
        
img = load_and_preprocess_image(image_path)
print(img.shape)
model = keras.models.load_model(r'CNN2DBbox\model\BboxModel.keras')
img = np.expand_dims(img, axis=0)
print(img.shape)
result = model.predict(img)
print(result)

result = result[0]

start_point1 = (int(result[4]),int(result[5]))
end_point1 = (int(result[4] + result[6]),  int(result[5] + result[7]))
start_point2 = (int(result[8]),int(result[9]))
end_point2 = (int(result[8]+ result[10]),  int(result[9] + result[11]))

print('start 1 :',start_point1 , " end 1 : ",end_point1, " start 2 : ", start_point2," end 2 : ", end_point2)

color1 = (255,0,0)
color2 = (0,255,0)
thickness = 1

image = cv2.imread(image_dir + '\\' + image_path + '\\Rendered_Image\\Bands\\band_1.png')
image = cv2.resize(image, (500,500))
image = cv2.rectangle(image,start_point1,end_point1,color1,thickness)
image = cv2.rectangle(image,start_point2,end_point2,color2,thickness)

cv2.imshow('boxes',image)
cv2.waitKey(0) 
