import keras 
import mapcalc
import dataGenerator
import os
from keras_retinanet.utils.eval import evaluate

image_dir = r'CNN2DBbox\data\MeldrickData\images'
label_dir = r'CNN2DBbox\data\MeldrickData\labels'

list_image_path = os.listdir(image_dir)
list_label_path = os.listdir(label_dir)

input_shape = (500,500,9)
img_size = (500,500)
batch_size = 1
epochs = 3
max_num_box = 2
num_classes = 2
lr = 0.001 # ça fait rien
len_dataset = 20

class MeanAveragePrecision(keras.callbacks.Callback):
    def __init__(self, val_data, score_threshold):
        super().__init__()
        self.validation_data = val_data
        self.score_threshold = score_threshold

    def on_epoch_end(self, epoch, logs=None):
        mAP =  evaluate(self.model, self.validation_data, score_threshold=self.score_threshold)
        print('\nMean Average Precision (mAP): {:.4f}'.format(mAP))
        

train_data = dataGenerator.DataGenerator(label_dir = label_dir, image_dir = image_dir,max_num_box = max_num_box, num_classes = num_classes, list_of_image_paths = list_image_path, list_of_label_paths = list_label_path, batch_size = batch_size, image_size = img_size)

callback = MeanAveragePrecision(train_data, 0.3)

model = model = keras.models.load_model(r'CNN2DBbox\model\BboxModel.keras')
model.evaluate(train_data, callbacks = callback)