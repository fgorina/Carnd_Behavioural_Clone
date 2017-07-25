import numpy as np
import random
import cv2
from datetime import datetime
import matplotlib.pyplot as plt
import os
import csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras import regularizers
from sklearn.preprocessing import LabelBinarizer

### training directory

test_data = "./test_data"


### Setup model saving directory
#directory = "./logs/keras_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_filename = "model.hf5"

model = load_model(model_filename)

image_list = []

for file in os.listdir(test_data):
    if file.endswith(".jpg"):
        image = plt.imread(test_data + "/" + file)
        image_list.append(image)

image_list = np.array(image_list)

preds = model.predict(image_list)


print(preds)
