import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout, Lambda

from keras.layers.convolutional import Conv2D, Cropping2D

from sklearn.utils import shuffle


### Loading Data.
# buildDirectory builds a list of images and steering values from a list of directories
# Directories must contain a driving_log.csv file and a IMG directory
# Corrects steering from left or right camera with "correction" value
#


def buildDirectory(filelist):

    rows = []

    correction = 0.65

    for filename in filelist:

        rootpath = filename
        prefix = rootpath + '/IMG/'  # subdirectory for images

        log = rootpath + '/driving_log.csv'

        gtFile = open(log)  # annotations file
        gtReader = csv.reader(gtFile, delimiter=',')  

        for arow in gtReader:
            if arow[0] != "center":
                ster = float(arow[3])  # We have the data!!!
                for i in range(0, 3):
                    if arow[i] != "*":
                        image_name = arow[i].split("/")[-1]
                        image_path = prefix + image_name

                        if i == 0:
                            c = 0.0
                        elif i == 1:
                            c = correction
                        else:
                            c = -correction
                        rows.append([image_path, ster+c])
        gtFile.close()

    return train_test_split(rows, test_size=0.2)

### This is the generator. Reads a list where each element is a image filename and a steering value
# and generates apropiate batches of img's and steering values.
#

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = batch_sample[0]
                image = plt.imread(name)
                angle = float(batch_sample[1])
                images.append(image)
                angles.append(angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield (X_train, y_train)

### training directories.
#
# Multiple training logs may be concatenated.
#
# Generated is generated with the generate program. It is used in som train processes.
#
# Doesn't seem to be of much use but surely may be interesting in some cases.
#

training_data = [ "./training_data/Generated", "./training_data/Run_2", "./training_data/Run_6","./training_data/Run_8", "./training_data/Run_11_Shorts","./training_data/Run_13_rec", "./training_data/Run_12", "./training_data/Run_10_Shorts"]


### Setup model saving filename. Includes date and time not to overwrite old models

model_filename = "model" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+".hf5"


# Define some parameters
epochs = 15


# Build model according Nvidia paper inserting Dropout layer

model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(((60,25),(24,24))))

model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu', padding='valid'))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu', padding='valid'))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu', padding='valid'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='valid'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='valid'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))


# Train the model


model.compile(optimizer='adam', loss='mse')


# Load the data and train the model


batch_size = 32

train_samples, validation_samples = buildDirectory(training_data)

# Build the generators

train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

print("Train Samples ", len(train_samples), "Validation Samples ", len(validation_samples))

model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/batch_size,
                    validation_data=validation_generator,
                    validation_steps=len(validation_samples)/batch_size, nb_epoch=epochs)

model.save(model_filename)

