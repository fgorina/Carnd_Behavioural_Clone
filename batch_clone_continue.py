import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split


from keras.models import load_model
from sklearn.utils import shuffle


### Continues training from a pretrained model




### Loading Data.
# First loads images
# Corrects from camera


def buildDirectory(filelist):

    rows = []

    correction = 0.3

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

## See batch_clone_1,py for explanation

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

### training directory

training_data = ["./training_data/Run_14_bends",  "./training_data/Run_15_bends", "./training_data/Generated"]

# Initial model

model_filename = "model_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".hf5"

#if not os.path.exists(directory):
#    os.makedirs(directory)


# Define some parameters
epochs = 15


# Train the model

model = load_model(model_filename)

# Load the data and train the model


batch_size = 256

train_samples, validation_samples = buildDirectory(training_data)


train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/32,
                    validation_data=validation_generator,
                    validation_steps=len(validation_samples)/32, nb_epoch=epochs)

model_new_filename = "new_"+model_filename
model.save(model_new_filename)

