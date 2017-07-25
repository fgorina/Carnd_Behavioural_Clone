import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import os
import csv

### Generates new data from actual logs.
#
# Generated format is with the same format as a log
# But left and right camera image name are *
#
# Transformations are car axis displacement and rotations.
#


dir_list = [ "./training_data/Run_2", "./training_data/Run_6","./training_data/Run_8", "./training_data/Run_11_Shorts","./training_data/Run_13_rec", "./training_data/Run_12", "./training_data/Run_10_Shorts"]
output_dir = "./training_data/Generated"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image_dir = output_dir + "/IMG"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)



## Generates new images

set_rows = []

### SHIFT  is whem we move the car left or right on the road :
#   - Horizon at y = 60
#   - Negative shift means go to the left of the road
#   - Positive shift means go to the right of the road
#
#   Shift moves baseline by the amount we require
#   and linearly less till it moves nothing at the horizon.
#   Over the horizont it is not moved
#
#   Correction to be added to the steering command of base image
#   is a linear proportion of a base turning defined by the left and right images.
#
#   left and right images move about 60 pixels
#

def shift(image, delta, base): #returns new image + new correction. delta<0 left, delta>0, right
    y0 = 60
    ymax = 160

    rows, cols = image.shape[:2]

    map_x = np.zeros((rows, cols), np.float32)
    map_y = np.zeros((rows, cols), np.float32)


    for i in range(0, rows):
        for j in range(0, cols):
            if i >= y0:
                map_x[i, j] = j + delta / (ymax-y0) * (i - y0)
            else:
                map_x[i, j] = j

            map_y[i, j] = i

    img1 = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)

    return img1, -delta * base / 60.0


### TURN  is whem we turn the car without moving it :
#   - Horizon at y = 60
#   - Negative turn  means clockwise turn
#   - Positive turn means counterclockwise
#
#   Shift moves over the horizon by the amount selected
#   and linearly less till it moves nothing at the bottom.
#
#   Correction to be added to the steering command of base image
#   is a linear proportion of a base turning defined by the left and right images.
#
#   It is computed siilary as in the turn case
#


def turn(image, delta, base): #returns new image + new correction. delta<0 left, delta>0, right
    y0 = 60
    ymax = 160

    rows, cols = image.shape[:2]

    map_x = np.zeros((rows, cols), np.float32)
    map_y = np.zeros((rows, cols), np.float32)


    for i in range(0, rows):
        for j in range(0, cols):
            if i >= y0:
                map_x[i, j] = j + delta * (ymax - i) / (ymax-y0)
            else:
                map_x[i, j] = j + delta

            map_y[i, j] = i

    img1 = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)

    return img1, delta * base / 60.0


## Creates a list of images from the list of log directories

def buildDirectory(filelist):

    rows = []

    correction = 0.65

    for filename in filelist:

        rootpath = filename
        prefix = rootpath + '/IMG/'  # subdirectory for images

        log = rootpath + '/driving_log.csv'

        output_dir = rootpath + "/GEN"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)



        gtFile = open(log)  # annotations file
        gtReader = csv.reader(gtFile, delimiter=',')

        for arow in gtReader:
            if arow[0] != "center":
                ster = float(arow[3])  # We have the data!!!
                for i in range(0, 3):
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

    return rows

## Generates the new rows

def generate(rows):

    '''Reads training log'''
    global ip_pos
    global set_rows


    nrows = len(rows)

    index = []

    i = 0   # Comptador per els noms de les imatges

    for row in rows:

        # Llegim  la imatge

        image_name = row[0]
        image = plt.imread(image_name)
        ster = float(row[1])  # We have the data!!!


        # Left and right images are displaced just 60 pixels
        #
        # 0.65 is the correction assigned to left and right camera images
        #

        nom = "/img_" + str(i)+".jpg"
        path = image_dir + nom
        i = i + 1

        off = random.gauss(0.0, 30.0)
        img1, str1 = shift(image, off, 0.65)

        index.append([path, "*", "*", ster+str1, 0, 0])
        plt.imsave(path, img1, format="jpg")

        # Rotations not used for the moment
        #nom = "/img_" + str(i)+".jpg"
        #path = image_dir + nom
        #i = i + 1


        #off = random.gauss(0.0, 30.0)
        #img1, str1 = turn(image, off, 0.3)
        #index.append([path, "*", "*", ster+str1, 0, 0])
        #plt.imsave(path, img1, format="jpg")

    with open(output_dir+'/driving_log.csv', "w") as f:
        writer = csv.writer(f)
        writer.writerows(index)


    return

rows = buildDirectory(dir_list)
generate(rows)
