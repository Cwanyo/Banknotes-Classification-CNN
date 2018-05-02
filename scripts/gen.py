import os
import re

import cv2
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt


def view_data_generator(input_data_dir, output_data_dir, img_size, batch_size):
    # Random distortion
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    # train_datagen = ImageDataGenerator(rescale=1. / 255)

    i = 0
    for batch in train_datagen.flow_from_directory(input_data_dir,
                                                   target_size=(img_size, img_size),
                                                   color_mode='rgb',
                                                   class_mode='categorical',
                                                   batch_size=batch_size,
                                                   shuffle=True, save_to_dir=output_data_dir):
        i += 1
        if i > 20:
            break


input_dir = '../files/thaibaht_photos_diff_2/train'
output_dir = '../files/thaibaht_photos_diff_2/train_gen'
view_data_generator(input_dir, output_dir, 128, 100)
