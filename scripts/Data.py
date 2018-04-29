import os
import re

import cv2
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt


class Dataset:

    def __init__(self):
        self.images_data = []
        self.labels_onehot = []
        # TODO - not require (optional)
        # self.images_name = []
        # self.labels_name = []


def read_image_resize_ratio(img_file, img_size):
    # Read image
    img = cv2.imread(img_file)

    # cv2 load image as BGR, so convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Find resize scale
    min_size = min(img.shape[0], img.shape[1])
    scale = 1 / (min_size / img_size)

    # Resize image to nearest img_size
    # REF: http://tanbakuchi.com/posts/comparison-of-openv-interpolation-algorithms/
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    # img = cv2.resize(img, dsize=None, fx=scale, fy=scale)

    # Crop center
    height, width, channels = img.shape
    upper_left = (int((width - img_size) / 2), int((height - img_size) / 2))
    bottom_right = (int((width - img_size) / 2) + img_size, int((height - img_size) / 2) + img_size)
    img = img[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]].copy()

    # Save as float 32
    img = img.astype(np.float32)

    # Get value between 0-1 from 0-255
    img = np.multiply(img, 1.0 / 255.0)

    return img


def read_image_resize(img_file, img_size):
    # Read image
    img = cv2.imread(img_file)

    # cv2 load image as BGR, so convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize image
    # REF: http://tanbakuchi.com/posts/comparison-of-openv-interpolation-algorithms/
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    # img = cv2.resize(img, (img_size, img_size))

    # Save as float 32
    img = img.astype(np.float32)

    # Get value between 0-1 from 0-255
    img = np.multiply(img, 1.0 / 255.0)

    return img


def load_data(img_dir, img_size):
    if not gfile.Exists(img_dir):
        print('Image directory ' + img_dir + ' not found.')
        return None

    # Get list of folders
    sub_dirs = [os.path.join(img_dir, item)
                for item in gfile.ListDirectory(img_dir)]
    sub_dirs = sorted(item for item in sub_dirs if gfile.IsDirectory(item))

    num_classes = len(sub_dirs)
    classes = []

    data = Dataset()

    # For each class
    for sub_dir in sub_dirs:
        extensions = ['jpg', 'jpeg']
        file_list = []

        # Get folder name
        dir_name = os.path.basename(sub_dir)

        if dir_name == img_dir:
            continue

        print('Loading images of class %s ...' % dir_name)

        # Get all images path with valid extension
        for extension in extensions:
            file_glob = os.path.join(img_dir, dir_name, '*.' + extension)
            file_list.extend(gfile.Glob(file_glob))

        if not file_list:
            print('No images found')
            continue

        label = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        classes.append(label)

        # Get index of class
        index = sub_dirs.index(sub_dir)

        # Dataset for each class
        img_data = []
        label_onehot = []

        img_name = []
        label_name = []

        # For each image in class
        for file in file_list:
            # Get name of file
            base_name = os.path.basename(file)
            img_name.append(base_name)

            # Get label name
            label_name.append(label)

            # Read image and resize, ratio resize -> center cropping
            img = read_image_resize_ratio(file, img_size)

            # Read image and resize, squashing
            # img = read_image_resize(file, img_size)

            # plt.imshow(img)
            # plt.show()

            img_data.append(img)

            # One-Hot Label
            loh = np.zeros(num_classes)
            loh[index] = 1.0

            label_onehot.append(loh)

        print('--loaded %d images' % len(img_data))

        # Shuffle the dataset with fixed random seed
        # img_data, label_onehot, img_name, label_name = shuffle(img_data, label_onehot, img_name, label_name,
        #                                                        random_state=2)

        data.images_data.extend(img_data)
        data.labels_onehot.extend(label_onehot)
        # TODO - not require (optional)
        # data.images_name.extend(img_name)
        # data.labels_name.extend(label_name)

    data.images_data = np.array(data.images_data)
    data.labels_onehot = np.array(data.labels_onehot)
    # TODO - not require (optional)
    # data.images_name = np.array(data.images_name)
    # data.labels_name = np.array(data.labels_name)

    x, y = shuffle(data.images_data, data.labels_onehot, random_state=2)

    print('Total = {} images'.format(len(data.images_data)))
    print('_________________________________________________________________')

    return x, y, classes


def load_data_generator(train_data_dir, valid_data_dir, img_size, batch_size):
    # Random distortion
    # train_datagen = ImageDataGenerator(
    #     rescale=1. / 255,
    #     shear_range=0.2,
    #     zoom_range=0.2,
    #     horizontal_flip=True)
    train_datagen = ImageDataGenerator(rescale=1. / 255)

    valid_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_size, img_size),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True, seed=4
    )

    valid_generator = valid_datagen.flow_from_directory(
        valid_data_dir,
        target_size=(img_size, img_size),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False
    )

    return train_generator, valid_generator
