import os
import re

import cv2
import numpy as np

from sklearn.utils import shuffle
from tensorflow.python.platform import gfile


class Dataset:

    def __init__(self):
        self.images_data = []
        self.labels_onehot = []
        self.images_name = []
        self.labels_name = []


def load_data(img_dir, img_size, validation_percentage, testing_percentage):
    if not gfile.Exists(img_dir):
        print('Image directory ' + img_dir + ' not found.')
        return None

    # Get list of folders
    sub_dirs = [os.path.join(img_dir, item)
                for item in gfile.ListDirectory(img_dir)]
    sub_dirs = sorted(item for item in sub_dirs if gfile.IsDirectory(item))

    num_classes = len(sub_dirs)
    classes = []

    training_data = Dataset()
    validation_data = Dataset()
    testing_data = Dataset()

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

            # Read image
            img = cv2.imread(file)

            # cv2 load image as BGR, so convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize image
            img = cv2.resize(img, (img_size, img_size))

            # Save as float 32
            img = img.astype(np.float32)

            # Get value between 0-1 from 0-255
            img = np.multiply(img, 1.0 / 255.0)

            img_data.append(img)

            # One-Hot Label
            loh = np.zeros(num_classes)
            loh[index] = 1.0

            label_onehot.append(loh)

        print('--loaded %d images' % len(img_data))

        print('--Shuffling images')

        # Shuffle the dataset with fixed random seed
        r_img_data, r_label_onehot, r_img_name, r_label_name = shuffle(img_data, label_onehot, img_name, label_name,
                                                                       random_state=2)

        # Split the dataset into training dataset, testing dataset and validation dataset
        s_img_data = split_dataset(r_img_data, validation_percentage, testing_percentage)
        s_label_onehot = split_dataset(r_label_onehot, validation_percentage, testing_percentage)
        s_img_name = split_dataset(r_img_name, validation_percentage, testing_percentage)
        s_label_name = split_dataset(r_label_name, validation_percentage, testing_percentage)

        print('--Splitting images (%.1f:%.1f:%.1f) -> Training[%d], Validation[%d] and Testing[%d]' %
              ((1.0 - validation_percentage - testing_percentage), validation_percentage, testing_percentage,
               len(s_img_data[0]), len(s_img_data[1]), len(s_img_data[2])))

        # Get training dataset [0], testing dataset [1] and validation dataset [2]
        training_data.images_data.extend(s_img_data[0])
        training_data.labels_onehot.extend(s_label_onehot[0])
        training_data.images_name.extend(s_img_name[0])
        training_data.labels_name.extend(s_label_name[0])

        validation_data.images_data.extend(s_img_data[1])
        validation_data.labels_onehot.extend(s_label_onehot[1])
        validation_data.images_name.extend(s_img_name[1])
        validation_data.labels_name.extend(s_label_name[1])

        testing_data.images_data.extend(s_img_data[2])
        testing_data.labels_onehot.extend(s_label_onehot[2])
        testing_data.images_name.extend(s_img_name[2])
        testing_data.labels_name.extend(s_label_name[2])

    # Convert to numpy array
    training_data.images_data = np.array(training_data.images_data)
    training_data.labels_onehot = np.array(training_data.labels_onehot)
    training_data.images_name = np.array(training_data.images_name)
    training_data.labels_name = np.array(training_data.labels_name)

    validation_data.images_data = np.array(validation_data.images_data)
    validation_data.labels_onehot = np.array(validation_data.labels_onehot)
    validation_data.images_name = np.array(validation_data.images_name)
    validation_data.labels_name = np.array(validation_data.labels_name)

    testing_data.images_data = np.array(testing_data.images_data)
    testing_data.labels_onehot = np.array(testing_data.labels_onehot)
    testing_data.images_name = np.array(testing_data.images_name)
    testing_data.labels_name = np.array(testing_data.labels_name)

    return training_data, validation_data, testing_data, classes


def split_dataset(dataset, validation_percentage, testing_percentage):
    return np.split(dataset, [int((1.0 - validation_percentage - testing_percentage) * len(dataset)),
                              int((1.0 - testing_percentage) * len(dataset))])


def read_datasets(img_dir, img_size, validation_percentage, testing_percentage):
    class Datasets:
        pass

    datasets = Datasets()

    training_data, validation_data, testing_data, classes = load_data(img_dir, img_size, validation_percentage,
                                                                      testing_percentage)

    datasets.training_data = training_data
    datasets.validation_data = validation_data
    datasets.testing_data = testing_data
    datasets.classes = classes

    print('_________________________________________________________________')
    print('Training   = {} images'.format(len(datasets.training_data.images_data)))
    print('Validation = {} images'.format(len(datasets.validation_data.images_data)))
    print('Testing    = {} images'.format(len(datasets.testing_data.images_data)))
    print('Total      = {} images'.format(
        len(datasets.training_data.images_data) + len(datasets.validation_data.images_data) + len(
            datasets.testing_data.images_data)))

    return datasets
