import os
import re

import cv2
import numpy as np

from tensorflow.python.platform import gfile


class Dataset:

    def __init__(self):
        self.images_data = []
        self.labels_onehot = []
        self.images_name = []
        self.labels_name = []


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
    #
    #

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

        # Shuffle the dataset with fixed random seed
        # img_data, label_onehot, img_name, label_name = shuffle(img_data, label_onehot, img_name, label_name,
        #                                                        random_state=2)

        data.images_data.extend(img_data)
        data.labels_onehot.extend(label_onehot)
        data.images_name.extend(img_name)
        data.labels_name.extend(label_name)

    data.images_data = np.array(data.images_data)
    data.labels_onehot = np.array(data.labels_onehot)
    data.images_name = np.array(data.images_name)
    data.labels_name = np.array(data.labels_name)

    print('Total = {} images'.format(len(data.images_data)))
    print('_________________________________________________________________')

    return data, classes

