import time
import cv2

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt


def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, 'rb') as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


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

    img = np.array(img)

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

    img = np.array(img)

    return img


def test():
    img_file = './d1.jpg'
    img_size = 128

    img = read_image_resize_ratio(img_file, img_size)

    plt.title('final')
    plt.imshow(img)
    plt.show()


def main():
    print('--start--')
    img_file = './t1.jpg'

    # runs = '2018-04-27_00-17-32,lr=0.001,b=32,e=30'
    runs = '2018-04-28_19-48-03,opt=Adam,lr=0.001,b=32,e=30'

    model_file = '../files/training_logs/' + runs + '/model/frozen_flower_convnet.pb'
    label_file = '../files/training_logs/' + runs + '/model/trained_labels.txt'

    img_size = 128
    num_channels = 3

    input_layer = 'conv2d_1_input'
    output_layer = 'dense_2/Softmax'

    graph = load_graph(model_file)

    img = read_image_resize_ratio(img_file, img_size)
    # img = read_image_resize(img_file, img_size)

    # plt.imshow(img)
    # plt.show()

    x = img.reshape(1, img_size, img_size, num_channels)

    input_name = 'import/' + input_layer
    output_name = 'import/' + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.Session(graph=graph) as sess:
        start = time.time()
        results = sess.run(output_operation.outputs[0],
                           {input_operation.outputs[0]: x})
        end = time.time()
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)

    print('\nEvaluation time (1-image): {:.3f}s\n'.format(end - start))

    for i in top_k:
        print(labels[i], results[i])

    print('--end--')


if __name__ == '__main__':
    main()
