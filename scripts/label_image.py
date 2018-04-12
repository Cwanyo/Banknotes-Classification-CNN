import time
import cv2

import numpy as np
import tensorflow as tf


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


def read_image(img_file, img_size):
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32)
    img = np.multiply(img, 1.0 / 255.0)
    img = np.array(img)
    return img


def main():
    print('--start--')
    img_file = '../20.jpg'
    model_file = '../files/models/2018-04-13_00-42-43,lr=1.0,b=32,e=30/frozen_banknotes_convnet.pb'
    label_file = '../files/models/2018-04-13_00-42-43,lr=1.0,b=32,e=30/trained_labels.txt'

    img_size = 128
    num_channels = 3

    input_layer = 'conv2d_1_input'
    output_layer = 'dense_2/Softmax'

    graph = load_graph(model_file)
    img = read_image(img_file, img_size)

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
