#!/usr/bin/env python
# -*- coding: utf-8 -*-
##
##  Module run.py
##  Python3 based module
##  Created:  Mon 10 17:21:55 GMT 2020 by Samaresh Nayak

"""
This module follows a fuzzy deep neural network model for image classification
on the CIFAR-10 dataset.

optional arguments:
  -h, --help            show this help message and exit
  --learning-rate LEARNING_RATE
                        Learning Rate of your classifier. Default 0.0001
  --epoch EPOCHS        Number of times you want to train your data. Default
                        100
  --batch-size BATCH_SIZE
                        Batch size for prediction. Default=16.
  --colour-image        Passing this argument will keep the coloured image
                        (RGB) during training. Default=False.
  --membership-layer-units MEMBERSHIP_LAYER_UNITS
                        Defines the number of units/nodes in the Membership
                        Function Layer
  --first-dr-layer-units DR_LAYER_1_UNITS
                        Defines the number of units in the first DR Layer
  --second-dr-layer-units DR_LAYER_2_UNITS
                        Defines the number of units in the second DR Layer
  --fusion-dr-layer-units FUSION_DR_LAYER_UNITS
                        Defines the number of units in the Fusion DR Layer
  --fusion-dr-layer-units FUSION_DR_LAYER_UNITS
                        Defines the number of units in the Fusion DR Layer
  --hide-graph          Hides the graph of results displayed via matplotlib

example usage:
    run.py --epoch 100 --batch-size 8 --learning-rate 0.001
           --membership-layer-units 256 --first-dr-layer-units 128
           --second-dr-layer-units 64
"""

__author__ ='Samaresh Nayak'
__version__ = '1.0'

import argparse
from pprint import pprint

import numpy as np
from keras.datasets import cifar10
from keras.utils import to_categorical
from matplotlib import pyplot
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Multiply, Concatenate
from tensorflow.keras.optimizers import Adam

from FuzzyLayer import FuzzyLayer

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# plot diagnostic learning curves
def summarise_diagnostics(history):
    # plot accuracy
    pyplot.subplot(211)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Accuracy')

    # plot loss
    pyplot.subplot(212)
    pyplot.title('Mean Squared Error Loss')
    pyplot.plot(history.history['loss'], color='orange', label='train')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Loss')

    pyplot.tight_layout()
    pyplot.show()


def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    # one hot encode labels
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY


def pre_process_data(train, test, keep_rgb=False):
    # convert from integers to floats (To normalize correctly)
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0

    # Keep RGB or convert it to Grey Image
    if keep_rgb:
        # Flatten the (32x3) dimensions to (96) i.e. Flatten the image for Dense layer
        colour_processed_train = train_norm.reshape(*train_norm.shape[:2], -1)
        colour_processed_test = test_norm.reshape(*test.shape[:2], -1)
    else:
        # convert to grey-scale image
        colour_processed_train = np.dot(train_norm[..., :3], [0.2989, 0.5870, 0.1140])
        colour_processed_test = np.dot(test_norm[..., :3], [0.2989, 0.5870, 0.1140])

    # Swap 1st Dimension of the data i.e. number of examples with 2nd Dimension i.e. Number of rows in an image
    # This is important to fetch the Membership function as it takes an input vector (1x32) and we have a matrix (32x32)
    x_train_swap = np.einsum('kli->lki', colour_processed_train)
    x_test_swap = np.einsum('kli->lki', colour_processed_test)

    # Since "Membership function" feeds an input vector we convert our data to list of vectors
    # 32 lists of (32 size) vector makes it a single training/testing example or an image
    x_train_multi_input = [x for x in x_train_swap]
    x_test_multi_input = [x for x in x_test_swap]

    # Return List of train/test examples
    return x_train_multi_input, x_test_multi_input


def prepare_model(input_len, input_shape, num_classes, parameters):
    fuzz_membership_layer = []
    model_inputs = []
    for vector in range(input_len):
        model_inputs.append(Input(shape=(input_shape,)))
        # Membership Function layer
        fuzz_membership_layer.append(FuzzyLayer(parameters.membership_layer_units)(model_inputs[vector]))
    # Fuzzy Rule Layer
    rule_layer = Multiply()(fuzz_membership_layer)

    inp = Concatenate()(model_inputs)
    # Input DR Layers
    dr_layer_1 = Dense(parameters.dr_layer_1_units, activation='sigmoid')(inp)
    dr_layer_2 = Dense(parameters.dr_layer_2_units, activation='sigmoid')(dr_layer_1)

    # Fusion Layer
    fusion_layer = Concatenate()([rule_layer, dr_layer_2])

    # Fusion DR Layer
    fusion_dr_layer = Dense(parameters.fusion_dr_layer_units, activation='sigmoid')(fusion_layer)

    # Task Driven Layer
    out = Dense(num_classes, activation='softmax')(fusion_dr_layer)
    model = Model(model_inputs, out)
    # compile model
    opt = Adam(learning_rate=parameters.learning_rate)
    model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
    return model


def parse_cli_parameters():
    parser = argparse.ArgumentParser(description="FuzzyDNN on CIFAR-10")
    parser.add_argument('--learning-rate', dest='learning_rate', default=10 ** -3, type=float,
                        help='Learning Rate of your classifier. Default 0.001')
    parser.add_argument('--epoch', dest='epochs', default=100, type=int,
                        help='Number of times you want to train your data. Default 100')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=16,
                        help='Batch size for prediction. Default=16.')
    parser.add_argument('--colour-image', dest='is_colour_image', action="store_true", default=False,
                        help='Passing this argument will keep the coloured image (RGB) during training. Default=False.')
    parser.add_argument('--membership-layer-units', dest='membership_layer_units', type=int, default=100,
                        help='Defines the number of units/nodes in the Membership Function Layer')
    parser.add_argument('--first-dr-layer-units', dest='dr_layer_1_units', type=int, default=100,
                        help='Defines the number of units in the first DR Layer')
    parser.add_argument('--second-dr-layer-units', dest='dr_layer_2_units', type=int, default=100,
                        help='Defines the number of units in the second DR Layer')
    parser.add_argument('--fusion-dr-layer-units', dest='fusion_dr_layer_units', type=int, default=100,
                        help='Defines the number of units in the Fusion DR Layer')
    parser.add_argument('--hide-graph', dest='should_hide_graph', action="store_true", default=False,
                        help='Hides the graph of results displayed via matplotlib')

    options = parser.parse_args()

    print("Starting with the following options:")
    pprint(vars(options))

    return options


def main():
    cli_parameters = parse_cli_parameters()
    X_train, y_train, X_test, y_test = load_dataset()
    X_train, X_test = pre_process_data(X_train, X_test, keep_rgb=cli_parameters.is_colour_image)

    # Defines the number of classes/categories and output vectors
    num_classes = y_test.shape[-1]
    # Defines the number of input vectors
    input_length = len(X_train)
    # Defines the shape of input layer
    input_shape = X_train[0].shape[-1]
    model = prepare_model(input_length, input_shape, num_classes, cli_parameters)
    # fit model
    history = model.fit(X_train, y_train, epochs=cli_parameters.epochs, batch_size=cli_parameters.batch_size)
    print('Evaluating the model')
    _, acc = model.evaluate(X_test, y_test)

    print('Model Evaluation Accuracy: {}'.format(acc))

    if not cli_parameters.should_hide_graph:
        summarise_diagnostics(history)


if __name__ == '__main__':
    main()
