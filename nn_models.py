# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 15:26:35 2025

@author: shagh
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def build_simple_ffnn(input_shape=2048, hidden_units=[64, 32], learning_rate=0.001 , **kwargs):
    model = models.Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    for units in hidden_units:
        model.add(layers.Dense(units, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def build_configurable_cnn_1d(input_shape=2048, conv_layers=[(64, 3)], ff_layers=[32], learning_rate=0.001 , **kwargs):
    model = models.Sequential()
    model.add(layers.Input(shape=(input_shape, 1)))

    # Add convolutional layers
    for filters, kernel_size in conv_layers:
        model.add(layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))
        model.add(layers.MaxPooling1D(pool_size=2))

    model.add(layers.Flatten())

    # Add fully connected layers
    for units in ff_layers:
        model.add(layers.Dense(units, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
