#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 02:10:29 2019

@author: yang
"""

from keras.applications import VGG16
import data
import tensorflow as tf
from keras import models
from keras import layers
from keras import optimizers


sess = tf.Session()

try:
    X_train
except NameError:
    X_train, y_train, X_test, y_test, file_name = data.CIFAR10_getData()
    X_train_resize = tf.image.resize_images(X_train, [48,48], method=0)
    X_test_resize = tf.image.resize_images(X_test, [48,48], method=0)
    X_train = sess.run(X_train_resize)
    X_test = sess.run(X_test_resize)
else:
    pass;


conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(48,48,3))
conv_base.summary()

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

dataagen = ImageDataGenerator(rescale=1./255)
batch_size = 100

def extract_features(x_data,x_label):
    features = np.zeros(shape=(x_data.shape[0], 1, 1, 512))
    labels = np.zeros(shape=(x_data.shape[0]))
    generator = dataagen.flow(x=x_data, y=x_label, batch_size=batch_size)
    i = 0
    for inputs_batch, label_batch in generator:
        feature_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i+1) * batch_size] = feature_batch
        labels[i * batch_size : (i+1) * batch_size] = label_batch
        i += 1
        if i * batch_size >= x_data.shape[0] / batch_size:
            break;
    
    return features, labels

try:
    train_features
except NameError:
    train_features, train_labels = extract_features(X_train, y_train)
    test_features, test_labels = extract_features(X_test, y_test)
    onehot = tf.one_hot(train_labels, depth=10)
    train_labels = sess.run(onehot)
    onehot1 = tf.one_hot(test_labels, depth=10)
    test_labels = sess.run(onehot1)
    train_features = np.reshape(train_features, (train_features.shape[0], 1*1*512))
    test_features = np.reshape(test_features, (test_features.shape[0], 1*1*512))
else:
    pass;

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=1*1*512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
              loss='categorical_crossentropy',
              metrics=['acc'])

history = model.fit(train_features, train_labels,
                    epochs=10,
                    batch_size=300,
                    validation_data=(test_features, test_labels))


        
        
        
        
        
        
        
        
        
        


