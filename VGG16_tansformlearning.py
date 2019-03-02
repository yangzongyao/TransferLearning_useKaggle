#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 00:05:46 2019

@author: yang
"""


from keras.applications import VGG16
import tensorflow as tf
import data

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

onehot = tf.one_hot(y_train, depth=10)
y_train_onehot = sess.run(onehot)
onehot1 = tf.one_hot(y_test, depth=10)
y_test_onehot = sess.run(onehot1)

conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(48,48,3))
#conv_base.summary()

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

conv_base.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

history = model.fit(x=X_train, y=y_train_onehot, batch_size=500, epochs=30, validation_data=(X_test, y_test_onehot))



