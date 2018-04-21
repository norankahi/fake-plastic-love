import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Flatten, Dense, Activation
from tensorflow.python.keras.models import Sequential

num_classes = 10

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Flatten(input_shape=(32,32,3)))
model.add(Dense(10))
model.add(Activation('softmax'))

opt = keras.optimizers.Adadelta(lr=5e-2, decay=1e-6)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=500, epochs=100, validation_data=(x_test, y_test), shuffle=True)

