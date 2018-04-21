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
x_train = x_train.reshape((-1, 32*32*3))
x_test = x_test.reshape((-1, 32*32*3))
y_train_v = keras.utils.to_categorical(y_train, num_classes)
y_test_v = keras.utils.to_categorical(y_test, num_classes)

from sklearn.linear_model import LogisticRegression

lm = LogisticRegression(solver='sag', max_iter=10, multi_class='multinomial').fit(x_train, y_train)

loss_train, loss_test = -np.mean(lm.predict_log_proba(x_train)*y_train_v)*num_classes, -np.mean(lm.predict_log_proba(x_test)*y_test_v)*num_classes
acc_train, acc_test = lm.score(x_train, y_train), lm.score(x_test, y_test)
print('Train Loss=%6f, Acc=%6f; Test Loss=%6f, Acc=%6f'%(loss_train, acc_train, loss_test, acc_test))

