from __future__ import print_function
import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np


# Settings
latent_dim = 2
img_dim = 4

num_classes = 16
num_train_examples = 3000 * num_classes
num_test_examples = 500 * num_classes




y_train = np.random.choice(list(range(num_classes)), size=(num_train_examples,))
y_test = np.random.choice(list(range(num_classes)), size=(num_test_examples,))

#x_train = keras.utils.to_categorical(y_train, num_classes)
#x_test = keras.utils.to_categorical(y_test, num_classes)

x_train = np.zeros((num_train_examples, img_dim))
x_test = np.zeros((num_test_examples, img_dim))

for i, y in enumerate(y_train):
    x_train[i] = np.array([int(x) for x in list('{:04b}'.format(y))])

for i, y in enumerate(y_test):
    x_test[i] = np.array([int(x) for x in list('{:04b}'.format(y))])

for x in x_train:
    for i in range(img_dim):
        if x[i] == 1:
            x[i] = x[i] - abs(np.random.normal(0, 0.05))
        elif x[i] == 0:
            x[i] = x[i] + abs(np.random.normal(0, 0.05))

for x in x_test:
    for i in range(img_dim):
        if x[i] == 1:
            x[i] = x[i] - abs(np.random.normal(0, 0.1))
        elif x[i] == 0:
            x[i] = x[i] + abs(np.random.normal(0, 0.1))


print(x_train[0:100])


#x_train = x_train + np.random.normal(0, 0.1, (x_train.shape[0],img_dim))
#x_test = x_test + np.random.normal(0, 0.1, (x_test.shape[0],img_dim))

for data, name in [(x_train, 'x_train'), (y_train, 'y_train'), (x_test, 'x_test'), (y_test, 'y_test')]:
    np.savetxt('Dataset/toy_dataset_' + name + '.txt', data, fmt='%f')

