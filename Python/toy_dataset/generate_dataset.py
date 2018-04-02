from __future__ import print_function
import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np


# Settings
latent_dim = 2
img_dim = 4

num_classes = 4
num_train_examples = 5000 * num_classes
num_test_examples = 1000 * num_classes


y_train = np.random.choice([0, 1, 2, 3], size=(num_train_examples,))
y_test = np.random.choice([0, 1, 2, 3], size=(num_test_examples,))

x_train = keras.utils.to_categorical(y_train, num_classes)
x_test = keras.utils.to_categorical(y_test, num_classes)

for x in x_train:
    for i in range(img_dim):
        if x[i] == 1:
            x[i] = x[i] - abs(np.random.normal(0, 0.1))
        elif x[i] == 0:
            x[i] = x[i] + abs(np.random.normal(0, 0.1))

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



