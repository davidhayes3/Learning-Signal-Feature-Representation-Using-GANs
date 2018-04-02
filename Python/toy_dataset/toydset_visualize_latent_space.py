from __future__ import print_function

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, merge, LeakyReLU, Dropout, concatenate, BatchNormalization
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import numpy as np


# Settings
latent_dim = 2
img_dim = 4

def encoder_model():
    model = Sequential()

    model.add(Dense(512, input_dim=img_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(latent_dim))

    return model


# Load dataset
x_train = np.loadtxt('Dataset/toy_dataset_x_train.txt', dtype=np.float32)
x_test = np.loadtxt('Dataset/toy_dataset_x_test.txt', dtype=np.float32)
y_train = np.loadtxt('Dataset/toy_dataset_y_train.txt', dtype=np.int)
y_test = np.loadtxt('Dataset/toy_dataset_y_test.txt', dtype=np.int)

# Rescale -1 to 1
x_train = (x_train - 0.5) / 0.5
x_test = (x_test - 0.5) / 0.5

num_classes = 4

e = encoder_model()
e.load_weights('Models/encoder.h5')
z = e.predict(x_train)


fig = plt.figure()
ax = fig.add_subplot(111)
colors = cm.Spectral(np.linspace(0, 1, num_classes))

xx = z[:,0]
yy = z[:,1]

labels = range(num_classes)

# plot the 2D data points
for i in range(num_classes):
    ax.scatter(xx[y_train==i], yy[y_train==i], color=colors[i], label=labels[i], s=10)

plt.axis('tight')
plt.legend(loc='best', scatterpoints=1, fontsize=5)
plt.savefig('foo.png')
plt.show()