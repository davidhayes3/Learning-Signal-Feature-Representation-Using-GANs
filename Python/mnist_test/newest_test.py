'''Trains a simple convnet on the MNIST dataset.
Gets over 99% test accuracy after 12 epochs
3 to 4 seconds per epoch on a TitanX GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
from keras_autoencoder.keras_conv_ae_models import encoder_model
from keras.callbacks import EarlyStopping
import numpy as np


z = np.random.normal(size=(10000, 2))
print(z)

plt.plot(z[:,0], z[:,1],'ro')
plt.show()
