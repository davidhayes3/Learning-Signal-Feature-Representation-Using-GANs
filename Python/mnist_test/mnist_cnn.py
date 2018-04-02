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

batch_size = 128
num_classes = 10
epochs = 100
rgb_dim = 1

callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0)]

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print('x_train shape:', x_train.shape)

plt.imshow(x_train[5])
plt.show()

def cnn(e):
    model = Sequential()
    model.add(e)
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model


# input image dimensions
img_rows, img_cols = x_train.shape[1], x_train.shape[2]


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], -1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], -1, img_rows, img_cols)
    input_shape = (rgb_dim, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, -1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, -1)
    input_shape = (img_rows, img_cols, rgb_dim)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
print(model.count_params())
model.add(Flatten())
print(model.count_params())
print(model.output_shape)
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
print(model.count_params())

e = encoder_model()
e.load_weights('encoder.h5')
model = cnn(e)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=callbacks,
          validation_split=1/12.)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
