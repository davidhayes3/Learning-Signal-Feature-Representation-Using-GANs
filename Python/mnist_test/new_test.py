from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense, Reshape, Flatten
from keras.models import Sequential
from keras import backend as K

# Define models
def encoder_model():
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     input_shape=(28, 28, 1))) # if channels_first
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Dense(1024, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    return model

def decoder_model():
    model = Sequential()
    model.add(Dense(input_dim=100, units=1024))
    model.add(Dense(128, activation='relu'))
    model.add(Reshape((4, 4, 8), input_shape=(128,)))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same', input_shape=(4,4,8)))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
    return model

def autoencoder_model(encoder, decoder):
    model = Sequential()
    model.add(encoder)
    model.add(decoder)
    return model

# Create models for encoder, decoder and combined autoencoder
e = encoder_model()
d = decoder_model()
print(e.output_shape, e.count_params(), d.input_shape, d.count_params())
autoencoder = autoencoder_model(e, d)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# To train it, we will use the original MNIST digits with shape (samples, 3, 28, 28), and we will just normal

from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

from keras.callbacks import TensorBoard

#tensorboard --logdir=/tmp/autoencoder

autoencoder.fit(x_train, x_train,
                epochs=1,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test)#,
                #callbacks=[TensorBoard(log_dir='/tmp/autoencoder')]
                )

# Save encoder and decoder models
'''e.save_weights('encoder.h5', True)
d.save_weights('decoder.h5', True)
autoencoder.save_weights('autoencoder.h5', True)'''

recon_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(recon_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

