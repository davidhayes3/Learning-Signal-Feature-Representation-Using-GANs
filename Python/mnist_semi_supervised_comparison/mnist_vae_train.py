from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, LeakyReLU
from keras.layers import Conv2D, Conv2DTranspose, Activation, BatchNormalization
from keras.models import Model, Sequential
from keras import backend as K
from keras import metrics
from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from matplotlib.pyplot import cm


# Define constants
img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
latent_dim = 100
epsilon_std = 1.0


# Load dataset

(X_train, _), (X_test, y_test) = mnist.load_data()

# Rescale -1 to 1
X_train = (X_train.astype(np.float32)) / 255.
X_train = np.expand_dims(X_train, axis=3)

X_test = (X_test.astype(np.float32)) / 255.
X_test = np.expand_dims(X_test, axis=3)



def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=epsilon_std)

    return z_mean + K.exp(z_log_var / 2) * epsilon


def encoder_model():
    x = Input(shape=img_shape)

    x_enc = Flatten()(x)
    x_enc = Dense(512)(x_enc)
    x_enc = LeakyReLU(alpha=0.2)(x_enc)
    x_enc = Dense(512)(x_enc)
    x_enc = LeakyReLU(alpha=0.2)(x_enc)

    z_mean = Dense(latent_dim)(x_enc)
    z_log_var = Dense(latent_dim)(x_enc)

    return Model(x, [z_mean, z_log_var])


# Define models


def generator_model():
    model = Sequential()

    model.add(Dense(512, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(np.prod(img_shape)))
    model.add(Activation('sigmoid'))
    model.add(Reshape(img_shape))

    return model


encoder = encoder_model()
generator = generator_model()


x = Input(shape=img_shape)

z_mean, z_log_var = encoder(x)

z = Lambda(sampling)([z_mean, z_log_var])

recon_x = generator(z)

# instantiate VAE model
vae = Model(x, recon_x)


# Compute VAE loss
xent_loss = K.sum(metrics.binary_crossentropy(x, recon_x), axis=1)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')


callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')]


vae.fit(X_train,
        shuffle=True,
        epochs=100,
        batch_size=100,
        callbacks=callbacks,
        validation_data=(X_test, None))


# Save encoder weights
encoder.save_weights('Models/mnist_vae_encoder.h5', True)