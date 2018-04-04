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


img_dim = 4
latent_dim = 2
num_classes = 16
epsilon_std = 1.0



def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=epsilon_std)

    return z_mean + K.exp(z_log_var / 2) * epsilon



def save_latent_vis(enc):
    z = enc.predict(x_train, batch_size=100)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = cm.Spectral(np.linspace(0, 1, num_classes))

    xx = z[:,0]
    yy = z[:,1]

    labels = range(num_classes)

    # plot the 2D data points
    for i in range(num_classes):
        ax.scatter(xx[y_train == i], yy[y_train == i], color=colors[i], label=labels[i], s=5)

    plt.axis('tight')
    plt.savefig('Images/toydset_vae_latent_1.png')




# Load dataset
x_train = np.loadtxt('Dataset/toy_dataset_x_train.txt', dtype=np.float32)
x_test = np.loadtxt('Dataset/toy_dataset_x_test.txt', dtype=np.float32)
y_train = np.loadtxt('Dataset/toy_dataset_y_train.txt', dtype=np.int)
y_test = np.loadtxt('Dataset/toy_dataset_y_test.txt', dtype=np.int)


def encoder_model():
    x = Input(shape=(img_dim,))

    x_enc = Dense(512)(x)
    x_enc = LeakyReLU(alpha=0.2)(x_enc)
    x_enc = BatchNormalization(momentum=0.8)(x_enc)
    x_enc = Dense(512)(x_enc)
    x_enc = LeakyReLU(alpha=0.2)(x_enc)
    x_enc = BatchNormalization(momentum=0.8)(x_enc)

    z_mean = Dense(latent_dim)(x_enc)
    z_log_var = Dense(latent_dim)(x_enc)

    return Model(x, [z_mean, z_log_var])


def generator_model():
    model = Sequential()

    model.add(Dense(512, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(img_dim))
    model.add(Activation('sigmoid'))

    return model


encoder = encoder_model()
generator = generator_model()


x = Input(shape=(img_dim,))

z_mean, z_log_var = encoder(x)

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

recon_x = generator(z)

# instantiate VAE model
vae = Model(x, recon_x)


# Compute VAE loss
recon_loss = img_dim * metrics.binary_crossentropy(x, recon_x)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(recon_loss + kl_loss)
vae.add_loss(vae_loss)

vae.compile(optimizer='rmsprop')


callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')]


vae.fit(x_train,
        shuffle=True,
        epochs=50,
        batch_size=100,
        callbacks=callbacks,
        validation_data=(x_test, None))

new_encoder = Model(x, z_mean)


save_latent_vis(new_encoder)
encoder.save_weights('Models/toydset_vae_encoder.h5', True)