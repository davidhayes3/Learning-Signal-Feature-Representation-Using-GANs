from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K
from matplotlib.pyplot import cm
from keras import metrics

import matplotlib.pyplot as plt

import numpy as np


# Define constants
img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
latent_dim = 100


# Load dataset

(X_train, _), (X_test, y_test) = mnist.load_data()

# Rescale -1 to 1
X_train = (X_train.astype(np.float32)) / 255.
X_train = np.expand_dims(X_train, axis=3)

X_test = (X_test.astype(np.float32)) / 255.
X_test = np.expand_dims(X_test, axis=3)


# Define models

def encoder_model():
    model = Sequential()

    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(latent_dim))

    return model


def generator_model():
    model = Sequential()

    model.add(Dense(512, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(img_shape)))
    model.add(Activation('sigmoid'))
    model.add(Reshape(img_shape))

    return model


def discriminator_model():
    z = Input(shape=(latent_dim,))

    model = Dense(1024)(z)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.5)(model)
    model = Dense(1024)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.5)(model)
    model = Dense(1024)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.5)(model)
    validity = Dense(1, activation='sigmoid')(model)

    return Model(z, validity)


optimizer = Adam(0.0002, 0.5)

# Build and compile the discriminator
discriminator = discriminator_model()
discriminator.compile(loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy'])

# Build and compile the encoder / decoder
encoder = encoder_model()
encoder.compile(loss=['binary_crossentropy'],
    optimizer=optimizer)

decoder = generator_model()
decoder.compile(loss=['mse'],
    optimizer=optimizer)


x = Input(shape=img_shape)

enc_x = encoder(x)
recon_x = decoder(enc_x)

discriminator.trainable = False

validity = discriminator(enc_x)

aae = Model(x, [recon_x, validity])


aae.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
    loss_weights=[0.5, 0.5],
    optimizer=optimizer)



# Train models

# Training hyperparameters
epochs = 20
batch_size = 100
epoch_save_interval = 1
num_batches = int(X_train.shape[0] / batch_size)
half_batch = int(batch_size / 2)


# Define arrays to hold progression of discriminator and bigan losses
d_batch_loss_trajectory = np.zeros(epochs * num_batches)
g_batch_loss_trajectory = np.zeros(epochs * num_batches)
d_epoch_loss_trajectory = np.zeros(epochs)
g_epoch_loss_trajectory = np.zeros(epochs)


for epoch in range(epochs):

    # Print current epoch number
    print("\nEpoch: " + str(epoch + 1) + "/" + str(epochs))

    # Set epoch losses to zero
    d_epoch_loss_sum = 0
    g_epoch_loss_sum = 0

    # Train on all batches
    for batch in range(num_batches):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select next batch of images from training set and encode
        imgs = X_train[batch * batch_size: (batch + 1) * batch_size]
        latent_fake = encoder.predict(imgs)

        latent_real = np.random.normal(size=(batch_size, latent_dim))

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(latent_real, valid)
        d_loss_fake = discriminator.train_on_batch(latent_fake, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        ## Record discriminator batch loss details
        d_batch_loss_trajectory[epoch * num_batches + batch] = d_loss[0]
        d_epoch_loss_sum += d_loss[0]


        # ---------------------
        #  Train Generator
        # ---------------------

        # Generator wants the discriminator to label the generated representations as valid
        valid_y = np.ones((batch_size, 1))

        # Train the generator
        g_loss = aae.train_on_batch(imgs, [imgs, valid_y])

        g_batch_loss_trajectory[epoch * num_batches + batch] = g_loss[0]
        g_epoch_loss_sum += g_loss[0]

        # Print progress
        print("[Epoch: %d, Batch: %d / %d] [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch + 1, batch, num_batches,
                                                                                      d_loss[0], 100 * d_loss[1],
                                                                                      g_loss[0]))

    # Get epoch loss data
    d_epoch_loss_trajectory[epoch] = d_epoch_loss_sum / num_batches
    g_epoch_loss_trajectory[epoch] = g_epoch_loss_sum / num_batches

    # If at save interval => save generated image samples
    if epoch % epoch_save_interval == 0:
        encoder.save_weights('Models/mnist_aae_encoder.h5')


# Save encoder weights
encoder.save_weights('Models/mnist_aae_encoder.h5')