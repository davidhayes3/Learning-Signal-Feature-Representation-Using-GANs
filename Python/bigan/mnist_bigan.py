from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K

import matplotlib.pyplot as plt

import numpy as np


img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
latent_dim = 100


def save_imgs(gen_imgs, epoch):
    r, c = 5, 5

    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig("mnist_bigan_%d.png" % epoch)
    plt.close()


def build_encoder():
    model = Sequential()

    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(latent_dim))

    model.summary()

    img = Input(shape=img_shape)
    z = model(img)

    return Model(img, z)

def build_generator():
    model = Sequential()

    model.add(Dense(512, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))

    model.summary()

    z = Input(shape=(latent_dim,))
    gen_img = model(z)

    return Model(z, gen_img)

def build_discriminator():

    z = Input(shape=(latent_dim,))
    img = Input(shape=img_shape)
    d_in = concatenate([z, Flatten()(img)])

    model = Dense(1024)(d_in)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.5)(model)
    model = Dense(1024)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.5)(model)
    model = Dense(1024)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.5)(model)
    validity = Dense(1, activation="sigmoid")(model)

    return Model([z, img], validity)

# Specify optimizer for models
optimizer = Adam(0.0002, 0.5)

# Build and compile the discriminator
discriminator = build_discriminator()
discriminator.compile(loss=['binary_crossentropy'],
                           optimizer=optimizer,
                           metrics=['accuracy'])

# Build and compile the generator
generator = build_generator()
generator.compile(loss=['binary_crossentropy'],
                       optimizer=optimizer)

# Build and compile the encoder
encoder = build_encoder()
encoder.compile(loss=['binary_crossentropy'],
                     optimizer=optimizer)

# The part of the bigan that trains the discriminator and encoder
discriminator.trainable = False

# Generate image from samples noise
z = Input(shape=(latent_dim,))
img_ = generator(z)

# Encode image
img = Input(shape=img_shape)
z_ = encoder(img)

# Latent -> img is fake, and img -> latent is valid
fake = discriminator([z, img_])
valid = discriminator([z_, img])

# Set up and compile the combined model
bigan_generator = Model([z, img], [fake, valid])
bigan_generator.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
                             optimizer=optimizer)

# Train models

epochs = 40000
batch_size = 32
save_interval = 400

# Load the dataset
(X_train, _), (_, _) = mnist.load_data()

# Rescale -1 to 1
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = np.expand_dims(X_train, axis=3)

half_batch = int(batch_size / 2)

for epoch in range(epochs):

    # ---------------------
    #  Train Discriminator
    # ---------------------

    # Sample noise and generate img
    z = np.random.normal(size=(half_batch, latent_dim))
    imgs_ = generator.predict(z)

    # Select a random half batch of images and encode
    idx = np.random.randint(0, X_train.shape[0], half_batch)
    imgs = X_train[idx]
    z_ = encoder.predict(imgs)

    valid = np.ones((half_batch, 1))
    fake = np.zeros((half_batch, 1))

    # Train the discriminator (img -> z is valid, z -> img is fake)
    d_loss_real = discriminator.train_on_batch([z_, imgs], valid)
    d_loss_fake = discriminator.train_on_batch([z, imgs_], fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # ---------------------
    #  Train Generator
    # ---------------------

    # Sample gaussian noise
    z = np.random.normal(size=(batch_size, latent_dim))

    # Select a random half batch of images
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    imgs = X_train[idx]

    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    # Train the generator (z -> img is valid and img -> z is is invalid)
    g_loss = bigan_generator.train_on_batch([z, imgs], [valid, fake])

    # Plot the progress
    print("%d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss[0]))

    # If at save interval => save generated image samples
    if epoch % save_interval == 0:
        # Select a random half batch of images
        z = np.random.normal(size=(25, latent_dim))
        gen_imgs = generator.predict(z)
        save_imgs(gen_imgs, epoch)