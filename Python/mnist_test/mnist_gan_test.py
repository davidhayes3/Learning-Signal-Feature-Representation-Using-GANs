from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np
from PIL import Image
import argparse
import math
import matplotlib.pyplot as plt
import coremltools
import os
import h5py


# Define generator model, takes latent vector of dimension 100 and outputs image of size 28 x 28 x1
def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, units=1024))
    print(model.count_params())
    model.add(Activation('tanh'))
    print(model.count_params())
    model.add(Dense(128 * 7 * 7))
    print(model.count_params())
    model.add(BatchNormalization())
    print(model.count_params())
    model.add(Activation('tanh'))
    print(model.count_params())
    model.add(Reshape((7, 7, 128), input_shape=(128 * 7 * 7,)))
    print(model.count_params())
    model.add(UpSampling2D(size=(2, 2)))
    print(model.count_params())
    model.add(Conv2D(64, (5, 5), padding='same'))
    print(model.count_params())
    model.add(Activation('tanh'))
    print(model.count_params())
    model.add(UpSampling2D(size=(2, 2)))
    print(model.count_params())
    model.add(Conv2D(1, (5, 5), padding='same'))
    print(model.count_params())
    model.add(Activation('tanh'))
    print(model.count_params())
    return model

# Define discriminator model, takes input image of dimension 28 x 28 x 1 and outputs a single value,
# indicating likelihood of input image being from training set
def discriminator_model():
    model = Sequential()
    model.add(Conv2D(64, (5, 5), padding='same', input_shape=(28, 28, 1)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

# Define model with generator and discriminator together, this is how generator is trained
def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    # Parameters of d cannot be altered while g is being trained
    d.trainable = False
    model.add(d)
    return model


# Combines batch of generated images into one image
def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height * shape[0], width * shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = \
            img[:, :, 0]
    return image

# Train GAN
def train(batch_size, epochs, digit=None, plot_loss=False):

    # Load MNIST data and split into train and test set
    (X_train, y_train), (X_test , _) = mnist.load_data()

    # If want to train on a certain digit type
    if digit is not None:
        # Find indices of digit in training set and reduce training set
        indices = [i for i, y in enumerate(y_train) if y == digit]
        X_train = X_train[indices]

    # Convert training set to float32 and normalize to range [-1,1]
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5

    # Something to do with RGB dimension
    X_train = X_train[:, :, :, None]

    # Define number of batches per epoch
    num_batches = int(X_train.shape[0] / batch_size)

    # Create models for generator, discriminator and combined g and d
    d = discriminator_model()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d)

    # Define arrays to hold progression of g and d loss
    d_loss_trajectory = np.zeros((1, epochs * num_batches))
    g_loss_trajectory = np.zeros((1, epochs * num_batches))

    # Define optimizer for generator and discriminator
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)

    # Compile loss functions and optimizers for g,d and d_on_g
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.compile(loss='binary_crossentropy', optimizer=d_optim)
    d.trainable = True

    # Train over 100 epochs
    for epoch in range(epochs):

        # Print epoch number and number of batches in epoch
        print("Epoch: %d/%d" % (epoch, epochs))
        print("Number of batches: %d" % num_batches)

        # Train on each batch in epoch
        for batch_number in range(num_batches):

            print("Batch number: %d/%d" % (batch_number, num_batches))

            # Create latent vector for generator
            noise = np.random.uniform(-1, 1, size=(batch_size, 100))
            # Select current batch from training images
            image_batch = X_train[batch_number * batch_size:(batch_number + 1) * batch_size]
            # Generate images from latent vector
            generated_images = g.predict(noise, verbose=0)

            # Every 20 batches create an image of the generated images
            if batch_number % 20 == 0:
                image = combine_images(generated_images)
                image = image * 127.5 + 127.5
                Image.fromarray(image.astype(np.uint8)).save(
                    str(epoch) + "_" + str(batch_number) + ".png")

            # Concatenate images from training set and generator images
            X = np.concatenate((image_batch, generated_images))

            # Label X for discriminator
            y = [1] * batch_size + [0] * batch_size

            # Train d on batch of real and fake images and report d's loss
            d_loss = d.train_on_batch(X, y)
            print("batch %d d_loss : %f" % (batch_number, d_loss))
            d_loss_trajectory[0, epoch * batch_size + batch_number] = d_loss

            # Generate different latent vectors to the ones d was just trained on
            noise = np.random.uniform(-1, 1, (batch_size, 100))

            # Hold d constant and train g based on d's output and report g's loss
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, [1] * batch_size) # Want d to output 1 for generator input
            print("batch %d g_loss : %f" % (batch_number, g_loss))
            g_loss_trajectory[0, epoch * batch_size + batch_number] = g_loss

            # Set d to be trainable again for next batch
            d.trainable = True

            # Every 10 batches, update the saved weights of g and d
            if batch_number % 10 == 9:
                g.save_weights('generator.h5', True)
                d.save_weights('discriminator.h5', True)


    if plot_loss == True:
        batch_numbers = np.arange(1, (epochs * num_batches) + 1)
        plt.figure(1)
        plt.plot(batch_numbers, d_loss_trajectory, 'r-', batch_numbers, d_loss_trajectory, 'b-')
        plt.legend(['Discriminator', 'Generator'], loc='upper right');
        plt.text(2, 2, r'$Batch Size = %d$', num_batches)
        plt.xlabel('Batch Number')
        plt.ylabel('Loss')
        plt.show()


# Generate
def generate(batch_size, nice=False):

    # Load generator
    g = generator_model()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('generator.h5')

    if nice:
        # Load discriminator
        d = discriminator_model()
        d.compile(loss='binary_crossentropy', optimizer="SGD")
        d.load_weights('discriminator.h5')

        # Generate latent vector and use g to produce new images
        noise = np.random.uniform(-1, 1, (batch_size * 20, 100))
        generated_images = g.predict(noise, verbose=1)

        # Predict likelihood of generated images being from training set
        d_pred = d.predict(generated_images, verbose=1)

        # Create index of all numbers from 0 to 20 * batch_size and reshape to be column vector
        index = np.arange(0, batch_size * 20)
        index.resize((batch_size * 20, 1))

        # Append d_pred and index and reverse
        pre_with_index = list(np.append(d_pred, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)

        # Create nice_images full of zeros
        nice_images = np.zeros((batch_size,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]

        # Gets last batch_size images from generated_images and combine
        for i in range(batch_size):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]

        image = combine_images(nice_images)

    else:
        # Generate batch_size number of images and combine to one image
        noise = np.random.uniform(-1, 1, (batch_size, 100))
        generated_images = g.predict(noise, verbose=1)
        image = combine_images(generated_images)

    # Convert back to pixel values and save
    image = image * 127.5 + 127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")


#batch_size = 128
#epochs = 100
#train(batch_size, epochs, plot_loss=True)

g = generator_model()

