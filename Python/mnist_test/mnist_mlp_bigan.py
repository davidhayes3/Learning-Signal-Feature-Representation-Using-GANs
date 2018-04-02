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

# Define constants
img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
latent_dim = 100


## Load the dataset

(X_train, _), (X_test, y_test) = mnist.load_data()

# Rescale -1 to 1
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = np.expand_dims(X_train, axis=3)

X_test = (X_test.astype(np.float32) - 127.5) / 127.5
X_test = np.expand_dims(X_test, axis=3)


# Function to save images
def save_imgs(gen_imgs, epoch):
    r, c = 5, 5

    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    count = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[count, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            count += 1
    fig.savefig("mnist_bigan_%d.png" % epoch)
    plt.close()


## Define models

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

    model.summary()

    img = Input(shape=img_shape)
    z = model(img)

    return Model(img, z)


def generator_model():
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

def discriminator_model():

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
discriminator = discriminator_model()
discriminator.compile(loss=['binary_crossentropy'],
                           optimizer=optimizer,
                           metrics=['accuracy'])

# Build and compile the generator
generator = generator_model()
generator.compile(loss=['binary_crossentropy'],
                       optimizer=optimizer)

# Build and compile the encoder
encoder = encoder_model()
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


## Train models

# Training hyperparameters
epochs = 20
batch_size = 128
epoch_save_interval = 10
num_batches = int(X_train.shape[0] / batch_size)
half_batch = int(batch_size / 2)

# Define arrays to hold progression of discriminator and bigan losses
d_batch_loss_trajectory = np.zeros(epochs * num_batches)
ge_batch_loss_trajectory = np.zeros(epochs * num_batches)
d_epoch_loss_trajectory = np.zeros(epochs)
ge_epoch_loss_trajectory = np.zeros(epochs)

# Train for set number of epochs
for epoch in range(epochs):

    # Print current epoch number
    print("\nEpoch: " + str(epoch + 1) + "/" + str(epochs))

    # Set epoch losses to zero
    d_epoch_loss_sum = 0
    ge_epoch_loss_sum = 0

    # Train on all batches
    for batch in range(num_batches):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select next batch of images from training set and encode
        imgs = X_train[batch * batch_size: (batch + 1) * batch_size]
        z_ = encoder.predict(imgs)

        ## Train d on full batch

        # Sample noise and generate img
        z = np.random.normal(size=(batch_size, latent_dim))
        imgs_ = generator.predict(z)

        # Create labels for discriminator inputs
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # Train the discriminator (img -> z is valid, z -> img is fake)
        d_loss_real = discriminator.train_on_batch([z_, imgs], valid)
        d_loss_fake = discriminator.train_on_batch([z, imgs_], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


        ## Train d on half batch

        '''# Sample noise and generate img
        z = np.random.normal(size=(half_batch, latent_dim))
        imgs_ = generator.predict(z)

        # Select a random half of image batch and encode
                # Select a random half of image batch and encode
        idx = np.random.randint(0, batch_size, half_batch)
        d_imgs = imgs[idx]
        z_ = encoder.predict(d_imgs)

        # Create labels for discriminator inputs
        valid = np.ones((half_batch, 1))
        fake = np.zeros((half_batch, 1))

        # Train the discriminator (img -> z is valid, z -> img is fake)
        d_loss_real = discriminator.train_on_batch([z_, d_imgs], valid)
        d_loss_fake = discriminator.train_on_batch([z, imgs_], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)'''


        ## Record discriminator batch loss details
        d_batch_loss_trajectory[epoch * num_batches + batch] = d_loss[0]
        d_epoch_loss_sum += d_loss[0]


        # ----------------------------
        #  Train Generator and Encoder
        # ----------------------------

        # Sample gaussian noise
        z = np.random.normal(size=(batch_size, latent_dim))

        # Set labels for generator/encoder training
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # Train the generator (z -> img is valid and img -> z is is invalid)
        ge_loss = bigan_generator.train_on_batch([z, imgs], [valid, fake])

        ge_batch_loss_trajectory[epoch * num_batches + batch] = ge_loss[0]
        ge_epoch_loss_sum += ge_loss[0]

        # Print progress
        print("[Epoch: %d, Batch: %d / %d] [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, batch, num_batches,
            d_loss[0], 100 * d_loss[1], ge_loss[0]))


    # Get epoch loss data
        d_epoch_loss_trajectory[epoch] = d_epoch_loss_sum / num_batches
        ge_epoch_loss_trajectory[epoch] = ge_epoch_loss_sum / num_batches

        # If at save interval, save generated image samples
        if epoch % epoch_save_interval == 0:
            z = np.random.normal(size=(25, latent_dim))
            gen_imgs = generator.predict(z)
            save_imgs(gen_imgs, epoch)


## Visualize Reconstruction

# Get initial data examples to train on
num_classes = 10
classes = np.arange(num_classes)
test_digit_indices = np.empty(0)

# Modify training set to contain set number of labels for each class
for class_index in range(num_classes):
    # Generate training set with even class distribution over all labels
    indices = [i for i, y in enumerate(y_test) if y == classes[class_index]]
    indices = np.asarray(indices)
    indices = indices[0:10]
    test_digit_indices = np.concatenate((test_digit_indices, indices))

test_digit_indices = test_digit_indices.astype(np.int)

# Generate test and reconstucted digit arrays
X_test = X_test[test_digit_indices]
recon_test = generator.predict(encoder.predict(X_test))
n = len(X_test)

# Plot test digits
for i in range(n):
    ax = plt.subplot(2 * num_classes, n / num_classes, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
