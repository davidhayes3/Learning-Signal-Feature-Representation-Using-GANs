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


img_dim = 4
latent_dim = 2
num_classes = 16


def save_latent_vis(encoder, epoch):
    z = encoder.predict(x_train)

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
    plt.savefig('Images/toydset_aae_latent_%d.png' % (epoch+1))


# Load dataset
x_train = np.loadtxt('Dataset/toy_dataset_x_train.txt', dtype=np.float32)
x_test = np.loadtxt('Dataset/toy_dataset_x_test.txt', dtype=np.float32)
y_train = np.loadtxt('Dataset/toy_dataset_y_train.txt', dtype=np.int)
y_test = np.loadtxt('Dataset/toy_dataset_y_test.txt', dtype=np.int)



def encoder_model():
    model = Sequential()

    model.add(Dense(512, input_dim=img_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512, input_dim=img_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(latent_dim))

    return model


def generator_model():
    model = Sequential()

    model.add(Dense(512, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512, input_dim=img_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(img_dim))
    model.add(Activation('sigmoid'))

    return model

def discriminator_model():
    model = Sequential()

    model.add(Dense(512, input_shape=(latent_dim,)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


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


x = Input(shape=(img_dim,))

enc_x = encoder(x)
recon_x = decoder(enc_x)

discriminator.trainable = False

validity = discriminator(enc_x)

aae = Model(x, [recon_x, validity])


aae.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
    loss_weights=[0.5, 0.5],
    optimizer=optimizer)

''''# Compute AAE loss
recon_loss = metrics.binary_crossentropy(x, recon_x)
adversarial_loss = metrics.binary_crossentropy(validity, 1)
aae_loss = K.mean(recon_loss, adversarial_loss)
#aae.add_loss(aae_loss)

aae.compile(optimizer=optimizer, loss=aae_loss)'''



## Train models

# Training hyperparameters
epochs = 20
batch_size = 200
epoch_save_interval = 1
num_batches = int(x_train.shape[0] / batch_size)
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
        imgs = x_train[batch * batch_size: (batch + 1) * batch_size]
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

        '''# Select a random half batch of images
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        imgs = x_train[idx]'''

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
        encoder.save_weights('Models/aae_encoder.h5')
        save_latent_vis(encoder, epoch)


## Visualize Reconstruction

# Get initial data examples to train on
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
x_test = x_test[test_digit_indices]
recon_test = decoder.predict(encoder.predict(x_test))

n = len(x_test)

x_test = 0.5 * x_test + 0.5
recon_test = 0.5 * recon_test + 0.5

# Plot test digits
for i in range(n):
    ax = plt.subplot(2 * num_classes, n / num_classes, i + 1)
    plt.imshow(x_test[i].reshape(2, 2))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

# Plot reconstructed test digits
for i in range(n):
    ax = plt.subplot(2 * num_classes, n / num_classes, i + 1 + n)
    plt.imshow(recon_test[i].reshape(2, 2))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()



## Plot loss curves

# Plot batch loss curves for g and d
plt.figure(1)
batch_numbers = np.arange((epochs * num_batches)) + 1
plt.plot(batch_numbers, d_batch_loss_trajectory, 'b-', batch_numbers, g_batch_loss_trajectory, 'r-')
plt.legend(['Discriminator', 'Generator/Encoder'], loc='upper right')
plt.xlabel('Batch Number')
plt.ylabel('Loss')
plt.show()


# Plot epoch loss curves for g and d
plt.figure(2)
epoch_numbers = np.arange(epochs) + 1
plt.plot(epoch_numbers, d_epoch_loss_trajectory, 'b-', epoch_numbers, g_epoch_loss_trajectory, 'r-')
plt.legend(['Discriminator', 'Generator/Encoder'], loc='upper left')
plt.xlabel('Epoch Number')
plt.ylabel('Average Minibatch Loss')
plt.savefig('mnist_bigan_valloss_%d_epochs_%d_bs.png' % (epochs, batch_size))
plt.show()

