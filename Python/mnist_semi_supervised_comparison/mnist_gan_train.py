import numpy as np
import keras
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, merge, LeakyReLU, Dropout, concatenate, BatchNormalization, Flatten, Reshape
import matplotlib.pyplot as plt
from keras.layers import Input
from keras.models import Model
from keras.datasets import mnist
from matplotlib.pyplot import cm


np.random.seed(1330)

# Define constants
img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
latent_dim = 100


# Function to save images
def save_imgs(gen_imgs, epoch):
    r, c = 5, 5

    # Rescale to
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    count = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[count].reshape(2, 2), cmap='gray')
            axs[i, j].axis('off')
            count += 1
    fig.savefig('Images/mnist_gan_%d.png' % (epoch+1))
    plt.close()


## Load the dataset

(x_train, _), (x_test, y_test) = mnist.load_data()

# Rescale -1 to 1
x_train = (x_train.astype(np.float32) - 127.5) / 127.5
x_train = np.expand_dims(x_train, axis=3)

x_test = (x_test.astype(np.float32) - 127.5) / 127.5
x_test = np.expand_dims(x_test, axis=3)


## Define models

def generator_model():
    model = Sequential()

    model.add(Dense(512, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(img_shape)))
    model.add(Activation('tanh'))
    model.add(Reshape(img_shape))

    return model


def discriminator_model():
    img = Input(shape=img_shape)

    model = Flatten()(img)
    model = Dense(1024)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.5)(model)
    model = Dense(1024)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.5)(model)
    model = Dense(1024)(model)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.5)(model)
    validity = Dense(1, activation='sigmoid')(model)

    return Model(img, validity)


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


# The part of the bigan that trains the discriminator and encoder
discriminator.trainable = False

# Generate image from samples noise
z = Input(shape=(latent_dim,))
gen_img = generator(z)
validity = discriminator(gen_img)

# Set up and compile the combined model
gan_generator = Model(z, validity)
gan_generator.compile(loss='binary_crossentropy', optimizer=optimizer)


## Train models

# Training hyperparameters
epochs = 20
batch_size = 100
epoch_save_interval = 1
num_batches = int(x_train.shape[0] / batch_size)
half_batch = int(batch_size / 2)

# Define arrays to hold progression of discriminator and bigan losses
d_batch_loss_trajectory = np.zeros(epochs * num_batches)
g_batch_loss_trajectory = np.zeros(epochs * num_batches)
d_epoch_loss_trajectory = np.zeros(epochs)
g_epoch_loss_trajectory = np.zeros(epochs)

# Train for set number of epochs
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

        # Sample noise and generate img
        z = np.random.normal(size=(batch_size, latent_dim))
        imgs_ = generator.predict(z)

        # Create labels for discriminator inputs
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # Train the discriminator (img -> z is valid, z -> img is fake)
        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(imgs_, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


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

        # Train the generator (z -> img is valid and img -> z is is invalid)
        g_loss = gan_generator.train_on_batch(z, valid)

        g_batch_loss_trajectory[epoch * num_batches + batch] = g_loss
        g_epoch_loss_sum += g_loss

        # Print progress
        print("[Epoch: %d, Batch: %d / %d] [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch+1, batch, num_batches,
            d_loss[0], 100 * d_loss[1], g_loss))


    # Get epoch loss data
    d_epoch_loss_trajectory[epoch] = d_epoch_loss_sum / num_batches
    g_epoch_loss_trajectory[epoch] = g_epoch_loss_sum / num_batches

    # If at save interval, save generated image samples
    if epoch % epoch_save_interval == 0:
        #z = np.random.normal(size=(25, latent_dim))
        #gen_imgs = generator.predict(z)
        #save_imgs(gen_imgs, epoch)
        generator.save_weights('Models/mnist_gan_generator.h5')


# Save learned generator
generator.save_weights('Models/mnist_gan_generator.h5')

