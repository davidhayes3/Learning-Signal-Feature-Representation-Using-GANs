import numpy as np
import keras
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, merge, LeakyReLU, Dropout, concatenate, BatchNormalization
import matplotlib.pyplot as plt
from keras.layers import Input
from keras.models import Model
from matplotlib.pyplot import cm


np.random.seed(1330)

# Settings
latent_dim = 2
img_dim = 4
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
    plt.savefig('Images/toydset_bigan_latent_%d.png' % (epoch+1))


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
    fig.savefig('Images/toydset_bigan_%d.png' % (epoch+1))
    plt.close()


# Load dataset
x_train = np.loadtxt('Dataset/toy_dataset_x_train.txt', dtype=np.float32)
x_test = np.loadtxt('Dataset/toy_dataset_x_test.txt', dtype=np.float32)
y_train = np.loadtxt('Dataset/toy_dataset_y_train.txt', dtype=np.int)
y_test = np.loadtxt('Dataset/toy_dataset_y_test.txt', dtype=np.int)

x_train = (x_train - 0.5) / 0.5
x_test = (x_test - 0.5) / 0.5


# Define models

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
    model.add(Activation('tanh'))

    return model

def discriminator_model():
    z_in = Input(shape=(2,))
    z = Dense(512)(z_in)
    z = LeakyReLU(alpha=0.2)(z)
    z = Dropout(0.5)(z)
    z = Dense(512)(z)
    z = LeakyReLU(alpha=0.2)(z)

    x_in = Input(shape=(4,))
    x = Dense(512)(x_in)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.5)(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)

    c = concatenate([z, x])
    c = Dropout(0.5)(c)
    c = Dense(1024)(c)
    c = LeakyReLU(alpha=0.2)(c)
    c = Dropout(0.5)(c)
    c = Dense(1)(c)
    validity = Activation('sigmoid')(c)

    return Model([z_in, x_in], validity)


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
img = Input(shape=(img_dim,))
z_ = encoder(img)

# Latent -> img is fake, and img -> latent is valid
fake = discriminator([z, img_])
valid = discriminator([z_, img])

# Set up and compile the combined model
bigan_generator = Model([z, img], [fake, valid])
bigan_generator.compile(loss=['binary_crossentropy', 'binary_crossentropy'],
                             optimizer=optimizer)


# Train models

# Training hyperparameters
epochs = 50
batch_size = 200
epoch_save_interval = 1
num_batches = int(x_train.shape[0] / batch_size)
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
        imgs = x_train[batch * batch_size: (batch + 1) * batch_size]
        z_ = encoder.predict(imgs)

        # Train d on full batch

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

        # Record discriminator batch loss details
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
        print("[Epoch: %d, Batch: %d / %d] [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch+1, batch, num_batches,
            d_loss[0], 100 * d_loss[1], ge_loss[0]))


    # Get epoch loss data
    d_epoch_loss_trajectory[epoch] = d_epoch_loss_sum / num_batches
    ge_epoch_loss_trajectory[epoch] = ge_epoch_loss_sum / num_batches

    # If at save interval, save generated image samples
    if epoch % epoch_save_interval == 0:
        z = np.random.normal(size=(25, latent_dim))
        gen_imgs = generator.predict(z)
        save_imgs(gen_imgs, epoch)
        encoder.save_weights('Models/bigan_encoder.h5')
        generator.save_weights('Models/bigan_generator.h5')
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
recon_test = generator.predict(encoder.predict(x_test))

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
plt.plot(batch_numbers, d_batch_loss_trajectory, 'b-', batch_numbers, ge_batch_loss_trajectory, 'r-')
plt.legend(['Discriminator', 'Generator/Encoder'], loc='upper right')
plt.xlabel('Batch Number')
plt.ylabel('Loss')
plt.show()


# Plot epoch loss curves for g and d
plt.figure(2)
epoch_numbers = np.arange(epochs) + 1
plt.plot(epoch_numbers, d_epoch_loss_trajectory, 'b-', epoch_numbers, ge_epoch_loss_trajectory, 'r-')
plt.legend(['Discriminator', 'Generator/Encoder'], loc='upper left')
plt.xlabel('Epoch Number')
plt.ylabel('Average Minibatch Loss')
plt.savefig('mnist_bigan_valloss_%d_epochs_%d_bs.png' % (epochs, batch_size))
plt.show()