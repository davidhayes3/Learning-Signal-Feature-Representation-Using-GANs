import keras.utils
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, merge, LeakyReLU, Dropout, concatenate, BatchNormalization, Flatten, Reshape
from keras.optimizers import Adam
from matplotlib.pyplot import cm

np.random.seed(1337) # for reproducibility

# Define constants
img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
latent_dim = 100


## Load the dataset

(x_train, _), (x_test, y_test) = mnist.load_data()

# Rescale -1 to 1
x_train = (x_train.astype(np.float32) - 127.5) / 127.5
x_train = np.expand_dims(x_train, axis=3)

x_test = (x_test.astype(np.float32) - 127.5) / 127.5
x_test = np.expand_dims(x_test, axis=3)

z_train = np.random.normal(size=(x_train.shape[0], latent_dim))
z_test = np.random.normal(size=(x_test.shape[0], latent_dim))



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
    model.add(Activation('tanh'))
    model.add(Reshape(img_shape))

    return model


def latent_reconstructor_model(d, e):
    model = Sequential()

    model.add(d)
    model.add(e)

    return model

optimizer = Adam(0.0002, 0.5)


# Create models for encoder, decoder and combined autoencoder
encoder = encoder_model()

generator = generator_model()
generator.load_weights('Models/mnist_gan_generator.h5')
generator.trainable = False

latent_regressor = latent_reconstructor_model(generator, encoder)



# Specify loss function and optimizer for autoencoder
latent_regressor.compile(optimizer='SGD', loss='mse',  metrics=['accuracy'])

callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')]

history = latent_regressor.fit(z_train, z_train,
                epochs=100,
                batch_size=100,
                shuffle=True,
                validation_data=(z_test, z_test),
                callbacks=callbacks,
                verbose=1
            )


# Save encoder and decoder models
encoder.save_weights('Models/mnist_latent_regressor_encoder.h5', True)
