import keras.utils
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, merge, LeakyReLU, Dropout, concatenate, Flatten, Reshape
from matplotlib.pyplot import cm

np.random.seed(1337) # for reproducibility


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
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(latent_dim))

    return model


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

def autoencoder_model(e, d):
    model = Sequential()

    model.add(e)
    model.add(d)

    return model


# Create models for encoder, decoder and combined autoencoder
encoder = encoder_model()
generator = generator_model()
autoencoder = autoencoder_model(encoder, generator)


# Specify loss function and optimizer for autoencoder
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy',  metrics=['accuracy'])

callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')]

history = autoencoder.fit(X_train, X_train,
                epochs=100,
                batch_size=100,
                shuffle=True,
                validation_data=(X_test, X_test),
                callbacks=callbacks,
                verbose=1
            )


# Save encoder and decoder models
encoder.save_weights('Models/mnist_basic_ae_encoder.h5', True)