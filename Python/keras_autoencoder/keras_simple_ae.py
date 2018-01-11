from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers

# Define encoder
def encoder_model():
    model = Sequential()
    model.add(Dense(128, activation='relu',
                    #activity_regularizer=regularizers.l1(10e-5), # Sparsity constraint on activity of hidden layers
                    input_shape=(784,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    return model

# Define decoder
def decoder_model():
    model = Sequential()
    model.add(Dense(64, input_dim=32, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(784, activation='sigmoid'))
    return model

# Define autoencoder
def autoencoder_model(encoder, decoder):
    model = Sequential()
    model.add(encoder)
    model.add(decoder)
    return model


# Create models for encoder, decoder and combined autoencoder
e = encoder_model()
d = decoder_model()
autoencoder = autoencoder_model(e, d)


# Import data

from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize all values between 0 and 1 and flatten the 28x28 images into vectors of size 784.

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

# Train autoencoder for 50 epochs:

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# Save encoder and decoder models
e.save_weights('encoder.h5', True)
d.save_weights('decoder.h5', True)
autoencoder.save_weights('autoencoder.h5', True)

# encode and decode some digits
# note that we take them from the *test* set
recon_imgs = autoencoder.predict(x_test)

# Use Matplotlib to plot some images
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))

for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1) # 2 rows, n cols, original in top rows
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(recon_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
