import numpy as np
from keras_conv_ae_models import encoder_model, decoder_model, autoencoder_model
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, TensorBoard

# Create models for encoder, decoder and combined autoencoder
e = encoder_model()
d = decoder_model()
autoencoder = autoencoder_model(e, d)

# Load and format data

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

# Train model

callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0),
             TensorBoard(log_dir='/tmp/autoencoder')]
#tensorboard --logdir=/tmp/autoencoder

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                verbose=1,
                callbacks=callbacks
                )

# Save learned models
e.save_weights('mnist_encoder.h5', True)
d.save_weights('mnist_decoder.h5', True)
autoencoder.save_weights('mnist_autoencoder.h5', True)

# Apply autoencoder to test images
recon_imgs = autoencoder.predict(x_test)

# Plot subset of reconstructed test images
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
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
