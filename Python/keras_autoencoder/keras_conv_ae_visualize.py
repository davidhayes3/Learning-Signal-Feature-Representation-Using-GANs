import numpy as np
from keras import backend as K
from keras.datasets import mnist
from keras_conv_ae_models import encoder_model, decoder_model
import matplotlib.pyplot as plt

# Load saved models for encoder and decoder
e = encoder_model()
e.load_weights('encoder.h5')

d = decoder_model()
d.load_weights('decoder.h5')

# Get weights from first layer of encoder
weights0 = e.layers[0].get_weights()[0] # get weights
#weights0 = e.layers[0].get_weights()[1] # get biases
weights0 = np.array(weights0).transpose() # transpose into suitable shape for visualizing

# Load and format data

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format


# Define function for mapping of first layer of encoder
layer_1_out = K.function([e.layers[0].input, K.learning_phase()],[e.layers[0].output])

# Get activation maps for first 10 images in test set


x_test = x_test[0:9,:,:,:]
recon_test = d.predict(e.predict(x_test))

img_num = 1 # choose image number
layer_1_activations = layer_1_out([x_test, 1])[0]
layer_1_activations = layer_1_activations[img_num].transpose() # transpose into shape suitable for plotting


# Visualization

# Display digit from test set along with the encoders filters and the activation map of this filter for each image
n = len(weights0)

plt.figure(figsize=(20,20))

# Plot test digit
ax = plt.subplot(3, n, 1)
plt.imshow(x_test[img_num].reshape(28,28))
plt.gray()
plt.title('Test Set Digit')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

# Plot reconstructed test digit
ax = plt.subplot(3, n, n)
plt.imshow(recon_test[img_num].reshape(28,28))
plt.gray()
plt.title('Reconstructed Digit')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

for i in range(n):
    # Display layer 1 features
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(weights0[i].reshape(3, 3))
    plt.gray()
    plt.title('Filter ' + str(i+1))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display layer 1 activation map
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(layer_1_activations[i].transpose())
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

# Give title to third row of images
plt.subplot(3, n, 1 + 1 + 2*n)
plt.title('Activation map of each filter')

plt.show()
