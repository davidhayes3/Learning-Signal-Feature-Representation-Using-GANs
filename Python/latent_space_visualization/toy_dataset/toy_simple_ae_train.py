import keras.utils
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, merge, LeakyReLU, Dropout, concatenate, BatchNormalization
from matplotlib.pyplot import cm

np.random.seed(1337) # for reproducibility

# Settings
latent_dim = 2
img_dim = 4
num_classes = 16


def save_latent_vis(encoder):
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
    plt.savefig('Images/toydset_basic_ae_latent.png')



# Load dataset
x_train = np.loadtxt('Dataset/toy_dataset_x_train.txt', dtype=np.float32)
x_test = np.loadtxt('Dataset/toy_dataset_x_test.txt', dtype=np.float32)
y_train = np.loadtxt('Dataset/toy_dataset_y_train.txt', dtype=np.int)
y_test = np.loadtxt('Dataset/toy_dataset_y_test.txt', dtype=np.int)

#print(x_test[0:100])

# Define models

def encoder_model():
    model = Sequential()

    model.add(Dense(512, input_dim=img_dim))
    model.add(LeakyReLU(alpha=0.2))
    #model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(latent_dim))

    return model


def generator_model():
    model = Sequential()

    model.add(Dense(512, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    #model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(img_dim))
    model.add(Activation('sigmoid'))

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
#print(encoder.count_params(), generator.count_params(), autoencoder.count_params())


# Specify loss function and optimizer for autoencoder
#autoencoder.compile(optimizer='adam', loss='mse',  metrics=['accuracy'])
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy',  metrics=['accuracy'])

callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')]

history = autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=100,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=callbacks,
                verbose=1
            )



save_latent_vis(encoder)


# Save encoder and decoder models
encoder.save_weights('Models/toydset_basic_ae_encoder.h5', True)


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



# Summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Training vs Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', ' Validation'], loc='lower right')
plt.show()

# Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training vs Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()


