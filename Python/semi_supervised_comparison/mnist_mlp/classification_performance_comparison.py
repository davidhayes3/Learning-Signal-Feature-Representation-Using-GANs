import keras
from keras import backend as K
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses


# Set random seed for reproducibility
np.random.seed(1330)


# Define constants
img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
latent_dim = 100


# Load and preprocess data

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Data for AE models
x_train_ae = x_train.astype('float32') / 255.
x_test_ae = x_test.astype('float32') / 255.
x_train_ae = np.reshape(x_train_ae, (len(x_train), img_rows, img_cols, channels))
x_test_ae = np.reshape(x_test_ae, (len(x_test), img_rows, img_cols, channels))

x_train_gan = (x_train.astype('float32') - 127.5) / 127.5
x_test_gan = (x_test.astype('float32') - 127.5) / 127.5
x_train_gan = np.reshape(x_train_gan, (len(x_train), img_rows, img_cols, channels))
x_test_gan = np.reshape(x_test_gan, (len(x_test), img_rows, img_cols, channels))

# Label data is same for both
y_train_one_hot = keras.utils.to_categorical(y_train, 10)
y_test_one_hot = keras.utils.to_categorical(y_test, 10)


# Define models

def gan_encoder_model():
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


def ae_encoder_model():
    model = Sequential()

    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(latent_dim))

    return model


def vae_encoder_model():
    x = Input(shape=img_shape)

    x_enc = Flatten()(x)
    x_enc = Dense(512)(x_enc)
    x_enc = LeakyReLU(alpha=0.2)(x_enc)
    x_enc = Dense(512)(x_enc)
    x_enc = LeakyReLU(alpha=0.2)(x_enc)

    z_mean = Dense(latent_dim)(x_enc)
    z_log_var = Dense(latent_dim)(x_enc)

    return Model(x, [z_mean, z_log_var])



def classifier_model(encoder):
    model = Sequential()

    model.add(encoder)
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    return model


def vae_classifier_model(vae_encoder):
    x = Input(shape=img_shape)

    z, sigma = vae_encoder(x)

    model = Dense(256, activation='relu')(z)
    model = Dropout(0.5)(model)
    prediction = Dense(10, activation='softmax')(model)

    return Model(x, prediction)



# Load encoders

basic_ae = ae_encoder_model()
dae = ae_encoder_model()
aae = ae_encoder_model()
vae = vae_encoder_model()
latent_regressor = gan_encoder_model()
bigan = gan_encoder_model()
posthoc_bigan = gan_encoder_model()


encoders = ((basic_ae, 'basic_ae'), (dae, 'dae'), (aae, 'aae'), (vae, 'vae'), (latent_regressor, 'latent_regressor'),
            (bigan, 'bigan'), (posthoc_bigan, 'posthoc_bigan'))


basic_ae.load_weights('Models/mnist_basic_ae_encoder.h5')
dae.load_weights('Models/mnist_dae_encoder.h5')
aae.load_weights('Models/mnist_aae_encoder.h5')
vae.load_weights('Models/mnist_vae_encoder.h5')
latent_regressor.load_weights('Models/mnist_latent_regressor_encoder.h5')
bigan.load_weights('Models/mnist_bigan_encoder.h5')
posthoc_bigan.load_weights('Models/mnist_posthoc_bigan_encoder.h5')


for encoder, name in encoders:
    encoder.trainable = False


# Hyperparameters and training specification for both models
epochs = 100
batch_size = 100
val_split = 1 / 5.

# Specify training stop criterion and when to save model weights
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

# Number of labelled examples to investigate
num_unlabelled = [100, 200, 500, 1000, 2000, 5000, 10000]
num_iterations = 5


# Arrays to hold accuracy of classifiers
classifier1_acc = np.zeros(len(num_unlabelled))
classifier2_acc = np.zeros(len(num_unlabelled))
classifier3_acc = np.zeros(len(num_unlabelled))
classifier4_acc = np.zeros(len(num_unlabelled))
classifier5_acc = np.zeros(len(num_unlabelled))
classifier6_acc = np.zeros(len(num_unlabelled))
classifier7_acc = np.zeros(len(num_unlabelled))
classifier8_acc = np.zeros(len(num_unlabelled))



# Loop through each quantity of enquiry
for index, num in enumerate(num_unlabelled):

    classifier1_score = 0
    classifier2_score = 0
    classifier3_score = 0
    classifier4_score = 0
    classifier5_score = 0
    classifier6_score = 0
    classifier7_score = 0
    classifier8_score = 0

    # Reduce size of training sets
    reduced_x_train_ae = x_train_ae[0:num, :, :, :]
    reduced_x_train_gan = x_train_gan[0:num, :, :, :]
    reduced_y_train = y_train_one_hot[0:num, :]

    # Average classification accuracy over num_iterations readings
    for iteration in range(num_iterations):

        # Print details of no. of labelled examples and iteration number
        print('Labelled Examples: ' + str(num) + ', Iteration: ' + str(iteration+1) + '/' + str(num_iterations))

        # Initialize classifiers

        # Classifier with e learned from autoencoder and frozen
        classifier1 = classifier_model(basic_ae)
        classifier2 = classifier_model(dae)
        classifier3 = classifier_model(aae)
        classifier4 = vae_classifier_model(vae)
        classifier5 = classifier_model(latent_regressor)
        classifier6 = classifier_model(bigan)
        classifier7 = classifier_model(posthoc_bigan)

        cnn = ae_encoder_model()
        classifier8 = classifier_model(cnn)


        # Compile models

        classifiers = (classifier1, classifier2, classifier3, classifier4, classifier5, classifier6, classifier7,
                       classifier8)

        for classifier in classifiers:
            classifier.compile(loss=keras.losses.categorical_crossentropy,
                                              optimizer=keras.optimizers.Adadelta(),
                                              metrics=['accuracy'])


        # Train models and save test accuracy

        # Classifier 1

        model_checkpoint = ModelCheckpoint('Models/classifier_1.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')
        callbacks = [early_stopping, model_checkpoint]

        classifier1.fit(reduced_x_train_ae, reduced_y_train,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          verbose=1,
                                          shuffle=True,
                                          callbacks=callbacks,
                                          validation_split=val_split)

        classifier1.load_weights('Models/classifier_1.h5')
        score = classifier1.evaluate(x_test_ae, y_test_one_hot, verbose=0)
        classifier1_score += score[1]


        # Classifier 2

        model_checkpoint = ModelCheckpoint('Models/classifier_2.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')
        callbacks = [early_stopping, model_checkpoint]

        classifier2.fit(reduced_x_train_ae, reduced_y_train,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          verbose=1,
                                          shuffle=True,
                                          callbacks=callbacks,
                                          validation_split=val_split)

        classifier2.load_weights('Models/classifier_2.h5')
        score = classifier2.evaluate(x_test_ae, y_test_one_hot, verbose=0)
        classifier2_score += score[1]


        # Classifier 3

        model_checkpoint = ModelCheckpoint('Models/classifier_3.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')
        callbacks = [early_stopping, model_checkpoint]

        classifier3.fit(reduced_x_train_ae, reduced_y_train,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          verbose=1,
                                          shuffle=True,
                                          callbacks=callbacks,
                                          validation_split=val_split)

        classifier3.load_weights('Models/classifier_3.h5')
        score = classifier3.evaluate(x_test_ae, y_test_one_hot, verbose=0)
        classifier3_score += score[1]


        # Classifier 4

        model_checkpoint = ModelCheckpoint('Models/classifier_4.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')
        callbacks = [early_stopping, model_checkpoint]

        classifier4.fit(reduced_x_train_ae, reduced_y_train,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          verbose=1,
                                          shuffle=True,
                                          callbacks=callbacks,
                                          validation_split=val_split)

        classifier4.load_weights('Models/classifier_4.h5')
        score = classifier4.evaluate(x_test_ae, y_test_one_hot, verbose=0)
        classifier4_score += score[1]



        # Classifier 5

        model_checkpoint = ModelCheckpoint('Models/classifier_5.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')
        callbacks = [early_stopping, model_checkpoint]

        classifier5.fit(reduced_x_train_gan, reduced_y_train,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          verbose=1,
                                          shuffle=True,
                                          callbacks=callbacks,
                                          validation_split=val_split)

        classifier5.load_weights('Models/classifier_5.h5')
        score = classifier5.evaluate(x_test_gan, y_test_one_hot, verbose=0)
        classifier5_score += score[1]



        # Classifier 6

        model_checkpoint = ModelCheckpoint('Models/classifier_6.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')
        callbacks = [early_stopping, model_checkpoint]

        classifier6.fit(reduced_x_train_gan, reduced_y_train,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          verbose=1,
                                          shuffle=True,
                                          callbacks=callbacks,
                                          validation_split=val_split)

        classifier6.load_weights('Models/classifier_6.h5')
        score = classifier6.evaluate(x_test_gan, y_test_one_hot, verbose=0)
        classifier6_score += score[1]



        # Classifier 7

        model_checkpoint = ModelCheckpoint('Models/classifier_7.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')
        callbacks = [early_stopping, model_checkpoint]

        classifier7.fit(reduced_x_train_gan, reduced_y_train,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          verbose=1,
                                          shuffle=True,
                                          callbacks=callbacks,
                                          validation_split=val_split)

        classifier7.load_weights('Models/classifier_7.h5')
        score = classifier7.evaluate(x_test_gan, y_test_one_hot, verbose=0)
        classifier7_score += score[1]



        # Classifier 8

        model_checkpoint = ModelCheckpoint('Models/classifier_8.h5', monitor='val_loss', verbose=1, save_best_only=True,
                                           mode='min')
        callbacks = [early_stopping, model_checkpoint]

        classifier8.fit(reduced_x_train_ae, reduced_y_train,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          verbose=1,
                                          shuffle=True,
                                          callbacks=callbacks,
                                          validation_split=val_split)

        classifier8.load_weights('Models/classifier_8.h5')
        score = classifier8.evaluate(x_test_ae, y_test_one_hot, verbose=0)
        classifier8_score += score[1]


    # Record average classification accuracy for each no. of labelled examples
    classifier1_acc[index] = 100 * classifier1_score / num_iterations
    classifier2_acc[index] = 100 * classifier2_score / num_iterations
    classifier3_acc[index] = 100 * classifier3_score / num_iterations
    classifier4_acc[index] = 100 * classifier4_score / num_iterations
    classifier5_acc[index] = 100 * classifier5_score / num_iterations
    classifier6_acc[index] = 100 * classifier6_score / num_iterations
    classifier7_acc[index] = 100 * classifier7_score / num_iterations
    classifier8_acc[index] = 100 * classifier8_score / num_iterations


# Save accuracies to file
np.savetxt('classifier1.txt', classifier1_acc, fmt='%f')
np.savetxt('classifier2.txt', classifier2_acc, fmt='%f')
np.savetxt('classifier3.txt', classifier3_acc, fmt='%f')
np.savetxt('classifier4.txt', classifier4_acc, fmt='%f')
np.savetxt('classifier5.txt', classifier5_acc, fmt='%f')
np.savetxt('classifier6.txt', classifier6_acc, fmt='%f')
np.savetxt('classifier7.txt', classifier7_acc, fmt='%f')
np.savetxt('classifier8.txt', classifier8_acc, fmt='%f')


# Print accuracies
print(classifier1_acc)
print(classifier2_acc)
print(classifier3_acc)
print(classifier4_acc)
print(classifier5_acc)
print(classifier6_acc)
print(classifier7_acc)
print(classifier8_acc)