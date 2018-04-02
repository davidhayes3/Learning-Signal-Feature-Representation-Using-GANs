from sklearn.metrics import confusion_matrix
from keras_conv_ae_models import encoder_model
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.datasets import mnist
import numpy as np
import keras
import itertools
import matplotlib.pyplot as plt


# Define classifier model
def cnn(e):
    model = Sequential()
    model.add(e)
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model

# Load models
e = encoder_model()
#e.load_weights('encoder.h5')
#e.trainable = False

# Define classifier model
mnist_classifier = cnn(e)

# Compile models
mnist_classifier.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# Specify training stopping criterion
callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0)]


## Implement algorithm

# Set parameters for algorithm
num_classes = 10
initial_num_labels = 1000
num_labels_per_class = np.int(initial_num_labels / num_classes)

# Create vector with name of all classes
classes = np.arange(num_classes)


## Generate initial training data set for classifier

# Load MNIST data and split into train and test set
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_train = x_train[:, :, :, None]
x_test = x_test[:, :, :, None]
y_test_one_hot = keras.utils.to_categorical(y_test, num_classes)

# Get initial data examples to train on
indices_initial = np.empty(0)

# Modify training set to contain set number of labels for each class
for class_index in range(num_classes):
    # Generate training set with even class distribution over all labels
    indices = [i for i, y in enumerate(y_train) if y == classes[class_index]]
    indices = np.asarray(indices)
    indices = indices[0:10]
    indices_initial = np.concatenate((indices_initial, indices))

# Sort indices so class examples are mixed up
indices_initial = np.sort(indices_initial)
indices_initial = indices_initial.astype(np.int)

# Reduce training vectors
x_train_initial = x_train[indices_initial]
y_train_initial = y_train[indices_initial]

# Convert label vectors to one-hot vectors
y_train_initial = keras.utils.to_categorical(y_train_initial, num_classes)
y_test_one_hot = keras.utils.to_categorical(y_test, num_classes)


# Train model on initial number of labelled examples
mnist_classifier.fit(x_train_initial, y_train_initial,
          batch_size=100,
          epochs=100,
          verbose=1,
          callbacks=callbacks,
          validation_split=1/10.)


# Compute and print test accuracy of model
score = mnist_classifier.evaluate(x_test, y_test_one_hot, verbose=0)
current_test_acc = score[1]
print("Overall test accuracy (%) with " + str(num_labels_per_class) + " labelled examples per class: "
              + str(100 * current_test_acc))


## Begin guided labelling algorithm with initial classifier

# Set desired test accuracy for data set
test_acc_desired = 0.99
num_labels_added_per_iter = 100

# Create unlabelled and labelled set
x_train_unlabelled = x_train
y_train_unlabelled = y_train
x_train_labelled = np.empty([0, 28, 28, 1])
y_train_labelled = np.empty(0)

num_iterations = np.int(x_train.shape[0] / num_labels_added_per_iter)
classifier_acc = np.zeros(num_iterations)
num_labels = np.zeros(num_iterations)

# Create array to hold test accuracy readings
class_test_accuracy = np.zeros(num_classes)

current_test_acc = 0

# Loop until test accuracy is at state of art level
for iteration in range(num_iterations):

    print("\nIteration " + str(iteration + 1) + "\n")

    if (current_test_acc < test_acc_desired):
        # Calculate entropy of classifier for all examples in unlabelled set
        predictions = mnist_classifier.predict(x_train_unlabelled)
        x_train_unlabelled_entropy = (-predictions * np.log2(predictions)).sum(axis=1)

        # Find indices of examples with 1000 highest entropy in unlabelled set
        max_entropy_indices = x_train_unlabelled_entropy.argsort()[-num_labels_added_per_iter:][::-1]

        # Add these examples to labelled set and remove from unlabelled set
        x_train_labelled = np.concatenate((x_train_labelled, x_train_unlabelled[max_entropy_indices]))
        y_train_labelled = np.concatenate((y_train_labelled, y_train_unlabelled[max_entropy_indices]))
        y_train_labelled_one_hot = keras.utils.to_categorical(y_train_labelled, num_classes)
        x_train_unlabelled = np.delete(x_train_unlabelled, max_entropy_indices, axis=0)
        y_train_unlabelled = np.delete(y_train_unlabelled, max_entropy_indices)

        # Re-initialize classifier
        mnist_classifier = cnn(e)

        # Compile model
        mnist_classifier.compile(loss=keras.losses.categorical_crossentropy,
                                 optimizer=keras.optimizers.Adadelta(),
                                 metrics=['accuracy'])

        # Train classifier

        print("Training on " + str(len(x_train_labelled)) + " most difficult examples\n")

        mnist_classifier.fit(x_train_labelled, y_train_labelled_one_hot,
                             batch_size=100,
                             epochs=100,
                             verbose=1,
                             shuffle=True,
                             callbacks=callbacks,
                             validation_split=1/10.)

        # Update and print test accuracy
        score = mnist_classifier.evaluate(x_test, y_test_one_hot, verbose=0)
        current_test_acc = score[1]
        classifier_acc[iteration] = current_test_acc
        num_labels[iteration] = x_train_labelled.shape[0]
        print("Test accuracy with " + str(len(x_train_labelled)) + " most difficult examples labelled: "
              + str(100 * current_test_acc) + "%\n")

        # Find accuracy of model on each class label
        for class_index in range(num_classes):
            indices = [i for i, y in enumerate(y_test) if y == classes[class_index]]
            x_test_one_class = x_test[indices]
            y_test_one_class = y_test[indices]
            y_test_one_class = keras.utils.to_categorical(y_test_one_class, num_classes)

            score = mnist_classifier.evaluate(x_test_one_class, y_test_one_class, verbose=0)
            class_test_accuracy[class_index] = 100 * score[1]

            # Print test accuracy for each digit
            print("Test accuracy for label " + str(classes[class_index]) + ": " + str(
                class_test_accuracy[class_index]) + "%\n")

        # Update number of examples added per iteration
        if num_labels_added_per_iter < 1000:
            num_labels_added_per_iter *= 2
        else:
            num_labels_added_per_iter = 1000

    else:
        break


## Analyse results of model obatined


## Plot test accuracy vs no.of labelled examples available

plt.figure(1)

plt.plot(num_labels, classifier_acc)
plt.xlabel("No. of labelled examples available")
plt.ylabel("MNIST Test Accuracy (%)")
plt.show()


## Generate confusion matrix

predictions = mnist_classifier.predict_classes(x_test)

plt.figure(2)

cm = confusion_matrix(y_test, predictions)

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)

tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, num_classes)
plt.yticks(tick_marks, num_classes)

fmt = 'd'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True labels')
plt.xlabel('Predicted labels')
plt.title('MNIST Confusion Matrix')
plt.show()