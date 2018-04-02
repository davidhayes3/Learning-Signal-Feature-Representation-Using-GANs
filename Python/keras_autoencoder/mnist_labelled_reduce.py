import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib as plt
import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.datasets import mnist
from keras_conv_ae_models import encoder_model

# Define number of classes and actual class names
num_classes = 10
classes = np.arange(num_classes)

# Load MNIST data and split into train and test set
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert label vectors to one-hot vectors
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Parameters for algorithm
initial_num_labels = 1000
num_labels_per_class = initial_num_labels / num_classes

# Generate vector with number of initial labelled examples for all
num_labels_per_class = np.full(num_classes, num_labels_per_class, dtype=int)

# Define classifier model
def cnn(e):
    model = Sequential()
    model.add(e)
    model.add(Dense(128))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    return model

# Load models
e = encoder_model()
e.load_weights('encoder.h5')
e.trainable = False

# Define classifier model
mnist_classifier = cnn(e)

# Compile models
mnist_classifier.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

indices_initial = []

# Modify training set to contain set number of labels for each class
for class_index in range(num_classes):
    # Generate training set with even class distribution over all labels
    indices = [i for i, y in enumerate(y_train) if (y == classes[class_index] and i < num_labels_per_class[class_index])]
    indices_initial = np.concatenate((indices_initial, indices), axis=1)

# Sort indices so class examples are mixed up
indices_initial = np.sort(indices_initial)

# Reduce training vectors
x_train_initial = x_train[indices_initial]
y_train_initial = y_train[indices_initial]

#callbacks = [EarlyStopping]

# Train model on initial number of labelled examples
mnist_classifier.fit(x_train_initial, y_train_initial,
          batch_size=100,
          epochs=100,
          verbose=1,
          #callbacks=callbacks,
          shuffle=True,
          validation_split=1 / 12.)

# Get vector of predictions
predictions = mnist_classifier.predict_classes(x_test, verbose=0)

# Compute and print test accuracy of model
score = mnist_classifier.evaluate(x_test, y_test, verbose=0)
print("Overall test accuracy (%) with " + str(num_labels_per_class) + " labelled examples per class: " + str(100 * score[1]))


## Compute test accuracies for each individual digit

# Create array to hold test accuracy readings
class_test_accuracy = np.zeros(num_classes)

# Find accuracy of model on each class label
for class_index in range(num_classes):
    indices = [i for i, y in enumerate(y_test) if y == classes[class_index]]
    X_test_one_class = x_test[indices]
    y_test_one_class = y_test[indices]

    score = mnist_classifier.evaluate(X_test_one_class, y_test_one_class, verbose=0)
    class_test_accuracy[class_index] = 100 * score[1]

    # Print test accuracy for each digit
    print("Test accuracy for label " + str(classes[class_index]) + ": " + str(class_test_accuracy[class_index]) + "%\n")



## Generate confusion matrix

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
