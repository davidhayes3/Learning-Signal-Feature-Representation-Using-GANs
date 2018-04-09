import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from data_funcs import rescale_image
from PIL import Image


# Combines batch of generated images into one image
def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height * shape[0], width * shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = img[:, :, 0]
    return image


# Every 20 batches create an image of the generated images
def save_imgs(path, gen_imgs, epoch=None):
    imgs = combine_images(gen_imgs)
    imgs = rescale_image(imgs)

    if epoch is not None:
        Image.fromarray(imgs.astype(np.uint8)).save(path + '_gen_imgs_%d.png' % epoch)
    else:
        Image.fromarray(imgs.astype(np.uint8)).save(path + '_recons.png')


'''# Function to save images
def save_imgs(path, gen_imgs, epoch):
    r, c = 5, 5

    fig, axs = plt.subplots(r, c)
    count = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[count].reshape(2, 2), cmap='gray')
            axs[i, j].axis('off')
            count += 1
    fig.savefig(path + '_gen_%d.png' % (epoch+1))
    plt.close()'''


# Function to plot batch loss curves for generator and discriminator loss
def plot_gan_batch_loss(path, epochs, num_batches, d_batch_loss_trajectory, g_batch_loss_trajectory):

    batch_numbers = np.arange((epochs * num_batches)) + 1

    plt.plot(batch_numbers, d_batch_loss_trajectory, 'b-', batch_numbers, g_batch_loss_trajectory, 'r-')
    plt.legend(['Discriminator', 'Generator'], loc='upper right')
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')

    plt.savefig(path + '_batchloss.png')
    plt.show()


# Function to plot epoch loss curves for g and d
def plot_gan_epoch_loss(path, epochs, d_epoch_loss_trajectory, g_epoch_loss_trajectory):

    epoch_numbers = np.arange(epochs) + 1

    plt.plot(epoch_numbers, d_epoch_loss_trajectory, 'b-', epoch_numbers, g_epoch_loss_trajectory, 'r-')
    plt.legend(['Discriminator', 'Generator'], loc='upper left')
    plt.xlabel('Epoch Number')
    plt.ylabel('Average Minibatch Loss')

    plt.savefig(path + '_epochloss.png')


# Function to plot reconstructions of test set examples
def save_reconstructions(path, num_classes, test_data, test_labels, generator, encoder, num_recons_per_class=10):
    # Get initial data examples to train on
    classes = np.arange(num_classes)
    test_digit_indices = np.empty(0)

    # Modify training set to contain set number of labels for each class
    for class_index in range(num_classes):
        # Generate training set with even class distribution over all labels
        indices = [i for i, y in enumerate(test_labels) if y == classes[class_index]]
        indices = np.asarray(indices)
        indices = indices[0:num_recons_per_class]
        test_digit_indices = np.concatenate((test_digit_indices, indices))

    test_digit_indices = test_digit_indices.astype(np.int)

    # Generate test and reconstructed digit arrays
    X_test = test_data[test_digit_indices]
    recon_test = generator.predict(encoder.predict(X_test))
    #n = len(X_test)

    save_imgs(path, recon_test)

    '''# Plot test digits
    for i in range(n):
        ax = plt.subplot(2 * num_classes, num_recons_per_class, i + 1)
        plt.imshow(X_test[i].reshape(2, 2))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    # Plot reconstructed test digits
    for i in range(n):
        ax = plt.subplot(2 * num_classes, num_recons_per_class, i + 1 + n)
        plt.imshow(recon_test[i].reshape(2, 2))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig(path + '_recons.png')'''


# Function to plot 2D latent space visualizations
def save_latent_vis(path, data, classes, encoder, epoch, num_classes):

    z = encoder.predict(data)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = cm.Spectral(np.linspace(0, 1, num_classes))

    xx = z[:,0]
    yy = z[:,1]

    labels = range(num_classes)

    # plot the 2D data points
    for i in range(num_classes):
        ax.scatter(xx[labels == i], yy[labels == i], color=colors[i], label=labels[i], s=5)

    plt.axis('tight')
    plt.savefig(path + '_latent_vis_%d.png' % (epoch+1))