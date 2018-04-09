
# Function to save models
def save_models(path, encoder, generator):
    encoder.save_weights('Models/cifar10_bigan_encoder.h5')

    if generator is not None:
        generator.save_weights(path + '_generator.h5')