import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Reshape
from keras.layers.core import Flatten, Dropout, Lambda, Activation
from keras.layers.merge import Concatenate
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.engine.topology import Layer


img_rows = 32
img_cols = 32
channels = 3
img_shape = (img_rows, img_cols, channels)
latent_dim = 64


class ConvMaxout(Layer):
    def __init__(self, n_piece, **kwargs):
        self.n_piece = n_piece
        super(ConvMaxout, self).__init__(**kwargs)

    def call(self, x):
        n = K.shape(x)[0]
        h = K.shape(x)[1]
        w = K.shape(x)[2]
        ch = K.shape(x)[3]
        x = K.reshape(x, (n, h, w, ch//self.n_piece, self.n_piece))
        x = K.max(x, axis=-1)
        return x

    def compute_output_shape(self, input_shape):
        n, h, w, ch = input_shape
        return (n, h, w, ch//self.n_piece)


def gan_generator_model():
    model = Sequential()

    model.add(Reshape(target_shape=(1,1,latent_dim), input_shape=(latent_dim,)))
    model.add(Conv2DTranspose(256, (4,4), strides=(1,1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.1))
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.1))
    model.add(Conv2DTranspose(64, (4,4), strides=(1,1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.1))
    model.add(Conv2DTranspose(32, (4,4), strides=(2,2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.1))
    model.add(Conv2DTranspose(32, (5,5), strides=(1,1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(32, (1,1), strides=(1,1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(3, (1,1), strides=(1,1)))
    model.add(Activation('sigmoid'))

    return model


def encoder_model():
    input = Input(shape=img_shape)

    x = Conv2D(32, (5,5), strides=(1,1))(input)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(64, (4,4), strides=(2,2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(128, (4,4), strides=(1,1))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(256, (4,4), strides=(2,2))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(512, (4,4), strides=(1,1))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(512, (1,1), strides=(1,1))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)

    mu = Conv2D(64, (1,1), strides=(1,1))(x)
    sigma = Conv2D(64, (1,1), strides=(1,1))(x)
    concatenated = Concatenate(axis=-1)([mu, sigma])

    output = Lambda(
        function=lambda x: x[:,:,:,:64] + K.exp(x[:,:,:,64:]) * K.random_normal(shape=K.shape(x[:,:,:,64:])),
        output_shape=(1,1,64))(concatenated)

    return Model(input, output)


def encoder_model():
    model = Sequential

    model.add(Conv2D(32, (5,5), strides=(1,1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(64, (4,4), strides=(2,2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(128, (4,4), strides=(1,1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(256, (4,4), strides=(2,2)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(512, (4,4), strides=(1,1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(512, (1,1), strides=(1,1)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.1))
    model.add(Conv2D(64, (1,1), strides=(1,1)))

    return model


def bigan_discriminator_model():
    z_in = Input(shape=(latent_dim,))
    x_in = Input(shape=img_shape)

    z = Dropout(0.2)(z_in)
    z = Conv2D(512, (1,1), strides=(1,1))(z)
    z = ConvMaxout(n_piece=2)(z)
    z = Dropout(0.5)(z)
    z = Conv2D(512, (1,1), strides=(1,1))(z)
    z = ConvMaxout(n_piece=2)(z)

    x = Dropout(0.2)(x_in)
    x = Conv2D(32, (5,5), strides=(1,1))(x)
    x = ConvMaxout(n_piece=2)(x)
    x = Dropout(0.5)(x)
    x = Conv2D(64, (4,4), strides=(2,2))(x)
    x = ConvMaxout(n_piece=2)(x)
    x = Dropout(0.5)(x)
    x = Conv2D(128, (4,4), strides=(1,1))(x)
    x = ConvMaxout(n_piece=2)(x)
    x = Dropout(0.5)(x)
    x = Conv2D(256, (4,4), strides=(2,2))(x)
    x = ConvMaxout(n_piece=2)(x)
    x = Dropout(0.5)(x)
    x = Conv2D(512, (4,4), strides=(1,1))(x)
    x = ConvMaxout(n_piece=2)(x)

    concatenated = Concatenate(axis=-1)([z, x])
    c = Dropout(0.5)(concatenated)
    c = Conv2D(1024, (1,1), strides=(1,1))(c)
    c = ConvMaxout(n_piece=2)(c)
    c = Dropout(0.5)(c)
    c = Conv2D(1024, (1,1), strides=(1,1))(c)
    c = ConvMaxout(n_piece=2)(c)
    c = Dropout(0.5)(c)
    c = Conv2D(1, (1,1), strides=(1,1), activation='sigmoid')(c)
    validity = Flatten()(c)

    return Model([z_in, x_in], validity)


def gan_discriminator_model():
    model = Sequential()

    model.add(Dropout(0.2, input_shape=img_shape))
    model.add(Conv2D(32, (5,5), strides=(1,1)))
    model.add(ConvMaxout(n_piece=2))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (4,4), strides=(2,2)))
    model.add(ConvMaxout(n_piece=2))
    model.add(Dropout(0.5))
    model.add(Conv2D(128, (4,4), strides=(1,1)))
    model.add(ConvMaxout(n_piece=2))
    model.add(Dropout(0.5))
    model.add(Conv2D(256, (4,4), strides=(2,2)))
    model.add(ConvMaxout(n_piece=2))
    model.add(Dropout(0.5))
    model.add(Conv2D(512, (4,4), strides=(1,1)))
    model.add(ConvMaxout(n_piece=2))

    model.add(Dropout(0.5))
    model.add(Conv2D(1024, (1,1), strides=(1,1)))
    model.add(ConvMaxout(n_piece=2))
    model.add(Dropout(0.5))
    model.add(Conv2D(1024, (1,1), strides=(1,1)))
    model.add(ConvMaxout(n_piece=2))
    model.add(Dropout(0.5))
    model.add(Conv2D(1, (1,1), strides=(1,1)))
    model.add(Activation('sigmoid'))
    model.add(Flatten())

    return model


def bigan_model(generator, encoder, discriminator):
    z = Input(shape=(latent_dim,))
    x = Input(shape=img_shape)

    x_ = generator(z)
    z_ = encoder(x)

    fake = discriminator([z, x_])
    valid = discriminator([z_, x])

    return Model([z, x], [fake, valid])